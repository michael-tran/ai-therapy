import * as ort from 'onnxruntime-web';
import { AutoTokenizer, env as transformersEnv } from '@xenova/transformers';

let tokenizer: any = null;
let session: ort.InferenceSession | null = null;

transformersEnv.localModelPath = import.meta.env.BASE_URL || '/';;
transformersEnv.allowLocalModels = true;
transformersEnv.allowRemoteModels = false;

const MODEL_PATH = import.meta.env.BASE_URL + 'model_int8.onnx';
const TOKENIZER_PATH = 'smol';
let EOS_TOKEN_ID: number | null = null;

async function initTokenizer() {
  if (tokenizer) return;
  tokenizer = await AutoTokenizer.from_pretrained(TOKENIZER_PATH, {
    local_files_only: true,
  });
  EOS_TOKEN_ID = tokenizer.eos_token_id ?? 2;
}

// async function initModel() {
//   if (session) return;
//   session = await ort.InferenceSession.create(MODEL_PATH, {
//     executionProviders: ['webgl'], // fallback to ['wasm'] if not available
//     graphOptimizationLevel: 'all',
//   });
// }

// paste into your ai.tsx (replace initModel or add new loader)
async function fetchArrayBufferOrThrow(url: string) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to fetch ${url}: ${r.status}`);
  return await r.arrayBuffer();
}

export async function initModelFromParts(partCount = 2) {
  if (session) return;
  // parts are served from public/, use BASE_URL to respect Vite base
  const base = import.meta.env.BASE_URL || '/';
  const parts: Uint8Array[] = [];
  for (let i = 1; i <= partCount; i++) {
    const name = `model_int8.onnx.part${String(i).padStart(2, '0')}`;
    const url = base + name;
    const ab = await fetchArrayBufferOrThrow(url);
    parts.push(new Uint8Array(ab));
  }

  const total = parts.reduce((s, p) => s + p.length, 0);
  const merged = new Uint8Array(total);
  let offset = 0;
  for (const chunk of parts) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  // create session from ArrayBuffer/Uint8Array
  // onnxruntime-web accepts ArrayBuffer/Uint8Array for create()
  session = await ort.InferenceSession.create(merged.buffer, {
    executionProviders: ['webgl', 'wasm'],
    graphOptimizationLevel: 'all',
  });
  console.log('Loaded model from parts; session ready');
}

// --- worker client helpers (add to src/ai.tsx) ---

let aiWorker: Worker | null = null;

export function startAiWorker() {
  if (aiWorker) return aiWorker;
  aiWorker = new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' });
  aiWorker.onmessage = (ev) => {
    // default no-op; generate function installs a per-call handler via Promises
    console.log('aiWorker message', ev.data);
  };
  return aiWorker;
}

export async function initWorkerSession() {
  const w = startAiWorker();
  return new Promise<void>((resolve, reject) => {
    const onMsg = (ev: MessageEvent) => {
      const m = ev.data;
      if (m?.type === 'inited') {
        w.removeEventListener('message', onMsg);
        resolve();
      } else if (m?.type === 'error') {
        w.removeEventListener('message', onMsg);
        reject(new Error(m.message || 'worker init error'));
      }
    };
    w.addEventListener('message', onMsg);

    const baseUrl = import.meta.env.BASE_URL || '/';
    const partCount = 3; // update this if you have more or fewer parts
    w.postMessage({ type: 'init', baseUrl, partCount });
  });
}

// Streamed generation: returns a promise that resolves to final text and accepts an onToken callback
export async function generateReplyStream(userText: string, onToken: (tokenId: number, tokenText: string) => void, maxLength = 120) {
  await initTokenizer();
  await initWorkerSession(); // ensure worker created AND session initialized
  startAiWorker();

  const prompt = `<|im_start|>system
You are an empathetic, non-judgmental therapist. Listen, validate feelings, and ask open-ended questions to guide self-reflection. Keep responses concise, supportive, and focused on the user. Do not give medical advice. Your name is Linda.
<|im_end|>
<|im_start|>user
${userText}
<|im_end|>
<|im_start|>assistant
`;
  const inputIds: number[] = await textToIds(prompt);

  return new Promise<string>((resolve, reject) => {
    if (!aiWorker) return reject(new Error('worker not created'));
    let partialIds: number[] = [];

    const onMessage = async (ev: MessageEvent) => {
      const m = ev.data;
      if (m?.type === 'token') {
        const tokenId: number = m.tokenId;
        partialIds.push(tokenId);
        const tokenText = await tokenizer.decode([tokenId], { skipSpecialTokens: true });
        onToken(tokenId, tokenText);
      } else if (m?.type === 'done') {
        aiWorker!.removeEventListener('message', onMessage);
        const finalText = await tokenizer.decode(partialIds, { skipSpecialTokens: true });
        resolve(finalText);
      } else if (m?.type === 'error') {
        aiWorker!.removeEventListener('message', onMessage);
        reject(new Error(m.message || 'worker error'));
      }
    };

    aiWorker.addEventListener('message', onMessage);
    aiWorker.postMessage({ type: 'generate', inputIds, maxLength, eosTokenId: EOS_TOKEN_ID });
  });
}

export async function textToIds(text: string): Promise<number[]> {
  await initTokenizer();
  return tokenizer.encode(text);
}

export async function idsToText(ids: number[]): Promise<string> {
  await initTokenizer();
  return tokenizer.decode(ids, { skipSpecialTokens: true });
}

function argMax(array: Float32Array | number[]) {
  let maxIndex = 0;
  let maxValue = array[0];
  for (let i = 1; i < array.length; i++) {
    if (array[i] > maxValue) {
      maxValue = array[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

function createTensor(
  name: string,
  values: number[],
  shape: number[],
): ort.Tensor {
  const meta = session?.inputMetadata as Record<string, any> | undefined;
  const type = meta?.[name]?.type ?? 'int64';
  if (type === 'int32') {
    return new ort.Tensor('int32', Int32Array.from(values), shape);
  }
  return new ort.Tensor(
    'int64',
    BigInt64Array.from(values.map((value) => BigInt(value))),
    shape,
  );
}

function createZeroTensor(name: string, shape: number[]): ort.Tensor {
  const meta = session?.inputMetadata as Record<string, any> | undefined;
  const type = meta?.[name]?.type ?? 'float32';
  const size = shape.reduce((acc, dim) => acc * dim, 1);

  if (type === 'int32') {
    return new ort.Tensor('int32', new Int32Array(size), shape);
  }

  if (type === 'int64') {
    return new ort.Tensor(
      'int64',
      BigInt64Array.from(new Array(size).fill(0).map(() => BigInt(0))),
      shape,
    );
  }

  return new ort.Tensor('float32', new Float32Array(size), shape);
}

function getShapeForInput(name: string, seqLen: number): number[] {
  if (name === 'input_ids' || name === 'attention_mask' || name === 'position_ids') {
    return [1, seqLen];
  }

  const meta = session?.inputMetadata as Record<string, any> | undefined;
  const dims = meta?.[name]?.dims;
  if (dims && dims.length > 0) {
    return dims.map((dim: number, idx: number) => {
      if (dim > 0) return dim;
      if (dim === -1) {
        if (name === 'input_ids' || name === 'attention_mask' || name === 'position_ids') {
          return idx === 1 ? seqLen : 1;
        }
        return idx === 2 ? 0 : 1;
      }
      return 1;
    });
  }

  return [1];
}

export async function runModelGreedy(
  inputIds: number[],
  maxLength = 50,
) {
  await initModelFromParts();
  if (!session) throw new Error('Model session not initialized');

  const generatedIds = [...inputIds];

  for (let step = 0; step < maxLength; step++) {
    const seqLen = generatedIds.length;
    const feeds: Record<string, ort.Tensor> = {};

    for (const name of session.inputNames) {
      if (name === 'input_ids') {
        feeds[name] = createTensor(name, generatedIds, [1, seqLen]);
      } else if (name === 'attention_mask') {
        feeds[name] = createTensor(
          name,
          new Array(seqLen).fill(1),
          [1, seqLen],
        );
      } else if (name === 'position_ids') {
        feeds[name] = createTensor(
          name,
          generatedIds.map((_, index) => index),
          [1, seqLen],
        );
      } else {
        const shape = getShapeForInput(name, seqLen);
        feeds[name] = createZeroTensor(name, shape);
      }
    }

    const outputs = await session.run(feeds);

    const logits = outputs.logits as ort.Tensor;
    if (!logits) {
      throw new Error(
        `No logits output found. available outputs: ${Object.keys(outputs).join(', ')}`,
      );
    }

    const vocabSize = logits.dims[2];
    const data = logits.data as Float32Array;
    const lastOffset = (logits.dims[1] - 1) * vocabSize;
    const nextTokenId = argMax(data.slice(lastOffset, lastOffset + vocabSize));

    await initTokenizer();

    const tokenText = await tokenizer.decode([nextTokenId], { skipSpecialTokens: true });
    console.log('step', step, 'nextTokenId', nextTokenId, 'tokenText', tokenText);

    if (nextTokenId === EOS_TOKEN_ID) break;
    generatedIds.push(nextTokenId);
  }

  return generatedIds.slice(inputIds.length);
}

export async function generateReply(userText: string) {
  const prompt = `<|im_start|>system
You are an empathetic, non-judgmental therapist. Listen, validate feelings, and ask open-ended questions to guide self-reflection. Keep responses concise, supportive, and focused on the user. Do not give medical advice. Your name is Linda.
<|im_end|>
<|im_start|>user
${userText}
<|im_end|>
<|im_start|>assistant
`;
  console.log('generateReply prompt length', prompt.length);
  const inputIds = await textToIds(prompt);
  console.log('inputIds', inputIds.length);
  const replyIds = await runModelGreedy(inputIds, 120);
  console.log('replyIds', replyIds.length);
  if (replyIds.length === 0) return '';
  return idsToText(replyIds);
}