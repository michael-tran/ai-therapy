import { AutoTokenizer, env as transformersEnv } from '@huggingface/transformers';

let tokenizer: any = null;

transformersEnv.localModelPath = import.meta.env.BASE_URL || '/';;
transformersEnv.allowLocalModels = true;
transformersEnv.allowRemoteModels = false;

// const MODEL_PATH = import.meta.env.BASE_URL + 'model_int8.onnx';
const TOKENIZER_PATH = 'smol360';
let EOS_TOKEN_ID: number | null = null;

async function initTokenizer() {
  if (tokenizer) return tokenizer;
  
  try {
    tokenizer = await AutoTokenizer.from_pretrained(TOKENIZER_PATH, {
      local_files_only: true,
    });
    
    // Fallback logic for EOS token ID
    EOS_TOKEN_ID = tokenizer.eos_token_id ?? 2;
    
    console.log("Tokenizer initialized successfully!");
    return tokenizer;
  } catch (error) {
    console.error("Failed to initialize tokenizer:", error);
    throw error;
  }
}

let aiWorker: Worker | null = null;

export function startAiWorker() {
  if (aiWorker) return aiWorker;
  aiWorker = new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' });
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
    const partCount = 6; // update this if you have more or fewer parts
    w.postMessage({ type: 'init', baseUrl, partCount });
  });
}

// Streamed generation: returns a promise that resolves to final text and accepts an onToken callback
export async function generateReplyStream(userText: string, onToken: (tokenId: number, tokenText: string) => void, maxLength = 120) {
  const tok = await initTokenizer();
  await initWorkerSession(); // ensure worker created AND session initialized
  startAiWorker();

  // Define the conversation structure
  const messages = [
    { 
      role: 'system', 
      content: 'You are an empathetic, non-judgmental therapist. Listen, validate feelings, and ask open-ended questions to guide self-reflection. Keep responses concise, supportive, and focused on the user. Do not give medical advice. Your name is Linda.' 
    },
    { 
      role: 'user', 
      content: userText 
    }
  ];
  const prompt = tok.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
  }) as string;
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

