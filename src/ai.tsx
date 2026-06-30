import { AutoTokenizer, env as transformersEnv } from '@huggingface/transformers';

let tokenizer: any = null;
let aiWorker: Worker | null = null;
let isWorkerInited = false;        // Track initialization state
let workerInitPromise: Promise<void> | null = null; // Prevent concurrent initialization calls

transformersEnv.localModelPath = import.meta.env.BASE_URL || '/';;
transformersEnv.allowLocalModels = true;
transformersEnv.allowRemoteModels = false;

let EOS_TOKEN_ID: number | null = null;

export interface Message {
  id: number;
  content: string;
  role: 'user' | 'system' | 'assistant';
}

const MODEL_CONFIGS = {
  smol: {
    tokenizerPath: 'smol/',
    partCount: 3,
    numHeads: 3
  },
  smol360: {
    tokenizerPath: 'smol360/',
    partCount: 7,
    numHeads: 5
  }
};

const ACTIVE_MODEL_KEY: 'smol' | 'smol360' = 'smol360';

const SELECTED_CONFIG = MODEL_CONFIGS[ACTIVE_MODEL_KEY];
const TOKENIZER_PATH = SELECTED_CONFIG.tokenizerPath+"config";

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

export function startAiWorker() {
  if (aiWorker) return aiWorker;
  aiWorker = new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' });
  return aiWorker;
}

export async function initWorkerSession() {
  if (isWorkerInited) return; // Skip if already initialized
  if (workerInitPromise) return workerInitPromise; // Return existing ongoing initialization
  workerInitPromise = new Promise<void>((resolve, reject) => {
    const w = startAiWorker();
    const onMsg = (ev: MessageEvent) => {
      const m = ev.data;
      if (m?.type === 'inited') {
        w.removeEventListener('message', onMsg);
        isWorkerInited = true; // Mark as done!
        resolve();
      } else if (m?.type === 'error') {
        w.removeEventListener('message', onMsg);
        workerInitPromise = null; // Allow retrying if it fails
        reject(new Error(m.message || 'worker init error'));
      }
    };
    w.addEventListener('message', onMsg);

    const baseUrl = import.meta.env.BASE_URL || '/';
    w.postMessage({ 
      type: 'init', 
      baseUrl: baseUrl+SELECTED_CONFIG.tokenizerPath, 
      partCount: SELECTED_CONFIG.partCount, 
      numHeads: SELECTED_CONFIG.numHeads 
    });
  });

  return workerInitPromise;
}

// Streamed generation: returns a promise that resolves to final text and accepts an onToken callback
export async function generateReplyStream(liveMessages: Message[], userText: string, onToken: (tokenId: number, tokenText: string) => void, maxLength = 120) {
  await initTokenizer();
  await initWorkerSession();
  startAiWorker();

  const formattedMessages = liveMessages.map(message => ({
    role: message.role,
    content: message.content // Map 'context' to 'content'
  }));

  formattedMessages.push({
    role: 'user',
    content: userText
  });
  formattedMessages.push({
    role: 'system', 
    content: 'You are a helpful assistant. Your name is Smol.' 
  });
  
  const prompt = tokenizer.apply_chat_template(formattedMessages.slice(-3), {
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

