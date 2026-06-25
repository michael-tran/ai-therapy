import * as ort from 'onnxruntime-web';

let session: ort.InferenceSession | null = null;

function createTensorFromNumbers(type: 'int32' | 'int64' | 'float32', values: number[], shape: number[]) {
  if (type === 'int32') {
    return new ort.Tensor('int32', Int32Array.from(values), shape);
  }
  if (type === 'int64') {
    const big = values.map((v) => BigInt(v));
    return new ort.Tensor('int64', BigInt64Array.from(big), shape);
  }
  return new ort.Tensor('float32', Float32Array.from(values), shape);
}

async function fetchArrayBufferOrThrow(url: string) {
  const r = await fetch(url);
  
  // 1. STRICTOR CHECK: Catch hidden 404/SPA route fallback pages
  if (!r.ok) {
    throw new Error(`HTTP Error fetching chunk! Status: ${r.status} for URL: ${url}`);
  }
  
  const contentType = r.headers.get('content-type') || '';
  if (contentType.includes('text/html')) {
    throw new Error(`Security/Routing Error: Server returned an HTML page instead of binary data for URL: ${url}. Check your public folder file names.`);
  }

  return await r.arrayBuffer();
}

ort.env.wasm.numThreads = 0;

async function loadModelFromParts(baseUrl: string, partCount: number): Promise<Uint8Array> {
  const parts: Uint8Array[] = [];
  for (let i = 1; i <= partCount; i++) {
    const name = `model_int8.onnx.part${String(i).padStart(2, '0')}`;
    const url = `${baseUrl}${name}`;
    const ab = await fetchArrayBufferOrThrow(url);
    parts.push(new Uint8Array(ab));
  }

  const total = parts.reduce((sum, p) => sum + p.length, 0);
  const merged = new Uint8Array(total);
  let offset = 0;
  for (const chunk of parts) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  return merged; // <-- Return the Uint8Array view directly
}

self.addEventListener('message', async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      const baseUrl = msg.baseUrl || '/';
      const partCount = msg.partCount ?? 6;

      const modelBuffer = await loadModelFromParts(baseUrl, partCount);
      console.log(`Successfully merged ${modelBuffer.byteLength} bytes.`);

      session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });

      // FIX: Robustly extract metadata by mapping directly over the verified inputNames array
      const cleanInputMetadata: Record<string, any> = {};
      
      session.inputNames.forEach((name) => {
        // Safe access on the internal metadata map
        const meta = (session.inputMetadata as any)[name];
        if (meta) {
          cleanInputMetadata[name] = {
            type: meta.type || 'float32',
            // Ensure dims is explicitly cloned out as a plain JS number array
            dims: Array.isArray(meta.dims) ? [...meta.dims] : (meta.shape ? [...meta.shape] : [])
          };
        } else {
          // Absolute safety fallback if the lookup key acts weirdly
          cleanInputMetadata[name] = { type: 'float32', dims: [] };
        }
      });

      // Verify it's no longer empty!
      console.log("Populated cleanInputMetadata:", cleanInputMetadata);

      self.postMessage({
        type: 'inited',
        inputNames: session.inputNames,
        outputNames: session.outputNames,
        inputMetadata: cleanInputMetadata,
      });
      return;
    }

    if (msg.type === 'generate') {
      if (!session) throw new Error('session not initialized');

      console.log("Inside Generate method")

      const inputIds: number[] = msg.inputIds;
      const maxLength: number = msg.maxLength ?? 50;
      const eosTokenId: number | null = msg.eosTokenId ?? null;

      const generated = [...inputIds];
      let lastOutputs: Record<string, ort.Tensor> | null = null;

      for (let step = 0; step < maxLength; step++) {
        const seqLen = generated.length;
        const feeds: Record<string, ort.Tensor> = {};

        for (const name of session.inputNames) {
          // inside the inputNames loop — replace the existing branch that builds 'shape'
          const meta = (session.inputMetadata as Record<string, any>)?.[name] || {};
          const type = meta.type ?? 'float32';

          // Known sequence inputs
          if (name === 'input_ids') {
            feeds[name] = createTensorFromNumbers(type === 'int32' ? 'int32' : 'int64', generated, [1, seqLen]);
          } else if (name === 'attention_mask') {
            feeds[name] = createTensorFromNumbers(type === 'int32' ? 'int32' : 'int64', new Array(seqLen).fill(1), [1, seqLen]);
          } else if (name === 'position_ids') {
            feeds[name] = createTensorFromNumbers(type === 'int32' ? 'int32' : 'int64', generated.map((_, i) => i), [1, seqLen]);
          } else if (name.toLowerCase().includes('past') || name.toLowerCase().includes('present') || name.toLowerCase().includes('key')) {
            // 1. Get the dynamic dimensions directly defined by the ONNX model's metadata
            const dims: number[] = Array.isArray(meta.dims) ? [...meta.dims] : [1, 3, 0, 64];
            
            // 2. Dynamically replace batch (-1 or dynamic string) with 1, and sequence length with 0 for the first iteration
            // Note: ONNX metadata sometimes uses string names or negative numbers for dynamic axes
            const shape = dims.map((d, idx) => {
              if (typeof d === 'number' && d > 0) return d;
              
              // Usually, index 0 is batch, index 2 is sequence length in [B, H, S, D]
              // If it's the sequence length axis, we start with 0 history.
              if (idx === 2 || d === 'sequence_length') return 0; 
              return 1; // Default fallback for batch size or dynamic heads
            });

            // 3. Check if the model returned previous outputs we can feed back in (True KV Caching)
            // If this isn't the first step, we look for the corresponding output tensor name from the last run
            const correspondingOutputName = name.replace('past_key_values', 'present_key_values').replace('past', 'present');
            // If we have previous cache outputs, feed them right back in
            if (step > 0 && lastOutputs && lastOutputs[correspondingOutputName]) {
              feeds[name] = lastOutputs[correspondingOutputName];
            } else {
              // Step 0 fallback: Generate initial empty cache tensors dynamically
              const dims: number[] = Array.isArray(meta.dims) ? [...meta.dims] : [1, 3, 0, 64];
              const shape = dims.map((d, idx) => {
                if (typeof d === 'number' && d > 0) return d;
                if (idx === 2 || d === 'sequence_length') return 0; 
                return 1;
              });
              

              const size = shape.reduce((a, b) => a * b, 1);
              const zeros = size > 0 ? new Array(size).fill(0) : [];
              const ttype = meta.type === 'int32' ? 'int32' : (meta.type === 'int64' ? 'int64' : 'float32');
              feeds[name] = createTensorFromNumbers(ttype as any, zeros, shape);
            }
          } else {
            const dims: number[] = Array.isArray(meta.dims) ? meta.dims : [1];
            const shape = dims.map((d) => (typeof d === 'number' && d > 0 ? d : 1));
            const size = shape.reduce((a, b) => a * b, 1);
            const zeros: number[] = size > 0 ? new Array(size).fill(0) : [];
            const ttype = meta.type === 'int32' ? 'int32' : (meta.type === 'int64' ? 'int64' : 'float32');
            feeds[name] = createTensorFromNumbers(ttype as any, zeros, shape);
          }
        }

        const outputs = await session.run(feeds);
        lastOutputs = outputs;

        const logits = outputs.logits as ort.Tensor | undefined;
        if (!logits) throw new Error('no logits output from model');

        const vocabSize = logits.dims[2];
        const data = logits.data as Float32Array;
        const lastOffset = (logits.dims[1] - 1) * vocabSize;

        let maxIdx = 0;
        let maxVal = data[lastOffset];
        for (let i = 1; i < vocabSize; i++) {
          const v = data[lastOffset + i];
          if (v > maxVal) {
            maxVal = v;
            maxIdx = i;
          }
        }
        const nextTokenId = maxIdx;

        if (eosTokenId !== null && nextTokenId === eosTokenId) break;

        self.postMessage({ type: 'token', tokenId: nextTokenId });

        generated.push(nextTokenId);
      }

      self.postMessage({ type: 'done' });
    }
  } catch (err: any) {
    self.postMessage({ type: 'error', message: String(err?.message ?? err) });
  }
});