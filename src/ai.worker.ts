// src/ai.worker.ts
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

self.addEventListener('message', async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      const modelPath = msg.modelPath || '/model_int8.onnx';
      session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['webgl', 'wasm'],
        graphOptimizationLevel: 'all',
      });

      self.postMessage({
        type: 'inited',
        inputNames: session.inputNames,
        outputNames: session.outputNames,
        inputMetadata: session.inputMetadata,
      });
      return;
    }

    if (msg.type === 'generate') {
      if (!session) throw new Error('session not initialized');

      const inputIds: number[] = msg.inputIds;
      const maxLength: number = msg.maxLength ?? 50;
      const eosTokenId: number | null = msg.eosTokenId ?? null;

      const generated = [...inputIds];

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
            // Ensure past/present KV inputs are rank-4 with seq axis zero: [batch, num_heads, seq_len(0), head_dim]
            // Use metadata if available to preserve ranks, otherwise fallback to [1,3,0,64].
            const dims: number[] = Array.isArray(meta.dims) && meta.dims.length >= 4 ? meta.dims : [1, 3, -1, 64];
            const shape = dims.map((d, idx) => {
              // keep positive dims, map -1 (sequence axis) to 0, fallback to 1; ensure rank 4
              if (d > 0) return d;
              if (d === -1) return 0;
              return idx === 2 ? 0 : 1;
            });
            const size = shape.reduce((a, b) => a * b, 1);
            const zeros = size > 0 ? new Array(size).fill(0) : [];
            const ttype = meta.type === 'int32' ? 'int32' : (meta.type === 'int64' ? 'int64' : 'float32');
            feeds[name] = createTensorFromNumbers(ttype as any, zeros, shape);
          } else {
            // generic fallback: build shape from metadata, map -1 -> 0
            const dims: number[] = Array.isArray(meta.dims) ? meta.dims : [1];
            const shape = dims.map((d) => (d > 0 ? d : (d === -1 ? 0 : 1)));
            const size = shape.reduce((a, b) => a * b, 1);
            const zeros: number[] = size > 0 ? new Array(size).fill(0) : [];
            const ttype = meta.type === 'int32' ? 'int32' : (meta.type === 'int64' ? 'int64' : 'float32');
            feeds[name] = createTensorFromNumbers(ttype as any, zeros, shape);
          }
        }

        const outputs = await session.run(feeds);

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

        self.postMessage({ type: 'token', tokenId: nextTokenId });

        if (eosTokenId !== null && nextTokenId === eosTokenId) break;

        generated.push(nextTokenId);
      }

      self.postMessage({ type: 'done' });
    }
  } catch (err: any) {
    self.postMessage({ type: 'error', message: String(err?.message ?? err) });
  }
});