# Laconic Phase-3: Scale-Up Design (locked 2026-10-14, while C-anneal trains)

Foundation (measured): born-sparse@k64 = 0.6% off dense (2.291 vs 2.278 bpc);
graceful below birth-k (2.61@32); function local (B@256 beats dense, full rows
hurt); addressability rises with concentration (energy recall 0.938@C512).

## Run P3-A: token-level row-native twin pair (the real gate at real vocab)

- Data: wikitext-2 + ultrachat sample (local), Qwen tokenizer (151k), seq 512
- Arch: 8L x d512, 8 heads, FFN 2048, tied embeddings (~28M params)
  (vocab head dominates; embeddings fp16)
- Twins, identical seed/data:
  A dense | C row-native: top-k INSIDE forward, k schedule {64 x 2500,
  48 x 1500, 32 x 2000} (anneal, per C-anneal result)
- Budget: 12k steps, batch 32, ~6-8h on GTX 1650 (fp32; no tensor cores)
- Eval: ppl at k in {16,32,48,64,128,full}; energy curves; probe recall on
  born-sparse activations (SimHash b64 first; learned hash if <0.85@C256)
- PASS = C@64 within 10% of A-full ppl. Then P3-B scales to 0.5B-class.

## Engine path (parallel, no GPU needed)

1. llama-alpha op: two-phase threading (select single-thread, sparse rows
   striped over threads) + int8-packed downT — expect 2-4x (target: alpha
   >= stock at k<=256).
2. Per-layer k allocation from the measured profile (L2/L3/L21 cheap,
   mid-layers expensive).
3. Size-independence test in-engine: pad row memory 100x (real ids), tok/s
   must stay flat — the honest version of the vacuous v0 bench.

## Rules of the house (unchanged)

Every number leak-checked; every "win" reproduced before it's a win;
conversion results (k~512 quality point) remain the shipped floor while
row-native training climbs. No killing GPU processes without asking —
the machine is shared.
