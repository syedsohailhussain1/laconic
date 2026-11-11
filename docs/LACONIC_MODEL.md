# Laconic-0.5B — the addressed-read language model (v0 status)

_Date: 2026-10-12/13. Everything below is measured on this machine, on
Qwen2.5-0.5B, held-out wikitext-2 test (32K tokens), FP32, unless stated._

## What Laconic is

One model, converted from an open-weight 0.5B by pure weight surgery
(rotation-free this round; calibration fitting only — no gradient training,
no teacher, no student, no router bolted on). Its MLPs are **addressed row
memories** read through computed addresses; per-token cost is set by **rows
read**, not rows stored:

```
MLP output(x) = bias0[cluster(x)] + b0                        background rows
              + ((x − x̄) · U_r) · Rows_r                       low-rank bulk (r rows)
              + Σ_{i ∈ topk_k |act−m|} act_i · W_down[:,i]    addressed rows (k)
```

- `cluster(x)` — 128-centroid background codebook (conversion-time k-means)
- `topk_k` selection over centered activations `act − m_c[cluster]`
- bulk fitted AFTER the row memory, on its residual (truncated-lstsq, rank ≤256)

Attention, embeddings, lm_head untouched this round (the repo's proven
multi-plane ternary K=3 conversion applies to them in the engine build).

## Measured quality surface (oracle6, all 24 layers converted)

| k \ r | 0 | 64 | 128 | 256 |
|---|---|---|---|---|
| 1024 | 22.99 | 21.93 | 21.38 | **21.05** |
| 512 | 39.78 | 31.40 | 28.35 | 25.88 |
| 256 | 108.7 | 53.18 | 41.40 | 33.13 |
| 128 | 367.0 | 98.86 | 62.16 | 44.46 |

Dense baseline: **16.86**. Best v0: k=1024, r=256 → 21.05 (+25% ppl) at
1280 row-equivalents/MLP ≈ **26% of dense MLP compute**. Every r=0 cell
reproduces the algebra exactly (k=full = 16.8637, bit-exact, verified 3×).

Evidence ladder that got here: raw top-k cliff (k=256→319) → global mean
(175) → cluster backgrounds (87) → + residual-ordered bulk (**33**).
Two classical fitting bugs found and fixed on the way: cross-cov "SVD LS"
(30-billion ppl — must ridge/lstsq with input covariance) and bulk/row
double-counting (bulk must be fitted on the row memory's residual).

## Live generation (alpha_runtime.py, k=512 r=256, ppl 25.9)

- "The largest planet in our solar system is" → " Jupiter, which is the
  largest planet in our solar system. Jupiter is the largest planet..."
- "Albert Einstein was a" → " great man, but he was also a great genius..."
- Coherent, grammatical, small-model-class. NOT salad — unlike every
  pure-lookup fast-mind attempt (bigram/kNN/signature) in this repo's history.

## Speed & size-independence — honest status

- Reference runtime (torch fp32 CPU, oracle selection = full gate/up matvec):
  6 tok/s. This is the QUALITY vehicle, not the engine.
- Engine cost model at k=512, r=256, ternary-packed rows (~224 B/row):
  ~400 KB touched per layer per token → ~9.6 MB/token → **~4-12K tok/s
  single-stream CPU** in the existing C++ engine style; adaptive-depth and
  temporal row caching push further. The 1M tok/s line needs k≈64-128 at
  quality — see frontier below.
- Size-independence: structural (cost = k + r + 1 rows by construction;
  stored-but-unselected rows are never touched) + the repo's dict-scale
  proof (5-14M q/s flat across 153,846× growth). A vacuous-pad bug in
  alpha_runtime's --bench was caught: the real 100×-rows measurement
  belongs to the engine build (address-based selection), not the oracle
  reference. Do not cite the 1.02× number.

## The frontier (named, ranked)

1. **Address quality (THE gap).** Sublinear probe recall measured:
   SimHash b64: 0.47-0.72 @ C=1024-2048; PQ-ADC(M=16): 0.26-0.46 (product
   ranking amplifies quantization noise). Needed: ≥0.9 @ C≤512.
   Next moves: learned hash (fit SimHash bits on calib co-activation),
   OPQ-rotation + stronger codebooks, HNSW graph at 4864 rows.
2. **Value-refit (GPTQ-on-selection)** — the house move not yet applied:
   refit W_down columns so kept rows absorb missed rows' mass (4864×4864
   ridge per layer, Hessian machinery already exists in rotate_gptq_export).
   Directly attacks the k=512→256 quality cliff.
3. **ReLUfication** — fit per-neuron affine SiLU≈α·ReLU(βg+γ): true zeros
   make top-k benign (literature: 90-95% sparsity viable). Heavier surgery.
4. **Adaptive depth** — implemented, currently buggy (73K ppl at τ=0.95,
   mean-k 3364): the energy-threshold rule misbehaves; debug in isolation.
5. **Scale path** — same conversion at 4B/27B; bulk+attention projections
   are the width-coupled term (named wall, attack with bitwise GPU kernels).
   mergekit becomes the row-union growth engine: in Laconic format, capacity
   is rows; merging models = row union at zero read cost (the thesis, made
   literal).

## Files

| File | Role |
|---|---|
| alpha_oracle{1..6}.py | the measurement ladder (quality surface above) |
| alpha_probe{2,3}.py | address recall measurements (SimHash, PQ-ADC) |
| alpha_runtime.py | end-to-end reference runtime (generation above) |
| _alpha_bg_calib.pt / _alpha_o6_fits.pt | cached conversion artifacts |
| _oracle6.json / _probe*.json | raw result tables |

## GATE EXPERIMENTS (2026-10-13 evening) — quality-at-small-k verdict

Four doors measured on 24 layers, held-out, leak-checked (an early
contradictory 0.187 at L12 failed arbitration — v1 artifact, not real):

1. VALUE-REFIT (joint LS over co-selected rows; naive lam=1e-4 AND centered
   selection + lam sweep 0.01/0.05/0.2): LOSES at 22/24 layers (e.g. L12
   base 0.683 vs refit 0.770-0.834; only L23 marginal 0.464->0.424).
   The dropped mass is NOT linearly absorbable into kept rows.
2. RELU-FICATION (per-neuron a*relu(g+b) fit): silu-fit quality varies
   (rel 0.17 L12, 0.50 L0); energy concentration improves (L12 top-512:
   63%->93%) but did not convert into lower reconstruction error.
3. LOW-RANK: best-case rel-err 0.67-0.80 on mid layers (function is NOT
   mostly linear/bulk). L0/L2/L3/L21/L23 are much more linear (0.17-0.48).
4. PER-LAYER SPARSITY PROFILE (the positive find): hugely non-uniform.
   k=128 base rel-err: L2=0.078, L3=0.083, L21=0.135 (near-free sparse)
   vs mid-layers 0.62-0.70 (need k~512+). Per-layer k allocation is real.

CONCLUSION (measured, final for conversion-only surgery): a dense-trained
SwiGLU MLP cannot be re-expressed in few addressed rows by re-valuing.
Quality point for converted models: k~512-1024 (~26% dense compute).
The millions live in the narration tier; model-at-millions requires
training-sparse/ternary from scratch (direction change, user decision).

## PHASE-2 GATE: BORN-SPARSE TRAINING — PASSED (2026-10-14 12:03)

Twin char-level transformers (6.6M params, 6L x D256, FFN 1024, wikitext-2,
6k steps, identical seed/data):

  A dense-trained:  full=2.278 bpc | @k64=4.554 (cliff) | @k32=7.020
  B born-sparse (top-k=64 in training from step 1):
     @k64 = 2.291   <- 0.6% off dense-full, using 6.25% of FFN rows
     @k32 = 2.613   <- graceful below birth-k (no cliff)
     @k256 = 2.272  <- slightly BEATS dense-full
     @k1024 = 2.640 <- untrained rows hurt: function lives in few rows
  energy64: A 45.4% -> B 69.4% (training concentrates the tail)

VERDICT: quality-at-small-k EXISTS when born sparse. The conversion wall
(4 doors, all negative) is overturned by training-native sparsity. Next:
scaled token-level runs, k-annealing, and addresser measurement on
born-sparse activations (concentration should make probing easier).
Files: alpha_native.py, _native.json, _native_ckpt_*.pt

## ANNEAL RESULT (2026-10-14 20:39): weaning 64->48->32 WORKS
C@32 = 2.306 bpc vs dense-full 2.278 (+1.2%) at 1/32 of FFN rows.
Best-k moved DOWN to the weaned 32 (specialization). C@16 = 2.78 still graceful.
Chain now: dense 2.278 | born@64 2.291 | annealed@32 2.306. Quality lives in 32 rows.
(KeyError in final print cosmetic; FINALs logged in _native2_console.log)

## P3-A TOKEN-LEVEL SCALE UP + OFFICIAL BATTERY (2026-10-18, final)

Born-sparse twin at 42M: C (8L×d512, FFN 2048, BPE-16k; k-anneal 64→48→32;
9000 steps) + C2 (continued anneal to 14750). The READ mechanism had to
graduate from torch's oracle top-k (which needs full compute) to a DEPLOYED
selector: `p3a_addr.bin` — a trained per-layer cluster addresser (128
clusters × D512 logistic + row buckets, 1.9 MB total:

    per layer: p_c = sigmoid(h·coef_c + b_c); walk clusters in desc-p order,
    gather bucket rows until C candidates; true acts only for candidates
    (biasless); read top-32 by |act|; no down bias.

Cost ∝ rows touched (C + 32), not rows stored (2048/layer). C++ engine:
`_p3c_release/p3a_engine_fast{.cpp,.exe}` (dense 85.5 · addr C=512 169.7 ·
addr C=64 197.4 tok/s CPU; C=2048 default bench 156.2).

### Cross-implementation parity (certified this session)

- `.hyper` export == ckpt C2: engine dbg dense x0..x8/ln/logits == torch
  dense to ~5e-6 (C checkpoint differs ⇒ lineage pinned).
- Engine oracle read set == torch full-set |act| top-32: **32/32 identical**
  (per-row acts agree; addresses agree).
- Engine addr walk+read == torch mirror: **32/32 identical** at C=512 AND
  C=2048 (instrumented engine prints SEL_ORACLE/SEL_ADDR; compare scripts
  `_usr_micro5.py`).
- Free-run greedy: torch mirror == engine strings (both modes, both prompts,
  string-identical in the official battery run). Free-run is chaotic at the
  argmax margins across FP-noise; readability is the comparison frame.
- Forensic caution recorded: a probe that omits the pos-0 attention residual
  (a=v, proj) reproduces WRONG L0 h and phantom set-mismatches. The engine
  was never wrong.

### The five claims on final weights (battery v3 → _p3a_extra.json)

1. **Quality — PASS.** Gate (val bpt): C k32 8.9573 vs A-dense k32 19.5661
   (A-full 9.4605). C2 continuation 8.6755 (best); engine addr deploy
   8.9316 ≈ oracle ceiling 8.6845 (+2.8%). Born-sparse small-k returns
   survive the token-level scale.
2. **Generation — PASS (deployed path).** Engine addr C=512 greedy from a
   natural prompt writes proper wikitext, including section structure:

       = = = = Filming = = = =
       The film was released on April 1, 2005, and received by critics
       with the film. The film was released on November 25, ...

   (torch addr-mirror reproduces this string identically; oracle-mode is
   similar "= = Plot = ="; dense drifts hardest: "released on the film,
   which was released on the film"). Cold-gallery val[:64] prompt: every
   path loops (42M exposure bias — k-native "M. K. K." loops recorded as
   the honest contrast; NOT the deployed selector's failure).
3. **Addressability — PASS, absolute, with the metric the founder was owed.**
   The trained addresser's read set vs oracle top-32, measured ON the
   addressed forward's own residuals (1024 positions × 8 layers):
   **0.953 / 0.958** at the shipped C=512 budget (L0 0.77 hard, L5+ ≈1.00);
   **0.989 / 1.000** at full bucket-walk — the addresser's cluster ranking
   IS the oracle ranking; only budget truncation costs ~5%.
   Raw untrained SimHash probe (old metric): A 0.202 / C 0.235 / C2 0.239
   — kept as the untrained-projection contrast, not the claim.
4. **Speed — PASS.** Bucketed-index addressed read vs dense FFN matvec, CPU
   block-0: C 1.41× / C2 1.74× this run (prior runs 1.13–1.69×; drift-timer
   bounds documented); flat @100× (ratio 1.87–2.05, drift-bounded);
   probes 9–10, candidates 40→30–66 (index-bits 9/15 per scale);
   documented-wrong linear-scan contrast: 30–39× blowup at 100× rows.
   C++ authority: 169.7 tok/s addr-C512 vs 85.5 dense (bpt-anchored).
5. **Size-independence — PASS.** Torch microbench flat at 100× padded rows
   (ratio ≈ 1.9-2.0, noise-bounded; pure-wall ~flat: C 4.399 µs C 1.683 µs).
   C++ pad engine authoritative: 32× padded rows, bit-identical outputs,
   flat speed, max row touched = old budget (2047), +3.2 GB resident.
   100× wall-time at µs scale is allocator-noise dominated in torch — the
   C++ engine is the claim-5 instrument.

Verdicts live in `_p3a_extra.json` (`claim2_verdict`, `claim3_verdict`,
`claim3_addr` with per-layer recall). Fast-training audit closed separately:
P3-F v5 = PPMI-SVD + ridge + JM-trigram (not a real LM trainer; isolated
model cannot generate).

## P3-A FINAL (2026-10-20): ROW-NATIVE BEATS DENSE — full claim stack validated

Token-level twins (42M params, BPE-16k, 9k steps, identical seed/data):
  A dense:   full=9.461 | k64=15.14 | k32=19.57 | k16=25.35   (cliff)
  C row-native (anneal 64-48-32):
    k32 = 8.957  <- BEATS dense-full by 5% at 1/64 of the rows
    k48 = 8.982 | k64 = 9.022 | k128 = 9.188 | full = 9.544 (worst!)
  The sparse-born model is BETTER sparse; its best k is the weaned 32.

CLAIM BATTERY (trained weights, _p3a_extra.json):
  1 QUALITY: C@32 8.96 < A@full 9.46                     PASS (strict win)
  2 GENERATION: oracle@k32 coherent wikitext prose;      PASS
    addressed output matches oracle; dense twin is MORE repetitive
  3 ADDRESSER: deployed index recall in=0.953/out=0.958   PASS
    @C512; 0.989/1.000 @C2048; probes=10
  4 SPEED: addressed read 414.8us vs dense 583.6us       PASS (1.41x,
    pure Python, FFN=2048; grows with FFN/k ratio and in C++)
  5 SIZE-INDEP: index widens 9->15 bits at 100x rows,    PASS
    occupancy constant (flat mechanism, proven pattern)

NEXT (per locked plan): Runpod A6000 pilot ~150M twin (USD 3-4) to
confirm at scale -> 0.5B-class row-native Laconic -> C++ engine merge.
