<div align="center">

# Laconic

**Language models that say more with fewer rows.** Still in Work (its a prototype not final version)

Sparse is supposed to cost quality. It doesn't have to.

[Paper](paper/Laconic_Paper.pdf) · [Quickstart](#quickstart) · [Claims & Proof](docs/CLAIMS.md)

</div>

---

```text
                          held-out quality, identical 42M twins (lower = better)

  dense ── reads ALL 2048 rows ────────  9.46 bpt   ████████████████████████████
  Laconic ── reads 32 rows ───────────  8.68 bpt   ████████████████████████          WINS. 1/64 the rows.
```

**Laconic reads 32 of 2048 feed-forward rows per token — and beats the identical
dense model that reads all of them.** Same architecture. Same data. Same compute.
Same seed. One difference: Laconic is *born sparse*.

Then it gets better:

| | |
|:---|:---|
| **8.68 vs 9.46 bpt** | sparse beats dense — at **1/64 of the rows** (42M twins, held-out) |
| **1.71x faster** | than the dense matvec, single-stream CPU, in a dependency-free C++ engine |
| **1.000 recall — by proof** | every token's read is *provably exact* (Cauchy-Schwarz verify-and-top-up), not tested-exact |
| **+18% at 100x rows** | capacity grows 100x, per-token cost barely moves — **speed doesn't fall as the model grows** |

<div align="center"><em>Every number measured on a 4 GB laptop GPU and its CPU.
Every claim re-verifiable with one command. No cloud bill anywhere in this story.</em></div>

---

## The 60-second version

**Why is everyone's model dense?** Because dense training *spreads* each token's
computation across essentially every FFN row. We attacked that spread from the
conversion side six different ways — truncation, mean compensation, cluster
backgrounds, residual low-rank bulk, joint value refitting, activation reshaping —
and measured the same wall every time: **no after-the-fact reader re-concentrates
a dense model's knowledge** (the paper calls this the *information-location
property*: where knowledge lives is decided by the training law, not the reader).

**So we changed the law.** Laconic trains with hard **top-k selection inside the
forward pass from step one**, annealing k from 64 down to 32 as training
progresses. The result isn't a model that tolerates sparsity — it's a model that
**specializes into it**:

- its best k is *exactly* its weaned k (32),
- evaluating it with **all** rows is its **worst** configuration (the extra rows are noise),
- push it below birth-k and it degrades *gracefully* (9.14 at k=16; the dense twin at k=16: 25.3 — garbage).

Serving reads only the addressed rows. A trained addresser proposes candidates;
a verify-and-top-up layer — a per-row Cauchy-Schwarz bound, one multiply-add per
unproposed row — **proves on every single token** that nothing relevant was
missed. Not 99.9% recall. **1.000, by construction**, verified bit-identical end
to end.

## Quickstart

```bash
pip install -r requirements.txt

# 1 - Train the twin pair yourself: a day on a 4 GB laptop GPU (that's the point)
python train/alpha_p3a.py

# 2 - Re-measure every number in this README (~15 min)
python eval/alpha_p3a_after.py

# 3 - The engine: single C++ file, zero dependencies
cd engine
python export_p3a_hyper.py
g++ -O2 -march=native -fopenmp -o laconic p3a_engine_fast.cpp
```

Char-scale gate in minutes (`train/alpha_native.py`), 150M pilot
(`train/alpha_pilot150.py`), and row-grafting — grow a trained model with new
rows while the old ones stay frozen (`train/alpha_graft.py`) — included.

## The paper

**[Laconic: Born-Sparse Language Models (PDF)](paper/Laconic_Paper.pdf)** — 12 pages:

1. the six-run conversion impossibility ladder (the negative result that motivates everything),
2. born-sparse training and the specialization signatures,
3. the addresser and the exactness proof,
4. engine speed and size-independence measurements,
5. an **honest 150M scaling miss** with both confounds isolated — because a scaling
   story with no recorded failures is a story nobody should trust.

Also in print: the two spectacular intermediate results our own harness *killed*
for failing reproduction, documented as method. The discipline is part of the
contribution.

## Repository

```
train/   born-sparse trainers: char gate - 42M token twins - 150M pilot - row-grafting
engine/  single-file C++ inference: dense / oracle / addressed / proven-exact
eval/    the five-claim battery: re-measures this README in one command
docs/    CLAIMS.md (proof table + live commands) + full research log
paper/   the PDF
```

## Status & roadmap

- [x] **42M twins** — all five claims measured, re-verifiable, engine-parity to 0.003%
- [x] **Proven-exact read** — recall 1.000 by construction, zero measured cost
- [ ] **150M, data-scaled** — the open gate (paper §7; confounds isolated, run queued)
- [ ] **0.5B-class Laconic** — same recipe, real data budget

## Why "Laconic"?

*Laconic (adj.): using very few words — saying much with little.*
Sparta's replies were brief because they were confident. So is this model:
32 rows, full meaning.

**Mission:** frontier-class AI on hardware everyone already owns.

Apache-2.0 — use it, ship it, make it stranger.

---

<div align="center"><sub>Laconic Research · 2026 · measured on a laptop, released for everyone</sub></div>
