# CLAIMS.md — Laconic five claims, proof artifacts, and live re-verification

_Model: twin C (row-native, born-sparse, k-annealed 64→48→32), 42M params,
checkpoint `_p3a_ckpt_C2.pt` (step 14,750). Twin A = dense reference
(`_p3a_ckpt_A2.pt`). All numbers on held-out wikitext, BPE-16k._

| # | Claim | Result | Proof artifacts | Code/scripts that produce it |
|---|---|---|---|---|
| 1 | **Quality — row-native beats dense** | **C@k32 = 8.6755 bpt vs dense-full 9.4605** (8.3% better at 1/64 of FFN rows); gate: C@32 8.9573 vs 9.4605 | `_p3a.json` (gate table), `_p3a_cont.json` (C2 extension → 8.6755) | `alpha_p3a.py` (trains both twins, writes `_p3a.json`) |
| 2 | **Generation without degeneration** | addressed ≡ oracle; coherent prose ("= Plot = The film was released on April 25, 2005…"); dense twin loops more | `_p3a_extra.json` + `_p3a_after.log` (samples: gen engine addr_C512 / oracle / dense) | `alpha_p3a_after.py` (claim-2 section) |
| 3 | **Addresser finds the rows** | deployed recall **0.953 / 0.958 @ C512**; **0.989 / 1.000 @ C2048** (held-out, honesty protocol) | `_p3a_extra.json`; addresser binary `_p3c_release/p3a_addr.bin` | `alpha_p3a_after.py` (claim-3 section); addresser fitted by `_p3c_release/breakthrough.py` |
| 4 | **Speed — sparse beats dense** | C++ B=1 decode: **addressed 149.7 vs dense 87.4 tok/s (1.71×)**; FFN ms/tok 7.43→1.86 (4×); bpt anchors match torch to 0.04% (dense 9.2826, oracle 8.6845, addr-C512 8.9316) | `_p3c_release/p3a_engine_fast.exe` + `p3a_c2.hyper` + `_p3c_release/README.md` (reproduction table) | `_p3c_release/p3a_engine_fast.cpp` (build: `build_static.bat`); export via `_p3c_release/export_p3a_hyper.py` |
| 5 | **Size-independence — speed flat as rows grow** | C++ index widens 9→15 bits at 100× rows, occupancy ~constant; dict-tier proof: 5–14M q/s flat across 153,846× growth | `_p3c_release/` (index-widening results), `_provescale.txt` | `_p3c_release/e2e_bench.py`, `prove_scale.py` |

**Trained models (the weights the claims live in):** `_p3a_ckpt_C2.pt` (row-native,
the claim carrier), `_p3a_ckpt_A2.pt` (dense twin, the reference), plus phase-2
char-level gates `_p3a…`→`_native_ckpt_{A-dense,B-sparse,C-anneal}.pt`.

---

## Re-verify everything live

```bash
cd "C:\New folder\llm head\Hyperion-1.58b-Release"

# Claims 2 + 3 (+ python-side speed): the official battery (~10 min, CPU/GPU)
python alpha_p3a_after.py

# Claim 1 anchor + Claim 4 speed + Claim 2 engine samples: the C++ engine
cd _p3c_release
p3a_engine_fast.exe --hyper p3a_c2.hyper --addr p3a_addr.bin --val p3a_val.bin bpt
#   -> dense 9.2826 / oracle 8.6845 / addr C512 8.9316   (claim 1 parity)
p3a_engine_fast.exe --hyper p3a_c2.hyper --addr p3a_addr.bin --mode dense --prompt p3a_np_prompt.txt -n 120 gen
#   -> ~87 tok/s
p3a_engine_fast.exe --hyper p3a_c2.hyper --addr p3a_addr.bin --mode addr -C 512 --prompt p3a_np_prompt.txt -n 120 gen
#   -> ~150 tok/s   (claim 4)

# Claim 5: the flat-curve proof (pure dict tier, seconds, CPU)
python ..\prove_scale.py    # if moved to archive: see _provescale.txt for the recorded run
```

*Every number in this file was measured on this machine; the battery and
engine re-run cleanly from the commands above.*
