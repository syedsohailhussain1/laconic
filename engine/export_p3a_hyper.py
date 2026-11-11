"""Export the 42M C2 checkpoint + addressers into the existing Hyperion
HYP2 .hyper format (all type-0 FP32) + a flat addresser bin, so the
existing C++ engine family can load it. Reads only; writes to temp."""
import os, struct, importlib.util
import numpy as np
import torch

HERE = r"C:\New folder\llm head\Hyperion-1.58b-Release"
FDIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", r"D:\PactGGUF\data\hfcache")
spec = importlib.util.spec_from_file_location("p3", os.path.join(HERE, "alpha_p3a.py"))
p3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p3)

V = 16000
s = torch.load(os.path.join(HERE, "_p3a_ckpt_C2.pt"), map_location="cpu", weights_only=False)
sd = {k: v for k, v in s["m"].items() if not k.startswith("head.")}
del s

# HYP2 writer: magic, vocab, n_tensors, then per tensor:
#   name_len, name, type(0), ndims, dims..., raw f32
out = os.path.join(FDIR, "p3a_c2.hyper")
tensors = []
for k, v in sd.items():
    tensors.append((k, v.float().contiguous()))
with open(out, "wb") as f:
    f.write(b"HYP2")
    f.write(struct.pack("<II", V, len(tensors)))
    for name, t in tensors:
        nb = name.encode()
        f.write(struct.pack("<I", len(nb))); f.write(nb)
        f.write(struct.pack("<I", 0))                    # type 0 = FP32
        f.write(struct.pack("<I", t.dim()))
        for d in t.shape:
            f.write(struct.pack("<I", d))
        f.write(t.numpy().tobytes())
print(f"hyper: {out}  {os.path.getsize(out)/1e6:.1f} MB  {len(tensors)} tensors")

# addresser bin (C++-friendly flat dump):
#   u32 L; per layer: u32 k, u32 D, coef f32[k*D], b f32[k],
#   u32 nrows[k] (may be 0), then rows i32[k*maxn]
ab = os.path.join(FDIR, "p3a_addr.bin")
with open(ab, "wb") as f:
    f.write(struct.pack("<I", 8))
    for l in range(8):
        z = np.load(os.path.join(FDIR, f"_p3c_addr_l{l}.npz"))
        coef = z["coef"].astype(np.float32)
        b = z["b"].astype(np.float32)
        rows = z["rows_pad"].astype(np.int32)
        lens = z["lens"].astype(np.int32)
        k, D = coef.shape
        f.write(struct.pack("<III", k, D, rows.shape[1]))
        f.write(coef.tobytes()); f.write(b.tobytes())
        f.write(lens.tobytes())
        f.write(rows.tobytes())
print(f"addr:  {ab}  {os.path.getsize(ab)/1e6:.1f} MB")

# prompt: 32 real eval-window token ids, comma-separated (family convention)
val_ids = None
try:
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(HERE, "_p3a_bpe.json"))
    enc = tk.encode_batch(p3.corpus_text())
    ids = [t for e in enc for t in e.ids]
    val_ids = ids[-len(ids)//20:]
except Exception as e:
    print("tok fallback", e)
    val_ids = list(range(40, 72))
pout = os.path.join(FDIR, "p3a_prompt.txt")
with open(pout, "w") as f:
    f.write(",".join(str(x) for x in val_ids[:32]))
print("prompt:", pout)
