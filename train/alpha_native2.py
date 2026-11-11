"""
alpha_native — THE BORN-SPARSE EXPERIMENT (phase-2 gate).

Question: quality-at-small-k was impossible by conversion (4 doors, all
measured). Does it EXIST when the model is TRAINED with top-k selection
active from birth?

Twin char-level transformers (identical seed/data/arch):
  A) dense-native: trained with full FFN; evaluated at k sweep (the cliff)
  B) sparse-native: trained with top-k=64 INSIDE the forward pass
Metrics: val bits/char at k in {32,64,128,256,full}; energy curves.

GATE: B@k64 vs A@k64 (must be far better) and vs A@full (how close).
Run:  python alpha_native.py          (trains A then B, ~30-45 min GPU)
"""
import os, time, json, math, glob
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", r"D:\PactGGUF\data\hfcache")
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "_native.log")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- config ----------------
D, LAYER, HEADS, FFN, SEQ = 256, 6, 4, 1024, 256
STEPS, BATCH, LR = 6000, 64, 3e-4
KS_EVAL = (16, 32, 48, 64, 128)
K_TRAIN_SPARSE = 64
ANNEAL = {0: 64, 3000: 48, 4500: 32}   # born 64 -> weaned 32
SEED = 7


def log(m):
    line = time.strftime("%H:%M:%S ") + str(m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_text():
    p = glob.glob(os.path.join(
        os.environ["HF_HOME"], "datasets--Salesforce--wikitext", "snapshots",
        "*", "wikitext-2-raw-v1", "train-00000-of-00001.parquet"))[0]
    import pandas as pd
    df = pd.read_parquet(p)
    return "\n".join(t for t in df["text"] if len(t.strip()) > 0)


def data():
    text = load_text()
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    n_val = len(ids) // 20
    train, val = ids[:-n_val], ids[-n_val:]
    log(f"chars={len(ids):,} vocab={len(chars)}")
    return train.to(DEV), val.to(DEV), len(chars)


class Block(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.gate = nn.Linear(d, ffn)
        self.up = nn.Linear(d, ffn)
        self.down = nn.Linear(ffn, d)
        self.pos_bias = None

    def forward(self, x, k):
        h = self.ln1(x)
        B, T, C = h.shape
        q = self.attn.in_proj_weight[:C]
        k_w = self.attn.in_proj_weight[C:2*C]
        v_w = self.attn.in_proj_weight[2*C:]
        qkv = F.linear(h, torch.cat([q, k_w, v_w])) + self.attn.in_proj_bias
        q, kk, v = qkv.chunk(3, dim=-1)
        hd = C // 4
        q = q.view(B, T, 4, hd).transpose(1, 2)
        kk = kk.view(B, T, 4, hd).transpose(1, 2)
        v = v.view(B, T, 4, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, kk, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, C)
        a = F.linear(a, self.attn.out_proj.weight, self.attn.out_proj.bias)
        x = x + a
        h = self.ln2(x)
        act = F.silu(self.gate(h)) * self.up(h)
        if k is not None and k < act.shape[-1]:
            idx = act.abs().topk(k, dim=-1).indices
            mask = torch.zeros_like(act).scatter_(-1, idx, 1.0)
            act = act * mask
        return x + self.down(act)


class TinyLM(nn.Module):
    def __init__(self, vocab, d, layers, heads, ffn):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList([Block(d, heads, ffn)
                                     for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.head.weight = self.emb.weight      # tied

    def forward(self, ids, k=None):
        T = ids.shape[1]
        x = self.emb(ids) + self.pos(torch.arange(T, device=ids.device))
        for b in self.blocks:
            x = b(x, k)
        return self.head(self.ln(x))


def batch_of(data, n):
    i = torch.randint(0, len(data) - SEQ - 1, (n,), device=data.device)
    off = torch.arange(SEQ, device=data.device)
    seq = data[i[:, None] + off[None, :]]
    tgt = data[i[:, None] + off[None, :] + 1]
    return seq, tgt


@torch.no_grad()
def evaluate(model, val, k, nb=24):
    model.eval()
    tot, cnt = 0.0, 0
    base = torch.randint(0, len(val) - SEQ - 1, (nb,), device=val.device)
    for j in base:
        seq = val[j:j + SEQ].unsqueeze(0)
        tgt = val[j + 1:j + SEQ + 1].unsqueeze(0)
        out = model(seq, k)
        tot += F.cross_entropy(out[0], tgt[0], reduction="sum").item()
        cnt += SEQ
    model.train()
    return tot / cnt / math.log(2)     # bits/char


@torch.no_grad()
def energy_at(model, val, k, nb=16):
    model.eval()
    fr = 0.0
    for j in torch.randint(0, len(val) - SEQ - 1, (nb,), device=val.device):
        seq = val[j:j + SEQ].unsqueeze(0)
        x = model.emb(seq) + model.pos(torch.arange(SEQ, device=seq.device))
        # capture acts of first block only (representative)
        h = model.blocks[0].ln2(x)
        act = F.silu(model.blocks[0].gate(h)) * model.blocks[0].up(h)
        srt = act.abs().sort(dim=-1, descending=True).values
        fr += (srt[:, :, :k].sum() / srt.sum()).mean().item()
    model.train()
    return fr / nb


def train(tag, train_data, val, vocab, k_train):
    torch.manual_seed(SEED)
    model = TinyLM(vocab, D, LAYER, HEADS, FFN).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    ckpt = os.path.join(HERE, f"_native_ckpt_{tag}.pt")
    start = 1
    if os.path.exists(ckpt):
        st_ = torch.load(ckpt, map_location=DEV, weights_only=False)
        model.load_state_dict(st_["m"]); opt.load_state_dict(st_["o"])
        sched.load_state_dict(st_["s"]); start = st_["step"] + 1
        log(f"[{tag}] resumed from step {start}")
    t0 = time.time()
    def k_at(step):
        ks = [v for s_, v in ANNEAL.items() if step >= s_]
        return ks[-1] if k_train is not None else None
    for step in range(start, STEPS + 1):
        k_now = k_at(step)
        seq, tgt = batch_of(train_data, BATCH)
        loss = F.cross_entropy(model(seq, k_now).flatten(0, 1), tgt.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == 1:
            bpc = evaluate(model, val, k_train)
            log(f"[{tag}] step {step:5d} loss={loss.item():.4f} "
                f"val@k={k_train}:{bpc:.4f} bpc ({time.time()-t0:.0f}s)")
            torch.save({"m": model.state_dict(), "o": opt.state_dict(),
                        "s": sched.state_dict(), "step": step}, ckpt)
    res = {"params": nparam, "k_train": k_train}
    for k in KS_EVAL:
        res[f"bpc_k{k}"] = round(evaluate(model, val, k), 4)
    res["energy64"] = round(energy_at(model, val, 64), 4)
    log(f"[{tag}] FINAL: {res}")
    del model, opt
    torch.cuda.empty_cache()
    return res


def main():
    log(f"=== ALPHA NATIVE: born-sparse gate (D={D} L={LAYER} FFN={FFN} "
        f"steps={STEPS}) ===")
    train_data, val, vocab = data()
    out = {}
    out["A_dense"] = train("A-dense", train_data, val, vocab, None)
    out["C_anneal"] = train("C-anneal", train_data, val, vocab, K_TRAIN_SPARSE)
    a, b = out["A_dense"], out["B_sparse64"]
    log("---- GATE ----")
    log(f"A@full={a['bpc_k' + str(FFN)]}  A@64={a['bpc_k64']}  "
        f"B@64={b['bpc_k64']}  (lower=better)")
    log(f"cliff-closed: A@64 - B@64 = {a['bpc_k64'] - b['bpc_k64']:.4f} bpc")
    with open(os.path.join(HERE, "_native.json"), "w") as f:
        json.dump(out, f, indent=1)
    log("done -> _native.json")


if __name__ == "__main__":
    main()
