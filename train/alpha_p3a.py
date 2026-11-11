"""
alpha_p3a — PHASE-3-A: token-level row-native twin pair (the real gate).

BPE-16k (trained on our corpus), 8L x d512, FFN 2048, seq 256, ~36M params.
  A: dense twin.  C: row-native, k anneal {64->48->32}, top-k in forward.
PASS = C@64 within 10% of A-full ppl. Checkpoints every 250 steps.
Queued behind C-anneal by the launcher. Log: _p3a.log
"""
import os, time, json, math, glob
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", r"D:\PactGGUF\data\hfcache")
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "_p3a.log")
DEV = "cuda"
D, LAYER, HEADS, FFN, SEQ = 512, 8, 8, 2048, 256
STEPS, BATCH, LR = 9000, 32, 3e-4
KS_EVAL = (16, 32, 48, 64, 128, FFN)
ANNEAL = {0: 64, 4000: 48, 6500: 32}
SEED = 7


def log(m):
    line = time.strftime("%H:%M:%S ") + str(m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def corpus_text():
    p = glob.glob(os.path.join(
        os.environ["HF_HOME"], "datasets--Salesforce--wikitext", "snapshots",
        "*", "wikitext-2-raw-v1", "train-00000-of-00001.parquet"))[0]
    import pandas as pd
    df = pd.read_parquet(p)
    return [t for t in df["text"] if len(t.strip()) > 0]


def build_tok():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    path = os.path.join(HERE, "_p3a_bpe.json")
    if os.path.exists(path):
        return Tokenizer.from_file(path), None
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=16000,
                             special_tokens=["<|endoftext|>"])
    chunks = corpus_text()
    tk.train_from_iterator(chunks, tr)
    tk.save(path)
    return tk, chunks


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(D)
        self.qkv = torch.nn.Linear(D, 3 * D)
        self.proj = torch.nn.Linear(D, D)
        self.ln2 = torch.nn.LayerNorm(D)
        self.gate = torch.nn.Linear(D, FFN)
        self.up = torch.nn.Linear(D, FFN)
        self.down = torch.nn.Linear(FFN, D)

    def forward(self, x, k):
        h = self.ln1(x)
        B, T, C = h.shape
        q, kk, v = F.linear(h, self.qkv.weight, self.qkv.bias).chunk(3, -1)
        hd = C // HEADS
        q = q.view(B, T, HEADS, hd).transpose(1, 2)
        kk = kk.view(B, T, HEADS, hd).transpose(1, 2)
        v = v.view(B, T, HEADS, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, kk, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        h = self.ln2(x)
        act = F.silu(self.gate(h)) * self.up(h)
        if k is not None and k < act.shape[-1]:
            idx = act.abs().topk(k, -1).indices
            act = act * torch.zeros_like(act).scatter_(-1, idx, 1.0)
        return x + self.down(act)


class LM(torch.nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = torch.nn.Embedding(V, D)
        self.pos = torch.nn.Embedding(SEQ, D)
        self.blocks = torch.nn.ModuleList([Block() for _ in range(LAYER)])
        self.ln = torch.nn.LayerNorm(D)
        self.head = torch.nn.Linear(D, V)
        self.head.weight = self.emb.weight

    def forward(self, ids, k=None):
        T = ids.shape[1]
        x = self.emb(ids) + self.pos(torch.arange(T, device=ids.device))
        for b in self.blocks:
            x = b(x, k)
        return self.head(self.ln(x))


@torch.no_grad()
def evaluate(m, val, k, nb=16):
    m.eval(); tot = 0.0; cnt = 0
    for j in range(0, nb):
        s = val[j * SEQ:(j + 1) * SEQ].unsqueeze(0)
        t = val[j * SEQ + 1:(j + 1) * SEQ + 1].unsqueeze(0)
        tot += F.cross_entropy(m(s, k)[0], t[0], reduction="sum").item()
        cnt += SEQ
    m.train()
    return tot / cnt / math.log(2)


def k_at(step, active):
    if not active:
        return None
    return [v for s, v in ANNEAL.items() if step >= s][-1]


def train(tag, ids, val, V, active):
    torch.manual_seed(SEED)
    m = LM(V).to(DEV)
    log(f"[{tag}] params={sum(p.numel() for p in m.parameters()):,}")
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    ck = os.path.join(HERE, f"_p3a_ckpt_{tag}.pt")
    start = 1
    if os.path.exists(ck):
        s = torch.load(ck, map_location=DEV, weights_only=False)
        m.load_state_dict(s["m"]); opt.load_state_dict(s["o"])
        sch.load_state_dict(s["s"]); start = s["step"] + 1
        log(f"[{tag}] resumed from {start}")
    off = torch.arange(SEQ, device=DEV)
    t0 = time.time()
    for step in range(start, STEPS + 1):
        i = torch.randint(0, len(ids) - SEQ - 1, (BATCH,), device=DEV)
        seq, tgt = ids[i[:, None] + off], ids[i[:, None] + off + 1]
        loss = F.cross_entropy(m(seq, k_at(step, active)).flatten(0, 1),
                               tgt.flatten())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 250 == 0 or step == 1:
            bpc = evaluate(m, val, k_at(step, active))
            log(f"[{tag}] step {step:5d} loss={loss.item():.4f} "
                f"val@k={k_at(step, active)}:{bpc:.4f} bpt ({time.time()-t0:.0f}s)")
            torch.save({"m": m.state_dict(), "o": opt.state_dict(),
                        "s": sch.state_dict(), "step": step}, ck)
    res = {}
    for k in KS_EVAL:
        res[f"k{k}"] = round(evaluate(m, val, k), 4)
    log(f"[{tag}] FINAL: {res}")
    return res


def main():
    log(f"=== P3-A: token-level twins (D={D} L={LAYER} FFN={FFN} "
        f"steps={STEPS}) ===")
    tk, _ = build_tok()
    chunks = corpus_text()
    enc = tk.encode_batch(chunks)
    ids = torch.tensor([t for e in enc for t in e.ids], dtype=torch.long)
    nval = len(ids) // 20
    val, tr = ids[-nval:].to(DEV), ids[:-nval].to(DEV)
    log(f"tokens={len(ids):,} vocab={tk.get_vocab_size()}")
    out = {}
    out["A_dense"] = train("A", tr, val, tk.get_vocab_size(), False)
    out["C_rownative"] = train("C", tr, val, tk.get_vocab_size(), True)
    log("---- P3-A GATE ----")
    log(f"A@full={out['A_dense'][f'k{FFN}']}  C@64={out['C_rownative']['k64']}"
        f"  C@32={out['C_rownative']['k32']}")
    with open(os.path.join(HERE, "_p3a.json"), "w") as f:
        json.dump(out, f, indent=1)
    log("done -> _p3a.json")


if __name__ == "__main__":
    main()
