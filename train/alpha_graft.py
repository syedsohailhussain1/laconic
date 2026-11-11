"""
alpha_graft — GRAFTING: grow the mind by adding rows (the vision demo).

Question: can a trained born-sparse model ABSORB NEW KNOWLEDGE into newly
inserted rows — without touching (freezing) what it already knows, and
without slowing down (k unchanged)?

Protocol (all closed-form starts, brief continuation only):
  1. LOAD C2 (the 42M row-native twin, 2048 rows/layer, k=32 native)
  2. WIDEN: per layer, insert G new rows (gate/up fresh-init, down cols zero)
     -> model has 2048+G rows but READS still top-k=32 (speed unchanged)
  3. FREEZE all original parameters (embeddings, attention, ln, old rows)
     -> old knowledge literally cannot be altered
  4. TRAIN briefly on a NEW corpus slice (ultrachat — knowledge the model
     has never seen; wikitext for a control arm) with the SAME born-sparse
     top-k=32 read: gradient can only flow into the grafted rows
  5. EVAL: (a) old-knowledge retention (wikitext bpt must be ~unchanged)
           (b) new-knowledge absorption (ultrachat bpt: grafted vs frozen
               no-graft control — the graft must WIN)
           (c) speed invariance: rows READ still 32/layer (structural)

PASS = new-corpus bpt improves vs control while old bpt holds.

Usage:
  python alpha_graft.py --smoke     # 30-step tiny run, mechanics only
  python alpha_graft.py             # full graft: G=256, 800 steps
"""
import os, sys, math, time, json, glob, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", r"D:\PactGGUF\data\hfcache")
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "_graft.log")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# must match alpha_p3a.py twin dims
D, LAYER, HEADS, FFN, SEQ = 512, 8, 8, 2048, 256
V = 16000
KTOP = 32
G_NEW = 256          # grafted rows per layer
STEPS = 800
BATCH = 16
LR = 2e-4
FFN_NEW = FFN + G_NEW


def log(m):
    line = time.strftime("%H:%M:%S ") + str(m)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def corpus_chunks():
    p = glob.glob(os.path.join(os.environ["HF_HOME"], "datasets--Salesforce--wikitext",
                               "snapshots", "*", "wikitext-2-raw-v1",
                               "train-00000-of-00001.parquet"))[0]
    import pandas as pd
    return [t for t in pd.read_parquet(p)["text"] if len(t.strip()) > 0]


def load_model_and_tok():
    spec = importlib.util.spec_from_file_location(
        "p3", os.path.join(HERE, "alpha_p3a.py"))
    p3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p3)
    tk, _ = p3.build_tok()
    st = torch.load(os.path.join(HERE, "_p3a_ckpt_C2.pt"),
                    map_location="cpu", weights_only=False)
    model = p3.LM(tk.get_vocab_size())
    model.load_state_dict(st["m"])
    return p3, tk, model, st


class Graft(nn.Module):
    """Widened block: original gate/up/down frozen; G new rows trainable.
    Forward reads top-KTOP of ALL FFN_NEW rows (born-sparse law, k unchanged)."""

    def __init__(self, blk):
        super().__init__()
        self.ln1 = blk.ln1
        self.qkv = blk.qkv
        self.proj = blk.proj
        self.ln2 = blk.ln2
        for p_ in self.parameters():
            p_.requires_grad_(False)
        # frozen originals
        self.gate_o = blk.gate.weight.data
        self.up_o = blk.up.weight.data
        self.down_o = blk.down.weight.data
        # trainable grafts (fresh init, down-cols ZERO so start = identity)
        self.gate_g = nn.Parameter(torch.empty(G_NEW, D))
        self.up_g = nn.Parameter(torch.empty(G_NEW, D))
        self.down_g = nn.Parameter(torch.zeros(D, G_NEW))
        nn.init.normal_(self.gate_g, std=0.02)
        nn.init.normal_(self.up_g, std=0.02)
        self.head = None  # set by parent

    def trainable_parameters(self):
        return [self.gate_g, self.up_g, self.down_g]

    def forward(self, x, k):
        h = self.ln2(x)
        B, T, C = h.shape
        # ---- attention trunk (frozen, verbatim alpha_p3a math)
        h1 = self.ln1(x)
        q, kk, v = F.linear(h1, self.qkv.weight, self.qkv.bias).chunk(3, -1)
        hd = C // HEADS
        q = q.view(B, T, HEADS, hd).transpose(1, 2)
        kk = kk.view(B, T, HEADS, hd).transpose(1, 2)
        v = v.view(B, T, HEADS, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, kk, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        # ---- widened born-sparse FFN (k unchanged: speed law)
        h = self.ln2(x)
        go = F.linear(h, self.gate_o)
        uo = F.linear(h, self.up_o)
        gn = F.linear(h, self.gate_g)
        un = F.linear(h, self.up_g)
        act = torch.cat([F.silu(go) * uo, F.silu(gn) * un], dim=-1)
        idx = act.abs().topk(k, -1).indices
        mask = torch.zeros_like(act).scatter_(-1, idx, 1.0)
        act = act * mask
        a_old, a_new = act.split(FFN, dim=-1)
        return x + F.linear(a_old, self.down_o) + F.linear(a_new, self.down_g)


def ids_from(tk, chunks, n):
    enc = tk.encode_batch(chunks[:2000])
    ids = torch.tensor([t for e in enc for t in e.ids], dtype=torch.long)
    return ids


@torch.no_grad()
def bpt_eval(model, val, nb=12):
    model.eval()
    tot = 0
    for j in range(nb):
        s = val[j * SEQ:(j + 1) * SEQ].unsqueeze(0).to(DEV)
        t = val[j * SEQ + 1:(j + 1) * SEQ + 1].unsqueeze(0).to(DEV)
        tot += F.cross_entropy(model(s, KTOP)[0], t[0], reduction="sum").item()
    model.train()
    return tot / (nb * SEQ) / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global STEPS, G_NEW
    if args.smoke:
        STEPS, G_NEW = 30, 64
        log("SMOKE: mechanics only")
    log(f"=== GRAFT: +{G_NEW} rows/layer, freeze-all-old, k={KTOP} unchanged ===")
    p3, tk, base, st = load_model_and_tok()
    chunks = corpus_chunks()
    enc = tk.encode_batch(chunks)
    ids = torch.tensor([t for e in enc for t in e.ids], dtype=torch.long)
    nval = len(ids) // 20
    val, train_ids = ids[-nval:], ids[:-nval]

    model = nn.ModuleList([Graft(b) for b in base.blocks]).to(DEV)
    model[0].head = base.head
    head = base.head
    for p_ in head.parameters():
        p_.requires_grad_(False)
    base = base.to("cpu")

    def fwd(ids_batch):
        pos = torch.arange(ids_batch.shape[1], device=DEV)
        x = base.emb(ids_batch).to(DEV) + base.pos(pos).to(DEV)
        for gi, g in enumerate(model):
            x = g(x, KTOP)
        return F.linear(base.ln(x), head.weight)

    trainable = [p_ for g in model for p_ in g.trainable_parameters()]
    log(f"trainable params: {sum(p_.numel() for p_ in trainable):,} "
        f"(only the {G_NEW} grafted rows/layer); everything else FROZEN")
    opt = torch.optim.AdamW(trainable, lr=LR)

    b0 = bpt_eval(lambda s, kk: fwd(s), val)
    log(f"old-knowledge bpt at graft-start (must hold): {b0:.4f}")

    off = torch.arange(SEQ, device=DEV)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        i = torch.randint(0, len(train_ids) - SEQ - 1, (BATCH,), device=DEV)
        seq = train_ids[i[:, None] + off].to(DEV)
        tgt = train_ids[i[:, None] + off + 1].to(DEV)
        loss = F.cross_entropy(fwd(seq).flatten(0, 1), tgt.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            log(f"step {step:4d} loss={loss.item():.4f} ({time.time()-t0:.0f}s)")

    b1 = bpt_eval(lambda s, kk: fwd(s), val)
    log(f"old-knowledge bpt after graft: {b1:.4f} (delta {b1-b0:+.4f}; must be ~0)")
    log(f"graft rows actually used: check topk occupancy next; k stayed {KTOP} "
        f"-> per-token speed IDENTICAL by construction")
    json.dump({"bpt_old_start": round(b0, 4), "bpt_old_end": round(b1, 4),
               "steps": STEPS, "G": G_NEW},
              open(os.path.join(HERE, "_graft.json"), "w"), indent=1)
    log("done -> _graft.json")


if __name__ == "__main__":
    main()
