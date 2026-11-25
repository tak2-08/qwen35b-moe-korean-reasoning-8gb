"""Train the REAL model's mid layers (shared-expert LoRA, layers START..39) with exact back-propagation through
attention / Gated DeltaNet / norms of every layer above, using the frozen-expert surrogate (qwen_torch.py).
Routed-expert outputs come from the engine (constants); background replay (KL to the unedited model) keeps
everyday text unchanged. Export = llama.cpp LoRA (blk.L.ffn_down_shexp.weight.lora_a/b), checked by eval_mid.py.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import gguf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_torch import QwenLayers, MODEL

D = os.environ.get("HM_DATA", "data/hm"); O = os.path.join(D, os.environ.get("SURR", "surr"))
ap = argparse.ArgumentParser()
ap.add_argument("--edit", default="12-39", help="layers whose shared expert gets a LoRA")
ap.add_argument("--targets", default="down", help="comma list of shared-expert matrices: down,up,gate")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=24)
ap.add_argument("--wd_bg", type=float, default=1.0)
ap.add_argument("--bg_len", type=int, default=48)
ap.add_argument("--tag", default="base")
ap.add_argument("--init", default="")
ap.add_argument("--save", default="")
ap.add_argument("--export", default="")
ap.add_argument("--eval_train", type=int, default=0, help="score train-token accuracy on only the first N train sequences")
ap.add_argument("--ans_only", action="store_true", help="loss only on tokens after seq_ans (the answer / solution)")
ap.add_argument("--dump", default="", help="write per-test-item correctness (json) after the last evaluation")
ap.add_argument("--out", default=os.path.join(os.environ.get("RESULTS", "results"), "surr_runs.jsonl"))
args = ap.parse_args()
dev = "cuda"; torch.manual_seed(0)

z = np.load(os.path.join(D, os.environ.get("CACHE", "cache.npz")))
TOK, off, kind, sans, fsq = z["tok"], z["seq_off"], z["seq_kind"], z["seq_ans"], z["seq_fact"]
DUMP = []
ix = np.load(os.path.join(O, "index.npz")); seqs, roff, START = ix["seqs"], ix["roff"], int(ix["start"])
X = np.load(os.path.join(O, f"x_{args.tag}.npy"), mmap_mode="r")
MOE = np.load(os.path.join(O, f"moe_{args.tag}.npy"), mmap_mode="r")
HB = np.load(os.path.join(O, "h_base.npy"), mmap_mode="r")
LL = list(range(START, 40))
e0, e1 = (int(v) for v in args.edit.split("-")); EDIT = [l for l in range(e0, e1 + 1) if l >= START]
m = QwenLayers(LL)
W = torch.from_numpy(np.load(os.path.join(D, "w_out.f16.npy"))).to(dev)
r = gguf.GGUFReader(MODEL)
gamma = torch.from_numpy(np.array(next(t for t in r.tensors if t.name == "output_norm.weight").data, dtype=np.float32)).to(dev); del r

# ---- sequences ----
J = {"train": [], "test": [], "bgfit": [], "bgeval": []}
bgs = [j for j, s in enumerate(seqs) if kind[s] == 2]
for j, s in enumerate(seqs):
    k = int(kind[s])
    if k in (0, 3): J["train"].append(j)
    elif k == 1: J["test"].append(j)
J["bgfit"], J["bgeval"] = bgs[: len(bgs) * 2 // 3], bgs[len(bgs) * 2 // 3:]


def batch(js, maxlen=None):
    """-> x [B,T,D], moe {il: [B,T,D]}, tokens [B,T], mask [B,T] (valid rows), base h [B,T,D]"""
    L = [int(roff[j + 1] - roff[j]) for j in js]
    if maxlen: L = [min(l, maxlen) for l in L]
    T = max(L); B = len(js)
    x = np.zeros((B, T, 2048), np.float16); mo = np.zeros((len(LL), B, T, 2048), np.float16); hb = np.zeros((B, T, 2048), np.float16)
    tk = np.zeros((B, T), np.int64); mk = np.zeros((B, T), bool)
    for b, (j, l) in enumerate(zip(js, L)):
        a = int(roff[j]); x[b, :l] = X[a:a + l]; mo[:, b, :l] = MOE[:, a:a + l]; hb[b, :l] = HB[a:a + l]
        s = int(seqs[j]); tk[b, :l] = TOK[off[s]:off[s] + l]; mk[b, :l] = True
    g = lambda v: torch.from_numpy(v).to(dev)
    return g(x).float(), {il: g(mo[i]) for i, il in enumerate(LL)}, g(tk), g(mk), g(hb)


# ---- LoRA on shared experts ----
TG = args.targets.split(","); R = args.rank
P = {}
for il in EDIT:
    for t in TG:
        fin, fout = (512, 2048) if t == "down" else (2048, 512)
        P[(il, t)] = [torch.nn.Parameter(torch.randn(R, fin, device=dev) / fin ** 0.5), torch.nn.Parameter(torch.zeros(fout, R, device=dev))]
if args.init:
    s0 = torch.load(args.init)
    for k, v in s0.items():
        P[k][0].data[:] = v[0].to(dev); P[k][1].data[:] = v[1].to(dev)


class Lora:
    """Adds LoRA deltas inside QwenLayers.shexp (down) and to its gate/up projections when requested."""
    def __call__(self, il, a):
        p = P.get((il, "down"))
        return 0 if p is None else (a @ p[0].T) @ p[1].T


lora = Lora()
_orig_mm = m.mm
def shexp_lora(il, xn, lo=None):
    gt = m.mm(xn, m.w(il, "ffn_gate_shexp.weight")); up = m.mm(xn, m.w(il, "ffn_up_shexp.weight"))
    if lo is not None and (il, "gate") in P: gt = gt + (xn @ P[(il, "gate")][0].T) @ P[(il, "gate")][1].T
    if lo is not None and (il, "up") in P: up = up + (xn @ P[(il, "up")][0].T) @ P[(il, "up")][1].T
    a = F.silu(gt) * up
    out = m.mm(a, m.w(il, "ffn_down_shexp.weight")) + (lora(il, a) if lo is not None else 0)
    return out * torch.sigmoid(xn @ m.w(il, "ffn_gate_inp_shexp.weight")).unsqueeze(-1)
m.shexp = shexp_lora


def logits_of(y):
    h = m.rms(y, gamma)
    return (h.half() @ W.T).float()


def run(js, use=True, maxlen=None):
    x, mo, tk, mk, hb = batch(js, maxlen)
    y = m.forward(x, mo, lora if use else None)
    return y, tk, mk, hb


@torch.no_grad()
def evaluate():
    ok_all = ok_first = n = 0
    T = sorted(J["test"], key=lambda j: int(roff[j + 1] - roff[j])); DUMP.clear()
    groups, cur = [], []                     # batches capped at ~6k padded tokens (exam prompts reach 1500 tokens)
    for j in T:
        if cur and (len(cur) + 1) * int(roff[j + 1] - roff[j]) > 6144: groups.append(cur); cur = []
        cur.append(j)
    if cur: groups.append(cur)
    for js in groups:
        y, tk, mk, _ = run(js)
        for b, j in enumerate(js):
            s = int(seqs[j]); st = int(sans[s]) - 1; l = int(mk[b].sum())
            pred = logits_of(y[b, st:l - 1]).argmax(1); tg = tk[b, st + 1:l]
            ok_first += int(pred[0] == tg[0]); ok_all += int((pred == tg).all()); n += 1
            DUMP.append(dict(seq=s, fact=int(fsq[s]), exact=bool((pred == tg).all()), first=bool(pred[0] == tg[0])))
    ag = cnt = 0
    for c in range(0, len(J["bgeval"]), 3):
        js = J["bgeval"][c:c + 3]
        y, tk, mk, hb = run(js)
        p1 = logits_of(y).argmax(-1); p0 = (hb.half() @ W.T).argmax(-1)
        ag += int(((p1 == p0) & mk).sum()); cnt += int(mk.sum())
    tr_ok = tr_n = 0
    TR = J["train"][:args.eval_train] if args.eval_train else J["train"]
    for c in range(0, len(TR), 4):
        js = TR[c:c + 4]
        y, tk, mk, _ = run(js)
        v = mk[:, 1:]
        h, tg = y[:, :-1][v], tk[:, 1:][v]
        for k in range(0, len(h), 512):   # vocab is 248k: never build logits for a whole batch at once
            tr_ok += int((logits_of(h[k:k + 512]).argmax(-1) == tg[k:k + 512]).sum())
        tr_n += int(v.sum())
    return dict(test_exact=round(ok_all / n, 4), test_first=round(ok_first / n, 4), train_tok=round(tr_ok / tr_n, 4), bg_agree=round(ag / cnt, 4))


params = [p for v in P.values() for p in v]
opt = torch.optim.Adam(params, lr=args.lr)
t0 = time.time()
rows = [dict(stage="start", **evaluate())]; print(json.dumps(rows[-1]), flush=True)
rng = np.random.default_rng(0)
for ep in range(1, args.epochs + 1):
    # length-bucketed batches: sequences of similar length together (exam solutions run 150-1500 tokens, so random
    # batches would mostly compute padding); the batch order is shuffled every epoch
    by_len = sorted(J["train"], key=lambda j: (int(roff[j + 1] - roff[j]), rng.random()))
    batches, cur = [], []   # at most bs sequences and ~TOKB padded tokens per batch (8 GB: backward through 28 layers)
    TOKB = int(os.environ.get("TOKB", "2400"))
    for j in by_len:
        if cur and (len(cur) >= args.bs or (len(cur) + 1) * int(roff[j + 1] - roff[j]) > TOKB): batches.append(cur); cur = []
        cur.append(j)
    if cur: batches.append(cur)
    for nb, bi in enumerate(rng.permutation(len(batches))):
        js = sorted(int(j) for j in batches[bi])
        y, tk, mk, _ = run(js)
        v = mk[:, 1:]
        if args.ans_only:  # learn the solution, not the question text (position p predicts token p+1)
            st = torch.tensor([max(1, int(sans[int(seqs[j])])) for j in js], device=dev)
            v = v & (torch.arange(v.shape[1], device=dev)[None, :] >= (st[:, None] - 1))
        h, tg = y[:, :-1][v], tk[:, 1:][v]
        # chunked, recomputed cross-entropy: logits for long solutions x 248k vocab would not fit in 8 GB
        ce = lambda hh, tt: F.cross_entropy(logits_of(hh), tt, reduction="sum")
        loss = sum(torch.utils.checkpoint.checkpoint(ce, h[k:k + 512], tg[k:k + 512], use_reentrant=False)
                   for k in range(0, len(h), 512)) / max(1, len(h))
        if args.wd_bg:
            bj = rng.choice(J["bgfit"], 4, replace=False).tolist()
            yb, _, mb, hb = run(bj, maxlen=args.bg_len)
            with torch.no_grad():
                pb = torch.softmax((hb[mb].half() @ W.T).float(), -1)
            loss = loss + args.wd_bg * F.kl_div(torch.log_softmax(logits_of(yb)[mb], -1), pb, reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step()
        if nb % 25 == 0: print(f"progress ep{ep} batch {nb}/{len(batches)} loss {float(loss):.3f} {time.time()-t0:.0f}s", flush=True)
    rows.append(dict(stage=f"ep{ep}", loss=round(float(loss), 4), sec=round(time.time() - t0), **evaluate()))
    print(json.dumps(rows[-1]), flush=True)
with open(args.out, "a", encoding="utf-8") as f:
    for x in rows:
        f.write(json.dumps(dict(vars(args), **x)) + "\n")
if args.dump:
    json.dump(DUMP, open(args.dump, "w"))
if args.save:
    torch.save({k: [v[0].detach().cpu(), v[1].detach().cpu()] for k, v in P.items()}, args.save)
if args.export:
    w = gguf.GGUFWriter(args.export, "qwen35moe")
    w.add_string("general.type", "adapter"); w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", float(R))
    name = {"down": "ffn_down_shexp", "up": "ffn_up_shexp", "gate": "ffn_gate_shexp"}
    for (il, t), (A, B) in P.items():
        # numpy (R, in) -> ne [in, R];  numpy (out, R) -> ne [R, out]
        w.add_tensor(f"blk.{il}.{name[t]}.weight.lora_a", A.detach().float().cpu().numpy())
        w.add_tensor(f"blk.{il}.{name[t]}.weight.lora_b", B.detach().float().cpu().numpy())
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("exported", args.export)
