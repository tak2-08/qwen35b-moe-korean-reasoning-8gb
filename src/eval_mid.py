"""Ground truth for a weight edit: load the LoRA into the real llama.cpp engine and measure
  test_exact / test_first : held-out paraphrased questions about the new facts (all answer tokens / first token)
  train_tok               : next-token accuracy on the training statements
  bg_agree                : unrelated Wikipedia text (rows never used for fitting): same top-1 token as the unedited model
usage: eval_mid.py adapter.gguf [scale] [label]"""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host import Host
D = os.environ.get("HM_DATA", "data/hm"); O = os.path.join(D, "mid")
path = sys.argv[1]; scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
label = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(path)
dev = "cuda"
z = np.load(os.path.join(D, "cache.npz"))
tok, off, kind, sans, H0 = z["tok"], z["seq_off"], z["seq_kind"], z["seq_ans"], z["h"]
W = torch.from_numpy(np.load(os.path.join(D, "w_out.f16.npy"))).to(dev)
pos = np.load(os.path.join(O, "pos.npy")); inrow = np.zeros(len(tok), bool); inrow[pos] = True
bg = [s for s in range(len(kind)) if kind[s] == 2 and inrow[off[s]]]
bg_eval = bg[len(bg) * 2 // 3:]
te = [s for s in range(len(kind)) if kind[s] == 1]
tr = [s for s in range(len(kind)) if kind[s] == 0]
host = Host()
host.set_lora("" if path == "none" else path, scale)
t0 = time.time()


def pred(h):
    return (torch.from_numpy(h).to(dev).half() @ W.T).argmax(1).cpu().numpy()


ok_all = ok_first = 0
for s in te:
    a, b = int(off[s]), int(off[s + 1]); st = int(sans[s]) - 1
    h, _ = host.extract(tok[a:b].tolist())
    p = pred(h)[st:-1]; tg = tok[a + st + 1:b]
    ok_first += int(p[0] == tg[0]); ok_all += int((p == tg).all())
tr_ok = tr_n = 0
for s in tr:
    a, b = int(off[s]), int(off[s + 1])
    h, _ = host.extract(tok[a:b].tolist())
    tr_ok += int((pred(h)[:-1] == tok[a + 1:b]).sum()); tr_n += b - a - 1
ag = n = 0
for s in bg_eval:
    a, b = int(off[s]), int(off[s + 1])
    h, _ = host.extract(tok[a:b].tolist())
    ag += int((pred(h)[:-1] == pred(H0[a:b].astype(np.float32))[:-1]).sum()); n += b - a - 1
host.close()
res = dict(label=label, scale=scale, test_exact=round(ok_all / len(te), 4), test_first=round(ok_first / len(te), 4),
           train_tok=round(tr_ok / tr_n, 4), bg_agree=round(ag / n, 4), sec=round(time.time() - t0))
print(json.dumps(res))
with open(os.path.join(os.environ.get("RESULTS", "results"), "mid_real.jsonl"), "a", encoding="utf-8") as f:
    f.write(json.dumps(res) + "\n")
