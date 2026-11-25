"""Cache what the frozen-expert surrogate needs: the residual entering layer START (l_out-(START-1)) and every
routed-expert output ffn_moe_out-L for L >= START, plus the engine's final h. Optional LORA = refresh with an edit.
Sequences: all train statements, all test prompts, the first NBG background texts truncated to BGLEN tokens."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host import Host
D = os.environ.get("HM_DATA", "data/hm"); O = os.path.join(D, os.environ.get("SURR", "surr")); os.makedirs(O, exist_ok=True)
START = int(os.environ.get("START", "12")); NBG = int(os.environ.get("NBG", "90")); BGLEN = int(os.environ.get("BGLEN", "160"))
tag = os.environ.get("TAG", "base"); LORA = os.environ.get("LORA", "")
z = np.load(os.path.join(D, os.environ.get("CACHE", "cache.npz")))
tok, off, kind = z["tok"], z["seq_off"], z["seq_kind"]
seqs = [s for s in range(len(kind)) if kind[s] != 2] + [s for s in range(len(kind)) if kind[s] == 2][:NBG]
lens = [min(int(off[s + 1] - off[s]), BGLEN) if kind[s] == 2 else int(off[s + 1] - off[s]) for s in seqs]
roff = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
np.savez(os.path.join(O, "index.npz"), seqs=np.array(seqs), roff=roff, start=START)
N, LL = int(roff[-1]), list(range(START, 40))
names = [f"l_out-{START-1}"] + [f"ffn_moe_out-{l}" for l in LL]
X = np.lib.format.open_memmap(os.path.join(O, f"x_{tag}.npy"), "w+", np.float16, (N, 2048))
M = np.lib.format.open_memmap(os.path.join(O, f"moe_{tag}.npy"), "w+", np.float16, (len(LL), N, 2048))
Hh = np.lib.format.open_memmap(os.path.join(O, f"h_{tag}.npy"), "w+", np.float16, (N, 2048))
host = Host(capture=",".join(names))
if LORA:
    host.set_lora(LORA, 1.0)
t0 = time.time()
for j, s in enumerate(seqs):
    a, n = int(off[s]), lens[j]; r = int(roff[j])
    h, _ = host.extract(tok[a:a + n].tolist())
    X[r:r + n] = host.caps[0]
    for i in range(len(LL)):
        M[i, r:r + n] = host.caps[1 + i]
    Hh[r:r + n] = h
    if j % 500 == 0:
        print(j, "/", len(seqs), f"{time.time()-t0:.0f}s", flush=True)
host.close(); X.flush(); M.flush(); Hh.flush()
print("saved", N, "rows", f"{time.time()-t0:.0f}s", flush=True)
