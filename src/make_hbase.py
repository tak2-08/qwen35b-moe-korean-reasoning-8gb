"""h_base.npy for a refresh capture (chunks after the first): the trainer reads the unedited model's final hidden state
only on background rows (KL replay target + everyday-text agreement), so only those rows are computed here, with the
base model and no adapter; every other row stays zero. Uses the index.npz written by extract_surr.py in SURR.
Usage: CACHE=... SURR=... python make_hbase.py"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host import Host
D = os.environ.get("HM_DATA", "data/hm"); O = os.path.join(D, os.environ["SURR"])
z = np.load(os.path.join(D, os.environ["CACHE"]))
tok, off, kind = z["tok"], z["seq_off"], z["seq_kind"]
ix = np.load(os.path.join(O, "index.npz")); seqs, roff = ix["seqs"], ix["roff"]
N = int(roff[-1])
H = np.lib.format.open_memmap(os.path.join(O, "h_base.npy"), "w+", np.float16, (N, 2048))
host = Host(); t0 = time.time(); n = 0
for j, s in enumerate(seqs):
    if kind[s] != 2: continue
    r, l = int(roff[j]), int(roff[j + 1] - roff[j])
    h, _ = host.extract(tok[int(off[s]):int(off[s]) + l].tolist())
    H[r:r + l] = h; n += 1
host.close(); H.flush()
print("h_base rows for", n, "background sequences", f"{time.time()-t0:.0f}s", flush=True)
