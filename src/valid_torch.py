import sys, os, itertools, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_torch import QwenLayers
V = np.load(os.path.join(os.environ.get("HM_DATA", "data/hm"), "valid.npz"))
seqs = sorted({k.split("|")[0] for k in V.files})
T = lambda k: torch.from_numpy(V[k]).cuda().float()
def rel(a, b): return float((a - b).norm() / b.norm())
m = QwenLayers([0, 3])
for bc, rp, qs in itertools.product(["interleave", "tile"], ["neox", "normal"], [True, False]):
    m.bcast, m.rope, m.qscale = bc, rp, qs
    e0 = e3 = 0
    for s in seqs:
        x0 = T(f"{s}|attn_residual-0") - T(f"{s}|linear_attn_out-0")
        xn = m.rms(x0, m.w(0, "attn_norm.weight"))
        e_norm = rel(xn, T(f"{s}|attn_norm-0"))
        with torch.no_grad():
            e0 += rel(m.linear_attn(0, xn), T(f"{s}|linear_attn_out-0"))
            xn3 = m.rms(T(f"{s}|l_out-2"), m.w(3, "attn_norm.weight"))
            e3 += rel(m.full_attn(3, xn3), T(f"{s}|attn_output-3"))
    print(bc, rp, qs, "norm", round(e_norm, 5), "linear0", round(e0 / len(seqs), 4), "full3", round(e3 / len(seqs), 4))
# shared expert + full layer check
m.bcast, m.rope, m.qscale = "interleave", "neox", True
with torch.no_grad():
    for s in seqs:
        y = m.layer(3, T(f"{s}|l_out-2"), T(f"{s}|ffn_moe_out-3"))
        print(s, "layer3 l_out rel err", round(rel(y, T(f"{s}|l_out-3")), 5))
