"""PyTorch re-implementation of Qwen3.6-35B-A3B (qwen35moe) decoder layers, matched to llama.cpp numerics.

Only the parts that are cheap to hold on an 8 GB GPU are implemented with weights: Gated DeltaNet (linear
attention), gated full attention, norms and the shared expert. The 256 routed experts are NOT loaded: their
output (ffn_moe_out) is taken from the real engine and treated as a constant (frozen-expert surrogate).
That gives exact gradients through attention / DeltaNet / norms / shared experts of every layer above the
edited one, so mid-layer weights can be trained without the 12.7 GB expert banks ever entering GPU memory.
"""
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import gguf


def delta_rule_loop(q, k, v, g, beta):
    """Gated delta rule, token by token (reference). q,k,v [B,T,H,D]; g log-decay [B,T,H]; beta [B,T,H]."""
    B, T, H, Dk = k.shape
    S = torch.zeros(B, H, Dk, v.shape[-1], device=k.device, dtype=torch.float32)
    dec = torch.exp(g); outs = []
    for t in range(T):
        S = S * dec[:, t].view(B, H, 1, 1)
        kt = k[:, t]
        delta = (v[:, t] - torch.einsum("bhkv,bhk->bhv", S, kt)) * beta[:, t].unsqueeze(-1)
        S = S + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t]))
    return torch.stack(outs, 1)


def delta_rule_chunked(q, k, v, g, beta, C=64):
    """Same recurrence computed 64 tokens at a time (WY / UT-transform form of the gated delta rule):
    inside a chunk everything is matrix products, only the chunk-to-chunk state is sequential. Exact up to fp."""
    B, T, H, Dk = k.shape; Dv = v.shape[-1]
    pad = (C - T % C) % C
    tr = lambda a: a.transpose(1, 2).float()                               # -> [B, H, T, ...]
    q, k, v, g, beta = tr(q), tr(k), tr(v), g.transpose(1, 2).float(), beta.transpose(1, 2).float()
    if pad:
        q, k, v = (F.pad(a, (0, 0, 0, pad)) for a in (q, k, v)); g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    N = (T + pad) // C
    q, k, v = (a.reshape(B, H, N, C, -1) for a in (q, k, v)); g, beta = g.reshape(B, H, N, C), beta.reshape(B, H, N, C)
    g = g.cumsum(-1)                                                        # within-chunk cumulative log decay
    kb, vb = k * beta[..., None], v * beta[..., None]
    low = torch.tril(torch.ones(C, C, dtype=torch.bool, device=k.device), -1)
    dm = torch.exp((g[..., :, None] - g[..., None, :]).masked_fill(~torch.tril(torch.ones_like(low)), -1e30))  # decay i<-j, j<=i
    L = ((kb @ k.transpose(-1, -2)) * dm).masked_fill(~low, 0)
    eye = torch.eye(C, device=k.device).expand_as(L)
    Tm = torch.linalg.solve_triangular(eye + L, eye, upper=False)          # (I + L)^-1
    u = Tm @ vb                                                             # pseudo-values
    w = Tm @ (kb * torch.exp(g)[..., None])
    S = torch.zeros(B, H, Dk, Dv, device=k.device)
    o = []
    for i in range(N):
        qi, ki, gi = q[:, :, i], k[:, :, i], g[:, :, i]
        att = (qi @ ki.transpose(-1, -2) * dm[:, :, i]).masked_fill(~torch.tril(torch.ones_like(low)), 0)
        vn = u[:, :, i] - w[:, :, i] @ S
        o.append((qi * torch.exp(gi)[..., None]) @ S + att @ vn)
        S = S * torch.exp(gi[..., -1])[..., None, None] + (ki * torch.exp(gi[..., -1:] - gi)[..., None]).transpose(-1, -2) @ vn
    o = torch.stack(o, 2).reshape(B, H, N * C, Dv)[:, :, :T]
    return o.transpose(1, 2)

MODEL = os.environ.get("MODEL", "models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf")


class QwenLayers:
    def __init__(self, layers, dev="cuda", dtype=torch.float16, bcast="tile", rope="neox", qscale=True):
        r = gguf.GGUFReader(MODEL)
        f = r.fields
        g = lambda k: f[k].parts[-1][0]
        self.eps = float(g("qwen35moe.attention.layer_norm_rms_epsilon"))
        self.n_rot = int(g("qwen35moe.rope.dimension_count"))
        self.theta = float(g("qwen35moe.rope.freq_base"))
        self.interval = int(g("qwen35moe.full_attention_interval"))
        self.dev, self.dtype, self.bcast, self.rope, self.qscale = dev, dtype, bcast, rope, qscale
        tens = {t.name: t for t in r.tensors}
        self.W = {}
        for il in layers:
            for t in [n for n in tens if n.startswith(f"blk.{il}.") and "_exps" not in n and "ffn_gate_inp.weight" not in n]:
                x = tens[t]
                a = gguf.quants.dequantize(x.data, x.tensor_type) if x.tensor_type.name not in ("F32", "F16") else np.array(x.data, dtype=np.float32)
                a = torch.from_numpy(np.ascontiguousarray(a)).to(dev)
                quant = x.tensor_type.name not in ("F32", "F16")
                self.W[t] = a.to(dtype) if (a.ndim == 2 and quant) else a.float()
        self.layers = list(layers)

    def is_full(self, il):
        return (il + 1) % self.interval == 0

    def w(self, il, name):
        return self.W[f"blk.{il}.{name}"]

    def rms(self, x, w):
        x32 = x.float()
        return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def mm(self, x, W):  # x [..., in] fp32, W [out, in]
        return (x.to(self.dtype) @ W.T).float()

    # ---------------- Gated DeltaNet ----------------
    def linear_attn(self, il, x):  # x: [B, T, 2048] (normed); right padding is harmless (causal)
        B, T = x.shape[:2]
        qkv = self.mm(x, self.w(il, "attn_qkv.weight"))                    # [B, T, 8192]
        z = self.mm(x, self.w(il, "attn_gate.weight"))                      # [B, T, 4096]
        beta = torch.sigmoid(x @ self.w(il, "ssm_beta.weight").T)          # [B, T, 32]
        alpha = x @ self.w(il, "ssm_alpha.weight").T + self.w(il, "ssm_dt.bias")
        gdec = F.softplus(alpha) * self.w(il, "ssm_a")                     # [B, T, 32] (log decay, <= 0)
        cw = self.w(il, "ssm_conv1d.weight")                                # [8192, 4]
        K = cw.shape[1]
        xp = F.pad(qkv.transpose(1, 2), (K - 1, 0))                         # [B, C, T+K-1]
        conv = F.silu(F.conv1d(xp, cw.unsqueeze(1), groups=cw.shape[0]).transpose(1, 2))  # [B, T, C]
        hk, nk, nv = 128, 16, 32
        q = conv[..., : hk * nk].reshape(B, T, nk, hk)
        k = conv[..., hk * nk: 2 * hk * nk].reshape(B, T, nk, hk)
        v = conv[..., 2 * hk * nk:].reshape(B, T, nv, hk)
        q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + self.eps)
        k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + self.eps)
        rep = nv // nk
        if self.bcast == "interleave":
            q, k = q.repeat_interleave(rep, 2), k.repeat_interleave(rep, 2)
        else:
            q, k = q.repeat(1, 1, rep, 1), k.repeat(1, 1, rep, 1)
        if self.qscale:
            q = q / math.sqrt(hk)
        o = (delta_rule_chunked if T > 16 else delta_rule_loop)(q, k, v, gdec, beta)   # [B, T, H, Dv]
        o = self.rms(o, self.w(il, "ssm_norm.weight")) * F.silu(z.view(B, T, nv, hk))
        return self.mm(o.reshape(B, T, nv * hk), self.w(il, "ssm_out.weight"))

    # ---------------- gated full attention ----------------
    def rope_apply(self, x, pos):  # x [B, T, H, D]
        n = self.n_rot
        inv = 1.0 / (self.theta ** (torch.arange(0, n, 2, device=x.device).float() / n))
        ang = pos.float()[:, None] * inv[None]                              # [T, n/2]
        cos, sin = ang.cos()[None, :, None], ang.sin()[None, :, None]
        xr, xp = x[..., :n], x[..., n:]
        if self.rope == "neox":
            a, b = xr[..., : n // 2], xr[..., n // 2:]
            xr = torch.cat([a * cos - b * sin, a * sin + b * cos], -1)
        else:
            a, b = xr[..., 0::2], xr[..., 1::2]
            xr = torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)
        return torch.cat([xr, xp], -1)

    def full_attn(self, il, x):  # x: [B, T, 2048]
        B, T = x.shape[:2]
        hd, nh, nkv = 256, 16, 2
        qf = self.mm(x, self.w(il, "attn_q.weight")).view(B, T, nh, 2 * hd)
        q, gate = qf[..., :hd], qf[..., hd:]
        k = self.mm(x, self.w(il, "attn_k.weight")).view(B, T, nkv, hd)
        v = self.mm(x, self.w(il, "attn_v.weight")).view(B, T, nkv, hd)
        q = self.rms(q, self.w(il, "attn_q_norm.weight"))
        k = self.rms(k, self.w(il, "attn_k_norm.weight"))
        pos = torch.arange(T, device=x.device)
        q, k = self.rope_apply(q, pos), self.rope_apply(k, pos)
        k = k.repeat_interleave(nh // nkv, 2); v = v.repeat_interleave(nh // nkv, 2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
                                           scale=1.0 / math.sqrt(hd)).transpose(1, 2)
        o = o * torch.sigmoid(gate)
        return self.mm(o.reshape(B, T, nh * hd), self.w(il, "attn_output.weight"))

    # ---------------- shared expert ----------------
    def shexp(self, il, xn, lora=None):
        gt = self.mm(xn, self.w(il, "ffn_gate_shexp.weight"))
        up = self.mm(xn, self.w(il, "ffn_up_shexp.weight"))
        a = F.silu(gt) * up                                                 # [T, 512]
        out = self.mm(a, self.w(il, "ffn_down_shexp.weight"))
        if lora is not None:
            out = out + lora(il, a)
        g = torch.sigmoid(xn @ self.w(il, "ffn_gate_inp_shexp.weight"))
        return out * g.unsqueeze(-1)

    def layer(self, il, x, moe_out, lora=None, attn_lora=None):
        """x: residual input [T, 2048]; moe_out: routed-expert output of this layer (constant from the engine)."""
        xn = self.rms(x, self.w(il, "attn_norm.weight"))
        a = self.full_attn(il, xn) if self.is_full(il) else self.linear_attn(il, xn)
        r = x + a
        pn = self.rms(r, self.w(il, "post_attention_norm.weight"))
        return r + moe_out + self.shexp(il, pn, lora)

    def forward(self, x, moe_outs, lora=None):
        """Run layers self.layers in order. x [B, T, 2048] (or [T, 2048]); moe_outs: dict il -> same shape."""
        if x.dim() == 2:
            return self.forward(x[None], {k: v[None] for k, v in moe_outs.items()}, lora)[0]
        from torch.utils.checkpoint import checkpoint
        for il in self.layers:
            if torch.is_grad_enabled():  # keep only layer inputs; recompute DeltaNet states during backward
                x = checkpoint(self.layer, il, x, moe_outs[il], lora, use_reentrant=False)
            else:
                x = self.layer(il, x, moe_outs[il], lora)
        return x
