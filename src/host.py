"""Client for bin/hm_host.exe (frozen model: per-token post-norm hidden states + MoE routing)."""
import os
import struct
import subprocess

import numpy as np

LAB = os.environ.get("LAB", ".")
MODEL = os.environ.get("MODEL", "models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf")
# same placement as the chat app (--n-cpu-moe 26): experts of layers 14..39 in system RAM
CPU_PAT = r"blk\.(1[4-9]|[23][0-9])\.ffn_(up|down|gate|gate_up)_exps"


class Host:
    def __init__(self, model=MODEL, ctx=2048, pattern=CPU_PAT, capture=""):
        env = dict(os.environ, HM_CPU_PATTERN=pattern, HM_CAPTURE=capture)
        self.p = subprocess.Popen([os.path.join(LAB, "bin", "hm_host.exe"), model, "1", str(ctx)],
                                  cwd=os.path.join(LAB, "bin"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(os.path.join(LAB, "results", "hm_host.log"), "w"), env=env)

    def _r(self, n):
        b = b""
        while len(b) < n:
            c = self.p.stdout.read(n - len(b))
            if not c:
                raise RuntimeError("hm_host exited (see results/hm_host.log)")
            b += c
        return b

    def tokenize(self, text):
        b = text.encode("utf-8")
        self.p.stdin.write(struct.pack("<ii", 1, len(b)) + b); self.p.stdin.flush()
        n = struct.unpack("<i", self._r(4))[0]
        return list(struct.unpack(f"<{n}i", self._r(4 * n))) if n else []

    def extract(self, toks):
        self.p.stdin.write(struct.pack("<ii", 2, len(toks)) + struct.pack(f"<{len(toks)}i", *toks)); self.p.stdin.flush()
        n, d = struct.unpack("<ii", self._r(8))
        h = np.frombuffer(self._r(4 * n * d), dtype=np.float32).reshape(n, d) if n else np.zeros((0, d), np.float32)
        L, k = struct.unpack("<ii", self._r(8))
        ids = np.frombuffer(self._r(4 * L * n * k), dtype=np.int32).reshape(L, n, k) if n else np.zeros((L, 0, k), np.int32)
        nc = struct.unpack("<i", self._r(4))[0]
        caps = []
        for _ in range(nc):
            w = struct.unpack("<i", self._r(4))[0]
            caps.append(np.frombuffer(self._r(4 * n * w), dtype=np.float32).reshape(n, w) if n else np.zeros((0, w), np.float32))
        self.caps = caps
        return h, ids

    def set_lora(self, path="", scale=1.0):
        b = path.encode("utf-8")
        self.p.stdin.write(struct.pack("<ii", 3, len(b)) + b + struct.pack("<f", scale)); self.p.stdin.flush()
        ok = struct.unpack("<i", self._r(4))[0]
        if not ok:
            raise RuntimeError("LoRA load failed: " + path)

    def close(self):
        try:
            self.p.stdin.write(struct.pack("<i", 0)); self.p.stdin.flush()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()
