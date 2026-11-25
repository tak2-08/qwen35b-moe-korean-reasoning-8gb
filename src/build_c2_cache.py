"""Token caches for c2 training: long step-by-step solutions (one verified solution per exam item) instead of the short
reference explanations the first stage (c1) learned from. The solution dataset itself is private and not distributed.
Layout per cache: kind 0 = training sequence, kind 1 = held-out monitor item, kind 2 = background replay text.
  - one solution per item; cot_filter.why() drops looping / repetitive solutions
  - the whole solution is trained, prompt + answer up to MAXLEN tokens
  - ~12% of each chunk is first-stage sequences (reference explanations, news text) so earlier knowledge is kept
  - items from the held-out exam year (>= HELDOUT_YEAR) are never used for training
Chunks are cut by token budget, because the activation capture costs ~175 KB of disk per token.

Inputs (JSONL, all under EXAM_DATA):
  items_train.jsonl     {"id", "prompt", "answer", "year"}            exam items (prompt includes the choices)
  solutions.jsonl       {"id", "solution"}                            verified step-by-step solution per item
  stage1_seq.jsonl      {"q", "a"}                                    first-stage sequences mixed in
  items_heldout.jsonl   {"prompt", "answer", "figure"}                held-out items (monitor only)
and HM_DATA/cache.npz (background replay texts, kind 2).
Usage: CHUNK=0 BUDGET=460000 MAXLEN=2000 python build_c2_cache.py  -> EXAM_DATA/cache_c2_{CHUNK}.npz"""
import io, json, os, random, sys
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host import Host
from cot_filter import why
D = os.environ.get("EXAM_DATA", "data/exam"); HMD = os.environ.get("HM_DATA", "data/hm")
CHUNK = int(os.environ.get("CHUNK", "0")); BUDGET = int(os.environ.get("BUDGET", "460000"))
MAXLEN = int(os.environ.get("MAXLEN", "2000")); NTEST = int(os.environ.get("NTEST", "150"))
HELDOUT_YEAR = int(os.environ.get("HELDOUT_YEAR", "2026"))
CH = "①②③④⑤"
ASK = "\n\n단계별로 풀고 마지막 줄에 '정답: ③' 형식으로 답하세요."


def chat(user, assistant):   # Qwen chat format with thinking off, as used at inference
    return (f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", f"{assistant}<|im_end|>\n")


def load(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")] if os.path.exists(p) else []


items = {r["id"]: r for r in load(os.path.join(D, "items_train.jsonl"))}
sol, drop = {}, 0
for r in load(os.path.join(D, "solutions.jsonl")):
    if r["id"] in sol or r["id"] not in items: continue
    if why(r["solution"]): drop += 1; continue
    if items[r["id"]].get("year", 0) >= HELDOUT_YEAR: continue
    sol[r["id"]] = r
cot = [chat(items[i]["prompt"] + ASK, sol[i]["solution"]) for i in sorted(sol)]
extra = [chat(r["q"], r["a"]) for r in load(os.path.join(D, "stage1_seq.jsonl"))]
random.Random(1234).shuffle(cot); random.Random(99).shuffle(extra)
print(f"solutions {len(sol)} (loop filter dropped {drop}), stage-1 sequences {len(extra)}")

host = Host(ctx=MAXLEN + 48)
enc = lambda s: host.tokenize(s)
tc = [(enc(u), enc(a)) for u, a in cot]
keep = [t for t in tc if len(t[0]) + len(t[1]) <= MAXLEN]
chunks, cur, used = [], [], 0
for t in keep:
    cur.append(t); used += len(t[0]) + len(t[1])
    if used >= BUDGET * 0.88: chunks.append(cur); cur, used = [], 0
if cur: chunks.append(cur)
print("chunks", len(chunks), [sum(len(a) + len(b) for a, b in c) for c in chunks])
train = list(chunks[CHUNK]) if CHUNK < len(chunks) else []
ex_used, j = 0, CHUNK * 400
while ex_used < BUDGET * 0.12 and j < len(extra):
    tu, ta = enc(extra[j][0]), enc(extra[j][1]); j += 1
    if len(tu) + len(ta) > MAXLEN: continue
    train.append((tu, ta)); ex_used += len(tu) + len(ta)
random.Random(CHUNK).shuffle(train)
print(f"chunk {CHUNK}: {len(train)} sequences, {sum(len(x)+len(y) for x, y in train)} tokens")

test = [r for r in load(os.path.join(D, "items_heldout.jsonl")) if not r.get("figure") and isinstance(r["answer"], int)]
random.Random(0).shuffle(test)
old = np.load(os.path.join(HMD, "cache.npz"))
tok, off, kind, fact, ans = [], [0], [], [], []
for tu, ta in train:
    tok += tu + ta; off.append(len(tok)); kind.append(0); fact.append(-1); ans.append(len(tu))
nt = 0
for r in test:
    if nt >= NTEST: break
    u, _ = chat(r["prompt"] + "\n\n정답 번호만 답하세요.", "")
    ta = enc(CH[r["answer"] - 1]); tu = enc(u + "정답: ") + ta[:1]; ta = ta[1:]
    if len(tu) > 1500: continue
    tok += tu + ta[:1]; off.append(len(tok)); kind.append(1); fact.append(nt); ans.append(len(tu)); nt += 1
for s in np.where(old["seq_kind"] == 2)[0]:
    t = old["tok"][old["seq_off"][s]:old["seq_off"][s + 1]].tolist()
    tok += t; off.append(len(tok)); kind.append(2); fact.append(-1); ans.append(-1)
host.close()
out = os.path.join(D, f"cache_c2_{CHUNK}.npz")
np.savez(out, tok=np.array(tok, np.int32), h=np.zeros((1, 1), np.float16), seq_off=np.array(off, np.int64),
         seq_kind=np.array(kind, np.int8), seq_fact=np.array(fact, np.int32), seq_ans=np.array(ans, np.int32))
print("saved", out, "sequences", len(kind), "tokens", len(tok))
