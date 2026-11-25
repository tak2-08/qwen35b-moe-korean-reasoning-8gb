"""Exam accuracy of the REAL chat engine (llama.cpp, same model file as the chat app) on held-out exam items (newest exam year, never trained),
optionally with a LoRA adapter. Two modes:
  direct  answer straight away: probability of ①..⑤ (or the first digit) as the next token after '정답:'
  cot     solve step by step (no thinking mode, the trained '1단계 ... 결론' style), then read '정답: ③'
Also reports background agreement-free sanity: share of CoT outputs that end with a parsable answer.
Usage: eval_exam.py [adapter.gguf|none] [label] [n_items]   (starts/stops its own llama-server on :8095)
Writes results to RESULTS/exam_eval.jsonl"""
import io, json, os, random, re, subprocess, sys, time, requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = os.environ.get("EXAM_DATA", "data/exam")
MODEL = os.environ.get("MODEL", "models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf")
ENGINE = os.environ.get("ENGINE", "llama-server")
ad = sys.argv[1] if len(sys.argv) > 1 else "none"; label = sys.argv[2] if len(sys.argv) > 2 else ad
NI = int(sys.argv[3]) if len(sys.argv) > 3 else 300
MODES = os.environ.get("MODES", "direct,cot").split(",")
CH = "①②③④⑤"; URL = "http://127.0.0.1:8095"

BENCH = os.environ.get("BENCH", "")   # e.g. kbench.jsonl: Korean benchmark with per-category scores
if BENCH:
    items = [json.loads(x) for x in open(os.path.join(D, BENCH), encoding="utf-8")][:NI]
    MODES = ["direct"]
else:
    items = [json.loads(x) for f in ("cot_heldout.jsonl", "cot_heldout_ocr.jsonl") if os.path.exists(os.path.join(D, f))
             for x in open(os.path.join(D, f), encoding="utf-8")]
    items = [r for r in items if not r.get("figure") and isinstance(r["answer"], int)]
    random.Random(0).shuffle(items); items = items[:NI]


def chat_prompt(user):
    # exactly the format used in training (thinking off): no dependence on server template options
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

args = [ENGINE, "-m", MODEL, "-ngl", "99", *os.environ.get("PLACE", "--n-cpu-moe 26").split(), "-c", "8192", "-fa", "on", "--jinja", "-t", "6",
        "-np", "1", "--host", "127.0.0.1", "--port", "8095", "--cache-ram", "0",
        "--reasoning", "auto" if "think" in MODES else "off", "--reasoning-budget", os.environ.get("TBUDGET", "3000")]
if ad != "none":
    args += ["--lora", ad]
srv = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=os.path.dirname(ENGINE))
try:
    for _ in range(300):
        try:
            if requests.get(URL + "/health", timeout=2).json().get("status") == "ok": break
        except Exception: pass
        time.sleep(2)
    tok = lambda s: requests.post(URL + "/tokenize", json={"content": s, "add_special": False}).json()["tokens"]
    # ①..⑤ are two tokens each and share the first one: put it in the prompt, compare the second
    PRE = tok(CH[0])[0]; assert all(tok(c)[0] == PRE for c in CH)
    cand = {i + 1: tok(c)[1] for i, c in enumerate(CH)}
    res = {"label": label, "adapter": ad, "n": len(items)}
    t0 = time.time()
    if "direct" in MODES:
        ok = 0; per = {}
        for r in items:
            p = tok(chat_prompt(r["prompt"] + "\n\n정답 번호만 답하세요.") + "정답: ") + [PRE]
            out = requests.post(URL + "/completion", json={"prompt": p, "n_predict": 1, "n_probs": 20, "temperature": 0}).json()
            if "completion_probabilities" not in out:
                print("server reply without probabilities:", str(out)[:300], "prompt tokens", len(p), flush=True)
                res.setdefault("skipped", 0); res["skipped"] += 1; continue
            probs = {x["id"]: x["logprob"] for x in out["completion_probabilities"][0]["top_logprobs"]}
            n = r.get("n", 5)
            best = max((k for k in cand if k <= n), key=lambda k: probs.get(cand[k], -1e9))
            hit = int(best == r["answer"]); ok += hit
            key = (r.get("src", "exam") + ":" + str(r.get("cat", r.get("subject", ""))))
            a, b = per.get(key, (0, 0)); per[key] = (a + hit, b + 1)
        res["direct_acc"] = round(ok / len(items), 4)
        src = {}
        for k, (a, b) in per.items():
            s = k.split(":")[0]; x, y = src.get(s, (0, 0)); src[s] = (x + a, y + b)
        res["by_src"] = {s: round(a / b, 3) for s, (a, b) in src.items()}
        res["by_cat"] = {k: f"{a}/{b}" for k, (a, b) in sorted(per.items())}
    if "cot" in MODES:
        ok = parsed = 0; NC = min(len(items), int(os.environ.get("NCOT", "120"))); hits = []
        COTMAX = int(os.environ.get("COTMAX", "900")); SAMP = json.loads(os.environ.get("SAMP", "{}"))   # c2: long answers, sampling tests
        loops = cut = 0; clen = []; outs = []
        for r in items[:NC]:
            msgs = [{"role": "user", "content": r["prompt"] + "\n\n단계별로 풀고 마지막 줄에 '정답: ③' 형식으로 답하세요."}]
            body = {"messages": msgs, "max_tokens": COTMAX, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            body.update(SAMP)
            out = requests.post(URL + "/v1/chat/completions", json=body, timeout=1800).json()
            txt = out["choices"][0]["message"]["content"]
            m = re.findall(r"정답\s*[:：]?\s*([①②③④⑤])", txt)
            if m:
                parsed += 1; ok += int(CH.index(m[-1]) + 1 == r["answer"])
            if len(hits) < 60: hits.append(int(bool(m) and CH.index(m[-1]) + 1 == r["answer"]))
            lines = [x.strip() for x in txt.split("\n") if len(x.strip()) >= 12]
            rep = max([lines.count(x) for x in set(lines)] or [0])          # same line 3+ times = a loop
            fin = out["choices"][0].get("finish_reason"); clen.append(out.get("usage", {}).get("completion_tokens", 0))
            loops += int(rep >= 3); cut += int(fin == "length")
            outs.append(dict(answer=r["answer"], got=(CH.index(m[-1]) + 1 if m else None), finish=fin, max_line_repeat=rep, text=txt))
        res.update(cot_acc=round(ok / NC, 4), cot_parsed=round(parsed / NC, 4), n_cot=NC, cot_acc_first60=round(sum(hits) / max(1, len(hits)), 4),
                   cot_loop_rate=round(loops / NC, 4), cot_cut_rate=round(cut / NC, 4), cot_avg_tokens=round(sum(clen) / max(1, len(clen))),
                   cot_max=COTMAX, samp=SAMP)
        with open(os.path.join(os.environ.get("EXAM_DATA", "data/exam"), f"cot_out_{label}.jsonl"), "w", encoding="utf-8") as f:
            for o in outs: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    if "think" in MODES:   # thinking mode ON (as the chat app can run it): same first items as the CoT run
        ok = parsed = sparsed = 0; NT = min(len(items), int(os.environ.get("NTHINK", "60"))); tlen = []
        for r in items[:NT]:
            msgs = [{"role": "user", "content": r["prompt"] + "\n\n마지막 줄에 '정답: ③' 형식으로 답하세요."}]
            out = requests.post(URL + "/v1/chat/completions", json={"messages": msgs, "max_tokens": int(os.environ.get("TBUDGET", "3000")) + 600,
                                "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}}, timeout=1800).json()
            msg = out["choices"][0]["message"]; txt = msg.get("content") or ""; rsn = msg.get("reasoning_content") or ""
            tlen.append(out.get("usage", {}).get("completion_tokens", 0))
            # lenient, same for every model: the visible reply first ('정답: ③', '정답은 ③', '답: ③'),
            # else the last such statement at the end of the thinking text (answer written before </think>)
            pat = r"(?:정답|답)\s*(?:은|는)?\s*[:：]?\s*(?:\*\*)?\s*([①②③④⑤])"
            m = re.findall(pat, txt) or re.findall(pat, rsn[-400:])
            strict = bool(re.findall(r"정답\s*[:：]?\s*([①②③④⑤])", txt))
            if m:
                parsed += 1; ok += int(CH.index(m[-1]) + 1 == r["answer"])
            sparsed += int(strict)
            with open(os.path.join(D, f"think_out_{label}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(answer=r["answer"], got=(CH.index(m[-1]) + 1 if m else None), content=txt[-600:],
                                        reasoning_tail=rsn[-600:], reasoning_len=len(rsn), tokens=tlen[-1],
                                        finish=out["choices"][0].get("finish_reason")), ensure_ascii=False) + "\n")
        res.update(think_acc=round(ok / NT, 4), think_parsed=round(parsed / NT, 4), think_strict_parsed=round(sparsed / NT, 4),
                   n_think=NT, think_avg_tokens=round(sum(tlen) / max(1, NT)))
    res["sec"] = round(time.time() - t0)
    print(json.dumps(res, ensure_ascii=False))
    with open(os.path.join(os.environ.get("RESULTS", "results"), "exam_eval.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
finally:
    srv.terminate()
    try: srv.wait(30)
    except Exception: srv.kill()
