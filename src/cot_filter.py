"""Quality filter for teacher step-by-step solutions (applied before training).
A solution is dropped when it loops or circles:
  - a long sentence (>= 40 chars) appears 2+ times verbatim, or any sentence (>= 15 chars) 6+ times
  - a 30-character window (whitespace removed) repeats 3+ times (catches looping formula lines)
  - self-check phrases ("다시 확인", "다시 계산", "검산", "정리하면" ...) appear 3+ times
  - more than one distinct final '정답:' value (the solution changed its answer)
Separately (in build_c2_cache.py) a solution is only used if its final '정답' equals the official answer key.
Usage as a module: from cot_filter import why;  why(text) -> "" (keep) or the reason"""
import re
from collections import Counter

SELF = re.compile(r"다시\s*확인|다시\s*계산|다시\s*검토|검산|정리하면|잘못\s*계산|아닌데")


def why(s):
    sents = [x.strip() for x in re.split(r"(?<=[.!?다])\s+|\n", s) if len(x.strip()) >= 15]
    c = Counter(sents)
    # long sentences must not repeat; short per-choice verdicts ("…옳지 않다.") may repeat up to 5 times
    if any(n >= 2 and len(t) >= 40 or n >= 6 for t, n in c.items()): return "repeat_sentence"
    flat = re.sub(r"\s+", "", s)
    w = Counter(flat[i:i + 30] for i in range(0, max(0, len(flat) - 30), 5))
    if w and w.most_common(1)[0][1] >= 3: return "repeat_span"
    if len(SELF.findall(s)) >= 3: return "self_check_loop"
    if len(set(re.findall(r"정답\s*[:：]\s*\**\s*([①②③④⑤]|\d+)", s))) > 1: return "answer_changed"
    return ""
