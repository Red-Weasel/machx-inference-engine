#!/usr/bin/env python3
"""The same held-out tasks WITHOUT Dream: the model answers each task directly through the engine (one chat request,
no tools, no harness), and its reply is graded by the same hidden tests. The comparison point for "does Dream help".

  direct_eval.py --tasks heldout.grpo.jsonl --label LABEL [--tries 3] [--max-tokens 32768] [--parallel 12]

Needs an engine on 127.0.0.1:11435 (baseline.sh's). Records: dream-coding-work/records/LABEL/<task>/<n>/{grade,reply}.json;
live view: dream-coding-work/live/LABEL/{metrics,attempts}.jsonl for Dream's Train tab.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import grpo_rewards as GR  # noqa: E402

W = os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.join("/media", os.environ.get("USER", ""), "Secondary Drive")), "dream-coding-work")
ap = argparse.ArgumentParser()
ap.add_argument("--tasks", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--tries", type=int, default=3)
ap.add_argument("--max-tokens", type=int, default=32768)
ap.add_argument("--parallel", type=int, default=12)
a = ap.parse_args()

tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
rec, live = os.path.join(W, "records", a.label), os.path.join(W, "live", a.label)
os.makedirs(live, exist_ok=True)
lock = threading.Lock()
done = {"n": 0, "p": 0}
t0 = time.time()


def ask(msgs):
    body = json.dumps({"model": "x", "messages": msgs, "max_tokens": a.max_tokens}).encode()
    r = urllib.request.Request("http://127.0.0.1:11435/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=7200) as f:
        j = json.load(f)
    m = j["choices"][0]["message"]
    return m.get("content") or "", m.get("reasoning_content") or "", j["choices"][0].get("finish_reason"), j.get("usage", {})


def one(job):
    t, n = job
    d = os.path.join(rec, t["id"], str(n))
    if os.path.exists(os.path.join(d, "grade.json")):
        return
    s = time.time()
    try:
        content, reasoning, fin, usage = ask(t["messages"])
        r, reason = GR.score(t, {"content": content, "reasoning": reasoning, "finish_reason": fin})
    except Exception as e:                                         # noqa: BLE001
        content, reasoning, fin, usage, r, reason = "", "", "error", {}, 0.0, f"error: {e}"
    os.makedirs(d, exist_ok=True)
    json.dump({"content": content, "reasoning": reasoning, "finish_reason": fin, "usage": usage},
              open(os.path.join(d, "reply.json"), "w"))
    json.dump({"passed": r >= 1.0, "reason": reason or "", "ended": "finished" if fin != "error" else "error",
               "seconds": round(time.time() - s, 1)}, open(os.path.join(d, "grade.json"), "w"))
    with lock:
        done["n"] += 1
        done["p"] += r >= 1.0
        with open(os.path.join(live, "attempts.jsonl"), "a") as f:
            f.write(json.dumps({"type": "attempt", "step": 0, "phase": "heldout", "task_id": t["id"], "family": t["family"],
                                "attempt": n + 1, "of": a.tries, "reward": float(r >= 1.0),
                                "reason": f"{reason or 'passed'} · direct answer, no Dream, no tools",
                                "finish_reason": fin, "tokens": usage.get("completion_tokens") or 0,
                                "content": content, "reasoning": reasoning, "advantage": None}) + "\n")
        with open(os.path.join(live, "metrics.jsonl"), "a") as f:
            f.write(json.dumps({"step": done["n"], "eval_success_rate": round(done["p"] / done["n"], 4),
                                "reward": round(done["p"] / done["n"], 4), "elapsed_seconds": round(time.time() - t0, 1)}) + "\n")


jobs = [(t, n) for n in range(a.tries) for t in tasks]
with cf.ThreadPoolExecutor(a.parallel) as ex:
    list(ex.map(one, jobs))
print(f"{a.label}: {done['p']} of {done['n']} passed (direct answers, no Dream)")
