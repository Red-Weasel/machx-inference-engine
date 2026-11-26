#!/usr/bin/env python3
"""Show Dream-baseline runs live in Dream: for each label, a metrics file and an attempts.jsonl (Dream's left-column
feed) under dream-coding-work/live/<label>/, attached to Dream's Train tab with `dream.train_runs attach`.

  live_feed.py --unit dream-baseline.service LABEL=MODEL_DIR [LABEL=MODEL_DIR ...]

One attempt row per graded Dream session: the task, attempt n of 3, passed or not and why, the steps it took, and the
session itself in readable form (each tool call and the final answer as "content", the model's thinking as "reasoning").
"""
import argparse
import json
import os
import subprocess
import time

W = os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.join("/media", os.environ.get("USER", ""), "Secondary Drive")), "dream-coding-work")
DREAM = os.environ.get("DREAM_DIR", os.path.expanduser("~/Desktop/Dream-Agent-Harness"))
CHECKED = os.path.expanduser("~/models/inkling2-train/data/checked/manifest.jsonl")

ap = argparse.ArgumentParser()
ap.add_argument("--unit", required=True)
ap.add_argument("pairs", nargs="+")
a = ap.parse_args()
pairs = [p.split("=", 1) for p in a.pairs]
n_tasks = sum(1 for l in open(CHECKED) if json.loads(l)["split"] == "heldout")


def text_of(c):
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def readable(msgs):
    out, think = [], []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        c = text_of(m.get("content"))
        if "</think>" in c:
            t, c = c.split("</think>", 1)
            think.append(t.replace("<think>", "").strip())
        if m.get("reasoning_content"):
            think.append(m["reasoning_content"].strip())
        if c.strip():
            out.append(c.strip())
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", {})
            args = f.get("arguments")
            args = args if isinstance(args, str) else json.dumps(args)
            out.append(f"→ {f.get('name')}({args[:400]}{'…' if len(args) > 400 else ''})")
    return "\n\n".join(out), "\n\n---\n\n".join(think)


def active():
    return subprocess.run(["systemctl", "--user", "is-active", "-q", a.unit]).returncode == 0


seen = {}
attached = set()
while True:
    still = active()
    for label, model in pairs:
        rec = os.path.join(W, "records", label)
        if not os.path.isdir(rec):
            continue
        live = os.path.join(W, "live", label)
        os.makedirs(live, exist_ok=True)
        metrics, feed = os.path.join(live, "metrics.jsonl"), os.path.join(live, "attempts.jsonl")
        if label not in attached:
            open(metrics, "a").close()
            subprocess.run([os.path.join(DREAM, ".venv/bin/python"), "-m", "dream.train_runs", "attach", "--unit", a.unit,
                            "--metrics", metrics, "--model", model, "--data", CHECKED, "--out", rec,
                            "--max-steps", str(n_tasks * 3)], cwd=DREAM, capture_output=True)
            with open(feed, "a") as f:
                f.write(json.dumps({"type": "phase", "step": 0, "phase": f"Dream baseline: {label}, {n_tasks} held-out tasks x 3",
                                    "t": round(time.time(), 1)}) + "\n")
            attached.add(label)
            seen[label] = set()
        rows, done = [], seen[label]
        for task in sorted(os.listdir(rec)):
            for n in sorted(os.listdir(os.path.join(rec, task))):
                d = os.path.join(rec, task, n)
                if (task, n) in done or not os.path.exists(os.path.join(d, "grade.json")):
                    continue
                g = json.load(open(os.path.join(d, "grade.json")))
                r = json.load(open(os.path.join(d, "result.json"))) if os.path.exists(os.path.join(d, "result.json")) else {}
                try:
                    msgs = json.load(open(os.path.join(d, "session.json"))).get("messages", [])
                except (OSError, ValueError):
                    msgs = []
                content, reasoning = readable(msgs)
                reason = g.get("reason") or "passed"
                rows.append({"type": "attempt", "step": 0, "phase": "heldout", "task_id": task, "family": task.rsplit("_", 1)[0],
                             "attempt": int(n) + 1, "of": 3, "reward": 1.0 if g["passed"] else 0.0,
                             "reason": f"{reason} · {r.get('requests', 0)} steps, {r.get('tool_calls', 0)} tool calls"
                                       f"{', ' + str(r.get('wrong_calls', 0) + r.get('failed_tool_calls', 0)) + ' broken' if (r.get('wrong_calls') or r.get('failed_tool_calls')) else ''}",
                             "finish_reason": g.get("ended"), "tokens": r.get("output_tokens") or 0,
                             "content": content, "reasoning": reasoning, "advantage": None})
                done.add((task, n))
        if rows:
            with open(feed, "a") as f:
                for x in rows:
                    f.write(json.dumps(x) + "\n")
            passed = sum(1 for l in open(feed) if '"type": "attempt"' in l and '"reward": 1.0' in l)
            total = sum(1 for l in open(feed) if '"type": "attempt"' in l)
            with open(metrics, "a") as f:
                f.write(json.dumps({"step": total, "eval_success_rate": round(passed / total, 4), "reward": round(passed / total, 4),
                                    "elapsed_seconds": 0}) + "\n")
    if not still:
        break
    time.sleep(10)
