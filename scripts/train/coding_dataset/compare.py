#!/usr/bin/env python3
"""Scorecard of Dream runs: how each model did inside Dream on the same tasks (records written by run.py).

  compare.py LABEL [LABEL ...]        e.g. compare.py base_plain_high base_cyber_high

Per label: tasks passed (attempts and per task), steps (model requests), tool calls, broken / wrong tool calls, failed
commands, loops (3+ identical tool calls in a row), answers left in chat instead of written to the answer file, sessions
that did not finish, output tokens and seconds. Then, on the attempts BOTH labels passed, the steps and tokens each
needed (the like-for-like efficiency comparison). Reads only the records; prints counts, never the sessions' text.
"""
import collections
import json
import os
import statistics
import sys

REC = os.path.join(os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.join("/media", os.environ.get("USER", ""), "Secondary Drive")), "dream-coding-work"), "records")


def load(label):
    out = []
    base = os.path.join(REC, label)
    for task in sorted(os.listdir(base)):
        for n in sorted(os.listdir(os.path.join(base, task))):
            d = os.path.join(base, task, n)
            if not os.path.exists(os.path.join(d, "grade.json")):
                continue
            g = json.load(open(os.path.join(d, "grade.json")))
            r = json.load(open(os.path.join(d, "result.json"))) if os.path.exists(os.path.join(d, "result.json")) else {}
            msgs = []
            try:
                msgs = json.load(open(os.path.join(d, "session.json"))).get("messages", [])
            except (OSError, ValueError):
                pass
            calls = [(c.get("function", {}).get("name"), json.dumps(c.get("function", {}).get("arguments"), sort_keys=True))
                     for m in msgs if m.get("role") == "assistant" for c in (m.get("tool_calls") or [])]
            run, loop = 1, False
            for i in range(1, len(calls)):
                run = run + 1 if calls[i] == calls[i - 1] else 1
                loop = loop or run >= 3
            out.append({"task": task, "family": task.rsplit("_", 1)[0], "n": int(n), "passed": bool(g["passed"]),
                        "reason": g.get("reason") or "passed", "ended": g.get("ended") or r.get("ended"),
                        "steps": r.get("requests") or 0, "calls": r.get("tool_calls") or 0,
                        "broken": (r.get("failed_tool_calls") or 0) + (r.get("wrong_calls") or 0),
                        "nonzero": r.get("nonzero_exits") or 0, "loop": loop,
                        "chat_answer": g.get("reason") == "no_answer_file" and (r.get("tool_calls") or 0) == 0,
                        "tokens": r.get("output_tokens") or 0, "seconds": r.get("seconds") or 0})
    return out


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def card(label, rs):
    n = len(rs)
    tasks = collections.defaultdict(list)
    for r in rs:
        tasks[r["task"]].append(r["passed"])
    fam = collections.defaultdict(lambda: [0, 0])
    for r in rs:
        fam[r["family"]][0] += r["passed"]
        fam[r["family"]][1] += 1
    print(f"\n== {label}: {n} attempts on {len(tasks)} tasks")
    print(f"  passed               {sum(r['passed'] for r in rs)} / {n} ({100 * mean([r['passed'] for r in rs]):.1f} %);"
          f" tasks passed at least once {sum(any(v) for v in tasks.values())} / {len(tasks)}")
    print("  by family            " + ", ".join(f"{f} {p}/{t}" for f, (p, t) in sorted(fam.items())))
    print(f"  steps (requests)     mean {mean([r['steps'] for r in rs]):.1f}, median {statistics.median([r['steps'] for r in rs]) if rs else 0}")
    print(f"  tool calls           mean {mean([r['calls'] for r in rs]):.1f}; broken or wrong {sum(r['broken'] for r in rs)} in total "
          f"({mean([r['broken'] for r in rs]):.2f} an attempt); failed commands {sum(r['nonzero'] for r in rs)}")
    print(f"  loops (3+ same call) {sum(r['loop'] for r in rs)} attempts")
    print(f"  answer left in chat  {sum(r['chat_answer'] for r in rs)} attempts (no tool used, no answer file)")
    print(f"  did not finish       {sum(1 for r in rs if r['ended'] != 'finished')} attempts "
          f"({dict(collections.Counter(r['ended'] for r in rs if r['ended'] != 'finished'))})")
    print(f"  output tokens        mean {mean([r['tokens'] for r in rs]):.0f}; seconds mean {mean([r['seconds'] for r in rs]):.0f}")
    print(f"  failure reasons      {dict(collections.Counter(r['reason'] for r in rs if not r['passed']))}")


def main():
    labels = sys.argv[1:]
    data = {l: load(l) for l in labels}
    for l in labels:
        card(l, data[l])
    if len(labels) == 2:
        a, b = labels
        pa = {(r["task"], r["n"]): r for r in data[a] if r["passed"]}
        pb = {(r["task"], r["n"]): r for r in data[b] if r["passed"]}
        both_tasks = {t for t, _ in pa} & {t for t, _ in pb}
        if both_tasks:
            sa = [r for r in pa.values() if r["task"] in both_tasks]
            sb = [r for r in pb.values() if r["task"] in both_tasks]
            print(f"\n== on the {len(both_tasks)} tasks both solved (their passing attempts)")
            for l, s in ((a, sa), (b, sb)):
                print(f"  {l:24s} steps {mean([r['steps'] for r in s]):.1f}, tool calls {mean([r['calls'] for r in s]):.1f}, "
                      f"tokens {mean([r['tokens'] for r in s]):.0f}, seconds {mean([r['seconds'] for r in s]):.0f}")


if __name__ == "__main__":
    main()
