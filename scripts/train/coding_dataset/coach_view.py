#!/usr/bin/env python3
"""The coach's view of Dream sessions: what the model did, step by step, and why it failed.

  coach_view.py LABEL                    one line per task: passes, failure reasons
  coach_view.py LABEL TASK [N]           the session(s) of one task, readable: the task, each step (how long it
                                         thought, what it said, the tool call, what came back), the grade, and for a
                                         failed test the hidden tests' own output re-run on the saved answer
"""
import collections
import glob
import json
import os
import subprocess
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

REC = C.RECORDS


def text(c):
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def cut(s, n):
    s = s.strip()
    return s if len(s) <= n else s[: n // 2] + f"\n   [... {len(s) - n} chars ...]\n" + s[-n // 2:]


def test_output(t, answer):
    d = os.path.join(C.WORK, "grade", uuid.uuid4().hex)
    os.makedirs(d)
    try:
        for name, content in t["files"].items():
            p = os.path.join(d, name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(content)
        open(os.path.join(d, t["answer_file"]), "w").write(answer)
        for cmd in t["check"]:
            full = C.scope("4G", C._base(["--bind", d, "/t", "--chdir", "/t", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
                                          "bash", "-c", f'ulimit -f {C.MAX_FILE_KB}; exec "$@"', "x"] + cmd))
            try:
                r = subprocess.run(full, capture_output=True, text=True, errors="replace", timeout=t.get("timeout", 60))
            except subprocess.TimeoutExpired:
                return "the tests timed out"
            if r.returncode:
                return (r.stdout + r.stderr)[-2500:]
        return "(the tests pass on this answer when re-run)"
    finally:
        subprocess.run(["rm", "-rf", d])


def last_written(msgs, name):
    """The answer file's content as the session last wrote it with write_file, or None."""
    out = None
    for m in msgs:
        for tc in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
            f = tc.get("function", {})
            a = f.get("arguments")
            try:
                a = json.loads(a) if isinstance(a, str) else (a or {})
            except ValueError:
                continue
            if f.get("name") == "write_file" and os.path.basename(str(a.get("path", ""))) == name and "content" in a:
                out = a["content"]
    return out


def summary(label):
    per = collections.defaultdict(list)
    for g in sorted(glob.glob(os.path.join(REC, label, "*", "*", "grade.json"))):
        d = os.path.dirname(g)
        per[os.path.basename(os.path.dirname(d))].append(json.load(open(g)))
    bad = 0
    for task, gs in sorted(per.items()):
        p = sum(g["passed"] for g in gs)
        if p < len(gs):
            bad += 1
            print(f"{task:28s} {p}/{len(gs)}  " + ", ".join(g["reason"] or g.get("ended") or "" for g in gs if not g["passed"]))
    print(f"\n{bad} of {len(per)} tasks with a failure")


def show(label, task, n=None):
    tasks = C.load_tasks()
    t = tasks[task]
    print(f"######## {task} ({t['language']}, answer file {t.get('answer_file')})\n--- TASK ---\n{cut(t['statement'], 3000)}\n")
    for d in sorted(glob.glob(os.path.join(REC, label, task, "*"))):
        if n is not None and os.path.basename(d) != str(n):
            continue
        g = json.load(open(os.path.join(d, "grade.json")))
        r = json.load(open(os.path.join(d, "result.json"))) if os.path.exists(os.path.join(d, "result.json")) else {}
        msgs = json.load(open(os.path.join(d, "session.json")))["messages"] if os.path.exists(os.path.join(d, "session.json")) else []
        print(f"=========== attempt {os.path.basename(d)}: {'PASSED' if g['passed'] else 'FAILED: ' + (g['reason'] or '')}"
              f" | ended {g.get('ended')} | {r.get('requests')} requests, {r.get('tool_calls')} tool calls, "
              f"{r.get('output_tokens')} tokens, {r.get('seconds')} s")
        step = 0
        for m in msgs:
            if m["role"] == "assistant":
                step += 1
                th = m.get("reasoning_content") or ""
                print(f"\n[{step}] THOUGHT {len(th)} chars: {cut(th, 700)}")
                if text(m.get("content")).strip():
                    print(f"[{step}] SAID: {cut(text(m.get('content')), 900)}")
                for tc in m.get("tool_calls") or []:
                    f = tc.get("function", {})
                    a = f.get("arguments")
                    a = a if isinstance(a, str) else json.dumps(a)
                    print(f"[{step}] CALL {f.get('name')}: {cut(a, 900)}")
            elif m["role"] == "tool":
                print(f"    -> RESULT: {cut(text(m.get('content')), 700)}")
        ans = glob.glob(os.path.join(d, "answer.*"))
        answer = open(ans[0]).read() if ans else last_written(msgs, t.get("answer_file") or "")
        if not g["passed"] and g["reason"] == "test_failed" and answer is not None:
            src = "the saved answer" if ans else "the last version it wrote with write_file (edits by shell are not in it)"
            print(f"\n--- HIDDEN TEST OUTPUT, on {src} ---\n{test_output(t, answer)}")
        print()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        summary(sys.argv[1])
    else:
        show(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
