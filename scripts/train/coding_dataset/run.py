#!/usr/bin/env python3
"""Run checked tasks through Dream's real agent loop and grade them with their hidden tests. Resumable from its records.

  run.py --cond page|nopage --split heldout|train [--ids a,b] [--attempts 3] [--stop-after 2] [--workers 12]
         [--minutes 15] [--hours H] [--page FILE]

One record per attempt: records/<label>/<task>/<attempt>/{result,session,grade}.json. A foreign engine or training run
stops everything at once (exit 3) and leaves the file records/STOPPED.
"""
import argparse, collections, json, os, shutil, signal, subprocess, sys, threading, time, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--cond", required=True, choices=["page", "nopage"])
ap.add_argument("--split", required=True, choices=["heldout", "train"])
ap.add_argument("--label", default="", help="records folder name; default <cond>_<split>")
ap.add_argument("--ids", default="")
ap.add_argument("--attempts", type=int, default=1)
ap.add_argument("--stop-after", type=int, default=2)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--minutes", type=float, default=15)
ap.add_argument("--hours", type=float, default=0)
ap.add_argument("--page", default=C.PAGE)
ap.add_argument("--slot-base", type=int, default=0)
ap.add_argument("--seed-files", type=int, default=1, help="1: a task that hands over a file to fix gets it in the workspace")
ap.add_argument("--notes", default="", help="coach's notes, JSONL {task_id, note}: appended to that task's prompt after "
                "C.COACH_MARK (training removes everything from the mark on, so the model learns the habit, not the note)")
ap.add_argument("--metrics", default="", help="Dream-format metrics file: step = tasks finished, eval_success_rate = share of attempts passing")
a = ap.parse_args()
label = a.label or f"{a.cond}_{a.split}"
base = os.path.join(C.RECORDS, label)
os.makedirs(base, exist_ok=True)
tasks = {k: t for k, t in C.load_tasks().items() if t["split"] == a.split}
if a.ids:
    tasks = {k: tasks[k] for k in a.ids.split(",")}
page = open(a.page).read().strip() if a.cond == "page" else None
lock, stop, live = threading.Lock(), threading.Event(), {}
state = {k: {"started": 0, "passes": 0, "busy": False} for k in tasks}
for k in tasks:
    d = os.path.join(base, k)
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        g = os.path.join(d, n, "grade.json")
        if os.path.exists(g):
            state[k]["started"] = max(state[k]["started"], int(n) + 1)
            state[k]["passes"] += bool(json.load(open(g))["passed"])
        else:
            shutil.rmtree(os.path.join(d, n), ignore_errors=True)
began = time.time()
tally = {"attempts": 0, "passed": 0, "out": 0, "done": set()}
for k in tasks:
    d = os.path.join(base, k)
    for n in os.listdir(d) if os.path.isdir(d) else []:
        g = os.path.join(d, n, "grade.json")
        if os.path.exists(g):
            tally["attempts"] += 1
            tally["passed"] += bool(json.load(open(g))["passed"])
            tally["done"].add(k)
start_done = len(tally["done"])


notes = {}
for line in open(a.notes) if a.notes else []:
    if line.strip():
        x = json.loads(line)
        notes[x["task_id"]] = x["note"].strip()


def write_metrics():
    if not a.metrics:
        return
    el = time.time() - began
    step = len(tally["done"])
    new = step - start_done
    eta = (len(tasks) - step) * el / new if new else None
    if a.hours:
        left = max(0, a.hours * 3600 - el)
        eta = left if eta is None else min(eta, left)
    row = {"step": step, "elapsed_seconds": round(el, 1), "eval_success_rate": round(tally["passed"] / tally["attempts"], 4) if tally["attempts"] else 0.0,
           "tokens_per_second": round(tally["out"] / el, 1) if el > 1 else 0.0}
    if eta is not None:
        row["eta_seconds"] = round(eta)
    with open(a.metrics, "a") as f:
        f.write(json.dumps(row) + "\n")


def next_job():
    with lock:
        if stop.is_set() or (a.hours and time.time() - began > a.hours * 3600):
            return None
        fam = collections.Counter()
        for k, s in state.items():
            fam[tasks[k]["family"]] += s["started"]
        for r in range(a.attempts):
            c = [k for k, s in state.items() if s["started"] == r and not s["busy"] and s["passes"] < a.stop_after]
            if c:
                k = min(c, key=lambda k: (fam[tasks[k]["family"]], k))
                state[k]["busy"] = True
                state[k]["started"] += 1
                return k, r
    return None


def final_text(msgs):
    if not msgs or msgs[-1].get("role") != "assistant" or msgs[-1].get("tool_calls"):
        return None
    c = msgs[-1].get("content")
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    c = c or ""
    if "</think>" in c:
        c = c.rsplit("</think>", 1)[1]
    return c.strip() or None


def one(k, n, slot):
    t, dest = tasks[k], os.path.join(base, k, str(n))
    cmd, ws, rec, home, tmp = C.dream_cmd(slot, a.minutes)
    for d in (ws, rec, home, tmp):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, mode=0o700)
    os.makedirs(os.path.join(rec, "root", "memory"))
    if a.seed_files:
        C.seed_workspace(t, ws)
    note = notes.get(k)
    open(os.path.join(rec, "prompt.md"), "w").write(t["statement"] + (f"\n\n{C.COACH_MARK}\n{note}\n" if note else ""))
    if page:
        open(os.path.join(rec, "root", "memory", "INSTRUCTIONS.md"), "w").write(page + "\n")
    out = open(os.path.join(rec, "stdout.txt"), "w")
    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    deadline, killed = time.time() + a.minutes * 60 + 120, False
    while p.poll() is None:
        time.sleep(1)
        if stop.is_set() or time.time() > deadline:
            os.makedirs(os.path.join(rec, "control"), exist_ok=True)
            open(os.path.join(rec, "control", "stop"), "w").close()
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            killed = True
    if stop.is_set():                                        # interrupted by the guard: not a result
        for d in (ws, rec, home, tmp):
            shutil.rmtree(d, ignore_errors=True)
        with lock:
            state[k]["started"] -= 1; state[k]["busy"] = False
        return
    res, msgs = {}, []
    try:
        res = json.load(open(os.path.join(rec, "result.json")))
        msgs = json.load(open(os.path.join(rec, "session.json"))).get("messages", [])
    except (OSError, ValueError):
        pass
    txt = C.text_answer(msgs) if t["language"] == "text" else final_text(msgs)
    passed, reason = False, "did_not_finish"
    if res.get("ended") == "finished" and not killed:
        passed, reason = C.grade(t, txt, ws) if (txt or t["language"] != "text") else (False, "no_final_answer")
    if t["language"] != "text" and os.path.isfile(os.path.join(ws, t["answer_file"])):     # the coach re-runs the tests on it
        shutil.copy(os.path.join(ws, t["answer_file"]), os.path.join(rec, "answer" + os.path.splitext(t["answer_file"])[1]))
    json.dump({"passed": passed, "reason": reason, "ended": res.get("ended"), "has_final": bool(txt), "slot": slot,
               "coached": bool(note)},
              open(os.path.join(rec, "grade.json"), "w"))
    os.makedirs(dest, exist_ok=True)
    for x in [f for f in os.listdir(rec) if f.startswith("answer.")] + ["result.json", "session.json", "stdout.txt", "grade.json"]:  # grade.json last: it marks the record done
        if os.path.exists(os.path.join(rec, x)):
            shutil.copy(os.path.join(rec, x), os.path.join(dest, x + ".part"))
            os.replace(os.path.join(dest, x + ".part"), os.path.join(dest, x))
    for d in (ws, rec, home, tmp):
        shutil.rmtree(d, ignore_errors=True)
    with lock:
        state[k]["busy"] = False
        state[k]["passes"] += passed
        tally["attempts"] += 1; tally["passed"] += bool(passed); tally["out"] += res.get("output_tokens") or 0
        tally["done"].add(k)
        write_metrics()
    print(json.dumps({"t": int(time.time() - began), "id": k, "n": n, "passed": passed, "reason": reason, "ended": res.get("ended"),
                      "calls": res.get("tool_calls"), "out": res.get("output_tokens"), "s": res.get("seconds")}), flush=True)


def worker(slot):
    while True:
        j = next_job()
        if j is None:
            with lock:
                more = not stop.is_set() and not (a.hours and time.time() - began > a.hours * 3600) and any(
                    s["busy"] for s in state.values()) and any(
                    s["started"] < a.attempts and s["passes"] < a.stop_after for s in state.values())
            if not more:
                return
            time.sleep(2)
            continue
        try:
            one(j[0], j[1], slot)
        except Exception as e:
            print(json.dumps({"id": j[0], "n": j[1], "error": repr(e)}), flush=True)
            with lock:
                state[j[0]]["busy"] = False


def mine():
    try:
        return int(subprocess.check_output(["systemctl", "--user", "show", "-p", "MainPID", "--value", "coding-ie"]).strip())
    except Exception:
        return -1


def guard():
    me = mine()
    while not stop.is_set():
        foreign = []
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
                comm = open(f"/proc/{pid}/comm").read().strip()
            except OSError:
                continue
            if comm == "ie" and int(pid) != me:
                foreign.append(f"ie pid {pid}")
            if comm.startswith("python") and any(c.endswith(b"machx_train.py") or c.endswith(b"machx-train") for c in cmd):
                foreign.append(f"machx_train pid {pid}")
        if foreign:
            open(os.path.join(C.RECORDS, "STOPPED"), "w").write("; ".join(foreign) + "\n")
            print(json.dumps({"stopped_for": foreign}), flush=True)
            stop.set()
            return
        time.sleep(15)


write_metrics()
threading.Thread(target=guard, daemon=True).start()
os.makedirs(C.LEASES, mode=0o700, exist_ok=True)
ws = [threading.Thread(target=worker, args=(a.slot_base + i,)) for i in range(a.workers)]
[w.start() for w in ws]
[w.join() for w in ws]
sys.exit(3 if stop.is_set() else 0)
