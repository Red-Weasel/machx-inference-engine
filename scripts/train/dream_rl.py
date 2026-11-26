#!/usr/bin/env python3
"""Dream-RL: reinforcement learning of a model FOR the Dream harness (the Hugging Face harness-adapter recipe, in Dream).

  dream_rl.py --model DIR --out NEW_DIR --live DIR [--steps 20] [--tasks-per-step 6] [--group 4] [--minutes 12]
              [--check-every 10] [--baseline-label base_plain_high]

Every attempt is a REAL Dream session: Dream's unchanged agent loop, system page and tools (python -m dream.task_run),
in a sandbox (no network, hidden tests out of view, memory-capped), served by the engine from the current weights.
The finished workspace is graded by the task's hidden tests: reward 1 if they pass, else 0 (a session that runs out of
time, stalls or never writes the answer scores 0). Per task, `group` attempts; advantage (r - mean) / std; the training
processes (inkling2/grpo_worker.py) learn from EVERY assistant turn of each session (its thinking and its tool calls,
with Dream's tool results as context), weighted by the session's advantage; then the engine serves the updated weights.
Held-out: the 52 held-out tasks (4 families never trained on). Step 0 is the plain model's Dream baseline (records of
--baseline-label, same tasks, same setup, 3 tries each); a 1-try check every --check-every steps; a final 3-try check.
The model folder is always written at the end; its TRAINING.md gives the before / after held-out scores.
Live in Dream: --live gets metrics.jsonl + attempts.jsonl (each session as it finishes), attached to the Train tab.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "coding_dataset"))
import common as C  # noqa: E402
import grpo  # noqa: E402

VENV = os.path.expanduser("~/models/inkling2-train")
SAMPLE_DIR = os.path.expanduser("~/.local/share/machx-train/dream-sample")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--live", required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--tasks-per-step", type=int, default=6)
ap.add_argument("--group", type=int, default=4)
ap.add_argument("--minutes", type=float, default=12)
ap.add_argument("--eval-minutes", type=float, default=15)
ap.add_argument("--check-every", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--max-len", type=int, default=12288)
ap.add_argument("--baseline-label", default="base_plain_high")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rft-per-task", type=int, default=0,
                help="first learn from the model's own passing Dream sessions already recorded (train tasks only): the N "
                     "that used the fewest output tokens per task, as plain supervised steps")
ap.add_argument("--rft-batch", type=int, default=24)
ap.add_argument("--rft-from", default="", help="comma-separated records labels to learn from instead of earlier Dream-RL "
                "runs (e.g. coached sessions); a coach's note is cut from each prompt before training")
ap.add_argument("--effort", default="", help="the engine's reasoning effort for Dream sessions (e.g. medium)")
a = ap.parse_args()

RUN = os.path.basename(a.out.rstrip("/"))
REC = os.path.join(C.RECORDS, "rl_" + RUN)
os.makedirs(a.live, exist_ok=True)
METRICS, FEED = os.path.join(a.live, "metrics.jsonl"), os.path.join(a.live, "attempts.jsonl")
feed_lock = threading.Lock()
STOP = {"flag": False}
signal.signal(signal.SIGTERM, lambda *_: STOP.__setitem__("flag", True))
signal.signal(signal.SIGINT, lambda *_: STOP.__setitem__("flag", True))
t_start = time.time()


def log(m):
    print(m, file=sys.stderr, flush=True)


def feed(*rows):
    with feed_lock, open(FEED, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def metric(row):
    with open(METRICS, "a") as f:
        f.write(json.dumps(dict(row, elapsed_seconds=round(time.time() - t_start, 1))) + "\n")


def phase(step, name):
    feed({"type": "phase", "step": step, "phase": name, "t": round(time.time(), 1)})


def readable(msgs):
    out, think = [], []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict)) if isinstance(c, list) else (c or "")
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


def final_text(msgs):
    if not msgs or msgs[-1].get("role") != "assistant" or msgs[-1].get("tool_calls"):
        return None
    c = msgs[-1].get("content")
    c = "".join(p.get("text", "") for p in c if isinstance(p, dict)) if isinstance(c, list) else (c or "")
    return c.rsplit("</think>", 1)[-1].strip() or None


def session(t, slot, minutes, dest):
    """One Dream session on task t in sandbox slot `slot`; graded; record kept in dest. -> dict"""
    cmd, ws, rec, home, tmp = C.dream_cmd(slot, minutes)
    for d in (ws, rec, home, tmp):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, mode=0o700)
    os.makedirs(os.path.join(rec, "root", "memory"))
    C.seed_workspace(t, ws)
    open(os.path.join(rec, "prompt.md"), "w").write(t["statement"])
    with open(os.path.join(rec, "stdout.txt"), "w") as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        deadline, killed = time.time() + minutes * 60 + 120, False
        while p.poll() is None:
            time.sleep(1)
            if time.time() > deadline:
                os.makedirs(os.path.join(rec, "control"), exist_ok=True)
                open(os.path.join(rec, "control", "stop"), "w").close()
                try:
                    p.wait(20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                killed = True
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
    if res.get("ended") not in (None, "finished"):
        reason = f"did_not_finish ({res.get('ended')})"
    g = {"passed": passed, "reason": reason or "passed", "ended": res.get("ended")}
    os.makedirs(dest, exist_ok=True)
    for x in ("result.json", "session.json"):
        if os.path.exists(os.path.join(rec, x)):
            shutil.copy(os.path.join(rec, x), os.path.join(dest, x))
    json.dump(g, open(os.path.join(dest, "grade.json"), "w"))
    for d in (ws, rec, home, tmp):
        shutil.rmtree(d, ignore_errors=True)
    return {"passed": passed, "reason": g["reason"], "ended": res.get("ended"), "msgs": msgs, "res": res}


def run_sessions(jobs, minutes, step, phase_name):
    """jobs: [(key, task, attempt, of, dest)] -> {key: result}; 12 at a time; each one fed to Dream as it finishes."""
    out, slots = {}, list(range(12))
    lock = threading.Lock()

    def one(j):
        key, t, n, of, dest = j
        if STOP["flag"]:
            return key, None
        with lock:
            slot = slots.pop()
        try:
            r = session(t, slot, minutes, dest)
        finally:
            with lock:
                slots.append(slot)
        content, reasoning = readable(r["msgs"])
        res = r["res"]
        feed({"type": "attempt", "step": step, "phase": phase_name, "task_id": t["id"], "family": t["family"],
              "attempt": n, "of": of, "reward": 1.0 if r["passed"] else 0.0,
              "reason": f"{r['reason']} · {res.get('requests', 0)} steps, {res.get('tool_calls', 0)} tool calls",
              "finish_reason": r["ended"], "tokens": res.get("output_tokens") or 0, "content": content,
              "reasoning": reasoning, "advantage": None})
        return key, r

    with cf.ThreadPoolExecutor(12) as ex:
        for key, r in ex.map(one, jobs):
            if r is not None:
                out[key] = r
    return out


def check(step, tries, engine):
    phase(step, f"held-out check ({len(held)} tasks x {tries})")
    jobs = [((t["id"], n), t, n + 1, tries, os.path.join(REC, f"check{step}", t["id"], str(n)))
            for t in held for n in range(tries)]
    engine.start()
    try:
        r = run_sessions(jobs, a.eval_minutes, step, "heldout")
    finally:
        engine.stop()
    p = sum(v["passed"] for v in r.values()) / max(1, len(r))
    log(f"held-out check at step {step}: {sum(v['passed'] for v in r.values())} of {len(r)} pass ({100 * p:.1f} %)")
    return p, len(r)


# ------------------------------------------------------------------------------------------------- setup
tasks = C.load_tasks()
train = sorted((t for t in tasks.values() if t["split"] == "train"), key=lambda t: t["id"])
held = sorted((t for t in tasks.values() if t["split"] == "heldout"), key=lambda t: t["id"])
order = list(range(len(train)))
random.Random(a.seed + 1).shuffle(order)
sample = max((os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR) if f.endswith(".json")), key=os.path.getmtime)
tools = json.load(open(sample))["tools"]

base = os.path.join(C.RECORDS, a.baseline_label)
bp = bn = 0
for t in os.listdir(base) if os.path.isdir(base) else []:
    for n in os.listdir(os.path.join(base, t)):
        g = os.path.join(base, t, n, "grade.json")
        if os.path.exists(g):
            bn += 1
            bp += bool(json.load(open(g))["passed"])
baseline = bp / bn if bn else None

work = a.out + ".work"
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
procs = []
for r, c in enumerate([0, 1]):
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=92G", "-p", "MemorySwapMax=0", "--",
           os.path.join(VENV, "xpu_env.sh"), os.path.join(VENV, "venv/bin/python"), "-I",
           os.path.join(HERE, "inkling2", "grpo_worker.py"), "--model", a.model, "--out", a.out, "--work", work,
           "--rank", str(r), "--world", "2", "--card", str(c), "--lr", str(a.lr), "--max-len", str(a.max_len),
           "--seed", str(a.seed)]
    env = dict(os.environ, ZE_AFFINITY_MASK=str(c), PYTORCH_ALLOC_CONF="expandable_segments:True")
    env = {k: v for k, v in env.items() if not k.startswith("PYTHON")}
    procs.append(subprocess.Popen(cmd, stdout=open(os.path.join(work, f"worker{r}.log"), "a"), stderr=subprocess.STDOUT,
                                  env=env, cwd=VENV))
engine = grpo.Engine(os.path.join(work, "policy"), 11435, 12, 131072, log)
engine.effort = a.effort or "high"
print(f"Dream-RL on {os.path.basename(a.model)}: {a.steps} steps of {a.tasks_per_step} tasks x {a.group} Dream sessions; "
      f"baseline {bp}/{bn}; live view {a.live}", flush=True)
phase(0, f"Dream-RL: {a.steps} steps x {a.tasks_per_step} tasks x {a.group} Dream sessions; loading the model")
if baseline is not None:
    metric({"step": 0, "eval_success_rate": baseline, "learning_rate": 0.0})
    feed({"type": "step", "step": 0, "mean_reward": None, "groups_used": 0, "groups": 0, "cut_off": 0,
          "heldout_pass": baseline, "heldout_step": 0, "seconds": 0})


def workers_ok():
    if os.path.exists(os.path.join(work, "failed")) or any(p.poll() not in (None, 0) for p in procs):
        raise RuntimeError("a training process failed (see the worker logs in " + work + ")")


def wait_file(path):
    while not os.path.exists(path):
        workers_ok()
        time.sleep(1)


def own_best_sessions():
    """The model's own passing Dream sessions on TRAIN tasks (records of earlier Dream-RL runs), fewest tokens first."""
    import glob
    ids, per = {t["id"] for t in train}, {}
    pats = ([os.path.join(C.RECORDS, l, "*", "*", "grade.json") for l in a.rft_from.split(",") if l]
            or [os.path.join(C.RECORDS, "rl_*", "step*", "*", "*", "grade.json")])
    for g in (x for p in pats for x in glob.glob(p)):
        d = os.path.dirname(g)
        task = os.path.basename(os.path.dirname(d))
        if task not in ids or not json.load(open(g))["passed"]:
            continue
        try:
            r = json.load(open(os.path.join(d, "result.json")))
            msgs = json.load(open(os.path.join(d, "session.json")))["messages"]
        except (OSError, ValueError, KeyError):
            continue
        msgs = C.strip_coach([m for m in msgs if m.get("role") in ("system", "user", "assistant", "tool")])
        per.setdefault(task, []).append((r.get("output_tokens") or 10 ** 9, d, msgs))
    out = []
    for task in sorted(per):
        out += [m for _, _, m in sorted(per[task], key=lambda x: x[0])[:a.rft_per_task]]
    random.Random(a.seed + 7).shuffle(out)
    return out, len(per)


secs = []
k = 0
try:
    for r in range(2):
        wait_file(os.path.join(work, f"parked{r}.0"))
    if a.rft_per_task:
        sess, n_tasks = own_best_sessions()
        log(f"learning first from {len(sess)} of its own passing Dream sessions on {n_tasks} training tasks")
        for i in range(0, len(sess), a.rft_batch):
            if STOP["flag"]:
                break
            ts, step = time.time(), k + 1
            phase(step, f"learning from its own best passing Dream sessions: {min(i + a.rft_batch, len(sess))} of {len(sess)}")
            batch = [{"messages": m, "tools": tools, "adv": 1.0, "all_turns": True} for m in sess[i:i + a.rft_batch]]
            with open(os.path.join(work, f"batch.{step}.json"), "w") as f:
                json.dump(batch, f)
            open(os.path.join(work, f"go.{step}"), "w").close()
            wait_file(os.path.join(work, f"result.{step}.json"))
            out = json.load(open(os.path.join(work, f"result.{step}.json")))
            for r in range(2):
                wait_file(os.path.join(work, f"parked{r}.{step}"))
            k = step
            metric({"step": k, "loss": out.get("loss"), "learning_rate": out.get("learning_rate", 0.0)})
            feed({"type": "step", "step": k, "mean_reward": None, "groups_used": len(batch), "groups": len(batch),
                  "cut_off": 0, "heldout_pass": None, "heldout_step": None, "seconds": round(time.time() - ts, 1)})
            log(f"step {k}: learned from {len(batch)} of its own passing sessions ({out.get('trained_tokens')} tokens), "
                f"{time.time() - ts:.0f} s")
    total = k + a.steps
    while k < total and not STOP["flag"]:
        ts = time.time()
        step = k + 1
        picks = [train[order[(k * a.tasks_per_step + j) % len(train)]] for j in range(a.tasks_per_step)]
        phase(step, f"writing attempts: {len(picks)} tasks x {a.group} Dream sessions")
        jobs = [((t["id"], g), t, g + 1, a.group, os.path.join(REC, f"step{step}", t["id"], str(g)))
                for t in picks for g in range(a.group)]
        engine.start()
        try:
            res = run_sessions(jobs, a.minutes, step, "train")
        finally:
            engine.stop()
        if STOP["flag"]:
            break
        phase(step, "training")
        batch, used, rs = [], 0, []
        for t in picks:
            grp = [(g, res[(t["id"], g)]) for g in range(a.group) if (t["id"], g) in res]
            vals = [1.0 if r["passed"] else 0.0 for _, r in grp]
            rs += vals
            if len(vals) < 2:
                continue
            mu = sum(vals) / len(vals)
            sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
            if sd < 1e-9:
                continue
            used += 1
            for (g, r), v in zip(grp, vals):
                msgs = [m for m in r["msgs"] if m.get("role") in ("system", "user", "assistant", "tool")]
                if any(m.get("role") == "assistant" for m in msgs):
                    batch.append({"messages": msgs, "tools": tools, "adv": (v - mu) / (sd + 1e-4), "all_turns": True})
        with open(os.path.join(work, f"batch.{step}.json"), "w") as f:
            json.dump(batch, f)
        open(os.path.join(work, f"go.{step}"), "w").close()
        wait_file(os.path.join(work, f"result.{step}.json"))
        out = json.load(open(os.path.join(work, f"result.{step}.json")))
        for r in range(2):
            wait_file(os.path.join(work, f"parked{r}.{step}"))
        k = step
        mean_r = sum(rs) / max(1, len(rs))
        secs.append(time.time() - ts)
        row = {"step": k, "reward": round(mean_r, 4), "learning_rate": out.get("learning_rate", 0.0),
               "eta_seconds": round(sum(secs[-3:]) / len(secs[-3:]) * (total - k))}
        hp = None
        if k % a.check_every == 0 and k < total:
            hp, _ = check(k, 1, engine)
            row["eval_success_rate"] = hp
        metric(row)
        feed({"type": "step", "step": k, "mean_reward": round(mean_r, 4), "groups_used": used, "groups": len(picks),
              "cut_off": sum(1 for r in res.values() if r["ended"] == "time_limit"), "heldout_pass": hp,
              "heldout_step": k if hp is not None else None, "seconds": round(time.time() - ts, 1)})
        log(f"step {k}/{total}: {sum(rs):.0f} of {len(rs)} Dream sessions passed, {used} of {len(picks)} tasks had a "
            f"signal, {time.time() - ts:.0f} s")
    final, n_final = (None, 0)
    if k > 0:
        final, n_final = check(k, 3, engine)
        metric({"step": k, "eval_success_rate": final})
        feed({"type": "step", "step": k, "mean_reward": None, "groups_used": 0, "groups": 0, "cut_off": 0,
              "heldout_pass": final, "heldout_step": k, "seconds": 0})
except Exception as e:                                                       # noqa: BLE001
    engine.stop()
    open(os.path.join(work, "failed"), "w").close()
    for p in procs:
        try:
            p.wait(120)
        except subprocess.TimeoutExpired:
            p.kill()
    print(f"Dream-RL failed: {str(e).splitlines()[0][:300] if str(e) else type(e).__name__}", file=sys.stderr, flush=True)
    sys.exit(1)

beat = final is not None and baseline is not None and final > baseline
md = f"""# Training record

- Trained by Dream-RL (reinforcement learning inside the Dream harness: every attempt a real Dream session, reward = the task's hidden tests pass), LoRA rank 16 on attention q/k/v/o, folded into the BF16 weights
- Base model: {os.path.basename(a.model)}
- Steps: {k}{f" (first, supervised on its own {a.rft_per_task} fewest-token passing Dream session(s) per training task)" if a.rft_per_task else ""}; RL steps {a.steps} of {a.tasks_per_step} tasks x {a.group} Dream sessions; learning rate {a.lr:g}
- Held-out in Dream (52 tasks of 4 families never trained on, 3 tries each): {f"{100 * baseline:.1f} %" if baseline is not None else "?"} before -> {f"{100 * final:.1f} %" if final is not None else "?"} after
"""
with open(os.path.join(work, "finish.json"), "w") as f:
    json.dump({"step": k, "training_md": md}, f)                         # every finished run keeps its model
for p in procs:
    try:
        p.wait(3600)
    except subprocess.TimeoutExpired:
        p.kill()
msg = (f"Dream-RL: held-out in Dream {100 * baseline:.1f} % before -> {100 * final:.1f} % after {k} steps"
       if final is not None and baseline is not None else "Dream-RL stopped before a final check")
if os.path.isdir(a.out):
    print(msg + ("" if beat else " (did not beat the plain model)") + f"; model written to {a.out}", flush=True)
    sys.exit(0)
print(msg + "; the model folder was not written (see the worker logs in " + work + ")", file=sys.stderr, flush=True)
sys.exit(3)
