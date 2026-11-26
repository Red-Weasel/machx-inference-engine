"""machx-train's GRPO mode (reinforcement learning with checked rewards), started by machx_train.py when the data
file's lines carry a "reward" (see grpo_rewards.py for the task format).

One step:
  1. the engine (ie serve) serves the CURRENT model (<work>/policy) on the cards and writes `group` attempts at each of
     `tasks_per_step` tasks (temperature 1.0, the model's recommended thinking effort);
  2. the engine stops; every attempt is scored by its task's reward (tests in capped scopes);
  3. advantages within each task's group: (r - mean) / (std + 1e-4); a group whose attempts all scored the same
     teaches nothing and is dropped;
  4. the training processes (inkling2/grpo_worker.py, one per card, parked in host memory while the engine ran) take
     one LoRA update on the kept attempts, then rewrite the policy folder's attention weights from the adapter.
Held-out check: before the first step and every `eval_every` steps, one attempt at each held-out task in the same
engine session; eval_success_rate = the share that pass. The model folder gets the adapter of the best check (early
stop after `patience` checks without improvement). Metrics: step, reward (the step's mean training reward),
eval_success_rate, learning_rate, elapsed_seconds, eta_seconds, tokens_per_second (the engine's writing speed).
"""
import concurrent.futures as cf
import datetime
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grpo_rewards as GR  # noqa: E402

ENGINE_ROOT = os.path.dirname(os.path.dirname(HERE))
ENV_SH = os.path.join(ENGINE_ROOT, "scripts", "env.sh")
IE_BIN = os.environ.get("MACHX_TRAIN_IE") or os.path.expanduser("~/00 - Inference Engine/build/src/ie")


def is_grpo_file(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    return isinstance(json.loads(line).get("reward"), dict)
                except (ValueError, AttributeError):
                    return False
    return False


def load_tasks(path, fail):
    tasks = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            t = json.loads(line)
            if not isinstance(t.get("messages"), list) or not t["messages"]:
                fail(f"line {n} of {os.path.basename(path)} has no 'messages'")
            if t["messages"][-1].get("role") != "user":
                fail(f"line {n} of {os.path.basename(path)}: a task's last message must be the user's")
            try:
                GR.validate(t.get("reward"), f"line {n} of {os.path.basename(path)}")
            except GR.RewardError as e:
                fail(str(e))
            t.setdefault("id", f"line{n}")
            tasks.append(t)
    if not tasks:
        fail(f"{os.path.basename(path)} has no tasks")
    return tasks


class Engine:
    def __init__(self, folder, port, parallel, ctx, log):
        self.folder, self.port, self.parallel, self.ctx, self.log = folder, port, parallel, ctx, log
        self.p = None

    def start(self):
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=96G", "-p", "MemorySwapMax=0", "--",
               "bash", "-c", f'source "{ENV_SH}" >/dev/null 2>&1; exec "{IE_BIN}" serve "{self.folder}" --gpus 2 '
               f'--parallel {self.parallel} --ctx {self.ctx} --port {self.port} --reasoning-effort {getattr(self, "effort", "high")}']
        self.logf = open(os.path.join(os.path.dirname(self.folder), "engine.log"), "a")
        t0 = time.time()
        self.p = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT)
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise RuntimeError("the engine stopped while loading the model (see engine.log in the work folder)")
            try:
                if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2).read().lower():
                    self.log(f"engine up in {time.time() - t0:.0f} s")
                    return
            except OSError:
                pass
            time.sleep(2)
        raise RuntimeError("the engine did not come up within 15 minutes")

    def stop(self):
        if self.p and self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(90)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait(30)
        self.p = None

    def ask(self, task, seed, temperature, max_new):
        body = {"messages": task["messages"], "temperature": temperature, "top_p": 1.0, "top_k": 0, "min_p": 0,
                "max_tokens": max_new, "seed": seed, "stream": False}
        if task.get("tools"):
            body["tools"] = task["tools"]
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        c = r["choices"][0]
        m = c.get("message") or {}
        return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
                "finish_reason": c.get("finish_reason"), "tokens": (r.get("usage") or {}).get("completion_tokens", 0)}

    def many(self, jobs, temperature, max_new, stop):
        """jobs: [(key, task, seed)] -> {key: reply}; the replies are written `parallel` at a time."""
        out = {}
        with cf.ThreadPoolExecutor(self.parallel) as ex:
            futs = {ex.submit(self.ask, t, s, temperature, max_new): k for k, t, s in jobs}
            for f in cf.as_completed(futs):
                if stop["flag"]:
                    break
                out[futs[f]] = f.result()
        return out


def run(a, spec, model, out, cards, fail, venv, here):
    log = lambda m: print(m, file=sys.stderr, flush=True)          # noqa: E731
    tasks = load_tasks(a.data, fail)
    if a.eval:
        held = load_tasks(a.eval, fail)
        train = tasks
    else:
        rnd = random.Random(a.seed)
        idx = list(range(len(tasks)))
        rnd.shuffle(idx)
        n = min(32, max(4, round(0.1 * len(tasks)))) if len(tasks) >= 20 else 0
        held, train = [tasks[i] for i in idx[:n]], [tasks[i] for i in idx[n:]]
    if not train:
        fail("the data file has no task left to train on")
    per_pass = math.ceil(len(train) / a.tasks_per_step)
    steps = min(a.max_steps, per_pass) if a.max_steps else per_pass
    every = a.eval_every or max(5, math.ceil(steps / 10))
    ctx = min(32768, a.max_new + 4096)

    work = out + ".work"
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work)
    open(a.metrics, "a").close()
    stop = {"flag": False}
    procs = []

    def on_term(*_):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    for r, c in enumerate(cards):
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={spec['host_gib_per_card'] + 20}G",
               "-p", "MemorySwapMax=0", "--", os.path.join(venv, "xpu_env.sh"), os.path.join(venv, "venv/bin/python"),
               "-I", os.path.join(here, "inkling2", "grpo_worker.py"), "--model", model, "--out", out, "--work", work,
               "--rank", str(r), "--world", str(len(cards)), "--card", str(c), "--lr", str(a.lr),
               "--max-len", str(a.max_len), "--seed", str(a.seed)]
        env = dict(os.environ, ZE_AFFINITY_MASK=str(c), PYTORCH_ALLOC_CONF="expandable_segments:True")
        procs.append(subprocess.Popen(cmd, stdout=open(os.path.join(work, f"worker{r}.log"), "a"),
                                      stderr=subprocess.STDOUT, env=env, cwd=venv))
    print(f"GRPO on {spec['name']}, card(s) {','.join(map(str, cards))}: {len(train)} training tasks, {len(held)} "
          f"held-out, {steps} steps of {a.tasks_per_step} tasks x {a.group} attempts; logs in {work}", flush=True)

    def check_workers():
        if os.path.exists(os.path.join(work, "failed")) or any(p.poll() not in (None, 0) for p in procs):
            reason = None
            for r in range(len(cards)):
                for line in open(os.path.join(work, f"worker{r}.log"), errors="replace"):
                    if line.startswith("training failed"):
                        reason = line.strip()
            raise RuntimeError(reason or "a training process stopped (see the worker logs in the work folder)")

    def wait_parked(k):
        while not all(os.path.exists(os.path.join(work, f"parked{r}.{k}")) for r in range(len(cards))):
            check_workers()
            time.sleep(1)

    engine = Engine(os.path.join(work, "policy"), a.port, a.parallel, ctx, log)
    t_start = time.time()
    order = list(range(len(train)))
    random.Random(a.seed + 1).shuffle(order)
    state = {"best": None, "best_step": 0, "since_best": 0, "evals": [], "rewards": []}
    stopped_by = "the data (one pass)" if steps == per_pass else f"--max-steps {a.max_steps}"
    step_secs = []

    def write(rec):
        with open(a.metrics, "a") as f:
            f.write(json.dumps(rec) + "\n")

    feed_path = os.path.join(os.path.dirname(os.path.abspath(a.metrics)), "attempts.jsonl")

    def feed(*recs):                                     # the live view (Dream reads it beside the metrics file)
        with open(feed_path, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    def phase(k, name):
        feed({"type": "phase", "step": k, "phase": name, "t": round(time.time(), 1)})

    def finish(step, md):
        with open(os.path.join(work, "finish.json.tmp"), "w") as f:
            json.dump({"step": step, "training_md": md}, f)
        os.replace(os.path.join(work, "finish.json.tmp"), os.path.join(work, "finish.json"))

    k = 0
    try:
        wait_parked(0)
        while True:
            ts = time.time()
            do_eval = bool(held) and (k == 0 or k % every == 0 or k == steps)
            jobs = []
            if k < steps:
                for j in range(a.tasks_per_step):
                    ti = order[(k * a.tasks_per_step + j) % len(train)]
                    for g in range(a.group):
                        jobs.append((("t", ti, g), train[ti], a.seed * 100003 + k * 1009 + j * 31 + g))
            if do_eval:
                jobs += [(("h", i, r), t, 7 + 101 * i + r) for i, t in enumerate(held) for r in range(a.eval_attempts)]
            if not jobs or stop["flag"]:
                break
            phase(k if k < steps else k, "writing attempts" if k < steps else "held-out check")
            engine.start()
            try:
                replies = engine.many(jobs, 1.0, a.max_new, stop)
            finally:
                engine.stop()
            if stop["flag"]:
                stopped_by = "a stop request"
                break
            phase(k, "scoring")
            gen_tok = sum(r["tokens"] for r in replies.values())
            gen_s = time.time() - ts
            with cf.ThreadPoolExecutor(8) as ex:
                keys = list(replies)
                tasks_of = {kk: (train if kk[0] == "t" else held)[kk[1]] for kk in keys}
                scored = dict(zip(keys, ex.map(lambda kk: GR.score(tasks_of[kk], replies[kk]), keys)))
            scores = {kk: v[0] for kk, v in scored.items()}
            adv_of = {}
            for j in range(a.tasks_per_step if k < steps else 0):
                ti = order[(k * a.tasks_per_step + j) % len(train)]
                vals = [scores[("t", ti, g)] for g in range(a.group) if ("t", ti, g) in scores]
                if len(vals) >= 2:
                    mu = sum(vals) / len(vals)
                    sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
                    if sd >= 1e-9:
                        for g in range(a.group):
                            if ("t", ti, g) in scores:
                                adv_of[("t", ti, g)] = (scores[("t", ti, g)] - mu) / (sd + 1e-4)
            feed(*[{"type": "attempt", "step": k + 1 if kk[0] == "t" else k, "phase": "train" if kk[0] == "t" else "heldout",
                    "task_id": tasks_of[kk].get("id"), "family": tasks_of[kk].get("family"),
                    "attempt": kk[2] + 1, "of": a.group if kk[0] == "t" else a.eval_attempts,
                    "reward": scored[kk][0], "reason": scored[kk][1], "finish_reason": replies[kk]["finish_reason"],
                    "tokens": replies[kk]["tokens"], "content": replies[kk]["content"],
                    "reasoning": replies[kk]["reasoning"], "advantage": adv_of.get(kk) if kk[0] == "t" else None}
                   for kk in sorted(keys, key=lambda x: (x[0] != "h", x[1], x[2]))])
            rec = {"step": k, "elapsed_seconds": round(time.time() - t_start, 1),
                   "tokens_per_second": round(gen_tok / max(1.0, gen_s), 1)}
            early = False
            if do_eval:
                hv = [v for kk, v in scores.items() if kk[0] == "h"]
                ev = sum(hv) / len(hv)
                state["evals"].append({"step": k, "pass": ev})
                rec["eval_success_rate"] = ev
                if state["best"] is None or ev > state["best"] + 1e-9:    # only a real gain counts: a tie with an
                    state["best"], state["best_step"], state["since_best"] = ev, k, 0    # earlier step is not one
                else:
                    state["since_best"] += 1
                    early = state["since_best"] >= a.patience
                log(f"held-out check at step {k}: {sum(hv):.0f} of {len(hv)} attempts pass ({100 * ev:.1f} %; "
                    f"{len(held)} tasks x {a.eval_attempts})")
            if k >= steps or early:
                if k == 0:
                    write(dict(rec, learning_rate=0.0))
                if early:
                    stopped_by = f"early stop: no held-out improvement in {a.patience} checks"
                break
            batch, groups_used, rsum, cut = [], 0, [], 0
            for j in range(a.tasks_per_step):
                ti = order[(k * a.tasks_per_step + j) % len(train)]
                rs = [(g, scores[("t", ti, g)]) for g in range(a.group) if ("t", ti, g) in scores]
                rsum += [v for _, v in rs]
                cut += sum(1 for g, _ in rs if replies[("t", ti, g)]["finish_reason"] == "length")
                vals = [v for _, v in rs]
                if len(vals) < 2:
                    continue
                mu = sum(vals) / len(vals)
                sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
                if sd < 1e-9:
                    continue
                groups_used += 1
                for g, v in rs:
                    rp = replies[("t", ti, g)]
                    batch.append({"messages": train[ti]["messages"] + [
                        {"role": "assistant", "content": rp["content"], "reasoning_content": rp["reasoning"]}],
                        "tools": train[ti].get("tools"), "adv": (v - mu) / (sd + 1e-4)})
            mean_r = sum(rsum) / max(1, len(rsum))
            state["rewards"].append(mean_r)
            if k == 0 and do_eval:
                write(dict(rec, learning_rate=0.0))
            k += 1
            phase(k, "training")
            with open(os.path.join(work, f"batch.{k}.json"), "w") as f:     # an empty batch: the workers just
                json.dump(batch, f)                                            # carry the adapter to step k
            open(os.path.join(work, f"go.{k}"), "w").close()
            while not os.path.exists(os.path.join(work, f"result.{k}.json")):
                check_workers()
                time.sleep(1)
            res = json.load(open(os.path.join(work, f"result.{k}.json")))
            wait_parked(k)
            step_secs.append(time.time() - ts)
            rec = {"step": k, "reward": round(mean_r, 4), "learning_rate": res.get("learning_rate", 0.0),
                   "elapsed_seconds": round(time.time() - t_start, 1),
                   "eta_seconds": round(sum(step_secs[-5:]) / len(step_secs[-5:]) * (steps - k) * 1.1),
                   "tokens_per_second": round(gen_tok / max(1.0, gen_s), 1),
                   "groups_used": groups_used, "groups": a.tasks_per_step, "cut_off": cut,
                   **({"grad_norm": res["grad_norm"]} if batch else {})}
            write(rec)
            feed({"type": "step", "step": k, "mean_reward": round(mean_r, 4), "groups_used": groups_used,
                  "groups": a.tasks_per_step, "cut_off": cut,
                  "heldout_pass": state["evals"][-1]["pass"] if do_eval else None,      # measured BEFORE this step's
                  "heldout_step": state["evals"][-1]["step"] if do_eval else None,      # update, i.e. at step k - 1
                  "seconds": round(time.time() - ts, 1)})
            log(f"step {k}/{steps}: mean reward {mean_r:.3f}, {groups_used} of {a.tasks_per_step} groups had a signal, "
                f"{cut} attempts cut off, {time.time() - ts:.0f} s")
            if stop["flag"]:
                stopped_by = "a stop request"
                break
    except Exception as e:                                       # noqa: BLE001
        engine.stop()
        open(os.path.join(work, "failed"), "w").close()
        for p in procs:
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
        fail(f"GRPO failed: {str(e).splitlines()[0][:300] if str(e) else type(e).__name__}")

    beat = bool(state["evals"]) and state["best_step"] > 0               # a step beat the starting model on held-out
    use = state["best_step"] if beat else k              # never better: the last step's model is still written
    e0 = state["evals"][0]["pass"] if state["evals"] else None
    eb = next((e["pass"] for e in state["evals"] if e["step"] == use), None)
    R = state["rewards"]
    md = f"""# Training record

- Trained: {datetime.date.today().isoformat()}, by machx-train GRPO (reinforcement learning with checked rewards; LoRA rank 16 on attention q/k/v/o of all 24 layers, folded into the BF16 weights)
- Base model: {os.path.basename(model)}
- Tasks: {os.path.basename(a.data)}: {len(train)} training tasks, {len(held)} held-out ({'from ' + os.path.basename(a.eval) if a.eval else 'held out of the data file'})
- Steps: {k} of {steps} planned, {a.tasks_per_step} tasks x {a.group} attempts a step, at most {a.max_new} new tokens an attempt, learning rate {a.lr:g}; ended by {stopped_by}
- Mean training reward (first / last 3 steps): {(sum(R[:3]) / len(R[:3])) if R else float('nan'):.3f} -> {(sum(R[-3:]) / len(R[-3:])) if R else float('nan'):.3f}
- Held-out pass rate: {f"{100 * e0:.1f} % before -> {100 * eb:.1f} % at step {use}" if (e0 is not None and eb is not None) else "no held-out check"}
- Weights saved from step {use} ({"the best held-out check" if beat else ("the last step; no step beat the starting model on held-out" if state["evals"] else "no held-out set")})
"""
    if state["evals"] and not beat:
        best_after = max((e["pass"] for e in state["evals"][1:]), default=None)
        nobeat = (f"the held-out pass rate never beat the starting model ({100 * state['evals'][0]['pass']:.1f} % before"
                  + (f", best {100 * best_after:.1f} % after" if best_after is not None else "")
                  + "); the last step's model is written anyway")
        log(nobeat)
    finish(use, md)
    rcs = []
    for p in procs:
        try:
            rcs.append(p.wait(3600))
        except subprocess.TimeoutExpired:
            p.kill()
            rcs.append(-9)
    if rcs and rcs[0] == 0 and os.path.isdir(out):
        shutil.rmtree(work, ignore_errors=True)
        print(f"model written to {out}", flush=True)
        return 0
    fail("GRPO ended without a model folder: " + ("it stopped before the first step" if use <= 0 else
                                                  f"see the worker logs in {work}"), 3 if stop["flag"] else 1)
