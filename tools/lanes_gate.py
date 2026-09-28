#!/usr/bin/env python3
"""P4 B8 (docs/lanes/LANES_SERVE.md): the generic `ie serve --parallel N` gate -- one command per criterion, for any arch.

  lanes_gate.py --ie BIN --model PATH --arch ARCH --gate GATE --out DIR [--port P] [--parallel N] [--ctx C] [--slot-ctx S]
                [--max-tokens M] [--n K] [--requests R] [--ref FILE] [--env K=V ...] [--serve-arg ARG ...]

It starts `BIN serve PATH --host 127.0.0.1 --port P --ctx C --parallel N [--slot-ctx S]` itself (server log in
DIR/<gate>_server.log), waits for /health, runs the gate, stops the server with POST /admin/shutdown and waits for it to exit.
Run it as ONE guarded GPU job, e.g.  scripts/ie-run-guarded --mem 240G --timeout 540 python3 tools/lanes_gate.py ...
The server is read through HTTP and procfs only (RssAnon, threads, its own VRAM from /proc/<pid>/fdinfo): nothing here talks
to the GPU. MiMo B4's and V4.1 B6b's test clients (tools/ds41_parallel_serve_test.py) made generic.

Gates (exit 0 = PASS):
  sse       --parallel 1 against HEAD: three fixed streaming requests as the FIRST requests of a fresh server (T 0.7 seed 1234;
            greedy with a tool; greedy thinking), raw SSE bytes with "id"/"created" masked, one md5 each. --save FILE writes
            them; --ref FILE (a HEAD capture) must match byte for byte.
  identity  greedy (the arch's identity env: CPU expert legs off), K conversations with different first system tokens: warm
            (each once, alone), solo (each again, alone), batch (all at once), cancel (all at once, request 1 closed after 6
            chunks). batch == solo and the cancel survivors == solo. --save/--ref compare solo texts across servers (e.g. the
            lanes' solo against --parallel 1).
  aba       greedy, no tools: solo -> K concurrent -> solo; the server's own /health "tokens" while exactly 1 / K / 1 lanes decode.
            Prints the ratio; --min-ratio X makes it a criterion.
  followup  greedy, K conversations: turn 1 of each alone, then turn 2 of all at once (the history + turn 1's reply + a new user
            message, restored from the prompt cache). --save/--ref compare both turns' texts across servers (e.g. turn 2 at
            --parallel 2 against --parallel 1); turn 2's cached_tokens must be > 0.
  memory    R requests in waves of K: RssAnon, the server's VRAM per card and its threads before, after wave 1 and at the end;
            growth after wave 1 < 64 MiB RssAnon and < 64 MiB VRAM per card, no thread growth.
  scale     P4 B14 (~/ds41_work/p60/b14/PLAN.md section 5): greedy, no tools; per repeat the chain A B(n1) A B(n2) ... A over
            --ns (each <= --parallel; 1 = the solo arms only), so every B arm sits between two solo arms of the same process
            (A-B-A). Every conversation is prefilled alone first (its cold TTFT), so the arms decode from the prompt cache. Each
            arm is sized to a steady window of >= --min-window s (all its lanes decoding; an arm that falls short is re-run once
            with more tokens, the learned per-lane rate then sizes the next repeats). Per arm: the aggregate tok/s over the
            window (client-side: content chunks in the window; plus the server's /health "tokens" while exactly N decode, when
            the arch reports it), per-lane tok/s, p50/p95 inter-chunk gap, TTFT (warm; with --arrival the last request of a B arm
            starts after the others decode: its TTFT under load), and each card's engine busy from /proc/<pid>/fdinfo
            drm-cycles (ie-glm-lanes-test's method) sampled at --util-hz, plus the VRAM peak. --cold adds, after the repeats, N
            NEW conversations at once (cold TTFT under contention). --xpu-smi also records `xpu-smi dump` beside it (raw; whether
            its fields report on the B70 under xe is unverified). Stops starting arms past --deadline s (then FAIL: incomplete).
            Medians and spreads over the repeats; everything in DIR/scale.json. --shared-prefix: agent traffic -- every
            request has the SAME system prompt (SYSTEM + --pad lines + --tools N tool schemas) and a unique user turn (so each
            arm's prompts restore that shared prefix from the prompt cache; the TTFTs show whether the restores serialise).
            The server log's lane VRAM lines and --slot-ctx are recorded.
  shared    P4 B15, a shared-prefix restore == a cold prefill: requests X and Y share one system prompt (SYSTEM + --pad lines +
            --tools schemas) and differ in the user turn; greedy, --max-tokens, one at a time in --order (xy or yx). The second
            must restore the shared prefix (cached_tokens > 0, below its prompt). --save / --ref compare X's and Y's texts across
            servers: run --order xy --save F on one server and --order yx --ref F on another, so each request is compared cold
            against restored.
  load      load the server, record its lane VRAM lines and /health, stop it (a refused load prints the refusal; exit 1):
            the admission probe for "the largest N that fits at --slot-ctx S".
Standard library only.
"""
import argparse
import hashlib
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

# Per-arch: the server env for byte identity (the CPU expert legs' split follows cache history) and what the lanes need.
ARCH = {
    "qwen4exp": {"identity_env": {}, "note": "Flash-Next: no CPU expert leg; --parallel > 1 needs two cards"},
    "deepseek41": {"identity_env": {"IE_DS41_CPU_MISS": "0"}, "note": "V4.1 B6b"},
    "mimo26": {"identity_env": {"IE_DS41_CPU_MISS": "0"}, "note": "MiMo B4 (its CPU leg is the shared Ds41ExpertTier)"},
    "glm5next": {"identity_env": {"IE_G5_CPU_MISS": "0"}, "note": "GLM (no lanes serving yet)"},
    "qwen35moe": {"identity_env": {}, "note": "the 35B-A3B crown split (P4 B10): no CPU expert leg; --parallel > 1 needs two cards"},
    "qwen35": {"identity_env": {}, "note": "the 27B two-card split (P4 B18): request lanes with rows; IE_QWEN35_LANES=0 = the slot banks "
                                          "(no lanes /health block there: scale measures client-side)"},
}

ap = argparse.ArgumentParser()
ap.add_argument("--ie", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--arch", required=True, choices=sorted(ARCH))
ap.add_argument("--gate", required=True, choices=["sse", "identity", "aba", "memory", "followup", "scale", "load", "shared"])
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=11497)
ap.add_argument("--parallel", type=int, default=2)
ap.add_argument("--ctx", type=int, default=32768)
ap.add_argument("--slot-ctx", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--requests", type=int, default=20)
ap.add_argument("--min-ratio", type=float, default=0.0)
ap.add_argument("--save", default="")
ap.add_argument("--ref", default="")
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--serve-arg", action="append", default=[])
ap.add_argument("--load-timeout", type=int, default=420)
ap.add_argument("--pad", type=int, default=0, help="log lines appended to the system prompt (~30 tokens each): long prompts")
ap.add_argument("--ns", default="1,2,3,4", help="scale: the concurrencies, each <= --parallel")
ap.add_argument("--repeats", type=int, default=3, help="scale: repeats of the A-B-A chain")
ap.add_argument("--min-window", type=float, default=20.0, help="scale: the steady window each arm must reach, s")
ap.add_argument("--deadline", type=float, default=540.0, help="scale: no arm starts past this many s from the tool's start")
ap.add_argument("--util-hz", type=float, default=10.0, help="scale: the fdinfo sampler's rate")
ap.add_argument("--arrival", action="store_true", help="scale: a B arm's last request arrives while the others decode")
ap.add_argument("--cold", action="store_true", help="scale: after the repeats, max(ns) NEW conversations at once")
ap.add_argument("--xpu-smi", action="store_true", help="scale: record `xpu-smi dump --metrics UTILIZATION,MEMORY` per card beside the run (raw)")
ap.add_argument("--shared-prefix", action="store_true", help="scale: agent traffic, one shared system prompt + unique user turns")
ap.add_argument("--tools", type=int, default=0, help="scale: N synthetic tool schemas in every request (agent traffic)")
ap.add_argument("--order", default="xy", choices=["xy", "yx"],
               help="shared: the order of the two requests (run xy on one server and yx on another: each request is cold once "
                    "and restored once)")
ap.add_argument("--task-text", default="", help="scale: the long reply task, {a}/{b} = the lane's range (default: counting; "
               "V4.1 refuses to count to 5000)")
A = ap.parse_args()
T_TOOL = time.time()
if A.port in (11435, 11440, 11441):
    sys.exit("refusing a production port")
os.makedirs(A.out, exist_ok=True)
fails = 0


def check(name, ok, evidence):
    global fails
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {evidence}", flush=True)


# ---- the server ------------------------------------------------------------------------------------------------------------
env = dict(os.environ)
if A.gate == "identity":
    env.update(ARCH[A.arch]["identity_env"])
for kv in A.env:
    k, _, v = kv.partition("=")
    env[k] = v
cmd = [A.ie, "serve", A.model, "--host", "127.0.0.1", "--port", str(A.port), "--ctx", str(A.ctx), "--parallel", str(A.parallel)]
if A.slot_ctx:
    cmd += ["--slot-ctx", str(A.slot_ctx)]
cmd += A.serve_arg
log_path = os.path.join(A.out, f"{A.gate}_server.log")
print(f"server: {' '.join(cmd)}  (env {', '.join(f'{k}={env[k]}' for k in sorted(env) if k.startswith('IE_'))}) -> {log_path}", flush=True)
srv = subprocess.Popen(cmd, stdout=open(log_path, "wb"), stderr=subprocess.STDOUT, env=env)
pid = srv.pid


def http_json(method, path, body=None, timeout=10):
    c = http.client.HTTPConnection("127.0.0.1", A.port, timeout=timeout)
    try:
        c.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
        return json.loads(c.getresponse().read() or b"{}")
    finally:
        c.close()


def health():
    try:
        return http_json("GET", "/health", timeout=5)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def stop_server():
    try:
        http_json("POST", "/admin/shutdown", {}, timeout=10)
    except Exception:  # noqa: BLE001
        pass
    try:
        rc = srv.wait(timeout=120)
    except subprocess.TimeoutExpired:
        print("server did not exit 120 s after /admin/shutdown: SIGTERM (never SIGKILL a model run)", flush=True)
        srv.send_signal(signal.SIGTERM)
        rc = srv.wait(timeout=120)
    check("server exits cleanly", rc == 0, f"exit {rc}")


t_load = time.time()
while True:
    if srv.poll() is not None:
        tail = [ln for ln in open(log_path, errors="replace").read().splitlines() if ln.strip()][-6:]
        print("server log tail:\n  " + "\n  ".join(tail), flush=True)
        sys.exit(f"server exited during load (rc {srv.returncode}); see {log_path}")
    if health().get("status") == "ok":
        break
    if time.time() - t_load > A.load_timeout:
        stop_server()
        sys.exit("server not ready in time")
    time.sleep(2)
print(f"server ready in {time.time() - t_load:.0f} s: {json.dumps(health())}", flush=True)


# ---- one streamed request (tools/ds41_parallel_serve_test.py's Req) ---------------------------------------------------------
TASKS = [
    "Explain how a two-card pipeline keeps both GPUs busy when two requests decode at once. Use three short paragraphs.",
    "Write a Python function that merges two sorted lists in linear time, with a docstring and two doctests.",
    "List ten practical checks before deploying a web service to production, one line each, most important first.",
    "Summarise the trade-offs between LRU, LFU and ARC cache eviction in a table with one row per policy.",
    "Describe what a prefix cache does in an LLM server and when it saves time, in plain language, about 150 words.",
    "Give a step-by-step plan to find why a nightly CI job fails with ModuleNotFoundError: yaml, as a numbered list.",
]
SYSTEM = ("You are a careful assistant. Answer exactly what is asked, in plain prose, without tool calls. "
          "When you are not certain of a claim, say so.")


class Req:
    def __init__(self, k, max_tokens, sys_prefix="", cancel_after=None, greedy=True, more=None, task=None, tools=None):
        self.k, self.max_tokens, self.sys_prefix, self.cancel_after, self.greedy = k, max_tokens, sys_prefix, cancel_after, greedy
        self.task, self.tools = task, tools
        self.more = more or []   # messages after the task (a follow-up turn: the assistant's reply, the next user message)
        self.content, self.reasoning, self.finish, self.usage, self.error = "", "", None, {}, None
        self.t0 = self.t_first = self.t1 = None
        self.chunks, self.cancelled = 0, False
        self.times = []   # arrival time of every content chunk (scale's windows and inter-chunk gaps)
        self.first = threading.Event()

    def run(self):
        pad = "".join(f"Record {i:04d}: station {i % 97} reported {(i * 37) % 1000} units, status {'nominal' if i % 5 else 'degraded'}.\n"
                      for i in range(A.pad))
        body = {"model": "m", "messages": [{"role": "system", "content": self.sys_prefix + SYSTEM + ("\n" + pad if pad else "")},
                                           {"role": "user", "content": self.task or TASKS[self.k % len(TASKS)]}] + self.more,
                "max_tokens": self.max_tokens, "stream": True, "stream_options": {"include_usage": True},
                "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}}
        body.update({"temperature": 0} if self.greedy else {"temperature": 0.7, "top_p": 0.95, "seed": 1000 + self.k})
        if self.tools:
            body["tools"] = self.tools
        self.t0 = time.time()
        c = http.client.HTTPConnection("127.0.0.1", A.port, timeout=1800)
        try:
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            if r.status != 200:
                self.error = f"HTTP {r.status}: {r.read()[:300]!r}"
                return
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if "error" in ev:
                    self.error = json.dumps(ev["error"])[:300]
                    break
                if ev.get("usage"):
                    self.usage = ev["usage"]
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("content") or d.get("reasoning_content"):
                        now = time.time()
                        if self.t_first is None:
                            self.t_first = now
                            self.first.set()
                        self.chunks += 1
                        self.times.append(now)
                    self.content += d.get("content") or ""
                    self.reasoning += d.get("reasoning_content") or ""
                    self.finish = ch.get("finish_reason") or self.finish
                if self.cancel_after is not None and self.chunks >= self.cancel_after:
                    self.cancelled = True
                    break
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        finally:
            c.close()
            self.t1 = time.time()
            self.first.set()   # (a failed request must not leave a waiter hanging)

    def ok(self):
        return self.error is None and self.finish in ("stop", "length") and bool(self.content.strip() or self.reasoning.strip())

    def tokens(self):
        return self.usage.get("completion_tokens", 0)

    def cached(self):
        return (self.usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)

    def text(self):
        return self.reasoning + "\u0000" + self.content

    def summary(self):
        span = self.t1 - (self.t_first or self.t0)
        return (f"req {self.k}: finish={self.finish} tokens={self.tokens()} prompt {self.usage.get('prompt_tokens')} ({self.cached()} cached) "
                f"TTFT {((self.t_first or self.t1) - self.t0):.2f} s, {self.t1 - self.t0:.1f} s, "
                f"{self.tokens() / span if span > 0.05 else 0:.1f} tok/s" + (f", cancelled after {self.chunks} chunks" if self.cancelled else "")
                + (f", ERROR {self.error}" if self.error else "") + f", text {(self.content or self.reasoning).strip()[:60]!r}")


def run_all(reqs):
    ths = [threading.Thread(target=r.run) for r in reqs]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return reqs


def diff_at(a, b):
    if a == b:
        return ""
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"at char {i} of {len(a)} / {len(b)}: {a[max(0, i - 30):i + 40]!r} vs {b[max(0, i - 30):i + 40]!r}"


def proc_status(key):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return -1


def vram_mib():
    tot, seen = {}, set()
    d = f"/proc/{pid}/fdinfo"
    for fd in os.listdir(d):
        try:
            txt = open(f"{d}/{fd}").read()
        except OSError:
            continue
        kv = dict(line.split(":", 1) for line in txt.splitlines() if ":" in line)
        pdev, cl = kv.get("drm-pdev", "").strip(), kv.get("drm-client-id", "").strip()
        vram = [v for k, v in kv.items() if k.startswith("drm-total-vram")]
        if not pdev or not vram or (pdev, cl) in seen:
            continue
        seen.add((pdev, cl))
        tot[pdev] = tot.get(pdev, 0) + sum(int(v.split()[0]) for v in vram) // 1024
    return [tot[k] for k in sorted(tot)]


def drm_fds():
    out = []
    for fd in os.listdir(f"/proc/{pid}/fd"):
        try:
            if os.readlink(f"/proc/{pid}/fd/{fd}").startswith("/dev/dri/"):
                out.append(fd)
        except OSError:
            pass
    return out


def fdinfo_snap(fds):
    """{(pdev, client, class): (cycles, total_cycles, capacity)} and {pdev: VRAM MiB} for the server's DRM clients (the
    xe drm-cycles-<class> / drm-total-cycles-<class> keys ie-glm-lanes-test's FdBusy reads)."""
    cyc, vram, seen = {}, {}, set()
    for fd in fds:
        try:
            txt = open(f"/proc/{pid}/fdinfo/{fd}").read()
        except OSError:
            continue
        kv = dict((a.strip(), b.strip()) for a, b in (line.split(":", 1) for line in txt.splitlines() if ":" in line))
        pdev, cl = kv.get("drm-pdev", ""), kv.get("drm-client-id", "")
        if not pdev or not cl or (pdev, cl) in seen:
            continue
        seen.add((pdev, cl))
        for k, v in kv.items():
            if k.startswith("drm-total-cycles-"):
                cls = k[17:]
                c = cyc.setdefault((pdev, cl, cls), [0, 0, 1])
                c[1] = int(v.split()[0])
            elif k.startswith("drm-cycles-"):
                cls = k[11:]
                c = cyc.setdefault((pdev, cl, cls), [0, 0, 1])
                c[0] = int(v.split()[0])
            elif k.startswith("drm-engine-capacity-"):
                c = cyc.setdefault((pdev, cl, k[20:]), [0, 0, 1])
                c[2] = max(1, int(v.split()[0]))
            elif k.startswith("drm-total-vram"):
                vram[pdev] = vram.get(pdev, 0) + int(v.split()[0]) // 1024
    return cyc, vram


class Util:
    """--util-hz sampler of the server's fdinfo engine cycles and VRAM (and /health, 4 Hz), for per-window busy %."""
    def __init__(self):
        self.samples, self.health, self.stop, self.fds, self.t_fds = [], [], threading.Event(), [], 0.0
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th_h = threading.Thread(target=self.run_health, daemon=True)

    def start(self):
        self.th.start()
        self.th_h.start()

    def run(self):
        dt = 1.0 / max(0.5, A.util_hz)
        while not self.stop.is_set():
            now = time.time()
            if now - self.t_fds > 5:
                try:
                    self.fds, self.t_fds = drm_fds(), now
                except OSError:
                    return
            cyc, vram = fdinfo_snap(self.fds)
            self.samples.append((now, cyc, vram))
            time.sleep(max(0.0, dt - (time.time() - now)))

    def run_health(self):
        while not self.stop.is_set():
            h = health()
            h["t"] = time.time()
            self.health.append(h)
            time.sleep(0.25)

    def close(self):
        self.stop.set()
        self.th.join()
        self.th_h.join()

    def busy(self, t0, t1):
        """Each card's busy % per engine class over [t0, t1] (the samples nearest inside), and the VRAM peak per card."""
        inside = [x for x in self.samples if t0 <= x[0] <= t1]
        if len(inside) < 2:
            return {}, {}
        a, b = inside[0][1], inside[-1][1]
        per = {}
        for key, (c1, tot1, cap) in b.items():
            if key not in a:
                continue
            c0, tot0, _ = a[key]
            if tot1 <= tot0 or c1 < c0:
                continue
            pdev, _, cls = key
            d = per.setdefault(pdev, {})
            d[cls] = d.get(cls, 0.0) + 100.0 * (c1 - c0) / (tot1 - tot0) / cap
        peak = {}
        for _, _, vr in inside:
            for pdev, mib in vr.items():
                peak[pdev] = max(peak.get(pdev, 0), mib)
        return per, peak

    def server_rate(self, t0, t1, lanes):
        """The server's /health tokens/s over [t0, t1] while exactly `lanes` decode (lanes archs only), and that span."""
        s = [h for h in self.health if t0 <= h["t"] <= t1 and h.get("decoding") == lanes and isinstance(h.get("tokens"), int)]
        if len(s) < 2 or s[-1]["t"] <= s[0]["t"]:
            return None, 0.0
        return (s[-1]["tokens"] - s[0]["tokens"]) / (s[-1]["t"] - s[0]["t"]), s[-1]["t"] - s[0]["t"]


def lane_log_lines():
    """The server log's lines about the lanes' VRAM / the expert tier they cost (every arch prints its own)."""
    try:
        txt = open(log_path, errors="replace").read().splitlines()
    except OSError:
        return []
    return [ln.strip()[:300] for ln in txt if ("lane" in ln or "[budget] card" in ln or "slot banks" in ln)
            and re.search(r"GiB|GB|MiB|slots/layer|ctx", ln)][:12]


def tool_schemas(n):
    out = []
    for i in range(n):
        out.append({"type": "function", "function": {
            "name": f"tool_{i:02d}_" + ["read_file", "search_code", "run_tests", "edit_file", "list_dir", "http_get"][i % 6],
            "description": (f"Tool {i}: " + "Performs one well-defined operation on the workspace and returns a JSON result with "
                            "a status field, a message field and a data field. Use it only when the task needs it. " * 3),
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "A workspace-relative path."},
                "query": {"type": "string", "description": "What to look for, as plain text or a regular expression."},
                "limit": {"type": "integer", "description": "The most results to return."}}, "required": ["path"]}}})
    return out


def pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q / 100.0 * (len(xs) - 1)))))
    return xs[i]


def median(xs):
    return pct(xs, 50)


# ---- gates -----------------------------------------------------------------------------------------------------------------
try:
    N, M = A.n, A.max_tokens
    if A.gate == "sse":
        mask = [(re.compile(rb'"id":"chatcmpl-[^"]*"'), b'"id":"chatcmpl-N"'), (re.compile(rb'"created":\d+'), b'"created":0')]
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        bodies = [
            {"model": "m", "messages": [{"role": "user", "content": "Explain in a few sentences why the sky is blue."}], "max_tokens": 96,
             "temperature": 0.7, "top_p": 0.95, "top_k": 0, "seed": 1234, "stream": True},
            {"model": "m", "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}], "tools": tools,
             "max_tokens": 96, "temperature": 0, "stream": True, "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}},
            {"model": "m", "messages": [{"role": "user", "content": "Is 391 a prime number? Think it through, then answer yes or no."}],
             "max_tokens": 160, "temperature": 0, "stream": True, "enable_thinking": True, "chat_template_kwargs": {"enable_thinking": True}},
        ]
        md5s = []
        for i, body in enumerate(bodies, 1):
            c = http.client.HTTPConnection("127.0.0.1", A.port, timeout=900)
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            raw = c.getresponse().read()
            c.close()
            for rx, rep in mask:
                raw = rx.sub(rep, raw)
            open(os.path.join(A.out, f"sse_{i}.sse"), "wb").write(raw)
            md5s.append(hashlib.md5(raw).hexdigest())
            print(f"sse {i}: {len(raw)} bytes md5 {md5s[-1]}", flush=True)
            check(f"sse {i} complete", b"data: [DONE]" in raw and b'"error"' not in raw, f"{raw.count(b'data: ')} events")
        if A.save:
            json.dump({"md5": md5s}, open(A.save, "w"))
        if A.ref:
            ref = json.load(open(A.ref))["md5"]
            check("SSE bytes == reference", ref == md5s, f"{sum(a == b for a, b in zip(ref, md5s))}/3 identical to {A.ref}")
        # after the captures (not part of them): one greedy reply's decode rate, client-side (first delta to the end)
        r = Req(0, M, "Rate. ")
        r.run()
        span = r.t1 - (r.t_first or r.t0)
        print(f"decode rate: {r.tokens()} tokens in {span:.2f} s = {r.tokens() / max(1e-9, span):.2f} tok/s (client-side); {r.summary()}", flush=True)
        check("rate request complete", r.ok(), r.finish)

    elif A.gate == "identity":
        sp = lambda k: f"Conversation {k}. "   # noqa: E731  (a different first token: nothing shared across conversations)
        warm = [Req(k, 8, sp(k)) for k in range(N)]
        for r in warm:
            r.run()
            print("warm", r.summary(), flush=True)
        check("warm complete", all(r.ok() for r in warm), f"finishes {[r.finish for r in warm]}")
        solo = []
        for k in range(N):
            r = Req(k, M, sp(k))
            r.run()
            solo.append(r)
            print("solo", r.summary(), flush=True)
        check("solo complete", all(r.ok() for r in solo), f"finishes {[r.finish for r in solo]}")
        batch = run_all([Req(k, M, sp(k)) for k in range(N)])
        for r in batch:
            print("batch", r.summary(), flush=True)
        same = [a.text() == b.text() for a, b in zip(solo, batch)]
        check("batch == solo", all(same) and all(r.ok() for r in batch),
              f"{sum(same)}/{N} identical; tokens solo {[r.tokens() for r in solo]} batch {[r.tokens() for r in batch]}; "
              + "; ".join(f"req {i} {diff_at(a.text(), b.text())}" for i, (a, b, s) in enumerate(zip(solo, batch, same)) if not s))
        K = 1 if N > 1 else 0
        canc = run_all([Req(k, M, sp(k), cancel_after=(6 if k == K else None)) for k in range(N)])
        for r in canc:
            print("cancel", r.summary(), flush=True)
        others = [a.text() == b.text() for k, (a, b) in enumerate(zip(solo, canc)) if k != K]
        check("cancel survivors == solo", canc[K].cancelled and all(others) and all(r.ok() for k, r in enumerate(canc) if k != K),
              f"request {K} closed after {canc[K].chunks} chunks; {sum(others)}/{N - 1} others identical; "
              + "; ".join(f"req {k} {diff_at(a.text(), b.text())}" for k, (a, b) in enumerate(zip(solo, canc)) if k != K and a.text() != b.text()))
        after = Req(K, 32, sp(K))
        after.run()
        check("server answers after the cancel", after.ok(), after.summary())
        # cold: every conversation NEW to this server, all at once -- prefills in a turn and through the lane pipe beside
        # decoding lanes; compared with a --parallel 1 server's cold arm (--save there, --ref here)
        cold = run_all([Req(k, M, f"Cold {k}. ") for k in range(N)])
        for r in cold:
            print("cold", r.summary(), flush=True)
        check("cold complete", all(r.ok() for r in cold), f"finishes {[r.finish for r in cold]}")
        if A.save:
            json.dump({"solo": [r.text() for r in solo], "cold": [r.text() for r in cold]}, open(A.save, "w"))
        if A.ref and "cold" in json.load(open(A.ref)):
            refc = json.load(open(A.ref))["cold"]
            eqc = [a == b.text() for a, b in zip(refc, cold)]
            check("cold (concurrent prefills) == reference cold", len(refc) == N and all(eqc),
                  f"{sum(eqc)}/{N} identical; " + "; ".join(f"req {i} {diff_at(refc[i], cold[i].text())}" for i in range(min(len(refc), N)) if not eqc[i]))
        if A.ref:
            ref = json.load(open(A.ref))["solo"]
            eqs = [a == b.text() for a, b in zip(ref, solo)]
            check("solo == reference solo", len(ref) == N and all(eqs),
                  f"{sum(eqs)}/{N} identical to {A.ref}; " + "; ".join(f"req {i} {diff_at(ref[i], solo[i].text())}" for i in range(min(len(ref), N)) if not eqs[i]))
        print(f"health at the end: {json.dumps(health())}; threads {proc_status('Threads')}", flush=True)

    elif A.gate == "followup":
        sp = lambda k: f"Follow {k}. "   # noqa: E731
        t1 = []
        for k in range(N):
            r = Req(k, M, sp(k))
            r.run()
            t1.append(r)
            print("turn 1", r.summary(), flush=True)
        check("turn 1 complete", all(r.ok() for r in t1), f"finishes {[r.finish for r in t1]}")
        nxt = "Now give the single most important point of your answer in one sentence."
        t2 = run_all([Req(k, M, sp(k), more=[{"role": "assistant", "content": t1[k].content}, {"role": "user", "content": nxt}])
                      for k in range(N)])
        for r in t2:
            print("turn 2", r.summary(), flush=True)
        check("turn 2 complete and restored from the prompt cache", all(r.ok() and r.cached() > 0 for r in t2),
              f"finishes {[r.finish for r in t2]}, cached {[r.cached() for r in t2]}, prompts {[r.usage.get('prompt_tokens') for r in t2]}")
        if A.save:
            json.dump({"t1": [r.text() for r in t1], "t2": [r.text() for r in t2], "cached": [r.cached() for r in t2]}, open(A.save, "w"))
        if A.ref:
            ref = json.load(open(A.ref))
            e1 = [a == b.text() for a, b in zip(ref["t1"], t1)]
            e2 = [a == b.text() for a, b in zip(ref["t2"], t2)]
            check("turn 1 == reference", len(ref["t1"]) == N and all(e1), f"{sum(e1)}/{N}; " + "; ".join(
                f"conv {i} {diff_at(ref['t1'][i], t1[i].text())}" for i in range(min(N, len(ref['t1']))) if not e1[i]))
            check("turn 2 == reference", len(ref["t2"]) == N and all(e2),
                  f"{sum(e2)}/{N}; cached ref {ref['cached']} here {[r.cached() for r in t2]}; " + "; ".join(
                      f"conv {i} {diff_at(ref['t2'][i], t2[i].text())}" for i in range(min(N, len(ref['t2']))) if not e2[i]))
        print(f"health at the end: {json.dumps(health())}", flush=True)

    elif A.gate == "shared":
        tools = tool_schemas(A.tools) if A.tools else None
        ux = "List three practical uses of a hash table, one line each, then stop."
        uy = "Explain in two short paragraphs why a B-tree suits disk storage."
        mk = {"x": lambda: Req(0, M, "", task=ux, tools=tools), "y": lambda: Req(1, M, "", task=uy, tools=tools)}
        got = {}
        for i, which in enumerate(A.order):
            r = mk[which]()
            r.run()
            got[which] = r
            print(f"{which} ({'first' if i == 0 else 'second'})", r.summary(), flush=True)
        second = got[A.order[1]]
        check("both complete", all(r.ok() for r in got.values()), f"finishes {[got[w].finish for w in A.order]}")
        check("the second request restored the shared prefix", 0 < second.cached() < second.usage.get("prompt_tokens", 0),
              f"cached {[got[w].cached() for w in A.order]} of prompts {[got[w].usage.get('prompt_tokens') for w in A.order]}")
        res = {w: {"text": got[w].text(), "cached": got[w].cached(), "prompt": got[w].usage.get("prompt_tokens")} for w in "xy"}
        if A.save:
            json.dump(res, open(A.save, "w"))
        if A.ref:
            ref = json.load(open(A.ref))
            for w in "xy":
                check(f"{w} == reference ({'restored' if got[w] is second else 'cold'} here, cached {res[w]['cached']}; "
                      f"reference cached {ref[w]['cached']})", ref[w]["text"] == res[w]["text"], diff_at(ref[w]["text"], res[w]["text"]))
        print(f"health at the end: {json.dumps(health())}", flush=True)

    elif A.gate == "aba":
        sp = lambda k: f"Conversation {k}. "   # noqa: E731
        samples, stop, phase, t_tr = [], threading.Event(), ["A1"], time.time()

        def sampler():
            while not stop.is_set():
                h = health()
                h["t"], h["arm"] = round(time.time() - t_tr, 2), phase[0]
                samples.append(h)
                time.sleep(0.5)

        def server_rate(arm, lanes):
            s = [h for h in samples if h.get("arm") == arm and h.get("decoding") == lanes and isinstance(h.get("tokens"), int)]
            if len(s) < 2 or s[-1]["t"] <= s[0]["t"]:
                return 0.0, 0.0
            return (s[-1]["tokens"] - s[0]["tokens"]) / (s[-1]["t"] - s[0]["t"]), s[-1]["t"] - s[0]["t"]

        for k in range(N):   # every prompt prefilled alone into its own lane first: the arms then mostly decode
            w = Req(k, 4, sp(k))
            w.run()
            print("warm", w.summary(), flush=True)
        st = threading.Thread(target=sampler)
        st.start()
        a1 = Req(0, M, sp(0))
        a1.run()
        print("solo A1", a1.summary(), flush=True)
        phase[0] = "B"
        batch = run_all([Req(k, M, sp(k)) for k in range(N)])
        for r in batch:
            print("concurrent", r.summary(), flush=True)
        phase[0] = "A2"
        a2 = Req(0, M, sp(0))
        a2.run()
        print("solo A2", a2.summary(), flush=True)
        time.sleep(0.6)
        stop.set()
        st.join()
        (s1, w1), (sb, wb), (s2, w2) = server_rate("A1", 1), server_rate("B", N), server_rate("A2", 1)
        ratio = sb / max(1e-9, (s1 + s2) / 2)
        print(f"A-B-A (server /health tokens while exactly 1 / {N} / 1 lanes decode): solo A1 {s1:.2f} tok/s over {w1:.1f} s, {N} concurrent "
              f"{sb:.2f} tok/s over {wb:.1f} s (x{ratio:.2f} the solo mean; per lane {sb / N:.2f}), solo A2 {s2:.2f} tok/s over {w2:.1f} s", flush=True)
        check("aba complete", a1.ok() and a2.ok() and all(r.ok() for r in batch), f"finishes {a1.finish} {[r.finish for r in batch]} {a2.finish}")
        check("aba windows measured", w1 > 2 and wb > 2 and w2 > 2, f"windows {w1:.1f} / {wb:.1f} / {w2:.1f} s (raise --max-tokens if short)")
        if A.min_ratio:
            check(f"aba ratio >= {A.min_ratio}", ratio >= A.min_ratio, f"x{ratio:.3f}")

    elif A.gate == "memory":
        R = A.requests
        rss0, vr0, th0 = proc_status("RssAnon") // 1024, vram_mib(), proc_status("Threads")
        print(f"before: RssAnon {rss0} MiB, VRAM {vr0} MiB, threads {th0}", flush=True)
        rss1 = vr1 = th1 = None
        done = wave = 0
        while done < R:
            n = min(N, R - done)
            reqs = run_all([Req(done + i, M, f"Run {done + i}. ") for i in range(n)])
            bad = [r.summary() for r in reqs if not r.ok()]
            done += n
            wave += 1
            if wave == 1:
                rss1, vr1, th1 = proc_status("RssAnon") // 1024, vram_mib(), proc_status("Threads")
            print(f"wave {wave}: {n} requests, finishes {[r.finish for r in reqs]}, RssAnon {proc_status('RssAnon') // 1024} MiB, "
                  f"VRAM {vram_mib()} MiB, threads {proc_status('Threads')}" + (f" BAD {bad}" if bad else ""), flush=True)
            check(f"wave {wave} complete", not bad, f"{n - len(bad)}/{n}")
        rss2, vr2, th2 = proc_status("RssAnon") // 1024, vram_mib(), proc_status("Threads")
        check("memory flat", rss2 - rss1 < 64 and len(vr1) == len(vr2) and all(abs(a - b) < 64 for a, b in zip(vr1, vr2)) and th2 <= th1,
              f"RssAnon {rss0} -> {rss1} (wave 1) -> {rss2} MiB; VRAM {vr0} -> {vr1} -> {vr2} MiB; threads {th0} -> {th1} -> {th2}")

    elif A.gate == "load":
        lines = lane_log_lines()
        print("lane VRAM lines:\n  " + "\n  ".join(lines), flush=True)
        print(f"health: {json.dumps(health())}; server VRAM per card {vram_mib()} MiB", flush=True)
        json.dump({"args": vars(A), "lane_lines": lines, "vram_mib": vram_mib(), "health": health()},
                  open(os.path.join(A.out, "load.json"), "w"), indent=1)
        check("loaded", True, f"--parallel {A.parallel} --slot-ctx {A.slot_ctx} --ctx {A.ctx}")

    elif A.gate == "scale":
        ns = [int(x) for x in A.ns.split(",") if x.strip()]
        if not ns or min(ns) < 1 or max(ns) > A.parallel:
            raise SystemExit(f"--ns {A.ns}: each must be 1..--parallel ({A.parallel})")
        bs = [n for n in ns if n > 1]
        maxn = max(ns)
        sp = (lambda k: "") if A.shared_prefix else (lambda k: f"Conversation {k}. ")   # noqa: E731
        tools = tool_schemas(A.tools) if A.tools else None
        arm_no = [0]   # agent traffic: every request's user turn is new (only the shared system prompt is cached)
        # a long greedy reply (the server has no ignore_eos on every arch): counting keeps a model going for thousands of tokens
        long_task = lambda k: ((f"Request {arm_no[0]}-{k}. " if A.shared_prefix else "") +   # noqa: E731
                               (A.task_text.format(a=1000 * k + 1, b=1000 * k + 5000) if A.task_text else
                                f"Count upward by ones from {1000 * k + 1} to {1000 * k + 5000}, writing every number in digits, "
                                "one per line, with nothing else. Do not skip, summarise or stop early."))
        lane_cap = A.slot_ctx or min(A.ctx, 32768)
        xpu = None
        if A.xpu_smi:
            try:
                # (the installed xpu-smi takes one --device per dump; on the B70 under xe most UTILIZATION / bandwidth fields
                # print N/A at idle -- 2026-09-27)
                xpu = [subprocess.Popen(["xpu-smi", "dump", "--device", str(dv), "--metrics", "UTILIZATION,MEMORY", "--interval", "1"],
                                        stdout=open(os.path.join(A.out, f"xpu_smi_dump_{dv}.txt"), "wb"), stderr=subprocess.STDOUT)
                       for dv in (0, 1)]
            except OSError as e:
                print(f"xpu-smi not started: {e}", flush=True)
        util = Util()
        util.start()
        res = {"args": vars(A), "cold_solo_ttft": [], "arms": [], "cold": None, "complete": False}
        warm = []
        for k in range(maxn):   # every conversation prefilled alone: its cold TTFT; the arms then decode from the cache
            w = Req(k, 4, sp(k), task=long_task(k), tools=tools)
            w.run()
            warm.append(w)
            print("warm", w.summary(), flush=True)
        check("warm complete", all(r.ok() for r in warm), f"finishes {[r.finish for r in warm]}")
        res["cold_solo_ttft"] = [round((r.t_first or r.t1) - r.t0, 3) for r in warm]
        prompt_tok = max(r.usage.get("prompt_tokens", 0) for r in warm)
        room = max(64, lane_cap - prompt_tok - 16)   # a reply longer than the lane's room is cut there by the server

        def window_of(reqs):
            """The steady window: every request past its first chunk and none finished; the chunks inside it."""
            if not all(r.t_first for r in reqs):
                return 0.0, 0.0, 0
            w0, w1 = max(r.t_first for r in reqs), min(r.times[-1] if r.times else r.t1 for r in reqs)
            if w1 <= w0:
                return w0, w1, 0
            return w0, w1, sum(sum(1 for t in r.times if w0 < t <= w1) for r in reqs)

        rate_lane = {}   # learned per-lane chunk rate per N (sizes the next arm)

        def run_arm(kind, n, rep_i):
            est = rate_lane.get(n) or ((rate_lane[1] * min(1.0, 2.0 / n)) if 1 in rate_lane else 0)
            toks = A.max_tokens if not est else int(est * (A.min_window + 8)) + 8
            for attempt in (0, 1):
                toks = min(max(toks, 32), room)
                if time.time() - T_TOOL + (toks / est if est else A.min_window) + 10 > A.deadline:
                    return None
                arm_no[0] += 1
                reqs = [Req(k, toks, sp(k), task=long_task(k), tools=tools) for k in range(n)]
                if A.arrival and n > 1:
                    ths = [threading.Thread(target=r.run) for r in reqs[:-1]]
                    for t in ths:
                        t.start()
                    for r in reqs[:-1]:
                        r.first.wait(timeout=600)
                    tl = threading.Thread(target=reqs[-1].run)
                    tl.start()
                    for t in ths + [tl]:
                        t.join()
                else:
                    run_all(reqs)
                w0, w1, nch = window_of(reqs)
                win = max(0.0, w1 - w0)
                agg = nch / win if win > 0 else 0.0
                if agg > 0:
                    rate_lane[n] = agg / n
                rl = rate_lane.get(n) or est
                if win >= A.min_window or attempt == 1 or not rl:
                    break
                # the window opens only when the LAST lane's first token is out: cover the first-token spread too
                firsts = [r.t_first for r in reqs if r.t_first]
                spread = (max(firsts) - min(firsts)) if firsts else 0.0
                toks = int(rl * (A.min_window + 4 + spread) * 1.15) + 8
                print(f"  {kind} N={n} window {win:.1f} s < {A.min_window}: re-run with {toks} tokens", flush=True)
            srv_rate, srv_span = util.server_rate(w0, w1, n) if win > 0 else (None, 0.0)
            busy, peak = util.busy(w0, w1) if win > 0 else ({}, {})
            lanes = []
            for r in reqs:
                gaps = [1000.0 * (b - a) for a, b in zip(r.times, r.times[1:]) if w0 < b <= w1]
                span = (r.times[-1] - r.t_first) if len(r.times) > 1 else 0.0
                lanes.append({"k": r.k, "ok": r.ok(), "finish": r.finish, "tokens": r.tokens(), "cached": r.cached(),
                              "ttft": round((r.t_first or r.t1) - r.t0, 3), "tok_s": round((len(r.times) - 1) / span, 2) if span > 0 else 0.0,
                              "gap_p50_ms": round(pct(gaps, 50), 2), "gap_p95_ms": round(pct(gaps, 95), 2)})
            pooled = [1000.0 * (b - a) for r in reqs for a, b in zip(r.times, r.times[1:]) if w0 < b <= w1]
            ttfts = sorted(x["ttft"] for x in lanes)
            arm = {"kind": kind, "n": n, "rep": rep_i, "max_tokens": toks, "prompt_tokens": max(x.usage.get("prompt_tokens", 0) for x in reqs),
                   "cached_tokens": [x["cached"] for x in lanes], "ttft_sorted": ttfts, "window_s": round(win, 2), "agg_tok_s": round(agg, 2),
                   "server_tok_s": round(srv_rate, 2) if srv_rate else None, "server_span_s": round(srv_span, 2),
                   "gap_p50_ms": round(pct(pooled, 50), 2), "gap_p95_ms": round(pct(pooled, 95), 2),
                   "arrival_ttft": lanes[-1]["ttft"] if (A.arrival and n > 1) else None,
                   "busy": busy, "vram_peak_mib": peak, "lanes": lanes, "ok": all(x["ok"] for x in lanes)}
            res["arms"].append(arm)
            bz = "; ".join(f"card {i} " + " ".join(f"{c} {v:.0f}%" for c, v in sorted(busy[pd].items()) if v > 0.5)
                           for i, pd in enumerate(sorted(busy)))
            print(f"{kind} N={n} rep {rep_i}: window {win:.1f} s, {agg:.2f} tok/s aggregate ({agg / n:.2f}/lane)"
                  + (f", server {srv_rate:.2f} over {srv_span:.1f} s" if srv_rate else "")
                  + f", gap p50 {arm['gap_p50_ms']:.1f} / p95 {arm['gap_p95_ms']:.1f} ms, TTFT {[x['ttft'] for x in lanes]}"
                  + (f" (arrival {arm['arrival_ttft']:.2f} s)" if arm["arrival_ttft"] is not None else "") + f"; busy {bz}", flush=True)
            return arm

        complete = True
        for rep_i in range(1, A.repeats + 1):
            prev_a = run_arm("A", 1, rep_i)
            if prev_a is None:
                complete = False
                break
            for n in bs:
                b = run_arm("B", n, rep_i)
                a2 = run_arm("A", 1, rep_i) if b is not None else None
                if b is None or a2 is None:
                    complete = False
                    break
                solo = (prev_a["agg_tok_s"] + a2["agg_tok_s"]) / 2
                b["ratio"] = round(b["agg_tok_s"] / solo, 3) if solo > 0 else 0.0
                b["solo_mean"] = round(solo, 2)
                prev_a = a2
            if not complete:
                break
        if complete and A.cold:
            if A.shared_prefix:   # B15: max(ns) NEW conversations at once that share one NEW system prompt (+ tools)
                arm_no[0] += 1
                cold = run_all([Req(k, 32, "Cold batch. ", task=long_task(k), tools=tools) for k in range(maxn)])
            else:
                cold = run_all([Req(k, 32, f"Cold {k}. ") for k in range(maxn)])
            res["cold"] = {"n": maxn, "ttft": [round((r.t_first or r.t1) - r.t0, 3) for r in cold], "ok": all(r.ok() for r in cold),
                           "cached": [r.cached() for r in cold]}
            print(f"cold x{maxn}: TTFT {res['cold']['ttft']}, cached {res['cold']['cached']}; health {json.dumps(health())}", flush=True)
            check("cold complete", res["cold"]["ok"], f"finishes {[r.finish for r in cold]}")
        util.close()
        for xp in (xpu or []):
            xp.send_signal(signal.SIGINT)
            try:
                xp.wait(timeout=10)
            except subprocess.TimeoutExpired:
                xp.kill()
        res["complete"] = complete
        res["slot_ctx"], res["lane_lines"], res["prompt_tokens"] = A.slot_ctx, lane_log_lines(), prompt_tok
        print(f"--slot-ctx {A.slot_ctx} (0 = min(ctx, 32768)); prompt {prompt_tok} tokens; lane VRAM lines:\n  " + "\n  ".join(res["lane_lines"]), flush=True)
        # the summary: per N the median and spread over the repeats
        print("\nsummary (median [min..max] over repeats):", flush=True)
        summ = {}
        for n in ns:
            arms = [a for a in res["arms"] if a["n"] == n and (n > 1 or a["kind"] == "A")]
            if not arms:
                continue
            agg = [a["agg_tok_s"] for a in arms]
            rat = [a["ratio"] for a in arms if "ratio" in a]
            p50 = [a["gap_p50_ms"] for a in arms]
            p95 = [a["gap_p95_ms"] for a in arms]
            summ[n] = {"agg_tok_s": median(agg), "agg_min": min(agg), "agg_max": max(agg), "ratio": median(rat) if rat else None,
                       "gap_p50_ms": median(p50), "gap_p95_ms": median(p95), "windows_s": [a["window_s"] for a in arms]}
            print(f"  N={n:2d}: {median(agg):8.2f} tok/s [{min(agg):.2f}..{max(agg):.2f}]"
                  + (f", x{median(rat):.2f} the solo mean [{min(rat):.2f}..{max(rat):.2f}]" if rat else "")
                  + f", per lane {median(agg) / n:.2f}, gap p50 {median(p50):.1f} / p95 {median(p95):.1f} ms, windows {summ[n]['windows_s']}",
                  flush=True)
        res["summary"] = summ
        json.dump(res, open(os.path.join(A.out, "scale.json"), "w"), indent=1)
        check("scale complete (every planned arm ran before --deadline)", complete, f"{len(res['arms'])} arms")
        check("scale arms all finished cleanly", all(a["ok"] for a in res["arms"]), f"{sum(not a['ok'] for a in res['arms'])} bad arm(s)")
        short = [(a["kind"], a["n"], a["rep"], a["window_s"]) for a in res["arms"] if a["window_s"] < A.min_window]
        check(f"every window >= {A.min_window} s", not short, f"short: {short}" if short else "all")
finally:
    stop_server()

print("PASS" if not fails else f"FAIL ({fails})", flush=True)
sys.exit(1 if fails else 0)
