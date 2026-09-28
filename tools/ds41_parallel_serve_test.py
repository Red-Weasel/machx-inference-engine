#!/usr/bin/env python3
"""P4 B6b (docs/deepseek41/P4_B6B_SERVE.md): several requests at once against a live `ie serve` of DeepSeek-V4.1-Flash at
--parallel N (MiMo B4's tools/mimo26/parallel_serve_test.py, adapted).

  ds41_parallel_serve_test.py <port> <server pid> <mode> [options]

Requests are Dream-shaped unless a mode says otherwise: a ~2.5k-token agent system prompt, 8 tools and a task, streamed.
The server pid is only read through procfs (RssAnon, threads, the process's own VRAM from /proc/<pid>/fdinfo): nothing here
talks to the GPU.

  concurrent  --n N --max-tokens M [--greedy] [--thinking]
      N requests at once (tools on). Every one must complete (stop / length / tool_calls, with text, reasoning or a tool
      call); with --thinking at least one reply must carry reasoning_content, and the tool calls must parse (the engine's
      native V4.1 parser, per lane). Prints TTFTs, tokens, the aggregate rate and /health (lanes_active, step_ms, ...).
      Criterion (b).
  identity    --n N --max-tokens M [--cancel-idx K] [--cancel-after C] [--save FILE] [--ref FILE]
      Greedy, thinking off, N conversations (a different first system token each; two tasks copy a paragraph, so prompt-
      lookup drafts and verifies). Warm: every conversation once, ONE AT A TIME, 8 tokens (each prefilled alone into its own
      lane). Solo: each again, one at a time. Batch: all at once. Cancel: all at once with request K closed after C content
      chunks. Every later arm is served from its lane's live state (cached = prompt - 1: the think-tag checkpoint), so the arms
      differ in the decode interleaving only. Batch == solo (f) and the cancel survivors == solo (c). Run the server with
      IE_DS41_CPU_MISS=0 (the CPU leg's split follows the cache history) and with IE_DS41_LOOKUP=0 for the lookup-off arm.
      --save writes the texts as JSON; --ref compares with a saved file (e.g. the same client against --parallel 1).
  evict       --n N --max-tokens M --ref FILE
      Right after `identity` in the same server (FILE = its --save): with every lane holding a conversation, a NEW one takes
      the least recently used lane (its conversation kept in a host slot first); then every conversation again, one at a
      time: each served from cache (cached = prompt - 1: its lane, or its host slot swapped in on another lane) and replying
      exactly as its solo run did. The prefix cache's eviction and a slot restored on another lane, CPU leg off.
  interrupt   [--after S] [--max-tokens M] [--save FILE] [--ref FILE]
      A ~7k-token prompt (4 prefill chunks) alone: its prefill runs in its serial turn with the cards pipelined; S seconds in
      (default 4) a short request arrives, A's prefill stops at the next chunk end and finishes through the lane pipe. Both
      must complete; A's text against a --parallel 1 server's (--save there, --ref here; CPU leg off).
  room        [--max-tokens M] [--target P]
      The reply-room rule (MiMo B4 gate case, sized to fit a job): with the server at --ctx 32768 --slot-ctx 4096, a prompt
      calibrated to ~P tokens (default 3,750: a 4,096 lane would leave ~346) asks for a long count with max_tokens M (700): the
      reply must reach M ("length"), i.e. the request took lane 0 and was not cut at a small lane's room. Then a short prompt
      with a small budget is served (on a small lane: the server's lane line shows it).
  ttft        --n N --max-tokens M
      TTFT of prompt 0 alone (a fresh server: a full prefill); then N-1 others decode (M tokens) and prompt 0 with a system
      prompt whose first token differs (nothing serves it) arrives: its TTFT while they decode, and the others' stall.
      Criterion (d): busy TTFT <= solo TTFT + one prefill chunk (the solo time of 2,048 prompt rows, at least 5 s).
  aba         --n N --max-tokens M [--warm] [--tasks I,J,..]
      Greedy, thinking off, no tools, prompt-lookup as served. Solo -> N concurrent -> solo. The primary figures are the
      SERVER's: /health every 0.5 s, each arm's rate = committed ids ("tokens") over the samples in which exactly 1 / N / 1
      lanes decode. --warm: every prompt first prefilled alone into its own lane (the arms then only decode).
  memory      --requests R --n N [--max-tokens M]
      R requests in waves of N: RssAnon, the server's own VRAM (fdinfo) and its thread count before, after wave 1 and at the
      end. Criterion (e): growth after wave 1 under 64 MiB RssAnon, under 64 MiB VRAM per card, and no thread growth. Run
      the server with IE_DS41_PROMPT_CACHE_GIB=0 (a kept conversation in a host slot is the prefix cache's design, not a leak).
  vision      --max-tokens M [--image FILE]
      A long text reply decodes on one lane while an image request (the Phase 57 fixture: "MACHX 4721" and three shapes)
      arrives on another: the image request takes a serial turn and must answer; then a follow-up in the image conversation
      must be served from cache (cached_tokens beyond the image). The text reply must complete.
  sse         --out PREFIX
      Criterion (a)'s byte method (docs/mimo26/IE_VITALS.md, MiMo B4): three fixed streaming requests as the FIRST requests of
      a fresh server, raw SSE bytes saved with "id" and "created" masked, one md5 each. 1: T 0.7 seed 1234 thinking on;
      2: greedy thinking off with a get_weather tool (a tool call); 3: greedy thinking on (reasoning_content). Run against HEAD
      and the branch at --parallel 1; the md5s must match (with IE_DS41_CPU_MISS=0: the CPU leg is not reproducible).
Standard library only. Prints ok / FAIL lines with their evidence; exit 0 = PASS.
"""
import base64
import hashlib
import http.client
import json
import os
import re
import sys
import threading
import time

port, pid, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


N = int(arg("--n", "2"))
MAX_TOKENS = int(arg("--max-tokens", "128"))
fails = 0


def check(name, ok, evidence):
    global fails
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {evidence}", flush=True)


# ---- Dream-shaped prompts (MiMo B4's) -------------------------------------------------------------------------------------
SYSTEM = """You are Dream, an autonomous software agent working inside a sandboxed workspace on the user's machine.

## Operating rules
1. You work in the workspace directory only. Files outside it are read-only and must never be modified, moved or deleted.
2. Before doing any work, restate the task in one sentence and define what DONE means as a checklist of 3-8 verifiable items.
3. Prefer small, reversible steps. After every change, run the narrowest check that proves it (a unit test, a type check, a
   dry run). Never claim a command ran or a test passed unless you saw its output.
4. When a step fails twice in the same way, stop and change approach; do not retry the same command a third time.
5. Never invent file paths, APIs, environment variables or command names. Inspect the repository first.
6. Say "Pineapple" in a sentence whenever you are not certain of a claim, so the user can spot the uncertainty.

## Tools
You have tools for the shell, files, search, the web and delegation. Call a tool when you need information or an effect; do
not narrate a tool call as text. Read a file before editing it. Use search_files to find where something lives before you
open files one by one. run_bash has public internet access through a filtering proxy. delegate_task hands a well-scoped
sub-task to a sub-agent with its own context: give it the goal, the files it may touch, the acceptance criteria and the
format of the report you expect back. Sub-agents cannot see this conversation.

## Memory
The project's memory lives in memory/ inside the workspace. memory/semantic/ holds durable facts about the project (one topic
per file, dated headers); memory/episodic/ holds session notes. When you learn something the next session must know, write
it down before the task ends. Never store secrets, tokens or personal data in memory.

## Output format
Reply in plain prose with short paragraphs. Use markdown bullets for checklists and tables for comparisons. Quote commands
and paths in backticks. When the task is done, end with a section "Result" that lists what changed, how it was verified and
what remains open. Keep the final report under 300 words unless the user asked for detail.

## Safety
Destructive actions (deleting more than one file, rewriting git history, force pushes, dropping database tables, sending
messages or emails, spending money) require the user's explicit approval in this conversation. Ask once, clearly, with the
exact command you intend to run. If the user is away, leave the action undone and say so in the Result.

## Style
Be direct. No filler, no praise, no apologies. When two interpretations of a request exist, name both and pick the one that
is cheaper to undo, saying which you picked. If a simpler approach exists than the one requested, say so before you start.

## Reference: the repository layout you usually meet
- src/ the application code; tests/ its tests (pytest); scripts/ maintenance scripts; docs/ design notes and runbooks.
- pyproject.toml or package.json declare the toolchain; read them before assuming a command such as `pytest` or `npm test`.
- A CLAUDE.md or AGENTS.md at the root carries project rules that override these defaults where they conflict.
- CI configuration lives in .github/workflows/; a failing check there is the first thing to reproduce locally.

## Reference: how to report a bug you found but did not fix
State the file and line, the observed behaviour, the expected behaviour, the smallest input that reproduces it, and your
confidence. Put it under "Open" in the Result. Do not fix it unless it blocks the task.

## Reference: working with the user's git repository
Commit only when asked. Stage the files you changed by name; never `git add -A`. Write the commit message as a short
imperative line, a blank line and a paragraph of why. Do not add co-author trailers. Do not push.
"""
TOOLS = [
    {"type": "function", "function": {"name": "run_bash", "description": "Run a shell command in the workspace and return stdout, stderr and the exit code.",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer", "description": "Seconds before the command is killed (default 120)."}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file from the workspace, optionally a line range.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a text file in the workspace.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List a directory's entries with sizes and types.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "search_files", "description": "Search file contents with a regular expression; returns path, line number and the matching line.",
                                      "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "glob": {"type": "string", "description": "Restrict to files matching this glob."}, "max_results": {"type": "integer"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "web_fetch", "description": "Fetch a URL through the filtering proxy and return its text content.",
                                      "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "max_chars": {"type": "integer"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "delegate_task", "description": "Hand a scoped sub-task to a sub-agent and return its report.",
                                      "parameters": {"type": "object", "properties": {"goal": {"type": "string"}, "files": {"type": "array", "items": {"type": "string"}}, "acceptance": {"type": "string"}, "report_format": {"type": "string"}}, "required": ["goal", "acceptance"]}}},
    {"type": "function", "function": {"name": "finish", "description": "End the task with a final report for the user.",
                                      "parameters": {"type": "object", "properties": {"result": {"type": "string"}, "open_items": {"type": "array", "items": {"type": "string"}}}, "required": ["result"]}}},
]
PARA = ("The engine keeps the two cards busy by splitting the forward into pipeline stages. Card zero runs the first twenty layers "
        "while card one runs the last twenty on the previous chunk, and the hidden stream crosses the PCIe link once per chunk. "
        "The prefix cache stores checkpoints at chunk boundaries so a returning conversation resumes where it left off.")
TASKS = [
    "The test suite in tests/ has one failing test, tests/test_prune.py::test_status_with_colons. Investigate and fix the bug in src/prune.py without changing the tests. Start by restating the task and your DONE checklist.",
    "Repeat the following paragraph exactly, word for word, then add one sentence about why it helps.\n\n" + PARA,
    "Write a design note docs/cache_eviction.md that compares LRU, LFU and ARC for our 16 GiB host slot cache, with a recommendation. Start by restating the task and your DONE checklist.",
    "Copy this paragraph twice, the second time with every sentence numbered, and nothing else.\n\n" + PARA,
    "The nightly CI job in .github/workflows/nightly.yml has been red for three days with 'ModuleNotFoundError: yaml'. Find the cause and propose the smallest fix. Start by restating the task and your DONE checklist.",
    "Add a --json flag to scripts/report.py so its summary prints as one JSON object instead of a table; keep the default output unchanged. Start by restating the task and your DONE checklist.",
    "Review src/server/admission.py for race conditions between acquire() and shutdown(); report each with a reproduction. Start by restating the task and your DONE checklist.",
    "Profile scripts/import_sessions.py on a 40 MB log: it takes 90 s. Find the hot spot and propose a fix with an expected speed-up. Start by restating the task and your DONE checklist.",
]


def messages(k, variant="", sys_prefix=""):
    # sys_prefix: a different first token makes the prompt share NO prefix with any lane or slot (a real prefill)
    return [{"role": "system", "content": sys_prefix + SYSTEM}, {"role": "user", "content": TASKS[k % len(TASKS)] + variant}]


# ---- one streamed request ---------------------------------------------------------------------------------------------------
class Req:
    def __init__(self, k, max_tokens, greedy=True, thinking=False, cancel_after=None, variant="", seed=1234, tools=True, sys_prefix="", msgs=None):
        self.k, self.max_tokens, self.greedy, self.thinking, self.cancel_after, self.variant = k, max_tokens, greedy, thinking, cancel_after, variant
        self.seed, self.tools, self.sys_prefix, self.msgs = seed, tools, sys_prefix, msgs
        self.content, self.reasoning, self.tool_calls, self.finish, self.usage, self.error = "", "", [], None, {}, None
        self.t0 = self.t_first = self.t1 = None
        self.chunks = 0
        self.times = []   # arrival time of every content / reasoning delta (~ one token each)
        self.cancelled = False

    def run(self):
        body = {"model": "m", "messages": self.msgs or messages(self.k, self.variant, self.sys_prefix), "max_tokens": self.max_tokens, "stream": True,
                "enable_thinking": self.thinking, "chat_template_kwargs": {"enable_thinking": self.thinking}}
        if self.tools:
            body["tools"] = TOOLS
        if self.greedy:
            body["temperature"] = 0
        else:
            body.update({"temperature": 0.7, "top_p": 0.95, "top_k": 0, "seed": self.seed})
        self.t0 = time.time()
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
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
                        if self.t_first is None:
                            self.t_first = time.time()
                        self.chunks += 1
                        self.times.append(time.time())
                    self.content += d.get("content") or ""
                    self.reasoning += d.get("reasoning_content") or ""
                    if d.get("tool_calls"):
                        self.tool_calls.extend(d["tool_calls"])
                    self.finish = ch.get("finish_reason") or self.finish
                if self.cancel_after is not None and self.chunks >= self.cancel_after:
                    self.cancelled = True
                    break
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        finally:
            c.close()
            self.t1 = time.time()

    def ok(self):
        return self.error is None and self.finish in ("stop", "length", "tool_calls") and bool(self.content.strip() or self.reasoning.strip() or self.tool_calls)

    def tokens(self):
        return self.usage.get("completion_tokens", 0)

    def text(self):
        return self.reasoning + "\u0000" + self.content + "\u0000" + json.dumps(self.tool_calls, sort_keys=True)

    def ttft(self):
        return (self.t_first or self.t1) - self.t0

    def cached(self):
        return (self.usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)

    def calls_parse(self):
        """every tool call carries a function name and JSON-object arguments"""
        try:
            return all(tc.get("function", {}).get("name") and isinstance(json.loads(tc["function"].get("arguments") or "{}"), dict) for tc in self.tool_calls)
        except Exception:  # noqa: BLE001
            return False

    def summary(self):
        span = self.t1 - (self.t_first or self.t0)   # a reply the server buffered (a tool call) lands in one delta: no span
        tps = f"{self.tokens() / span:.1f} tok/s" if span > 0.05 else "one delta (reply buffered), no client-side rate"
        calls = ", calls " + ",".join(tc.get("function", {}).get("name", "?") for tc in self.tool_calls) if self.tool_calls else ""
        return (f"req {self.k}: finish={self.finish} tokens={self.tokens()} prompt {self.usage.get('prompt_tokens')} ({self.cached()} cached) "
                f"TTFT {self.ttft():.2f} s, {self.t1 - self.t0:.1f} s total, {tps}{calls}, reasoning {len(self.reasoning)} chars"
                + (f", cancelled after {self.chunks} chunks" if self.cancelled else "") + (f", ERROR {self.error}" if self.error else "")
                + f", text {(self.content or self.reasoning).strip()[:60]!r}")


def steady(reqs):
    """The window in which every request decodes -- from the last first-token to the first finish -- and the deltas landing in it."""
    if any(r.t_first is None for r in reqs):
        return 0.0, 0, 0.0
    w0, w1 = max(r.t_first for r in reqs), min(r.t1 for r in reqs)
    n = sum(1 for r in reqs for t in r.times if w0 < t <= w1)
    return (w1 - w0), n, (n / (w1 - w0) if w1 > w0 else 0.0)


def solo_tps(r):
    return r.tokens() / max(1e-9, r.t1 - (r.t_first or r.t0))


def diff_at(a, b):
    """The first differing character of two texts and the context around it ("" when equal)."""
    if a == b:
        return ""
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"at char {i} of {len(a)} / {len(b)}: {a[max(0, i - 30):i + 40]!r} vs {b[max(0, i - 30):i + 40]!r}"


def run_all(reqs, stagger=0.0):
    ths = []
    for r in reqs:
        t = threading.Thread(target=r.run)
        t.start()
        ths.append(t)
        if stagger:
            time.sleep(stagger)
    for t in ths:
        t.join()
    return reqs


def health():
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.request("GET", "/health")
        d = json.loads(c.getresponse().read())
        c.close()
        return d
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def proc_status(key):
    for l in open(f"/proc/{pid}/status"):
        if l.startswith(key + ":"):
            return int(l.split()[1])
    return -1


def rss_anon_mib():
    return proc_status("RssAnon") // 1024


def n_threads():
    return proc_status("Threads")


def vram_mib():
    """The server's own VRAM per card (drm-pdev), MiB: drm-total-vram of its DRM clients in /proc/<pid>/fdinfo (procfs only)"""
    tot, seen = {}, set()
    d = f"/proc/{pid}/fdinfo"
    for fd in os.listdir(d):
        try:
            txt = open(f"{d}/{fd}").read()
        except OSError:
            continue
        kv = dict(l.split(":", 1) for l in txt.splitlines() if ":" in l)
        pdev, cl = kv.get("drm-pdev", "").strip(), kv.get("drm-client-id", "").strip()
        vram = [v for k, v in kv.items() if k.startswith("drm-total-vram")]
        if not pdev or not vram or (pdev, cl) in seen:
            continue
        seen.add((pdev, cl))
        tot[pdev] = tot.get(pdev, 0) + sum(int(v.split()[0]) for v in vram) // 1024
    return [tot[k] for k in sorted(tot)]


# ---- modes -----------------------------------------------------------------------------------------------------------------------
if mode == "concurrent":
    greedy, thinking = "--greedy" in sys.argv, "--thinking" in sys.argv
    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append(health())
            time.sleep(0.5)

    st = threading.Thread(target=sampler)
    st.start()
    th0 = n_threads()
    reqs = run_all([Req(k, MAX_TOKENS, greedy=greedy, thinking=thinking, seed=1000 + k) for k in range(N)])
    stop.set()
    st.join()
    for r in reqs:
        print(r.summary(), flush=True)
    t_first, t_last = min(r.t0 for r in reqs), max(r.t1 for r in reqs)
    tot = sum(r.tokens() for r in reqs)
    dec0 = min((r.t_first for r in reqs if r.t_first), default=t_first)
    check("complete", all(r.ok() for r in reqs), f"{sum(r.ok() for r in reqs)}/{N} requests finished with text, reasoning or a tool call; finishes {[r.finish for r in reqs]}")
    check("tool calls parse", all(r.calls_parse() for r in reqs), f"{sum(len(r.tool_calls) for r in reqs)} calls over {sum(bool(r.tool_calls) for r in reqs)} replies")
    if thinking:
        check("reasoning_content per lane", any(r.reasoning.strip() for r in reqs), f"{sum(bool(r.reasoning.strip()) for r in reqs)}/{N} replies carry reasoning")
    sw, sn, sr = steady(reqs)
    print(f"aggregate: {tot} completion tokens over {N} requests in {t_last - t_first:.1f} s wall (first start to last finish) = {tot / (t_last - t_first):.2f} tok/s; "
          f"decode window (first token to last finish) {t_last - dec0:.1f} s = {tot / max(1e-9, t_last - dec0):.2f} tok/s; "
          f"STEADY window (all {N} decoding) {sw:.1f} s, {sn} deltas = {sr:.2f} tok/s; TTFTs {[round(r.ttft(), 2) for r in reqs]}; "
          f"server threads {th0} -> {n_threads()}", flush=True)
    la = [s.get("lanes_active") for s in samples if isinstance(s.get("lanes_active"), int)]
    sm = [s.get("step_ms") for s in samples if isinstance(s.get("step_ms"), (int, float)) and s.get("step_ms") > 0]
    rp = [s.get("rows_per_step") for s in samples if isinstance(s.get("rows_per_step"), (int, float)) and s.get("rows_per_step") > 0]
    check("health", bool(la) and max(la) >= min(2, N) and bool(sm) and bool(rp),
          f"{len(samples)} samples: lanes_active max {max(la) if la else None}, step_ms {min(sm) if sm else None}-{max(sm) if sm else None}, "
          f"rows_per_step {min(rp) if rp else None}-{max(rp) if rp else None}; last {json.dumps(samples[-1]) if samples else None}")

elif mode == "identity":
    K, C = int(arg("--cancel-idx", "1")), int(arg("--cancel-after", "6"))
    # each conversation gets its own system prompt (a different first token): the solo runs land on N different lanes and the
    # batch is served from their live states, so the arms differ in the decode interleaving only
    sp = lambda k: f"Conversation {k}. "   # noqa: E731
    warm = []
    for k in range(N):
        r = Req(k, 8, sys_prefix=sp(k))
        r.run()
        warm.append(r)
        print("warm", r.summary(), flush=True)
    check("warm complete", all(r.ok() for r in warm), f"finishes {[r.finish for r in warm]}")
    check("warm: each conversation prefilled alone into a fresh lane", all(r.cached() == 0 for r in warm), f"cached {[r.cached() for r in warm]}")
    solo = []
    for k in range(N):
        r = Req(k, MAX_TOKENS, sys_prefix=sp(k))
        r.run()
        solo.append(r)
        print("solo", r.summary(), flush=True)
    check("solo complete", all(r.ok() for r in solo), f"finishes {[r.finish for r in solo]}")
    check("solo served from its lane (cached = prompt - 1)", all(r.cached() == r.usage.get("prompt_tokens", 0) - 1 for r in solo), f"cached {[r.cached() for r in solo]}")
    batch = run_all([Req(k, MAX_TOKENS, sys_prefix=sp(k)) for k in range(N)])
    for r in batch:
        print("batch", r.summary(), flush=True)
    check("batch served from its lane", all(r.cached() == r.usage.get("prompt_tokens", 0) - 1 for r in batch), f"cached {[r.cached() for r in batch]}")
    if "--save" in sys.argv:
        json.dump({"solo": [r.text() for r in solo], "batch": [r.text() for r in batch]}, open(arg("--save", ""), "w"))
    if "--ref" in sys.argv:
        ref = json.load(open(arg("--ref", "")))
        eqs = [a == b.text() for a, b in zip(ref["solo"], solo)]
        check("solo == reference solo", len(ref["solo"]) == N and all(eqs),
              f"{sum(eqs)}/{N} identical to {arg('--ref', '')}; " + "; ".join(f"req {i} {diff_at(ref['solo'][i], solo[i].text())}" for i in range(min(len(ref['solo']), N)) if not eqs[i]))
    same = [a.text() == b.text() for a, b in zip(solo, batch)]
    check("batch == solo (f)", all(same) and all(r.ok() for r in batch),
          f"{sum(same)}/{N} identical texts; tokens solo {[r.tokens() for r in solo]} batch {[r.tokens() for r in batch]}; "
          + "; ".join(f"req {i} {diff_at(a.text(), b.text())}" for i, (a, b, sm) in enumerate(zip(solo, batch, same)) if not sm))
    canc = run_all([Req(k, MAX_TOKENS, cancel_after=(C if k == K else None), sys_prefix=sp(k)) for k in range(N)])
    for r in canc:
        print("cancel", r.summary(), flush=True)
    others = [(a.text() == b.text()) for k, (a, b) in enumerate(zip(solo, canc)) if k != K]
    check("cancel (c)", canc[K].cancelled and all(others) and all(r.ok() for k, r in enumerate(canc) if k != K),
          f"request {K} closed after {canc[K].chunks} chunks; {sum(others)}/{N - 1} other texts identical to solo; finishes {[r.finish for r in canc]}; "
          + "; ".join(f"req {k} {diff_at(a.text(), b.text())}" for k, (a, b) in enumerate(zip(solo, canc)) if k != K and a.text() != b.text()))
    after = Req(K, 32, sys_prefix=sp(K))
    after.run()
    check("server answers after the cancel", after.ok(), after.summary())
    print(f"health at the end: {json.dumps(health())}; server threads {n_threads()}", flush=True)

elif mode == "evict":
    # Run right after `identity` in the same server, with its --save file as --ref (the solo texts). Every lane holds a
    # conversation; a NEW conversation takes the least recently used lane (whose conversation is kept in a host slot first);
    # then every conversation again, one at a time: each must be served from cache -- its own lane, or its host slot swapped
    # in on another lane (that lane's conversation kept in a slot in turn) -- and reply exactly as its solo run did.
    ref = json.load(open(arg("--ref", "")))
    sp = lambda k: f"Conversation {k}. "   # noqa: E731
    x = Req(0, 16, sys_prefix="Conversation X. ")
    x.run()
    print("new", x.summary(), flush=True)
    check("a new conversation takes a lane", x.ok() and x.cached() < 64, f"cached {x.cached()}")
    again = []
    for k in range(N):
        r = Req(k, MAX_TOKENS, sys_prefix=sp(k))
        r.run()
        again.append(r)
        print("again", r.summary(), flush=True)
    same = [a.text() == b for a, b in zip(again, ref["solo"])]
    check("every conversation served from cache (its lane or a host slot)", all(r.cached() == r.usage.get("prompt_tokens", 0) - 1 for r in again),
          f"cached {[r.cached() for r in again]} of {[r.usage.get('prompt_tokens') for r in again]}")
    check("every conversation replies as its solo run", len(ref["solo"]) == N and all(same),
          f"{sum(same)}/{N} identical; " + "; ".join(f"req {i} {diff_at(ref['solo'][i], again[i].text())}" for i in range(N) if not same[i]))

elif mode == "interrupt":
    # A long prompt (4 prefill chunks) arrives alone: its prefill runs in its serial turn with the cards pipelined over the chunks.
    # --after S seconds later a short request B arrives and waits for the turn: A's prefill stops at the next chunk end, B takes
    # the turn, and A's remaining chunks go through the lane pipe beside B. Both must complete; with --save / --ref, A's text is
    # compared across servers (at --parallel 1 nothing interrupts A: the reference).
    filler = "".join(f"Record {i:04d}: the relay at station {i % 97} reported {(i * 37) % 1000} units at {i % 24:02d}:{(i * 7) % 60:02d}, "
                     f"status {'nominal' if i % 5 else 'degraded'}, operator note {i * 13 % 211}.\n" for i in range(260))
    msgs_a = [{"role": "system", "content": "Interrupt A. You summarise station logs precisely."},
              {"role": "user", "content": filler + "\nList the records whose status is degraded, by record number, then count them."}]
    a = Req(0, int(arg("--max-tokens", "64")), tools=False, msgs=msgs_a)
    b = Req(1, 32, tools=False, sys_prefix="Interrupt B. ")
    ta = threading.Thread(target=a.run)
    ta.start()
    time.sleep(float(arg("--after", "4")))
    b.run()
    ta.join()
    print("A", a.summary(), flush=True)
    print("B", b.summary(), flush=True)
    check("both complete", a.ok() and b.ok(), f"A {a.finish} ({a.usage.get('prompt_tokens')} prompt tokens, TTFT {a.ttft():.1f} s), B {b.finish} (TTFT {b.ttft():.1f} s)")
    if "--save" in sys.argv:
        json.dump({"A": a.text()}, open(arg("--save", ""), "w"))
    if "--ref" in sys.argv:
        ref = json.load(open(arg("--ref", "")))
        check("A == reference A (the prefill handed from the turn to the lane pipe mid-plan)", ref["A"] == a.text(), diff_at(ref["A"], a.text()) or "identical")

elif mode == "room":
    # The MiMo B4 gate's case in a size that fits a job: run the server with --slot-ctx 4096 (lanes 1.. hold 4,096 positions). A
    # ~3,600-token prompt asking for a long count with max_tokens M (default 700) leaves under M on a small lane, so the reply-room
    # rule must put it on lane 0 (32,768): the reply reaches M ("length"), not the small lane's room. Then a short prompt with a
    # small budget must still get a reply (it takes a small lane: the server log's lane line says which).
    M = int(arg("--max-tokens", "700"))
    target = int(arg("--target", "3750"))   # prompt tokens: a 4,096 lane then leaves ~346 < M
    line = lambda i: f"Station {i:03d} log: pressure {(i * 37) % 1000} hPa, valve {i % 7}, crew {i % 5}.\n"   # noqa: E731
    ask = "\nIgnore the logs above. Count from 1 to 3000, one number per line, nothing else."
    def room_msgs(n, tag):   # (a plain system prompt: Dream's asks for tool calls, and a tool call ends the reply early)
        return [{"role": "system", "content": tag + "You answer exactly as asked, in plain text."}, {"role": "user", "content": "".join(line(i) for i in range(n)) + ask}]
    # calibrate the tokens per log line with two 1-token probes (no filler / 100 lines), then size the prompt to `target`
    p0, p1 = Req(2, 1, tools=False, msgs=room_msgs(0, "Probe 0. ")), Req(3, 1, tools=False, msgs=room_msgs(100, "Probe 1. "))
    p0.run(); p1.run()
    t0_, t1_ = p0.usage.get("prompt_tokens", 0), p1.usage.get("prompt_tokens", 0)
    per_line = max(1e-3, (t1_ - t0_) / 100.0)
    n_lines = max(0, int((target - t0_) / per_line))
    print(f"calibration: {t0_} tokens without the logs, {t1_} with 100 lines ({per_line:.1f} per line): {n_lines} lines for ~{target}", flush=True)
    big = Req(0, M, tools=False, msgs=room_msgs(n_lines, "Room A. "))
    big.run()
    print("big", big.summary(), flush=True)
    pt = big.usage.get("prompt_tokens", 0)
    small_room = 4096 - pt
    # the failure this catches: finish "length" short of max_tokens (the reply cut at a small lane's room, ~350 tokens)
    check("the long reply is not cut at a small lane's room", big.ok() and not (big.finish == "length" and big.tokens() < M),
          f"prompt {pt} tokens (a 4,096 lane would leave {small_room}), max_tokens {M}: {big.tokens()} tokens, finish {big.finish}"
          + ("" if small_room < M else " -- NOTE: the prompt came out too short for this case to discriminate (room >= max_tokens)"))
    short = Req(1, 24, tools=False, sys_prefix="Room B. ")
    short.run()
    print("short", short.summary(), flush=True)
    check("a short prompt with a small budget is served", short.ok(), short.summary())

elif mode == "ttft":
    solo = Req(0, 24)
    solo.run()
    print("solo", solo.summary(), flush=True)
    pt = solo.usage.get("prompt_tokens", 0)
    chunk_s = max(5.0, solo.ttft() * 2048.0 / max(1, pt))
    others = [Req(k, MAX_TOKENS, variant=" (variant B)") for k in range(1, N)]
    probe = Req(0, 24, sys_prefix="Session B. ")   # shares no prefix with any lane or slot: a full prefill while the others decode
    ths = [threading.Thread(target=r.run) for r in others]
    for t in ths:
        t.start()
    t_wait = time.time()
    while any(r.t_first is None for r in others) and time.time() - t_wait < 300:
        time.sleep(0.1)
    time.sleep(1.0)
    before = [r.chunks for r in others]
    tb = time.time()
    pth = threading.Thread(target=probe.run)
    pth.start()
    while probe.t_first is None and pth.is_alive():
        time.sleep(0.05)
    during = [r.chunks for r in others]
    td = time.time()
    pth.join()
    for t in ths:
        t.join()
    for r in others:
        print("other", r.summary(), flush=True)
    print("probe", probe.summary(), flush=True)
    print(f"others' chunks during the probe's prefill ({td - tb:.1f} s): {[(d - b) for b, d in zip(before, during)]} (a stalled lane shows 0-2); "
          f"health {json.dumps(health())}", flush=True)
    check("ttft (d)", probe.ok() and probe.ttft() <= solo.ttft() + chunk_s,
          f"solo TTFT {solo.ttft():.2f} s ({pt} prompt tokens; one 2,048-row chunk ~ {chunk_s:.1f} s), busy TTFT {probe.ttft():.2f} s with {N - 1} lane(s) decoding: "
          f"+{probe.ttft() - solo.ttft():.2f} s; probe finish {probe.finish}, cached {probe.cached()}")

elif mode == "aba":
    sp = lambda k: f"Conversation {k}. "   # noqa: E731
    tk = [int(x) for x in arg("--tasks", ",".join(str(k) for k in range(N))).split(",")]   # the task of conversation k (TASKS index)
    samples = []
    stop = threading.Event()
    t_trace0 = time.time()
    phase = ["A1"]

    def sampler():   # /health every 0.5 s through all three arms, each sample tagged with its arm
        while not stop.is_set():
            h = health()
            h["t"] = round(time.time() - t_trace0, 2)
            h["arm"] = phase[0]
            samples.append(h)
            time.sleep(0.5)

    def server_rate(arm, lanes):
        """The server's decode rate in an arm: /health "tokens" over the samples in which exactly `lanes` lanes decode"""
        s = [h for h in samples if h.get("arm") == arm and h.get("decoding") == lanes and isinstance(h.get("tokens"), int)]
        if len(s) < 2 or s[-1]["t"] <= s[0]["t"]:
            return 0.0, 0.0
        return (s[-1]["tokens"] - s[0]["tokens"]) / (s[-1]["t"] - s[0]["t"]), s[-1]["t"] - s[0]["t"]

    if "--warm" in sys.argv:   # every prompt prefilled alone into its own lane first: the arms then only decode (cached = prompt - 1)
        for k in range(N):
            w = Req(tk[k], 4, sys_prefix=sp(k), tools=False)
            w.run()
            print("warm", w.summary(), flush=True)
    st = threading.Thread(target=sampler)
    st.start()
    a1 = Req(tk[0], MAX_TOKENS, sys_prefix=sp(0), tools=False)
    a1.run()
    print("solo A1", a1.summary(), flush=True)
    phase[0] = "B"
    batch = run_all([Req(tk[k], MAX_TOKENS, sys_prefix=sp(k), tools=False) for k in range(N)])
    for r in batch:
        print("concurrent", r.summary(), flush=True)
    phase[0] = "A2"
    a2 = Req(tk[0], MAX_TOKENS, sys_prefix=sp(0), tools=False)
    a2.run()
    print("solo A2", a2.summary(), flush=True)
    time.sleep(0.6)
    stop.set()
    st.join()
    sw, sn, sr = steady(batch)
    (s1, w1), (sb, wb), (s2, w2) = server_rate("A1", 1), server_rate("B", N), server_rate("A2", 1)
    print("concurrent arm by lanes decoding (server): " + ", ".join(f"{k}: {r:.2f} tok/s over {w:.1f} s" for k in range(N, 0, -1)
                                                            for r, w in [server_rate("B", k)] if w > 0), flush=True)
    tot = sum(r.tokens() for r in batch)
    t_first, t_last = min(r.t0 for r in batch), max(r.t1 for r in batch)
    print(f"A-B-A (server, /health tokens while exactly 1 / {N} / 1 lanes decode): solo A1 {s1:.2f} tok/s over {w1:.1f} s, {N} concurrent "
          f"{sb:.2f} tok/s over {wb:.1f} s (x{sb / max(1e-9, (s1 + s2) / 2):.2f} the solo mean; per lane {sb / N:.2f}), solo A2 {s2:.2f} tok/s over {w2:.1f} s", flush=True)
    print(f"A-B-A (client, SSE deltas): solo {solo_tps(a1):.2f} / {solo_tps(a2):.2f} tok/s; {N} concurrent: steady window {sw:.1f} s, {sn} deltas = {sr:.2f} tok/s; "
          f"whole arm {tot} tokens in {t_last - t_first:.1f} s = {tot / (t_last - t_first):.2f} tok/s incl. prefills; TTFTs {[round(r.ttft(), 2) for r in batch]}", flush=True)
    trace = [(h["t"], h.get("arm"), h.get("lanes_active"), h.get("decoding"), h.get("step_ms"), h.get("rows_per_step"), h.get("turns"), h.get("paused_ms"), h.get("handovers"))
             for h in samples if "step_ms" in h and h.get("arm") == "B"]
    print("health trace, concurrent arm (t s, arm, lanes_active, decoding, step_ms EMA, rows_per_step, turns, paused_ms, handovers): "
          + " | ".join(" ".join(str(x) for x in row) for row in trace[::max(1, len(trace) // 24)]), flush=True)
    if samples:
        print(f"health at the end: {json.dumps(samples[-1])}; server threads {n_threads()}", flush=True)
    check("aba complete", a1.ok() and a2.ok() and all(r.ok() for r in batch), f"finishes {a1.finish} {[r.finish for r in batch]} {a2.finish}")

elif mode == "memory":
    R = int(arg("--requests", "20"))
    rss0, vr0, th0 = rss_anon_mib(), vram_mib(), n_threads()
    print(f"before: RssAnon {rss0} MiB, VRAM {vr0} MiB, threads {th0}", flush=True)
    rss1 = vr1 = th1 = None
    done = 0
    wave = 0
    while done < R:
        n = min(N, R - done)
        reqs = run_all([Req((done + i) % len(TASKS), MAX_TOKENS, variant=f" (run {done + i})") for i in range(n)])
        bad = [r.summary() for r in reqs if not r.ok()]
        done += n
        wave += 1
        if wave == 1:
            rss1, vr1, th1 = rss_anon_mib(), vram_mib(), n_threads()
            print(f"after wave 1 ({n} requests): RssAnon {rss1} MiB, VRAM {vr1} MiB, threads {th1}", flush=True)
        print(f"wave {wave}: {n} requests, {sum(r.tokens() for r in reqs)} tokens, finishes {[r.finish for r in reqs]}, RssAnon {rss_anon_mib()} MiB, "
              f"VRAM {vram_mib()} MiB, threads {n_threads()}" + (f" BAD {bad}" if bad else ""), flush=True)
        check(f"wave {wave} complete", not bad, f"{n - len(bad)}/{n}")
    rss2, vr2, th2 = rss_anon_mib(), vram_mib(), n_threads()
    print(f"end: RssAnon {rss2} MiB, VRAM {vr2} MiB, threads {th0} -> {th1} (wave 1) -> {th2}; health {json.dumps(health())}", flush=True)
    check("memory (e)", rss2 - rss1 < 64 and len(vr1) == len(vr2) and all(abs(a - b) < 64 for a, b in zip(vr1, vr2)) and th2 <= th1,
          f"RssAnon {rss0} -> {rss1} (wave 1) -> {rss2} MiB over {R} requests (growth after wave 1: {rss2 - rss1} MiB); VRAM {vr0} -> {vr1} -> {vr2} MiB; "
          f"threads {th0} -> {th1} -> {th2}")

elif mode == "vision":
    img_path = arg("--image", "")
    if not img_path:
        sys.exit("vision mode needs --image <png>")
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(img_path, "rb").read()).decode()}}
    sys_msg = {"role": "system", "content": "You are a careful assistant. Answer briefly and exactly."}
    q0 = "What text is written in this image? Then list each shape you see with its color."
    long = Req(0, MAX_TOKENS, sys_prefix="Conversation 0. ", tools=False,
               variant=" Write the whole note in this reply as prose, at least 500 words, without calling any tool.")
    lt = threading.Thread(target=long.run)
    lt.start()
    while long.t_first is None and lt.is_alive():
        time.sleep(0.05)
    time.sleep(1.0)
    a = Req(1, 200, tools=False, msgs=[sys_msg, {"role": "user", "content": [img, {"type": "text", "text": q0}]}])
    chunks_before = long.chunks
    a.run()
    print("image A", a.summary(), flush=True)
    print(f"the text reply made {long.chunks - chunks_before} chunks while the image request ran ({a.t1 - a.t0:.1f} s)", flush=True)
    b = Req(1, 120, tools=False, msgs=[sys_msg, {"role": "user", "content": [img, {"type": "text", "text": q0}]}, {"role": "assistant", "content": a.content},
                                       {"role": "user", "content": "Which shape is on the right side, and what color is it?"}])
    b.run()
    print("image B (follow-up)", b.summary(), flush=True)
    lt.join()
    print("text", long.summary(), flush=True)
    check("image request answers while another lane decodes", a.ok() and "4721" in a.content, f"{a.content.strip()[:160]!r}")
    check("follow-up served from cache past the image", b.ok() and b.cached() > 206, f"cached {b.cached()} of {b.usage.get('prompt_tokens')}; {b.content.strip()[:120]!r}")
    check("the decoding text reply completes", long.ok(), long.summary())

elif mode == "sse":
    out = arg("--out", "sse")
    mask = [(re.compile(rb'"id":"chatcmpl-\d+"'), b'"id":"chatcmpl-N"'), (re.compile(rb'"created":\d+'), b'"created":0')]
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    q = "Explain in a few sentences why the sky is blue."
    reqs = [
        {"model": "m", "messages": [{"role": "user", "content": q}], "max_tokens": 96, "temperature": 0.7, "top_p": 0.95, "top_k": 0, "seed": 1234, "stream": True},
        {"model": "m", "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}], "tools": tools,
         "max_tokens": 96, "temperature": 0, "stream": True, "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}},
        {"model": "m", "messages": [{"role": "user", "content": "Is 391 a prime number? Think it through, then answer yes or no."}], "max_tokens": 160,
         "temperature": 0, "stream": True, "enable_thinking": True, "chat_template_kwargs": {"enable_thinking": True}},
    ]
    for i, body in enumerate(reqs, 1):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        raw = c.getresponse().read()
        c.close()
        for rx, rep in mask:
            raw = rx.sub(rep, raw)
        with open(f"{out}_{i}.sse", "wb") as f:
            f.write(raw)
        evs = [json.loads(l[6:]) for l in raw.decode("utf-8", "replace").split("\n") if l.startswith("data: {")]
        text = "".join((e.get("choices") or [{}])[0].get("delta", {}).get("content") or "" for e in evs if e.get("choices"))
        reason = "".join((e.get("choices") or [{}])[0].get("delta", {}).get("reasoning_content") or "" for e in evs if e.get("choices"))
        calls = [tc for e in evs if e.get("choices") for tc in (e["choices"][0].get("delta", {}).get("tool_calls") or [])]
        print(f"sse {i}: {len(raw)} bytes md5 {hashlib.md5(raw).hexdigest()}  content {text.strip()[:70]!r} reasoning {len(reason)} chars calls "
              f"{[tc.get('function', {}).get('name') for tc in calls]}", flush=True)
        check(f"sse {i} complete", b"data: [DONE]" in raw and b'"error"' not in raw, f"{len(evs)} events")
else:
    sys.exit(f"unknown mode {mode}")

print("PASS" if not fails else f"FAIL ({fails})", flush=True)
sys.exit(1 if fails else 0)
