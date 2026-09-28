#!/usr/bin/env python3
"""P4 B4 (docs/mimo26/P4_B4_SERVE.md): several requests at once against a live `ie serve` of MiMo-V2.6 at --parallel N.

  parallel_serve_test.py <port> <server pid> <mode> [options]

Every request is Dream-shaped: a ~2.5k-token agent system prompt, 8 tools and a task, streamed. Modes:

  concurrent  --n N --max-tokens M [--greedy] [--thinking]
      N requests at once. Every one must complete (finish stop / length / tool_calls with text or a tool call). Prints each
      request's TTFT, tokens and tok/s, the AGGREGATE tok/s (all completion tokens / the wall from the first start to the last
      finish) and the /health fields (lanes_active, step_ms, rows_per_step) sampled while they run. Criterion (b).
  identity    --n N --max-tokens M [--cancel-idx K] [--cancel-after C] [--save FILE] [--ref FILE [--ref-solo]] [--diverge-ok]
      Greedy, thinking off, N distinct conversations. Phase 0 (warm): every conversation once, ONE AT A TIME, 8 tokens, so
      each prompt is prefilled alone in one chunk (the same bytes in every server) and sits in its own lane. Every later arm
      is served from its lane's live state (cached = prompt - 1), so the arms differ in the DECODE grouping only -- not in how
      the prompt was prefilled (a prompt prefilled in 512-row chunks has different KV bytes from one prefilled in one chunk:
      the prefill GEMMs follow M, and a greedy near-tie moves). Phase 1: every prompt solo, one at a time. Phase 2: all N at
      once. Phase 3: all N at once with request K closed after C content chunks. Every completed request's text must equal
      its solo text, and the first divergence is printed with its character position. Criteria (c) and (f), in the SERVED
      default (B3's auto groups) with only the CPU expert leg off (IE_DS41_CPU_MISS=0: its split follows the stream-slot
      history). --save writes the solo and batch texts as JSON; --ref compares with a saved file: the batch texts must be
      identical (plain rows in both servers), the solo texts are reported (with the drafter on, a lone lane drafts, and the
      verify's multi-row step moves greedy near-ties -- the P5 finding, the same as at --parallel 1); --ref-solo makes the solo
      comparison a check too (P4 B5: budget 0 against the B4 binary: the same lone-lane path). --diverge-ok (P4 B5, the
      draft budget: batched lanes draft fewer rows per step than a lone lane, and a verify step's row count moves greedy
      near-ties): the batch / cancel texts that differ from solo are reported with their first divergence instead of failing
      (every request must still complete, the cancelled one closed), and the teacher-forced scan decides whether the
      divergences are near-ties.
  ttft        --n N --max-tokens M
      TTFT of prompt 0 alone (a fresh server: a full prefill); then N-1 others start decoding (max_tokens M) and 1 s later
      prompt 0 with a system prompt that differs from its first token (so nothing serves it: a full prefill again) starts: its
      TTFT while they decode, and how many chunks the others produced meanwhile (their stall). Criterion (d):
      the busy TTFT must be within the solo TTFT plus one prefill chunk (the solo time of 2,048 prompt rows, at least 5 s).
  aba         --n N --max-tokens M [--prompt-dir DIR] [--warm] [--n-list N1,N2,...]
      Greedy, thinking off, NO tools. Solo request (a lone lane: the drafter drafts) -> N concurrent requests (plain rows, or the
      draft budget's rows, P4 B5) -> solo again; --n-list runs one concurrent arm per count between the solos (A1, B2, B4, A2
      for "2,4"), each arm's rate over the samples with exactly that many lanes decoding, and each arm's drafter figures (/health
      draft_*: passes, their ms, drafts offered / kept, per token). The primary figures are the SERVER's: /health is sampled every 0.5 s through all the arms, and each arm's
      rate is the committed ids ("tokens") over the samples in which exactly 1 / N / 1 lanes decode ("decoding") -- after a
      <tool_call> the server buffers the rest of a reply, so the SSE deltas stop counting tokens (a held-out transcript's reply
      is often a tool call). Also reported from the client: each solo's tok/s (first token to finish), the concurrent arm's
      steady-window deltas (from the last first-token to the first finish) and the per-request token counts over their own
      decode spans (streamed replies only), and a /health trace of the concurrent arm (lanes, step_ms, rows_per_step, turns,
      paused_ms, the gate's ms). --prompt-dir: the prompts are that directory's files 03.txt, 04.txt, ... (the held-out Dream
      transcripts; B3's lanes test ran 03-06), each sent as one user message with the chat template, instead of the
      synthetic Dream-shaped prompts. --warm: every prompt is first prefilled alone into its own lane (4 tokens), so the arms
      are served from the lanes' live states and only decode (with long held-out prompts the concurrent arrivals' prefills
      otherwise leave almost no window in which all N decode).
  turns       --max-tokens M [--every S]
      The serial turn's cost to a decoding lane (the pipe pause / resume). Greedy, thinking off, no tools. A short
      conversation is warmed into its own lane; then one long reply (M tokens, a fresh prompt) decodes while that short
      conversation is asked again and again (4 tokens each, S seconds apart, one at a time): each short request is served from
      its lane and takes two serial turns (the prefix step, the prompt-end snapshot) while the long one decodes. Reports the
      long reply's tok/s (first token to finish), the turns taken during it and the pipe's paused ms per turn (/health), and the
      short requests' TTFTs. Run it with and without IE_MIMO26_PIPE_PAUSE=0 (stop and restart the stage threads at every turn),
      and with IE_MIMO26_STEP_TRACE=1 for the step times around each turn in the server log.
  memory      --requests R --n N [--max-tokens M]
      R requests in waves of N; RssAnon (/proc/<pid>/status) and each card's VRAM (xpu-smi) before, after wave 1 and at the
      end: the growth after wave 1 must stay under 64 MiB of RssAnon and 64 MiB of VRAM. Criterion (e). Run the server with
      IE_MIMO26_PROMPT_CACHE_GIB=0: with host slots on, every distinct 2.6k-token conversation is legitimately KEPT in host RAM
      (~0.5 GiB each, up to the 16 GiB budget), which is the P7 design, not a leak.
Standard library only. Prints ok / FAIL lines with their evidence; exit 0 = PASS.
"""
import hashlib
import http.client
import json
import subprocess
import sys
import threading
import time

port, pid, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


N = int(arg("--n", "4"))
MAX_TOKENS = int(arg("--max-tokens", "128"))
fails = 0


def check(name, ok, evidence):
    global fails
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {evidence}", flush=True)


# ---- Dream-shaped prompts -------------------------------------------------------------------------------------------------
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
TASKS = [
    "The test suite in tests/ has one failing test, tests/test_prune.py::test_status_with_colons. Investigate and fix the bug in src/prune.py without changing the tests. Start by restating the task and your DONE checklist.",
    "Write a design note docs/cache_eviction.md that compares LRU, LFU and ARC for our 16 GiB host slot cache, with a recommendation. Start by restating the task and your DONE checklist.",
    "The nightly CI job in .github/workflows/nightly.yml has been red for three days with 'ModuleNotFoundError: yaml'. Find the cause and propose the smallest fix. Start by restating the task and your DONE checklist.",
    "Add a --json flag to scripts/report.py so its summary prints as one JSON object instead of a table; keep the default output unchanged. Start by restating the task and your DONE checklist.",
    "Review src/server/admission.py for race conditions between acquire() and shutdown(); report each with a reproduction. Start by restating the task and your DONE checklist.",
    "Our README's install section is out of date: it names python 3.9 and pip, while pyproject.toml requires 3.12 and uv. Rewrite that section. Start by restating the task and your DONE checklist.",
    "Profile scripts/import_sessions.py on a 40 MB log: it takes 90 s. Find the hot spot and propose a fix with an expected speed-up. Start by restating the task and your DONE checklist.",
    "Migrate tests/ from unittest to pytest style without changing what is asserted; list every file you would touch first. Start by restating the task and your DONE checklist.",
]


PROMPT_DIR = arg("--prompt-dir", "")
PROMPT_FILES = sorted(f for f in __import__("os").listdir(PROMPT_DIR) if f.endswith(".txt")) if PROMPT_DIR else []
if "--prompt-files" in sys.argv:   # (P4 B5) fail here, not inside a request thread
    missing = [n for n in arg("--prompt-files", "").split(",") if f"{n}.txt" not in PROMPT_FILES]
    if not PROMPT_DIR or missing:
        sys.exit(f"--prompt-files needs --prompt-dir holding every file: missing {missing}" if PROMPT_DIR else "--prompt-files needs --prompt-dir")


def messages(k, variant="", sys_prefix=""):
    # sys_prefix: a different first token makes the prompt share NO prefix with any lane or slot (a real prefill)
    if PROMPT_FILES:   # a held-out transcript as one user message: 03, 04, ... (B3's lanes test ran 03-06; 00-02 are one 68-token prompt)
        files = [f for f in PROMPT_FILES if f not in ("00.txt", "01.txt", "02.txt")]
        if "--prompt-files" in sys.argv:   # (P4 B5) the files by name, in this order: e.g. 03,04,05,07 (06's reply is a ~40-token tool call)
            files = [f"{n}.txt" for n in arg("--prompt-files", "").split(",")]
        text = open(f"{PROMPT_DIR}/{files[k % len(files)]}", encoding="utf-8", errors="replace").read()
        return [{"role": "user", "content": sys_prefix + text + variant}]
    return [{"role": "system", "content": sys_prefix + SYSTEM}, {"role": "user", "content": TASKS[k % len(TASKS)] + variant}]


# ---- one streamed request ---------------------------------------------------------------------------------------------------
class Req:
    def __init__(self, k, max_tokens, greedy=True, thinking=False, cancel_after=None, variant="", seed=1234, tools=True, sys_prefix=""):
        self.k, self.max_tokens, self.greedy, self.thinking, self.cancel_after, self.variant = k, max_tokens, greedy, thinking, cancel_after, variant
        self.seed, self.tools, self.sys_prefix = seed, tools, sys_prefix
        self.content, self.reasoning, self.tool_calls, self.finish, self.usage, self.error = "", "", [], None, {}, None
        self.t0 = self.t_first = self.t1 = None
        self.chunks = 0
        self.times = []   # arrival time of every content / reasoning delta (~ one token each)
        self.cancelled = False

    def run(self):
        body = {"model": "m", "messages": messages(self.k, self.variant, self.sys_prefix), "max_tokens": self.max_tokens, "stream": True,
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
                self.error = f"HTTP {r.status}: {r.read()[:200]!r}"
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

    def summary(self):
        span = self.t1 - (self.t_first or self.t0)   # a reply the server buffered (after a <tool_call>) lands in one delta: no span
        tps = f"{self.tokens() / span:.1f} tok/s" if span > 0.05 else "one delta (reply buffered), no client-side rate"
        return (f"req {self.k}: finish={self.finish} tokens={self.tokens()} prompt {self.usage.get('prompt_tokens')} ({self.cached()} cached) "
                f"TTFT {self.ttft():.2f} s, {self.t1 - self.t0:.1f} s total, {tps}"
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


def rss_anon_mib():
    for l in open(f"/proc/{pid}/status"):
        if l.startswith("RssAnon:"):
            return int(l.split()[1]) // 1024
    return -1


def n_threads():
    for l in open(f"/proc/{pid}/status"):
        if l.startswith("Threads:"):
            return int(l.split()[1])
    return -1


def vram_mib():
    out = []
    for d in (0, 1):
        try:
            txt = subprocess.run(["xpu-smi", "stats", "-d", str(d)], capture_output=True, text=True, timeout=60).stdout
            line = next(l for l in txt.splitlines() if "GPU Memory Used" in l)
            out.append(int(line.split("avg:")[1].split(",")[0].strip()))
        except Exception:  # noqa: BLE001
            out.append(-1)
    return out


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
    reqs = run_all([Req(k, MAX_TOKENS, greedy=greedy, thinking=thinking, seed=1000 + k) for k in range(N)])
    stop.set()
    st.join()
    for r in reqs:
        print(r.summary(), flush=True)
    t_first, t_last = min(r.t0 for r in reqs), max(r.t1 for r in reqs)
    tot = sum(r.tokens() for r in reqs)
    dec0 = min((r.t_first for r in reqs if r.t_first), default=t_first)
    check("complete", all(r.ok() for r in reqs), f"{sum(r.ok() for r in reqs)}/{N} requests finished with text or a tool call; finishes {[r.finish for r in reqs]}")
    sw, sn, sr = steady(reqs)
    print(f"aggregate: {tot} completion tokens over {N} requests in {t_last - t_first:.1f} s wall (first start to last finish) = {tot / (t_last - t_first):.2f} tok/s; "
          f"decode window (first token to last finish) {t_last - dec0:.1f} s = {tot / max(1e-9, t_last - dec0):.2f} tok/s; "
          f"STEADY window (all {N} decoding: last first-token to first finish) {sw:.1f} s, {sn} deltas = {sr:.2f} tok/s; TTFTs {[round(r.ttft(), 2) for r in reqs]}", flush=True)
    la = [s.get("lanes_active") for s in samples if isinstance(s.get("lanes_active"), int)]
    sm = [s.get("step_ms") for s in samples if isinstance(s.get("step_ms"), (int, float)) and s.get("step_ms") > 0]
    rp = [s.get("rows_per_step") for s in samples if isinstance(s.get("rows_per_step"), (int, float)) and s.get("rows_per_step") > 0]
    check("health", bool(la) and max(la) >= 2 and bool(sm) and bool(rp),
          f"{len(samples)} samples: lanes_active max {max(la) if la else None}, step_ms {min(sm) if sm else None}-{max(sm) if sm else None}, "
          f"rows_per_step {min(rp) if rp else None}-{max(rp) if rp else None}; last {json.dumps(samples[-1]) if samples else None}")

elif mode == "identity":
    K, C = int(arg("--cancel-idx", "1")), int(arg("--cancel-after", "6"))
    # each conversation gets its own system prompt (a different first token): the solo runs land on N different lanes and the
    # batch is served from their live states, so the arms differ in the DECODE grouping only. Prompts sharing the ~1.6k-token
    # system prefix are served as continuations of each other's lane (the P7 rule) and re-prefilled with a different chunking,
    # which changes the prompt's KV bytes (the prefill kernels follow M) -- a real effect, but not the one under test.
    sp = lambda k: f"Conversation {k}. "   # noqa: E731
    warm = []
    for k in range(N):
        r = Req(k, 8, sys_prefix=sp(k))
        r.run()
        warm.append(r)
        print("warm", r.summary(), flush=True)
    check("warm complete", all(r.ok() for r in warm), f"finishes {[r.finish for r in warm]}")
    check("warm: each conversation prefilled alone in a fresh lane", all(r.cached() == 0 for r in warm), f"cached {[r.cached() for r in warm]}")
    solo = []
    for k in range(N):
        r = Req(k, MAX_TOKENS, sys_prefix=sp(k))
        r.run()
        solo.append(r)
        print("solo", r.summary(), flush=True)
    check("solo complete", all(r.ok() for r in solo), f"finishes {[r.finish for r in solo]}")
    check("solo served from its lane", all(r.cached() == r.usage.get("prompt_tokens", 0) - 1 for r in solo), f"cached {[r.cached() for r in solo]}")
    batch = run_all([Req(k, MAX_TOKENS, sys_prefix=sp(k)) for k in range(N)])
    for r in batch:
        print("batch", r.summary(), flush=True)
    check("batch served from its lane", all(r.cached() == r.usage.get("prompt_tokens", 0) - 1 for r in batch), f"cached {[r.cached() for r in batch]}")
    if "--save" in sys.argv:
        json.dump({"solo": [r.text() for r in solo], "batch": [r.text() for r in batch]}, open(arg("--save", ""), "w"))
    if "--ref" in sys.argv:
        ref = json.load(open(arg("--ref", "")))
        eqb = [a == b.text() for a, b in zip(ref["batch"], batch)]
        check("batch == reference batch (plain rows in both servers)", len(ref["batch"]) == N and all(eqb),
              f"{sum(eqb)}/{N} identical to {arg('--ref', '')}; " + "; ".join(f"req {i} {diff_at(ref['batch'][i], batch[i].text())}" for i in range(min(len(ref['batch']), N)) if not eqb[i]))
        eqs = [a == b.text() for a, b in zip(ref["solo"], solo)]
        ev_s = f"{sum(eqs)}/{N} identical; " + "; ".join(f"req {i} {diff_at(ref['solo'][i], solo[i].text())}" for i in range(min(len(ref['solo']), N)) if not eqs[i])
        if "--ref-solo" in sys.argv:   # (P4 B5) the solo arms must match too: the same lone-lane drafted path in both servers
            check("solo == reference solo", len(ref["solo"]) == N and all(eqs), ev_s)
        else:
            print(f"info solo vs reference solo: {ev_s}", flush=True)
    same = [a.text() == b.text() for a, b in zip(solo, batch)]
    diverge_ok = "--diverge-ok" in sys.argv   # (P4 B5: batched lanes draft other row counts than a lone lane -- reported, not failed)
    ev_f = (f"{sum(same)}/{N} identical texts; tokens solo {[r.tokens() for r in solo]} batch {[r.tokens() for r in batch]}; "
            + "; ".join(f"req {i} {diff_at(a.text(), b.text())}" for i, (a, b, sm) in enumerate(zip(solo, batch, same)) if not sm))
    if diverge_ok:
        check("batch complete", all(r.ok() for r in batch), f"finishes {[r.finish for r in batch]}")
        print(f"info batch vs solo (--diverge-ok): {ev_f}", flush=True)
    else:
        check("batch == solo (f)", all(same) and all(r.ok() for r in batch), ev_f)
    canc = run_all([Req(k, MAX_TOKENS, cancel_after=(C if k == K else None), sys_prefix=sp(k)) for k in range(N)])
    for r in canc:
        print("cancel", r.summary(), flush=True)
    others = [(a.text() == b.text()) for k, (a, b) in enumerate(zip(solo, canc)) if k != K]
    ev_c = (f"request {K} closed after {canc[K].chunks} chunks; {sum(others)}/{N - 1} other texts identical to solo; finishes {[r.finish for r in canc]}; "
            + "; ".join(f"req {k} {diff_at(a.text(), b.text())}" for k, (a, b) in enumerate(zip(solo, canc)) if k != K and a.text() != b.text()))
    if diverge_ok:
        check("cancel (c) completes", canc[K].cancelled and all(r.ok() for k, r in enumerate(canc) if k != K), ev_c)
    else:
        check("cancel (c)", canc[K].cancelled and all(others) and all(r.ok() for k, r in enumerate(canc) if k != K), ev_c)
    print("cancel vs batch (the same lane states): " + ("; ".join(f"req {k} {diff_at(a.text(), b.text())}" for k, (a, b) in enumerate(zip(batch, canc)) if k != K and a.text() != b.text()) or "identical"), flush=True)
    after = Req(K, 32, sys_prefix=sp(K))
    after.run()
    check("server answers after the cancel", after.ok(), after.summary())

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
    # wait until every other request has its first token (they decode), then start the probe
    t_wait = time.time()
    while any(r.t_first is None for r in others) and time.time() - t_wait < 300:
        time.sleep(0.1)
    time.sleep(1.0)
    paces = []   # the others' committed chunks over time: before / during / after the probe's prefill
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
    stall = [(d - b) for b, d in zip(before, during)]
    print(f"others' chunks during the probe's prefill ({td - tb:.1f} s): {stall} (a stalled lane shows 0-2)", flush=True)
    check("ttft (d)", probe.ok() and probe.ttft() <= solo.ttft() + chunk_s,
          f"solo TTFT {solo.ttft():.2f} s ({pt} prompt tokens; one 2,048-row chunk ~ {chunk_s:.1f} s), busy TTFT {probe.ttft():.2f} s with {N - 1} lanes decoding: "
          f"+{probe.ttft() - solo.ttft():.2f} s; probe finish {probe.finish}, cached {probe.usage.get('prompt_tokens_details', {}).get('cached_tokens')}")

elif mode == "aba":
    sp = lambda k: f"Conversation {k}. "   # noqa: E731
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

    def spans(arm, lanes, key):
        """The deltas of /health fields over the consecutive sample pairs that both have exactly `lanes` lanes decoding in `arm`
        (a gap between two runs of such samples -- a solo's prefix step, another lane's prefill -- is left out); 0.5 s resolution."""
        d, t = {}, 0.0
        for a, b in zip(samples, samples[1:]):
            if all(h.get("arm") == arm and h.get("decoding") == lanes and isinstance(h.get(key), int) for h in (a, b)):
                t += b["t"] - a["t"]
                for k, v in b.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(a.get(k), (int, float)):
                        d[k] = d.get(k, 0) + v - a[k]
        return d, t

    def server_rate(arm, lanes):
        """The server's own decode rate in an arm: /health "tokens" (ids committed, every lane) over the samples in which exactly
        `lanes` lanes decode -- independent of how the SSE deltas land (after a <tool_call> the server buffers the reply)."""
        d, t = spans(arm, lanes, "tokens")
        return (d.get("tokens", 0) / t if t > 0 else 0.0), t

    def drafter(arm, lanes):
        """(P4 B5) the drafter over the samples in which exactly `lanes` lanes decode: /health draft_* deltas -- passes, their ms, the
        drafts offered and kept -- and the committed ids over the same samples (0.5 s resolution: the edges are approximate)."""
        d, t = spans(arm, lanes, "draft_calls")
        if t <= 0:
            return "no samples"
        return (f"{d['draft_calls']} draft passes ({d['draft_ms'] / max(1, d['draft_calls']):.1f} ms each, {d['draft_ms'] / max(1, d['tokens']):.2f} ms per token), "
                f"{d['draft_offered']} drafts offered, {d['draft_accepted']} kept ({d['draft_accepted'] / max(1, d['draft_offered']):.0%}); "
                f"{d['tokens']} tokens in {d.get('decode_steps', 0)} lane steps ({d['tokens'] / max(1, d.get('decode_steps', 0)):.2f} per step), "
                f"{d.get('grouped_steps', 0)} of them in a group of several lanes")

    counts = [int(x) for x in arg("--n-list", str(N)).split(",")]
    NW = max(counts)
    if "--warm" in sys.argv:   # every prompt prefilled alone into its own lane first: the arms then only decode (cached = prompt - 1)
        for k in range(NW):
            w = Req(k, 4, sys_prefix=sp(k), tools=False)
            w.run()
            print("warm", w.summary(), flush=True)
    st = threading.Thread(target=sampler)
    st.start()
    # --solo-all (P4 B5): each solo arm runs every prompt of the concurrent arms alone, one after another, so the solo reference is
    # the same prompt mix (a prompt's drafted solo rate follows its acceptance); else prompt 0 alone, as B4
    solo_ks = list(range(NW)) if "--solo-all" in sys.argv else [0]
    a1s = []
    for k in solo_ks:
        r = Req(k, MAX_TOKENS, sys_prefix=sp(k), tools=False)
        r.run()
        print("solo A1", r.summary(), flush=True)
        a1s.append(r)
    a1 = a1s[0]
    batches = {}
    for n in counts:
        phase[0] = f"B{n}"
        batches[n] = run_all([Req(k, MAX_TOKENS, sys_prefix=sp(k), tools=False) for k in range(n)])
        for r in batches[n]:
            print(f"concurrent B{n}", r.summary(), flush=True)
    phase[0] = "A2"
    a2s = []
    for k in solo_ks:
        r = Req(k, MAX_TOKENS, sys_prefix=sp(k), tools=False)
        r.run()
        print("solo A2", r.summary(), flush=True)
        a2s.append(r)
    a2 = a2s[0]
    time.sleep(0.6)
    stop.set()
    st.join()
    (s1, w1), (s2, w2) = server_rate("A1", 1), server_rate("A2", 1)
    print(f"solo drafter: A1 {drafter('A1', 1)}; A2 {drafter('A2', 1)}", flush=True)
    for n in counts:
        batch, arm = batches[n], f"B{n}"
        sw, sn, sr = steady(batch)
        sb, wb = server_rate(arm, n)
        print(f"concurrent arm {arm} by lanes decoding (server): " + ", ".join(f"{k}: {r:.2f} tok/s over {w:.1f} s" for k in range(n, 0, -1)
                                                                     for r, w in [server_rate(arm, k)] if w > 0), flush=True)
        print(f"concurrent arm {arm} drafter (exactly {n} decoding): {drafter(arm, n)}", flush=True)
        tot = sum(r.tokens() for r in batch)
        streamed = [r for r in batch if r.chunks > 1 and r.t1 - (r.t_first or r.t1) > 0.5]   # (a buffered reply has no decode span on the client)
        by_tokens = sum(solo_tps(r) for r in streamed)
        t_first, t_last = min(r.t0 for r in batch), max(r.t1 for r in batch)
        print(f"A-B-A (server, /health tokens while exactly 1 / {n} / 1 lanes decode): solo A1 {s1:.2f} tok/s over {w1:.1f} s, {n} concurrent "
              f"{sb:.2f} tok/s over {wb:.1f} s ({sb / max(1e-9, (s1 + s2) / 2):.2f}x the solo mean; per lane {sb / n:.2f}), solo A2 {s2:.2f} tok/s over {w2:.1f} s", flush=True)
        print(f"A-B-A (client, SSE deltas): solo {solo_tps(a1):.2f} / {solo_tps(a2):.2f} tok/s (the drafter drafts for a lone lane); {n} concurrent: steady window "
              f"{sw:.1f} s, {sn} deltas = {sr:.2f} tok/s; from token counts {by_tokens:.2f} tok/s over {len(streamed)}/{n} streamed replies (each one's tokens "
              f"over its own decode time, summed); whole arm {tot} tokens in {t_last - t_first:.1f} s = {tot / (t_last - t_first):.2f} tok/s incl. prefills; "
              f"TTFTs {[round(r.ttft(), 2) for r in batch]}", flush=True)
        trace = [(h["t"], h.get("arm"), h.get("lanes_active"), h.get("decoding"), h.get("step_ms"), h.get("rows_per_step"), h.get("turns"), h.get("paused_ms"),
                  h.get("gate_ms_1"), h.get("gate_ms_n")) for h in samples if "step_ms" in h and h.get("arm") == arm]
        print(f"health trace, concurrent arm {arm} (t s, arm, lanes_active, decoding, step_ms EMA, rows_per_step, turns, paused_ms, gate_ms_1, gate_ms_n): "
              + " | ".join(" ".join(str(x) for x in row) for row in trace[::max(1, len(trace) // 24)]), flush=True)
    if samples:
        print(f"health at the end: {json.dumps(samples[-1])}; server threads {n_threads()}", flush=True)
    check("aba complete", all(r.ok() for r in a1s + a2s) and all(r.ok() for n in counts for r in batches[n]),
          f"finishes {[r.finish for r in a1s]} {[[r.finish for r in batches[n]] for n in counts]} {[r.finish for r in a2s]}")

elif mode == "turns":
    every = float(arg("--every", "1.0"))
    th_turns0 = n_threads()
    warm = Req(3, 4, sys_prefix="Conversation 1. ", tools=False)
    warm.run()
    print("warm", warm.summary(), flush=True)
    long = Req(1, MAX_TOKENS, sys_prefix="Conversation 0. ", tools=False,
               variant=" Write the whole note in this reply as prose, at least 600 words, without calling any tool.")
    lt = threading.Thread(target=long.run)
    lt.start()
    while long.t_first is None and lt.is_alive():
        time.sleep(0.02)
    h1 = health()
    shorts = []
    while lt.is_alive():
        r = Req(3, 4, sys_prefix="Conversation 1. ", tools=False)
        r.run()
        if long.t1 is None or r.t1 <= long.t1:   # (only the shorts that ran entirely inside the long decode)
            shorts.append(r)
        time.sleep(every)
    lt.join()
    h2 = health()
    print("long", long.summary(), flush=True)
    dturns, dpaused = h2.get("turns", 0) - h1.get("turns", 0), h2.get("paused_ms", 0) - h1.get("paused_ms", 0)
    ttfts = sorted(r.ttft() for r in shorts)
    med = ttfts[len(ttfts) // 2] if ttfts else 0.0
    print(f"turns: long reply {long.tokens()} tokens at {solo_tps(long):.2f} tok/s (first token to finish, {long.t1 - (long.t_first or long.t0):.1f} s); "
          f"{len(shorts)} short requests inside it, cached {sorted(set(r.cached() for r in shorts))}, TTFT median {med:.3f} s "
          f"(min {min(ttfts, default=0):.3f}, max {max(ttfts, default=0):.3f}); {dturns} turns, the pipe paused {dpaused:.0f} ms for them "
          f"({dpaused / max(1, dturns):.1f} ms per turn); server threads {th_turns0} -> {n_threads()}; health at the end {json.dumps(h2)}", flush=True)
    check("turns complete", long.ok() and all(r.ok() for r in shorts) and len(shorts) >= 3,
          f"long finish {long.finish}, {sum(r.ok() for r in shorts)}/{len(shorts)} short requests ok")

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
              f"threads {n_threads()}" + (f" BAD {bad}" if bad else ""), flush=True)
        check(f"wave {wave} complete", not bad, f"{n - len(bad)}/{n}")
    rss2, vr2, th2 = rss_anon_mib(), vram_mib(), n_threads()
    print(f"end: RssAnon {rss2} MiB, VRAM {vr2} MiB, threads {th0} -> {th1} (wave 1) -> {th2}; health {json.dumps(health())}", flush=True)
    check("memory (e)", rss2 - rss1 < 64 and all(abs(a - b) < 64 for a, b in zip(vr1, vr2)),
          f"RssAnon {rss0} -> {rss1} (wave 1) -> {rss2} MiB over {R} requests (growth after wave 1: {rss2 - rss1} MiB); VRAM {vr0} -> {vr1} -> {vr2} MiB")
else:
    sys.exit(f"unknown mode {mode}")

print("PASS" if not fails else f"FAIL ({fails})", flush=True)
sys.exit(1 if fails else 0)
