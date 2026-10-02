#!/usr/bin/env python3
"""P4 B29: the swarm replay -- a prefill-heavy multi-lane agent workload against a RUNNING OpenAI-compatible server.

  swarm_replay.py --port P --out FILE.json [--scenario workers|wave|both] [--lanes N] [--lead] [--seed S] [...]

The shape is the Dream swarm of 2026-09-30 (~/Desktop/Dream-Agent-Harness/var/logs/machx.log 21:10-21:49, the 35B-A3B crown split
at `ie serve --gpus 2 --parallel 16 --ctx 262144`: lane 0 ctx 262144, lanes 1..15 ctx 32768):
  workers  N worker conversations at once, each: one shared ~1,100-token system prompt (the log's "1087 cached" restores) + a
           ~300-token task; then turns, each appending a 1-9K-token tool output (real repo file text) and asking for a
           100-300-token reply, until the conversation would pass --conv-cap tokens (the worker lanes hold 32,768). In the log the
           worker prompts grew 1.4K -> 24K over ~8 turns, mostly "(1087 cached)": the prompt cache lost the conversations.
  wave     N workers launched TOGETHER whose system prompt is ONE identical ~30K-token text (the log's 21:33:15 wave: 15 prompts
           of ~30,270 tokens sharing a 30,181-token prefix, all "(0 cached)", 723-734 s each at 41 tok/s), one short task each;
           then --wave-turns follow-ups (a 200..--wave-append-max append each, so they stay on the 32,768 worker lanes).
  both     workers, then wave.
  --lead   adds one lead conversation (lane 0's: a --lead-tokens history, ~54K in the log) with the same per-turn appends.
  needle   the long-prefill correctness check (P4 B29's tile-attention gate): per --needle-tokens size and --needle-depths
           depth, ONE request whose repo-text haystack hides a unique fact ("The vault code for <tag> is <number>.") and asks
           for it (32 tokens, greedy); PASS = the reply contains the number. Sequential, nothing else running; exit 1 on a miss.

Text: files under src/ and include/ of this repo, picked by --seed (deterministic for an A/B), cut to the target size at
--chars-per-token (3.2 = a rough code ratio; the server's own prompt_tokens are what the report uses). Requests are streamed,
thinking off, temperature 0, max_tokens drawn per turn in [--reply-min, --reply-max].

Report (stdout + --out JSON), per conversation and aggregate: per request prompt / cached / new tokens, TTFT, prefill tok/s (new
tokens / TTFT), decode tok/s (completion tokens / (end - first chunk)), turn latency; aggregate client-side decode tok/s over
the run, p50/p95 of the per-request rates and latencies, and the server's /health "tokens" rate (lanes archs) sampled at 1 Hz.
The run fails (exit 1) only on request errors. It does not start or stop a server: run it beside one, e.g.
  build/src/ie serve MODEL --gpus 2 --parallel 16 --ctx 262144 --port 8090
  python3 tools/swarm_replay.py --port 8090 --scenario both --lanes 15 --lead --out runs/A.json
Standard library only.
"""
import argparse
import http.client
import json
import os
import random
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scenario", choices=["workers", "wave", "both", "needle"], default="workers")
ap.add_argument("--needle-tokens", default="32000,54000", help="needle: haystack sizes (comma separated)")
ap.add_argument("--needle-depths", default="0.1,0.5,0.9", help="needle: where the fact sits, as a fraction of the haystack")
ap.add_argument("--lanes", type=int, default=15, help="concurrent worker conversations")
ap.add_argument("--lead", action="store_true", help="add the lead conversation (lane 0's long history)")
ap.add_argument("--lead-tokens", type=int, default=54000)
ap.add_argument("--lead-turns", type=int, default=4)
ap.add_argument("--system-tokens", type=int, default=1100, help="workers: the shared system prompt")
ap.add_argument("--task-tokens", type=int, default=300)
ap.add_argument("--append-min", type=int, default=1000)
ap.add_argument("--append-max", type=int, default=9000)
ap.add_argument("--reply-min", type=int, default=100)
ap.add_argument("--reply-max", type=int, default=300)
ap.add_argument("--turns", type=int, default=8, help="workers: the most turns per conversation")
ap.add_argument("--conv-cap", type=int, default=30000, help="a conversation ends before its prompt would pass this (lanes 1..15 hold 32768)")
ap.add_argument("--wave-tokens", type=int, default=29000, help="wave: the identical shared system prompt (kept under the 32768 worker lanes)")
ap.add_argument("--wave-turns", type=int, default=1, help="wave: follow-up turns after the first")
ap.add_argument("--wave-append-max", type=int, default=1500, help="wave: a follow-up's append, 200..this (the log's were 239-316)")
ap.add_argument("--stagger", type=float, default=0.0, help="workers: seconds between conversation starts")
ap.add_argument("--chars-per-token", type=float, default=3.2)
ap.add_argument("--seed", type=int, default=29)
ap.add_argument("--timeout", type=float, default=3600.0, help="per request")
ap.add_argument("--dry-run", action="store_true", help="print the planned turns (estimated tokens) and exit; no server")
A = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def corpus():
    files = []
    for top in ("src", "include"):
        for d, _, fs in os.walk(os.path.join(ROOT, top)):
            files += [os.path.join(d, f) for f in fs if f.endswith((".cpp", ".hpp", ".h"))]
    files.sort()
    if not files:
        sys.exit("swarm_replay: no source files under src/ or include/")
    return files


FILES = corpus()


def file_text(rng, tokens):
    """About `tokens` tokens of real repo text: whole files from a seeded pick, the last one cut."""
    want, out = int(tokens * A.chars_per_token), []
    while sum(len(s) for s in out) < want:
        p = rng.choice(FILES)
        with open(p, encoding="utf-8", errors="replace") as f:
            body = f.read()
        out.append(f"==> {os.path.relpath(p, ROOT)} <==\n{body}\n")
    s = "".join(out)
    return s[:want]


def est(text):
    return int(len(text) / A.chars_per_token)


# ---- the plan: per conversation a fixed system prompt + task, and per turn an append and a reply cap ---------------------
class Conv:
    def __init__(self, name, system, task, appends, replies, start_delay=0.0):
        self.name, self.system, self.task, self.appends, self.replies, self.start_delay = name, system, task, appends, replies, start_delay
        self.reqs = []   # one dict per turn


def plan():
    rng = random.Random(A.seed)
    convs = []
    tasks = ["Read the files the tool returns and report every function that touches device memory, one line each.",
             "Review the tool output for error paths that drop an error message; list file:line and the fix.",
             "Summarise what the returned code does and which env knobs change its behaviour.",
             "Find the places in the returned code that allocate per request and say whether each is freed."]

    def turns(n_max, start_tokens):
        appends, replies, total = [], [], start_tokens
        for t in range(n_max):
            a = 0 if t == 0 else rng.randint(A.append_min, A.append_max)
            r = rng.randint(A.reply_min, A.reply_max)
            if total + a + r + 64 > A.conv_cap:
                break
            appends.append(a); replies.append(r)
            total += a + r + 64
        return appends, replies

    if A.scenario in ("workers", "both"):
        system = "You are a worker agent of a coding swarm. " + file_text(rng, A.system_tokens)
        for i in range(A.lanes):
            task = tasks[i % len(tasks)] + "\n" + file_text(rng, A.task_tokens)
            ap_, rp = turns(A.turns, A.system_tokens + A.task_tokens)
            convs.append(Conv(f"worker{i:02d}", system, task, ap_, rp, start_delay=i * A.stagger))
    if A.scenario in ("wave", "both"):
        system = "You are a worker agent; the lead's shared context follows.\n" + file_text(rng, A.wave_tokens)
        for i in range(A.lanes):
            task = f"Sub-task {i}: " + tasks[i % len(tasks)]
            ap_ = [0] + [rng.randint(200, max(200, A.wave_append_max)) for _ in range(A.wave_turns)]
            rp = [rng.randint(A.reply_min, A.reply_max) for _ in ap_]
            convs.append(Conv(f"wave{i:02d}", system, task, ap_, rp))
    if A.lead:
        system = "You are the lead agent of a coding swarm."
        task = "Plan the work over this history.\n" + file_text(rng, A.lead_tokens)
        ap_ = [0] + [rng.randint(A.append_min, A.append_max) for _ in range(A.lead_turns - 1)]
        rp = [rng.randint(A.reply_min, A.reply_max) for _ in ap_]
        convs.append(Conv("lead", system, task, ap_, rp))
    # the append texts, drawn after the plan so a change of one knob moves as little else as possible
    for c in convs:
        c.append_text = [file_text(rng, a) if a else "" for a in c.appends]
    return convs


# ---- one streamed turn ---------------------------------------------------------------------------------------------------
def stream(messages, max_tokens):
    body = {"model": "m", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = {"max_tokens": max_tokens, "t0": time.time(), "t_first": None, "t1": None, "content": "", "usage": {}, "finish": None,
         "error": None}
    c = http.client.HTTPConnection(A.host, A.port, timeout=A.timeout)
    try:
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        resp = c.getresponse()
        if resp.status != 200:
            r["error"] = f"HTTP {resp.status}: {resp.read()[:300]!r}"
            return r
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if "error" in ev:
                r["error"] = json.dumps(ev["error"])[:300]
                break
            if ev.get("usage"):
                r["usage"] = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if (d.get("content") or d.get("reasoning_content")) and r["t_first"] is None:
                    r["t_first"] = time.time()
                r["content"] += d.get("content") or d.get("reasoning_content") or ""
                r["finish"] = ch.get("finish_reason") or r["finish"]
    except Exception as e:  # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {e}"
    finally:
        c.close()
        r["t1"] = time.time()
    return r


def run_conv(c, t_start):
    time.sleep(c.start_delay)
    msgs = [{"role": "system", "content": c.system}, {"role": "user", "content": c.task}]
    prev_prompt = 0
    for t, (txt, cap) in enumerate(zip(c.append_text, c.replies)):
        if t > 0:
            msgs.append({"role": "user", "content": "Tool output:\n```\n" + txt + "\n```\nContinue the task with this output."})
        r = stream(msgs, cap)
        u = r["usage"]
        p, cached, n = u.get("prompt_tokens", 0), (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0), u.get("completion_tokens", 0)
        ttft = (r["t_first"] or r["t1"]) - r["t0"]
        dspan = r["t1"] - (r["t_first"] or r["t1"])
        rec = {"conv": c.name, "turn": t, "t_start": r["t0"] - t_start, "t_end": r["t1"] - t_start, "prompt": p, "cached": cached,
               "new": max(0, p - cached), "completion": n, "max_tokens": cap, "finish": r["finish"], "ttft_s": round(ttft, 3),
               "prefill_tps": round((p - cached) / ttft, 1) if ttft > 0 and p > cached else None,
               "decode_tps": round((n - 1) / dspan, 2) if dspan > 0.05 and n > 1 else None,
               "latency_s": round(r["t1"] - r["t0"], 3), "error": r["error"],
               # P4 B29: a follow-up restored its conversation = it cached at least the previous turn's prompt end (the
               # server snapshots at the stable boundary = the prompt minus its generation prompt, a few tokens; 32 of slack)
               "restored_prev": (cached >= prev_prompt - 32) if t > 0 and prev_prompt else None}
        prev_prompt = p
        c.reqs.append(rec)
        print(f"[{rec['t_end']:7.1f}s] {c.name} turn {t}: prompt {p} ({cached} cached) TTFT {ttft:.1f}s "
              f"prefill {rec['prefill_tps']} tok/s, {n} tok decode {rec['decode_tps']} tok/s, {rec['latency_s']:.1f}s"
              + (f" ERROR {r['error']}" if r["error"] else ""), flush=True)
        if r["error"]:
            return
        msgs.append({"role": "assistant", "content": r["content"]})


def run_group(convs):
    t_start = time.time()
    ths = [threading.Thread(target=run_conv, args=(c, t_start)) for c in convs]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    return t_start, time.time()


class Health:
    """/health at 1 Hz: the lanes block's tokens / decoding / lanes_active / prefill_queue / paused_ms (absent on other archs)."""
    def __init__(self):
        self.s, self.stop = [], threading.Event()
        self.th = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                c = http.client.HTTPConnection(A.host, A.port, timeout=5)
                c.request("GET", "/health")
                h = json.loads(c.getresponse().read() or b"{}")
                c.close()
            except Exception as e:  # noqa: BLE001
                h = {"error": str(e)}
            h["t"] = time.time()
            self.s.append(h)
            self.stop.wait(1.0)

    def rate(self, t0, t1):
        s = [h for h in self.s if t0 <= h["t"] <= t1 and isinstance(h.get("tokens"), int)]
        if len(s) < 2 or s[-1]["t"] <= s[0]["t"]:
            return None
        return round((s[-1]["tokens"] - s[0]["tokens"]) / (s[-1]["t"] - s[0]["t"]), 2)


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else None


def summarize(name, convs, t0, t1, health):
    reqs = [r for c in convs for r in c.reqs]
    wall = t1 - t0
    comp = sum(r["completion"] for r in reqs)
    new = sum(r["new"] for r in reqs)
    s = {"group": name, "wall_s": round(wall, 1), "requests": len(reqs), "errors": sum(1 for r in reqs if r["error"]),
         "completion_tokens": comp, "prefilled_new_tokens": new, "cached_tokens": sum(r["cached"] for r in reqs),
         "agg_decode_tps_client": round(comp / wall, 2) if wall > 0 else None,
         "agg_prefill_tps_client": round(new / wall, 1) if wall > 0 else None,
         "server_tokens_rate": health.rate(t0, t1),
         "decode_tps_p50": pct([r["decode_tps"] for r in reqs], 0.5), "decode_tps_p5": pct([r["decode_tps"] for r in reqs], 0.05),
         "prefill_tps_p50": pct([r["prefill_tps"] for r in reqs], 0.5),
         "latency_p50_s": pct([r["latency_s"] for r in reqs], 0.5), "latency_p95_s": pct([r["latency_s"] for r in reqs], 0.95),
         "ttft_p50_s": pct([r["ttft_s"] for r in reqs], 0.5), "ttft_p95_s": pct([r["ttft_s"] for r in reqs], 0.95),
         "followups": sum(1 for r in reqs if r["restored_prev"] is not None),
         "followups_restored_prev": sum(1 for r in reqs if r["restored_prev"]),
         "per_conv": {}}
    for c in convs:
        cr = c.reqs
        dspan = sum(r["latency_s"] - r["ttft_s"] for r in cr)
        s["per_conv"][c.name] = {"turns": len(cr), "completion": sum(r["completion"] for r in cr), "new": sum(r["new"] for r in cr),
                                 "cached": sum(r["cached"] for r in cr),
                                 "decode_tps": round(sum(r["completion"] for r in cr) / dspan, 2) if dspan > 0 else None,
                                 "busy_s": round(sum(r["latency_s"] for r in cr), 1)}
    print(f"== {name}: {s['requests']} requests in {s['wall_s']} s; decode {comp} tok = {s['agg_decode_tps_client']} tok/s aggregate "
          f"(server /health {s['server_tokens_rate']}), per request p50 {s['decode_tps_p50']} p5 {s['decode_tps_p5']} tok/s; "
          f"prefill {new} new tok = {s['agg_prefill_tps_client']} tok/s aggregate, p50 {s['prefill_tps_p50']} tok/s per request; "
          f"cached {s['cached_tokens']} tok, {s['followups_restored_prev']} of {s['followups']} follow-up(s) restored the previous turn; "
          f"latency p50 {s['latency_p50_s']} p95 {s['latency_p95_s']} s; TTFT p50 {s['ttft_p50_s']} p95 {s['ttft_p95_s']} s; "
          f"errors {s['errors']}", flush=True)
    return s


def run_needles():
    rng = random.Random(A.seed)
    res, miss = [], 0
    for n in [int(x) for x in A.needle_tokens.split(",") if x]:
        hay = file_text(rng, n)
        for d in [float(x) for x in A.needle_depths.split(",") if x]:
            tag, code = f"site-{rng.randint(100, 999)}", str(rng.randint(10_000_000, 99_999_999))
            cut = hay.rfind("\n", 0, int(len(hay) * d)) + 1
            # (a per-request first line: the prompt cache shares only the short system turn, so each needle is a cold prefill)
            text = f"Request {tag}-{code[:3]}.\n" + hay[:cut] + f"The vault code for {tag} is {code}.\n" + hay[cut:]
            msgs = [{"role": "system", "content": "You read files carefully and answer exactly."},
                    {"role": "user", "content": text + f"\n\nWhat is the vault code for {tag}? Answer with the number only."}]
            r = stream(msgs, 32)
            u = r["usage"]
            ok = r["error"] is None and code in r["content"]
            miss += not ok
            rec = {"tokens": n, "depth": d, "prompt": u.get("prompt_tokens"),
                   "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                   "ttft_s": round((r["t_first"] or r["t1"]) - r["t0"], 2), "code": code, "reply": r["content"][:80], "ok": ok,
                   "error": r["error"]}
            res.append(rec)
            print(f"{'PASS' if ok else 'MISS'} needle {n} tok at {d:.2f}: prompt {rec['prompt']} ({rec['cached']} cached), "
                  f"TTFT {rec['ttft_s']} s, want {code}, got {rec['reply']!r}" + (f" ERROR {r['error']}" if r["error"] else ""),
                  flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    with open(A.out, "w") as f:
        json.dump({"args": vars(A), "needles": res}, f, indent=1)
    print(f"needles: {len(res) - miss} / {len(res)} found; wrote {A.out}")
    return 1 if miss else 0


def main():
    if A.scenario == "needle":
        return run_needles()
    convs = plan()
    if A.dry_run:
        for c in convs:
            total = est(c.system) + est(c.task)
            line = []
            for a, r in zip(c.appends, c.replies):
                total += a
                line.append(f"{total}+{r}")
                total += r
            print(f"{c.name}: system ~{est(c.system)} task ~{est(c.task)}; turns (est. prompt+reply): {' '.join(line)}")
        return 0
    health = Health()
    health.th.start()
    groups = []
    order = []
    if A.scenario in ("workers", "both"):
        order.append(("workers", [c for c in convs if c.name.startswith("worker")]))
    if A.scenario in ("wave", "both"):
        order.append(("wave", [c for c in convs if c.name.startswith("wave")]))
    lead = [c for c in convs if c.name == "lead"]
    for i, (name, cs) in enumerate(order):
        cs = cs + (lead if i == 0 else [])   # the lead runs beside the first group
        t0, t1 = run_group(cs)
        groups.append(summarize(name + ("+lead" if i == 0 and lead else ""), cs, t0, t1, health))
    health.stop.set()
    out = {"args": vars(A), "groups": groups, "requests": [r for c in convs for r in c.reqs], "health": health.s}
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    with open(A.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {A.out}")
    return 1 if any(g["errors"] for g in groups) else 0


if __name__ == "__main__":
    sys.exit(main())
