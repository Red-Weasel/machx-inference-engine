#!/usr/bin/env python3
"""Soak traffic for a running `ie serve` (OpenAI chat route): the shapes a long agent session produces, for N minutes.

  conversation A   one long conversation that grows turn by turn (greedy); every --rewrite-every turns an EARLIER message
                   is rewritten in place (what an agent harness does when it trims a tool result) and every
                   --retouch-every turns the last assistant message is edited -- the cases a prompt cache must survive
  job B            the same medium prompt again and again on another lane (greedy): every reply must be byte-identical
  disconnects      a streamed request the client abandons after a few chunks
  bursts           a few short requests at once (the request lanes)

Fails (exit 1) on: a non-200 or empty reply, job B's reply changing, a new GPU fault / reset line in the kernel log
(journalctl -k), the server's RSS growing more than --rss-growth-gib, the server process gone. Prints a JSON summary.
The server is NOT started here: scripts/soak/soak.sh starts it with a real launch line and reads its log afterwards.

  python soak_serve.py --base-url http://127.0.0.1:11497 --minutes 60 --doc README.md [--pid SERVER_PID]
"""
import argparse, json, os, random, re, subprocess, sys, threading, time, urllib.request

def post(url, body, timeout=1800):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())

def stream_and_drop(url, body, chunks):
    body = dict(body, stream=True)
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data:"):
                n += 1
                if n >= chunks: break
    return n

def kernel_faults(since):
    try:
        out = subprocess.run(["journalctl", "-k", "--since", since, "--no-pager"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    return [l for l in out.splitlines() if re.search(r"xe .*(reset|fault)|i915.*(reset|hang|wedged)|GT\d: .*reset", l, re.I)]

def rss_gib(pid):
    try:
        for l in open(f"/proc/{pid}/status"):
            if l.startswith("VmRSS:"): return int(l.split()[1]) / 1048576.0
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--a-chars", type=int, default=60000, help="conversation A's first document, characters")
    ap.add_argument("--b-chars", type=int, default=45000, help="job B's prompt, characters")
    ap.add_argument("--a-max-tokens", type=int, default=150000, help="conversation A starts over above this prompt size")
    ap.add_argument("--rewrite-every", type=int, default=5)
    ap.add_argument("--retouch-every", type=int, default=7)
    ap.add_argument("--rss-growth-gib", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    doc = open(a.doc, encoding="utf-8").read()
    t_start = time.time(); since = time.strftime("%Y-%m-%d %H:%M:%S"); deadline = t_start + a.minutes * 60
    S = dict(turns_a=0, restarts_a=0, max_prompt_tokens=0, rewrites=0, retouches=0, runs_b=0, b_changed=0, disconnects=0, bursts=0, errors=[], a_seconds=[], b_seconds=[])
    lock = threading.Lock()
    def err(m):
        with lock: S["errors"].append(m)
        print("ERROR:", m, flush=True)
    rss0 = rss_gib(a.pid) if a.pid else None

    def conv_a():
        first = {"role": "user", "content": doc[:a.a_chars] + "\n\nSummarise the first section in one sentence."}
        msgs = [first]
        off = a.a_chars
        while time.time() < deadline:
            try:
                t0 = time.time()
                st, r = post(a.base_url, {"messages": msgs, "max_tokens": 96, "temperature": 0})
                m = r["choices"][0]["message"]; text = m.get("content") or ""
                if st != 200 or not (text or m.get("reasoning_content")): err(f"A turn {S['turns_a']}: empty reply"); return
                pt = int((r.get("usage") or {}).get("prompt_tokens") or 0)
                with lock: S["turns_a"] += 1; S["a_seconds"].append(round(time.time() - t0, 1)); S["max_prompt_tokens"] = max(S["max_prompt_tokens"], pt)
                if pt > a.a_max_tokens:      # a new conversation (the context has a limit; a real session compacts or starts over)
                    msgs = [dict(first, content=f"(conversation {S['restarts_a'] + 2}) " + first["content"])]; off = a.a_chars
                    with lock: S["restarts_a"] += 1
                    continue
                msgs.append({"role": "assistant", "content": text or "(no visible answer)"})
                piece = doc[off:off + 3000] or doc[:3000]; off += 3000
                msgs.append({"role": "user", "content": "Tool result:\n" + piece + "\n\nWhat changed in this part? One sentence."})
                n = S["turns_a"]
                if a.rewrite_every and n % a.rewrite_every == 0 and len(msgs) > 6:      # an earlier tool result trimmed in place
                    k = rng.randrange(2, len(msgs) - 3, 2)
                    msgs[k] = dict(msgs[k], content=msgs[k]["content"][:400] + "\n[trimmed]")
                    with lock: S["rewrites"] += 1
                elif a.retouch_every and n % a.retouch_every == 0:                      # the last assistant message edited
                    msgs[-2] = dict(msgs[-2], content="(edited) " + msgs[-2]["content"])
                    with lock: S["retouches"] += 1
            except Exception as x:
                err(f"A: {type(x).__name__}: {x}"); return

    def job_b():
        body = {"messages": [{"role": "user", "content": doc[-a.b_chars:] + "\n\nList three product names from the text."}], "max_tokens": 128, "temperature": 0}
        first = None
        while time.time() < deadline:
            try:
                t0 = time.time()
                st, r = post(a.base_url, body)
                m = json.dumps(r["choices"][0]["message"], sort_keys=True)
                with lock: S["runs_b"] += 1; S["b_seconds"].append(round(time.time() - t0, 1))
                if first is None: first = m
                elif m != first:
                    with lock: S["b_changed"] += 1
                    err(f"B run {S['runs_b']}: the reply to the same greedy prompt changed")
                time.sleep(rng.uniform(5, 25))
            except Exception as x:
                err(f"B: {type(x).__name__}: {x}"); return

    def chaos():
        while time.time() < deadline:
            time.sleep(rng.uniform(40, 90))
            if time.time() >= deadline: break
            try:
                if rng.random() < 0.5:
                    stream_and_drop(a.base_url, {"messages": [{"role": "user", "content": "Count from 1 to 400, one number a line."}], "max_tokens": 600, "temperature": 0.7}, rng.randrange(3, 30))
                    with lock: S["disconnects"] += 1
                else:
                    ths = [threading.Thread(target=lambda i=i: post(a.base_url, {"messages": [{"role": "user", "content": f"Give one fact about the number {i}."}], "max_tokens": 48, "temperature": 0.7})) for i in range(3)]
                    [t.start() for t in ths]; [t.join() for t in ths]
                    with lock: S["bursts"] += 1
            except Exception as x:
                err(f"chaos: {type(x).__name__}: {x}")

    ths = [threading.Thread(target=f, daemon=True) for f in (conv_a, job_b, chaos)]
    [t.start() for t in ths]
    while time.time() < deadline and any(t.is_alive() for t in ths[:2]) and not S["errors"]:
        time.sleep(10)
        if a.pid and rss_gib(a.pid) is None: err("the server process is gone"); break
        kf = kernel_faults(since)
        if kf: err(f"kernel log: {kf[0][:200]}"); break
    for t in ths: t.join(timeout=900)
    kf = kernel_faults(since)
    rss1 = rss_gib(a.pid) if a.pid else None
    S["kernel_fault_lines"] = len(kf) if kf is not None else None
    S["rss_gib"] = [rss0, rss1]
    if rss0 and rss1 and rss1 - rss0 > a.rss_growth_gib: err(f"server RSS grew {rss1 - rss0:.1f} GiB")
    if kf: err(f"{len(kf)} GPU fault / reset line(s) in the kernel log")
    S["minutes"] = round((time.time() - t_start) / 60, 1)
    S["ok"] = not S["errors"]
    S["a_seconds"] = S["a_seconds"][-200:]; S["b_seconds"] = S["b_seconds"][-200:]
    print(json.dumps(S, indent=1))
    return 0 if S["ok"] else 1

if __name__ == "__main__":
    sys.exit(main())
