#!/usr/bin/env python3
"""#29 (docs/server_idle_spin_watchdog_2026-09-24.md): disconnect clients mid-generation, then watch the idle server.

On 2026-09-19 `ie serve` burned ~900 % CPU for eight hours with nothing in flight after a client disconnected
mid-generation. One deliberate repro did not spin. This probe repeats the disconnect many ways and then checks that the
server goes quiet. Stdlib only; talks to one server over loopback.

Each round opens a raw socket, POSTs /v1/chat/completions (stream=true by default), reads the reply until `--after` SSE
`data:` events have arrived (or `--after-s` seconds for --nonstream), then closes the socket: odd rounds with an orderly
FIN, even rounds with an RST (SO_LINGER 0), because the two reach the server differently. The server should log
"[req] stream client disconnected mid-generation" and free the slot. The probe waits for /health to show inflight 0 before
the next round. After the last round it stays idle for `--idle` seconds (default 90: longer than the watchdog's 60 s
window) and reports the server process's CPU rate over that wait (from /proc/<pid>/stat), its thread count before and
after, and /health's idle_spin field.

PASS when the idle CPU rate is under --max-idle-cores (default 0.5), idle_spin.seconds is 0 (or absent) and every round
disconnected mid-generation (a reply that finished before the cut is reported, and fails the run).

  tools/serve_disconnect_probe.py --port 11441 --pid "$(pgrep -f 'build/src/[i]e serve')" --rounds 20
"""
import argparse
import http.client
import json
import os
import socket
import struct
import sys
import time

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--pid", type=int, required=True, help="the ie serve process (for /proc CPU and thread counts)")
ap.add_argument("--rounds", type=int, default=20)
ap.add_argument("--after", type=int, default=40, help="streaming: SSE data events to read before the cut")
ap.add_argument("--after-s", type=float, default=8.0, help="--nonstream: seconds to wait before the cut")
ap.add_argument("--nonstream", action="store_true", help="non-streaming requests, cut after --after-s seconds")
ap.add_argument("--max-tokens", type=int, default=2000)
ap.add_argument("--idle", type=float, default=90.0, help="seconds to watch the idle server after the last round")
ap.add_argument("--max-idle-cores", type=float, default=0.5)
ap.add_argument("--round-timeout", type=float, default=600.0)
args = ap.parse_args()

CLK = os.sysconf("SC_CLK_TCK")


def proc_cpu_ticks(pid):
    with open(f"/proc/{pid}/stat") as f:
        s = f.read()
    rest = s[s.rindex(")") + 2:].split()
    return int(rest[11]) + int(rest[12])            # utime + stime (fields 14 and 15)


def proc_threads(pid):
    return len(os.listdir(f"/proc/{pid}/task"))


def health():
    c = http.client.HTTPConnection(args.host, args.port, timeout=10)
    try:
        c.request("GET", "/health")
        return json.loads(c.getresponse().read())
    finally:
        c.close()


def wait_idle(limit_s):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        try:
            h = health()
            if h.get("inflight", 1) == 0 and h.get("queued", 1) == 0:
                return time.time() - t0, h
        except OSError:
            pass
        time.sleep(0.25)
    return None, None


def model_id():
    c = http.client.HTTPConnection(args.host, args.port, timeout=30)
    try:
        c.request("GET", "/v1/models")
        return json.loads(c.getresponse().read())["data"][0]["id"]
    finally:
        c.close()


def one_round(i, model):
    rst = i % 2 == 0
    body = json.dumps({
        "model": model, "stream": not args.nonstream, "max_tokens": args.max_tokens, "temperature": 0.7,
        "messages": [{"role": "user", "content":
                      f"Round {i}. Write a long, detailed essay (at least 1500 words) about the history of the "
                      "printing press, with many concrete examples. Do not stop early."}]}).encode()
    req = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {args.host}:{args.port}\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(body)}\r\nConnection: keep-alive\r\n\r\n").encode() + body
    s = socket.create_connection((args.host, args.port), timeout=args.round_timeout)
    t0 = time.time()
    s.sendall(req)
    got = b""; events = 0; finished = False
    try:
        if args.nonstream:
            s.settimeout(args.after_s)
            try:
                got = s.recv(65536)                  # a reply this early means the generation already finished
                finished = bool(got)
            except socket.timeout:
                pass
        else:
            while events < args.after:
                chunk = s.recv(65536)
                if not chunk:
                    finished = True; break
                got += chunk
                events = got.count(b"data: ")
                if b"[DONE]" in got:
                    finished = True; break
    finally:
        if rst:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        s.close()
    cut_s = time.time() - t0
    freed_s, h = wait_idle(args.round_timeout)
    status = got.split(b"\r\n", 1)[0].decode(errors="replace") if got else "(no bytes yet)"
    print(f"round {i:2d}: {'RST' if rst else 'FIN'} after {events if not args.nonstream else '-'} events / {cut_s:.1f} s; "
          f"{status}; {'FINISHED BEFORE THE CUT' if finished else 'cut mid-generation'}; "
          f"slot free {('%.1f s later' % freed_s) if freed_s is not None else 'NEVER (round timeout)'}", flush=True)
    return (not finished) and freed_s is not None


model = model_id()
thr0 = proc_threads(args.pid)
print(f"server pid {args.pid}, model {model}, {thr0} threads; {args.rounds} rounds, "
      f"{'non-streaming' if args.nonstream else 'streaming'}", flush=True)
ok_rounds = sum(one_round(i, model) for i in range(1, args.rounds + 1))
thr1 = proc_threads(args.pid)
c0, t0 = proc_cpu_ticks(args.pid), time.time()
time.sleep(args.idle)
c1, t1 = proc_cpu_ticks(args.pid), time.time()
cores = (c1 - c0) / CLK / (t1 - t0)
h = health()
spin = h.get("idle_spin")
thr2 = proc_threads(args.pid)
print(f"idle {t1 - t0:.0f} s after the last round: {cores:.3f} cores; threads {thr0} before, {thr1} after the rounds, "
      f"{thr2} after the wait; /health idle_spin {json.dumps(spin)}; inflight {h.get('inflight')} queued {h.get('queued')}")
passed = ok_rounds == args.rounds and cores < args.max_idle_cores and (spin is None or spin.get("seconds", 0) == 0)
print(f"DISCONNECT PROBE: {'PASS' if passed else 'FAIL'} ({ok_rounds}/{args.rounds} rounds cut mid-generation and freed; "
      f"idle rate {cores:.3f} vs {args.max_idle_cores} cores)")
sys.exit(0 if passed else 1)
