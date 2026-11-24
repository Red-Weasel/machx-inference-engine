#!/usr/bin/env python3
"""Sampled replies through `ie serve --parallel N` (one short run: start the server, send a chunk, stop it).

  python sample_run.py --ie BIN --model DIR --card 0 --port P --prompts CHUNK.jsonl --seed S --out OUT.jsonl
                       [--temp 1.0] [--top-p 1.0] [--max-tokens 2048] [--parallel 12]

Each prompt is one chat request (default effort, thinking on) with the given temperature and seed (seed + prompt index,
so every prompt has its own stream). OUT lines: name, seed, content (-> "text"), reasoning, finish_reason, tokens.
The server is started with --cards CARD --gpus 1 and killed (its own PID only) when the chunk is done.
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser()
    for k in ("--ie", "--model", "--prompts", "--out"): ap.add_argument(k, required=True)
    ap.add_argument("--card", default="0"); ap.add_argument("--port", type=int, default=18741); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048); ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=4096); ap.add_argument("--log", default="")
    a = ap.parse_args()
    P = [json.loads(l) for l in open(a.prompts) if l.strip()]
    log = open(a.log or a.out + ".serve.log", "w")
    srv = subprocess.Popen([a.ie, "serve", a.model, "--cards", a.card, "--gpus", "1", "--ctx", str(a.ctx), "--port", str(a.port),
                            "--parallel", str(a.parallel)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{a.port}"
    try:
        t0 = time.time()
        while True:
            if srv.poll() is not None: sys.exit(f"server exited ({srv.returncode}); see {log.name}")
            try: urllib.request.urlopen(url + "/v1/models", timeout=2); break
            except Exception: time.sleep(1)
            if time.time() - t0 > 180: sys.exit("server not ready in 180 s")

        def one(i):
            p = P[i]
            body = {"model": "x", "messages": [{"role": "user", "content": p["text"]}], "max_tokens": a.max_tokens,
                    "temperature": a.temp, "top_p": a.top_p, "seed": a.seed * 100003 + i, "stream": False}
            req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=1800))
            ch = r["choices"][0]; m = ch["message"]
            return {"name": p["name"], "seed": a.seed, "text": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
                    "tool_calls": m.get("tool_calls") or [], "finish_reason": ch.get("finish_reason"),
                    "tokens": r.get("usage", {}).get("completion_tokens", 0)}
        with ThreadPoolExecutor(a.parallel) as ex: rows = list(ex.map(one, range(len(P))))
        with open(a.out, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        print(f"{a.out}: {len(rows)} replies, {sum(r['tokens'] for r in rows)} tokens, {time.time() - t0:.0f} s")
    finally:
        try: os.killpg(srv.pid, signal.SIGTERM); srv.wait(30)
        except Exception: os.killpg(srv.pid, signal.SIGKILL)


if __name__ == "__main__": main()
