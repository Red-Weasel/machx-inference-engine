#!/usr/bin/env python3
"""#48 (docs/deepseek41/103): one chat request with a long system prompt, and what the prompt cache served it.

The system prompt is the first --chars characters of --text (by default this repository's
src/model/deepseek41_forward.cpp), so the V4.1 server writes a disk entry at the first-user boundary when that prefix is
4,096 tokens or more (Phase 47). Prints one JSON line: prompt tokens, cached tokens, finish reason, wall time. Run it in
a fresh server process after a rebuild: a disk entry that survived shows as cached tokens covering the system prefix.

  tools/ds41_prompt_cache_probe.py --port 11441 --min-cached 4096     # PASS iff at least 4,096 tokens came from cache
  tools/ds41_prompt_cache_probe.py --port 11441 --max-cached 0        # PASS iff nothing did
Stdlib only.
"""
import argparse
import http.client
import json
import os
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--text", default=os.path.join(here, "..", "src", "model", "deepseek41_forward.cpp"))
ap.add_argument("--chars", type=int, default=40000)
ap.add_argument("--question", default="In one sentence: what does the code above do?")
ap.add_argument("--max-tokens", type=int, default=16)
ap.add_argument("--min-cached", type=int, default=None)
ap.add_argument("--max-cached", type=int, default=None)
args = ap.parse_args()

with open(args.text, encoding="utf-8", errors="replace") as f:
    system = f.read(args.chars)
c = http.client.HTTPConnection(args.host, args.port, timeout=3600)
c.request("GET", "/v1/models")
model = json.loads(c.getresponse().read())["data"][0]["id"]
body = {"model": model, "stream": False, "max_tokens": args.max_tokens, "temperature": 0,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": args.question}]}
t0 = time.time()
c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
r = c.getresponse()
raw = r.read()
wall = time.time() - t0
if r.status != 200:
    print(json.dumps({"http_status": r.status, "body": raw[:400].decode(errors="replace")}))
    sys.exit(2)
d = json.loads(raw)
u = d.get("usage", {})
cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
out = {"prompt_tokens": u.get("prompt_tokens"), "cached_tokens": cached, "completion_tokens": u.get("completion_tokens"),
       "finish_reason": d["choices"][0].get("finish_reason"), "wall_s": round(wall, 2)}
ok = (args.min_cached is None or cached >= args.min_cached) and (args.max_cached is None or cached <= args.max_cached)
out["verdict"] = "PASS" if ok else "FAIL"
print(json.dumps(out))
sys.exit(0 if ok else 1)
