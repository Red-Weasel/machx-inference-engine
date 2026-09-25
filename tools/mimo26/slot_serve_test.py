#!/usr/bin/env python3
"""Fix-list #70 (docs/mimo26/P7_FIX64_FIX70.md): the verifier-then-main pattern against a live `ie serve` of MiMo-V2.6.

  slot_serve_test.py <port> <text_file> [--main-chars N] [--sides K] [--side-chars A:B] [--shared-system CHARS]

1. MAIN turn 1: one long user message (the first N characters of text_file) and a question; streamed, thinking off,
   max_tokens 64. Reports its prompt tokens, time to the first content delta (TTFT) and total time.
2. SIDES: K requests, each a different slice of text_file (sizes spread over A..B characters), max_tokens 16 --
   Dream's verifier-style calls. Before #70 each one replaced the live state.
3. MAIN turn 2: turn 1's messages + its reply (content echoed) + a new question. Reports usage.cached_tokens and TTFT.
PASS when turn 2's cached tokens cover >= 90 % of turn 1's prompt. Run it A/B on one build: the server started with
IE_MIMO26_PROMPT_CACHE_GIB=0 (the live conversation only = the behaviour before #70), then with the default.
--shared-system CHARS: every request starts with the same system message of that many characters, so the sides share a
prefix with the main conversation (the branch rule must keep the main one before cutting it back to that prefix).
Standard library only; greedy (temperature 0).
"""
import http.client
import json
import sys
import time


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


port, text_file = int(sys.argv[1]), sys.argv[2]
main_chars = int(arg("--main-chars", "400000"))
n_sides = int(arg("--sides", "10"))
side_lo, side_hi = (int(x) for x in arg("--side-chars", "12000:32000").split(":"))
shared = int(arg("--shared-system", "0"))
text = open(text_file, encoding="utf-8", errors="replace").read()
need = main_chars + n_sides * side_hi + shared
if len(text) < need:
    sys.exit(f"{text_file}: {len(text)} characters, need {need}")
system = [{"role": "system", "content": text[len(text) - shared:]}] if shared else []   # from the end: no slice overlaps it


def stream(messages, max_tokens, timeout=3600):
    """One streamed chat request: (content, usage dict, ttft seconds, total seconds)."""
    body = {"model": "mimo", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "enable_thinking": False}   # (the server sends a usage chunk before [DONE] on every stream)
    t0 = time.time()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    content, usage, ttft = "", {}, None
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if "error" in ev:
            sys.exit(f"server error: {ev['error']}")
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            delta = ch.get("delta", {})
            if (delta.get("content") or delta.get("reasoning_content")) and ttft is None:
                ttft = time.time() - t0
            content += delta.get("content") or ""
    c.close()
    return content, usage, ttft if ttft is not None else time.time() - t0, time.time() - t0


def cached(u):
    return (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


main1 = system + [{"role": "user", "content": text[:main_chars] + "\n\nIn one sentence: what is the text above mostly about?"}]
reply, u1, ttft1, tot1 = stream(main1, 64)
print(f"main turn 1: {u1.get('prompt_tokens')} prompt tokens ({cached(u1)} cached), TTFT {ttft1:.1f} s, total {tot1:.1f} s")

at = main_chars
for k in range(n_sides):
    size = side_lo + (side_hi - side_lo) * k // max(1, n_sides - 1)
    msgs = system + [{"role": "user", "content": text[at:at + size] + "\n\nIs the passage above consistent? Answer yes or no."}]
    at += side_hi
    _, u, ttft, tot = stream(msgs, 16)
    print(f"side {k + 1}/{n_sides}: {u.get('prompt_tokens')} prompt tokens ({cached(u)} cached), TTFT {ttft:.1f} s")

main2 = main1 + [{"role": "assistant", "content": reply},
                 {"role": "user", "content": "Name one person or place the text mentions."}]
_, u2, ttft2, tot2 = stream(main2, 64)
p1, c2 = u1.get("prompt_tokens", 0), cached(u2)
print(f"main turn 2: {u2.get('prompt_tokens')} prompt tokens ({c2} cached), TTFT {ttft2:.1f} s, total {tot2:.1f} s")
ok = p1 > 0 and c2 >= 0.9 * p1
print(f"{'PASS' if ok else 'FAIL'}: turn 2 served {c2} of turn 1's {p1} prompt tokens from cache ({100.0 * c2 / max(1, p1):.1f} %); "
      f"TTFT {ttft2:.1f} s vs turn 1's {ttft1:.1f} s")
sys.exit(0 if ok else 1)
