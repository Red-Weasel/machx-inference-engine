#!/usr/bin/env python3
"""P6.2 gate clauses (c) and (d) against a live `ie serve` of MiMo-V2.6 (docs/mimo26/00_PORT_PLAN.md).
  vision_serve_test.py <port> <fixture_dir> [--mode cache|dflash] [--out answers.json]
Greedy (temperature 0, thinking off) image requests through /v1/chat/completions with data: URIs.
--mode cache (default), on one server:
  A1  fixture_0 + question              -> answer A1 (nothing cached)
  B1  fixture_1 + the same question      -> answer B1; usage cached_tokens must stop BEFORE the image (the text prefix only)
  A2  fixture_0 + question again         -> answer A2 == A1 (the image's ids differ from B's: A is recomputed from its image)
  B2  fixture_1 + question again         -> answer B2 == B1
  A3  fixture_0 + question again         -> A3 == A1 and cached_tokens == prompt_tokens - 1 (the whole prompt hit)
  T2  fixture_2 + question               -> a different-size image (4 tokens) answers at all
--mode dflash: answers for the three fixtures, written to --out; run once with the drafter on and once with
IE_MIMO26_DFLASH=0, then compare the two files (they must be identical: lossless speculation).
Every check prints ok/FAIL with its evidence; exit 1 on any FAIL.
"""
import base64
import http.client
import json
import sys

port, fixtures = int(sys.argv[1]), sys.argv[2]
mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "cache"
out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
QUESTION = "Describe this image in one sentence: the shapes, their colours, and any text."
fails = 0


def check(name, ok, evidence):
    global fails
    fails += not ok
    print(f"{'ok ' if ok else 'FAIL'} {name}: {evidence}")


def data_uri(k):
    with open(f"{fixtures}/fixture_{k}.png", "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def ask(k, max_tokens=48):
    body = {"model": "m", "temperature": 0, "max_tokens": max_tokens, "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_uri(k)}},
                                                      {"type": "text", "text": QUESTION}]}]}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    d = json.loads(r.read())
    if "error" in d or "choices" not in d:
        sys.exit(f"request for fixture {k} failed: {json.dumps(d)[:400]}")
    msg = d["choices"][0]["message"]
    usage = d.get("usage", {})
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    return {"content": msg.get("content") or "", "finish": d["choices"][0]["finish_reason"],
            "prompt_tokens": usage.get("prompt_tokens"), "cached": cached}


if mode == "cache":
    a1 = ask(0)
    print(f"    A1 prompt {a1['prompt_tokens']} cached {a1['cached']}: {a1['content']!r}")
    b1 = ask(1)
    print(f"    B1 prompt {b1['prompt_tokens']} cached {b1['cached']}: {b1['content']!r}")
    # the image starts after "<|im_start|>user\n<|vision_start|>": a handful of text tokens; a cache hit must not reach into it
    check("B1 cache stops before the image", b1["cached"] < 16, f"cached {b1['cached']} of {b1['prompt_tokens']} (text prefix only)")
    check("B1 answers", len(b1["content"].strip()) > 0 and b1["content"] != a1["content"], f"{b1['content']!r}")
    a2 = ask(0)
    check("A2 == A1 (A recomputed from its own image after B)", a2["content"] == a1["content"], f"cached {a2['cached']}; {a2['content']!r}")
    b2 = ask(1)
    check("B2 == B1", b2["content"] == b1["content"], f"cached {b2['cached']}; {b2['content']!r}")
    a3 = ask(0)
    check("A3 == A1", a3["content"] == a1["content"], f"{a3['content']!r}")
    a4 = ask(0)
    check("A4 hits the whole prompt", a4["cached"] == a4["prompt_tokens"] - 1, f"cached {a4['cached']} of {a4['prompt_tokens']}")
    # the replay computes the last prompt row alone (a one-row forward) where A1 computed it inside a 328-row prefill:
    # the engine's batched and single-row kernels differ at fp16 scale, so a near-tie can resolve differently
    div = next((i for i in range(min(len(a1["content"]), len(a4["content"]))) if a1["content"][i] != a4["content"][i]),
               None if a1["content"] == a4["content"] else min(len(a1["content"]), len(a4["content"])))
    check("A4 == A1 (cached replay)", a4["content"] == a1["content"],
          "identical" if div is None else f"differs from char {div}: A1 ...{a1['content'][max(0, div - 30):div + 40]!r} | A4 ...{a4['content'][max(0, div - 30):div + 40]!r}")
    t2 = ask(2)
    check("T2 (4-token image) answers", len(t2["content"].strip()) > 0, f"prompt {t2['prompt_tokens']}: {t2['content']!r}")
    answers = {"0": a1, "1": b1, "2": t2}
else:
    answers = {str(k): ask(k, 64) for k in range(3)}
    for k, a in answers.items():
        print(f"    fixture {k}: prompt {a['prompt_tokens']} finish {a['finish']}: {a['content']!r}")
if out_path:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=1)
    print(f"answers -> {out_path}")
print("PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
