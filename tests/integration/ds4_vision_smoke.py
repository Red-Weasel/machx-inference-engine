#!/usr/bin/env python3
"""End-to-end vision smoke against a running `ie serve` (OpenAI chat endpoint).

    python3 tests/integration/ds4_vision_smoke.py --port 11435 --image red.png --prompt "What colour is this image? Answer in one word." [--repeat 2]

Sends the image as a base64 data URL in an `image_url` content part, greedy
(temperature 0), prints the answer, token counts and timings; with --repeat
compares the answers for determinism. Text-only when --image is omitted.
Exit 0 = request(s) succeeded (and, with --expect, the expected substring was
found case-insensitively in every answer); 1 otherwise.
"""
import argparse
import base64
import json
import mimetypes
import sys
import time
import urllib.request


def ask(port, prompt, image, max_tokens, timeout, no_think=False):
    content = []
    for img in image or []:
        mime = mimetypes.guess_type(img)[0] or "image/png"
        b64 = base64.b64encode(open(img, "rb").read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    body = {"model": "local", "messages": [{"role": "user", "content": content if image else prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": False}
    if no_think:
        body["enable_thinking"] = False
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--image", action="append", help="image file; repeat for several images (in order)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--expect", help="substring that must appear (case-insensitive) in every answer")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--no-think", action="store_true", help="send enable_thinking=false (direct answer)")
    a = ap.parse_args()
    answers = []
    for i in range(a.repeat):
        try:
            out, dt = ask(a.port, a.prompt, a.image, a.max_tokens, a.timeout, a.no_think)
        except Exception as e:  # noqa: BLE001
            print(f"request failed: {e}")
            return 1
        ch = out["choices"][0]
        text = ch["message"].get("content") or ""
        usage = out.get("usage", {})
        answers.append(text)
        print(json.dumps({"run": i, "finish": ch.get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens"),
                          "completion_tokens": usage.get("completion_tokens"), "seconds": round(dt, 2),
                          "answer": text.strip()[:400]}, ensure_ascii=False))
    ok = True
    if a.repeat > 1:
        same = all(x == answers[0] for x in answers)
        print(f"deterministic: {same}")
        ok &= same
    if a.expect:
        hit = all(a.expect.lower() in x.lower() for x in answers)
        print(f"expect '{a.expect}': {hit}")
        ok &= hit
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
