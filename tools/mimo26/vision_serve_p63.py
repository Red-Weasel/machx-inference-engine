#!/usr/bin/env python3
"""P6.3 clauses (a) and (b) against a live `ie serve` of MiMo-V2.6 (docs/mimo26/00_PORT_PLAN.md).
  vision_serve_p63.py <port> <fixture_dir> <model_dir> <ie binary> [--out answers.json] [--expect-ready 0|1]
(a) each fixture as a PNG data URI and as a JPEG (quality 90, encoded here with Pillow): greedy answers recorded verbatim
    and checked for the things drawn (fixture 0: red circle, blue square, green triangle, "MiMo 42"; fixture 1: dark UI,
    sidebar, "Send" button, "Rocket build"; fixture 2: a diagonal/black line);
(b) `ie capabilities <model_dir>` reports features.vision true for MiMo; GET /props carries vision.ready (P11's field) --
    with --expect-ready the value must match (1: the tower is staged; 0: IE_MIMO26_VISION=0 with a reason).
Prints ok/FAIL per check; exit 1 on any FAIL.
"""
import base64
import http.client
import io
import json
import subprocess
import sys

from PIL import Image

port, fixtures, model_dir, ie_bin = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
expect_ready = int(sys.argv[sys.argv.index("--expect-ready") + 1]) if "--expect-ready" in sys.argv else None
QUESTION = "Describe this image in one sentence: the shapes, their colours, and any text."
EXPECT = {0: [["circle"], ["square"], ["triangle"], ["mimo 42", "mimo"]],
          1: [["dark"], ["sidebar", "side bar", "navigation"], ["send"], ["rocket", "log", "build"]],
          2: [["line", "stripe", "stroke", "streak"], ["black", "dark"]]}
fails = 0


def check(name, ok, evidence):
    global fails
    fails += not ok
    print(f"{'ok ' if ok else 'FAIL'} {name}: {evidence}")


def get(path):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("GET", path)
    return json.loads(c.getresponse().read())


def ask(uri, max_tokens=64):
    body = {"model": "m", "temperature": 0, "max_tokens": max_tokens, "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                                                      {"type": "text", "text": QUESTION}]}]}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read())
    if "error" in d or "choices" not in d:
        return {"error": json.dumps(d)[:300]}
    return {"content": d["choices"][0]["message"].get("content") or "", "usage": d.get("usage", {})}


# (b) readiness and the capability flag
props = get("/props")
vision = props.get("vision") if isinstance(props, dict) else None
check("/props carries vision.ready", isinstance(vision, dict) and isinstance(vision.get("ready"), bool), json.dumps(vision))
if expect_ready is not None and isinstance(vision, dict):
    check(f"/props vision.ready == {bool(expect_ready)}", vision.get("ready") is bool(expect_ready) and (bool(expect_ready) or bool(vision.get("reason"))),
          f"ready={vision.get('ready')} reason={vision.get('reason')!r} image_tokens={vision.get('image_tokens')}")
caps = subprocess.run([ie_bin, "capabilities", model_dir], capture_output=True, text=True, timeout=120)
try:
    flag = json.loads(caps.stdout).get("features", {}).get("vision")
except Exception:
    flag = f"unparseable: {caps.stdout[:120]!r} {caps.stderr[:120]!r}"
check("ie capabilities features.vision is true", flag is True, f"{flag!r}")

answers = {}
if expect_ready == 0:
    r = ask("data:image/png;base64," + base64.b64encode(open(f"{fixtures}/fixture_0.png", "rb").read()).decode())
    check("an image request is refused with the reason", "error" in r and "image input" in r["error"], r.get("error", r.get("content"))[:200])
else:
    for k in range(3):
        png = open(f"{fixtures}/fixture_{k}.png", "rb").read()
        buf = io.BytesIO()
        Image.open(io.BytesIO(png)).convert("RGB").save(buf, format="JPEG", quality=90)
        for fmt, data in (("png", png), ("jpeg", buf.getvalue())):
            r = ask(f"data:image/{fmt};base64," + base64.b64encode(data).decode())
            answers[f"{k}.{fmt}"] = r
            if "error" in r:
                check(f"fixture {k} {fmt} answers", False, r["error"])
                continue
            text = r["content"].lower()
            missing = [alts[0] for alts in EXPECT[k] if not any(a in text for a in alts)]
            check(f"fixture {k} {fmt} names what is drawn", not missing,
                  f"prompt {r['usage'].get('prompt_tokens')} tokens; missing {missing}; {r['content']!r}")
if out_path:
    json.dump({"props": props, "answers": answers}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
