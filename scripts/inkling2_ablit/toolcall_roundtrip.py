#!/usr/bin/env python3
"""One tool-call round trip through `ie serve` (OpenAI chat completions): the model must call get_weather, gets the tool's
result back, and must answer from it. Prints the finish reasons, the call, and the final answer (neutral content).

  python toolcall_roundtrip.py --port P [--effort low]
"""
import argparse, json, urllib.request

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, required=True); ap.add_argument("--effort", default="low")
    a = ap.parse_args()
    msgs = [{"role": "user", "content": "What's the weather in Lisbon right now? Use the tool."}]
    base = {"model": "x", "tools": TOOLS, "temperature": 0, "max_tokens": 1024, "reasoning_effort": a.effort}
    r1 = post(a.port, dict(base, messages=msgs))
    m1 = r1["choices"][0]["message"]; f1 = r1["choices"][0]["finish_reason"]
    calls = m1.get("tool_calls") or []
    print("turn 1 finish:", f1, "| tool calls:", json.dumps([c["function"] for c in calls]))
    if not calls: print("RESULT: FAIL (no tool call)"); return 1
    msgs.append({"role": "assistant", "content": m1.get("content") or "", "tool_calls": calls})
    msgs.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"city": "Lisbon", "temp_c": 19, "sky": "light rain"})})
    r2 = post(a.port, dict(base, messages=msgs))
    m2 = r2["choices"][0]["message"]; f2 = r2["choices"][0]["finish_reason"]
    ans = m2.get("content") or ""
    print("turn 2 finish:", f2, "| answer:", json.dumps(ans[:300]))
    ok = calls[0]["function"]["name"] == "get_weather" and "lisbon" in calls[0]["function"]["arguments"].lower() and f2 == "stop" \
        and "19" in ans and "rain" in ans.lower()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__": raise SystemExit(main())
