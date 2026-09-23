#!/usr/bin/env python3
"""P3b gate (a)/(b)/(d) against a live `ie serve` of MiMo-V2.6 (docs/mimo26/00_PORT_PLAN.md).

  serve_test.py <port> <server pid> [--requests 20]

Checks, each printed ok/FAIL with its evidence:
  chat       non-streaming, thinking off: 17 * 23 -> content contains 391
  stream     streaming, thinking on: reasoning_content deltas, then content deltas, finish_reason stop
  tool_call  a get_weather tool: finish_reason tool_calls, name get_weather, arguments.city mentions Paris
  tool_turn  the tool result fed back: the final content uses it (18)
  cancel     a stream closed after 5 chunks; the next request still answers
  reuse      turn 2 extends turn 1 (its answer + a new question): usage cached_tokens > 0
  memory     RssAnon of the server over N short requests: the growth after the first 3 (warm-up)
  props      GET /props carries "vision": {"ready": bool, "reason": str} -- the readiness of THIS load (P11)
"""
import http.client
import json
import sys
import time

port, pid = int(sys.argv[1]), int(sys.argv[2])
n_mem = int(sys.argv[sys.argv.index("--requests") + 1]) if "--requests" in sys.argv else 20
fails = 0


def post(body, stream=False, timeout=900):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if not stream:
        data = json.loads(r.read())
        c.close()
        return data
    return c, r


def sse(r, max_events=None):
    n = 0
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            return
        yield json.loads(line[6:])
        n += 1
        if max_events and n >= max_events:
            return


def check(name, ok, evidence):
    global fails
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {evidence}", flush=True)


def rss_anon():
    for l in open(f"/proc/{pid}/status"):
        if l.startswith("RssAnon:"):
            return int(l.split()[1]) // 1024
    return -1


# chat
d = post({"model": "m", "messages": [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
          "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, "enable_thinking": False})
msg = d["choices"][0]["message"]
check("chat", "391" in (msg.get("content") or ""), f"content={msg.get('content')!r} finish={d['choices'][0]['finish_reason']} usage={d.get('usage')}")

# stream with thinking
c, r = post({"model": "m", "messages": [{"role": "user", "content": "In one sentence: why does ice float on water?"}],
             "max_tokens": 600, "temperature": 0, "stream": True}, stream=True)
reason, content, fin = "", "", None
for ev in sse(r):
    ch = ev.get("choices", [{}])[0] if ev.get("choices") else {}
    dl = ch.get("delta", {})
    reason += dl.get("reasoning_content") or ""
    content += dl.get("content") or ""
    fin = ch.get("finish_reason") or fin
c.close()
check("stream", bool(reason.strip()) and bool(content.strip()) and fin == "stop",
      f"reasoning {len(reason)} chars, content={content.strip()[:100]!r}, finish={fin}")

# tool call
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
q = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
d = post({"model": "m", "messages": q, "tools": tools, "max_tokens": 600, "temperature": 0})
ch = d["choices"][0]
calls = ch["message"].get("tool_calls") or []
args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
check("tool_call", ch["finish_reason"] == "tool_calls" and calls and calls[0]["function"]["name"] == "get_weather" and "paris" in json.dumps(args).lower(),
      f"finish={ch['finish_reason']} calls={calls} content={ch['message'].get('content')!r}")

# tool result turn
if calls:
    turn = q + [{"role": "assistant", "content": ch["message"].get("content") or "", "tool_calls": calls,
                 "reasoning_content": ch["message"].get("reasoning_content") or ""},
                {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "content": json.dumps({"temp_c": 18, "sky": "clear"})}]
    d = post({"model": "m", "messages": turn, "tools": tools, "max_tokens": 600, "temperature": 0})
    m2 = d["choices"][0]["message"].get("content") or ""
    check("tool_turn", "18" in m2, f"content={m2.strip()[:160]!r} finish={d['choices'][0]['finish_reason']}")
else:
    check("tool_turn", False, "no tool call to answer")

# cancel
c, r = post({"model": "m", "messages": [{"role": "user", "content": "Count from 1 to 300, one number per line."}],
             "max_tokens": 1200, "temperature": 0, "stream": True, "enable_thinking": False}, stream=True)
got = list(sse(r, max_events=5))
c.close()
time.sleep(2)
d = post({"model": "m", "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 16, "temperature": 0, "enable_thinking": False})
check("cancel", len(got) == 5 and bool(d["choices"][0]["message"].get("content")), f"{len(got)} events before close; next: {d['choices'][0]['message'].get('content')!r}")

# prefix reuse
t1 = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Name a prime number between 10 and 20."}]
d1 = post({"model": "m", "messages": t1, "max_tokens": 32, "temperature": 0, "enable_thinking": False})
a1 = d1["choices"][0]["message"].get("content") or ""
t2 = t1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "And one between 20 and 30?"}]
d2 = post({"model": "m", "messages": t2, "max_tokens": 32, "temperature": 0, "enable_thinking": False})
cached = d2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
check("reuse", cached > 0, f"turn 2: prompt {d2['usage']['prompt_tokens']} tokens, cached {cached}; answer {d2['choices'][0]['message'].get('content')!r}")

# memory
samples = []
for i in range(n_mem):
    post({"model": "m", "messages": [{"role": "user", "content": f"Reply with the number {i}."}], "max_tokens": 12, "temperature": 0, "enable_thinking": False})
    samples.append(rss_anon())
growth = samples[-1] - samples[min(2, len(samples) - 1)]
check("memory", growth < 64, f"RssAnon MiB over {n_mem} requests: {samples[:3]} ... {samples[-3:]} (growth after warm-up {growth} MiB)")
c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
c.request("GET", "/props")
props = json.loads(c.getresponse().read())
vision = props.get("vision") if isinstance(props, dict) else None
check("props", isinstance(vision, dict) and isinstance(vision.get("ready"), bool) and isinstance(vision.get("reason"), str)
      and (vision["ready"] or bool(vision["reason"])), f"vision={vision}")
print("PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
