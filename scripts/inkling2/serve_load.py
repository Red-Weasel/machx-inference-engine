#!/usr/bin/env python3
"""scripts/inkling2/serve_load.py — drive `ie serve <inkling2 dir> --parallel N` over HTTP (standard library only).

  serve_load.py URL same   [--n 8] [--tokens 128]       N different chat prompts, greedy: each at once with the others,
                                                         then each alone; prints PASS when every reply is byte-identical
  serve_load.py URL tput   --conc 1,2,4,8,12 [--tokens 256]   C concurrent streaming requests: total and per-request tok/s
  serve_load.py URL mixed  [--tokens 64]                 a long generation, short chats, a tool-call round trip, a client
                                                         that disconnects mid-stream, a request over the slot context
"""
import argparse, json, socket, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

TOPICS = ["the water cycle", "how a bicycle gear works", "why the sky is blue", "the history of the printing press",
          "how vaccines train the immune system", "what a hash table is", "how tides form", "the rules of chess openings",
          "photosynthesis", "how compilers optimise loops", "the causes of the French revolution", "how bees communicate",
          "plate tectonics", "how a refrigerator works", "the life of a star", "what inflation is"]


def post(url, body, stream=False, timeout=900):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat(url, msgs, tokens, stream=False, extra=None):
    body = {"model": "x", "messages": msgs, "max_tokens": tokens, "temperature": 0, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}
    if extra: body.update(extra)
    t0 = time.time()
    if not stream:
        with post(url, body) as r:
            j = json.load(r)
        return j, time.time() - t0, None
    first = None
    text = ""
    usage = None
    with post(url, body, stream=True) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            j = json.loads(d)
            if j.get("usage"): usage = j["usage"]
            for ch in j.get("choices", []):
                dl = ch.get("delta") or {}
                piece = dl.get("content") or ""
                if (piece or dl.get("reasoning_content")) and first is None: first = time.time() - t0
                text += piece
    return {"text": text, "usage": usage}, time.time() - t0, first


def reply_text(j):
    m = j["choices"][0]["message"]
    return (m.get("content") or "") + "|" + json.dumps(m.get("tool_calls") or [])


def cmd_same(a):
    prompts = [[{"role": "user", "content": f"Explain {TOPICS[i % len(TOPICS)]} in a few paragraphs."}] for i in range(a.n)]
    with ThreadPoolExecutor(a.n) as ex:
        together = list(ex.map(lambda p: chat(a.url, p, a.tokens)[0], prompts))
    alone = [chat(a.url, p, a.tokens)[0] for p in prompts]
    bad = 0
    for i, (x, y) in enumerate(zip(together, alone)):
        same = reply_text(x) == reply_text(y)
        bad += not same
        print(f"prompt {i:2d}: {x['usage']['completion_tokens']:4d} tokens together, {y['usage']['completion_tokens']:4d} alone: "
              f"{'identical' if same else 'DIFFERENT'}")
    print("RESULT (same):", "PASS" if bad == 0 else f"FAIL ({bad} of {a.n} differ)")
    return bad == 0


def cmd_tput(a):
    for c in [int(x) for x in a.conc.split(",")]:
        prompts = [[{"role": "user", "content": f"Write a long, detailed essay about {TOPICS[i % len(TOPICS)]}."}] for i in range(c)]
        t0 = time.time()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(lambda p: chat(a.url, p, a.tokens, stream=True, extra={"ignore_eos": True,
                                                                                     "stream_options": {"include_usage": True},
                                                                                     "chat_template_kwargs": {"enable_thinking": True}}), prompts))
        wall = time.time() - t0
        toks = [r[0]["usage"]["completion_tokens"] if r[0]["usage"] else 0 for r in res]
        per = [t / max(1e-9, r[1] - (r[2] or 0)) for t, r in zip(toks, res)]
        ttft = [r[2] or 0 for r in res]
        print(f"concurrent {c:2d}: total {sum(toks) / wall:7.1f} tok/s ({sum(toks)} tokens in {wall:6.1f} s); per request decode "
              f"min {min(per):5.1f} / mean {sum(per) / len(per):5.1f} / max {max(per):5.1f} tok/s; first token mean {sum(ttft) / len(ttft):5.2f} s",
              flush=True)


def cmd_mixed(a):
    ok = True
    results = {}
    def long_gen():
        j, t, f = chat(a.url, [{"role": "user", "content": "Write a very long story about a lighthouse keeper."}], 1024, stream=True,
                       extra={"stream_options": {"include_usage": True}})
        results["long"] = (j["usage"], t)
    def short(i):
        j, t, _ = chat(a.url, [{"role": "user", "content": f"In one sentence: what is {TOPICS[i]}?"}], 64)
        results[f"short{i}"] = (j["choices"][0]["finish_reason"], j["usage"]["completion_tokens"], t)
    def tool():
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        msgs = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
        j, _, _ = chat(a.url, msgs, 256, extra={"tools": tools})
        m = j["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        if not calls:
            results["tool"] = ("no tool call", j["choices"][0]["finish_reason"]); return
        msgs += [m, {"role": "tool", "tool_call_id": calls[0].get("id", "0"), "content": json.dumps({"temp_c": 14, "sky": "cloudy"})}]
        j2, _, _ = chat(a.url, msgs, 128, extra={"tools": tools})
        results["tool"] = (calls[0]["function"]["name"], calls[0]["function"]["arguments"], j2["choices"][0]["finish_reason"],
                           (j2["choices"][0]["message"].get("content") or "")[:80])
    def drop():
        body = {"model": "x", "messages": [{"role": "user", "content": "Count from 1 to 1000, one number per line."}],
                "max_tokens": 2000, "temperature": 0, "stream": True}
        r = post(a.url, body, stream=True)
        n = 0
        for line in r:
            if line.startswith(b"data:"): n += 1
            if n >= 20: break
        r.fp.raw._sock.shutdown(socket.SHUT_RDWR) if hasattr(r.fp, "raw") else None
        r.close()
        results["drop"] = f"closed after {n} chunks"
    def over():
        big = "word " * 40000
        try:
            j, _, _ = chat(a.url, [{"role": "user", "content": big}], 16)
            results["over"] = ("answered", j["choices"][0]["finish_reason"])
        except urllib.error.HTTPError as e:
            results["over"] = (e.code, e.read()[:200].decode(errors="replace"))
    th = [threading.Thread(target=long_gen), threading.Thread(target=tool), threading.Thread(target=drop), threading.Thread(target=over)]
    th += [threading.Thread(target=short, args=(i,)) for i in range(4)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    for k in sorted(results): print(f"{k:7s}: {results[k]}")
    with urllib.request.urlopen(a.url + "/health", timeout=10) as r: print("health :", r.read()[:400].decode())
    j, _, _ = chat(a.url, [{"role": "user", "content": "Say hello."}], 16)
    print("after  :", j["choices"][0]["finish_reason"], repr(j["choices"][0]["message"].get("content"))[:80])
    print(f"wall {time.time() - t0:.1f} s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("url"); p.add_argument("mode", choices=["same", "tput", "mixed"])
    p.add_argument("--n", type=int, default=8); p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--conc", default="1,2,4,8,12")
    a = p.parse_args()
    sys.exit(0 if {"same": cmd_same, "tput": cmd_tput, "mixed": cmd_mixed}[a.mode](a) in (None, True) else 1)
