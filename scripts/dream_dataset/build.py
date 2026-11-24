#!/usr/bin/env python3
"""Build a fine-tuning dataset from Dream's raw model-call traces (read-only on Dream's files).

  build.py --out DIR [--roots ~/Desktop ~] [--keep strict|loose|all] [--local-only] [--scrub]

Source: every project's .dream/raw/<session>/<id>.{meta.json,request.json,response.bin}. One example per model call: the
request (system, tools, the conversation so far) is the context, the decoded reply is the training target ("last turn only":
the history's assistant turns are context, not targets -- the harness may have trimmed or rewritten them).

Every wire format Dream records is read: chat/completions (the engine, local models; an SSE delta stream), OpenAI Responses
(Codex / GPT / Grok; the response.completed event holds the whole reply), Anthropic Messages (Claude) and Google
streamGenerateContent (Gemini). Every example carries "source_model"
and "provider", so variants come from one build (all models / --local-only). Quality flags per example:
  complete        the stream reached its end (finish_reason / response.completed); Dream's meta status is not reliable
  finish          stop / tool_calls / length / ...
  bad_tool_json   a call's arguments are not a JSON object
  unknown_tool    a call names a tool the request did not offer (not set when the request offers Dream's dispatcher
                  tool_call / tool_schema: Dream lists its other tools in the prompt and runs them by name)
  repetition      the reply repeats one 40-character chunk 6+ times
  tool_error_next the next call of the session shows an error result for one of this reply's calls
--keep strict (default): complete, finish stop/tool_calls, no flag set; loose: complete and not length; all: everything.
--scrub replaces home paths, e-mail addresses, IPv4 addresses and key-like tokens (for anything published).
Output: DIR/dataset.jsonl ({"source_model","provider","session","id","created_at","tools","messages","flags"}) and
DIR/stats.json. docs/research/2026-11-20-dream-dataset-howto.md is the guide.
"""
import argparse, collections, glob, gzip, json, os, re, sys

LOCAL_PROVIDERS = {"machx"}


def jload(path):
    with open(path, "rb") as f:
        b = f.read()
    if b[:2] == b"\x1f\x8b": b = gzip.decompress(b)
    return b


# ------------------------------------------------------------------------------------------------ chat/completions
def decode_chat_stream(b):
    """SSE deltas (or one JSON body) -> (assistant message, finish_reason, complete)."""
    content, reasoning, calls, finish, done = [], [], {}, None, False
    if b.lstrip()[:1] == b"{":
        try:
            j = json.loads(b)
            ch = j["choices"][0]; m = ch.get("message", {})
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("reasoning_content"): msg["reasoning_content"] = m["reasoning_content"]
            if m.get("tool_calls"): msg["tool_calls"] = m["tool_calls"]
            return msg, ch.get("finish_reason"), ch.get("finish_reason") is not None
        except (ValueError, KeyError, IndexError):
            return None, None, False
    for line in b.split(b"\n"):
        if not line.startswith(b"data: "): continue
        d = line[6:].strip()
        if d == b"[DONE]": done = True; continue
        try: j = json.loads(d)
        except ValueError: continue
        for ch in j.get("choices") or []:
            dl = ch.get("delta") or {}
            if dl.get("content"): content.append(dl["content"])
            if dl.get("reasoning_content"): reasoning.append(dl["reasoning_content"])
            for tc in dl.get("tool_calls") or []:
                k = tc.get("index", 0)
                c = calls.setdefault(k, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"): c["id"] = tc["id"]
                f = tc.get("function") or {}
                if f.get("name"): c["function"]["name"] += f["name"]
                if f.get("arguments"): c["function"]["arguments"] += f["arguments"]
            if ch.get("finish_reason"): finish = ch["finish_reason"]
    msg = {"role": "assistant", "content": "".join(content)}
    if reasoning: msg["reasoning_content"] = "".join(reasoning)
    if calls: msg["tool_calls"] = [calls[k] for k in sorted(calls)]
    return msg, finish, finish is not None or done


# ------------------------------------------------------------------------------------------------ OpenAI Responses
def _text(parts):
    if isinstance(parts, str): return parts
    return "".join(p.get("text", "") for p in parts or [] if isinstance(p, dict))


def responses_items_to_messages(items):
    """Responses input / output items -> chat messages (function calls joined into one assistant turn)."""
    msgs, pending_reasoning = [], []
    def assistant():
        if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1].get("_closed"): return msgs[-1]
        m = {"role": "assistant", "content": ""}
        if pending_reasoning: m["reasoning_content"] = "\n".join(pending_reasoning); pending_reasoning.clear()
        msgs.append(m); return m
    for it in items or []:
        t = it.get("type") or ("message" if "role" in it else None)
        if t == "message":
            role = it.get("role", "user")
            if role == "assistant":
                a = assistant(); a["content"] += _text(it.get("content"))
            else:
                if msgs and msgs[-1]["role"] == "assistant": msgs[-1]["_closed"] = True
                msgs.append({"role": "system" if role in ("system", "developer") else role, "content": _text(it.get("content"))})
        elif t == "reasoning":
            s = _text(it.get("summary")) or _text(it.get("content"))
            if s: pending_reasoning.append(s)
        elif t == "function_call":
            a = assistant()
            a.setdefault("tool_calls", []).append({"id": it.get("call_id", ""), "type": "function",
                                                   "function": {"name": it.get("name", ""), "arguments": it.get("arguments", "")}})
        elif t == "function_call_output":
            if msgs and msgs[-1]["role"] == "assistant": msgs[-1]["_closed"] = True
            out = it.get("output")
            msgs.append({"role": "tool", "tool_call_id": it.get("call_id", ""), "content": out if isinstance(out, str) else json.dumps(out)})
    for m in msgs: m.pop("_closed", None)
    return msgs


def decode_responses_stream(b):
    for line in b.split(b"\n"):
        if not line.startswith(b"data: "): continue
        try: j = json.loads(line[6:])
        except ValueError: continue
        if j.get("type") == "response.completed":
            r = j.get("response") or {}
            ms = responses_items_to_messages(r.get("output"))
            msg = ms[-1] if ms and ms[-1]["role"] == "assistant" else {"role": "assistant", "content": ""}
            finish = "tool_calls" if msg.get("tool_calls") else ("length" if (r.get("incomplete_details") or {}).get("reason") == "max_output_tokens" else "stop")
            return msg, finish, True
    return None, None, False


def responses_request(rq):
    msgs = []
    if rq.get("instructions"): msgs.append({"role": "system", "content": rq["instructions"]})
    msgs += responses_items_to_messages(rq.get("input"))
    tools = []
    for t in rq.get("tools") or []:
        if t.get("type") == "function":
            p = t.get("parameters")
            if isinstance(p, str):
                try: p = json.loads(p)
                except ValueError: pass
            tools.append({"type": "function", "function": {"name": t.get("name", ""), "description": t.get("description", ""), "parameters": p}})
    return msgs, tools



# ------------------------------------------------------------------------------------------------ Anthropic Messages
def _blocks(c):
    return [{"type": "text", "text": c}] if isinstance(c, str) else (c or [])


def anthropic_request(rq):
    msgs = []
    sysm = rq.get("system")
    if sysm: msgs.append({"role": "system", "content": sysm if isinstance(sysm, str) else _text(sysm)})
    for m in rq.get("messages") or []:
        text, calls, reasoning = [], [], []
        for b in _blocks(m.get("content")):
            t = b.get("type")
            if t == "text": text.append(b.get("text", ""))
            elif t == "thinking": reasoning.append(b.get("thinking", ""))
            elif t == "tool_use":
                calls.append({"id": b.get("id", ""), "type": "function", "function": {"name": b.get("name", ""), "arguments": json.dumps(b.get("input", {}))}})
            elif t == "tool_result":
                c = b.get("content"); msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""), "content": c if isinstance(c, str) else _text(c)})
        if m.get("role") == "assistant":
            a = {"role": "assistant", "content": "".join(text)}
            if reasoning: a["reasoning_content"] = "".join(reasoning)
            if calls: a["tool_calls"] = calls
            msgs.append(a)
        elif text: msgs.append({"role": "user", "content": "".join(text)})
    tools = [{"type": "function", "function": {"name": t.get("name", ""), "description": t.get("description", ""), "parameters": t.get("input_schema")}}
             for t in rq.get("tools") or [] if t.get("name")]
    return msgs, tools


def decode_anthropic_stream(b):
    blocks, stop, done = {}, None, False
    for line in b.split(b"\n"):
        if not line.startswith(b"data: "): continue
        try: j = json.loads(line[6:])
        except ValueError: continue
        t = j.get("type")
        if t == "content_block_start": blocks[j.get("index", 0)] = dict(j.get("content_block") or {}, _json="")
        elif t == "content_block_delta":
            bl = blocks.setdefault(j.get("index", 0), {"type": "text"}); d = j.get("delta") or {}
            if d.get("type") == "text_delta": bl["text"] = bl.get("text", "") + d.get("text", "")
            elif d.get("type") == "thinking_delta": bl["thinking"] = bl.get("thinking", "") + d.get("thinking", "")
            elif d.get("type") == "input_json_delta": bl["_json"] = bl.get("_json", "") + d.get("partial_json", "")
        elif t == "message_delta": stop = (j.get("delta") or {}).get("stop_reason") or stop
        elif t == "message_stop": done = True
    if not blocks and not done: return None, None, False
    msg = {"role": "assistant", "content": ""}; calls = []
    for k in sorted(blocks):
        bl = blocks[k]
        if bl.get("type") == "text": msg["content"] += bl.get("text", "")
        elif bl.get("type") == "thinking": msg["reasoning_content"] = msg.get("reasoning_content", "") + bl.get("thinking", "")
        elif bl.get("type") == "tool_use":
            calls.append({"id": bl.get("id", ""), "type": "function", "function": {"name": bl.get("name", ""), "arguments": bl.get("_json") or "{}"}})
    if calls: msg["tool_calls"] = calls
    finish = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop"}.get(stop, stop)
    return msg, finish, done


# ------------------------------------------------------------------------------------------------ Google Gemini
def gemini_request(rq):
    r = rq.get("request", rq); msgs = []
    si = r.get("systemInstruction")
    if si: msgs.append({"role": "system", "content": _text((si or {}).get("parts"))})
    n = 0; pending = collections.deque()
    for c in r.get("contents") or []:
        text, calls, reasoning = [], [], []
        for part in c.get("parts") or []:
            if "functionCall" in part:
                fc = part["functionCall"]; cid = fc.get("id") or f"g{n}"; n += 1; pending.append(cid)
                calls.append({"id": cid, "type": "function", "function": {"name": fc.get("name", ""), "arguments": json.dumps(fc.get("args", {}))}})
            elif "functionResponse" in part:
                fr = part["functionResponse"]; cid = fr.get("id") or (pending.popleft() if pending else "")
                res = fr.get("response"); msgs.append({"role": "tool", "tool_call_id": cid, "content": res if isinstance(res, str) else json.dumps(res)})
            elif part.get("thought"): reasoning.append(part.get("text", ""))
            elif "text" in part: text.append(part["text"])
        if c.get("role") == "model":
            a = {"role": "assistant", "content": "".join(text)}
            if reasoning: a["reasoning_content"] = "".join(reasoning)
            if calls: a["tool_calls"] = calls
            msgs.append(a)
        elif text: msgs.append({"role": "user", "content": "".join(text)})
    tools = []
    for t in r.get("tools") or []:
        for f in t.get("functionDeclarations") or []:
            tools.append({"type": "function", "function": {"name": f.get("name", ""), "description": f.get("description", ""),
                                                           "parameters": f.get("parameters") or f.get("parametersJsonSchema")}})
    return msgs, tools


def decode_gemini_stream(b):
    msg = {"role": "assistant", "content": ""}; calls = []; finish = None
    chunks = []
    if b.lstrip()[:1] in (b"[", b"{"):
        try:
            j = json.loads(b); chunks = j if isinstance(j, list) else [j]
        except ValueError: pass
    else:
        for line in b.split(b"\n"):
            if line.startswith(b"data: "):
                try: chunks.append(json.loads(line[6:]))
                except ValueError: pass
    for j in chunks:
        r = j.get("response", j)
        for cand in r.get("candidates") or []:
            for part in (cand.get("content") or {}).get("parts") or []:
                if "functionCall" in part:
                    fc = part["functionCall"]
                    calls.append({"id": fc.get("id") or f"g{len(calls)}", "type": "function", "function": {"name": fc.get("name", ""), "arguments": json.dumps(fc.get("args", {}))}})
                elif part.get("thought"): msg["reasoning_content"] = msg.get("reasoning_content", "") + part.get("text", "")
                elif "text" in part: msg["content"] += part["text"]
            if cand.get("finishReason"): finish = cand["finishReason"]
    if not chunks: return None, None, False
    if calls: msg["tool_calls"] = calls
    f = "tool_calls" if calls and finish in ("STOP", None) else {"STOP": "stop", "MAX_TOKENS": "length"}.get(finish, finish)
    return msg, f, finish is not None

# ------------------------------------------------------------------------------------------------ checks + scrub
def tool_names(tools):
    return {(t.get("function") or t).get("name") for t in tools or []}


def repetition(text, n=40, k=6):
    if len(text) < n * k: return False
    seen = collections.Counter(text[i:i + n] for i in range(0, len(text) - n, n))
    return bool(seen) and seen.most_common(1)[0][1] >= k


ERR = re.compile(r"^\s*(error|Error|ERROR|Traceback|exit code [1-9]|command failed|No such file)", re.M)
SCRUB = [(re.compile(r"/home/[A-Za-z0-9_.-]+"), "/home/user"),
         (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "user@example.com"),
         (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "0.0.0.0"),
         (re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|xox[abp]-[A-Za-z0-9-]{10,})\b"), "[REDACTED]")]


def scrub(o):
    if isinstance(o, str):
        for r, s in SCRUB: o = r.sub(s, o)
        return o
    if isinstance(o, list): return [scrub(x) for x in o]
    if isinstance(o, dict): return {k: scrub(v) for k, v in o.items()}
    return o


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="*", default=[os.path.expanduser("~")])
    ap.add_argument("--keep", choices=["strict", "loose", "all"], default="strict")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--scrub", action="store_true")
    a = ap.parse_args()
    raw_dirs = set()
    for root in a.roots:   # <project>/.dream/raw at most 4 levels down; hidden folders other than .dream skipped
        root = os.path.expanduser(root); depth0 = root.rstrip("/").count("/")
        for cur, dirs, _ in os.walk(root):
            if os.path.basename(cur) == ".dream" and os.path.isdir(os.path.join(cur, "raw")):
                raw_dirs.add(os.path.realpath(os.path.join(cur, "raw")))
            dirs[:] = [x for x in dirs if (x == ".dream" or not x.startswith(".")) and x not in ("node_modules", "Expansion")]
            if cur.count("/") - depth0 >= 4: dirs[:] = [x for x in dirs if x == ".dream"]
    os.makedirs(a.out, exist_ok=True)
    stats = collections.defaultdict(collections.Counter)
    sessions = collections.defaultdict(list)
    for d in sorted(raw_dirs):
        for meta in glob.glob(os.path.join(d, "*", "*.meta.json")):
            try: m = json.load(open(meta))
            except (ValueError, OSError): continue
            sessions[(d, m.get("session_id") or os.path.basename(os.path.dirname(meta)))].append((m.get("created_at", ""), meta, m))
    n_out = 0
    with open(os.path.join(a.out, "dataset.jsonl"), "w") as out:
        for key, calls in sessions.items():
            calls.sort()
            decoded = []
            for _, meta, m in calls:
                base = meta[:-len(".meta.json")]
                model, prov = m.get("model") or "unknown", m.get("provider") or "unknown"
                st = stats[model]; st["calls"] += 1
                if a.local_only and prov not in LOCAL_PROVIDERS: st["skipped_not_local"] += 1; decoded.append(None); continue
                try:
                    rq = json.loads(jload(base + ".request.json")); rb = jload(base + ".response.bin")
                except (OSError, ValueError, EOFError):   # (a capture cut mid-gzip)
                    st["unreadable"] += 1; decoded.append(None); continue
                ep = m.get("endpoint") or ""
                if ep == "messages" or prov.startswith("anthropic"):
                    msgs, tools = anthropic_request(rq); reply, finish, complete = decode_anthropic_stream(rb)
                elif "GenerateContent" in ep or prov.startswith("google"):
                    msgs, tools = gemini_request(rq); reply, finish, complete = decode_gemini_stream(rb)
                elif ep == "responses" or "input" in rq:
                    msgs, tools = responses_request(rq); reply, finish, complete = decode_responses_stream(rb)
                else:
                    msgs, tools = list(rq.get("messages") or []), list(rq.get("tools") or []); reply, finish, complete = decode_chat_stream(rb)
                if reply is None: st["undecodable"] += 1; decoded.append(None); continue
                names = tool_names(tools)
                open_names = bool(names & {"tool_call", "tool_schema"})   # Dream's lazy tools: any listed-in-prompt name runs
                bad_json = unknown = False
                for c in reply.get("tool_calls") or []:
                    f = c.get("function") or {}
                    try: ok = isinstance(json.loads(f.get("arguments") or "{}"), dict)
                    except ValueError: ok = False
                    bad_json |= not ok
                    unknown |= bool(names) and not open_names and f.get("name") not in names
                flags = {"complete": complete, "finish": finish, "bad_tool_json": bad_json, "unknown_tool": unknown,
                         "repetition": repetition((reply.get("content") or "") + (reply.get("reasoning_content") or "")),
                         "tool_error_next": False}
                decoded.append({"source_model": model, "provider": prov, "session": key[1], "id": m.get("id"),
                                "created_at": m.get("created_at"), "tools": tools, "messages": msgs + [reply], "flags": flags,
                                "_ids": [c.get("id") for c in reply.get("tool_calls") or []]})
            # the next call's tool results for this reply's calls
            for i, ex in enumerate(decoded):
                if not ex or not ex["_ids"]: continue
                nxt = next((e for e in decoded[i + 1:] if e), None)
                if nxt:
                    res = {mm.get("tool_call_id"): mm.get("content", "") for mm in nxt["messages"] if mm.get("role") == "tool"}
                    ex["flags"]["tool_error_next"] = any(isinstance(res.get(c), str) and ERR.search(res[c][:400]) for c in ex["_ids"] if c)
            for ex in decoded:
                if not ex: continue
                ex.pop("_ids"); f, st = ex["flags"], stats[ex["source_model"]]
                for k in ("bad_tool_json", "unknown_tool", "repetition", "tool_error_next"):
                    if f[k]: st[k] += 1
                st["complete" if f["complete"] else "incomplete"] += 1
                st["finish_" + str(f["finish"])] += 1
                if a.keep == "strict": keep = f["complete"] and f["finish"] in ("stop", "tool_calls") and not any(f[k] for k in ("bad_tool_json", "unknown_tool", "repetition", "tool_error_next"))
                elif a.keep == "loose": keep = f["complete"] and f["finish"] != "length"
                else: keep = True
                if not keep: continue
                st["kept"] += 1; n_out += 1
                out.write(json.dumps(scrub(ex) if a.scrub else ex, ensure_ascii=False) + "\n")
    json.dump({"raw_dirs": sorted(raw_dirs), "keep": a.keep, "local_only": a.local_only, "scrub": a.scrub, "examples": n_out,
               "per_model": {k: dict(v) for k, v in stats.items()}}, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(f"{n_out} examples -> {a.out}/dataset.jsonl")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]["calls"]):
        print(f"  {k:40s} calls {v['calls']:6d}  complete {v['complete']:6d}  kept {v['kept']:6d}  "
              f"bad json {v['bad_tool_json']}  unknown tool {v['unknown_tool']}  repetition {v['repetition']}  tool error next {v['tool_error_next']}")


if __name__ == "__main__":
    main()
