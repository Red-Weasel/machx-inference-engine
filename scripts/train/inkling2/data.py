"""Training data for Inkling 2 Mini: Dream's recorded-session files rendered the way Dream's live requests look.

Input: one JSON object per line, {"messages": [...]} (other keys ignored; an optional "tools" array is used as the
record's own tool declarations). Roles system / user / assistant / tool. Assistant turns hold Qwen-style text:
optional "<think>...</think>", then text and/or "<tool_call><function=NAME><parameter=K>v</parameter></function>
</tool_call>" blocks; or already-structured reasoning_content / tool_calls. "weight": 0/1 on assistant turns selects the
trained turns (no weight on any turn = every assistant turn is trained).

Rendering (what `ie serve` builds for Dream's request; the engine's renderer is golden-tested against this template):
- tool declarations: the record's "tools" if it has them, else the `tools` array of a Dream request sample
  (python -m dream.request_sample): the live request carries the tools in its `tools` field, the template renders them
  as one tool_declare block. A Qwen tool preamble left in a system page ("# Tools ... <tools>...</tools> ...
  </IMPORTANT>") is a rendering artefact of the recorded files and is removed from the page.
- assistant turns: reasoning -> a thinking block, text -> a text block, each call -> an invoke block with the
  arguments as an object (values parsed as JSON where the declared schema does not say "string").
- tool results: role tool, named after the call they answer (the k-th result after a turn answers its k-th call),
  as the engine names a result from its tool_call_id.
- thinking effort: left to the template default (0.9): Dream's request sets no reasoning_effort.
Loss: on every token the model writes for a trained turn (<|message_model|> ... <|content_model_end_sampling|>).

Windowing: an example longer than max_len keeps the system page and the first user message whole and drops the
oldest whole exchanges after them (an assistant turn with its tool results, or a user turn) until it fits: the way
Dream itself compacts a long history. A record that does not fit becomes the fewest windows: each ends at the latest trained turn
not yet covered, keeps as much recent history as fits, and trains every trained turn inside it. Only a trained turn that does not fit even with no history is cut at max_len (counted).
Records of one session that are prefixes of a longer record are merged into it (their trained turn becomes a trained
turn of the longer record) so the shared history is computed once.
Errors name the line and the problem, never the text.
"""
import glob
import json
import os
import re

_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)
_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
_QWEN_TOOLS = re.compile(r"^\s*# Tools\s*\n.*?</IMPORTANT>\s*", re.S)
SAMPLE_DIR = os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.expanduser("~/data")), "inkling2-train-exports/dream-sample")
ROLES = ("system", "user", "assistant", "tool")


class DataError(ValueError):
    pass


def newest_sample(folder=SAMPLE_DIR):
    c = [p for p in glob.glob(os.path.join(folder, "*.json")) if _is_sample(p)]
    return max(c, key=os.path.getmtime) if c else None


def _is_sample(p):
    try:
        s = json.load(open(p))
        return isinstance(s, dict) and isinstance(s.get("tools"), list) and "system" in s
    except (OSError, ValueError):
        return False


def load_sample(path):
    s = json.load(open(path))
    if not isinstance(s.get("tools"), list):
        raise DataError(f"{os.path.basename(path)} is not a Dream request sample (no tools array)")
    return s


def _schemas(tools):
    out = {}
    for t in tools or []:
        fn = t.get("function", t)
        out[fn.get("name")] = ((fn.get("parameters") or {}).get("properties") or {})
    return out


def _value(v, prop):
    if isinstance(prop, dict) and prop.get("type") == "string":
        return v
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return v


def convert(messages, tools, line_no):
    """Dream / Qwen-text messages -> (Inkling messages, weights). Raises DataError on a malformed record."""
    if not isinstance(messages, list) or not messages:
        raise DataError(f"line {line_no}: 'messages' is missing or empty")
    schemas = _schemas(tools)
    out, weights, pending = [], [], []
    for j, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            raise DataError(f"line {line_no}: message {j} has no known role (system, user, assistant, tool)")
        role, content = m["role"], m.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            if isinstance(content, list) and all(isinstance(p, dict) and p.get("type", "text") == "text" for p in content):
                content = "".join(p.get("text", "") for p in content)
            else:
                raise DataError(f"line {line_no}: message {j} content is not text")
        if role == "system":
            if j:
                raise DataError(f"line {line_no}: a system message after the first message")
            out.append({"role": "system", "content": _QWEN_TOOLS.sub("", content, count=1)})
            weights.append(0)
            continue
        if role == "tool":
            msg = {"role": "tool", "content": content}
            if pending:
                msg["tool_call_id"] = pending.pop(0)
            elif m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
            out.append(msg)
            weights.append(0)
            continue
        pending = []
        if role == "user":
            out.append({"role": "user", "content": content})
            weights.append(0)
            continue
        w = m.get("weight", None)
        if w not in (None, 0, 1, True, False, 0.0, 1.0):
            raise DataError(f"line {line_no}: message {j} weight is not 0 or 1")
        a = {"role": "assistant"}
        text = content
        rc = m.get("reasoning_content")
        mt = _THINK.match(text)
        if mt:
            rc = (mt.group(1).strip() or None) if rc is None else rc
            text = text[mt.end():]
        if rc:
            a["reasoning_content"] = rc.strip() if isinstance(rc, str) else rc
        calls = []
        for k, tc in enumerate(m.get("tool_calls") or []):
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    raise DataError(f"line {line_no}: message {j} has tool-call arguments that are not JSON")
            if not isinstance(fn.get("name"), str) or not isinstance(args, dict):
                raise DataError(f"line {line_no}: message {j} has a malformed tool call")
            calls.append({"id": tc.get("id") or f"c{j}_{k}", "type": "function", "function": {"name": fn["name"], "arguments": args}})
        for k, (name, body) in enumerate(_CALL.findall(text)):
            name = name.strip()
            props = schemas.get(name, {})
            args = {pk: _value(pv, props.get(pk)) for pk, pv in _PARAM.findall(body)}
            calls.append({"id": f"c{j}_{len(calls)}", "type": "function", "function": {"name": name, "arguments": args}})
        if "<tool_call>" in _CALL.sub("", text):
            raise DataError(f"line {line_no}: message {j} has a tool call that does not parse")
        a["content"] = _CALL.sub("", text).strip()
        if calls:
            a["tool_calls"] = calls
            pending = [c["id"] for c in calls]
        out.append(a)
        weights.append(None if w is None else int(w))
    if all(w is None for w, m in zip(weights, out) if m["role"] == "assistant"):
        weights = [1 if m["role"] == "assistant" else 0 for m in out]
    else:
        weights = [int(w or 0) for w in weights]
    if not any(weights):
        raise DataError(f"line {line_no}: no assistant turn to train on")
    return out, weights


class Renderer:
    def __init__(self, tok):
        self.tok = tok

    def segments(self, msgs, tools):
        """Token ids per message (the template is concatenative: message i's text is the tail of the render up to i).
        Segment 0 also holds the tool declarations; the trailing effort line, when no non-system message came, is
        never needed (every example has a user turn)."""
        prev, segs = "", []
        for i in range(len(msgs)):
            r = self.tok.apply_chat_template(msgs[:i + 1], tools=tools or None, tokenize=False)
            if not r.startswith(prev):
                raise DataError("internal: the template is not concatenative for this record")
            segs.append(self.tok(r[len(prev):], add_special_tokens=False)["input_ids"])
            prev = r
        return segs, prev


def _units(msgs, start):
    """Droppable units after index `start`: a user turn, or an assistant turn with the tool results after it."""
    units, i = [], start
    while i < len(msgs):
        j = i + 1
        if msgs[i]["role"] == "assistant":
            while j < len(msgs) and msgs[j]["role"] == "tool":
                j += 1
        units.append((i, j))
        i = j
    return units


def window(segs, weights, msgs, max_len, stats):
    """Examples (ids, labels) from one rendered record under the windowing rule (see the module doc)."""
    total = sum(len(s) for s in segs)
    trained = [i for i, w in enumerate(weights) if w]

    def build(keep):
        ids, labels = [], []
        for i in keep:
            ids += segs[i]
            labels += segs[i] if (weights[i] and i in tset) else [-100] * len(segs[i])
        return ids, labels

    if total <= max_len:
        tset = set(trained)
        return [build(range(len(segs)))]
    stats["windowed"] = stats.get("windowed", 0) + 1
    first_user = next((i for i, m in enumerate(msgs) if m["role"] == "user"), 0)
    head = list(range(first_user + 1))
    hsize = sum(len(segs[i]) for i in head)
    units = _units(msgs, first_user + 1)
    remaining = [t for t in trained if t > first_user]
    out = []
    while remaining:
        t = remaining[-1]                                         # the latest trained turn not yet in a window
        body = [u for u in units if u[0] <= t]
        last = body.pop()                                         # the unit holding t: kept, cut after t
        keep_last = list(range(last[0], t + 1))
        size = hsize + sum(len(segs[i]) for i in keep_last)
        kept = []
        for u in reversed(body):                                  # newest history first, whole exchanges
            n = sum(len(segs[i]) for i in range(*u))
            if size + n > max_len:
                break
            kept.insert(0, u)
            size += n
        stats["exchanges_dropped"] = stats.get("exchanges_dropped", 0) + len(body) - len(kept)
        keep = head + [i for u in kept for i in range(*u)] + keep_last
        tset = {i for i in remaining if i in keep}                # every trained turn inside this window
        remaining = [i for i in remaining if i not in tset]
        ids, labels = build(keep)
        if len(ids) > max_len:
            stats["cut_at_max_len"] = stats.get("cut_at_max_len", 0) + 1
            ids, labels = ids[:max_len], labels[:max_len]
        if any(l != -100 for l in labels[1:]):
            out.append((ids, labels))
    return out


def _key(m):
    return json.dumps({k: m.get(k) for k in ("role", "content", "reasoning_content", "tool_calls")}, sort_keys=True)


def merge_prefixes(records):
    """records: list of (msgs, weights). A record whose messages are a prefix of a longer record of the input is
    folded into it (its trained turns become trained turns there). Returns the merged list and how many were folded."""
    keyed = [[_key(m) for m in msgs] for msgs, _ in records]
    order = sorted(range(len(records)), key=lambda i: -len(records[i][0]))
    by_first = {}
    for i in order:
        by_first.setdefault(tuple(keyed[i][:2]), []).append(i)
    absorbed, weights = set(), [list(w) for _, w in records]
    for i in sorted(range(len(records)), key=lambda i: len(records[i][0])):
        for j in by_first.get(tuple(keyed[i][:2]), []):
            if j == i or j in absorbed or len(keyed[j]) <= len(keyed[i]):
                continue
            if keyed[j][:len(keyed[i])] == keyed[i]:
                for k, w in enumerate(weights[i]):
                    if w:
                        weights[j][k] = 1
                absorbed.add(i)
                break
    return [(records[i][0], weights[i]) for i in range(len(records)) if i not in absorbed], len(absorbed)


def load_file(path, tok, tools_default, max_len, max_examples=None, merge=True, log=print):
    """Read, check, convert, merge, render and window a data file. Returns (examples, stats). Never prints text."""
    recs, stats = [], {"file": os.path.basename(path), "records": 0}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                raise DataError(f"line {n} of {os.path.basename(path)} is not JSON")
            if not isinstance(r, dict) or "messages" not in r:
                raise DataError(f"line {n} of {os.path.basename(path)} is not an object with a 'messages' list")
            tools = r.get("tools") if isinstance(r.get("tools"), list) else tools_default
            msgs, w = convert(r["messages"], tools, n)
            recs.append((msgs, w, tools))
            if max_examples and len(recs) >= max_examples:
                break
    if not recs:
        raise DataError(f"{os.path.basename(path)} has no records")
    stats["records"] = len(recs)
    groups = {}
    for msgs, w, tools in recs:
        groups.setdefault(json.dumps(tools, sort_keys=True), (tools, []))[1].append((msgs, w))
    rend = Renderer(tok)
    examples, folded, lengths = [], 0, []
    for tools, rs in groups.values():
        if merge:
            rs, k = merge_prefixes(rs)
            folded += k
        for msgs, w in rs:
            segs, _ = rend.segments(msgs, tools)
            lengths.append(sum(len(s) for s in segs))
            for ids, labels in window(segs, w, msgs, max_len, stats):
                examples.append({"ids": ids, "labels": labels, "rec": len(lengths) - 1})
    lengths.sort()
    stats.update(merged_into_longer=folded, rendered_records=len(lengths),
                 median_tokens=lengths[len(lengths) // 2], max_tokens=lengths[-1], examples=len(examples),
                 tokens=sum(len(e["ids"]) for e in examples),
                 trained_tokens=sum(sum(1 for x in e["labels"][1:] if x != -100) for e in examples))
    return examples, stats
