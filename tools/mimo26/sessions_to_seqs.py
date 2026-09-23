#!/usr/bin/env python3
"""P4 lever 1 inputs (docs/mimo26/00_PORT_PLAN.md): real Dream agent sessions -> MiMo-V2.6 chat-rendered token sequences.

  sessions_to_seqs.py <model_dir> <sessions_dir> <out_dir> [--train N] [--heldout M] [--cap T]

Each session log (role user / assistant / tool_use / tool_result lines) becomes MiMo chat messages -- assistant text and
its tool calls one assistant turn, each tool_result a tool turn -- rendered with the checkpoint's own chat template
(transformers, byte-identical to the engine's; tools/mimo26/check_template.py) and cut at T tokens. The assistant turns'
token ranges are the COUNTED positions (the decode workload). Sessions are split by time order: every k-th session is
held out, so the held-out set shares no conversation with the profile.
Writes train.jsonl (for ie-mimo26-profile) and heldout_prompts/NN.txt: a held-out conversation cut at the start of an
assistant turn (the prompt a decode benchmark continues), plus heldout.json with their token counts.
"""
import json
import os
import sys

from transformers import PreTrainedTokenizerFast

model, sess_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
arg = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
n_train, n_held, cap = arg("--train", 40), arg("--heldout", 10), arg("--cap", 8192)
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model, "tokenizer.json"))
tmpl = open(os.path.join(model, "chat_template.jinja"), encoding="utf-8").read()
os.makedirs(os.path.join(out, "heldout_prompts"), exist_ok=True)


def messages(path):
    msgs = []
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        role, content = e.get("role"), e.get("content") or ""
        if role == "user":
            msgs.append({"role": "user", "content": content})
        elif role == "assistant":
            if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls"):
                msgs[-1]["content"] += "\n\n" + content
            else:
                msgs.append({"role": "assistant", "content": content})
        elif role == "tool_use":
            try:
                args = json.loads(content) if content.strip().startswith("{") else {"input": content}
            except json.JSONDecodeError:
                args = {"input": content}
            if not msgs or msgs[-1]["role"] != "assistant":
                msgs.append({"role": "assistant", "content": ""})
            msgs[-1].setdefault("tool_calls", []).append({"type": "function", "function": {"name": e.get("tool") or "tool", "arguments": args}})
        elif role == "tool_result":
            msgs.append({"role": "tool", "content": content[:6000]})
    return msgs


def render(msgs, gen=False):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False, chat_template=tmpl)


def ids_of(text):
    return tok(text, add_special_tokens=False)["input_ids"]


files = sorted(f for f in os.listdir(sess_dir) if f.endswith(".jsonl"))
usable = []
for f in files:
    m = messages(os.path.join(sess_dir, f))
    if sum(1 for x in m if x["role"] == "assistant") >= 3 and m and m[0]["role"] == "user":
        usable.append((f, m))
k = max(2, len(usable) // max(1, n_held))
held = [u for i, u in enumerate(usable) if i % k == k - 1][:n_held]
held_names = {f for f, _ in held}
train = [u for u in usable if u[0] not in held_names]
step = max(1, len(train) // n_train)
train = train[::step][:n_train]

n_tok = n_cnt = 0
with open(os.path.join(out, "train.jsonl"), "w") as fo:
    for f, m in train:
        spans, prev = [], 0
        for i in range(len(m)):
            cur = len(ids_of(render(m[:i + 1])))
            if m[i]["role"] == "assistant":
                spans.append([prev, min(cur, cap)])
            prev = cur
            if cur >= cap:
                break
        ids = ids_of(render(m[:i + 1]))[:cap]
        spans = [s for s in spans if s[0] < s[1]]
        n_tok += len(ids); n_cnt += sum(b - a for a, b in spans)
        fo.write(json.dumps({"session": f, "ids": ids, "count": spans}) + "\n")
meta = {}
for j, (f, m) in enumerate(held):
    # the last assistant turn that starts within 6000 tokens: the prompt is everything before it + the generation prompt
    best = None
    for i in range(1, len(m)):
        if m[i]["role"] == "assistant":
            text = render(m[:i], gen=True)
            n = len(ids_of(text))
            if n > 6000:
                break
            best = (i, text, n)
    if best:
        name = f"{j:02d}.txt"
        open(os.path.join(out, "heldout_prompts", name), "w", encoding="utf-8").write(best[1])
        meta[name] = {"session": f, "turn": best[0], "tokens": best[2]}
json.dump(meta, open(os.path.join(out, "heldout.json"), "w"), indent=1)
print(f"{len(usable)} usable sessions; train {len(train)} sequences, {n_tok} tokens, {n_cnt} counted (assistant) positions; "
      f"held-out prompts {len(meta)}: {[v['tokens'] for v in meta.values()]}")
