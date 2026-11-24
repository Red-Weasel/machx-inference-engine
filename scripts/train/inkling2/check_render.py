"""Converter check: Dream's text-form sessions (dream_local_*.jsonl) vs their OpenAI-shaped twins (*.openai.jsonl,
structured reasoning / tool calls with ids and typed arguments, the tools array), both rendered with the model's
template. Prints counts only.

  xpu_env.sh venv/bin/python -I check_render.py MODEL_DIR TEXT.jsonl TWIN.openai.jsonl [N]
"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from inkling2.data import Renderer, convert

model, text_f, twin_f = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 400
tok = AutoTokenizer.from_pretrained(model)
rend = Renderer(tok)
twins = {}
for line in open(twin_f):
    r = json.loads(line)
    twins[r["id"]] = r
c = collections.Counter()
for n, line in enumerate(open(text_f), 1):
    if n > N:
        break
    r = json.loads(line)
    t = twins.get(r["id"])
    if t is None:
        c["no twin"] += 1
        continue
    msgs, w = convert(r["messages"], t["tools"], n)
    c["page equal after removing the tool preamble"] += msgs[0]["content"] == t["messages"][0]["content"]
    # the twin's call ids repeat across turns (the engine's call_chatcmpl_N); made unique per turn here, so the
    # comparison is of the conversion, not of that engine id bug (reported separately)
    tm, turn = [], 0
    for m in t["messages"]:
        m = dict(m)
        if m.get("tool_calls"):
            turn += 1
            m["tool_calls"] = [dict(tc, id=f"t{turn}_{tc['id']}", function=dict(tc["function"], arguments=json.loads(tc["function"]["arguments"] or "{}")))
                               for tc in m["tool_calls"]]
        elif m["role"] == "tool" and m.get("tool_call_id"):
            m["tool_call_id"] = f"t{turn}_{m['tool_call_id']}"
        tm.append(m)
    segs, full = rend.segments(msgs, t["tools"])
    mine = [x for s in segs for x in s]
    c["segment ids == ids of the whole render"] += mine == tok(full, add_special_tokens=False)["input_ids"]
    ref = tok.apply_chat_template(tm, tools=t["tools"], tokenize=True, return_dict=False)
    same = mine == ref
    c["identical token ids"] += same
    if not same:
        # where the first difference is: which role's message (no text printed)
        k = next(i for i, (x, y) in enumerate(zip(mine + [-1], ref + [-1])) if x != y)
        pos, role = 0, "?"
        for s, m in zip(segs, msgs):
            if pos + len(s) > k:
                role = m["role"] + (" call" if m.get("tool_calls") else "")
                break
            pos += len(s)
        c[f"first difference in a {role} message"] += 1
    c["records"] += 1
print(json.dumps(c, indent=1))
