"""What is alive at the backward's memory peak of one training step (allocation trace replayed to the peak, live bytes
grouped by the deepest source line in this package or in the Inkling model code).

  xpu_env.sh venv/bin/python -I memtrace.py MODEL_DIR T
"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from inkling2 import model as M

model_dir, T = sys.argv[1], int(sys.argv[2])
dev = "xpu:0"
tok, model, tracker, abl = M.load(model_dir, dev)
ad = M.add_lora(model)
stream = []
for line in open(os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.expanduser("~/data")), "dream-dataset/sft/dream_local_train.openai.jsonl")):
    r = json.loads(line)
    msgs = []
    for m in r["messages"]:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [dict(tc, function=dict(tc["function"], arguments=json.loads(tc["function"]["arguments"] or "{}"))) for tc in m["tool_calls"]]
        msgs.append(m)
    stream += tok.apply_chat_template(msgs, tools=r.get("tools"), tokenize=True, return_dict=False)
    if len(stream) > T + 10:
        break
ids = stream[:T]
labels = [-100] * int(T * 0.85) + ids[int(T * 0.85):]
loss = M.example_loss(model, tracker, ids, labels, dev)[0]
torch.xpu.memory._record_memory_history(max_entries=2_000_000)
loss.backward()
snap = torch.xpu.memory._snapshot()
torch.xpu.memory._record_memory_history(enabled=None)


def where(frames):
    keep = [f for f in frames if ("inkling2" in f["filename"]) and "torch/" not in f["filename"]]
    f = keep[0] if keep else (frames[0] if frames else {"filename": "?", "line": 0, "name": "?"})
    return f"{os.path.basename(f['filename'])}:{f['line']} {f['name']}"


live, cur, peak, at_peak = {}, 0, 0, None
for ev in snap["device_traces"][0]:
    a = ev["action"]
    if a == "alloc":
        live[ev["addr"]] = (ev["size"], where(ev.get("frames", [])))
        cur += ev["size"]
        if cur > peak:
            peak, at_peak = cur, dict(live)
    elif a in ("free_completed",):
        if ev["addr"] in live:
            cur -= live.pop(ev["addr"])[0]
by = collections.Counter()
for size, w in at_peak.values():
    by[w] += size
print(f"backward allocations alive at its peak: {peak / 2**30:.2f} GiB (on top of what the forward left)")
for w, s in by.most_common(25):
    print(f"  {s / 2**30:7.3f} GiB  {w}")
