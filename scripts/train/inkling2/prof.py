"""Where a training step spends card time and memory.

  xpu_env.sh venv/bin/python -I prof.py MODEL_DIR OUT_JSON
(1) torch.profiler on one 4,096-token step: the top kernels by self card time; (2) peak card memory of one step at
4,096 tokens for attention blocks of 256 / 512 / 1,024 query rows; (3) seconds and peak at 8,192 and 12,288 with the
smallest block that helps.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from inkling2 import model as M

model_dir, out = sys.argv[1], sys.argv[2]
dev = "xpu:0"
tok, model, tracker, abl = M.load(model_dir, dev)
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
    if len(stream) > 13000:
        break
ad = M.add_lora(model)
params = [p for w in ad.values() for p in (w.lora_A, w.lora_B)]
res = {}


def step(T):
    ids = stream[:T]
    labels = [-100] * int(T * 0.85) + ids[int(T * 0.85):]
    for p in params:
        p.grad = None
    l = M.example_loss(model, tracker, ids, labels, dev)[0]
    l.backward()
    torch.xpu.synchronize(dev)


def measure(T, reps=1):
    import gc
    gc.collect(); torch.xpu.empty_cache(); torch.xpu.reset_peak_memory_stats(dev)
    t = time.time()
    try:
        for _ in range(reps):
            step(T)
    except torch.OutOfMemoryError:
        return {"tokens": T, "attn_chunk": M.ATTN_CHUNK, "out_of_memory": True}
    return {"tokens": T, "attn_chunk": M.ATTN_CHUNK, "seconds": (time.time() - t) / reps,
            "peak_gib": torch.xpu.max_memory_allocated(dev) / 2**30}


step(1024)
from torch.profiler import ProfilerActivity, profile
M.ATTN_CHUNK = 512
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
    step(4096)
table = prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=30, max_name_column_width=60)
print(table, flush=True)
res["profile_table"] = table
res["memory_4096"] = []
for c in (1024, 512, 256):
    M.ATTN_CHUNK = c
    r = measure(4096)
    res["memory_4096"].append(r)
    print(json.dumps(r), flush=True)
best = min((r for r in res["memory_4096"] if "peak_gib" in r), key=lambda r: r["peak_gib"])
M.ATTN_CHUNK = best["attn_chunk"]
res["long"] = []
for T in (8192, 12288):
    r = measure(T)
    res["long"].append(r)
    print(json.dumps(r), flush=True)
json.dump(res, open(out, "w"), indent=1)
