"""Card memory of one training step: held after the forward (what the backward will need) and the peaks of the forward
and the backward, at a few lengths; and a per-call trace of the backward's peak (which call, how much).

  xpu_env.sh venv/bin/python -I memprobe.py MODEL_DIR T1,T2,...
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from inkling2 import model as M

model_dir, Ts = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
dev = "xpu:0"
tok, model, tracker, abl = M.load(model_dir, dev)
ad = M.add_lora(model)
params = [p for w in ad.values() for p in (w.lora_A, w.lora_B)]
G = 2**30
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
    if len(stream) > max(Ts) + 10:
        break
base = torch.xpu.memory_allocated(dev) / G
print(f"weights on the card: {base:.2f} GiB", flush=True)
for T in Ts:
    ids = stream[:T]
    labels = [-100] * int(T * 0.85) + ids[int(T * 0.85):]
    for p in params:
        p.grad = None
    torch.xpu.empty_cache()
    torch.xpu.reset_peak_memory_stats(dev)
    try:
        loss = M.example_loss(model, tracker, ids, labels, dev)[0]
        held = torch.xpu.memory_allocated(dev) / G
        fpeak = torch.xpu.max_memory_allocated(dev) / G
        torch.xpu.reset_peak_memory_stats(dev)
        loss.backward()
        bpeak = torch.xpu.max_memory_allocated(dev) / G
        print(json.dumps({"tokens": T, "held_after_forward_gib": round(held - base, 2),
                          "forward_peak_gib": round(fpeak - base, 2), "backward_peak_gib": round(bpeak - base, 2),
                          "total_peak_gib": round(max(fpeak, bpeak), 2)}), flush=True)
    except torch.OutOfMemoryError:
        print(json.dumps({"tokens": T, "out_of_memory": True}), flush=True)
        break
    del loss
