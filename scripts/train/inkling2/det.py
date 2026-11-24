"""Run-to-run spread of the card forward / gradient, with the reference's index_add combine and the deterministic one.

  xpu_env.sh venv/bin/python -I det.py MODEL_DIR OUT_JSON
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from inkling2 import compressed_experts as CE
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
    if len(stream) > 2000:
        break
ad = M.add_lora(model)
params = [p for w in ad.values() for p in (w.lora_A, w.lora_B)]
g = torch.Generator().manual_seed(1)
with torch.no_grad():
    for w in ad.values():
        w.lora_B.copy_(torch.randn(w.lora_B.shape, generator=g) * 1e-3)
ids = stream[:1024]
labels = [-100] * 768 + ids[768:]
res = {}
for det in (False, True):
    CE.DETERMINISTIC = det
    with torch.no_grad():
        ls = [M.example_loss(model, tracker, ids, labels, dev)[0].item() for _ in range(4)]
    grads, gl = [], []
    for _ in range(3):
        for p in params:
            p.grad = None
        l = M.example_loss(model, tracker, ids, labels, dev)[0]
        l.backward()
        gl.append(l.item())
        grads.append(torch.cat([p.grad.flatten().float().cpu() for p in params]))
    cs = [torch.nn.functional.cosine_similarity(grads[i], grads[j], dim=0).item() for i, j in ((0, 1), (0, 2), (1, 2))]
    res["deterministic" if det else "index_add"] = {"forward_losses": ls, "grad_losses": gl, "grad_cos_pairs": cs}
    print(json.dumps(res), flush=True)
json.dump(res, open(out, "w"), indent=1)
