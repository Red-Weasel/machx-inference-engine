"""Where two PROCESSES first differ: hashes of every parameter / buffer, then of every module output of one forward.

  divergence.py run MODEL_DIR OUT.json [T]     (twice, in two processes)
  divergence.py cmp A.json B.json
Prints names and counts only.
"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

if sys.argv[1] == "cmp":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    pd = [k for k in a["params"] if a["params"][k] != b["params"].get(k)]
    print(json.dumps({"params": len(a["params"]), "params_differ": len(pd), "first_params_differ": pd[:12]}))
    n = min(len(a["outputs"]), len(b["outputs"]))
    first = next((i for i in range(n) if a["outputs"][i] != b["outputs"][i]), None)
    print(json.dumps({"module_outputs": n, "first_differing_index": first,
                      "around": a["outputs"][max(0, first - 2):first + 2] if first is not None else None,
                      "differing": sum(a["outputs"][i] != b["outputs"][i] for i in range(n))}))
    sys.exit(0)

from inkling2 import model as M
model_dir, out = sys.argv[2], sys.argv[3]
T = int(sys.argv[4]) if len(sys.argv) > 4 else 256
dev = "xpu:0"
tok, model, tracker, abl = M.load(model_dir, dev)
lm = model.model.language_model


def h(t):
    t = t.detach()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.md5(t.cpu().contiguous().numpy().tobytes()).hexdigest()[:12]


params = {}
for k, p in list(lm.named_parameters()) + list(lm.named_buffers()):
    if p.numel() < 5_000_000:              # the large expert code tensors come straight from the file
        params[k] = h(p)
outs = []
names = {m: n for n, m in lm.named_modules()}


def hook(m, args, o):
    x = o[0] if isinstance(o, tuple) else o
    if torch.is_tensor(x):
        outs.append(f"{names[m]}:{h(x)}")


for m in lm.modules():
    if not isinstance(m, (torch.nn.ModuleList,)) and m is not lm:
        m.register_forward_hook(hook)
torch.manual_seed(0)
ids = torch.randint(1000, 150000, (1, T), device="cpu").to(dev)
with torch.no_grad():
    M.lm_forward(model, ids)
json.dump({"params": params, "outputs": outs}, open(out, "w"))
print(len(params), len(outs))
