#!/usr/bin/env python3
"""Phase 5a golden: run DeepSeek-V4.1's OWN MoE block on CPU and dump inputs/outputs.

Constructs the reference's `MoE(0, args)` straight from inference/model.py — its Gate, its
Expert, its routing, its SwiGLU clamps — with only the six leaf kernels replaced (see
kernel.py in this directory). Loads layer 0's real weights and runs a deterministic input.

  usage: golden_moe.py <out_dir> [model_dir] [n_tokens]
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # our kernel.py shim must win
D = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
sys.path.append(os.path.join(D, "inference"))  # engram / image_processor / vision

import torch
from safetensors import safe_open

import model as ref

OUT = sys.argv[1]
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 8
os.makedirs(OUT, exist_ok=True)

cfg = json.load(open(f"{D}/config.json"))
tc = cfg["text_config"]

# ModelArgs field names are the config keys, but a few differ; map only what MoE reads.
args = ref.ModelArgs(
    dtype="fp8",
    expert_dtype="fp4",
    dim=tc["hidden_size"],
    moe_inter_dim=tc["moe_intermediate_size"],
    n_layers=tc["num_hidden_layers"],
    n_routed_experts=tc["n_routed_experts"],
    n_shared_experts=tc["n_shared_experts"],
    n_activated_experts=tc["num_experts_per_tok"],
    score_func=tc["scoring_func"],
    norm_topk_prob=tc["norm_topk_prob"],
    route_scale=tc["routed_scaling_factor"],
    swiglu_limit=tc["swiglu_limit"],
    vision_n_layers=cfg["vision_config"]["num_hidden_layers"],
)
ref.default_dtype = torch.float8_e4m3fn
ref.world_size, ref.rank = 1, 0

# ---- weight access ---------------------------------------------------------------
wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
_handles = {}
def get(name):
    sh = wm[name]
    if sh not in _handles:
        _handles[sh] = safe_open(f"{D}/{sh}", framework="pt")
    return _handles[sh].get_tensor(name)

def put(lin, base, fp4):
    """Load `<base>.weight` (+ `.scale`) into a reference Linear, keeping the dtype its
    `linear()` dispatches on."""
    w = get(f"{base}.weight")
    if fp4:
        w = w.view(torch.uint8).view(torch.float4_e2m1fn_x2)
    lin.weight = torch.nn.Parameter(w, requires_grad=False)
    s = get(f"{base}.scale")
    lin.weight.scale = lin.scale = torch.nn.Parameter(s, requires_grad=False)

print(f"building reference MoE(0) — {args.n_routed_experts} experts, dim {args.dim}, "
      f"inter {args.moe_inter_dim}, top-{args.n_activated_experts}, score '{args.score_func}'")
moe = ref.MoE(0, args)

moe.gate.weight = torch.nn.Parameter(get("layers.0.ffn.gate.weight").float(), requires_grad=False)
moe.gate.bias   = torch.nn.Parameter(get("layers.0.ffn.gate.bias").float(),   requires_grad=False)
moe.gate.bias_vl= torch.nn.Parameter(get("layers.0.ffn.gate.bias_vl").float(),requires_grad=False)
for e in range(args.n_routed_experts):
    for nm in ("w1", "w2", "w3"):
        put(getattr(moe.experts[e], nm), f"layers.0.ffn.experts.{e}.{nm}", fp4=True)
for nm in ("w1", "w2", "w3"):
    put(getattr(moe.shared_experts, nm), f"layers.0.ffn.shared_experts.{nm}", fp4=False)
print("weights bound (lazily mmapped; only selected experts are ever read)")

# ---- deterministic input ---------------------------------------------------------
g = torch.Generator().manual_seed(20260912)
# std 0.13, the scale of the real post-ffn_norm input (measured 0.1269 on the block golden).
# The first version used 0.02, and the Phase 5 gate showed why that matters: SwiGLU's gate/up
# asymmetry -- the only thing a w1/w3 swap perturbs -- shrinks with the activations, so at
# std 0.02 a swapped bank passed the engine test at 1.9e-2. At the real scale it cannot.
x = (torch.randn(1, NTOK, args.dim, generator=g) * 0.13).float()

with torch.inference_mode():
    flat = x.view(-1, args.dim)
    weights, indices = moe.gate(flat, None)
    y = moe(x)

print(f"routed: indices {tuple(indices.shape)}, unique experts hit "
      f"{len(set(indices.flatten().tolist()))}")
print(f"y: {tuple(y.shape)}  absmax {y.abs().max():.6f}  std {y.std():.6f}")

# ---- self-check: recompute the block independently of the reference modules -------
# The reference's own code is the authority on COMPOSITION, but the wiring into it is mine
# (which safetensors tensor lands on which Linear). A w1/w3 swap, a transposed weight, or a
# missed clamp would produce a plausible golden and poison everything downstream. So recompute
# the same block from the raw bytes with numpy semantics and require agreement.
with torch.inference_mode():
    import kernel as K
    xf = x.view(-1, args.dim).float()
    scores = torch.nn.functional.softplus(xf @ moe.gate.weight.float().T).sqrt()
    idx = (scores + moe.gate.bias.float()).topk(args.n_activated_experts, dim=-1)[1]
    wts = scores.gather(1, idx)
    wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * args.route_scale
    assert torch.equal(idx, indices), "self-check: routing indices disagree with the reference"
    assert torch.allclose(wts, weights, atol=0, rtol=0), "self-check: routing weights disagree"

    L = args.swiglu_limit
    def ffn(w1, w3, w2, xin, scale=None):
        gate = xin @ w1.T
        up = xin @ w3.T
        if L > 0:
            up, gate = up.clamp(-L, L), gate.clamp(max=L)
        h = torch.nn.functional.silu(gate) * up
        if scale is not None:
            h = scale * h
        return h @ w2.T

    y_routed = torch.zeros_like(xf)
    y2 = y_routed
    for e in sorted(set(idx.flatten().tolist())):
        rows, top = torch.where(idx == e)
        w1 = K.dequant_fp4(get(f"layers.0.ffn.experts.{e}.w1.weight").view(torch.uint8),
                           get(f"layers.0.ffn.experts.{e}.w1.scale"))
        w3 = K.dequant_fp4(get(f"layers.0.ffn.experts.{e}.w3.weight").view(torch.uint8),
                           get(f"layers.0.ffn.experts.{e}.w3.scale"))
        w2 = K.dequant_fp4(get(f"layers.0.ffn.experts.{e}.w2.weight").view(torch.uint8),
                           get(f"layers.0.ffn.experts.{e}.w2.scale"))
        y2[rows] += ffn(w1, w3, w2, xf[rows], wts[rows, top, None])
    y_routed = y2.clone()      # routed contribution only, before the shared expert
    sh = {n: K.dequant_fp8(get(f"layers.0.ffn.shared_experts.{n}.weight"),
                           get(f"layers.0.ffn.shared_experts.{n}.scale"))
          for n in ("w1", "w3", "w2")}
    y2 += ffn(sh["w1"], sh["w3"], sh["w2"], xf)
    d = (y2 - y.view(-1, args.dim).float()).abs().max().item()
    rel = d / y.abs().max().item()
    print(f"self-check: independent recompute max|diff| {d:.3e} (rel {rel:.2e})")
    assert rel < 1e-6, f"self-check FAILED: the golden disagrees with an independent recompute ({rel:.2e})"

def dump(name, t):
    t.detach().contiguous().float().numpy().tofile(f"{OUT}/{name}.f32")
dump("moe_in", x)
dump("moe_out", y)
dump("moe_routed_out", y_routed)
dump("gate_weights", weights)
indices.detach().contiguous().to(torch.int32).numpy().tofile(f"{OUT}/gate_indices.i32")
json.dump({"n_tokens": NTOK, "dim": args.dim, "topk": args.n_activated_experts,
           "n_routed": args.n_routed_experts, "seed": 20260912,
           "inter": args.moe_inter_dim, "swiglu_limit": args.swiglu_limit,
           "route_scale": args.route_scale, "score_func": args.score_func},
          open(f"{OUT}/moe_meta.json", "w"), indent=2)
print(f"wrote goldens to {OUT}")
