#!/usr/bin/env python
"""
Component-level numerical parity harness for the DeepSeek-V4 engine port.

Full end-to-end bit-exactness against the reference is NOT achievable: the official weights are fp8
and our GGUF is a 3rd-party IQ3_XXS/MXFP4 requant. So we gate PER COMPONENT instead — instantiate each
reference module with deterministic random weights, run deterministic input through it, and dump
(weights, input, output) as raw little-endian f32. The C++ side loads the same blobs, runs its kernel,
and asserts a tolerance match. That isolates "did I implement the math right" from "is the quant lossy".

Usage:  python3 parity_dump.py <outdir>
"""
import json, os, sys, struct
import torch
import torch.nn as nn

DEFAULT_OUT = os.path.expanduser("~/.cache/ie-deepseek4-parity/parity")
OUT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(1234)
torch.use_deterministic_algorithms(True)

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
import transformers.models.deepseek_v4.modeling_deepseek_v4 as M

CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds4-hf", "config.json")
raw = json.load(open(CFG_PATH))
cfg = DeepseekV4Config(**{k: v for k, v in raw.items() if k != "architectures"})

manifest = {"config_source": CFG_PATH, "torch": torch.__version__, "components": []}

def dump(name, arr):
    """Write a tensor as raw little-endian f32 + return its shape."""
    a = arr.detach().to(torch.float32).contiguous().cpu().numpy()
    path = os.path.join(OUT, name + ".f32")
    a.tofile(path)
    return {"file": name + ".f32", "shape": list(a.shape), "numel": int(a.size)}

def record(comp, mod, inputs, outputs, notes=""):
    entry = {"component": comp, "notes": notes, "params": {}, "inputs": {}, "outputs": {}}
    for pn, p in mod.named_parameters():
        entry["params"][pn] = dump(f"{comp}.param.{pn.replace('.','_')}", p)
    for i, t in enumerate(inputs):
        entry["inputs"][f"in{i}"] = dump(f"{comp}.in{i}", t)
    for i, t in enumerate(outputs):
        entry["outputs"][f"out{i}"] = dump(f"{comp}.out{i}", t)
    manifest["components"].append(entry)
    print(f"  OK  {comp}: {len(entry['params'])} params, "
          f"{[v['shape'] for v in entry['inputs'].values()]} -> {[v['shape'] for v in entry['outputs'].values()]}")

def attempt(label, fn):
    try:
        fn()
    except Exception as e:
        print(f"  SKIP {label}: {type(e).__name__}: {str(e)[:160]}")
        manifest["components"].append({"component": label, "error": f"{type(e).__name__}: {str(e)[:300]}"})

print(f"config: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
      f"hc_mult {getattr(cfg,'hc_mult','?')}, experts {getattr(cfg,'n_routed_experts','?')}")
print("dumping components ->", OUT)

B, T = 1, 8
H = cfg.hidden_size

# ---- 1. HyperConnection: touches every layer; 4 parallel residual streams + Sinkhorn mixing ----
def do_hc():
    hc = M.DeepseekV4HyperConnection(cfg).eval()
    hcm = getattr(cfg, "hc_mult", 4)
    x = torch.randn(B, T, hcm, H, dtype=torch.float32)
    with torch.no_grad():
        out = hc(x)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    record("hyper_connection", hc, [x], [o for o in outs if torch.is_tensor(o)],
           f"hc_mult={hcm}, sinkhorn_iters={getattr(cfg,'hc_sinkhorn_iters','?')}")
attempt("hyper_connection", do_hc)

# ---- 2. GroupedLinear: the o_groups=8 / o_lora_rank=1024 output projection ----
# Construct EXACTLY as Attention.__init__ does (modeling_deepseek_v4.py:792-794):
#   GroupedLinear(num_heads*head_dim // o_groups, o_groups*o_lora_rank, o_groups)
# and feed it the real [B,S,o_groups,in_per_group] layout, not [B,S,hidden].
def do_gl():
    g  = getattr(cfg, "o_groups", 8)
    r  = getattr(cfg, "o_lora_rank", 1024)
    ipg = cfg.num_attention_heads * cfg.head_dim // g
    gl = M.DeepseekV4GroupedLinear(ipg, g * r, g).eval()
    x = torch.randn(B, T, g, ipg, dtype=torch.float32)
    with torch.no_grad():
        out = gl(x)
    record("grouped_linear", gl, [x], [out],
           f"o_groups={g}, o_lora_rank={r}, in_per_group={ipg}")
attempt("grouped_linear", do_gl)

# ---- 3. TopKRouter: sqrtsoftplus scoring + noaux_tc bias correction + renorm + scale ----
def do_router():
    r = M.DeepseekV4TopKRouter(cfg).eval()
    x = torch.randn(B * T, H, dtype=torch.float32)
    with torch.no_grad():
        out = r(x)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    record("topk_router", r, [x], [o for o in outs if torch.is_tensor(o)],
           f"scoring={getattr(cfg,'scoring_func','?')}, topk_method={getattr(cfg,'topk_method','?')}, "
           f"top_k={getattr(cfg,'num_experts_per_tok','?')}, scale={getattr(cfg,'routed_scaling_factor','?')}")
attempt("topk_router", do_router)

# ---- 4. HashRouter: layers 0-2 (ffn_gate_tid2eid) ----
def do_hash():
    r = M.DeepseekV4HashRouter(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    x = torch.randn(B * T, H, dtype=torch.float32)
    with torch.no_grad():
        try:    out = r(x, ids)
        except TypeError: out = r(ids)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    record("hash_router", r, [x, ids.to(torch.float32)], [o for o in outs if torch.is_tensor(o)],
           f"num_hash_layers={getattr(cfg,'num_hash_layers','?')}")
attempt("hash_router", do_hash)

# ---- 5. RMSNorm variants ----
def do_norm():
    n = M.DeepseekV4RMSNorm(H, eps=cfg.rms_norm_eps).eval()
    with torch.no_grad():
        n.weight.copy_(torch.randn(H))
        x = torch.randn(B, T, H, dtype=torch.float32)
        out = n(x)
    record("rms_norm", n, [x], [out], f"eps={cfg.rms_norm_eps}")
attempt("rms_norm", do_norm)

# ---- 6. MLP (shared expert) with swiglu clamp ----
def do_mlp():
    mlp = M.DeepseekV4MLP(cfg).eval()
    x = torch.randn(B, T, H, dtype=torch.float32)
    with torch.no_grad():
        out = mlp(x)
    record("mlp_shared_expert", mlp, [x], [out], f"swiglu_limit={getattr(cfg,'swiglu_limit','?')}")
attempt("mlp_shared_expert", do_mlp)

json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
ok = [c for c in manifest["components"] if "error" not in c]
print(f"\n{len(ok)}/{len(manifest['components'])} components dumped -> {OUT}/manifest.json")
for c in manifest["components"]:
    if "error" in c:
        print(f"  FAILED {c['component']}: {c['error'][:120]}")
