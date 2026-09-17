#!/usr/bin/env python
"""
Extended component parity harness — captures ground truth for EVERY DeepseekV4 submodule by running a
real (but tiny) DeepseekV4Model forward with hooks attached.

Why hooks and not standalone instantiation: Attention, the HCA/CSA compressors, the Indexer and
SparseMoeBlock all require live cache objects, position_embeddings, q_residual and layer_idx. Building
those by hand is guesswork and would gate the engine against inputs the real model never produces.
Running a real forward and recording what each module actually receives and returns is both easier and
strictly more trustworthy.

The model is shrunk (few layers, 8 experts, small vocab/hidden) so it fits in CPU RAM. Per-component
MATH is identical to the full model; only sizes differ. The layer schedule is preserved so that a
sliding, a CSA and an HCA layer are all exercised.

Why component-level at all: official weights are fp8 and our GGUF is a 3rd-party IQ3_XXS/MXFP4 requant,
so end-to-end bit-exactness is impossible. Component dumps isolate math correctness from quant error.

Usage:  python3 parity_dump2.py <outdir> [--layers N] [--hidden N]
"""
import json, os, sys, copy
import torch

def _positional(argv):
    """Positional args only — skip both '--opt' flags AND the value that follows them."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False; continue
        if a.startswith("--"):
            skip = True; continue
        out.append(a)
    return out

args = _positional(sys.argv[1:])
# Default to a STABLE path, not an ephemeral /tmp session scratchpad. The Phase 2 gate observed that
# when the blob dir vanishes the parity test prints SKIPPED and exits 0, which ctest reports as a green
# pass — so the evidence must outlive the session that produced it.
DEFAULT_OUT = os.path.expanduser("~/.cache/ie-deepseek4-parity/parity2")
OUT = args[0] if args else DEFAULT_OUT
def opt(name, default):
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default

os.makedirs(OUT, exist_ok=True)
torch.manual_seed(4321)

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
import transformers.models.deepseek_v4.modeling_deepseek_v4 as M

HERE = os.path.dirname(os.path.abspath(__file__))
# $DS4_HF_CONFIG points at a downloaded DeepSeek-V4-Flash config.json; the
# repo-local ds4-hf/ copy is the default. (A stale absolute scratch path used
# to sit here — it only ever resolved on one machine.)
CFG_PATH = next((c for c in [
    os.environ.get("DS4_HF_CONFIG", ""),
    os.path.join(HERE, "ds4-hf", "config.json"),
] if c and os.path.exists(c)), None)
if CFG_PATH is None:
    sys.exit("config.json not found — hf download deepseek-ai/DeepSeek-V4-Flash-0731 config.json")

raw = {k: v for k, v in json.load(open(CFG_PATH)).items() if k != "architectures"}
full = DeepseekV4Config(**raw)
full_layer_types = list(getattr(full, "layer_types", []) or [])

# Shrink. Keep head_dim / rope dims / hc_mult / compress rates EXACTLY as the real model — those drive
# the math we are gating. Shrink only counts and widths that are pure scale.
N_LAYERS = opt("--layers", 4)
small = copy.deepcopy(raw)
small.update(
    num_hidden_layers=N_LAYERS,
    hidden_size=opt("--hidden", 512),
    n_routed_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=128,
    intermediate_size=128,
    vocab_size=512,
    num_attention_heads=4,
    index_n_heads=4,
    o_groups=2,
    q_lora_rank=128,
    o_lora_rank=128,
    index_topk=8,
    max_position_embeddings=4096,
)
# Preserve the real schedule shape: sliding, sliding, CSA, HCA, ... truncated to N_LAYERS.
small["compress_ratios"] = raw["compress_ratios"][:N_LAYERS]
small.pop("layer_types", None)
cfg = DeepseekV4Config(**small)
lt = list(getattr(cfg, "layer_types", []) or [])
mlt = list(getattr(cfg, "mlp_layer_types", []) or [])
print(f"tiny model: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
      f"{cfg.n_routed_experts} experts | layer_types={lt} | mlp={mlt}")
print(f"(full model schedule for reference: {full_layer_types[:6]} ... {len(full_layer_types)} layers)")

model = M.DeepseekV4ForCausalLM(cfg).eval()

# HashRouter registers tid2eid as torch.zeros(vocab_size, top_k) and _init_weights does
# init.zeros_(module.tid2eid) — real values come from the checkpoint (modeling:1071,1238). A randomly
# initialised model therefore has an ALL-ZERO lookup table, which makes the dumped hash-router indices
# degenerate: a consumer with a transposed/mis-strided lookup would still match. The Phase 2 gate caught
# exactly that blind spot. Populate the table with real, distinct expert ids so stride errors are
# discriminable. Row-major, stride = top_k, matching modeling:1079 tid2eid[input_ids.reshape(-1)].
_n_hash = 0
for _m in model.modules():
    if isinstance(_m, M.DeepseekV4HashRouter) and hasattr(_m, "tid2eid"):
        with torch.no_grad():
            _m.tid2eid.copy_(torch.randint(0, cfg.n_routed_experts,
                                           _m.tid2eid.shape, dtype=_m.tid2eid.dtype))
        _n_hash += 1
print(f"populated tid2eid on {_n_hash} hash router(s) with random expert ids "
      f"(shape {tuple(model.model.layers[0].mlp.gate.tid2eid.shape) if _n_hash else 'n/a'})")

manifest = {
    "config_source": CFG_PATH,
    "torch": torch.__version__,
    "tiny_config": {k: small[k] for k in
                    ["num_hidden_layers", "hidden_size", "n_routed_experts", "num_experts_per_tok",
                     "num_attention_heads", "o_groups", "q_lora_rank", "index_topk"]},
    "preserved_from_full": {k: raw.get(k) for k in
                            ["head_dim", "qk_rope_head_dim", "hc_mult", "hc_sinkhorn_iters", "hc_eps",
                             "index_head_dim", "sliding_window", "compress_rope_theta", "rope_theta",
                             "swiglu_limit", "routed_scaling_factor", "scoring_func", "topk_method",
                             "norm_topk_prob", "rms_norm_eps"]},
    "tiny_layer_types": lt,
    "tiny_mlp_layer_types": mlt,
    "full_layer_types": full_layer_types,
    "components": [],
}

def dump(name, t):
    a = t.detach().to(torch.float32).contiguous().cpu().numpy()
    a.tofile(os.path.join(OUT, name + ".f32"))
    return {"file": name + ".f32", "shape": list(a.shape), "numel": int(a.size)}

def flat(x, out, prefix):
    """Recursively record tensors from arbitrary nested args/returns."""
    if torch.is_tensor(x):
        out.append((prefix, x)); return
    if isinstance(x, (tuple, list)):
        for i, v in enumerate(x): flat(v, out, f"{prefix}_{i}")
    elif isinstance(x, dict):
        for k, v in x.items(): flat(v, out, f"{prefix}_{k}")

INTERESTING = (
    M.DeepseekV4Attention, M.DeepseekV4HCACompressor, M.DeepseekV4CSACompressor,
    M.DeepseekV4Indexer, M.DeepseekV4IndexerScorer, M.DeepseekV4HyperConnection,
    M.DeepseekV4HyperHead, M.DeepseekV4SparseMoeBlock, M.DeepseekV4Experts,
    M.DeepseekV4TopKRouter, M.DeepseekV4HashRouter, M.DeepseekV4GroupedLinear,
    M.DeepseekV4RMSNorm, M.DeepseekV4UnweightedRMSNorm, M.DeepseekV4MLP,
    M.DeepseekV4DecoderLayer,
)
seen = {}

def make_hook(name, mod):
    def hook(_m, inputs, kwargs, output):
        # record only the FIRST invocation of each module instance
        if name in seen: return
        seen[name] = True
        e = {"component": name, "class": type(mod).__name__, "params": {}, "inputs": {}, "outputs": {}}
        for pn, p in mod.named_parameters(recurse=False):
            e["params"][pn] = dump(f"{name}.param.{pn.replace('.','_')}", p)
        # Buffers matter as much as parameters here: HashRouter's tid2eid lookup table is a registered
        # BUFFER (modeling:1071), so a params-only dump would omit the very table a consumer needs to
        # reproduce the lookup. Same for any rope inv_freq caches.
        for bn, b in mod.named_buffers(recurse=False):
            if torch.is_tensor(b):
                e["params"][f"buffer:{bn}"] = dump(f"{name}.buffer.{bn.replace('.','_')}", b)
        ins = []
        flat(list(inputs), ins, "arg")
        flat(kwargs, ins, "kw")
        for k, t in ins:
            e["inputs"][k] = dump(f"{name}.in.{k}", t)
        outs = []
        flat(output, outs, "out")
        for k, t in outs:
            e["outputs"][k] = dump(f"{name}.{k}", t)
        manifest["components"].append(e)
    return hook

for name, mod in model.named_modules():
    if isinstance(mod, INTERESTING):
        mod.register_forward_hook(make_hook(name or type(mod).__name__, mod), with_kwargs=True)

T = 32
ids = torch.randint(0, cfg.vocab_size, (1, T))
print(f"running forward, seq_len={T} ...")
with torch.no_grad():
    out = model(input_ids=ids, use_cache=True)
logits = out.logits
manifest["forward"] = {
    "input_ids": dump("forward.input_ids", ids.to(torch.float32)),
    "logits": dump("forward.logits", logits),
}
print(f"forward OK, logits {tuple(logits.shape)}, finite={bool(torch.isfinite(logits).all())}")

json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
by_class = {}
for c in manifest["components"]:
    by_class.setdefault(c["class"], 0)
    by_class[c["class"]] += 1
print(f"\n{len(manifest['components'])} module invocations captured -> {OUT}/manifest.json")
for k in sorted(by_class):
    print(f"  {k:36s} x{by_class[k]}")
