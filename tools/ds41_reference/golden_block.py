#!/usr/bin/env python3
"""Phase 5c golden: run DeepSeek-V4.1's OWN Block(0) on CPU and dump every sub-block boundary.

Layer 0 is the simplest layer in the model and therefore the right first target: compress_ratio
is 0, so it is sliding-window attention only -- no compressor, no indexer, no engram (those are
layers 1 and 14). Everything else about it is the general case: latent q/kv, one shared KV head,
attn_sink, the inverse RoPE on the attention output, the block-diagonal wo_a, mHC, and the
384-expert MoE.

  usage: golden_block.py <out_dir> [model_dir] [n_tokens]
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
D = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
sys.path.append(os.path.join(D, "inference"))

import torch
from safetensors import safe_open
import kernel as K
import model as ref

OUT = sys.argv[1]
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 8
os.makedirs(OUT, exist_ok=True)

cfg = json.load(open(f"{D}/config.json"))
tc = cfg["text_config"]
args = ref.ModelArgs(
    dtype="fp8", expert_dtype="fp4",
    max_batch_size=1, max_seq_len=4096,
    dim=tc["hidden_size"], moe_inter_dim=tc["moe_intermediate_size"],
    n_layers=tc["num_hidden_layers"], n_heads=tc["num_attention_heads"],
    n_routed_experts=tc["n_routed_experts"], n_shared_experts=tc["n_shared_experts"],
    n_activated_experts=tc["num_experts_per_tok"], score_func=tc["scoring_func"],
    norm_topk_prob=tc["norm_topk_prob"], route_scale=tc["routed_scaling_factor"],
    swiglu_limit=tc["swiglu_limit"], q_lora_rank=tc["q_lora_rank"],
    head_dim=tc["head_dim"], rope_head_dim=tc["qk_rope_head_dim"],
    norm_eps=tc["rms_norm_eps"], o_groups=tc["o_groups"], o_lora_rank=tc["o_lora_rank"],
    window_size=tc["sliding_window"], compress_ratios=tuple(tc["compress_ratios"]),
    kv_source_layers=tuple(tc["kv_source_layer_ids"]),
    index_source_layers=tuple(tc["index_source_layer_ids"]),
    compress_rope_theta=tc["compress_rope_theta"], rope_theta=tc["rope_theta"],
    original_seq_len=tc["rope_scaling"]["original_max_position_embeddings"],
    rope_factor=tc["rope_scaling"]["factor"], beta_fast=tc["rope_scaling"]["beta_fast"],
    beta_slow=tc["rope_scaling"]["beta_slow"], index_n_heads=tc["index_n_heads"],
    index_head_dim=tc["index_head_dim"], index_topk=tc["index_topk"],
    hc_mult=tc["hc_mult"], hc_sinkhorn_iters=tc["hc_sinkhorn_iters"], hc_eps=tc["hc_eps"],
    vision_n_layers=cfg["vision_config"]["num_hidden_layers"],
)
ref.default_dtype = torch.float8_e4m3fn
ref.world_size, ref.rank = 1, 0
assert args.compress_ratios[0] == 0, "layer 0 should be window-only"

wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
_h = {}
def get(n):
    sh = wm[n]
    if sh not in _h: _h[sh] = safe_open(f"{D}/{sh}", framework="pt")
    return _h[sh].get_tensor(n)

def P(t): return torch.nn.Parameter(t, requires_grad=False)

def put(lin, base, kind):
    """kind: 'fp4' keeps the packed dtype for linear()'s dispatch, 'fp8' likewise,
    'bf16' dequantises now (wo_a: the reference indexes .weight directly for a
    block-diagonal einsum, so it must already be a real float -- convert.py does this; here it
    is kept fp32 because the golden is full precision throughout)."""
    w = get(f"{base}.weight")
    if kind == "fp4":
        lin.weight = P(w.view(torch.uint8).view(torch.float4_e2m1fn_x2))
        lin.weight.scale = lin.scale = P(get(f"{base}.scale"))
    elif kind == "fp8":
        lin.weight = P(w)
        lin.weight.scale = lin.scale = P(get(f"{base}.scale"))
    else:
        # fp32, not the bf16 convert.py produces: this golden is full precision
        # throughout, and the einsum must share a dtype with the fp32 attention output.
        lin.weight = P(K.dequant_fp8(w, get(f"{base}.scale")))
        lin.scale = None

print("building reference Block(0) …")
blk = ref.Block(0, args, None)
a = blk.attn
p = "layers.0."
blk.attn_norm.weight = P(get(p + "attn_norm.weight").float())
blk.ffn_norm.weight  = P(get(p + "ffn_norm.weight").float())
a.q_norm.weight      = P(get(p + "attn.q_norm.weight").float())
a.kv_norm.weight     = P(get(p + "attn.kv_norm.weight").float())
a.attn_sink          = P(get(p + "attn.attn_sink").float())
put(a.wq_a, p + "attn.wq_a", "fp8")
put(a.wq_b, p + "attn.wq_b", "fp8")
put(a.wkv,  p + "attn.wkv",  "fp8")
put(a.wo_a, p + "attn.wo_a", "bf16")   # block-diagonal: needs a real float dtype
put(a.wo_b, p + "attn.wo_b", "fp8")
for nm in ("attn_fn", "attn_base", "attn_scale", "ffn_fn", "ffn_base", "ffn_scale"):
    setattr(blk, "hc_" + nm, P(get(p + "hc_" + nm).float()))
blk.ffn.gate.weight  = P(get(p + "ffn.gate.weight").float())
blk.ffn.gate.bias    = P(get(p + "ffn.gate.bias").float())
blk.ffn.gate.bias_vl = P(get(p + "ffn.gate.bias_vl").float())
for e in range(args.n_routed_experts):
    for nm in ("w1", "w2", "w3"):
        put(getattr(blk.ffn.experts[e], nm), f"{p}ffn.experts.{e}.{nm}", "fp4")
for nm in ("w1", "w2", "w3"):
    put(getattr(blk.ffn.shared_experts, nm), f"{p}ffn.shared_experts.{nm}", "fp8")
print("weights bound")

g = torch.Generator().manual_seed(20260912)
h = (torch.randn(1, NTOK, args.hc_mult, args.dim, generator=g) * 0.02).float()
pre_mix = ref.make_identity_pre_mix(h, args.hc_mult)

with torch.inference_mode():
    # sub-block boundaries, captured on the way through
    attn_pre, attn_post, attn_comb = blk.hc_mixes(h, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)
    xa = blk.hc_pre(h, pre_mix)
    xn = blk.attn_norm(xa)
    ao = a(xn, 0)

    # Stage-by-stage, recomputed with the module's own submodules so the engine can be
    # compared at each boundary rather than only at the end. A single end-to-end number
    # would say something is wrong without saying where -- the lesson Phase 4 paid for.
    fc = a.freqs_cis[0:NTOK]
    rd = a.rope_head_dim
    qr_ = a.q_norm(a.wq_a(xn))
    q_  = a.wq_b(qr_).unflatten(-1, (a.n_local_heads, a.head_dim))
    ref.apply_rotary_emb(q_[..., -rd:], fc)
    kv_ = a.kv_norm(a.wkv(xn))
    ref.apply_rotary_emb(kv_[..., -rd:], fc)
    tk_ = ref.get_window_topk_idxs(a.window_size, 1, NTOK, 0)
    o_  = K.sparse_attn(q_, kv_, a.attn_sink, tk_, a.softmax_scale)
    o_r = o_.clone()
    ref.apply_rotary_emb(o_r[..., -rd:], fc, True)
    og  = o_r.view(1, NTOK, a.n_local_groups, -1)
    wa  = a.wo_a.weight.view(a.n_local_groups, a.o_lora_rank, -1)
    oa  = torch.einsum("bsgd,grd->bsgr", og, wa)
    ob  = a.wo_b(oa.flatten(2))
    assert torch.allclose(ob, ao, atol=0, rtol=0), "stage recompute must reproduce Attention.forward"

    # The FFN half, recomputed with the block's own submodules so the MoE inside the BLOCK has
    # a golden of its own. golden_moe.py's moe_out uses a different, synthetic input, so
    # comparing the block's MoE against it is meaningless -- which is exactly the mistake the
    # first version of ds41_block_test made, hidden behind a tolerance of 1e10.
    st1_ = blk.hc_post(ao, h, attn_post, attn_comb)
    ffn_pre_, ffn_post_, ffn_comb_ = blk.hc_mixes(st1_, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base)
    xf_ = blk.ffn_norm(blk.hc_pre(st1_, attn_pre))
    moe_ = blk.ffn(xf_, None)
    out_ = blk.hc_post(moe_, st1_, ffn_post_, ffn_comb_)

    out, ffn_pre = blk(h, 0, pre_mix, None)
    assert torch.allclose(out_, out, atol=0, rtol=0), "block stage recompute must reproduce Block.forward"

print(f"stage recompute == Attention.forward: exact")
print(f"attn out  absmax {ao.abs().max():.6f}  std {ao.std():.6f}")
print("block stage recompute == Block.forward: exact")
print(f"block out {tuple(out.shape)}  absmax {out.abs().max():.6f}  std {out.std():.6f}")
print(f"hc: pre {tuple(attn_pre.shape)} post {tuple(attn_post.shape)} comb {tuple(attn_comb.shape)}")
print(f"comb row sums {attn_comb.sum(-1).flatten()[:4].tolist()}  col sums "
      f"{attn_comb.sum(-2).flatten()[:4].tolist()}")

def dump(n, t): t.detach().contiguous().float().numpy().tofile(f"{OUT}/{n}.f32")
for n, t in [("blk_ffn_in", xf_), ("blk_moe_out", moe_), ("blk_ffn_post", ffn_post_),
             ("blk_ffn_comb", ffn_comb_),
             ("attn_qr", qr_), ("attn_q_roped", q_), ("attn_kv_roped", kv_),
             ("attn_o_raw", o_), ("attn_o_unroped", o_r), ("attn_wo_a_out", oa),
             ("blk_in", h), ("blk_pre_mix", pre_mix), ("blk_attn_pre", attn_pre),
             ("blk_attn_post", attn_post), ("blk_attn_comb", attn_comb),
             ("blk_attn_in", xn), ("blk_attn_out", ao), ("blk_out", out),
             ("blk_ffn_pre", ffn_pre)]:
    dump(n, t)
json.dump({"n_tokens": NTOK, "dim": args.dim, "hc_mult": args.hc_mult,
           "n_heads": args.n_heads, "head_dim": args.head_dim,
           "rope_head_dim": args.rope_head_dim, "o_groups": args.o_groups,
           "o_lora_rank": args.o_lora_rank, "window": args.window_size,
           "norm_eps": args.norm_eps, "seed": 20260912},
          open(f"{OUT}/blk_meta.json", "w"), indent=2)
print(f"wrote goldens to {OUT}")
