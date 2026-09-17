#!/usr/bin/env python3
"""Phase 6b golden: the reference's own Attention(2) -- a compress_ratio-2 SOURCE layer -- at
T=1536, dumping every stage the engine must reproduce: the compressor latent, the index keys,
the index scores (with their causal mask), the selected indices, and the attention output.

T is 1536 on purpose: topk = min(512, T//2) selects EVERY compressed position below 1,024
tokens, and the indexer's ranking would have no effect at all. At 1536 there are 768 compressed
positions and late queries keep 512 of them -- the selection is real, and the script asserts so.

  usage: golden_attn2.py <out_dir> [model_dir] [T]
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
T = int(sys.argv[3]) if len(sys.argv) > 3 else 1536
sys.path.insert(0, HERE); sys.path.append(os.path.join(D, "inference"))

import torch
from safetensors import safe_open
import kernel as K
import model as ref

cfg = json.load(open(f"{D}/config.json")); tc = cfg["text_config"]
args = ref.ModelArgs(
    dtype="fp8", max_batch_size=1, max_seq_len=4096,
    dim=tc["hidden_size"], n_layers=tc["num_hidden_layers"], n_heads=tc["num_attention_heads"],
    q_lora_rank=tc["q_lora_rank"], head_dim=tc["head_dim"], rope_head_dim=tc["qk_rope_head_dim"],
    norm_eps=tc["rms_norm_eps"], o_groups=tc["o_groups"], o_lora_rank=tc["o_lora_rank"],
    window_size=tc["sliding_window"], compress_ratios=tuple(tc["compress_ratios"]),
    kv_source_layers=tuple(tc["kv_source_layer_ids"]),
    index_source_layers=tuple(tc["index_source_layer_ids"]),
    compress_rope_theta=tc["compress_rope_theta"], rope_theta=tc["rope_theta"],
    original_seq_len=tc["rope_scaling"]["original_max_position_embeddings"],
    rope_factor=tc["rope_scaling"]["factor"], beta_fast=tc["rope_scaling"]["beta_fast"],
    beta_slow=tc["rope_scaling"]["beta_slow"], index_n_heads=tc["index_n_heads"],
    index_head_dim=tc["index_head_dim"], index_topk=tc["index_topk"],
    candidate_source_layer=tc["candidate_source_layer_id"],
    candidate_topk_blocks=tc["candidate_topk_blocks"], candidate_block_size=tc["candidate_block_size"],
)
ref.default_dtype = torch.float8_e4m3fn; ref.world_size, ref.rank = 1, 0
LAYER = 2
assert args.compress_ratios[LAYER] == 2 and LAYER in args.kv_source_layers and LAYER in args.index_source_layers

wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]; _h = {}
def get(n):
    sh = wm[n]
    if sh not in _h: _h[sh] = safe_open(f"{D}/{sh}", framework="pt")
    return _h[sh].get_tensor(n)
def P(t): return torch.nn.Parameter(t, requires_grad=False)
def put(lin, base, kind):
    w = get(f"{base}.weight")
    if kind == "fp8":
        lin.weight = P(w); lin.weight.scale = lin.scale = P(get(f"{base}.scale"))
    elif kind == "fp8->f32":
        lin.weight = P(K.dequant_fp8(w, get(f"{base}.scale"))); lin.scale = None
    else:  # bf16 -> fp32 for a full-precision golden
        lin.weight = P(w.float()); lin.scale = None

a = ref.Attention(LAYER, args)
p = f"layers.{LAYER}."
a.q_norm.weight = P(get(p + "attn.q_norm.weight").float())
a.kv_norm.weight = P(get(p + "attn.kv_norm.weight").float())
a.attn_sink = P(get(p + "attn.attn_sink").float())
put(a.wq_a, p + "attn.wq_a", "fp8"); put(a.wq_b, p + "attn.wq_b", "fp8")
put(a.wkv, p + "attn.wkv", "fp8"); put(a.wo_a, p + "attn.wo_a", "fp8->f32"); put(a.wo_b, p + "attn.wo_b", "fp8")
c = a.compressor
c.norm.weight = P(get(p + "attn.compressor.norm.weight").float())
put(c.wkv, p + "attn.compressor.wkv", "bf16"); put(c.wgate, p + "attn.compressor.wgate", "bf16")
ix = a.indexer
put(ix.wq_b, p + "attn.indexer.wq_b", "fp8"); put(ix.weights_proj, p + "attn.indexer.weights_proj", "bf16")
put(ix.wk, p + "attn.indexer.wk", "bf16")
ix.k_norm.weight = P(get(p + "attn.indexer.k_norm.weight").float())
print(f"Attention({LAYER}) bound: ratio {a.compress_ratio}, kv_source {a.is_kv_source}, index_source {a.is_index_source}")

g = torch.Generator().manual_seed(20260912)
x = (torch.randn(1, T, args.dim, generator=g) * 0.1).float()   # post-attn_norm input

# ---- capture the stages by running the module's own pieces in its own order -------------
with torch.inference_mode():
    fc = a.freqs_cis[0:T]; rd = a.rope_head_dim; ratio = a.compress_ratio
    qr = a.q_norm(a.wq_a(x))
    latent = c(x, 0)                                     # RoPE-free, normed  [1, T//2, 512]
    ix.freqs_cis = a.freqs_cis
    # index keys, exactly as Indexer.forward's owner branch
    freqs_g = a.freqs_cis[: T - T % ratio : ratio]
    k = ix.k_norm(ix.wk(latent)); ref.apply_rotary_emb(k[..., -rd:], freqs_g)
    # index query + scores, as Indexer.forward
    q_i = ix.wq_b(qr).unflatten(-1, (ix.n_local_heads, ix.index_head_dim))
    ref.apply_rotary_emb(q_i[..., -rd:], fc)
    weights = ix.weights_proj(x) * (ix.softmax_scale * ix.n_heads ** -0.5)
    score = torch.einsum("bshd,btd->bsht", q_i, k)
    score = (score.relu() * weights.unsqueeze(-1)).sum(dim=2)          # [1, T, T//2]
    compress_lens = (torch.arange(1, T + 1) // ratio).unsqueeze(-1)
    score_masked = score.masked_fill(torch.arange(T // ratio) >= compress_lens, -torch.inf)
    # the real thing, end to end
    out = a(x, 0)
    idxs = ref.shared_attn.topk_idxs                                   # [1, T, topk], offset by window
    ckv = ref.shared_attn.compress_kv[:1, : T // ratio]                # RoPE'd latent

# selection must be real: the last query can reach 768 and keeps at most 512
reach_last = int(compress_lens[-1]); kept_last = int((idxs[0, -1] >= 0).sum())
print(f"last query: reachable {reach_last}, selected {kept_last}  ({'REAL selection' if kept_last < reach_last else 'VACUOUS -- everything selected'})")
assert kept_last < reach_last, "T too small: the indexer selects everything and the test is vacuous"
print(f"out {tuple(out.shape)} absmax {out.abs().max():.4f}   latent absmax {latent.abs().max():.4f}   "
      f"scores finite {torch.isfinite(score_masked).float().mean():.3f}")

def dump(n, t): t.detach().contiguous().float().numpy().tofile(f"{OUT}/{n}.f32")
dump("a2_x", x); dump("a2_qr", qr); dump("a2_latent", latent); dump("a2_index_k", k)
dump("a2_index_q", q_i); dump("a2_weights", weights); dump("a2_score", score_masked)
dump("a2_ckv_roped", ckv); dump("a2_out", out)
idxs.to(torch.int32).numpy().tofile(f"{OUT}/a2_topk.i32")
json.dump({"T": T, "layer": LAYER, "ratio": ratio, "n_comp": T // ratio, "topk": idxs.shape[-1],
           "window": a.window_size, "index_heads": ix.n_local_heads, "index_head_dim": ix.index_head_dim,
           "offset": T}, open(f"{OUT}/a2_meta.json", "w"))
print(f"wrote layer-{LAYER} attention goldens to {OUT}")
