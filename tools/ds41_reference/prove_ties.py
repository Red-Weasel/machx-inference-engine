#!/usr/bin/env python3
"""Turn the 6b boundary-tie EXCUSE into a PROOF (Phase 6 gate, finding 5).

The engine's top-512 differs from the reference's on 30 of 1,536 queries, all at near-tied
entries. The excuse was: swapping a near-tied entry cannot move the output much. The proof:
run the REFERENCE's own sparse_attn with the ENGINE's selected indices for exactly those
queries and show the output lands at the ordinary fp16 floor -- i.e. the entire 7.1e-3 was the
selection, not the arithmetic, and the selection difference is itself benign.

  usage: prove_ties.py <golden_dir> [model_dir]
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = sys.argv[1]; D = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
sys.path.insert(0, HERE); sys.path.append(os.path.join(D, "inference"))
import torch
import kernel as K
import model as ref

meta = json.load(open(f"{G}/a2_meta.json"))
T, NC, TOPK, WIN, NH, HD = meta["T"], meta["n_comp"], meta["topk"], meta["window"], 64, 512
q_ = None  # rebuilt below from the goldens: q is not dumped, so recompute the pieces we need
# We need q (roped), kv_win (roped), ckv (roped), sinks. The golden dumped ckv_roped and out,
# but not q/kv_win. Recompute them from a2_x through the reference's own Attention pieces.
cfg = json.load(open(f"{D}/config.json")); tc = cfg["text_config"]; rs = tc["rope_scaling"]
args = ref.ModelArgs(dtype="fp8", max_batch_size=1, max_seq_len=4096, dim=tc["hidden_size"],
    n_layers=tc["num_hidden_layers"], n_heads=tc["num_attention_heads"], q_lora_rank=tc["q_lora_rank"],
    head_dim=tc["head_dim"], rope_head_dim=tc["qk_rope_head_dim"], norm_eps=tc["rms_norm_eps"],
    o_groups=tc["o_groups"], o_lora_rank=tc["o_lora_rank"], window_size=tc["sliding_window"],
    compress_ratios=tuple(tc["compress_ratios"]), kv_source_layers=tuple(tc["kv_source_layer_ids"]),
    index_source_layers=tuple(tc["index_source_layer_ids"]), compress_rope_theta=tc["compress_rope_theta"],
    rope_theta=tc["rope_theta"], original_seq_len=rs["original_max_position_embeddings"],
    rope_factor=rs["factor"], beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"],
    index_n_heads=tc["index_n_heads"], index_head_dim=tc["index_head_dim"], index_topk=tc["index_topk"],
    candidate_source_layer=tc["candidate_source_layer_id"], candidate_topk_blocks=tc["candidate_topk_blocks"],
    candidate_block_size=tc["candidate_block_size"])
ref.default_dtype = torch.float8_e4m3fn; ref.world_size, ref.rank = 1, 0
from safetensors import safe_open
wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]; _h = {}
def get(n):
    sh = wm[n]
    if sh not in _h: _h[sh] = safe_open(f"{D}/{sh}", framework="pt")
    return _h[sh].get_tensor(n)
def P(t): return torch.nn.Parameter(t, requires_grad=False)
a = ref.Attention(2, args); p = "layers.2."
a.q_norm.weight = P(get(p+"attn.q_norm.weight").float()); a.kv_norm.weight = P(get(p+"attn.kv_norm.weight").float())
a.attn_sink = P(get(p+"attn.attn_sink").float())
for nm in ("wq_a","wq_b","wkv","wo_b"):
    lin = getattr(a, nm); lin.weight = P(get(f"{p}attn.{nm}.weight")); lin.weight.scale = lin.scale = P(get(f"{p}attn.{nm}.scale"))
a.wo_a.weight = P(K.dequant_fp8(get(p+"attn.wo_a.weight"), get(p+"attn.wo_a.scale"))); a.wo_a.scale = None

x = torch.from_numpy(np.fromfile(f"{G}/a2_x.f32", dtype=np.float32).reshape(1, T, -1))
ckv = torch.from_numpy(np.fromfile(f"{G}/a2_ckv_roped.f32", dtype=np.float32).reshape(1, NC, HD))
g_topk = torch.from_numpy(np.fromfile(f"{G}/a2_topk.i32", dtype=np.int32).reshape(1, T, TOPK))
E = f"{G}/eng"                                              # the engine's artifacts live apart from the goldens
e_topk = torch.from_numpy(np.fromfile(f"{E}/a2_engine_topk.i32", dtype=np.int32).reshape(1, T, TOPK))
tie_q = np.fromfile(f"{E}/a2_engine_tie_queries.i32", dtype=np.int32); tie_q = tie_q[tie_q >= 0]
e_meta = json.load(open(f"{E}/a2_engine_meta.json"))
e_out = torch.from_numpy(np.fromfile(f"{E}/a2_engine_out.f32", dtype=np.float32).reshape(1, T, -1))
# PROVENANCE: the tie list, the topk and the output must all be from the same run of the test
# (gate finding: nothing used to notice mixed dumps). The test writes its own measured numbers.
assert e_meta["T"] == T and e_meta["n_tie"] == len(tie_q), f"dump mismatch: meta says {e_meta['n_tie']} ties, file has {len(tie_q)}"
ENGINE_TIE_ERR = e_meta["tie_rel"]                          # measured by the test, not a literal
g_out = torch.from_numpy(np.fromfile(f"{G}/a2_out.f32", dtype=np.float32).reshape(1, T, -1))
print(f"{len(tie_q)} tie queries: {tie_q.tolist()[:10]}{' ...' if len(tie_q) > 10 else ''}")

with torch.inference_mode():
    fc = a.freqs_cis[0:T]; rd = a.rope_head_dim
    qr = a.q_norm(a.wq_a(x)); q = a.wq_b(qr).unflatten(-1, (a.n_local_heads, a.head_dim)); ref.apply_rotary_emb(q[..., -rd:], fc)
    kvw = a.kv_norm(a.wkv(x)); ref.apply_rotary_emb(kvw[..., -rd:], fc)
    kv = torch.cat([kvw, ckv], dim=1)
    win_idx = ref.get_window_topk_idxs(a.window_size, 1, T, 0)
    def run(comp_idx):
        # the engine's compressed indices are segment-relative, the reference's carry +offset(T)
        idx = torch.cat([win_idx, comp_idx], dim=-1)
        o = K.sparse_attn(q, kv, a.attn_sink, idx, a.softmax_scale)
        ref.apply_rotary_emb(o[..., -rd:], fc, True)
        og = o.view(1, T, a.n_local_groups, -1); wa = a.wo_a.weight.view(a.n_local_groups, a.o_lora_rank, -1)
        return a.wo_b(torch.einsum("bsgd,grd->bsgr", og, wa).flatten(2))
    out_ref = run(g_topk)                                       # sanity: must reproduce a2_out exactly
    e_shift = torch.where(e_topk >= 0, e_topk + T, -1)          # engine indices -> reference convention
    out_eng = run(e_shift)
scale = g_out.abs().max().item()
print(f"sanity: reference selection reproduces a2_out: max|diff| {(out_ref - g_out).abs().max().item():.3e}")
d_all = (out_eng - g_out).abs().max().item() / scale
d_tie = (out_eng[0, tie_q] - g_out[0, tie_q]).abs().max().item() / scale
mask = torch.ones(T, dtype=torch.bool); mask[tie_q] = False
d_same = (out_eng[0, mask] - g_out[0, mask]).abs().max().item() / scale
print(f"reference attention run WITH THE ENGINE'S selection:")
print(f"   over the {len(tie_q)} tie queries : rel {d_tie:.3e}   <- the effect of the swapped entries alone, in fp32")
print(f"   over the other {T-len(tie_q)}      : rel {d_same:.3e}   <- must be 0 (identical selection)")
# and compare to the engine's own error on those queries from the test (7.07e-3 published)
# d_same is not exactly 0: sparse_attn sums in index order and the engine's rows list the same
# set in a different order, so fp32 summation-order noise (~1e-7) remains (the gate confirmed
# that sorting the indices makes the two selections byte-identical and d_same exactly 0).
# THE DIRECT PROOF (gate finding 4): compare the engine's OWN output against the reference run on
# the engine's own selection. On the tie queries this must be at or below the engine's ordinary
# fp16 floor -- then the whole excess against the golden was the selection, in the engine too.
d_direct_tie = (e_out[0, tie_q] - out_eng[0, tie_q]).abs().max().item() / scale
d_direct_same = (e_out[0, mask] - out_eng[0, mask]).abs().max().item() / scale
print(f"engine output vs reference-on-the-engine's-selection:")
print(f"   tie queries  : rel {d_direct_tie:.3e}   <- must be at/below the fp16 floor")
print(f"   other queries: rel {d_direct_same:.3e}   <- the ordinary fp16 floor ({e_meta['same_rel']:.3e} per the test)")
consistent = abs(d_tie - ENGINE_TIE_ERR) < 0.2 * max(d_tie, ENGINE_TIE_ERR)
ok = d_same < 1e-6 and d_tie < 1e-2 and consistent and d_direct_tie <= 1.5 * d_direct_same
print("VERDICT:", f"PROVED — the reference, given the engine's selection, moves {d_tie:.3e} on the tie queries "
      f"(the engine measured {ENGINE_TIE_ERR:.3e} on them); and the engine's own output agrees with that reference run to "
      f"{d_direct_tie:.3e} there, at/below its {d_direct_same:.3e} floor elsewhere. The excess is the selection, not the arithmetic."
      if ok else f"NOT PROVED (d_same {d_same:.1e}, d_tie {d_tie:.3e} vs engine {ENGINE_TIE_ERR:.3e}, direct {d_direct_tie:.3e} vs floor {d_direct_same:.3e})")
sys.exit(0 if ok else 1)
