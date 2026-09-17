#!/usr/bin/env python3
"""The decode golden: the reference's own generate loop -- a prefill, then single-token steps
at start_pos = n -- with every cache carried inside DeepSeek's modules (window ring buffers,
compressed KV, index keys, compressor partial-group state, the engram hash cache).

All 40 Blocks stay alive. That is affordable because expert weights are np.memmap-backed: the
OS pages in only the experts a token routes to, so 40 live blocks cost RAM for the dense
weights and nothing for the 269 GB of experts. Dumps per-step logits and the chosen token, and
for the FIRST decode step every layer's output stream, so the engine's decode can be checked
layer by layer exactly as prefill was.

  usage: golden_decode.py <out_dir> [model_dir] [n_decode] [prompt]
"""
import gc, json, os, struct, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
N_DECODE = int(sys.argv[3]) if len(sys.argv) > 3 else 4
PROMPT = sys.argv[4] if len(sys.argv) > 4 else "The capital of France is Paris. The capital of Germany is"
sys.path.insert(0, HERE); sys.path.append(os.path.join(D, "inference"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors import safe_open
import kernel as K
import engram as E
import model as ref
import warnings; warnings.filterwarnings("ignore")

cfg = json.load(open(f"{D}/config.json")); tc = cfg["text_config"]; rs = tc["rope_scaling"]
args = ref.ModelArgs(
    dtype="fp8", expert_dtype="fp4", max_batch_size=1, max_seq_len=4096,
    vocab_size=tc["vocab_size"], dim=tc["hidden_size"], moe_inter_dim=tc["moe_intermediate_size"],
    n_layers=tc["num_hidden_layers"], n_mtp_layers=tc["num_nextn_predict_layers"],
    n_heads=tc["num_attention_heads"], n_routed_experts=tc["n_routed_experts"],
    n_shared_experts=tc["n_shared_experts"], n_activated_experts=tc["num_experts_per_tok"],
    score_func=tc["scoring_func"], norm_topk_prob=tc["norm_topk_prob"],
    route_scale=tc["routed_scaling_factor"], swiglu_limit=tc["swiglu_limit"],
    q_lora_rank=tc["q_lora_rank"], head_dim=tc["head_dim"], rope_head_dim=tc["qk_rope_head_dim"],
    norm_eps=tc["rms_norm_eps"], o_groups=tc["o_groups"], o_lora_rank=tc["o_lora_rank"],
    window_size=tc["sliding_window"], compress_ratios=tuple(tc["compress_ratios"]),
    kv_source_layers=tuple(tc["kv_source_layer_ids"]), index_source_layers=tuple(tc["index_source_layer_ids"]),
    compress_rope_theta=tc["compress_rope_theta"], rope_theta=tc["rope_theta"],
    original_seq_len=rs["original_max_position_embeddings"], rope_factor=rs["factor"],
    beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"],
    index_n_heads=tc["index_n_heads"], index_head_dim=tc["index_head_dim"], index_topk=tc["index_topk"],
    candidate_source_layer=tc["candidate_source_layer_id"], candidate_topk_blocks=tc["candidate_topk_blocks"],
    candidate_block_size=tc["candidate_block_size"], hc_mult=tc["hc_mult"],
    hc_sinkhorn_iters=tc["hc_sinkhorn_iters"], hc_eps=tc["hc_eps"],
    engram_layer_ids=tuple(tc["engram_layer_ids"]), engram_num_embeddings=tuple(tc["engram_num_embeddings"]),
    engram_max_ngram_size=tc["engram_max_ngram_size"], engram_vocab_size=tc["engram_vocab_size"],
    engram_n_heads=tc["engram_n_heads"], engram_head_dim=tc["engram_head_dim"],
    engram_pad_id=tc["engram_pad_token_id"], engram_compressed_vocab_size=tc["engram_compressed_vocab_size"],
    vision_n_layers=cfg["vision_config"]["num_hidden_layers"], image_token_id=cfg["image_token_id"],
)
ref.default_dtype = torch.float8_e4m3fn; ref.world_size, ref.rank = 1, 0

wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]; _h = {}; _hdr = {}
def get(n):
    sh = wm[n]
    if sh not in _h: _h[sh] = safe_open(f"{D}/{sh}", framework="pt")
    return _h[sh].get_tensor(n)
def locate(n):
    sh = wm[n]; p = f"{D}/{sh}"
    if p not in _hdr:
        with open(p, "rb") as f:
            hl, = struct.unpack("<Q", f.read(8)); _hdr[p] = (json.loads(f.read(hl)), 8 + hl)
    hdr, base = _hdr[p]; v = hdr[n]; return p, base + v["data_offsets"][0], v["shape"]
def mm_u8(n):
    p, off, sh = locate(n); return torch.from_numpy(np.memmap(p, dtype=np.uint8, mode="r", offset=off, shape=tuple(sh)))
def P(t): return torch.nn.Parameter(t, requires_grad=False)
def put(lin, base, kind):
    if kind == "fp4":     # memmap-backed: nothing is read until an expert is routed to
        lin.weight = P(mm_u8(f"{base}.weight").view(torch.float4_e2m1fn_x2))
        lin.weight.scale = lin.scale = P(mm_u8(f"{base}.scale").view(torch.float8_e8m0fnu))
        return
    w = get(f"{base}.weight")
    if kind == "fp8":
        lin.weight = P(w); lin.weight.scale = lin.scale = P(get(f"{base}.scale"))
    elif kind == "fp8->f32":
        lin.weight = P(K.dequant_fp8(w, get(f"{base}.scale"))); lin.scale = None
    else:
        lin.weight = P(w.float()); lin.scale = None

class MmapEngramEmbedding(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        pw, ow, shw = locate(name + ".weight"); ps, os_, shs = locate(name + ".scale")
        self.w = np.memmap(pw, dtype=np.uint8, mode="r", offset=ow, shape=tuple(shw))
        self.s = np.memmap(ps, dtype=np.uint8, mode="r", offset=os_, shape=tuple(shs))
    def forward(self, idx):
        flat = idx.reshape(-1).numpy()
        rows = torch.from_numpy(np.ascontiguousarray(self.w[flat])).view(torch.float8_e4m3fn).float()
        sc   = torch.from_numpy(np.ascontiguousarray(self.s[flat])).view(torch.float8_e8m0fnu).float()
        return (rows.unflatten(-1, (-1, 32)) * sc.unsqueeze(-1)).flatten(-2).to(torch.bfloat16).view(*idx.shape, -1)

def load_block(L):
    p = f"layers.{L}."
    ref.ParallelEngramEmbedding = lambda n, d, _p=p: MmapEngramEmbedding(_p + "engram.embed")
    blk = ref.Block(L, args, layout)
    blk.attn_norm.weight = P(get(p + "attn_norm.weight").float()); blk.ffn_norm.weight = P(get(p + "ffn_norm.weight").float())
    a = blk.attn
    a.q_norm.weight = P(get(p + "attn.q_norm.weight").float()); a.kv_norm.weight = P(get(p + "attn.kv_norm.weight").float())
    a.attn_sink = P(get(p + "attn.attn_sink").float())
    put(a.wq_a, p + "attn.wq_a", "fp8"); put(a.wq_b, p + "attn.wq_b", "fp8"); put(a.wkv, p + "attn.wkv", "fp8")
    put(a.wo_a, p + "attn.wo_a", "fp8->f32"); put(a.wo_b, p + "attn.wo_b", "fp8")
    if a.compressor is not None:
        c = a.compressor
        c.norm.weight = P(get(p + "attn.compressor.norm.weight").float()); put(c.wkv, p + "attn.compressor.wkv", "bf16")
        if hasattr(c, "wgate"): put(c.wgate, p + "attn.compressor.wgate", "bf16")
    if a.indexer is not None:
        ix = a.indexer
        put(ix.wq_b, p + "attn.indexer.wq_b", "fp8"); put(ix.weights_proj, p + "attn.indexer.weights_proj", "bf16")
        if ix.owns_k:
            put(ix.wk, p + "attn.indexer.wk", "bf16"); ix.k_norm.weight = P(get(p + "attn.indexer.k_norm.weight").float())
    for nm in ("attn_fn", "attn_base", "attn_scale", "ffn_fn", "ffn_base", "ffn_scale"):
        setattr(blk, "hc_" + nm, P(get(p + "hc_" + nm).float()))
    blk.ffn.gate.weight = P(get(p + "ffn.gate.weight").float()); blk.ffn.gate.bias = P(get(p + "ffn.gate.bias").float())
    blk.ffn.gate.bias_vl = P(get(p + "ffn.gate.bias_vl").float())
    for e in range(args.n_routed_experts):
        for nm in ("w1", "w2", "w3"): put(getattr(blk.ffn.experts[e], nm), f"{p}ffn.experts.{e}.{nm}", "fp4")
    for nm in ("w1", "w2", "w3"): put(getattr(blk.ffn.shared_experts, nm), f"{p}ffn.shared_experts.{nm}", "fp8")
    if blk.engram is not None:
        put(blk.engram.wkv, p + "engram.wkv", "fp8")
        blk.engram.q_weight = P(get(p + "engram.q_weight").float()); blk.engram.k_weight = P(get(p + "engram.k_weight").float())
    return blk

layout = E.EngramLayout.from_args(args)
tok = AutoTokenizer.from_pretrained(D)
hasher = E.NgramHashState(args, layout, tok)
embed_w = get("embed.weight").float(); norm_w = get("norm.weight").float(); head_w = get("head.weight").float()
ref.shared_attn = ref.SharedAttentionRuntime()

t0 = time.time()
blocks = [load_block(L) for L in range(args.n_layers)]
print(f"all {args.n_layers} blocks alive with memmap-backed experts: {time.time()-t0:.0f}s", flush=True)

def dump(n, t): t.detach().contiguous().float().numpy().tofile(f"{OUT}/{n}.f32")
def dump_i64(n, t): t.detach().contiguous().numpy().astype(np.int64).tofile(f"{OUT}/{n}.i64")

# Phase 9 criteria 1 and 6 (docs/deepseek41/19): the state the engine must carry across the
# prefill/decode boundary, read straight out of the reference's module buffers after `n` positions.
#   {prefix}_window_{L}      [window_size, head_dim]  the ring as held (slot = pos % win)
#   {prefix}_comp_{L}        [n // ratio, head_dim]    compressed latents (kv-source layers)
#   {prefix}_idxk_{L}        [n // ratio, index_head_dim]  index keys (kv-source layers)
#   {prefix}_kvstate_{L}     [ratio, head_dim]         the compressor's partial group (ratio 2: live at odd n)
#   {prefix}_scorestate_{L}  [ratio, head_dim]
#   {prefix}_hashcache       [n] int64                 compressed ids the 4-gram look-back reads
CACHE_LAYERS = (0, 2, 20)      # window at every layer; 2 = ratio-2 source, 20 = ratio-1 source
def dump_caches(prefix, n):
    for L in CACHE_LAYERS:
        a = blocks[L].attn
        dump(f"{prefix}_window_{L}", a.window_kv_cache[0])
        if a.compressor is not None:
            r = a.compress_ratio
            dump(f"{prefix}_comp_{L}", a.compress_kv_cache[0, : n // r])
            if a.indexer is not None and getattr(a.indexer, "k_cache", None) is not None:
                dump(f"{prefix}_idxk_{L}", a.indexer.k_cache[0, : n // r])
            if getattr(a.compressor, "kv_state", None) is not None:
                dump(f"{prefix}_kvstate_{L}", a.compressor.kv_state[0]); dump(f"{prefix}_scorestate_{L}", a.compressor.score_state[0])
    dump_i64(f"{prefix}_hashcache", hasher.cache[0, :n])

@torch.inference_mode()
def forward(ids, start_pos, dump_prefix=None):
    """Transformer.forward, verbatim in structure, over the live blocks."""
    hashes = hasher(ids, start_pos, None)
    if dump_prefix: dump_i64(f"{dump_prefix}_hashes", hashes)      # [1, L, n_engram_layers, n_hash_cols]
    h = F.embedding(ids, embed_w).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    pre_mix = ref.make_identity_pre_mix(h, args.hc_mult)
    for L, blk in enumerate(blocks):
        # the router's decision per token (golden_model.py's hook): indices, weights, and the
        # 6th-vs-7th score margin, so the engine's own-router divergences can be judged as
        # near-tie flips and its arithmetic checked with the routing forced (Phase 7's method)
        route = {}
        def gate_hook(mod, inp, out, _r=route):
            xg = inp[0].float()
            scores = torch.nn.functional.softplus(xg @ mod.weight.float().T).sqrt()
            sel = (scores + mod.bias).topk(mod.topk + 1, dim=-1)
            _r["weights"], _r["indices"] = out[0].clone(), out[1].clone()
            _r["gap"] = (sel.values[:, mod.topk - 1] - sel.values[:, mod.topk]).clone()
        hook = blk.ffn.gate.register_forward_hook(gate_hook) if dump_prefix else None
        if blk.engram is not None:
            h = blk.engram(h, hashes[:, :, blk.engram.layer_hash_index, :], None)
            if dump_prefix: dump(f"{dump_prefix}_engram_out_{L}", h)
        h, pre_mix = blk(h, start_pos, pre_mix, None)
        if dump_prefix:
            dump(f"{dump_prefix}_layer_out_{L}", h)
            hook.remove()
            route["indices"].to(torch.int32).numpy().tofile(f"{OUT}/{dump_prefix}_route_idx_{L}.i32")
            dump(f"{dump_prefix}_route_w_{L}", route["weights"]); dump(f"{dump_prefix}_route_gap_{L}", route["gap"])
    h = blocks[-1].hc_pre(h, pre_mix)
    hn = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + args.norm_eps) * norm_w
    return F.linear(hn[:, -1].float(), head_w)          # logits at the last position

ids = tok(PROMPT, return_tensors="pt", add_special_tokens=True)["input_ids"]
L0 = ids.shape[1]
print(f"prompt {PROMPT!r} -> {L0} tokens", flush=True)

t0 = time.time()
logits = forward(ids, 0, dump_prefix="p")
dump_caches("d_pre", L0)                                            # criterion 1: the state decode starts from
print(f"prefill {L0} tokens: {time.time()-t0:.0f}s, top-1 {tok.decode([int(logits.argmax())])!r}", flush=True)
steps = []
all_ids = ids[0].tolist()
for step in range(N_DECODE):
    nxt = int(logits.argmax(-1))
    steps.append({"pos": len(all_ids), "token": nxt, "text": tok.decode([nxt]),
                  "top5": logits[0].topk(5).indices.tolist()})
    logits.numpy().tofile(f"{OUT}/d_logits_step{step}.f32")
    all_ids.append(nxt)
    t1 = time.time()
    # the reference feeds ONLY the new token, at start_pos = its position
    logits = forward(torch.tensor([[nxt]]), len(all_ids) - 1, dump_prefix="d" if step == 0 else f"s{step}")
    dump_caches(f"d_step{step}", len(all_ids))                      # the state after this step (odd n = live partial group)
    print(f"  step {step}: fed {nxt} ({tok.decode([nxt])!r}) at pos {len(all_ids)-1}: {time.time()-t1:.0f}s -> "
          f"next {tok.decode([int(logits.argmax())])!r}", flush=True)
final = int(logits.argmax()); all_ids.append(final)
steps.append({"pos": len(all_ids) - 1, "token": final, "text": tok.decode([final]), "top5": logits[0].topk(5).indices.tolist()})
logits.numpy().tofile(f"{OUT}/d_logits_step{N_DECODE}.f32")

print(f"\ngenerated: {tok.decode(all_ids[L0:])!r}")
print(f"full: {tok.decode(all_ids)!r}")
json.dump({"prompt": PROMPT, "prompt_ids": ids[0].tolist(), "all_ids": all_ids, "n_decode": N_DECODE,
           "steps": steps, "vocab": args.vocab_size, "dim": args.dim, "hc_mult": args.hc_mult,
           "n_layers": args.n_layers}, open(f"{OUT}/d_meta.json", "w"), indent=1)
print(f"wrote decode goldens to {OUT}")
