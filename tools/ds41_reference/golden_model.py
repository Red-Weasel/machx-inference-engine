#!/usr/bin/env python3
"""The full-model golden: the reference's forward through ALL 40 layers for a real prompt,
giving reference logits and the first predicted token.

`Transformer(args)` cannot be constructed here -- it would torch.empty 269 GB of FP4 experts --
but Transformer.forward (model.py:1242) is a 30-line loop. This script runs that loop with each
Block(L) built, loaded and run in turn as the reference's OWN code (engram via the mmap-backed
table, experts mmap'd and dequantised only when routed to). It dumps every layer's output
stream so the engine can be checked layer by layer, plus the logits.

  usage: golden_model.py <out_dir> [model_dir] [prompt]
"""
import gc, json, os, struct, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
PROMPT = sys.argv[3] if len(sys.argv) > 3 else "The capital of France is Paris. The capital of Germany is"
sys.path.insert(0, HERE); sys.path.append(os.path.join(D, "inference"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors import safe_open
import kernel as K
import engram as E
import model as ref

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

wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]; _h = {}
def get(n):
    sh = wm[n]
    if sh not in _h: _h[sh] = safe_open(f"{D}/{sh}", framework="pt")
    return _h[sh].get_tensor(n)
def locate(n):
    sh = wm[n]; p = f"{D}/{sh}"
    with open(p, "rb") as f:
        hl, = struct.unpack("<Q", f.read(8)); hdr = json.loads(f.read(hl))
    v = hdr[n]; return p, 8 + hl + v["data_offsets"][0], v["shape"]
def P(t): return torch.nn.Parameter(t, requires_grad=False)
def put(lin, base, kind):
    w = get(f"{base}.weight")
    if kind == "fp4":
        lin.weight = P(w.view(torch.uint8).view(torch.float4_e2m1fn_x2)); lin.weight.scale = lin.scale = P(get(f"{base}.scale"))
    elif kind == "fp8":
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

# ---- Transformer.forward, unrolled --------------------------------------------------------
layout = E.EngramLayout.from_args(args)
tok = AutoTokenizer.from_pretrained(D)
hasher = E.NgramHashState(args, layout, tok)
ids = tok(PROMPT, return_tensors="pt", add_special_tokens=True)["input_ids"]
L_ = ids.shape[1]
print(f"prompt: {PROMPT!r} -> {L_} tokens {ids[0].tolist()}", flush=True)

embed_w = get("embed.weight").float()
ref.shared_attn = ref.SharedAttentionRuntime()
ref.shared_attn.__class__  # keep a reference so model.py's global is the one layers use
with torch.inference_mode():
    hashes = hasher(ids, 0, None)
    h = F.embedding(ids, embed_w)
    h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
    pre_mix = ref.make_identity_pre_mix(h, args.hc_mult)
    def dump(n, t): t.detach().contiguous().float().numpy().tofile(f"{OUT}/{n}.f32")
    dump("m_embed", h)
    t0 = time.time()
    for L in range(args.n_layers):
        tl = time.time()
        blk = load_block(L)
        route = {}
        def gate_hook(mod, inp, out, _L=L, _r=route):
            xg = inp[0].float()
            scores = torch.nn.functional.softplus(xg @ mod.weight.float().T).sqrt()
            sel = (scores + mod.bias).topk(mod.topk + 1, dim=-1)          # 7 = top-6 plus the runner-up
            _r["weights"], _r["indices"] = out[0].clone(), out[1].clone()
            _r["gap"] = (sel.values[:, mod.topk - 1] - sel.values[:, mod.topk]).clone()   # 6th minus 7th
        hook = blk.ffn.gate.register_forward_hook(gate_hook)
        if blk.engram is not None:
            h = blk.engram(h, hashes[:, :, blk.engram.layer_hash_index, :], None)
            dump(f"m_engram_out_{L}", h)
        h, pre_mix = blk(h, 0, pre_mix, None)
        dump(f"m_layer_out_{L}", h); dump(f"m_ffn_pre_{L}", pre_mix)
        hook.remove()
        route["indices"].to(torch.int32).numpy().tofile(f"{OUT}/m_route_idx_{L}.i32")
        dump(f"m_route_w_{L}", route["weights"]); dump(f"m_route_gap_{L}", route["gap"])
        kind = ("kv+idx src" if blk.attn.is_kv_source and blk.attn.is_index_source else
                "idx src" if blk.attn.is_index_source else "consumer" if blk.attn.compress_ratio else "window")
        print(f"  layer {L:2d} r{blk.attn.compress_ratio} {kind:10s} engram={blk.engram is not None!s:5s} "
              f"absmax {h.abs().max():8.4f}  {time.time()-tl:5.1f}s", flush=True)
        last = blk
        del blk; gc.collect()
    h = last.hc_pre(h, pre_mix)                                   # final collapse with the last ffn_pre
    normw = get("norm.weight").float()
    hn = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + args.norm_eps) * normw
    logits = F.linear(hn.float(), get("head.weight").float())     # all positions
    dump("m_final_collapsed", h); dump("m_logits", logits)
    top = logits[0, -1].topk(5)
    print(f"\n40 layers in {time.time()-t0:.0f}s")
    print(f"next-token top-5: {[(tok.decode([i]), round(v.item(),3)) for v, i in zip(top.values, top.indices)]}")
    nxt = logits[0].argmax(-1)
    print(f"greedy continuation from each position: {tok.decode(nxt.tolist())!r}")
json.dump({"prompt": PROMPT, "ids": ids[0].tolist(), "L": L_, "n_layers": args.n_layers,
           "dim": args.dim, "hc_mult": args.hc_mult, "vocab": args.vocab_size,
           "next_token": int(logits[0, -1].argmax()), "next_token_text": tok.decode([int(logits[0, -1].argmax())])},
          open(f"{OUT}/m_meta.json", "w"), indent=1)
print(f"wrote full-model goldens to {OUT}")
