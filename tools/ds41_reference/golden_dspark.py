#!/usr/bin/env python3
"""DSpark P1 golden: the reference's own `forward_spec` (model.py:1275-1282) -- the drafter's
three DSparkBlocks, the tied head, the Markov head and the confidence head -- run on the CPU in
fp32 over the EXISTING decode golden's forced steps, every stage boundary dumped.

Where the forced inputs come from (golden_decode.py -> <golden>/decode2):
  * `main_hidden` at position p is what Transformer.forward collects at model.py:1264-1271: the
    mean over the hc copies of the stream ENTERING each target layer (37, 38, 39), concatenated
    in that order. The engram runs before the capture (1262-1266), but layers 37-39 carry none
    (engram_layer_ids = [1, 14]), so the stream entering layer L is the previous layer's output:
    `{prefix}_layer_out_{L-1}.f32`, the `h` golden_decode.py dumps after `blk(...)`, laid out
    [1, T, hc, dim].  prefix p = the 12-token prefill (T = 12); d = decode step 0 (pos 12);
    s1 / s2 / s3 = steps 1-3 (pos 13-15), one row each.
  * `input_ids` for the pass at position p is the token the backbone sampled from that step,
    all_ids[p + 1] == argmax(d_logits_step{k+1}) (asserted here).
  * the window rings are seeded exactly as the reference seeds them: forward_spec at start_pos 0
    over the prefill's main_hidden (model.py:1044-1052), then one ring write per decode position
    (1065) by running forward_spec at 12, 13, 14, 15 in order -- so the ring at 15 is the
    reference's own, not a reconstruction.

Position algebra (model.py:1275-1282; generate.py:69-76): the pass at start_pos = p consumes
main_hidden(p) and t_{p+1}; draft row j sits at position p+1+j; output_ids = [t_{p+1}, d_1 .. d_5]
are positions p+1 .. p+6, so d_1 .. d_5 are drafts for positions p+2 .. p+6.

Same conventions as golden_decode.py: fp8 dense weights stay fp8 and are dequantised exactly per
call by kernel.py's fp8_gemm; wo_a is dequantised to fp32 up front (the block-diagonal einsum
reads .weight directly); experts are np.memmap-backed FP4 and only routed-to slots are ever read;
act_quant / fp4_act_quant are identity; temperature 0, i.e. argmax (model.py:1288-1289).

Every pass is self-checked: the head (norm -> tied head -> Markov bias chain -> argmax ->
confidence) is recomputed independently from the raw weights and must agree; the index set every
draft row attends to must be [0 .. min(win, p+1)) ++ [win .. win+5) (model.py:1021-1029); the ring
slot p % win must hold this stage's RoPE'd kv_norm(wkv(main_x)) (1040-1041, 1065); and the
non-causal control: perturbing the NOISE token's embedding must move draft row 0's attention
output (rows 1-4 feed row 0 -- the block is bidirectional, so five 1-row passes cannot
reproduce it).

  usage: golden_dspark.py <out_dir> [model_dir] [decode_golden_dir] [n_positions]
         golden_dspark.py --selftest [out_dir]     tiny random shapes, one core, no checkpoint
"""
import json, os, struct, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SELFTEST = "--selftest" in sys.argv
argv = [a for a in sys.argv[1:] if a != "--selftest"]
D = argv[1] if len(argv) > 1 else os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
if SELFTEST:
    OUT = argv[0] if argv else os.path.join(tempfile.gettempdir(), "dspark_selftest")
else:
    if not argv: sys.exit(__doc__)
    OUT = argv[0]
DD = argv[2] if len(argv) > 2 else f"{D}/ie_golden/decode2"
N_POS = int(argv[3]) if len(argv) > 3 else None
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE); sys.path.append(os.path.join(D, "inference"))   # our kernel.py shim must win

import torch
import torch.nn.functional as F
if SELFTEST: torch.set_num_threads(1)
import kernel as K
import model as ref
import warnings; warnings.filterwarnings("ignore")

ref.world_size, ref.rank = 1, 0
T_START = time.time()

# ---- dumps ---------------------------------------------------------------------------------
manifest = {}
def save(name, t):
    a = t.detach().contiguous().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    ext = ".i32" if a.dtype.kind in "iu" else ".f32"
    a = a.astype(np.int32 if ext == ".i32" else np.float32)
    a.tofile(f"{OUT}/{name}{ext}"); manifest[name + ext] = list(a.shape)

# ---- the model args ------------------------------------------------------------------------
if SELFTEST:
    # dim 64, 4 heads, 8 experts / 2 activated, vocab 512, 2 stages, window 16 (small enough to
    # wrap), 3 target layers so main_proj is [dim, 3 dim] like the real one
    args = ref.ModelArgs(
        dtype="bf16", expert_dtype=None, max_batch_size=1, max_seq_len=256, temperature=0,
        vocab_size=512, dim=64, moe_inter_dim=32, n_layers=4, n_mtp_layers=2, n_heads=4,
        n_routed_experts=8, n_shared_experts=1, n_activated_experts=2, score_func="sqrtsoftplus",
        norm_topk_prob=True, route_scale=1.5, swiglu_limit=10.0,
        q_lora_rank=32, head_dim=32, rope_head_dim=16, norm_eps=1e-20, o_groups=2, o_lora_rank=16,
        window_size=16, compress_ratios=(0,) * 6, kv_source_layers=(), index_source_layers=(),
        rope_theta=10000.0, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        dspark_block_size=5, dspark_noise_token_id=511, dspark_target_layer_ids=(1, 2, 3),
        dspark_markov_rank=8, dspark_n_routed_experts=8, dspark_n_activated_experts=2, vision_n_layers=0,
    )
    ref.default_dtype = torch.float32          # plain fp32 Linear weights: linear() -> F.linear
else:
    cfg = json.load(open(f"{D}/config.json")); tc = cfg["text_config"]; rs = tc["rope_scaling"]
    args = ref.ModelArgs(
        dtype="fp8", expert_dtype="fp4", max_batch_size=1, max_seq_len=4096, temperature=0,
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
        engram_layer_ids=tuple(tc["engram_layer_ids"]),
        vision_n_layers=cfg["vision_config"]["num_hidden_layers"], image_token_id=cfg["image_token_id"],
        # the drafter (config.json text_config; docs/deepseek41/48 A.0)
        dspark_block_size=tc["dspark_block_size"], dspark_noise_token_id=tc["dspark_noise_token_id"],
        dspark_target_layer_ids=tuple(tc["dspark_target_layer_ids"]), dspark_markov_rank=tc["dspark_markov_rank"],
        dspark_n_routed_experts=tc["dspark_n_routed_experts"], dspark_n_activated_experts=tc["dspark_num_experts_per_tok"],
    )
    ref.default_dtype = torch.float8_e4m3fn
    assert args.temperature == 0, "the golden is greedy: sample() must be argmax (model.py:1287)"

H, HC, WIN, B = args.dim, args.hc_mult, args.window_size, args.dspark_block_size
NS, V, TGT = args.n_mtp_layers, args.vocab_size, args.dspark_target_layer_ids
TK = args.get_moe_config(args.n_layers)[1]
assert all(args.compress_ratios[args.n_layers + s] == 0 for s in range(NS)), "MTP layers are window-only"

# ---- the stages ----------------------------------------------------------------------------
def P(t): return torch.nn.Parameter(t, requires_grad=False)

if not SELFTEST:
    wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]; _h = {}; _hdr = {}
    from safetensors import safe_open
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

    def load_stage(s):
        """mtp.{s}: a Block's 30 tensors bound exactly as golden_decode.load_block binds
        layers.{L}, plus the six DSpark extras (docs/deepseek41/48 C.3)."""
        p = f"mtp.{s}."
        blk = ref.DSparkBlock(args.n_layers + s, args)
        assert blk.attn.compressor is None and blk.attn.indexer is None and blk.engram is None
        blk.attn_norm.weight = P(get(p + "attn_norm.weight").float()); blk.ffn_norm.weight = P(get(p + "ffn_norm.weight").float())
        a = blk.attn
        a.q_norm.weight = P(get(p + "attn.q_norm.weight").float()); a.kv_norm.weight = P(get(p + "attn.kv_norm.weight").float())
        a.attn_sink = P(get(p + "attn.attn_sink").float())
        put(a.wq_a, p + "attn.wq_a", "fp8"); put(a.wq_b, p + "attn.wq_b", "fp8"); put(a.wkv, p + "attn.wkv", "fp8")
        put(a.wo_a, p + "attn.wo_a", "fp8->f32"); put(a.wo_b, p + "attn.wo_b", "fp8")
        for nm in ("attn_fn", "attn_base", "attn_scale", "ffn_fn", "ffn_base", "ffn_scale"):
            setattr(blk, "hc_" + nm, P(get(p + "hc_" + nm).float()))
        blk.ffn.gate.weight = P(get(p + "ffn.gate.weight").float()); blk.ffn.gate.bias = P(get(p + "ffn.gate.bias").float())
        if blk.ffn.gate.bias_vl is not None: blk.ffn.gate.bias_vl = P(get(p + "ffn.gate.bias_vl").float())
        for e in range(blk.ffn.n_routed_experts):
            for nm in ("w1", "w2", "w3"): put(getattr(blk.ffn.experts[e], nm), f"{p}ffn.experts.{e}.{nm}", "fp4")
        for nm in ("w1", "w2", "w3"): put(getattr(blk.ffn.shared_experts, nm), f"{p}ffn.shared_experts.{nm}", "fp8")
        if s == 0:                                                  # model.py:1111-1114
            put(blk.main_proj, p + "main_proj", "fp8"); blk.main_norm.weight = P(get(p + "main_norm.weight").float())
        if s == NS - 1:                                             # model.py:1115-1118
            blk.norm.weight = P(get(p + "norm.weight").float())
            blk.markov_head.embed.weight = P(get(p + "markov_head.embed.weight").float())
            blk.markov_head.head.weight = P(get(p + "markov_head.head.weight").float())
            blk.confidence_head.proj.weight = P(get(p + "confidence_head.proj.weight").float())
        return blk

    t0 = time.time()
    stages = [load_stage(s) for s in range(NS)]
    # the tied embedding and head (model.py:1212-1213): the backbone's own, fp32 as in golden_decode
    emb = ref.ParallelEmbedding(V, H); emb.weight = P(get("embed.weight").float())
    head = ref.ParallelHead(V, H, args.norm_eps, args.hc_eps); head.weight = P(get("head.weight").float())
    print(f"{NS} DSpark stages bound ({stages[0].ffn.n_routed_experts} memmap-backed experts each, top-{TK}), "
          f"embed + head fp32: {time.time()-t0:.0f}s", flush=True)
else:
    g = torch.Generator().manual_seed(20260914)
    def init_stage(blk):
        for n, p in blk.named_parameters():
            if n.endswith("norm.weight") or n.endswith("_scale"): p.data.fill_(1.0)
            elif n.endswith("_base") or n.endswith("bias") or n.endswith("bias_vl"): p.data.zero_()
            elif n.endswith("attn_sink"): p.data.normal_(0, 0.1, generator=g)
            else: p.data.normal_(0, 1.0 / np.sqrt(p.shape[-1]), generator=g)
        # wo_a is built bf16 whatever default_dtype is (model.py:645-649) and the einsum at 1072
        # reads .weight directly: fp32 here, as put(..., "fp8->f32") makes it in the real run
        blk.attn.wo_a.weight = P(blk.attn.wo_a.weight.float()); blk.attn.wo_a.scale = None
        return blk
    stages = [init_stage(ref.DSparkBlock(args.n_layers + s, args)) for s in range(NS)]
    emb = ref.ParallelEmbedding(V, H); emb.weight.data.normal_(0, 1.0, generator=g)
    head = ref.ParallelHead(V, H, args.norm_eps, args.hc_eps); head.weight.data.normal_(0, 1.0 / np.sqrt(H), generator=g)
for blk in stages: blk.embed, blk.head = emb, head          # every stage, as Transformer.__init__ does (1210-1213)
last = stages[-1]

# ---- forward_spec, with the stage boundaries captured ---------------------------------------
def gate_capture(mod, inp, out):
    """golden_decode.py's router hook: indices, weights, and the TK-th vs (TK+1)-th score margin."""
    assert mod.score_func == "sqrtsoftplus"
    scores = F.softplus(inp[0].float() @ mod.weight.float().T / mod.gate_temp).sqrt()
    sel = (scores + mod.bias).topk(mod.topk + 1, dim=-1)
    return {"weights": out[0].clone(), "indices": out[1].clone(), "gap": (sel.values[:, mod.topk - 1] - sel.values[:, mod.topk]).clone()}

def run_spec(main_hidden, input_id, start_pos, dump=None):
    """Transformer.forward_spec (model.py:1275-1282), verbatim in structure, over the live stages.
    start_pos 0 only seeds the rings (1123-1125, 1280-1281). Returns everything captured."""
    ids = torch.tensor([input_id], dtype=torch.int64)
    cap, hooks = {}, []
    def grab(name, fn):
        def hook(mod, inp, out): cap.setdefault(name, []).append(fn(mod, inp, out))
        return hook
    for s, blk in enumerate(stages):
        hooks += [blk.attn_norm.register_forward_hook(grab(f"attn_in_{s}", lambda m, i, o: o.clone())),
                  blk.attn.register_forward_hook(grab(f"attn_out_{s}", lambda m, i, o: o.clone())),
                  blk.ffn_norm.register_forward_hook(grab(f"ffn_in_{s}", lambda m, i, o: o.clone())),
                  blk.ffn.register_forward_hook(grab(f"moe_out_{s}", lambda m, i, o: o.clone())),
                  blk.ffn.gate.register_forward_hook(grab(f"route_{s}", gate_capture))]
    hooks += [last.norm.register_forward_hook(grab("head_in", lambda m, i, o: i[0].clone())),
              last.head.register_forward_hook(grab("logits_base", lambda m, i, o: o.clone())),   # before the in-place Markov add (1151)
              last.markov_head.register_forward_hook(grab("markov", lambda m, i, o: (o[0].clone(), o[1].clone()))),
              last.confidence_head.register_forward_hook(grab("confidence", lambda m, i, o: o.clone()))]
    topk_seen = []; orig = ref.get_dspark_topk_idxs                    # the index set the module asks for
    def spy(*a, **kw):
        t = orig(*a, **kw); topk_seen.append(t.clone()); return t
    ref.get_dspark_topk_idxs = spy
    try:
        with torch.inference_mode():
            h, main_x = stages[0].forward_embed(main_hidden, ids)
            embed = h.clone()
            pre_mix = ref.make_identity_pre_mix(h, HC)
            outs = []
            for blk in stages:
                h, pre_mix = blk(h, start_pos, pre_mix, main_x)
                outs.append((h.clone(), pre_mix.clone()))
            res = None if start_pos == 0 else last.forward_head(h, pre_mix, ids)
    finally:
        ref.get_dspark_topk_idxs = orig
        for hk in hooks: hk.remove()
    r = {"main_hidden": main_hidden, "main_x": main_x, "embed": embed, "outs": outs, "cap": cap,
         "topk": topk_seen, "rings": [blk.attn.window_kv_cache[0].clone() for blk in stages], "res": res}
    if dump:
        save(dump + "main_hidden", main_hidden); save(dump + "main_x", main_x)
        for s in range(NS): save(dump + f"window_{s}", r["rings"][s])
        if res is not None:
            save(dump + "embed", embed)
            for s in range(NS):
                for nm in ("attn_in", "attn_out", "ffn_in", "moe_out"): save(dump + f"{nm}_{s}", cap[f"{nm}_{s}"][0])
                save(dump + f"layer_out_{s}", outs[s][0]); save(dump + f"ffn_pre_{s}", outs[s][1])
                rt = cap[f"route_{s}"][0]
                save(dump + f"route_idx_{s}", rt["indices"].to(torch.int32)); save(dump + f"route_w_{s}", rt["weights"]); save(dump + f"route_gap_{s}", rt["gap"])
                save(dump + f"topk_{s}", topk_seen[s][0])
            save(dump + "head_in", cap["head_in"][0]); save(dump + "logits_base", cap["logits_base"][0][0])
            save(dump + "logits", res[1][0]); save(dump + "markov_bias", torch.cat([m[0] for m in cap["markov"]]))
            save(dump + "markov_embed", torch.cat([m[1] for m in cap["markov"]]))
            save(dump + "output_ids", res[0][0].to(torch.int32)); save(dump + "confidence", res[2][0])
    return r

# ---- self-checks ---------------------------------------------------------------------------
def check_pass(r, input_id, pos):
    """The pass's own consistency, independent of the modules' composition."""
    cap, res = r["cap"], r["res"]
    # (a) the index set: all filled ring slots, then all five draft rows -- no causal mask (1021-1029)
    want = torch.cat([torch.arange(min(WIN, pos + 1)), WIN + torch.arange(B)]).int().view(1, 1, -1).expand(1, B, -1)
    assert len(r["topk"]) == NS and all(torch.equal(t, want) for t in r["topk"]), "draft index set"
    # (b) the ring: slot pos % win holds this stage's RoPE'd kv_norm(wkv(main_x)) (1040-1041, 1065)
    for s, blk in enumerate(stages):
        a = blk.attn
        with torch.inference_mode():
            row = a.kv_norm(a.wkv(r["main_x"])).clone(); ref.apply_rotary_emb(row[..., -a.rope_head_dim:], a.freqs_cis[pos:pos + 1])
        assert torch.allclose(row[0, 0], r["rings"][s][pos % WIN], rtol=1e-5, atol=1e-6), f"ring slot, stage {s}"
    # (c) the head, recomputed from the raw weights (1144-1156, 1077-1097)
    x = cap["head_in"][0][0]                                                     # [B, H] hc-collapsed, pre-norm
    xn = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + args.norm_eps) * last.norm.weight
    base = xn @ last.head.weight.T
    me, mh, cw = last.markov_head.embed.weight, last.markov_head.head.weight, last.confidence_head.proj.weight[0]
    ids, embs, biases = [input_id], [], []
    for i in range(B):
        e = me[ids[i]]; bias = mh @ e
        ids.append(int((base[i] + bias).argmax())); embs.append(e); biases.append(bias)
    conf = torch.cat([x, torch.stack(embs)], -1) @ cw
    assert torch.allclose(base, cap["logits_base"][0][0], rtol=1e-4, atol=1e-4), "base logits"
    assert torch.allclose(base + torch.stack(biases), res[1][0], rtol=1e-4, atol=1e-4), "biased logits"
    assert ids == res[0][0].tolist(), f"Markov chain: {ids} vs {res[0][0].tolist()}"
    assert torch.allclose(conf, res[2][0], rtol=1e-4, atol=1e-5), "confidence"
    assert res[0][0, 0].item() == input_id
    # (d) shapes and finiteness of everything the engine test will read
    shapes = {"main_x": (1, 1, H), "embed": (1, B, HC, H), "head_in": (1, B, H)}
    for s in range(NS):
        for nm in ("attn_in", "attn_out", "ffn_in", "moe_out"): shapes[f"{nm}_{s}"] = (1, B, H)
    for k, sh in shapes.items():
        t = r[k] if k in r else cap[k][0]
        assert tuple(t.shape) == sh and torch.isfinite(t).all(), f"{k} {tuple(t.shape)}"
    for s in range(NS):
        assert tuple(r["outs"][s][0].shape) == (1, B, HC, H) and torch.isfinite(r["outs"][s][0]).all()
        assert tuple(r["outs"][s][1].shape) == (1, B, HC) and tuple(cap[f"route_{s}"][0]["indices"].shape) == (B, TK)
        assert tuple(r["rings"][s].shape) == (WIN, args.head_dim) and torch.isfinite(r["rings"][s]).all()
    assert tuple(res[1].shape) == (1, B, V) and torch.isfinite(res[1]).all()
    assert tuple(res[2].shape) == (1, B) and torch.isfinite(res[2]).all()
    assert tuple(res[0].shape) == (1, B + 1) and all(0 <= t < V for t in ids)
    # each draft row's top-1 minus top-2 of its (biased) logits: the engine test's near-tie rule
    # for the discrete output_ids check, as ds41_decode_test.cpp:273 uses the backbone's margins
    top2 = res[1][0].topk(2, dim=-1).values
    return ids, conf, (top2[:, 0] - top2[:, 1]).tolist()

def noncausal_control(main_hidden, input_id, pos, ref_r):
    """Perturb the noise token's embedding rows 1-4 embed and re-run at the same position (the
    ring write is idempotent). Row 0's attention output must move: the drafter is bidirectional
    over its block (docs/deepseek41/48 A.3 fact 2)."""
    w = emb.weight.data; nid = args.dspark_noise_token_id
    saved = w[nid].clone(); w[nid] += 0.1 * torch.randn(H, generator=torch.Generator().manual_seed(1))
    try: r2 = run_spec(main_hidden, input_id, pos)
    finally: w[nid] = saved
    a0, b0 = ref_r["cap"]["attn_out_0"][0][0, 0], r2["cap"]["attn_out_0"][0][0, 0]
    rel = (a0 - b0).abs().max().item() / a0.abs().max().item()
    assert torch.equal(ref_r["embed"][0, 0], r2["embed"][0, 0]), "row 0's own input did not change"
    assert rel > 1e-4, f"row 0 unaffected by rows 1-4 ({rel:.1e}): the block would be causal"
    return rel

# ---- the run -------------------------------------------------------------------------------
def prefill_seed(main_hidden, token, dump):
    """forward_spec at start_pos 0: each stage's ring takes the last min(T, win) rows of
    kv_norm(wkv(main_x)), slot = pos % win (model.py:1044-1052). Checked slot by slot."""
    r = run_spec(main_hidden, token, 0, dump)
    T = main_hidden.shape[1]
    for s, blk in enumerate(stages):
        a = blk.attn
        with torch.inference_mode():
            rows = a.kv_norm(a.wkv(r["main_x"])).clone(); ref.apply_rotary_emb(rows[..., -a.rope_head_dim:], a.freqs_cis[:T])
        for p in range(max(0, T - WIN), T):
            assert torch.allclose(rows[0, p], r["rings"][s][p % WIN], rtol=1e-5, atol=1e-6), f"prefill ring slot {p}, stage {s}"
    return r

meta = {"selftest": SELFTEST, "dim": H, "hc_mult": HC, "window": WIN, "block_size": B, "n_stages": NS, "vocab": V,
        "n_routed": args.get_moe_config(args.n_layers)[0], "topk": TK, "markov_rank": args.dspark_markov_rank,
        "noise_token_id": args.dspark_noise_token_id, "target_layer_ids": list(TGT), "head_dim": args.head_dim,
        "rope_head_dim": args.rope_head_dim, "norm_eps": args.norm_eps, "positions": []}

if SELFTEST:
    gen = torch.Generator().manual_seed(7)
    for T0, label in ((20, "wrapped: prefill 20 > window 16, then 20, 21"), (6, "partial: prefill 6, then 6, 7")):
        for blk in stages: blk.attn.window_kv_cache.zero_()          # a fresh Attention's ring
        mh = torch.randn(1, T0, len(TGT) * H, generator=gen)
        prefill_seed(mh, 1, f"st{T0}_prefill_")
        for pos in (T0, T0 + 1):
            mh1 = torch.randn(1, 1, len(TGT) * H, generator=gen); tok = int(torch.randint(0, V - 1, (1,), generator=gen))
            r = run_spec(mh1, tok, pos, f"st{T0}_p{pos}_")
            ids, conf, margins = check_pass(r, tok, pos)
            ctrl = noncausal_control(mh1, tok, pos, r)
            meta["positions"].append({"case": label, "pos": pos, "input_id": tok, "output_ids": ids, "confidence": conf.tolist(),
                                      "draft_margin": margins, "control_rel": ctrl})
            print(f"  {label:45s} pos {pos:2d}: ids {ids} conf {[round(c, 3) for c in conf.tolist()]} row-0 control {ctrl:.2e}")
    el = time.time() - T_START
    print(f"selftest OK: {len(manifest)} dumps in {OUT}, {el:.1f}s on {torch.get_num_threads()} thread")
    assert el < 5.0, f"selftest took {el:.1f}s (> 5s)"
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(D)
    dm = json.load(open(f"{DD}/d_meta.json"))
    prompt_ids, all_ids, n_dec = dm["prompt_ids"], dm["all_ids"], dm["n_decode"]
    T0 = len(prompt_ids); assert dm["dim"] == H and dm["hc_mult"] == HC and dm["n_layers"] == args.n_layers
    n_pos = min(N_POS or n_dec, n_dec)
    consumed = []
    def main_hidden_from(prefix, T):
        """model.py:1262-1271: the mean over hc of the stream entering each target layer, i.e.
        the engram output at L if L is an engram layer, else the previous layer's output."""
        parts = []
        for L in TGT:
            n = f"{prefix}_engram_out_{L}" if L in args.engram_layer_ids else f"{prefix}_layer_out_{L - 1}"
            a = np.fromfile(f"{DD}/{n}.f32", np.float32); assert a.size == T * HC * H, f"{n}: {a.size} floats"
            consumed.append(n + ".f32"); parts.append(torch.from_numpy(a.reshape(1, T, HC, H)).mean(2))
        return torch.cat(parts, -1)

    t0 = time.time()
    mh_p = main_hidden_from("p", T0)
    prefill_seed(mh_p, all_ids[T0], "sp_prefill_")
    print(f"prefill seeding from p_layer_out_{{{','.join(str(L-1) for L in TGT)}}} ({T0} positions): {time.time()-t0:.0f}s", flush=True)
    for k in range(n_pos):
        pos, prefix = T0 + k, ("d" if k == 0 else f"s{k}")
        tok_in = all_ids[pos + 1]                                        # the backbone's sample from this step
        g = np.fromfile(f"{DD}/d_logits_step{k + 1}.f32", np.float32); consumed.append(f"d_logits_step{k + 1}.f32")
        assert int(g.argmax()) == tok_in, f"step {k}: all_ids[{pos + 1}] = {tok_in} but argmax(d_logits_step{k + 1}) = {int(g.argmax())}"
        t1 = time.time()
        r = run_spec(main_hidden_from(prefix, 1), tok_in, pos, f"sp{pos}_")
        ids, conf, margins = check_pass(r, tok_in, pos)
        ctrl = noncausal_control(main_hidden_from(prefix, 1), tok_in, pos, r) if k == 0 else None
        # informational: drafts d_1..d_5 predict positions pos+2..pos+6; the backbone's own greedy tokens exist up to len(all_ids)-1
        cmp = [(ids[1 + j], all_ids[pos + 2 + j]) for j in range(B) if pos + 2 + j < len(all_ids)]
        flips = [int(cap["gap"].min() < 3e-3) for cap in (r["cap"][f"route_{s}"][0] for s in range(NS))]
        rec = {"pos": pos, "prefix": prefix, "input_id": tok_in, "output_ids": ids, "confidence": conf.tolist(),
               "draft_margin": margins, "drafts_text": [tok.decode([t]) for t in ids[1:]], "backbone_match": [a == b for a, b in cmp],
               "route_gap_min": [float(r["cap"][f"route_{s}"][0]["gap"].min()) for s in range(NS)],
               "unique_experts": [int(r["cap"][f"route_{s}"][0]["indices"].unique().numel()) for s in range(NS)],
               "control_rel": ctrl, "seconds": round(time.time() - t1, 1)}
        meta["positions"].append(rec)
        print(f"  pos {pos}: in {tok_in} ({tok.decode([tok_in])!r}) -> drafts {ids[1:]} {[tok.decode([t]) for t in ids[1:]]}"
              f"  conf {[round(c, 3) for c in conf.tolist()]}  backbone match {sum(a == b for a, b in cmp)}/{len(cmp)}"
              f"  near-tie stages {flips}  {rec['seconds']}s" + (f"  row-0 control {ctrl:.2e}" if ctrl is not None else ""), flush=True)
    meta.update({"prompt": dm["prompt"], "prompt_ids": prompt_ids, "all_ids": all_ids, "decode_golden_dir": DD,
                 "consumed": sorted(set(consumed)), "seconds_total": round(time.time() - T_START, 1)})
    print(f"wrote DSpark goldens to {OUT} in {time.time()-T_START:.0f}s")

meta["files"] = manifest
json.dump(meta, open(f"{OUT}/sp_meta.json", "w"), indent=1)
