"""Mini qwen4exp end-to-end oracle (2026-11-08, Qwen 4 day-one readiness).

Builds a TINY random model out of the HF reference classes (modeling_qwen4_exp.py, the same file the block oracle runs),
writes it as a GGUF in llama.cpp's qwen4exp layout (every converter transform gguf_weights.py inverts, applied forward),
and saves the reference's fp32 CPU logits for a fixed token sequence. `ie-q4e-mini-logits` runs the engine on the same
file; `compare` scores the two. No download, no real weights: 4 blocks (3 Gated DeltaNet + 1 full attention), 16 experts.

Why: Qwen 4 may change the shape of what Flash-Next ships (the 27B rumour: dense FFN, hidden 5120, 4 KV heads, no QSA
weights). Each variant below is one such shape; the engine must load it and reproduce the reference's logits before the
real weights finish downloading.

Weights the GGUF stores as Q8_0 are quantized -> dequantized IN THE HF MODEL before the reference runs, so both sides
compute with the same weights; F16-stored tensors are rounded to fp16 likewise. What remains is compute precision.

  python mini_model.py build --variant flashnext --out-dir DIR [--tokens 24] [--seed 7]
  python mini_model.py compare --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

import qwen4exp_ref as ref

BASE = dict(
    num_hidden_layers=4,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    num_experts=16,
    num_experts_per_tok=4,
    vocab_size=4096,
    eos_token_id=4094,
    bos_token_id=4094,
    ple_layer_ids=[2],              # 1-based: block 1 (as Flash-Next)
    ngram_vocab_size_base=1000,     # 16 hash heads of ~1000 rows (the real table is ~20M rows a head)
    max_position_embeddings=4096,
)

# name -> (config overrides, extras). Extras: no_qsa (no indexer keys / tensors in the GGUF), dense_ffn (a dense SwiGLU MLP of
# that width replaces the MoE block in every layer: Qwen4ExpTextMLP, ffn_gate / ffn_up / ffn_down, feed_forward_length).
VARIANTS = {
    "flashnext": ({}, {}),
    "kv4": ({"num_key_value_heads": 4}, {}),
    "noqsa": ({}, {"no_qsa": True}),
    "h5120": ({"hidden_size": 5120}, {}),
    "dense": ({"hidden_size": 5120, "num_key_value_heads": 4}, {"no_qsa": True, "dense_ffn": 13824}),
    "dense2560": ({}, {"dense_ffn": 9216}),
    # the file named qwen4 with its keys under qwen4. (llama.cpp's qwen35 / qwen35moe naming, should Qwen 4 follow it)
    "arch_qwen4": ({}, {"arch": "qwen4"}),
    # the wider sweep (other sizes of the same parts a Qwen 4 tier may pick)
    "hd128": ({"head_dim": 128, "rope_parameters": {"mrope_interleaved": True, "mrope_section": [6, 5, 5],
                                                    "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                                                    "rope_type": "default"}}, {}),
    "q32kv8": ({"num_attention_heads": 32, "num_key_value_heads": 8}, {}),
    "moe64": ({"num_experts": 64, "num_experts_per_tok": 8, "moe_intermediate_size": 1024,
               "shared_expert_intermediate_size": 1024}, {}),
    "ssm32": ({"linear_num_value_heads": 32}, {}),
    "nople": ({"ple_layer_ids": []}, {}),
    "hc8": ({"hc_count": 8, "ple_layer_ids": []}, {}),
    "lowrank512": ({"hc_lowrank": 512}, {}),
    "fused_gu": ({}, {"fused_gu": True}),   # the experts' gate / up as one ffn_gate_up_exps bank
    # NEGATIVE CONTROL (must FAIL): the full-attention q / k norm gammas written without the converter's +1 -- one wrong
    # transform in one block; a gate that passes this is too weak. Not in the default list.
    "neg_qknorm": ({}, {"neg_qknorm": True}),
}


def _gguf():
    g = ref.import_gguf()
    import gguf.quants as gq
    return g, gq


# ---------------------------------------------------------------------------------------------------------------------
# model

def build_model(cfg, extras, seed):
    mod = ref.load_modeling()
    torch.manual_seed(seed)
    m = torch.nn.Module()
    m.embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    m.layers = torch.nn.ModuleList()
    for il in range(cfg.num_hidden_layers):
        lyr = mod.Qwen4ExpTextDecoderLayer(cfg, il)
        if extras.get("dense_ffn"):
            lyr.mlp = mod.Qwen4ExpTextMLP(cfg, intermediate_size=extras["dense_ffn"])
        m.layers.append(lyr)
    m.mixer = mod.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
    m.lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    gen = torch.Generator().manual_seed(seed)

    def rn(shape, sd):
        return torch.randn(*shape, generator=gen) * sd

    with torch.no_grad():
        for name, p in m.named_parameters():
            leaf = name.split(".")[-2] if name.endswith(".weight") else name.split(".")[-1]
            if name.endswith("linear_attn.norm.weight"):              # RMSNormGated: raw gamma (ones-init)
                p.copy_(1.0 + rn(p.shape, 0.1))
            elif "norm" in leaf or leaf in ("q_layernorm", "k_layernorm"):   # zero-centred RMSNorm: gamma = 1 + w
                p.copy_(rn(p.shape, 0.1))
            elif name.endswith("A_log"):
                p.copy_(torch.log(torch.rand(p.shape, generator=gen) * 7.5 + 0.5))
            elif name.endswith("dt_bias"):
                p.copy_(1.0 + rn(p.shape, 0.1))
            elif "conv1d" in name:
                p.copy_(rn(p.shape, 1.0 / math.sqrt(p.shape[-1])))
            elif name.endswith("experts.gate_up_proj"):
                p.copy_(rn(p.shape, 1.0 / math.sqrt(p.shape[-1])))
            elif name.endswith("experts.down_proj"):
                p.copy_(rn(p.shape, 1.0 / math.sqrt(p.shape[-1])))
            elif name.endswith("mlp.gate.weight"):                      # router: decisive picks (few near-ties)
                p.copy_(rn(p.shape, 3.0 / math.sqrt(p.shape[-1])))
            elif name.startswith("embed") or "ngram_embedding" in name:
                p.copy_(rn(p.shape, 0.5))
            elif name.startswith("lm_head"):                            # logits std ~3: the argmax check means something
                p.copy_(rn(p.shape, 8.0 / math.sqrt(p.shape[-1])))
            elif p.dim() == 2:
                p.copy_(rn(p.shape, 1.0 / math.sqrt(p.shape[-1])))
            else:
                p.copy_(rn(p.shape, 0.1))
    return mod, m.float().eval()


def router_gaps(cfg, m):
    """Hooks every MoE router; returns (dict, remove): dict[layer] = per-token gap between the k-th and (k+1)-th router
    logit. A gap at F16 resolution (~0.002-0.004 at these magnitudes) is a near-tie the engine's F16 routing may break the
    other way -- a different expert, a legitimate precision effect, not a bug."""
    gaps, hooks = {}, []
    for il, lyr in enumerate(m.layers):
        if not hasattr(lyr.mlp, "gate") or not hasattr(lyr.mlp.gate, "num_experts"):
            continue
        def h(_m, _inp, out, il=il):
            lg = out[0] if isinstance(out, tuple) else out
            srt = torch.sort(lg.float(), dim=-1, descending=True).values
            k = cfg.num_experts_per_tok
            gaps[il] = (srt[:, k - 1] - srt[:, k]).numpy()
        hooks.append(lyr.mlp.gate.register_forward_hook(h))
    return gaps, lambda: [x.remove() for x in hooks]


def forward(mod, cfg, m, ids):
    T = ids.shape[1]
    cos, sin = ref.make_position_embeddings(mod, cfg, T)
    mask = ref.make_full_attn_mask(T)
    with torch.no_grad():
        h = m.embed(ids).repeat(1, 1, cfg.hc_count)
        for lyr in m.layers:
            h = lyr(h, position_embeddings=(cos, sin), attention_mask=mask, conv_mask=None, past_key_values=None,
                    ple_input_ids=ids if lyr.ple is not None else None)
        return m.lm_head(m.mixer(h))[0]


# ---------------------------------------------------------------------------------------------------------------------
# HF -> GGUF (the converter's transforms; gguf_weights.py documents and inverts each one)

def grouped_to_tiled(t, dim, n_k, v_per_k, head_dim):
    """I4 forward: HF V heads grouped by K head -> the converter's tiled order (gguf_head[r*n_k+k] = hf_head[k*v_per_k+r])."""
    shape = list(t.shape)
    d = dim if dim >= 0 else dim + len(shape)
    ns = shape[:d] + [n_k, v_per_k, head_dim] + shape[d + 1:]
    t = t.reshape(*ns)
    perm = list(range(len(ns)))
    perm[d], perm[d + 1] = perm[d + 1], perm[d]
    return t.permute(*perm).contiguous().reshape(*shape)


Q8 = "Q8_0"
F16 = "F16"
F32 = "F32"


def round_to(p, kind, gq, QT):
    """Make the HF parameter hold exactly what the GGUF will (Q8_0 along the last dim, or fp16)."""
    with torch.no_grad():
        a = p.detach().float().numpy()
        if kind == Q8:
            assert a.shape[-1] % 32 == 0, (p.shape, "Q8_0 needs the row width a multiple of 32")
            b = gq.dequantize(gq.quantize(a, QT.Q8_0), QT.Q8_0).reshape(a.shape)
            p.copy_(torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)))
        elif kind == F16:
            p.copy_(torch.from_numpy(a.astype(np.float16).astype(np.float32)))


def tensor_plan(cfg, m, extras):
    """[(gguf name, kind, producer)] where producer() returns the HF-oriented fp32 array the converter would write,
    plus the list of (param, kind) to round before the reference runs."""
    plan, rounds = [], []
    nk, nv, kd, vd = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    vpk = nv // nk
    KEY = nk * kd

    def P(name, kind, fn, *params):
        plan.append((name, kind, fn))
        for p in params:
            rounds.append((p, kind))

    def t2n(x):
        return x.detach().float().numpy()

    P("token_embd.weight", Q8, lambda: t2n(m.embed.weight), m.embed.weight)
    P("output.weight", Q8, lambda: t2n(m.lm_head.weight), m.lm_head.weight)
    P("output_hc_norm.weight", F32, lambda: t2n(m.mixer.hc_norm.weight) + 1.0)
    P("output_hc_down.weight", F16, lambda: t2n(m.mixer.input_mix_weight_down.weight), m.mixer.input_mix_weight_down.weight)
    P("output_hc_up.weight", F16, lambda: t2n(m.mixer.input_mix_weight_up.weight), m.mixer.input_mix_weight_up.weight)

    for il, lyr in enumerate(m.layers):
        B = f"blk.{il}."
        if cfg.layer_types[il] == "linear_attention":
            la = lyr.linear_attn
            def qkv(la=la):
                w = t2n(la.in_proj_qkv.weight)
                return np.concatenate([w[: 2 * KEY], t2n(grouped_to_tiled(torch.from_numpy(w[2 * KEY:]), 0, nk, vpk, vd))], 0)
            P(B + "attn_qkv.weight", Q8, qkv, la.in_proj_qkv.weight)
            P(B + "attn_gate.weight", Q8, lambda la=la: t2n(grouped_to_tiled(la.in_proj_z.weight, 0, nk, vpk, vd)), la.in_proj_z.weight)
            P(B + "ssm_alpha.weight", F32, lambda la=la: t2n(grouped_to_tiled(la.in_proj_a.weight, 0, nk, vpk, 1)))
            P(B + "ssm_beta.weight", F32, lambda la=la: t2n(grouped_to_tiled(la.in_proj_b.weight, 0, nk, vpk, 1)))
            P(B + "ssm_a", F32, lambda la=la: t2n(grouped_to_tiled((-torch.exp(la.A_log)).unsqueeze(-1), 0, nk, vpk, 1).squeeze(-1)))
            P(B + "ssm_dt.bias", F32, lambda la=la: t2n(grouped_to_tiled(la.dt_bias.unsqueeze(-1), 0, nk, vpk, 1).squeeze(-1)))
            def conv(la=la):
                c = t2n(la.conv1d.weight.squeeze(1))
                return np.concatenate([c[: 2 * KEY], t2n(grouped_to_tiled(torch.from_numpy(c[2 * KEY:]), 0, nk, vpk, vd))], 0)
            P(B + "ssm_conv1d.weight", F32, conv)
            P(B + "ssm_out.weight", Q8, lambda la=la: t2n(grouped_to_tiled(la.out_proj.weight, 1, nk, vpk, vd)), la.out_proj.weight)
            P(B + "ssm_norm.weight", F32, lambda la=la: t2n(la.norm.weight))
        else:
            sa = lyr.self_attn
            P(B + "attn_q.weight", Q8, lambda sa=sa: t2n(sa.q_proj.weight), sa.q_proj.weight)
            P(B + "attn_k.weight", Q8, lambda sa=sa: t2n(sa.k_proj.weight), sa.k_proj.weight)
            P(B + "attn_v.weight", Q8, lambda sa=sa: t2n(sa.v_proj.weight), sa.v_proj.weight)
            P(B + "attn_output.weight", Q8, lambda sa=sa: t2n(sa.o_proj.weight), sa.o_proj.weight)
            g1 = 0.0 if extras.get("neg_qknorm") else 1.0
            P(B + "attn_q_norm.weight", F32, lambda sa=sa, g1=g1: t2n(sa.q_norm.weight) + g1)
            P(B + "attn_k_norm.weight", F32, lambda sa=sa, g1=g1: t2n(sa.k_norm.weight) + g1)
            if not extras.get("no_qsa"):
                ix = sa.indexer
                nq = ix.index_n_heads * ix.index_head_dim
                P(B + "indexer.q_proj.weight", F16, lambda ix=ix, nq=nq: t2n(ix.index_qk_proj.weight)[:nq], ix.index_qk_proj.weight)
                P(B + "indexer.k_proj.weight", F16, lambda ix=ix, nq=nq: t2n(ix.index_qk_proj.weight)[nq:])
                P(B + "indexer.q_norm.weight", F32, lambda ix=ix: t2n(ix.q_layernorm.weight) + 1.0)
                P(B + "indexer.k_norm.weight", F32, lambda ix=ix: t2n(ix.k_layernorm.weight) + 1.0)
        for hf, gg in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
            hc = getattr(lyr, hf)
            P(B + f"{gg}_norm.weight", F32, lambda hc=hc: t2n(hc.hc_norm.weight) + 1.0)
            P(B + f"{gg}_down.weight", F16, lambda hc=hc: t2n(hc.input_mix_weight_down.weight), hc.input_mix_weight_down.weight)
            P(B + f"{gg}_up.weight", F16, lambda hc=hc: t2n(hc.input_mix_weight_up.weight), hc.input_mix_weight_up.weight)
            P(B + f"{gg}_inject.weight", F32, lambda hc=hc: t2n(hc.block_inject_weight.weight))
        mlp = lyr.mlp
        if extras.get("dense_ffn"):
            P(B + "ffn_gate.weight", Q8, lambda mlp=mlp: t2n(mlp.gate_proj.weight), mlp.gate_proj.weight)
            P(B + "ffn_up.weight", Q8, lambda mlp=mlp: t2n(mlp.up_proj.weight), mlp.up_proj.weight)
            P(B + "ffn_down.weight", Q8, lambda mlp=mlp: t2n(mlp.down_proj.weight), mlp.down_proj.weight)
        else:
            F = cfg.moe_intermediate_size
            P(B + "ffn_gate_inp.weight", F16, lambda mlp=mlp: t2n(mlp.gate.weight), mlp.gate.weight)
            if extras.get("fused_gu"):   # llama.cpp --fuse-gate-up-exps: one [E, 2F, H] bank, HF's gate_up_proj as is
                P(B + "ffn_gate_up_exps.weight", Q8, lambda mlp=mlp: t2n(mlp.experts.gate_up_proj), mlp.experts.gate_up_proj)
            else:
                P(B + "ffn_gate_exps.weight", Q8, lambda mlp=mlp: t2n(mlp.experts.gate_up_proj)[:, :F], mlp.experts.gate_up_proj)
                P(B + "ffn_up_exps.weight", Q8, lambda mlp=mlp: t2n(mlp.experts.gate_up_proj)[:, F:])
            P(B + "ffn_down_exps.weight", Q8, lambda mlp=mlp: t2n(mlp.experts.down_proj), mlp.experts.down_proj)
            se = mlp.shared_expert
            P(B + "ffn_gate_shexp.weight", Q8, lambda se=se: t2n(se.gate_proj.weight), se.gate_proj.weight)
            P(B + "ffn_up_shexp.weight", Q8, lambda se=se: t2n(se.up_proj.weight), se.up_proj.weight)
            P(B + "ffn_down_shexp.weight", Q8, lambda se=se: t2n(se.down_proj.weight), se.down_proj.weight)
            P(B + "ffn_gate_inp_shexp.weight", F16, lambda mlp=mlp: t2n(mlp.shared_expert_gate.weight).reshape(-1),
              mlp.shared_expert_gate.weight)
        if lyr.ple is not None:
            pl = lyr.ple
            P(B + "ple_key.weight", F16, lambda pl=pl: t2n(pl.key_proj.weight), pl.key_proj.weight)
            P(B + "ple_value.weight", F16, lambda pl=pl: t2n(pl.value_proj.weight), pl.value_proj.weight)
            P(B + "ple_norm_key.weight", F32, lambda pl=pl: t2n(pl.norm_key.weight) + 1.0)
            P(B + "ple_norm_query.weight", F32, lambda pl=pl: t2n(pl.norm_query.weight) + 1.0)
            P(B + "ple_norm_conv.weight", F32, lambda pl=pl: t2n(pl.norm_conv.weight) + 1.0)
            P(B + "ple_conv1d.weight", F32, lambda pl=pl: t2n(pl.conv1d.weight.squeeze(1)))
    ples = [lyr.ple for lyr in m.layers if lyr.ple is not None]
    if ples:
        P("per_layer_token_embd.weight", Q8, lambda: t2n(ples[0].ple_embedding.ngram_embedding.weight),
          ples[0].ple_embedding.ngram_embedding.weight)
    return plan, rounds


def write_gguf(path, cfg, m, extras, plan):
    gguf, gq = _gguf()
    QT = gguf.GGMLQuantizationType
    VT = gguf.GGUFValueType
    arch = extras.get("arch", "qwen4exp")
    w = gguf.GGUFWriter(path, arch)
    A = arch + "."
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    w.add_uint32(A + "block_count", cfg.num_hidden_layers)
    w.add_uint32(A + "context_length", cfg.max_position_embeddings)
    w.add_uint32(A + "embedding_length", cfg.hidden_size)
    w.add_uint32(A + "attention.head_count", cfg.num_attention_heads)
    w.add_uint32(A + "attention.head_count_kv", cfg.num_key_value_heads)
    w.add_uint32(A + "attention.key_length", cfg.head_dim)
    w.add_uint32(A + "attention.value_length", cfg.head_dim)
    w.add_float32(A + "attention.layer_norm_rms_epsilon", cfg.rms_norm_eps)
    w.add_uint32(A + "full_attention_interval", cfg.full_attention_interval)
    rp = cfg.rope_parameters
    rope_dim = int(cfg.head_dim * rp["partial_rotary_factor"])
    w.add_uint32(A + "rope.dimension_count", rope_dim)
    w.add_float32(A + "rope.freq_base", float(rp["rope_theta"]))
    w.add_array(A + "rope.dimension_sections", list(rp["mrope_section"]) + [0])
    w.add_uint32(A + "ssm.state_size", cfg.linear_key_head_dim)
    w.add_uint32(A + "ssm.conv_kernel", cfg.linear_conv_kernel_dim)
    w.add_uint32(A + "ssm.group_count", cfg.linear_num_key_heads)
    w.add_uint32(A + "ssm.time_step_rank", cfg.linear_num_value_heads)
    w.add_uint32(A + "ssm.inner_size", cfg.linear_num_value_heads * cfg.linear_value_head_dim)
    w.add_uint32(A + "hyper_connection.count", cfg.hc_count)
    w.add_uint32(A + "hyper_connection.low_rank", cfg.hc_lowrank)
    if not extras.get("no_qsa"):
        w.add_uint32(A + "attention.indexer.head_count", cfg.indexer_n_heads)
        w.add_uint32(A + "attention.indexer.key_length", cfg.indexer_head_dim)
        w.add_uint32(A + "attention.indexer.top_k", cfg.indexer_budget)
        w.add_array(A + "attention.compress_ratios",
                    [cfg.indexer_compress_ratio if i in full else 0 for i in range(cfg.num_hidden_layers)])
    if extras.get("dense_ffn"):
        w.add_uint32(A + "feed_forward_length", extras["dense_ffn"])
    else:
        w.add_uint32(A + "expert_count", cfg.num_experts)
        w.add_uint32(A + "expert_used_count", cfg.num_experts_per_tok)
        w.add_uint32(A + "expert_feed_forward_length", cfg.moe_intermediate_size)
        w.add_uint32(A + "expert_shared_feed_forward_length", cfg.shared_expert_intermediate_size)
    ples = [lyr.ple for lyr in m.layers if lyr.ple is not None]
    if ples:
        ne = ples[0].ple_embedding
        w.add_array(A + "ple.layers", [int(i) - 1 for i in cfg.ple_layer_ids])
        w.add_uint32(A + "ple.ngram_size", cfg.ngram_size)
        w.add_uint32(A + "ple.heads_per_ngram", cfg.heads_per_ngram)
        w.add_uint32(A + "ple.conv_kernel", cfg.ple_conv_kernel_size)
        w.add_uint32(A + "embedding_length_per_layer_input", cfg.ple_embed_dim // ne.ngram_heads)
        w.add_uint32(A + "ple.eos_token_id", int(ne.eos_token_id))
        w.add_uint32(A + "ple.image_token_id", cfg.vocab_size - 1)
        w.add_key_value(A + "ple.layer_multipliers", [int(x) for x in ne.layer_multipliers.tolist()], VT.ARRAY, VT.UINT64)
        w.add_key_value(A + "ple.head_offsets", [int(x) for x in ne.ngram_heads_offsets.tolist()], VT.ARRAY, VT.UINT64)
        w.add_key_value(A + "ple.head_vocab_sizes", [int(x) for x in ne.ngram_heads_vocab_sizes.tolist()], VT.ARRAY, VT.UINT64)
    if extras.get("tokenizer_from"):
        copy_tokenizer(w, extras["tokenizer_from"])
    else:
        w.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(cfg.vocab_size)])
    for name, kind, fn in plan:
        a = np.ascontiguousarray(fn(), dtype=np.float32)
        if kind == Q8:
            w.add_tensor(name, gq.quantize(a, QT.Q8_0), raw_dtype=QT.Q8_0)   # (the byte shape gives the element shape)
        elif kind == F16:
            w.add_tensor(name, a.astype(np.float16))
        else:
            w.add_tensor(name, a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def tokenizer_fields(path):
    """[(name, value, type, sub_type)] of every tokenizer.* key in a GGUF (the real model's tokenizer, for a mini file
    that `ie serve` can load and chat with)."""
    gguf, _ = _gguf()
    r = gguf.GGUFReader(path)
    out = []
    for f in r.fields.values():
        if not f.name.startswith("tokenizer."):
            continue
        vt = f.types[0]
        sub = f.types[1] if vt == gguf.GGUFValueType.ARRAY and len(f.types) > 1 else None
        out.append((f.name, f.contents(), vt, sub))
    return out


def copy_tokenizer(w, path):
    for name, val, vt, sub in tokenizer_fields(path):
        w.add_key_value(name, val, vt, sub)


def make_config(variant, tokenizer_from=None, cycles=1, max_pos=None):
    over, extras = VARIANTS[variant]
    extras = dict(extras)
    base = dict(BASE)
    if cycles > 1:
        # deeper models for the N-stage chain gate (ie-qwen4exp-run2 needs >= 2 blocks a stage): the 4-block cycle repeated
        base.update(num_hidden_layers=4 * cycles, layer_types=BASE["layer_types"] * cycles)
    if max_pos:
        base.update(max_position_embeddings=max_pos)
    if tokenizer_from:
        # The real vocabulary (Flash-Next: 248320 tokens, EOS / PLE reset 248044): the file then carries the real
        # tokenizer and `ie serve` loads it -- the engine-level smoke for a shape. The logits stay random-weight logits.
        f = {n: v for n, v, _, _ in tokenizer_fields(tokenizer_from)}
        eos = int(f.get("tokenizer.ggml.eos_token_id", 248044))
        pad = int(f.get("tokenizer.ggml.padding_token_id", eos))
        base.update(vocab_size=len(f["tokenizer.ggml.tokens"]), eos_token_id=pad, bos_token_id=pad)
        extras["tokenizer_from"] = tokenizer_from
    cfg = ref.build_text_config(**{**base, **over})
    return cfg, extras


def cmd_build(args):
    cfg, extras = make_config(args.variant, args.tokenizer_from, args.cycles, args.max_pos)
    os.makedirs(args.out_dir, exist_ok=True)
    mod, m = build_model(cfg, extras, args.seed)
    gguf, gq = _gguf()
    plan, rounds = tensor_plan(cfg, m, extras)
    for p, kind in rounds:
        round_to(p, kind, gq, gguf.GGMLQuantizationType)
    gen = torch.Generator().manual_seed(args.seed + 1)
    ids = torch.randint(0, cfg.vocab_size - 2, (1, args.tokens), generator=gen)
    ids[0, args.tokens // 2] = cfg.eos_token_id          # one EOS mid-sequence: the PLE n-gram reset
    gaps, unhook = router_gaps(cfg, m)
    logits = forward(mod, cfg, m, ids)
    unhook()
    gap = np.min(np.stack(list(gaps.values())), axis=0) if gaps else np.full(args.tokens, np.inf)
    gap.astype(np.float32).tofile(os.path.join(args.out_dir, "ref_router_gap.f32"))
    path = os.path.join(args.out_dir, "mini.gguf")
    write_gguf(path, cfg, m, extras, plan)
    ids.numpy().astype(np.int32).tofile(os.path.join(args.out_dir, "ids.i32"))
    logits.numpy().astype(np.float32).tofile(os.path.join(args.out_dir, "ref_logits.f32"))
    meta = dict(variant=args.variant, tokens=args.tokens, vocab=cfg.vocab_size, seed=args.seed, extras=extras,
                hidden=cfg.hidden_size, kv_heads=cfg.num_key_value_heads, layers=cfg.num_hidden_layers,
                gguf_bytes=os.path.getsize(path))
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=1)
    print(f"built {args.variant}: {path} ({meta['gguf_bytes'] / 2**20:.1f} MiB), {args.tokens} tokens, "
          f"ref logits std {float(logits.std()):.3f}")


def cmd_compare(args):
    meta = json.load(open(os.path.join(args.out_dir, "meta.json")))
    T, V = meta["tokens"], meta["vocab"]
    ref_l = np.fromfile(os.path.join(args.out_dir, "ref_logits.f32"), dtype=np.float32).reshape(T, V)
    eng = np.fromfile(os.path.join(args.out_dir, "engine_logits.f32"), dtype=np.float32).reshape(-1, V)
    dec, pre_last, split_last = eng[:T], eng[T], eng[T + 1]

    def score(a, b):
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        rel = float(np.max(np.abs(a - b)) / (np.std(b) + 1e-30))
        # an argmax difference counts only where the reference's top-1 leads by > argmax_margin x the row's std (a closer
        # top two is an output tie: the order carries no signal at these precisions; cosine still scores the row)
        top2 = np.sort(b)[-2:]
        decisive = (top2[1] - top2[0]) > args.argmax_margin * (np.std(b) + 1e-30)
        return cos, rel, int(np.argmax(a) == np.argmax(b) or not decisive)

    rows = [score(dec[t], ref_l[t]) for t in range(T)]
    # Near-tie rows: the reference's k-th / (k+1)-th router logits within args.tie_gap in some layer (ref_router_gap.f32,
    # the min over layers per token). The engine's F16 routing may pick the other expert there: those rows are held to
    # args.tie_cos and their argmax does not count; every other row to args.min_cos.
    gp = os.path.join(args.out_dir, "ref_router_gap.f32")
    gap = np.fromfile(gp, dtype=np.float32) if os.path.exists(gp) else np.full(T, np.inf, dtype=np.float32)
    tie = [bool(gap[t] < args.tie_gap) for t in range(T)]
    strict = [r for t, r in enumerate(rows) if not tie[t]]
    tied = [r for t, r in enumerate(rows) if tie[t]]
    cmin = min(r[0] for r in strict) if strict else 1.0
    tmin = min(r[0] for r in tied) if tied else 1.0
    rmax = max(r[1] for r in rows)
    am = sum(r[2] for r in strict)
    cp, rp, ap = score(pre_last, ref_l[T - 1])
    cs, rs, as_ = score(split_last, ref_l[T - 1])
    print(f"[{meta['variant']}] decode vs ref: min cos {cmin:.6f}  max |d|/std {rmax:.4f}  argmax {am}/{len(strict)}"
          + (f"  (+{len(tied)} near-tie row(s) {[t for t in range(T) if tie[t]]}: min cos {tmin:.6f})" if tied else ""))
    print(f"[{meta['variant']}] prefill last vs ref: cos {cp:.6f}  |d|/std {rp:.4f}  argmax {'=' if ap else 'DIFFERS'}")
    print(f"[{meta['variant']}] split prefill last vs ref: cos {cs:.6f}  |d|/std {rs:.4f}  argmax {'=' if as_ else 'DIFFERS'}")
    # The split prefill's second chunk (T - T/2 rows after a cached prefix) runs the small-batch int8-activation kernels and
    # reads the cached KV / DeltaNet state, as decode does: it is held to the decode floor. (2026-11-08: at 8 blocks the
    # cached path's 1 - cos is ~2e-4 on every shape -- dense 2.4e-4, flashnext 1.3e-4, flat over positions, the engine's
    # own decode vs prefill as far apart -- while one-shot prefill stays at ~2e-5; the 4-block calibration had held the
    # split row to the one-shot bar.) The one-shot row stays the sharp instrument (neg_qknorm: 0.9985).
    ok = (cmin >= args.min_cos and tmin >= args.tie_cos and cp >= args.prefill_cos and cs >= args.min_cos and
          am >= len(strict) - args.argmax_slack)
    print(f"[{meta['variant']}] {'PASS' if ok else 'FAIL'} (decode + split prefill cos >= {args.min_cos}, prefill >= {args.prefill_cos}, "
          f"near-tie rows >= {args.tie_cos}, "
          f"argmax misses <= {args.argmax_slack})")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--tokens", type=int, default=24)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--cycles", type=int, default=1,
                   help="repeat the 4-block layer cycle (2 = 8 blocks: enough for a 4-stage ie-qwen4exp-run2 chain)")
    b.add_argument("--max-pos", type=int, default=None, help="max_position_embeddings (default 4096)")
    b.add_argument("--tokenizer-from", default=None,
                   help="copy tokenizer.* from this GGUF (the real vocabulary): the file then loads in ie serve")
    c = sp.add_parser("compare")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--min-cos", type=float, default=0.999)
    # prefill is the sharper instrument (fp16 activations, no int8 rounding): correct runs 0.99989-0.99998 over 14 shapes,
    # the neg_qknorm control (one wrong norm transform) 0.9985
    c.add_argument("--prefill-cos", type=float, default=0.9998)
    c.add_argument("--argmax-slack", type=int, default=1)
    c.add_argument("--tie-gap", type=float, default=0.02, help="a router gap below this marks a near-tie row")
    c.add_argument("--argmax-margin", type=float, default=0.05,
                   help="an argmax miss counts when the reference's top-1 leads by more than this x the row's std")
    c.add_argument("--tie-cos", type=float, default=0.995, help="the cosine floor of a near-tie row")
    args = ap.parse_args()
    if args.cmd == "build":
        cmd_build(args)
        return 0
    return cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
