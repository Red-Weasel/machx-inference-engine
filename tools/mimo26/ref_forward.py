#!/usr/bin/env python3
"""An fp32 reference forward of MiMo-V2.6 straight from the checkpoint (docs/mimo26/00_PORT_PLAN.md, P2 arbiter).

  ref_forward.py <model_dir> <ids.txt> <out.bin> [--max-seqs N] [--dump-dir DIR]

Semantics transcribed from the checkpoint's own modeling_mimo_v2.py (MiMoV2Attention / MiMoV2MoEGate / MiMoV2MoE /
MiMoV2RotaryEmbedding), weights dequantised EXACTLY (FP8 x F32 128-block scale_inv with llama.cpp's TP-aware qkv
regroup -- verified bit-exact against the engine in P1; MXFP4 E2M1 x 2^(e-127); BF16), activations fp32 throughout
(norms, softmax and routing in fp32/fp64). No fp16 anywhere: this is the "exact math" both the engine and llama.cpp
approximate. Writes MLOG (tools/mimo26/compare_logits.py) with the logits at every position of each sequence.
--dump-dir writes the residual stream after every layer, x_L{L}_s{S}.f32 [T, H], for a per-layer bisect.
"""
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

torch.set_grad_enabled(False)
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


class Store:
    def __init__(self, d):
        self.d = d
        self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.h = {}

    def get(self, name):
        f = self.idx[name]
        if f not in self.h:
            self.h[f] = safe_open(os.path.join(self.d, f), framework="pt")
        return self.h[f].get_tensor(name)


def fp8_dequant(w, s, bs=128):
    n, k = w.shape
    sf = s.float().repeat_interleave(bs, 0)[:n].repeat_interleave(bs, 1)[:, :k]
    return w.float() * sf


def mxfp4(st, name, n, k):
    qs = st.get(name).view(torch.uint8).reshape(n, k // 2)
    sc = st.get(name + "_scale").view(torch.uint8).reshape(n, k // 32)
    codes = torch.stack([qs & 0xF, qs >> 4], dim=-1).reshape(n, k).long()
    vals = E2M1[codes].reshape(n, k // 32, 32)
    scale = torch.ldexp(torch.ones_like(sc, dtype=torch.float32), sc.int() - 127)
    return (vals * scale.unsqueeze(-1)).reshape(n, k)


def rms(x, w, eps):
    v = x.double().pow(2).mean(-1, keepdim=True)
    return (x.double() * torch.rsqrt(v + eps)).float() * w


def rope_neox(x, pos, dim, theta):
    # x [T, heads, hd]: rotate the first `dim` dims, rotate_half pairing (i, i + dim/2)
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    ang = pos.double()[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1).float()[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1).float()[:, None, :]
    r, rest = x[..., :dim], x[..., dim:]
    h = dim // 2
    rot = torch.cat([-r[..., h:], r[..., :h]], -1)
    return torch.cat([r * cos + rot * sin, rest], -1)


def main():
    model, ids_path, out_path = sys.argv[1:4]
    max_seqs = int(sys.argv[sys.argv.index("--max-seqs") + 1]) if "--max-seqs" in sys.argv else None
    dump = sys.argv[sys.argv.index("--dump-dir") + 1] if "--dump-dir" in sys.argv else None
    if dump:
        os.makedirs(dump, exist_ok=True)
    cfg = json.load(open(os.path.join(model, "config.json")))
    st = Store(model)
    seqs = [list(map(int, l.replace(",", " ").split())) for l in open(ids_path) if l.strip()]
    if max_seqs:
        seqs = seqs[:max_seqs]
    H, nq, hd, vhd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["head_dim"], cfg["v_head_dim"]
    E, TK, EF, FI = cfg["n_routed_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"], cfg["intermediate_size"]
    eps, vscale = cfg["layernorm_epsilon"], cfg["attention_value_scale"]
    rdim = int(hd * cfg["partial_rotary_factor"])
    sys.path.insert(0, os.path.expanduser("~/llama.cpp-mimo"))
    from conversion.mimo import MimoV2Model

    emb = st.get("model.embed_tokens.weight")
    xs = [emb[torch.tensor(s)].float() for s in seqs]                       # per sequence [T, H]
    t0 = time.time()
    for L in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{L}."
        swa = cfg["hybrid_layer_pattern"][L] == 1
        nkv = cfg["swa_num_key_value_heads" if swa else "num_key_value_heads"]
        theta = cfg["swa_rope_theta"] if swa else cfg["rope_theta"]
        wqkv = MimoV2Model._tp_aware_qkv_dequant(st.get(p + "self_attn.qkv_proj.weight"), st.get(p + "self_attn.qkv_proj.weight_scale_inv"),
                                                 nq, nkv, hd, vhd).float()
        wo = st.get(p + "self_attn.o_proj.weight").float()
        an = st.get(p + "input_layernorm.weight").float()
        fn = st.get(p + "post_attention_layernorm.weight").float()
        sink = st.get(p + "self_attn.attention_sink_bias").float() if swa else None
        moe = cfg["moe_layer_freq"][L] == 1
        new_xs = []
        ffn_in = []
        for s, x in enumerate(xs):
            T = x.shape[0]
            h = rms(x, an, eps)
            qkv = h @ wqkv.T
            q = qkv[:, :nq * hd].reshape(T, nq, hd)
            k = qkv[:, nq * hd:nq * hd + nkv * hd].reshape(T, nkv, hd)
            v = qkv[:, nq * hd + nkv * hd:].reshape(T, nkv, vhd) * vscale
            pos = torch.arange(T)
            q, k = rope_neox(q, pos, rdim, theta), rope_neox(k, pos, rdim, theta)
            g = nq // nkv
            kk = k.repeat_interleave(g, dim=1).permute(1, 0, 2)            # [nq, T, hd]
            vv = v.repeat_interleave(g, dim=1).permute(1, 0, 2)            # [nq, T, vhd]
            sc = (q.permute(1, 0, 2).double() @ kk.double().transpose(1, 2)) / math.sqrt(hd)   # [nq, T, T]
            i = torch.arange(T)[:, None]; j = torch.arange(T)[None, :]
            allowed = j <= i
            if swa:
                allowed = allowed & (i - j < cfg["sliding_window"])
            sc = sc.masked_fill(~allowed[None], float("-inf"))
            if sink is not None:
                sc = torch.cat([sc, sink.double()[:, None, None].expand(nq, T, 1)], -1)
            pr = torch.softmax(sc, -1)
            if sink is not None:
                pr = pr[..., :-1]
            att = (pr @ vv.double()).float().permute(1, 0, 2).reshape(T, nq * vhd)
            x = x + att @ wo.T
            new_xs.append(x)
            ffn_in.append(rms(x, fn, eps))
        if not moe:
            wg = fp8_dequant(st.get(p + "mlp.gate_proj.weight"), st.get(p + "mlp.gate_proj.weight_scale_inv"))
            wu = fp8_dequant(st.get(p + "mlp.up_proj.weight"), st.get(p + "mlp.up_proj.weight_scale_inv"))
            wd = fp8_dequant(st.get(p + "mlp.down_proj.weight"), st.get(p + "mlp.down_proj.weight_scale_inv"))
            for s in range(len(xs)):
                h = ffn_in[s]
                new_xs[s] = new_xs[s] + (torch.nn.functional.silu(h @ wg.T) * (h @ wu.T)) @ wd.T
        else:
            gw = st.get(p + "mlp.gate.weight").float(); gb = st.get(p + "mlp.gate.e_score_correction_bias").float()
            allh = torch.cat(ffn_in, 0)
            scores = torch.sigmoid(allh @ gw.T)
            idx = torch.topk(scores + gb[None], TK, dim=-1).indices
            w = scores.gather(1, idx); w = w / (w.sum(-1, keepdim=True) + 1e-20)
            y = torch.zeros_like(allh)
            for e in torch.unique(idx).tolist():
                rows, slot = torch.where(idx == e)
                ep = p + f"mlp.experts.{e}."
                wg = mxfp4(st, ep + "gate_proj.weight", EF, H); wu = mxfp4(st, ep + "up_proj.weight", EF, H); wd = mxfp4(st, ep + "down_proj.weight", H, EF)
                hh = allh[rows]
                y.index_add_(0, rows, ((torch.nn.functional.silu(hh @ wg.T) * (hh @ wu.T)) @ wd.T) * w[rows, slot][:, None])
            off = 0
            for s in range(len(xs)):
                T = xs[s].shape[0]; new_xs[s] = new_xs[s] + y[off:off + T]; off += T
        xs = new_xs
        if dump:
            for s, x in enumerate(xs):
                x.numpy().astype("<f4").tofile(os.path.join(dump, f"x_L{L}_s{s}.f32"))
        print(f"layer {L} done ({time.time() - t0:.0f} s)", flush=True)
    nw = st.get("model.norm.weight").float(); head = st.get("lm_head.weight").float()
    with open(out_path, "wb") as f:
        f.write(b"MLOG"); f.write(struct.pack("<II", len(seqs), head.shape[0]))
        for s, x in zip(seqs, xs):
            lg = (rms(x, nw, eps) @ head.T).numpy().astype("<f4")
            f.write(struct.pack("<I", len(s))); f.write(np.asarray(s, dtype="<i4").tobytes()); f.write(lg.tobytes())
    print("wrote", out_path)


if __name__ == "__main__":
    main()
