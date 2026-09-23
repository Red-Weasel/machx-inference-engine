#!/usr/bin/env python3
"""P5 feasibility (docs/mimo26/00_PORT_PLAN.md, P5 DESIGN): the DFlash drafter's acceptance on the engine's own greedy
continuation, offline -- before any engine integration.

  dflash_accept.py <model_dir> <dump_dir> <ids.txt> <n_prompt> [--variants plain,sink] [--max-pos N] [--call N]
                   [--mask-emb file|target]

<dump_dir> holds the engine's residual stream after every layer for ONE forward over ids (IE_MIMO26_DUMP=<dir> with
ie-mimo26-score --ubatch >= len(ids): x_L{L}_s0.f32, [T, 4096] fp32). ids.txt: the prompt's ids followed by the
engine's greedy continuation (one line). For every continuation position p the drafter sees the context features of
positions < p (the last 1024: "at most 1,024 backbone context positions preceding the anchor", report 2.4), drafts the 7
tokens after the anchor token[p], and the accepted length is the matching prefix of token[p+1 .. p+7] -- the greedy
verify's acceptance. Prints the mean accepted drafts for k = 1..7 and the tokens per pass (k accepted + 1).

Variants: plain = dflash.py's forward as shipped (no sink, no value scale); sink = the checkpoint's
self_attn.attention_sink_bias as a virtual key in the softmax denominator and dflash_config.attention_value_scale on
the attention output (the main model's conventions). The better acceptance is the trained form. The mask positions use
dflash/mask_embedding.pt (the learned vector; the target's own embedding row for the mask id is all zeros) unless
--mask-emb target.
"""
import json
import math
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

torch.set_grad_enabled(False)
model_dir, dump_dir, ids_path, n_prompt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
variants = arg("--variants", "plain,sink").split(",")
max_pos = int(arg("--max-pos", "256"))
mask_emb = arg("--mask-emb", "file")
call = int(arg("--call", "0"))   # the forward() call whose dump to read (x_L{L}_s{call}.f32)
ddir = os.path.join(model_dir, "dflash")
cfg = json.load(open(os.path.join(ddir, "config.json")))
dc = cfg["dflash_config"]
H, NQ, NKV, HD, NL = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["num_hidden_layers"]
RD = int(HD * cfg["partial_rotary_factor"])
BLOCK, MASK, WIN, EPS, THETA = dc["block_size"], dc["mask_token_id"], cfg["sliding_window"], cfg["rms_norm_eps"], cfg["rope_theta"]
VSCALE, TL = dc["attention_value_scale"], dc["target_layer_ids"]

ids = [int(x) for x in open(ids_path).read().replace(",", " ").replace("[", " ").replace("]", " ").split()]
T = len(ids)
print(f"{T} ids ({n_prompt} prompt + {T - n_prompt} continuation); drafter {NL} layers, heads {NQ}/{NKV} x {HD}, rope {RD}, "
      f"window {WIN}, block {BLOCK}, target layers {TL}, value scale {VSCALE}", flush=True)

# ---- weights (bf16 -> fp32 on the CPU) ----
W = {}
with safe_open(os.path.join(ddir, "dflash_draft_model.safetensors"), "pt") as f:
    for k in f.keys():
        W[k] = f.get_tensor(k).float()
idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
for name in ("model.embed_tokens.weight", "lm_head.weight"):
    with safe_open(os.path.join(model_dir, idx[name]), "pt") as f:
        W[name] = f.get_tensor(name).float()
if mask_emb == "file":
    me = torch.load(os.path.join(ddir, "mask_embedding.pt"), map_location="cpu", weights_only=False)
    assert int(me["mask_token_id"]) == MASK, "mask_embedding.pt names another mask id"
    W["model.embed_tokens.weight"][MASK] = me["embedding"].float()
print(f"weights loaded (mask embedding: {mask_emb})", flush=True)


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(x, pos):   # x [n, heads, HD]; NEOX rotate-half over the first RD dims, theta THETA
    inv = 1.0 / (THETA ** (torch.arange(0, RD, 2, dtype=torch.float32) / RD))
    ang = pos[:, None].float() * inv[None, :]
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :], torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]
    xr, xp = x[..., :RD], x[..., RD:]
    x1, x2 = xr[..., : RD // 2], xr[..., RD // 2:]
    rot = torch.cat([-x2, x1], -1)
    return torch.cat([xr * cos + rot * sin, xp], -1)


# ---- context features: the residual after the target layers, [T, 5 * H] ----
feats = []
for L in TL:
    a = np.fromfile(os.path.join(dump_dir, f"x_L{L}_s{call}.f32"), dtype=np.float32)
    assert a.size == T * H, f"x_L{L}_s{call}.f32 has {a.size} floats, expected {T} x {H}"
    feats.append(torch.from_numpy(a.reshape(T, H)))
feats = torch.cat(feats, -1)
ctx = rms(feats @ W["fc.weight"].T, W["hidden_norm.weight"])   # [T, H]


def layer_kv(l, h, pos):   # context rows -> this layer's K (normed, roped) and V
    p = f"layers.{l}.self_attn."
    k = (h @ W[p + "k_proj.weight"].T).view(-1, NKV, HD)
    v = (h @ W[p + "v_proj.weight"].T).view(-1, NKV, HD)
    return rope(rms(k, W[p + "k_norm.weight"]), pos), v


# every layer's context K/V for every position, once (positions 0..T-1)
pos_all = torch.arange(T)
CK, CV = zip(*[layer_kv(l, ctx, pos_all) for l in range(NL)])


def draft(p, variant):
    """7 drafts after the anchor ids[p] (block at positions p .. p + 7), context = positions [p - WIN, p)."""
    c0 = max(0, p - WIN)
    h = W["model.embed_tokens.weight"][torch.tensor([ids[p]] + [MASK] * (BLOCK - 1))]
    bpos = torch.arange(p, p + BLOCK)
    for l in range(NL):
        pre = f"layers.{l}."
        x = rms(h, W[pre + "input_layernorm.weight"])
        q = rope(rms((x @ W[pre + "self_attn.q_proj.weight"].T).view(BLOCK, NQ, HD), W[pre + "self_attn.q_norm.weight"]), bpos)
        kb, vb = layer_kv(l, x, bpos)
        K = torch.cat([CK[l][c0:p], kb], 0)                     # [n_keys, NKV, HD]
        V = torch.cat([CV[l][c0:p], vb], 0)
        g = NQ // NKV
        Kx, Vx = K.repeat_interleave(g, 1), V.repeat_interleave(g, 1)          # [n_keys, NQ, HD]
        s = torch.einsum("bhd,khd->hbk", q, Kx) / math.sqrt(HD)                 # [NQ, BLOCK, n_keys]
        if variant == "sink":
            sink = W[pre + "self_attn.attention_sink_bias"].view(NQ, 1, 1).expand(NQ, BLOCK, 1)
            pr = torch.softmax(torch.cat([s, sink], -1), -1)[..., :-1]
        else:
            pr = torch.softmax(s, -1)
        o = torch.einsum("hbk,khd->bhd", pr, Vx).reshape(BLOCK, NQ * HD)
        o = o @ W[pre + "self_attn.o_proj.weight"].T
        if variant == "sink":
            o = o * VSCALE
        h = h + o
        x = rms(h, W[pre + "post_attention_layernorm.weight"])
        m = torch.nn.functional.silu(x @ W[pre + "mlp.gate_proj.weight"].T) * (x @ W[pre + "mlp.up_proj.weight"].T)
        h = h + m @ W[pre + "mlp.down_proj.weight"].T
    h = rms(h, W["norm.weight"])
    return (h[1:] @ W["lm_head.weight"].T).argmax(-1).tolist()                  # rows 1..7


for variant in variants:
    acc_k = np.zeros(BLOCK)   # acc_k[k] = sum over positions of the accepted drafts when k drafts are offered
    n = 0
    for p in range(n_prompt, min(T - BLOCK, n_prompt + max_pos)):
        d = draft(p, variant)
        truth = ids[p + 1: p + BLOCK]
        run = 0
        while run < len(d) and d[run] == truth[run]:
            run += 1
        for k in range(1, BLOCK):
            acc_k[k] += min(run, k)
        n += 1
        if n % 32 == 0:
            print(f"  [{variant}] {n} positions, mean accepted at k=7: {acc_k[7] / n:.2f}", flush=True)
    print(f"{variant} (mask {mask_emb}): {n} positions; mean accepted drafts by k: " +
          ", ".join(f"k{k} {acc_k[k] / n:.2f}" for k in range(1, BLOCK)) +
          "; tokens per pass (accepted + 1): " + ", ".join(f"k{k} {acc_k[k] / n + 1:.2f}" for k in range(1, BLOCK)), flush=True)
