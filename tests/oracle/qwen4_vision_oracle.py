#!/usr/bin/env python3
"""Pure-numpy golden oracle for the qwen4exp vision tower (docs/qwen4/16_vision_port.md).

Loads mmproj-F16.gguf directly and reproduces Qwen4ExpVisionModel.forward exactly
(reference: modeling_qwen4_exp.py:1660-1960 + vision_utils.py). Emits a golden
dump the C++ unit test (qwen4_vision_cpu_test) compares against.

Usage:
  python3 qwen4_vision_oracle.py [--mmproj PATH] [--out PATH] [--h 64] [--w 96]

The input image is a deterministic synthetic gradient (no file decode), height/
width in PIXELS (multiples of 32).
"""
import argparse
import os
import numpy as np
from gguf import GGUFReader

P = 16          # patch size
MERGE = 2       # spatial merge
HID = 1152
HEADS = 16
HD = HID // HEADS          # 72
ROT = HD // 2              # 36 (18 h-freqs + 18 w-freqs)
FFN = 4304
DEPTH = 27
GRID_SIDE = 48             # sqrt(2304) learned pos table
OUT_D = 2560
LN_EPS = 1e-6
VIS_THETA = 1e4


def t(reader, name):
    for te in reader.tensors:
        if te.name == name:
            a = np.asarray(te.data)
            return a.astype(np.float32)
    raise KeyError(name)


def layer_norm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + LN_EPS) * w + b


def gelu_tanh(x):  # gelu_pytorch_tanh (ViT MLP)
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def gelu_erf_np(x):  # nn.GELU default (merger)
    # erf via numpy: 0.5*x*(1+erf(x/sqrt(2))). math.erf is scalar; use vectorized
    # rational erf-free identity: erf(z) = 2*Phi(z*sqrt(2))-1 — implement with
    # np.vectorize(math.erf) for exactness (oracle speed is irrelevant).
    import math
    verf = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + verf(x / math.sqrt(2.0)))


def rotate_half(x):
    a, b = x[..., :ROT], x[..., ROT:]
    return np.concatenate([-b, a], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmproj", default=os.path.expanduser("~/models/Qwen3.8-Flash-Next-GGUF/mmproj-F16.gguf"))
    ap.add_argument("--out", default="qwen4_vision_golden.npz")
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=96)
    args = ap.parse_args()
    assert args.h % (P * MERGE) == 0 and args.w % (P * MERGE) == 0

    r = GGUFReader(args.mmproj)
    gh, gw = args.h // P, args.w // P            # patch grid
    n_patch = gh * gw

    # ---- synthetic image, [3, H, W] f32 in [0,1], deterministic -------------
    yy, xx = np.mgrid[0:args.h, 0:args.w].astype(np.float32)
    img = np.stack([
        xx / args.w,
        yy / args.h,
        0.5 + 0.5 * np.sin(xx * 0.37 + yy * 0.11),
    ])
    img = (img - 0.5) / 0.5                       # normalize mean=std=0.5

    # ---- block-major patch extraction, per-patch layout [C, T, ph, pw] ------
    # (h//m, m, w//m, m) -> (h//m, w//m, m, m): 2x2 merge blocks contiguous.
    patches = np.zeros((n_patch, 3, 2, P, P), np.float32)
    idx = 0
    for br in range(gh // MERGE):
        for bc in range(gw // MERGE):
            for ir in range(MERGE):
                for ic in range(MERGE):
                    pr, pc = br * MERGE + ir, bc * MERGE + ic
                    px = img[:, pr * P:(pr + 1) * P, pc * P:(pc + 1) * P]
                    patches[idx, :, 0] = px
                    patches[idx, :, 1] = px       # temporal duplicate (images)
                    idx += 1
    # position ids (h, w) per patch, same block-major order
    pos = np.zeros((n_patch, 2), np.int64)
    idx = 0
    for br in range(gh // MERGE):
        for bc in range(gw // MERGE):
            for ir in range(MERGE):
                for ic in range(MERGE):
                    pos[idx] = (br * MERGE + ir, bc * MERGE + ic)
                    idx += 1

    # ---- patch embed --------------------------------------------------------
    w0 = t(r, "v.patch_embd.weight").reshape(HID, 3, P, P)
    w1 = t(r, "v.patch_embd.weight.1").reshape(HID, 3, P, P)
    pb = t(r, "v.patch_embd.bias")
    W = np.concatenate([w0[:, :, None], w1[:, :, None]], axis=2).reshape(HID, -1)
    x = patches.reshape(n_patch, -1) @ W.T + pb   # [N, 1152]

    # ---- interpolated learned pos embed (bilinear, align_corners=True) ------
    table = t(r, "v.position_embd.weight").reshape(GRID_SIDE * GRID_SIDE, HID)

    def taps(index, size):
        src = index * (GRID_SIDE - 1) / max(size - 1, 1)
        f = np.floor(src)
        tp = np.stack([f, f + 1], -1).clip(0, GRID_SIDE - 1).astype(np.int64)
        d = np.abs(src[:, None] - f[:, None] - np.array([0.0, 1.0]))
        return tp, np.clip(1.0 - d, 0.0, None)

    ht, hw_ = taps(pos[:, 0].astype(np.float64), gh)
    wt, ww_ = taps(pos[:, 1].astype(np.float64), gw)
    ind = (ht[:, :, None] * GRID_SIDE + wt[:, None, :]).reshape(n_patch, 4)
    wgt = (hw_[:, :, None] * ww_[:, None, :]).reshape(n_patch, 4)
    x = x + (table[ind] * wgt[:, :, None].astype(np.float32)).sum(1)

    # ---- rope tables --------------------------------------------------------
    inv = VIS_THETA ** (-np.arange(0, ROT, 2, dtype=np.float32) / ROT)  # [18]
    ang = np.concatenate([pos[:, 0:1] * inv[None], pos[:, 1:2] * inv[None]], 1)  # [N,36]
    emb = np.concatenate([ang, ang], 1)           # [N,72]
    cos, sin = np.cos(emb), np.sin(emb)

    # ---- 27 blocks ----------------------------------------------------------
    for L in range(DEPTH):
        pfx = f"v.blk.{L}."
        h = layer_norm(x, t(r, pfx + "ln1.weight"), t(r, pfx + "ln1.bias"))
        qkv = h @ t(r, pfx + "attn_qkv.weight").reshape(3 * HID, HID).T + t(r, pfx + "attn_qkv.bias")
        q, k, v = [qkv[:, i * HID:(i + 1) * HID].reshape(n_patch, HEADS, HD) for i in range(3)]
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        o = np.zeros_like(q)
        for hd in range(HEADS):
            s = (q[:, hd] @ k[:, hd].T) * (HD ** -0.5)
            s = s - s.max(-1, keepdims=True)
            p = np.exp(s); p /= p.sum(-1, keepdims=True)
            o[:, hd] = p @ v[:, hd]
        x = x + o.reshape(n_patch, HID) @ t(r, pfx + "attn_out.weight").reshape(HID, HID).T \
              + t(r, pfx + "attn_out.bias")
        h = layer_norm(x, t(r, pfx + "ln2.weight"), t(r, pfx + "ln2.bias"))
        h = gelu_tanh(h @ t(r, pfx + "ffn_up.weight").reshape(FFN, HID).T + t(r, pfx + "ffn_up.bias"))
        x = x + h @ t(r, pfx + "ffn_down.weight").reshape(HID, FFN).T + t(r, pfx + "ffn_down.bias")

    # ---- merger -------------------------------------------------------------
    x = layer_norm(x, t(r, "v.post_ln.weight"), t(r, "v.post_ln.bias"))
    x = x.reshape(n_patch // 4, 4 * HID)
    x = gelu_erf_np(x @ t(r, "mm.0.weight").reshape(4 * HID, 4 * HID).T + t(r, "mm.0.bias"))
    x = x @ t(r, "mm.2.weight").reshape(OUT_D, 4 * HID).T + t(r, "mm.2.bias")

    print(f"golden: {x.shape} mean={x.mean():.6f} std={x.std():.6f} "
          f"[0,:4]={np.round(x[0, :4], 4)}")
    # Flat binary for the C++ test: u32 H, u32 W, u32 Nm, f32 img[3*H*W], f32 emb[Nm*2560]
    with open(args.out, "wb") as f:
        np.array([args.h, args.w, x.shape[0]], np.uint32).tofile(f)
        img.astype(np.float32).tofile(f)
        x.astype(np.float32).tofile(f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
