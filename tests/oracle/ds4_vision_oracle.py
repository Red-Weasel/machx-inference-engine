#!/usr/bin/env python3
"""Pure-numpy oracle for the DeepSeek-V4-Flash-Vision-Exp vision path.

Reads the native vision sidecar (safetensors, BF16) directly — the same bytes the
C++ tower will read — and reproduces, in fp32, the official reference
(`inference/vision.py`, `inference/image_processor.py` of
deepseek-ai/DeepSeek-V4-Flash-Vision-Exp):

    preprocess -> patch embed -> 32 x {RMSNorm, 2-D RoPE attention, SwiGLU MLP}
    -> RMSNorm -> 3x3 unfold aligner (Linear, GELU-erf, Linear)
    -> N-layout block (compress pad, IMAGE_START, column-interleaved row pairs,
       pad_last, IMAGE_END) with the four control rows

and writes one golden file:

    magic  "DS4VORC1"
    u32    n_vit_h, n_vit_w, n_llm_h, n_llm_w, start_pos, n_rows, hidden, n_patches
    f32    pixels[3 * n_vit_h*14 * n_vit_w*14]      CHW, normalised, BF16-rounded
    i32    types[n_rows]                             0=START 1=PAD 2=IMAGE 3=NEWLINE 4=END
    i32    perm[n_llm_h * n_llm_w]                   aligner row for each IMAGE slot, in order
    f32    aligned[n_llm_h*n_llm_w * hidden]         aligner output, grid order (before perm)
    f32    rows[n_rows * hidden]                     final block rows the LM receives

Usage:
    python3 tests/oracle/ds4_vision_oracle.py --tower T.safetensors --config config.json \
        --image carrots.jpeg --out golden.bin [--start-pos 0]
    python3 tests/oracle/ds4_vision_oracle.py ... --synthetic 224x336   (deterministic gradient)
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4
MAGIC = b"DS4VORC1"


# ---------------------------------------------------------------- sidecar ---
def load_sidecar(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        f.seek(0)
        blob = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        assert info["dtype"] == "BF16", (name, info["dtype"])
        a, b = info["data_offsets"]
        u16 = np.frombuffer(blob[base + a: base + b], dtype="<u2")
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        out[name] = f32.reshape(info["shape"])
    return out


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round fp32 to the nearest BF16 (ties to even), return fp32."""
    u = x.astype(np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return u.view(np.float32)


# ------------------------------------------------------------ preprocess ---
def grid_tokens(best_h, best_w, patch, ratio):
    n_llm_h = math.ceil((best_h // patch) / ratio)
    n_llm_w = math.ceil((best_w // patch) / ratio)
    num = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num += n_llm_w + 1
    num += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num


def solve_resize_ratio(height, width, patch, ratio, max_n_token):
    r = height / width
    max_w_f = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_f = max_w_f * r
    if max_w_f < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_w = max_w * patch * ratio
        best_h = max_h * patch * ratio
    elif max_h_f < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_w = max_w * patch * ratio
        best_h = max_h * patch * ratio
    else:
        max_w = math.floor(max_w_f)
        max_h = math.floor(max_h_f)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(max_w * patch * ratio / width, max_h * patch * ratio / height)
        best_w = math.floor(width * beta / patch) * patch
        best_h = math.floor(height * beta / patch) * patch
    n_llm_h, n_llm_w, num = grid_tokens(best_h, best_w, patch, ratio)
    return n_llm_h, n_llm_w, best_h, best_w, num


def safe_resize(height, width, best_h, best_w, patch, ratio, max_n_token):
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num = grid_tokens(best_h, best_w, patch, ratio)
    budget = max_n_token
    while num > max_n_token:
        n_llm_h, n_llm_w, best_h, best_w, num = solve_resize_ratio(height, width, patch, ratio, budget)
        budget -= 1
    return n_llm_h, n_llm_w, best_h, best_w


def plan_resize(width, height, cfg):
    """Official load_image() geometry; returns (best_w, best_h, n_vit_h, n_vit_w, n_llm_h, n_llm_w, plain)."""
    p = cfg["vision_patch_size"]
    w0, h0 = width, height
    if cfg["vision_max_wh_ratio"] is not None and width > height * cfg["vision_max_wh_ratio"]:
        width = height * cfg["vision_max_wh_ratio"]
    if 0 < width * height < cfg["vision_min_pixels"]:
        r = (cfg["vision_min_pixels"] / (width * height)) ** 0.5
        width = int(width * r)
        height = int(height * r)
    best_w = math.ceil(width / p) * p
    best_h = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_h, best_w = safe_resize(
        height, width, best_h, best_w, p, cfg["vision_downsample_ratio"], cfg["vision_max_n_token"])
    n_vit_h, n_vit_w = best_h // p, best_w // p
    plain = cfg["vision_max_wh_ratio"] is not None and w0 >= cfg["vision_max_wh_ratio"] * h0
    return best_w, best_h, n_vit_h, n_vit_w, n_llm_h, n_llm_w, plain


def preprocess_pil(image, cfg):
    from PIL import Image, ImageOps
    p = cfg["vision_patch_size"]
    best_w, best_h, n_vit_h, n_vit_w, n_llm_h, n_llm_w, plain = plan_resize(*image.size, cfg)
    if plain:
        image = image.resize((best_w, best_h))
    else:
        image = ImageOps.pad(image, (best_w, best_h), color=(127, 127, 127))
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    x = bf16_round((x - 0.5) / 0.5)                       # official casts to bf16 here
    return x, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def synthetic_pixels(h, w):
    """Deterministic gradient, already sized to multiples of 14 — no PIL."""
    yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    r = xx * np.ones_like(yy)
    g = yy * np.ones_like(xx)
    b = np.sin(3.0 * xx) * np.cos(2.0 * yy)
    return bf16_round(np.stack([r, g, b]).astype(np.float32))


def patchify(x, p):
    c, H, W = x.shape
    nh, nw = H // p, W // p
    return x.reshape(c, nh, p, nw, p).transpose(1, 3, 0, 2, 4).reshape(nh * nw, c * p * p)


# ---------------------------------------------------------------- tower ---
def rms_norm(x, w, eps=1e-6):
    y = x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))
    return w * y


def vision_cos_sin(n_h, n_w, dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))     # [dim/2]
    hpos = np.repeat(np.arange(n_h, dtype=np.float32)[:, None], n_w, axis=1)
    wpos = np.repeat(np.arange(n_w, dtype=np.float32)[None, :], n_h, axis=0)
    freqs = np.stack([hpos, wpos], axis=-1).reshape(-1, 2, 1) * inv          # [n, 2, dim/2]
    freqs = freqs.reshape(freqs.shape[0], -1)                                # [n, dim]  (h-freqs ‖ w-freqs)
    return np.cos(freqs)[:, None, :], np.sin(freqs)[:, None, :]              # [n, 1, dim]


def apply_rotary(x, cos, sin):
    x1, x2 = np.split(x, 2, axis=-1)
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_erf(x):
    from math import erf
    v = np.vectorize(erf, otypes=[np.float64])
    return (0.5 * x.astype(np.float64) * (1.0 + v(x.astype(np.float64) / math.sqrt(2.0)))).astype(np.float32)


def vit_forward(W, patches, n_h, n_w, cfg):
    dim = cfg["vision_dim"]
    nh = cfg["vision_n_heads"]
    hd = dim // nh
    rope_dim = hd // 2
    x = patches @ W["vision.patch_embed.proj.weight"].T + W["vision.patch_embed.proj.bias"]
    cos, sin = vision_cos_sin(n_h, n_w, rope_dim, cfg["vision_rope_theta"])
    n = x.shape[0]
    scale = 1.0 / math.sqrt(hd)
    for L in range(cfg["vision_n_layers"]):
        pre = f"vision.blocks.{L}."
        h = rms_norm(x, W[pre + "norm1.weight"])
        qkv = h @ W[pre + "attn.wqkv.weight"].T + W[pre + "attn.wqkv.bias"]
        q, k, v = (t.reshape(n, nh, hd) for t in np.split(qkv, 3, axis=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        qh, kh, vh = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)   # [nh, n, hd]
        s = (qh @ kh.transpose(0, 2, 1)) * scale
        s = s - s.max(axis=-1, keepdims=True)
        pr = np.exp(s)
        pr /= pr.sum(axis=-1, keepdims=True)
        o = (pr @ vh).transpose(1, 0, 2).reshape(n, dim)
        x = x + (o @ W[pre + "attn.wo.weight"].T + W[pre + "attn.wo.bias"])
        h = rms_norm(x, W[pre + "norm2.weight"])
        gu = h @ W[pre + "mlp.w1.weight"].T
        gate, up = np.split(gu, 2, axis=-1)
        x = x + (silu(gate) * up) @ W[pre + "mlp.w2.weight"].T
    return rms_norm(x, W["vision.norm.weight"])


def aligner_forward(W, x, n_h, n_w, cfg):
    r = cfg["vision_downsample_ratio"]
    dim = cfg["vision_dim"]
    g = x.reshape(n_h, n_w, dim).transpose(2, 0, 1)                    # [C, n_h, n_w]
    ph, pw = (-n_h) % r, (-n_w) % r
    g = np.pad(g, ((0, 0), (0, ph), (0, pw)))
    H, Wd = g.shape[1], g.shape[2]
    bh, bw = H // r, Wd // r
    # F.unfold(kernel r, stride r): feature index = c*r*r + kh*r + kw, block index row-major
    blocks = g.reshape(dim, bh, r, bw, r).transpose(1, 3, 0, 2, 4).reshape(bh * bw, dim * r * r)
    h = blocks @ W["aligner.w1.weight"].T + W["aligner.w1.bias"]
    h = gelu_erf(h)
    return h @ W["aligner.w2.weight"].T + W["aligner.w2.bias"]


# --------------------------------------------------------------- layout ---
def build_image_block(n_llm_h, n_llm_w, start_pos):
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = np.array(([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_PAD] * (row_len * pad_h),
                     dtype=np.int64)
    order = np.arange(rows * row_len).reshape(rows // 2, 2, row_len).transpose(0, 2, 1).reshape(-1)
    image_idx = np.full(rows * row_len, -1, dtype=np.int64)
    image_idx.reshape(rows, row_len)[:n_llm_h, :n_llm_w] = np.arange(n_llm_h * n_llm_w).reshape(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = np.concatenate([
        np.full(compress_pad, IMAGE_PAD, dtype=np.int64),
        np.array([IMAGE_START]),
        types[order],
        np.full(pad_last, IMAGE_PAD, dtype=np.int64),
        np.array([IMAGE_END]),
    ])
    return types.astype(np.int32), perm.astype(np.int32)


def assemble_rows(W, aligned, types, perm):
    controls = np.stack([W["image_start"], W["image_pad"], W["image_pad"], W["image_newline"], W["image_end"]])
    rows = controls[types].copy()
    rows[types == IMAGE] = aligned[perm]
    return rows.astype(np.float32)


# ------------------------------------------------------------------ main ---
def run(W, cfg, pixels, n_vit_h, n_vit_w, n_llm_h, n_llm_w, start_pos):
    p = cfg["vision_patch_size"]
    patches = patchify(pixels, p)
    feats = vit_forward(W, patches, n_vit_h, n_vit_w, cfg)
    aligned = aligner_forward(W, feats, n_vit_h, n_vit_w, cfg)
    assert aligned.shape[0] == n_llm_h * n_llm_w, (aligned.shape, n_llm_h, n_llm_w)
    types, perm = build_image_block(n_llm_h, n_llm_w, start_pos)
    rows = assemble_rows(W, aligned, types, perm)
    return patches, aligned, types, perm, rows


def write_golden(path, pixels, n_vit_h, n_vit_w, n_llm_h, n_llm_w, start_pos, aligned, types, perm, rows):
    hidden = rows.shape[1]
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<8I", n_vit_h, n_vit_w, n_llm_h, n_llm_w, start_pos, rows.shape[0], hidden,
                            n_vit_h * n_vit_w))
        f.write(np.ascontiguousarray(pixels, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(types, dtype="<i4").tobytes())
        f.write(np.ascontiguousarray(perm, dtype="<i4").tobytes())
        f.write(np.ascontiguousarray(aligned, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(rows, dtype="<f4").tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--synthetic", help="HxW pixels (multiples of 14), e.g. 224x336")
    ap.add_argument("--start-pos", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    W = load_sidecar(a.tower)
    assert len(W) == 267, len(W)
    p = cfg["vision_patch_size"]
    if a.synthetic:
        h, w = (int(v) for v in a.synthetic.lower().split("x"))
        assert h % p == 0 and w % p == 0
        _, _, n_vit_h, n_vit_w, n_llm_h, n_llm_w, _ = plan_resize(w, h, cfg)
        assert (n_vit_h * p, n_vit_w * p) == (h, w), "synthetic size must survive plan_resize unchanged"
        pixels = synthetic_pixels(h, w)
    else:
        from PIL import Image
        with Image.open(a.image) as im:
            pixels, n_vit_h, n_vit_w, n_llm_h, n_llm_w = preprocess_pil(im.convert("RGB"), cfg)
    patches, aligned, types, perm, rows = run(W, cfg, pixels, n_vit_h, n_vit_w, n_llm_h, n_llm_w, a.start_pos)
    assert rows.shape[0] <= cfg["vision_max_n_token"] and np.isfinite(rows).all()
    write_golden(a.out, pixels, n_vit_h, n_vit_w, n_llm_h, n_llm_w, a.start_pos, aligned, types, perm, rows)
    print(json.dumps({
        "vit_grid": [n_vit_h, n_vit_w], "llm_grid": [n_llm_h, n_llm_w], "patches": int(patches.shape[0]),
        "rows": int(rows.shape[0]), "grid_tokens": grid_tokens(n_vit_h * p, n_vit_w * p, p,
                                                              cfg["vision_downsample_ratio"])[2],
        "start_pos": a.start_pos, "aligned_mean": float(aligned.mean()), "aligned_std": float(aligned.std()),
        "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
