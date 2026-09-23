#!/usr/bin/env python3
"""P6.0 fp32 CPU reference for the MiMo-V2.6 vision tower. See docs/mimo26/00_PORT_PLAN.md, P6.0.

Transcribed from the checkpoint modeling_mimo_v2.py, lines 585-911. Merger biases are absent and read as zero.
Usage: vision_ref.py <model_dir> <out_dir> [--threads N]
"""
import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from safetensors import safe_open
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil, smart_resize

BLOCKS_SAVED = (0, 5, 9, 27)


def load_weights(model_dir, dtype):
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    names = [n for n in idx if n.startswith("visual.")]
    shards = sorted({idx[n] for n in names})
    w = {}
    for shard in shards:
        with safe_open(str(model_dir / shard), framework="pt") as f:
            for n in names:
                if idx[n] == shard:
                    w[n[len("visual."):]] = f.get_tensor(n).to(dtype)
    return w


def patch_positions(gh, gw, merge=2):
    """(h, w) of every patch in merge-block order (reference rot_pos_emb)."""
    h = torch.arange(gh).unsqueeze(1).expand(-1, gw)
    h = h.reshape(gh // merge, merge, gw // merge, merge).permute(0, 2, 1, 3).flatten()
    w = torch.arange(gw).unsqueeze(0).expand(gh, -1)
    w = w.reshape(gh // merge, merge, gw // merge, merge).permute(0, 2, 1, 3).flatten()
    return torch.stack([h, w], dim=-1)


def rope_table(gh, gw, head_dim=64, theta=10000.0):
    """[N, head_dim] angle table: 16 freqs for h, 16 for w, the pair repeated (reference forward)."""
    dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    freqs = torch.outer(torch.arange(max(gh, gw), dtype=torch.float), inv_freq)
    r = freqs[patch_positions(gh, gw)].flatten(1)
    return torch.cat((r, r), dim=-1)


def column_index(gh, gw, merge=2):
    """Merge blocks in column-major order (reference get_window_index_1d, col=True, one image)."""
    lh, lw = gh // merge, gw // merge
    return torch.arange(lh * lw).reshape(lh, lw).t().reshape(-1)


def by_block(t, index, unit=4):
    """Reorder a sequence by whole merge blocks (reference apply_index)."""
    return t.unflatten(0, (-1, unit))[index].flatten(0, 1)


def rotate(x, cos, sin):
    """Reference _apply_rotary_pos_emb_vision for one tensor: fp32 math, cast back."""
    xf = x.float()
    half = xf.shape[-1] // 2
    rot = torch.cat((-xf[..., half:], xf[..., :half]), dim=-1)
    return (xf * cos.unsqueeze(-2) + rot * sin.unsqueeze(-2)).to(x.dtype)


def attention(x, W, i, cos, sin, windowed, band, heads=32, kv_heads=8, hd=64):
    """Reference MiMoVisionAttention.forward for one image: GQA, 2-D rope, optional band mask and key-0 sink bias."""
    n = x.shape[0]
    qkv = F.linear(x, W[f"blocks.{i}.attn.qkv.weight"], W[f"blocks.{i}.attn.qkv.bias"])
    q = qkv[:, :heads * hd].view(n, heads, hd)
    k = qkv[:, heads * hd:(heads + kv_heads) * hd].view(n, kv_heads, hd)
    v = qkv[:, (heads + kv_heads) * hd:].view(n, kv_heads, hd)
    q, k = rotate(q, cos, sin), rotate(k, cos, sin)
    q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    k = k.repeat_interleave(heads // kv_heads, dim=1)
    v = v.repeat_interleave(heads // kv_heads, dim=1)
    sinks = W.get(f"blocks.{i}.attn.sinks")
    out = torch.empty(1, heads, n, hd, dtype=x.dtype)
    for h in range(heads):
        mask = None
        if windowed:
            mask = torch.zeros(n, n, dtype=x.dtype)
            mask.masked_fill_(band, float("-inf"))
        if sinks is not None:
            mask = torch.zeros(n, n, dtype=x.dtype) if mask is None else mask
            mask[:, 0] += sinks[h]
        m = None if mask is None else mask.view(1, 1, n, n)
        out[:, h:h + 1] = F.scaled_dot_product_attention(q[:, h:h + 1], k[:, h:h + 1], v[:, h:h + 1],
                                                         attn_mask=m, scale=hd ** -0.5)
    out = out.squeeze(0).transpose(0, 1).reshape(n, heads * hd)
    return F.linear(out, W[f"blocks.{i}.attn.proj.weight"], W[f"blocks.{i}.attn.proj.bias"])


def block(x, W, i, cos, sin, windowed, band):
    """Reference MiMoVisionBlock: pre-RMSNorm attention and SwiGLU, residual each."""
    hdim = (x.shape[-1],)
    a = F.rms_norm(x, hdim, W[f"blocks.{i}.norm1.weight"], eps=1e-6)
    x = x + attention(a, W, i, cos, sin, windowed, band)
    m = F.rms_norm(x, hdim, W[f"blocks.{i}.norm2.weight"], eps=1e-6)
    g = F.linear(m, W[f"blocks.{i}.mlp.gate_proj.weight"], W[f"blocks.{i}.mlp.gate_proj.bias"])
    u = F.linear(m, W[f"blocks.{i}.mlp.up_proj.weight"], W[f"blocks.{i}.mlp.up_proj.bias"])
    return x + F.linear(F.silu(g) * u, W[f"blocks.{i}.mlp.down_proj.weight"], W[f"blocks.{i}.mlp.down_proj.bias"])


def merger(x, W):
    """Reference MiMoVisionPatchMerger with the checkpoint's missing biases as zero."""
    y = F.layer_norm(x, (x.shape[-1],), W["merger.ln_q.weight"], None, eps=1e-6).reshape(-1, 4 * x.shape[-1])
    y = F.gelu(F.linear(y, W["merger.mlp.0.weight"]))
    return F.linear(y, W["merger.mlp.2.weight"])


def tower(pv, grid, W, cfg, dtype, save=None):
    """Reference MiMoVisionTransformer.forward for one image (grid_t 1)."""
    t, gh, gw = grid
    assert t == 1, "one still image"
    x = F.conv3d(pv.to(dtype).view(-1, 3, 2, 16, 16), W["patch_embed.proj.weight"], stride=(2, 16, 16))
    x = x.view(-1, cfg["hidden_size"])
    emb = rope_table(gh, gw)
    col = column_index(gh, gw)
    back = torch.argsort(col)
    row_cs = (emb.cos(), emb.sin())
    ce = by_block(emb, col)
    col_cs = (ce.cos(), ce.sin())
    r = torch.arange(x.shape[0])
    band = (r.unsqueeze(1) - r.unsqueeze(0)).abs() > cfg["visual_token_window_size"]
    kinds, full = cfg["vit_window_attn_types"], set(cfg["fullatt_block_indexes"])
    in_col = False
    for i in range(cfg["depth"]):
        k = kinds[i]
        if k == 1 and (i == 0 or kinds[i - 1] != 1):
            x, in_col = by_block(x, col), True
        if i > 0 and k != 1 and kinds[i - 1] == 1:
            x, in_col = by_block(x, back), False
        cos, sin = col_cs if k == 1 else row_cs
        x = block(x, W, i, cos, sin, i not in full, band)
        if save is not None and i in BLOCKS_SAVED:
            save[i] = (by_block(x, back) if in_col else x).float().clone()
    return merger(x, W)


def compare(test, ref):
    """rel-L2 and the row-cosine distribution of test against ref (the P6.1 bar's terms)."""
    a, b = test.double(), ref.double()
    cos = F.cosine_similarity(a, b, dim=1)
    return {"rows": int(a.shape[0]), "rel_l2": ((a - b).norm() / b.norm()).item(),
            "min_row_cos": cos.min().item(), "p05_row_cos": torch.quantile(cos, 0.05).item(),
            "rows_below_0.99": int((cos < 0.99).sum()), "rows_below_0.95": int((cos < 0.95).sum())}


def fixtures():
    """Synthetic only: shapes and text 640x480; a UI-like 1920x1080 frame; a tiny 28x100 image under min_pixels."""
    a = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(a)
    d.ellipse((60, 60, 220, 220), fill=(220, 30, 30))
    d.rectangle((300, 80, 460, 240), fill=(30, 60, 220))
    d.polygon([(520, 400), (620, 400), (570, 300)], fill=(30, 170, 60))
    d.text((80, 330), "MiMo 42", fill="black", font_size=64)
    b = Image.new("RGB", (1920, 1080), (18, 22, 36))
    d = ImageDraw.Draw(b)
    d.rectangle((0, 0, 260, 1080), fill=(10, 14, 26))
    d.rectangle((260, 0, 1920, 70), fill=(24, 30, 48))
    for k, label in enumerate(["Home", "Chat", "Studio", "Files", "Settings"]):
        d.text((40, 120 + 60 * k), label, fill=(200, 210, 230), font_size=28)
    d.text((300, 20), "Rocket build - frame 24 of 96", fill=(230, 230, 240), font_size=30)
    for k in range(8):
        d.text((320, 140 + 70 * k), f"step {k + 1}: render check frame, mean luminance {90 + 7 * k}",
               fill=(180, 190, 210), font_size=26)
    d.rounded_rectangle((1500, 900, 1860, 980), radius=16, fill=(120, 90, 230))
    d.text((1560, 920), "Send", fill="white", font_size=34)
    c = Image.new("RGB", (28, 100), (250, 240, 200))
    ImageDraw.Draw(c).line((0, 0, 27, 99), fill=(0, 0, 0), width=3)
    return [a, b, c]


def processor(pc):
    return Qwen2VLImageProcessorPil(min_pixels=pc["min_pixels"], max_pixels=pc["max_pixels"],
                                 size={"shortest_edge": pc["min_pixels"], "longest_edge": pc["max_pixels"]},
                                 patch_size=pc["patch_size"], temporal_patch_size=pc["temporal_patch_size"],
                                 merge_size=pc["merge_size"], image_mean=pc["image_mean"], image_std=pc["image_std"])


def sweep(pc):
    rng = np.random.default_rng(20260923)
    sizes = [(100, 28), (28, 100), (480, 640), (1080, 1920), (1920, 1080), (3000, 4000), (4000, 3000),
             (32, 32), (33, 33), (4096, 4096)] + [tuple(int(v) for v in p) for p in rng.integers(16, 4200, size=(110, 2))]
    rows = []
    for h, w in sizes:
        try:
            hb, wb = smart_resize(h, w, factor=pc["patch_size"] * pc["merge_size"],
                                  min_pixels=pc["min_pixels"], max_pixels=pc["max_pixels"])
            rows.append(f"{h}\t{w}\t{hb}\t{wb}")
        except ValueError as e:
            rows.append(f"{h}\t{w}\terror\t{str(e).splitlines()[0][:60]}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=pathlib.Path)
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.model_dir / "config.json").read_text())["vision_config"]
    pc = json.loads((args.model_dir / "preprocessor_config.json").read_text())
    W32 = load_weights(args.model_dir, torch.float32)
    W16 = {k: v.to(torch.bfloat16) for k, v in W32.items()}
    (out / "sweep.tsv").write_text("h\tw\th_bar\tw_bar\n" + "\n".join(sweep(pc)) + "\n")
    proc = processor(pc)
    grids, yard = {}, {}
    for k, img in enumerate(fixtures()):
        img.save(out / f"fixture_{k}.png")
        enc = proc(images=[img], return_tensors="pt")
        pv, grid = enc["pixel_values"].float(), [int(v) for v in enc["image_grid_thw"][0]]
        grids[k] = {"size_wh": list(img.size), "grid_thw": grid, "patches": int(pv.shape[0]), "merged": int(pv.shape[0]) // 4}
        pv.numpy().astype("<f4").tofile(out / f"pixel_values_{k}.f32")
        saved = {}
        with torch.no_grad():
            ref = tower(pv, grid, W32, cfg, torch.float32, saved)
            half = tower(pv, grid, W16, cfg, torch.bfloat16)
        ref.numpy().astype("<f4").tofile(out / f"merged_{k}.f32")
        for i, h in saved.items():
            h.numpy().astype("<f4").tofile(out / f"block{i}_{k}.f32")
        yard[k] = compare(half.float(), ref)
        print(f"fixture {k}: grid {grid} patches {pv.shape[0]} -> {ref.shape[0]} tokens; bf16 vs fp32 {yard[k]}", flush=True)
    (out / "grids.json").write_text(json.dumps(grids, indent=1))
    (out / "yardstick.json").write_text(json.dumps(yard, indent=1))


if __name__ == "__main__":
    main()
