#!/usr/bin/env python3
"""Hy4-flash: the reference image encoder on the REAL weights, as an oracle for the engine's tower.

  <reference venv>/bin/python vision_oracle.py <weights_dir> <out_dir> [--size 640x512] [--image file.png]

Loads the checkpoint's `vit.*` tensors into transformers' HYV4FlashVisionTransformer (float32, CPU), runs one image
and writes to <out_dir>:
  image.png            the test image (a synthetic one unless --image is given)
  pixel_values.f32     [patches, 768] what the reference image processor produced (row-major patches, (c, y, x) each)
  block_NN.f32         [patches, 1152] the hidden state after blocks 0, 13 and 26
  merged.f32           [tokens, 3072] the projector's output, a newline vector closing every merged row
  meta.json            grid_h, grid_w, patches, tokens, the resized size
The tower runs in fp32 here (the release runs it in bf16): the engine's fp16 tower is compared with this.
"""
import argparse, json, os, re, struct, sys, zlib
import numpy as np
import torch

def load_vit(weights_dir):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(weights_dir, "model.safetensors.index.json")))["weight_map"]
    sd = {}
    for name, shard in idx.items():
        if not name.startswith("vit."): continue
        with safe_open(os.path.join(weights_dir, shard), "pt") as f: t = f.get_tensor(name).float()
        k = name[4:]
        k = re.sub(r"^layers\.(\d+)\.input_layernorm\.", r"layers.\1.layer_norm1.", k)
        k = re.sub(r"^layers\.(\d+)\.post_attention_layernorm\.", r"layers.\1.layer_norm2.", k)
        k = k.replace("mlp.dense_h_to_4h", "mlp.fc1").replace("mlp.dense_4h_to_h", "mlp.fc2")
        k = k.replace("perceive.norm.", "patch_merger.before_rms.").replace("perceive.proj.0.", "patch_merger.proj_conv.").replace("perceive.proj.2.", "patch_merger.proj_out.")
        k = k.replace("perceive.image_newline", "patch_merger.image_newline")
        sd[k] = t
    return sd

def synthetic(w, h):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([255 * x / w, 255 * y / h, 127 + 127 * np.sin(x / 23.0) * np.cos(y / 17.0)], -1)
    img[h // 4: h // 2, w // 5: w // 2] = [220, 30, 30]                       # a red block
    cy, cx = int(h * 0.68), int(w * 0.7); rr = min(w, h) // 6
    img[(y - cy) ** 2 + (x - cx) ** 2 < rr * rr] = [30, 60, 220]              # a blue disc
    return np.clip(img, 0, 255).astype(np.uint8)

def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))
    def ch(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b""))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("weights"); ap.add_argument("out"); ap.add_argument("--size", default="640x512"); ap.add_argument("--image")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashVisionConfig
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import HYV4FlashVisionTransformer
    from transformers.models.hy_v4_flash.image_processing_pil_hy_v4_flash import HYV4FlashImageProcessorPil   # (PIL's bicubic: the engine's resize)
    from PIL import Image
    vc = json.load(open(os.path.join(a.weights, "config.json")))["vision_config"]
    fields = HYV4FlashVisionConfig.__dataclass_fields__ if hasattr(HYV4FlashVisionConfig, "__dataclass_fields__") else {}
    cfg = HYV4FlashVisionConfig(**{k: v for k, v in vc.items() if k in fields or k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "patch_size", "spatial_merge_size", "max_image_size", "out_hidden_size", "rms_norm_eps", "hidden_act", "num_channels", "interpolate_mode")})
    model = HYV4FlashVisionTransformer(cfg).eval().float()
    missing, unexpected = model.load_state_dict(load_vit(a.weights), strict=False)
    print("missing:", list(missing)[:6], "unexpected:", list(unexpected)[:6])
    if missing or unexpected: sys.exit("the checkpoint's vision tensors do not fill the reference tower")
    if a.image: img = Image.open(a.image).convert("RGB")
    else:
        w, h = (int(v) for v in a.size.split("x")); rgb = synthetic(w, h); write_png(os.path.join(a.out, "image.png"), rgb); img = Image.fromarray(rgb)
    proc = HYV4FlashImageProcessorPil()
    feats = proc(images=[img], return_tensors="pt")
    pv, grid = feats["pixel_values"].float(), feats["image_grid_thw"]
    _, gh, gw = (int(v) for v in grid[0])
    keep = {}
    for i in (0, 13, len(model.layers) - 1):
        model.layers[i].register_forward_hook(lambda m, args, out, i=i: keep.__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach()))
    with torch.no_grad(): out = model(pixel_values=pv, grid_thw=grid)
    merged = out.pooler_output.reshape(-1, cfg.out_hidden_size)
    pv.numpy().astype("<f4").tofile(os.path.join(a.out, "pixel_values.f32"))
    for i, t in keep.items(): t.reshape(-1, cfg.hidden_size).numpy().astype("<f4").tofile(os.path.join(a.out, f"block_{i:02d}.f32"))
    merged.numpy().astype("<f4").tofile(os.path.join(a.out, "merged.f32"))
    json.dump({"grid_h": gh, "grid_w": gw, "patches": gh * gw, "tokens": int(merged.shape[0]), "height": gh * cfg.patch_size, "width": gw * cfg.patch_size,
               "blocks": sorted(keep), "eps": cfg.rms_norm_eps, "source_size": [img.size[1], img.size[0]]}, open(os.path.join(a.out, "meta.json"), "w"))
    print(f"grid {gh} x {gw}: {gh * gw} patches -> {merged.shape[0]} tokens; merged rms {merged.pow(2).mean().sqrt():.4f}, pixel rms {pv.pow(2).mean().sqrt():.4f}")

if __name__ == "__main__": main()
