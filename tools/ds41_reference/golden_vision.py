#!/usr/bin/env python3
"""Phase 55 golden: the checkpoint's own vision.py + image_processor.py on real weights (CPU, fp32).

Writes, under <out_dir>:
  geom.txt           "w h n_llm_h n_llm_w best_h best_w" per line -- official plan_image_grid over a size sweep
  img_<k>.png        the fixture images (deterministic, drawn here)
  golden_<k>.bin     magic "DS41VOR1", u32[8] = n_vit_h, n_vit_w, n_llm_h, n_llm_w, n_types, hidden, n_patches, 0,
                     then f32 pixels [3*H*W] (the normalised bf16 tensor load_image patchifies),
                     f32 aligner rows [n_llm_h*n_llm_w * hidden] in grid order, i32 types [n_types]
The ViT and the aligner are the shipped modules with the shipped weights upcast to fp32: the engine's fp16 tower is
measured against the exact function, not against another reduced-precision run.

  usage: golden_vision.py <out_dir> [model_dir]
"""
import json, os, struct, sys, types as pytypes
import numpy as np

D = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
sys.path.append(os.path.join(D, "inference"))

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from safetensors import safe_open
import vision as V
import image_processor as IP

torch.set_grad_enabled(False)
torch.set_num_threads(16)
a = json.load(open(f"{D}/inference/config.json"))
args = pytypes.SimpleNamespace(**{k: v for k, v in a.items() if k.startswith("vision_") or k in ("dim", "image_token_id")})

# ---- 1. geometry sweep (pure function of the size) -------------------------------------------------------------
sizes = [(w, h) for w in (1, 13, 14, 15, 100, 333, 544, 640, 1000, 1024, 1366, 1920, 2560, 3840, 8000)
                for h in (1, 14, 97, 480, 544, 768, 1080, 1440, 2160, 5000)]
sizes += [(30000, 20), (20, 30000), (545, 543), (2000, 2000), (1179, 2556), (2556, 1179)]
with open(f"{OUT}/geom.txt", "w") as f:
    for w, h in sizes:
        n_llm_h, n_llm_w, bh, bw = IP.plan_image_grid(w, h, args)
        f.write(f"{w} {h} {n_llm_h} {n_llm_w} {bh} {bw}\n")
print(f"geom.txt: {len(sizes)} sizes")

# ---- 2. fixtures -----------------------------------------------------------------------------------------------
def font(px):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", px)
    except OSError: return ImageFont.load_default()

def fixture_text():                      # 640x480: big text + shapes (also the end-to-end "read this" image)
    im = Image.new("RGB", (640, 480), (255, 255, 255)); d = ImageDraw.Draw(im)
    d.rectangle((40, 40, 240, 200), fill=(220, 30, 30)); d.ellipse((380, 50, 580, 250), fill=(30, 60, 220))
    d.polygon([(320, 260), (220, 440), (420, 440)], fill=(30, 170, 60))
    d.text((60, 300), "MACHX 4721", font=font(72), fill=(0, 0, 0))
    return im

def fixture_screen():                    # 1920x1080: a screenshot-like page; hits the 1,024-token budget
    im = Image.new("RGB", (1920, 1080), (24, 26, 32)); d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1920, 64), fill=(45, 50, 62)); d.text((24, 12), "Dream — Studio", font=font(36), fill=(235, 235, 240))
    for i in range(12):
        y = 110 + i * 78
        d.rectangle((40, y, 1880, y + 60), outline=(90, 96, 110), width=2)
        d.text((60, y + 12), f"row {i:02d}   value {(i * 7919) % 1000:03d}   status {'OK' if i % 3 else 'FAIL'}", font=font(30),
               fill=(120, 220, 140) if i % 3 else (240, 110, 110))
    return im

def fixture_tall():                      # 200x900: below min_pixels -> upscaled; tall aspect -> grey side bars
    rng = np.random.default_rng(20260918)
    arr = (rng.random((900, 200, 3)) * 255).astype(np.uint8)
    arr[::50] = 0
    return Image.fromarray(arr, "RGB")

fixtures = [fixture_text(), fixture_screen(), fixture_tall()]

# ---- 3. the shipped modules, shipped weights, fp32 ---------------------------------------------------------------
torch.set_default_dtype(torch.float32)
vit, ali = V.ViT(args), V.Aligner(args)
index = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
want = {k: f for k, f in index.items() if k.startswith(("vision.", "aligner."))}
sd_v, sd_a = {}, {}
for shard in sorted(set(want.values())):
    with safe_open(f"{D}/{shard}", "pt") as st:
        for k in (k for k, f in want.items() if f == shard):
            t = st.get_tensor(k).float()
            (sd_v if k.startswith("vision.") else sd_a)[k.split(".", 1)[1]] = t
vit.load_state_dict(sd_v, strict=True); ali.load_state_dict(sd_a, strict=True)
vit.eval(); ali.eval()
print(f"weights: {len(sd_v)} vision + {len(sd_a)} aligner tensors, fp32")

for k, im in enumerate(fixtures):
    path = f"{OUT}/img_{k}.png"; im.save(path)
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = IP.load_image({"url": path}, args)      # the shipped loader
    # the pixel tensor load_image patchified, rebuilt from its patches (exact inverse of the reshape/permute)
    p = args.vision_patch_size
    x = patches.float().reshape(n_vit_h, n_vit_w, 3, p, p).permute(2, 0, 3, 1, 4).reshape(3, n_vit_h * p, n_vit_w * p)
    rows = ali(vit(patches.float(), n_vit_h, n_vit_w), n_vit_h, n_vit_w)                  # [n_llm_h*n_llm_w, dim]
    assert rows.shape == (n_llm_h * n_llm_w, args.dim), rows.shape
    ty = IP.image_token_types(n_llm_h, n_llm_w).to(torch.int32)
    with open(f"{OUT}/golden_{k}.bin", "wb") as f:
        f.write(b"DS41VOR1")
        f.write(struct.pack("<8I", n_vit_h, n_vit_w, n_llm_h, n_llm_w, ty.numel(), args.dim, n_vit_h * n_vit_w, 0))
        f.write(x.contiguous().numpy().astype("<f4").tobytes())
        f.write(rows.contiguous().numpy().astype("<f4").tobytes())
        f.write(ty.numpy().astype("<i4").tobytes())
    print(f"golden_{k}.bin: image {im.width}x{im.height} -> vit {n_vit_h}x{n_vit_w} ({n_vit_h * n_vit_w} patches), "
          f"llm {n_llm_h}x{n_llm_w}, {ty.numel()} tokens, rows |mean| {rows.abs().mean():.4f} finite {bool(torch.isfinite(rows).all())}")
print("done")
