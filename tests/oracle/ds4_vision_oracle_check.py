#!/usr/bin/env python3
"""P1 gate: numpy oracle vs the OFFICIAL torch reference modules, same weights, same pixels.

Runs `inference/vision.py` (ViT, Aligner) and `inference/image_processor.py`
(`build_image_block`) from the deepseek-ai/DeepSeek-V4-Flash-Vision-Exp repo in fp32
on CPU against a golden file written by ds4_vision_oracle.py. Reports max|Δ| and
min row-cosine for the aligner output, and exact equality for types/perm.

    ~/venv/bin/python tests/oracle/ds4_vision_oracle_check.py \
        --official-dir <repo>/inference --tower T.safetensors --config config.json --golden golden.bin
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MAGIC = b"DS4VORC1"


def read_golden(path: Path):
    b = path.read_bytes()
    assert b[:8] == MAGIC, b[:8]
    n_vit_h, n_vit_w, n_llm_h, n_llm_w, start_pos, n_rows, hidden, n_patches = struct.unpack("<8I", b[8:40])
    off = 40
    def take(dtype, count):
        nonlocal off
        arr = np.frombuffer(b, dtype=dtype, count=count, offset=off)
        off += arr.nbytes
        return arr
    pixels = take("<f4", 3 * n_vit_h * 14 * n_vit_w * 14).reshape(3, n_vit_h * 14, n_vit_w * 14)
    types = take("<i4", n_rows)
    perm = take("<i4", n_llm_h * n_llm_w)
    aligned = take("<f4", n_llm_h * n_llm_w * hidden).reshape(n_llm_h * n_llm_w, hidden)
    rows = take("<f4", n_rows * hidden).reshape(n_rows, hidden)
    assert off == len(b), (off, len(b))
    return dict(n_vit_h=n_vit_h, n_vit_w=n_vit_w, n_llm_h=n_llm_h, n_llm_w=n_llm_w, start_pos=start_pos,
                pixels=pixels, types=types, perm=perm, aligned=aligned, rows=rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-dir", required=True, type=Path)
    ap.add_argument("--tower", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--golden", required=True, type=Path)
    ap.add_argument("--max-abs", type=float, default=1e-3)
    ap.add_argument("--min-cos", type=float, default=0.9999)
    ap.add_argument("--dtype", default="float32", help="run the official modules in float32 or bfloat16")
    a = ap.parse_args()

    sys.path.insert(0, str(a.official_dir))
    import torch
    from safetensors.torch import load_file
    from vision import ViT, Aligner                     # official
    from image_processor import build_image_block      # official

    torch.manual_seed(0)
    cfg = json.loads(a.config.read_text())
    args = SimpleNamespace(**cfg, dim=cfg["hidden_size"])
    g = read_golden(a.golden)

    state = load_file(str(a.tower), device="cpu")
    vit, al = ViT(args), Aligner(args)
    vit.load_state_dict({k.removeprefix("vision."): v for k, v in state.items() if k.startswith("vision.")}, strict=True)
    al.load_state_dict({k.removeprefix("aligner."): v for k, v in state.items() if k.startswith("aligner.")}, strict=True)
    dt = getattr(torch, a.dtype)
    vit.to(dt).eval(); al.to(dt).eval()

    p = cfg["vision_patch_size"]
    x = torch.from_numpy(g["pixels"].copy())
    patches = x.reshape(3, g["n_vit_h"], p, g["n_vit_w"], p).permute(1, 3, 0, 2, 4).reshape(-1, 3, p, p)
    with torch.inference_mode():
        ref = al(vit(patches.to(dt), g["n_vit_h"], g["n_vit_w"]), g["n_vit_h"], g["n_vit_w"]).float().numpy()
    types_ref, perm_ref = build_image_block(g["n_llm_h"], g["n_llm_w"], int(g["start_pos"]))
    types_ref, perm_ref = types_ref.numpy().astype(np.int32), perm_ref.numpy().astype(np.int32)

    ours = g["aligned"]
    diff = np.abs(ours - ref)
    cos = np.sum(ours * ref, -1) / (np.linalg.norm(ours, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-30)
    types_ok = types_ref.shape == g["types"].shape and bool((types_ref == g["types"]).all())
    perm_ok = perm_ref.shape == g["perm"].shape and bool((perm_ref == g["perm"]).all())
    rep = {
        "rows": int(ref.shape[0]), "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
        "ref_abs_mean": float(np.abs(ref).mean()), "min_cos": float(cos.min()),
        "types_equal": types_ok, "perm_equal": perm_ok,
        "pass": bool(diff.max() < a.max_abs and cos.min() > a.min_cos and types_ok and perm_ok),
    }
    print(json.dumps(rep))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
