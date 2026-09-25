#!/usr/bin/env python3
"""#74 (docs/mimo26/P7_FIX64_FIX70.md section 6): the engine's MoE output row against an fp64 recomputation from the checkpoint.

  moe_agree.py <model_dir> <moe_L{L}_pos{P}[tag].bin> <moeout_L{L}_pos{P}[tag].f32>

The .bin is the row's post-norm MoE input, top-k ids and routing weights (ie-mimo26-nan-probe --dump, the
IE_MIMO26_CHECK_DUMP layout); the .f32 is the engine's MoE output row [H]. Prints whether the engine row is finite, its
relative L2 error against sum_k w_k * down_k(silu(gate_k x) * up_k x) in fp64, and the largest |difference|.
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_recompute import mxfp4  # noqa: E402 -- the same MXFP4 dequant as the P2 recompute


def main():
    model, path_in, path_out = sys.argv[1], sys.argv[2], sys.argv[3]
    L = int(re.search(r"moe_L(\d+)_", path_in).group(1))
    cfg = json.load(open(os.path.join(model, "config.json")))
    H, EF, TK = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["num_experts_per_tok"]
    raw = open(path_in, "rb").read()
    x = np.frombuffer(raw[:4 * H], dtype="<f4").astype(np.float64)
    ids = np.frombuffer(raw[4 * H:4 * H + 4 * TK], dtype="<i4")
    w = np.frombuffer(raw[4 * H + 4 * TK:4 * H + 8 * TK], dtype="<f4").astype(np.float64)
    eng = np.fromfile(path_out, dtype="<f4")
    if eng.size != H:
        sys.exit(f"{path_out}: {eng.size} values, expected {H}")
    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    y = np.zeros(H)
    for e, we in zip(ids, w):
        p = f"model.layers.{L}.mlp.experts.{e}."
        g = mxfp4(model, index, p + "gate_proj.weight", EF, H) @ x
        u = mxfp4(model, index, p + "up_proj.weight", EF, H) @ x
        y += we * (mxfp4(model, index, p + "down_proj.weight", H, EF) @ (g / (1 + np.exp(-g)) * u))
    finite = bool(np.isfinite(eng).all())
    print(f"layer {L}: engine row finite: {finite} ({int((~np.isfinite(eng)).sum())} non-finite of {H})")
    if not finite:
        sys.exit(1)
    d = eng.astype(np.float64) - y
    rel = np.linalg.norm(d) / max(np.linalg.norm(y), 1e-30)
    i = int(np.abs(d).argmax())
    print(f"fp64 reference: max|y| {np.abs(y).max():.2f}; engine vs reference: relative L2 {rel:.3e}, "
          f"max|diff| {np.abs(d).max():.4f} at dim {i} (engine {eng[i]:.4f}, reference {y[i]:.4f}), cosine "
          f"{float(eng @ y / (np.linalg.norm(eng) * np.linalg.norm(y))):.6f}")


if __name__ == "__main__":
    main()
