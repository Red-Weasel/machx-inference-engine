#!/usr/bin/env python3
"""P2 debugging (docs/mimo26/00_PORT_PLAN.md): one MoE row recomputed in fp32/fp64 on the host from the checkpoint.

  expert_recompute.py <model_dir> <moe_L{L}_pos{p}.bin>

The .bin is what IE_MIMO26_CHECK_DUMP writes: the post-norm input row (H f32), the top-k expert ids (i32) and routing
weights (f32). For each expert: max |gate|, |up|, |silu(gate)*up| and |down output|, then the weighted sum -- which of
them would overflow an fp16 store (65504) and by how much.
"""
import json
import os
import re
import struct
import sys

import numpy as np
from safetensors import safe_open

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def mxfp4(model, index, name, N, K):
    with safe_open(os.path.join(model, index[name]), framework="np") as f:
        qs = f.get_tensor(name).reshape(N, K // 2)
    with safe_open(os.path.join(model, index[name + "_scale"]), framework="np") as f:
        sc = f.get_tensor(name + "_scale").reshape(N, K // 32)
    lo, hi = qs & 0xF, qs >> 4
    codes = np.stack([lo, hi], axis=-1).reshape(N, K)
    scale = np.ldexp(1.0, sc.astype(np.int64) - 127)
    return E2M1[codes] * np.repeat(scale, 32, axis=1)


def main():
    model, path = sys.argv[1], sys.argv[2]
    L = int(re.search(r"moe_L(\d+)_", path).group(1))
    cfg = json.load(open(os.path.join(model, "config.json")))
    H, EF, TK = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["num_experts_per_tok"]
    raw = open(path, "rb").read()
    x = np.frombuffer(raw[:4 * H], dtype="<f4").astype(np.float64)
    ids = np.frombuffer(raw[4 * H:4 * H + 4 * TK], dtype="<i4")
    w = np.frombuffer(raw[4 * H + 4 * TK:4 * H + 8 * TK], dtype="<f4")
    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    print(f"layer {L}: max|x| {np.abs(x).max():.2f}, experts {list(ids)}, weights {[round(float(v), 4) for v in w]}")
    y = np.zeros(H)
    for e, we in zip(ids, w):
        p = f"model.layers.{L}.mlp.experts.{e}."
        g = mxfp4(model, index, p + "gate_proj.weight", EF, H) @ x
        u = mxfp4(model, index, p + "up_proj.weight", EF, H) @ x
        h = g / (1 + np.exp(-g)) * u
        d = mxfp4(model, index, p + "down_proj.weight", H, EF) @ h
        y += we * d
        flag = " <-- fp16 OVERFLOW" if max(np.abs(g).max(), np.abs(u).max(), np.abs(h).max(), np.abs(d).max()) > 65504 else ""
        print(f"  expert {e:3d} w {we:.4f}: max|gate| {np.abs(g).max():10.1f} max|up| {np.abs(u).max():10.1f} "
              f"max|h| {np.abs(h).max():12.1f} max|down| {np.abs(d).max():12.1f} (dim {int(np.abs(d).argmax())}){flag}")
    print(f"weighted sum: max|y| {np.abs(y).max():.1f} at dim {int(np.abs(y).argmax())}")


if __name__ == "__main__":
    main()
