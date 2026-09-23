#!/usr/bin/env python3
"""P2 arbiter (docs/mimo26/00_PORT_PLAN.md): per-layer distance of the engine's residual stream from the fp32 reference.

  bisect_layers.py <ref dump dir> <engine dump dir> <n_layers> <seq lengths...>

Both dirs hold x_L{L}_s{S}.f32 [T, H] (tools/mimo26/ref_forward.py --dump-dir, IE_MIMO26_DUMP). Per layer: the
relative L2 error ||x_eng - x_ref|| / ||x_ref|| over all positions (median and max over positions), and the worst
position. Smooth growth at fp16-noise scale = no localized defect; a step at one layer = look there.
"""
import os
import sys

import numpy as np


def main():
    rd, ed, nL = sys.argv[1], sys.argv[2], int(sys.argv[3])
    lens = list(map(int, sys.argv[4:]))
    H = 4096
    print(f"{'layer':>5} {'median rel':>11} {'max rel':>9} {'worst (seq,pos)':>16}")
    for L in range(nL):
        rels = []
        for s, T in enumerate(lens):
            a = np.fromfile(os.path.join(rd, f"x_L{L}_s{s}.f32"), dtype="<f4").reshape(T, H).astype(np.float64)
            b = np.fromfile(os.path.join(ed, f"x_L{L}_s{s}.f32"), dtype="<f4").reshape(T, H).astype(np.float64)
            r = np.linalg.norm(b - a, axis=1) / np.maximum(np.linalg.norm(a, axis=1), 1e-30)
            rels += [(float(r[t]), s, t) for t in range(T)]
        v = np.array([x[0] for x in rels]); w = max(rels)
        print(f"{L:5d} {np.median(v):11.2e} {v.max():9.2e} {('(%d,%d)' % (w[1], w[2])):>16}")


if __name__ == "__main__":
    main()
