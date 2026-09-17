#!/usr/bin/env python3
"""npz -> raw-bin exporter for the qwen4exp C++ parity gate.

Usage: python3 qwen4exp_npz_to_bins.py <in.npz> <out_dir>

Writes one <key>.bin per array (fp32 arrays as float32 little-endian,
integer arrays as int32) plus manifest.txt lines: `<key> <dtype> <shape,>`.
"""
import sys, os
import numpy as np

def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    z = np.load(src)
    with open(os.path.join(out, "manifest.txt"), "w") as mf:
        for k in z.files:
            a = np.asarray(z[k])
            if a.dtype.kind in "SU":   # string metadata (layer_type etc.) — skip
                continue
            if a.dtype.kind in "iu":
                a = a.astype(np.int32)
                dt = "i32"
            else:
                a = a.astype(np.float32)
                dt = "f32"
            a.tofile(os.path.join(out, f"{k}.bin"))
            mf.write(f"{k} {dt} {','.join(str(d) for d in a.shape)}\n")
    print(f"wrote {len(z.files)} arrays to {out}")

if __name__ == "__main__":
    main()
