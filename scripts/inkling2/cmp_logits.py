#!/usr/bin/env python3
"""Compare two raw float32 logit files (ie-inkling2-run --dump-steps .../step0.f32): relative L2, max |d|, top-1, top-10.

  python -I cmp_logits.py A.f32 B.f32 [--tol 1e-4]     (exit 1 when relL2 > tol or the top-10 lists differ)
"""
import argparse, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--tol", type=float, default=1e-4)
a = ap.parse_args()
x, y = np.fromfile(a.a, np.float32), np.fromfile(a.b, np.float32)
rel = float(np.linalg.norm(x - y) / np.linalg.norm(y))
tx, ty = np.argsort(-x, kind="stable")[:10], np.argsort(-y, kind="stable")[:10]
same = tx.tolist() == ty.tolist()
print(f"relL2 {rel:.3e}  max|d| {np.abs(x - y).max():.3e}  top-1 {int(tx[0])} / {int(ty[0])}  top-10 {'identical' if same else 'DIFFERENT'}")
sys.exit(0 if rel <= a.tol and same else 1)
