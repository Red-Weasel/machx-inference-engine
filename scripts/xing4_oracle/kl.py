#!/usr/bin/env python3
"""KL of the engine's next-token distributions against the reference oracle's, row by row.

  python kl.py <oracle dir> <engine logits.f32> [--skip 0]

Prints the mean / max KL(ref || engine), the share of rows with the same top token, and the worst rows.
"""
import json, os, sys
import numpy as np

def lsm(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))

def main():
    od, ef = sys.argv[1], sys.argv[2]
    skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0
    meta = json.load(open(os.path.join(od, "meta.json")))
    T, V = meta["T"], meta["vocab"]
    ref = np.fromfile(os.path.join(od, "logits.f32"), np.float32).reshape(T, V)[skip:].astype(np.float64)
    eng = np.fromfile(ef, np.float32).reshape(-1, V)[skip:T].astype(np.float64)
    lr, le = lsm(ref), lsm(eng)
    kl = (np.exp(lr) * (lr - le)).sum(-1)
    same = (ref.argmax(-1) == eng.argmax(-1))
    print(f"rows {len(kl)}: KL mean {kl.mean():.5f}  median {np.median(kl):.5f}  max {kl.max():.4f}  same top token {same.sum()} / {len(kl)}")
    worst = np.argsort(-kl)[:5]
    print("worst rows:", ", ".join(f"{int(i) + skip}:{kl[i]:.3f}" for i in worst))

if __name__ == "__main__":
    main()
