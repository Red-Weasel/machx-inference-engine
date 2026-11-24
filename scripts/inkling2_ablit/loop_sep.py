#!/usr/bin/env python3
"""Where "about to conclude" separates from "keep reasoning" / "looping" in Inkling 2 Mini's stream (neutral prompts).

  python loop_sep.py --a CLOSE.f32 --b MID.f32 [--label close-vs-mid] [--top 12] [--save-dirs D.npy] [--energy REF.f32 ...]

A / B: ie-inkling2-run --batch --acts-out --acts-last N --acts-mean recordings ([48 calls][4 kinds][2304] per reply).
For every (call, kind): the mean-difference direction fitted on even rows, d' and AUROC on the odd rows, and the
direction's share of the energy of ordinary states (mean (x.r)^2 / mean |x|^2 over the --energy recordings, same call
and kind; default: A and B together). Prints the top (call, kind) by AUROC and a per-kind summary over calls.
--save-dirs writes the full-data unit directions [48][4][2304] for direction building.
"""
import argparse
import numpy as np

H, NC, NK = 2304, 48, 4
KIND = ["partial", "attn_w", "moe_w", "moe_in"]


def load(p):
    x = np.fromfile(p, dtype=np.float32)
    assert x.size % (NC * NK * H) == 0, p
    return x.reshape(-1, NC, NK, H).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True); ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--label", default=""); ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--save-dirs"); ap.add_argument("--energy", nargs="*")
    a = ap.parse_args()
    A = np.concatenate([load(p) for p in a.a]); B = np.concatenate([load(p) for p in a.b])
    E = np.concatenate([load(p) for p in a.energy]) if a.energy else np.concatenate([A, B])
    print(f"{a.label}: A {len(A)} rows, B {len(B)} rows, energy reference {len(E)} rows")
    res, dirs = [], np.zeros((NC, NK, H))
    for c in range(NC):
        for k in range(NK):
            u = A[0::2, c, k].mean(0) - B[0::2, c, k].mean(0); u /= max(np.linalg.norm(u), 1e-30)
            pa, pb = A[1::2, c, k] @ u, B[1::2, c, k] @ u
            d = (pa.mean() - pb.mean()) / np.sqrt(0.5 * (pa.var(ddof=1) + pb.var(ddof=1)) + 1e-30)
            au = float((pa[:, None] > pb[None, :]).mean() + 0.5 * (pa[:, None] == pb[None, :]).mean())
            f = A[:, c, k].mean(0) - B[:, c, k].mean(0); f /= max(np.linalg.norm(f), 1e-30); dirs[c, k] = f
            e = E[:, c, k]; share = float(((e @ f) ** 2).mean() / ((e * e).sum(1).mean() + 1e-30))
            res.append((au, d, c, k, share))
    res.sort(reverse=True)
    print(" call kind     AUROC    d'   energy share")
    for au, d, c, k, sh in res[:a.top]: print(f"  {c:3d} {KIND[k]:8s} {au:.3f} {d:6.2f}  {100 * sh:.2f} %")
    for k in range(NK):
        r = [x for x in res if x[3] == k]; best = max(r)
        print(f"kind {KIND[k]:8s}: best call {best[2]} AUROC {best[0]:.3f}; calls with AUROC >= 0.85: {sorted(x[2] for x in r if x[0] >= 0.85)}")
    # do neighbouring calls agree (one direction or many)?
    P = dirs[:, 0]
    cs = [float(P[c] @ P[c + 1]) for c in range(NC - 1)]
    print("partial-state direction cosine with the next call, calls 0..46:", " ".join(f"{x:.2f}" for x in cs))
    if a.save_dirs: np.save(a.save_dirs, dirs.astype(np.float32)); print("saved", a.save_dirs)


if __name__ == "__main__": main()
