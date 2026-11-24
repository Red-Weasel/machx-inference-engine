#!/usr/bin/env python3
"""Check an ablation set inside the running model: what is left along its directions, per call and recorded kind.

  python check_orth.py --acts A.f32 --ablation SET.json [--P 1]

A.f32 = ie-inkling2-run --batch --acts-out recorded WITH the set active. Kinds: 0 the partial state after the call (must
be ~0 along the set when the set covers every write: it is a sum of ablated writes, or at a block boundary the ablated
attention write plus the ablated MoE write), 1 / 2 the attention / MoE writes BEFORE ablation (not 0: what was removed),
3 the MoE read-out u (a mix of RMS-normalised block states and partials, so ~0 too -- except where the mlp controller of
passes 2-3 multiplies it elementwise with a message, which is a READ, never written back).
Prints, per call, max over prompts and positions of |V r| / |v| (the norm of the part in the set's span, relative).
"""
import argparse, json, os
import numpy as np

H, NC, NK = 2304, 48, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True); ap.add_argument("--ablation", required=True); ap.add_argument("--P", type=int, default=1)
    a = ap.parse_args()
    j = json.load(open(a.ablation))
    R = np.fromfile(os.path.join(os.path.dirname(os.path.abspath(a.ablation)), j["file"]), dtype=np.float32).reshape(j["rows"], H).astype(np.float64)
    X = np.fromfile(a.acts, dtype=np.float32).reshape(-1, a.P, NC, NK, H).astype(np.float64)
    sch = [(L, 1) for L in range(6)] + [(L, p) for p in (1, 2, 3) for L in range(6, 18)] + [(L, 1) for L in range(18, 24)]
    print(f"{X.shape[0]} prompts x {a.P} positions; set: {j['rows']} direction(s)")
    print(f"{'call':>4} {'layer':>5} {'pass':>4} " + " ".join(f"{k:>12s}" for k in ["partial", "attn_w(pre)", "moe_w(pre)", "moe_in"]))
    worst = 0.0
    for c in range(NC):
        cells = []
        for k in range(NK):
            v = X[:, :, c, k].reshape(-1, H)
            rel = np.linalg.norm(v @ R.T, axis=1) / np.maximum(np.linalg.norm(v, axis=1), 1e-30)
            cells.append(rel.max())
        worst = max(worst, cells[0])
        print(f"{c:>4} {sch[c][0]:>5} {sch[c][1]:>4} " + " ".join(f"{x:12.2e}" for x in cells))
    print(f"largest relative component left in the partial state over all calls: {worst:.2e}")


if __name__ == "__main__": main()
