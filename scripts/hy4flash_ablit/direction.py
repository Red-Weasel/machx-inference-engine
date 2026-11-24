#!/usr/bin/env python3
"""Hy4-flash abliteration, step 2: the refusal direction from recorded activations.

  python direction.py --harmful acts_h.f32 --harmless acts_b.f32 --out <dir> [--layer K] [--stream mean|0|1|2|3]
                      [--min-depth 0.2] [--max-depth 0.8] [--layers 42] [--hc 4] [--dim 3072]

The two files are ie-hy4flash-run --batch --acts-out recordings: [prompts, layers, hc, dim] fp32, the residual streams
ENTERING every layer at each prompt's last position (layer k's entry = the state after layer k - 1). For every layer and
stream (and for the mean of the streams) the direction is the difference of the class means; the table shows how well
it separates the classes (d', AUROC), its size relative to the activations, and how the streams' directions agree.
Default choice: the layer with the best d' inside the depth window, on the mean of the streams. Writes direction.npy
([dim] unit vector, fp32) and direction.json (the table and the choice).
"""
import argparse, json, os
import numpy as np

def separation(xh, xb):
    mu_h, mu_b = xh.mean(0), xb.mean(0)
    r = mu_h - mu_b; rn = np.linalg.norm(r); u = r / max(rn, 1e-30)
    ph, pb = xh @ u, xb @ u
    dprime = (ph.mean() - pb.mean()) / np.sqrt(0.5 * (ph.var(ddof=1) + pb.var(ddof=1)) + 1e-30)
    auroc = (ph[:, None] > pb[None, :]).mean()
    rel = rn / np.linalg.norm(np.concatenate([xh, xb]), axis=1).mean()
    return u, {"dprime": float(dprime), "auroc": float(auroc), "diff_norm": float(rn), "rel_norm": float(rel)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful", required=True); ap.add_argument("--harmless", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int); ap.add_argument("--stream", default="mean")
    ap.add_argument("--min-depth", type=float, default=0.2); ap.add_argument("--max-depth", type=float, default=0.8)
    ap.add_argument("--hc", type=int, default=4); ap.add_argument("--dim", type=int, default=3072); ap.add_argument("--layers", type=int, default=42)
    a = ap.parse_args()
    def load(p):
        x = np.fromfile(p, dtype=np.float32); per = a.hc * a.dim
        return x, per
    xh, per = load(a.harmful); xb, _ = load(a.harmless)
    # layers from the file sizes: both files hold whole prompts of the same layer count
    L = a.layers
    if xh.size % (L * per) or xb.size % (L * per): raise SystemExit(f"the files do not hold whole prompts of {L} layers x {a.hc} streams x {a.dim}")
    xh = xh.reshape(-1, L, a.hc, a.dim).astype(np.float64); xb = xb.reshape(-1, L, a.hc, a.dim).astype(np.float64)
    print(f"activations: {xh.shape[0]} harmful, {xb.shape[0]} harmless, {L} layers, {a.hc} streams, dim {a.dim}")
    if not (np.isfinite(xh).all() and np.isfinite(xb).all()): raise SystemExit("non-finite activations")
    views = ["mean"] + [str(i) for i in range(a.hc)]
    def view(x, v): return x.mean(2) if v == "mean" else x[:, :, int(v)]
    table, dirs = [], {}
    for k in range(L):
        row = {"layer": k}
        for v in views:
            u, st = separation(view(xh, v)[:, k], view(xb, v)[:, k]); dirs[(k, v)] = u; row[v] = st
        row["stream_cos_min"] = float(min(abs(dirs[(k, str(i))] @ dirs[(k, str(j))]) for i in range(a.hc) for j in range(i)))
        row["cos_mean_vs_streams"] = [float(dirs[(k, "mean")] @ dirs[(k, str(i))]) for i in range(a.hc)]
        row["cos_prev_layer"] = float(dirs[(k, a.stream)] @ dirs[(k - 1, a.stream)]) if k else float("nan")
        table.append(row)
    lo, hi = int(round(a.min_depth * L)), int(round(a.max_depth * L))
    best = max((r for r in table if lo <= r["layer"] <= hi), key=lambda r: r[a.stream]["dprime"])
    chosen = a.layer if a.layer is not None else best["layer"]
    print(f"\nstream view '{a.stream}'; candidates: layers {lo}..{hi}")
    print(f"{'layer':>5} {'dprime':>8} {'auroc':>7} {'|diff|':>8} {'rel':>6} {'cos_prev':>8} {'streams agree (min |cos|)':>26}   d' per stream")
    for r in table:
        st = r[a.stream]; mark = " <== chosen" if r["layer"] == chosen else (" (best in window)" if r["layer"] == best["layer"] else "")
        print(f"{r['layer']:>5} {st['dprime']:>8.3f} {st['auroc']:>7.4f} {st['diff_norm']:>8.3f} {st['rel_norm']:>6.3f} {r['cos_prev_layer']:>8.3f} {r['stream_cos_min']:>26.3f}   "
              + " ".join(f"{r[str(i)]['dprime']:.2f}" for i in range(a.hc)) + mark)
    u = dirs[(chosen, a.stream)].astype(np.float32)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "direction.npy"), u)
    np.save(os.path.join(a.out, "directions_all_layers.npy"), np.stack([dirs[(k, a.stream)] for k in range(L)]).astype(np.float32))
    json.dump({"layer": chosen, "stream": a.stream, "best_in_window": best["layer"], "window": [lo, hi], "n_harmful": int(xh.shape[0]),
               "n_harmless": int(xb.shape[0]), "chosen_stats": table[chosen][a.stream], "table": table}, open(os.path.join(a.out, "direction.json"), "w"), indent=1)
    print(f"\nchosen: layer {chosen} ({'given' if a.layer is not None else 'best d-prime in the window'}), d' {table[chosen][a.stream]['dprime']:.3f}, AUROC {table[chosen][a.stream]['auroc']:.4f}; saved to {a.out}")

if __name__ == "__main__": main()
