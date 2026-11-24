#!/usr/bin/env python3
"""Compare one ie-inkling2-run --dump-calls folder with the reference goldens (SPEC.md section 16).

  python -I compare.py --dump DIR --goldens GOLD --mode a|b|c [--routes-ref routes_ref.json]

Prints a 48-row table (call, layer, pass, max |d|, relative L2), the final-logit error, both top-10 lists and the
expert-pick differences, then a PASS / FAIL line for the mode's criteria:
  a  teacher-forced:              every call <= 5e-3
  b  end to end, reference picks: every call <= 1.2e-2, logits rel L2 <= 8e-3, top-1 == 152717
  c  free-running:                top-1 == 152717 (calls above 2e-2 are listed with the pick swaps that explain them)
Exit status 0 on PASS, 1 on FAIL.
"""
import argparse, glob, json, os, sys
import numpy as np

H = 2304
TOP1 = 152717
# SPEC 16.3: the near-tie picks where an fp32 implementation's own top-8 differs from the reference's (call, token)
SPEC_NEAR_TIES = {(2, 3), (10, 3), (31, 2), (37, 1)}


def golden_file(d, i):
    for pat in (f"call{i:02d}_layer[0-9][0-9]_o.npy", f"call{i:02d}_layer[0-9][0-9].npy"):
        hit = sorted(glob.glob(os.path.join(d, pat)))
        if hit:
            return hit[0]
    sys.exit(f"no golden for call {i} in {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--mode", choices=["a", "b", "c"], required=True)
    ap.add_argument("--routes-ref", default=None)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.goldens, "meta.json")))
    routes = json.load(open(os.path.join(a.dump, "routes.json")))
    ref = json.load(open(a.routes_ref)) if a.routes_ref else None
    T = len(meta["ids"])
    sched = [c["layer"] for c in meta["calls"]] if "calls" in meta else None

    print(f"{'call':>4} {'layer':>5} {'pass':>4} {'max|d|':>9} {'relL2':>10}")
    rels, worst = [], (0.0, -1)
    seen = {}
    for i in range(48):
        g = np.load(golden_file(a.goldens, i)).astype(np.float32).reshape(-1, H)
        fn = sorted(glob.glob(os.path.join(a.dump, f"call{i:02d}_layer*.f32")))
        if not fn:
            sys.exit(f"no dump for call {i} in {a.dump}")
        layer = int(os.path.basename(fn[0])[12:14])
        if sched is not None and sched[i] != layer:
            sys.exit(f"call {i}: dump says layer {layer}, goldens say {sched[i]}")
        seen[layer] = seen.get(layer, 0) + 1
        m = np.fromfile(fn[0], dtype=np.float32).reshape(-1, H)
        d = m - g
        rel = float(np.linalg.norm(d) / np.linalg.norm(g))
        rels.append(rel)
        if rel > worst[0]:
            worst = (rel, i)
        flag = ""
        if a.mode == "a" and rel > 5e-3: flag = "  <-- > 5e-3"
        if a.mode == "b" and rel > 1.2e-2: flag = "  <-- > 1.2e-2"
        if a.mode == "c" and rel > 2e-2: flag = "  > 2e-2"
        p = seen[layer] if 6 <= layer < 18 else "-"
        print(f"{i:4d} {layer:5d} {p!s:>4} {np.abs(d).max():9.5f} {rel:10.3e}{flag}")
    print(f"worst call: {worst[1]} relL2 {worst[0]:.3e}")

    glast = np.load(os.path.join(a.goldens, "logits_last.npy")).astype(np.float32)
    mine = np.fromfile(os.path.join(a.dump, "logits_last.f32"), dtype=np.float32)[: glast.size]
    d = mine - glast
    lrel = float(np.linalg.norm(d) / np.linalg.norm(glast))
    top = np.argsort(-mine, kind="stable")[:10]
    gt = [t[0] for t in meta["top10"]]
    print(f"final logits (last position): relL2 {lrel:.3e}  max|d| {np.abs(d).max():.5f}")
    print("top-10 mine  :", [(int(t), round(float(mine[t]), 4)) for t in top])
    print("top-10 golden:", [(t[0], round(t[1], 4)) for t in meta["top10"]])
    tie_order = all(glast[int(t)] == glast[g] for t, g in zip(top, gt))
    print(f"top-1 {int(top[0])} ({'same' if int(top[0]) == TOP1 else 'DIFFERENT'}); top-10 set "
          f"{'same' if set(gt) == set(top.tolist()) else 'different'}; order "
          f"{'same' if gt == top.tolist() else ('same up to exact ties in the golden' if tie_order else 'different')}")

    # expert picks: this run's own top-8 against the reference's
    if ref is not None:
        diffs = []
        for i in range(48):
            for t in range(T):
                own = sorted(routes["own"][i][t])
                if own != sorted(ref[i][t]):
                    diffs.append((i, t, routes["margin"][i][t]))
        what = "imposed (reference) picks" if a.mode in ("a", "b") else "reference's picks"
        print(f"own top-8 differs from the {what} in {len(diffs)} of {48 * T} (call, token) pairs"
              + ("; in free-running mode differences after the first swap follow from it" if a.mode == "c" else ""))
        for i, t, mg in diffs:
            tag = "  (named in SPEC 16.3)" if (i, t) in SPEC_NEAR_TIES else ""
            print(f"   call {i:2d} token {t}: own 8th-9th margin {mg:.2e}{tag}")

    if a.mode == "a":
        ok = max(rels) <= 5e-3
        crit = "every call <= 5e-3"
    elif a.mode == "b":
        ok = max(rels) <= 1.2e-2 and lrel <= 8e-3 and int(top[0]) == TOP1
        crit = "every call <= 1.2e-2, logits <= 8e-3, top-1 152717"
    else:
        ok = int(top[0]) == TOP1
        crit = "top-1 152717"
        over = [i for i, r in enumerate(rels) if r > 2e-2]
        print(f"calls above 2e-2: {over if over else 'none'}")
    print(f"RESULT ({a.mode}): {'PASS' if ok else 'FAIL'} ({crit})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
