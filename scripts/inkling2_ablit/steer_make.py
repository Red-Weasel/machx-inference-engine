#!/usr/bin/env python3
"""Build steering ("mode": "add") sets for Inkling 2 Mini from two recordings (e.g. close vs mid reasoning stretches).

  python steer_make.py --a CLOSE.f32 --b MID.f32 --out DIR NAME=SPEC [NAME=SPEC ...]

SPEC = KINDS:CALLS:BETA[:SHAPE], e.g. am:7-41:0.5  a:12-29:1  m:7-41:0.5:peak24w8
  KINDS  a = attention writes, m = MoE writes (am = both)
  CALLS  A-B (inclusive) or a comma list
  BETA   multiplier of the raw class-mean difference (mean_A - mean_B) of that write at that call: the vector added
         to the write is BETA * profile(call) * (mean_A - mean_B)  (not unit: the "rows" hold the raw difference)
  SHAPE  optional per-call profile: peakCwW = exp(-((c - C) / W)^2) ; default flat 1
  randfixS  optional: the control with ONE random direction (seed S) on every write (adds up like the difference does)
  randS  optional: the control: a random direction (seed S) of the same norm on every write, instead of the difference
Also: NAME=remove:CALL:KIND:STRENGTH  a removal set of the unit partial-state ("kind" 0..3) direction at that call on
every write (for the "remove the keep-checking component" test). Prints each set's added norm relative to the mean
write norm per call (the size check). Writes DIR/NAME.json + .f32 and appends NAME<TAB>path to DIR/list.tsv.
"""
import argparse, json, math, os
import numpy as np

H, NC, NK = 2304, 48, 4


def load(p): return np.fromfile(p, dtype=np.float32).reshape(-1, NC, NK, H).astype(np.float64)


def calls(s):
    if "-" in s: a, b = map(int, s.split("-")); return list(range(a, b + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("sets", nargs="+")
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    diff = A.mean(0) - B.mean(0)                          # [48][4][H]
    wnorm = np.sqrt((np.concatenate([A, B]) ** 2).sum(-1)).mean(0)   # [48][4] mean norm of each recorded vector
    os.makedirs(a.out, exist_ok=True)
    lst = open(os.path.join(a.out, "list.tsv"), "a")
    for item in a.sets:
        name, spec = item.split("=", 1)
        rows, groups, rel = [], [], []
        f = spec.split(":")
        if f[0] == "remove":
            c, k, s = int(f[1]), int(f[2]), float(f[3])
            u = diff[c, k] / np.linalg.norm(diff[c, k]); rows.append(u)
            groups.append({"rows": [0, 1], "strength": s, "embed": True})
            groups += [{"rows": [0, 1], "strength": s, "attn": "all"}, {"rows": [0, 1], "strength": s, "moe": "all"}]
        else:
            kinds, cs, beta = f[0], calls(f[1]), float(f[2])
            prof, rng = (lambda c: 1.0), None
            for opt in f[3:]:
                if opt.startswith("peak"):
                    C, W = opt[4:].split("w"); C, W = float(C), float(W); prof = lambda c, C=C, W=W: math.exp(-((c - C) / W) ** 2)
                elif opt.startswith("randfix"):                     # one random direction for every write (coherent)
                    fixed = np.random.default_rng(int(opt[7:] or 0)).standard_normal(H); rng = "fix"
                elif opt.startswith("rand"): rng = np.random.default_rng(int(opt[4:] or 0))
            for c in cs:
                for kch, k, key in (("a", 1, "attn"), ("m", 2, "moe")):
                    if kch not in kinds: continue
                    w = beta * prof(c)
                    if w < 1e-3: continue
                    v = diff[c, k]
                    if rng is not None:                       # control: a random direction of the same size
                        r = fixed if isinstance(rng, str) else rng.standard_normal(H)
                        v = r / np.linalg.norm(r) * np.linalg.norm(v)
                    rows.append(v); groups.append({"rows": [len(rows) - 1, 1], "strength": round(w, 5), "mode": "add", key: [c]})
                    rel.append(w * np.linalg.norm(diff[c, k]) / wnorm[c, k])
        np.asarray(rows, dtype=np.float32).tofile(os.path.join(a.out, name + ".f32"))
        j = {"format": "ie-ablation-v1", "arch": "inkling2_loop", "hidden": H, "file": name + ".f32", "rows": len(rows),
             "groups": groups, "name": name, "spec": spec}
        json.dump(j, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
        lst.write(f"{name}\t{os.path.abspath(os.path.join(a.out, name + '.json'))}\n")
        print(f"{name}: {len(groups)} groups" + (f"; added norm / write norm: mean {np.mean(rel):.3f}, max {np.max(rel):.3f}" if rel else ""))


if __name__ == "__main__": main()
