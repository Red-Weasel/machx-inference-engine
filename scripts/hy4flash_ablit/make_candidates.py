#!/usr/bin/env python3
"""Hy4-flash abliteration search: write candidate direction sets for ie-hy4flash-run --ablate-list.

  python make_candidates.py --acts-harmful H.f32 --acts-harmless B.f32 --out DIR [--layers 42] [--hc 4] [--dim 3072]
                            [--extra-basis FILE:K] RECIPE [RECIPE ...]

The two files are any two classes to tell apart ("harmful" = the one to move away from): harmful against harmless
prompts, or -- under a candidate that already works partly -- the prompts it still declines against the ones it answers.
Each file's prompt count is taken from its size. --extra-basis adds the K rows of an earlier set to every removal set
(the result is orthonormalised again), so a second round can extend the first.

The activation files are --acts-out dumps: [prompts, layers, streams, dim] fp32, the streams after each layer at a prompt's
last token. A direction is a normalised difference of class means. RECIPE (optionally followed by @STRENGTH and
+MASKFILE):
  none                      the unmodified model
  means:20,25,30            the stream-averaged direction of each listed layer
  streams:20,25,30          every stream's own direction at each listed layer (4 per layer)
  streams:19-41             ... of a layer range (inclusive); 19-41/2 = every second layer
  top:16:19-41              the 16 leading principal directions of the per-stream directions of the range
  steer:19-41@1.0           no removal: ADD alpha * (harmless mean - harmful mean) of every stream after each listed layer
                            (a control vector; alpha 1 moves an average harmful activation onto the average harmless one)
  streams:20,25,30,35,41&steer:19-41@0.5     a removal set and a steering set together (@ after the steering part = alpha)
A set is written as an orthonormal basis (SVD; components below --rank-tol of the largest singular value are dropped: the
directions of neighbouring layers are near-copies, and what separates near-copies is noise) and a line
"name <tab> file <tab> k <tab> strength [<tab> mask]" is appended to DIR/list.tsv.
"""
import argparse, os
import numpy as np

def layers_of(spec):
    out = []
    for part in spec.split(","):
        step = 1
        if "/" in part: part, s = part.split("/"); step = int(s)
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1, step))
        else: out.append(int(part))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-harmful", required=True); ap.add_argument("--acts-harmless", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--extra-basis"); ap.add_argument("--layers", type=int, default=42); ap.add_argument("--hc", type=int, default=4); ap.add_argument("--dim", type=int, default=3072)
    ap.add_argument("--rank-tol", type=float, default=0.03)
    ap.add_argument("recipes", nargs="+")
    a = ap.parse_args()
    def class_mean(path):
        x = np.memmap(path, dtype=np.float32, mode="r"); per = a.layers * a.hc * a.dim
        assert x.size % per == 0, f"{path}: not a whole number of [layers, streams, dim] records"
        return x.reshape(-1, a.layers, a.hc, a.dim).astype(np.float64).mean(0), x.size // per
    mh, nh = class_mean(a.acts_harmful); mb, nb = class_mean(a.acts_harmless)                          # [layers, hc, dim]
    print(f"{nh} prompts in the first class, {nb} in the second")
    extra = None
    if a.extra_basis:
        f, k = a.extra_basis.rsplit(":", 1); extra = np.fromfile(f, dtype=np.float32).reshape(int(k), a.dim).astype(np.float64)
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else None             # (a stream the classes do not differ in: the first layers at a template token)
    dstream = lambda L, i: unit(mh[L, i] - mb[L, i])
    dmean = lambda L: unit(mh[L].mean(0) - mb[L].mean(0))
    stack = lambda vs: np.array([v for v in vs if v is not None])
    os.makedirs(a.out, exist_ok=True)
    lines = []
    for rec in a.recipes:
        mask = ""; strength = 1.0; body = rec; steer = ""; alpha = 0.0
        if "+" in body: body, mask = body.split("+", 1)
        name = body.replace(":", "_").replace(",", "_").replace("/", "s").replace("@", "_x").replace("&", "_and_") + ("_mask" if mask else "")
        if "&" in body or body.startswith("steer:"):                       # a steering part: steer:LAYERS@ALPHA
            body, _, sp = body.rpartition("&") if "&" in body else ("none", "", body)
            sl, al = sp.split(":", 1)[1].split("@"); alpha = float(al)
            V = np.zeros((a.layers, a.hc, a.dim), dtype=np.float32)
            for L in layers_of(sl): V[L] = (mb[L] - mh[L]).astype(np.float32)
            steer = os.path.join(a.out, "steer_" + sl.replace(",", "_").replace("/", "s") + ".f32"); V.tofile(steer)
        tail = f"\t{mask}\t{steer}\t{alpha}"
        if "@" in body: body, s = body.split("@", 1); strength = float(s)
        if body == "none": lines.append(f"{name}\tnone\t0\t1.0{tail}"); print(f"{name}: steering only, alpha {alpha}" if steer else name); continue
        kind, *rest = body.split(":")
        if kind == "means": V = stack([dmean(L) for L in layers_of(rest[0])]); keep = None
        elif kind == "streams": V = stack([dstream(L, i) for L in layers_of(rest[0]) for i in range(a.hc)]); keep = None
        elif kind == "top": V = stack([dstream(L, i) for L in layers_of(rest[1]) for i in range(a.hc)]); keep = int(rest[0])
        else: raise SystemExit(f"unknown recipe {rec}")
        _, sv, Vt = np.linalg.svd(V, full_matrices=False)
        k = int((sv > a.rank_tol * sv[0]).sum()) if keep is None else min(keep, len(sv))
        B = Vt[:k]; k_own = k
        if extra is not None:                      # the earlier set first, then what the new one adds to it
            Q, _ = np.linalg.qr(np.concatenate([extra, B], 0).T); B = Q.T; k = len(B)
        B = B.astype(np.float32)
        path = os.path.join(a.out, name + ".f32"); B.tofile(path)
        lines.append(f"{name}\t{path}\t{k}\t{strength}{tail}")
        print(f"{name}: {len(V)} directions -> rank {k_own} (singular values {sv[0]:.2f} .. {sv[k_own - 1]:.3f}){f', {k} with the earlier set' if extra is not None else ''}, strength {strength}" + (f", mask {mask}" if mask else ""))
    with open(os.path.join(a.out, "list.tsv"), "w") as f: f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} candidates -> {os.path.join(a.out, 'list.tsv')}")

if __name__ == "__main__": main()
