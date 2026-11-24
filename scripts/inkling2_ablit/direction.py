#!/usr/bin/env python3
"""Inkling 2 Mini abliteration, step 2: where harmful and harmless prompts separate, and candidate direction sets.

  python direction.py table --harmful H.f32 --harmless B.f32 [--pos 1] [--out table.json]
  python direction.py make  --harmful H.f32 --harmless B.f32 --out DIR RECIPE [RECIPE ...]

H / B are ie-inkling2-run --batch --acts-out recordings: per prompt [P positions][48 calls][4 kinds][2304] float32
(--acts-last P; kinds: 0 partial state after the call, 1 attention write, 2 MoE write, 3 MoE read-out u). The default
recording (--open-turn --acts-last 2) holds position 0 = the prompt's last token (<|end_message|> of the user turn) and
position 1 = <|message_model|>, the token the model opens its own turn with (the "first generated position").

table: per call, for each kind, the difference-of-means direction's d' and AUROC (computed on a half of the prompts the
direction was NOT computed from: train on even prompts, score odd ones) and how the call's direction agrees with the
neighbours'.

make: RECIPE = name=SPEC[@STRENGTH][/WHERE]. SPEC:
  call:K:C          the direction of kind K at call C (position --pos)
  calls:K:A-B       every call A..B's direction of kind K, as one orthonormal set (rank by --rank-tol)
  top:N:K:A-B       the N leading principal directions of calls A..B's directions
  mean:K:A-B        one direction: the normalised mean of calls A..B's unit directions
WHERE: all (default: embedding + every attention and MoE write) | attn | moe | calls:A-B (attention + MoE writes of
those calls only). Writes DIR/<name>.json + DIR/<name>.f32 (ie-ablation-v1) and appends "name<TAB>path" to DIR/list.tsv.
"""
import argparse, json, os
import numpy as np

H, NC, NK = 2304, 48, 4
KIND = ["partial", "attn_w", "moe_w", "moe_in"]


def load(path, P):
    x = np.memmap(path, dtype=np.float32, mode="r")
    per = P * NC * NK * H
    assert x.size % per == 0, f"{path}: not whole [{P}, 48, 4, 2304] records"
    return x.reshape(-1, P, NC, NK, H)


def sep(xh, xb):
    """d' and AUROC of the mean-difference direction fitted on even rows, scored on odd rows."""
    u = xh[0::2].mean(0) - xb[0::2].mean(0)
    n = np.linalg.norm(u)
    if n < 1e-12: return 0.0, 0.5, u
    u /= n
    ph, pb = xh[1::2] @ u, xb[1::2] @ u
    d = (ph.mean() - pb.mean()) / np.sqrt(0.5 * (ph.var(ddof=1) + pb.var(ddof=1)) + 1e-30)
    au = (ph[:, None] > pb[None, :]).mean()
    return float(d), float(au), u


def unit_dirs(Xh, Xb, pos, kind):
    """[48, H] unit mean-difference directions of all prompts at (pos, kind)."""
    out = np.zeros((NC, H))
    for c in range(NC):
        v = Xh[:, pos, c, kind].astype(np.float64).mean(0) - Xb[:, pos, c, kind].astype(np.float64).mean(0)
        out[c] = v / max(np.linalg.norm(v), 1e-30)
    return out


def calls_of(spec):
    out = []
    for part in spec.split(","):
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return out


def schedule():
    s = [(L, 1) for L in range(6)] + [(L, p) for p in (1, 2, 3) for L in range(6, 18)] + [(L, 1) for L in range(18, 24)]
    return s


def cmd_table(a):
    Xh, Xb = load(a.harmful, a.P), load(a.harmless, a.P)
    print(f"{Xh.shape[0]} harmful, {Xb.shape[0]} harmless prompts; position {a.pos}; d' / AUROC on held-out halves")
    sch = schedule()
    rows = []
    prev = None
    print(f"{'call':>4} {'layer':>5} {'pass':>4} | " + " | ".join(f"{k:>16s}" for k in KIND) + " | cos(prev) partial | cos(pos0,pos1) partial")
    for c in range(NC):
        row = {"call": c, "layer": sch[c][0], "pass": sch[c][1]}
        cells = []
        for k in range(NK):
            d, au, u = sep(Xh[:, a.pos, c, k].astype(np.float64), Xb[:, a.pos, c, k].astype(np.float64))
            row[KIND[k]] = {"dprime": d, "auroc": au}
            cells.append(f"{d:7.2f} {au:8.4f}")
            if k == 0: up = u
        cp = float(up @ prev) if prev is not None else float("nan")
        prev = up
        cpp = float("nan")
        if a.P > 1:
            u0 = Xh[:, 0, c, 0].astype(np.float64).mean(0) - Xb[:, 0, c, 0].astype(np.float64).mean(0)
            u1 = Xh[:, 1, c, 0].astype(np.float64).mean(0) - Xb[:, 1, c, 0].astype(np.float64).mean(0)
            cpp = float(u0 @ u1 / (np.linalg.norm(u0) * np.linalg.norm(u1)))
        row["cos_prev"] = cp; row["cos_pos01"] = cpp
        rows.append(row)
        print(f"{c:>4} {sch[c][0]:>5} {sch[c][1]:>4} | " + " | ".join(cells) + f" | {cp:17.3f} | {cpp:21.3f}")
    if a.out: json.dump(rows, open(a.out, "w"), indent=1)


def cmd_make(a):
    Xh, Xb = load(a.harmful, a.P), load(a.harmless, a.P)
    os.makedirs(a.out, exist_ok=True)
    cache = {}
    def D(kind):
        if kind not in cache: cache[kind] = unit_dirs(Xh, Xb, a.pos, kind)
        return cache[kind]
    lines = []
    for rec in a.recipes:
        name, spec = rec.split("=", 1)
        where = "all"
        if "/" in spec: spec, where = spec.split("/", 1)
        strength = 1.0
        if "@" in spec: spec, s = spec.split("@"); strength = float(s)
        f = spec.split(":")
        if f[0] == "local":                        # each call's own direction on that call's two writes only
            cs = calls_of(f[2]); B = D(int(f[1]))[cs]
            B.astype(np.float32).tofile(os.path.join(a.out, name + ".f32"))
            groups = [{"rows": [i, 1], "strength": strength, "attn": [c], "moe": [c]} for i, c in enumerate(cs)]
            meta = {"format": "ie-ablation-v1", "arch": "inkling2_loop", "hidden": H, "file": name + ".f32", "rows": len(cs),
                    "groups": groups, "recipe": rec, "position": a.pos, "kinds": KIND}
            p = os.path.abspath(os.path.join(a.out, name + ".json"))
            json.dump(meta, open(p, "w"), indent=1)
            lines.append(f"{name}\t{p}")
            print(f"{name}: {len(cs)} per-call direction(s), strength {strength}")
            continue
        if f[0] == "call": V = D(int(f[1]))[[int(f[2])]]
        elif f[0] == "calls": V = D(int(f[1]))[calls_of(f[2])]
        elif f[0] == "top": V = D(int(f[2]))[calls_of(f[3])]
        elif f[0] == "mean":
            v = D(int(f[1]))[calls_of(f[2])].mean(0); V = (v / np.linalg.norm(v))[None]
        else: raise SystemExit(f"unknown recipe {rec}")
        _, sv, Vt = np.linalg.svd(V, full_matrices=False)
        if f[0] == "top": k = min(int(f[1]), len(sv))
        else: k = int((sv > a.rank_tol * sv[0]).sum())
        B = Vt[:k]
        Q, _ = np.linalg.qr(B.T); B = Q.T          # orthonormal to float64 precision
        B.astype(np.float32).tofile(os.path.join(a.out, name + ".f32"))
        g = {"rows": [0, k], "strength": strength}
        if where == "all": g.update(embed=True, attn="all", moe="all")
        elif where == "attn": g.update(attn="all")
        elif where == "moe": g.update(moe="all")
        elif where.startswith("calls:"):
            cs = calls_of(where[6:]); g.update(attn=cs, moe=cs)
        else: raise SystemExit(f"unknown WHERE {where}")
        meta = {"format": "ie-ablation-v1", "arch": "inkling2_loop", "hidden": H, "file": name + ".f32", "rows": k, "groups": [g],
                "recipe": rec, "position": a.pos, "kinds": KIND,
                "source": {"harmful": os.path.abspath(a.harmful), "harmless": os.path.abspath(a.harmless),
                           "n_harmful": int(Xh.shape[0]), "n_harmless": int(Xb.shape[0])}}
        p = os.path.abspath(os.path.join(a.out, name + ".json"))
        json.dump(meta, open(p, "w"), indent=1)
        lines.append(f"{name}\t{p}")
        print(f"{name}: {len(V)} direction(s) -> rank {k} (singular values {sv[0]:.3f} .. {sv[k - 1]:.3f}), strength {strength}, where {where}")
    with open(os.path.join(a.out, "list.tsv"), "a") as fo: fo.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "make"])
    ap.add_argument("--harmful", required=True); ap.add_argument("--harmless", required=True)
    ap.add_argument("--P", type=int, default=2); ap.add_argument("--pos", type=int, default=1)
    ap.add_argument("--out"); ap.add_argument("--rank-tol", type=float, default=0.03)
    ap.add_argument("recipes", nargs="*")
    a = ap.parse_intermixed_args()
    cmd_table(a) if a.cmd == "table" else cmd_make(a)


if __name__ == "__main__": main()
