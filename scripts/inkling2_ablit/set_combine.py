#!/usr/bin/env python3
"""Combine Inkling 2 ablation / steering sets into one ie-ablation-v1 file (rows concatenated, group rows shifted).

  python set_combine.py --out OUT.json A.json B.json [...]   (writes OUT.json + OUT.f32; "file" is OUT's basename)

Each input keeps its groups unchanged (removal or "mode": "add"); the engine allows one removal and one add group per
write, so combining a removal set with a steering set is valid; two sets of the same mode on one write are refused.
"""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("sets", nargs="+")
    ap.add_argument("--name", default=""); a = ap.parse_args()
    rows, groups, parts = [], [], []
    seen = {}
    for p in a.sets:
        j = json.load(open(p))
        d = np.fromfile(os.path.join(os.path.dirname(os.path.abspath(p)), j["file"]), dtype=np.float32).reshape(-1, j["hidden"])
        off = sum(len(r) for r in rows); rows.append(d)
        for g in j["groups"]:
            g = dict(g); g["rows"] = [g["rows"][0] + off, g["rows"][1]]
            mode = g.get("mode", "remove")
            for key in ("embed", "attn", "moe"):
                if key not in g or g[key] is False: continue
                calls = ["e"] if key == "embed" else (list(range(48)) if g[key] == "all" else g[key])
                for c in calls:
                    k = (mode, key, c)
                    if k in seen: raise SystemExit(f"{p}: second {mode} group on {key} {c} (first from {seen[k]})")
                    seen[k] = p
            groups.append(g)
        parts.append({"from": os.path.abspath(p), "name": j.get("name", ""), "version": j.get("version", "")})
    out = os.path.abspath(a.out); base = os.path.splitext(out)[0]
    np.concatenate(rows).astype(np.float32).tofile(base + ".f32")
    j = {"format": "ie-ablation-v1", "arch": "inkling2_loop", "hidden": int(rows[0].shape[1]), "file": os.path.basename(base) + ".f32",
         "rows": int(sum(len(r) for r in rows)), "groups": groups, "name": a.name, "combined_from": parts}
    json.dump(j, open(out, "w"), indent=1)
    print(f"{out}: {len(groups)} groups, {j['rows']} rows")


if __name__ == "__main__": main()
