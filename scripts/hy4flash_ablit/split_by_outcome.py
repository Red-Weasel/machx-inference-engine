#!/usr/bin/env python3
"""Hy4-flash abliteration: split a candidate's recorded activations by what it did with each prompt.

  python split_by_outcome.py --acts A.f32 --gens G.jsonl --out PREFIX [--layers 42] [--hc 4] [--dim 3072]

A.f32 is ie-hy4flash-run --acts-out for ONE candidate over a batch (one [layers, streams, dim] record a prompt, in batch
order) and G.jsonl its --gen-out. Prompts whose answer reads as declined go to PREFIX.declined.f32, clearly answered
ones to PREFIX.complied.f32 (behavior.classify; "unclear" answers are left out). make_candidates.py then finds what
still separates the two -- the part of the refusal the candidate does not remove.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from behavior import classify

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--acts", required=True); ap.add_argument("--gens", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=42); ap.add_argument("--hc", type=int, default=4); ap.add_argument("--dim", type=int, default=3072)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.gens, encoding="utf-8", errors="replace") if l.strip()]
    x = np.memmap(a.acts, dtype=np.float32, mode="r").reshape(-1, a.layers, a.hc, a.dim)
    assert len(x) == len(rows), (len(x), len(rows))
    kind = [classify(r["text"]) for r in rows]
    for k in ("declined", "complied"):
        idx = [i for i, v in enumerate(kind) if v == k]
        np.asarray(x[idx]).tofile(f"{a.out}.{k}.f32")
        print(f"{k}: {len(idx)} prompts -> {a.out}.{k}.f32")
    print(f"unclear (left out): {kind.count('unclear')}")

if __name__ == "__main__": main()
