#!/usr/bin/env python3
"""Compare two ie-ds41-lanes-test --dump files (2026-10-20: a 2-card run against a 4-card one).

Every trace -- the solo sequences and the slot-arm ones -- must match byte for byte: the same tokens, the same logits FNV after
the prompt and after every decode step, the same final position. Prints one line per trace and exits 1 on any difference.
  usage: ds41_lanes_dump_compare.py A.dump B.dump
"""
import sys


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            w = line.split()
            if not w:
                continue
            i_pos, i_tok, i_lg = w.index("n_pos"), w.index("tokens"), w.index("logits")
            name = " ".join(w[:i_pos])
            out[name] = (int(w[i_pos + 1]), w[i_tok + 1:i_lg], w[i_lg + 1:])
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1].strip())
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])
    bad = 0
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            print(f"[FAIL] {name}: only in {'A' if name in a else 'B'}")
            bad += 1
            continue
        (pa, ta, la), (pb, tb, lb) = a[name], b[name]
        if (pa, ta, la) == (pb, tb, lb):
            print(f"[ ok ] {name}: identical ({len(ta)} tokens, {len(la)} logits rows, n_pos {pa})")
            continue
        bad += 1
        step = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
        tok = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        print(f"[FAIL] {name}: logits differ first at step {step}, tokens at {tok}, n_pos {pa} vs {pb}")
    print(f"\n{len(set(a) | set(b)) - bad} of {len(set(a) | set(b))} traces identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
