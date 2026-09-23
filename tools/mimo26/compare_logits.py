#!/usr/bin/env python3
"""P2 gate clause (b) (docs/mimo26/00_PORT_PLAN.md): per-position comparison of two MLOG logit files.

  compare_logits.py <oracle.bin> <engine.bin>

MLOG: "MLOG", n_seq (u32), vocab (u32), then per sequence n (u32), ids (n i32), logits (n x vocab f32)
(tools/mimo26/oracle_logits.cpp and ie-mimo26-score). For every position of every sequence: top-1 agreement,
KL(oracle || engine) over the full vocab in float64, cosine of the logit vectors, and max |dlogit|. At every top-1
disagreement it prints both sides' top-2 margins, so a near-tie is visible as one.
"""
import struct
import sys

import numpy as np


def read(path):
    out = []
    with open(path, "rb") as f:
        assert f.read(4) == b"MLOG"
        n_seq, V = struct.unpack("<II", f.read(8))
        for _ in range(n_seq):
            (n,) = struct.unpack("<I", f.read(4))
            ids = np.frombuffer(f.read(4 * n), dtype="<i4")
            lg = np.frombuffer(f.read(4 * n * V), dtype="<f4").reshape(n, V)
            out.append((ids, lg))
    return V, out


def lsm(x):
    x = x.astype(np.float64)
    return x - (x.max() + np.log(np.exp(x - x.max()).sum()))


def main():
    Vo, orc = read(sys.argv[1])
    Ve, eng = read(sys.argv[2])
    assert Vo == Ve and len(orc) == len(eng), (Vo, Ve, len(orc), len(eng))
    all_top1 = all_n = 0
    kls, coss = [], []
    for si, ((io, lo), (ie_, le)) in enumerate(zip(orc, eng)):
        assert np.array_equal(io, ie_), f"sequence {si}: token ids differ"
        top1 = 0
        skl = []
        for p in range(len(io)):
            a, b = lo[p], le[p]
            la, lb = lsm(a), lsm(b)
            kl = float(np.sum(np.exp(la) * (la - lb)))
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            ta, tb = int(np.argmax(a)), int(np.argmax(b))
            if ta == tb:
                top1 += 1
            else:
                sa = np.sort(a)[-2:]; sb = np.sort(b)[-2:]
                print(f"  seq {si} pos {p}: top-1 oracle {ta} engine {tb} | oracle margin {sa[1] - sa[0]:.4f} (engine's pick at "
                      f"{a[tb] - a[ta]:+.4f}) | engine margin {sb[1] - sb[0]:.4f} (oracle's pick at {b[ta] - b[tb]:+.4f})")
            skl.append(kl); coss.append(cos)
        kls += skl
        dl = float(np.abs(lo.astype(np.float64) - le.astype(np.float64)).max())
        print(f"seq {si}: {len(io)} positions, top-1 {top1}/{len(io)}, mean KL {np.mean(skl):.5f} nats (max {np.max(skl):.5f}), "
              f"min cos {min(coss[-len(io):]):.5f}, max |dlogit| {dl:.3f}")
        all_top1 += top1; all_n += len(io)
    print(f"OVERALL: top-1 {all_top1}/{all_n} ({100 * all_top1 / all_n:.2f}%), mean KL {np.mean(kls):.5f} nats, "
          f"max KL {np.max(kls):.5f}, min cos {np.min(coss):.5f}")


if __name__ == "__main__":
    main()
