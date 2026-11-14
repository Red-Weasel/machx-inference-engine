#!/usr/bin/env python3
"""Read the activations IE_DS41_ACT_DUMP=<dir> records (the residual stream entering each layer, mean over the hc copies).

  python ds41_acts.py <dir>                 summary: lanes, positions, layers, bytes
  python ds41_acts.py <dir> --npz out.npz   one array [positions, layers, dim] per lane (float32; NaN where a layer is missing),
                                            with the positions and the token ids

As a module: load(dir) -> {lane: (positions[int], ids[int], acts[P, L, dim])}. A position recorded more than once
(a rollback, a re-read prompt) keeps the LAST record.
"""
import glob, struct, sys
import numpy as np

MAGIC = 0x31544341

def records(path):
    with open(path, "rb") as f:
        while True:
            h = f.read(24)
            if len(h) < 24: return
            magic, lane, layer, pos, rows, dim = struct.unpack("<6I", h)
            if magic != MAGIC: raise ValueError(f"{path}: bad record at byte {f.tell() - 24}")
            ids = np.frombuffer(f.read(4 * rows), dtype=np.int32)
            x = np.frombuffer(f.read(4 * rows * dim), dtype=np.float32)
            if x.size != rows * dim: return                      # a record cut short (the server was still writing)
            yield lane, layer, pos, ids, x.reshape(rows, dim)

def load(d):
    by = {}
    for path in sorted(glob.glob(d.rstrip("/") + "/acts.card*.bin")):
        for lane, layer, pos, ids, x in records(path):
            for r in range(x.shape[0]):
                e = by.setdefault(lane, {}).setdefault(pos + r, [int(ids[r]), {}, layer])
                e[1][layer] = x[r]                              # the later record of a (position, layer) wins
                if layer <= e[2]: e[0], e[2] = int(ids[r]), layer   # the token id: the lowest layer's last record (the files are not interleaved in time)
    out = {}
    for lane, m in by.items():
        P = sorted(m); layers = sorted({L for p in P for L in m[p][1]}); dim = next(iter(m[P[0]][1].values())).size
        a = np.full((len(P), max(layers) + 1, dim), np.nan, dtype=np.float32)
        for i, p in enumerate(P):
            for L, v in m[p][1].items(): a[i, L] = v
        out[lane] = (np.array(P), np.array([m[p][0] for p in P]), a)
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    data = load(sys.argv[1])
    if not data: sys.exit("no records")
    for lane, (P, ids, a) in sorted(data.items()):
        full = int(np.sum(~np.isnan(a[:, :, 0]).any(axis=1)))
        print(f"lane {lane}: {len(P)} positions ({P[0]}..{P[-1]}), {a.shape[1]} layers x {a.shape[2]} dims, {full} with every layer, "
              f"{a.nbytes / 2**20:.0f} MiB; mean |x| at layers 0 / mid / last: "
              + " / ".join(f"{np.nanmean(np.abs(a[:, L])):.3f}" for L in (0, a.shape[1] // 2, a.shape[1] - 1)))
    if "--npz" in sys.argv:
        o = sys.argv[sys.argv.index("--npz") + 1]
        np.savez_compressed(o, **{f"lane{l}_{k}": v for l, t in data.items() for k, v in zip(("pos", "ids", "acts"), t)}); print("wrote", o)
