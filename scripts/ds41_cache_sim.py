#!/usr/bin/env python3
"""What would another VRAM residency policy miss on REAL decode traffic of the shared expert tier (MiMo-V2.6, DeepSeek-V4.1)?

Input: the files IE_DS41_PICK_TRACE=<prefix> writes (one per card): uint32 = (layer << 16) | (resident << 14) | (tier << 12) | expert,
each step's layer opened by 0xFFFF0000 | rows. Every layer has S VRAM slots (today: a fixed set from the ranking file + 8 FIFO slots).
Replays each layer's accesses through the policies of q4e_cache_sim.py at the same S, plus:

  today        what the engine did (static tier + the stream slots), from the trace's own flags
  best-static  the best fixed set in hindsight (this trace's own most-used experts) + the FIFO slots: what a perfect ranking file gives
  belady       the optimum with knowledge of the future

  python ds41_cache_sim.py <prefix> [--stream 8] [--extra 0,8,16]
"""
import argparse, glob, os, sys
from collections import Counter, defaultdict, deque
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q4e_cache_sim as Q

def fifo_static(seq, static, F):
    q = deque(); inq = set(); m = 0
    for e in seq:
        if e in static or e in inq: continue
        m += 1
        if F:
            if len(q) >= F: inq.discard(q.popleft())
            q.append(e); inq.add(e)
    return m

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--stream", type=int, default=8)
    ap.add_argument("--extra", default="0"); ap.add_argument("--split", type=int, default=0, help="report the trace in N equal parts too")
    a = ap.parse_args()
    files = sorted(glob.glob(a.prefix + ".L*.u32"))
    if not files: sys.exit("no trace files")
    pol = [("lru", Q.lru), ("slru", Q.slru), ("lfu-age", lambda s, S: Q.lfu_age(s, S, age=2000)), ("arc", Q.arc), ("belady", Q.belady)]
    for f in files:
        x = np.fromfile(f, dtype=np.uint32)
        seqs = defaultdict(list); today = defaultdict(int); nstat = {}; statics = defaultdict(set); rows = []
        for v in x:
            v = int(v)
            if v >> 16 == 0xFFFF: rows.append(v & 0xFFFF); continue
            L, e, tier, res = v >> 16, v & 0xFFF, (v >> 12) & 3, (v >> 14) & 1
            seqs[L].append(e)
            if tier == 0: statics[L].add(e)
            if not (tier == 0 or res): today[L] += 1
        layers = sorted(seqs); n = sum(len(seqs[L]) for L in layers)
        ns = max(len(statics[L]) for L in layers)   # (a lower bound of the static tier: the static experts the trace touched)
        print(f"{os.path.basename(f)}: layers {layers[0]}..{layers[-1]}, {len(rows) // len(layers)} steps, {n} expert uses "
              f"({n / max(1, len(rows)):.1f} a layer-step, {np.mean(rows):.1f} rows), static experts seen a layer: up to {ns}")
        t = sum(today.values()); print(f"  today (static + stream slots, as run)        misses {t:7d} = {100 * t / n:5.1f} %")
        for ex in [int(v) for v in a.extra.split(",")]:
            S = ns + a.stream + ex; res = {}
            bs = 0
            for L in layers:
                top = set(e for e, _ in Counter(seqs[L]).most_common(S - a.stream)); bs += fifo_static(seqs[L], top, a.stream)
            res["best-static"] = bs
            for name, fn in pol: res[name] = sum(fn(seqs[L], S) for L in layers)
            print(f"  S = {S} slots a layer (+{ex}):  " + "  ".join(f"{k} {100 * v / n:.1f} %" for k, v in res.items()))
        if a.split > 1:
            S = ns + a.stream
            for name, fn in [("lru", Q.lru), ("lfu-age", lambda s, S, flags=False: Q.lfu_age(s, S, flags, age=2000)), ("belady", Q.belady)]:
                parts = np.zeros(a.split); tot = np.zeros(a.split)
                for L in layers:
                    F = np.array(fn(seqs[L], S, True), dtype=float); idx = (np.arange(len(F)) * a.split // len(F))
                    for p in range(a.split): parts[p] += F[idx == p].sum(); tot[p] += (idx == p).sum()
                print(f"  {name:8s} by part: " + " ".join(f"{100 * parts[p] / tot[p]:.1f}" for p in range(a.split)))
            parts = np.zeros(a.split); tot = np.zeros(a.split)
            for L in layers:
                pass
if __name__ == "__main__": main()
