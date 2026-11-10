#!/usr/bin/env python3
"""What would another expert-cache policy, or another slot count, miss on REAL traffic?

Input: the files IE_Q4E_PICK_TRACE=<prefix> writes (uint32 = (layer << 16) | expert, every host-routed expert access in
order; run the server with IE_Q4E_DEV_ROUTE=0). Each layer has its own cache of S slots; the engine evicts the least
recently used (LRU). This replays every layer's accesses through:

  lru      the engine's policy today
  slru     segmented LRU: a new expert enters a probation segment (20 % of the slots); a second use promotes it to the
           protected segment, so one-off experts cannot push out the regulars
  slru-stamp  the same idea with only a last-use stamp and one flag a slot (what the engine's cache already has room for)
  lfu-age  least frequently used, counts halved every `age` accesses of the layer
  arc      Megiddo & Modha's adaptive replacement cache (recency and frequency lists, sized by what the ghosts say)
  belady   the optimum with full knowledge of the future -- no real policy can beat it; the room there is to gain

  python q4e_cache_sim.py <prefix> [--slots 155,221] [--layers-step 1] [--max-accesses 4000000]
"""
import argparse, glob, heapq, sys
from collections import OrderedDict, defaultdict
import numpy as np

def lru(seq, S, flags=False):
    c = OrderedDict(); m = 0; F = []
    for e in seq:
        hit = e in c
        if hit: c.move_to_end(e)
        else:
            m += 1
            if len(c) >= S: c.popitem(last=False)
            c[e] = 1
        if flags: F.append(not hit)
    return F if flags else m

def slru(seq, S, flags=False, prob_frac=0.2):
    P = max(1, int(S * prob_frac)); Q = S - P
    prob, prot = OrderedDict(), OrderedDict(); m = 0; F = []
    for e in seq:
        if flags: F.append(e not in prot and e not in prob)
        if e in prot: prot.move_to_end(e)
        elif e in prob:
            del prob[e]; prot[e] = 1
            if len(prot) > Q:
                d, _ = prot.popitem(last=False); prob[d] = 1
                if len(prob) > P: prob.popitem(last=False)
        else:
            m += 1
            if len(prob) >= P: prob.popitem(last=False)
            prob[e] = 1
    return F if flags else m

def slru_stamp(seq, S, flags=False, prob_frac=0.2):
    """SLRU as the engine could keep it with what it already has: a last-use stamp a slot plus one flag (protected or not).
    A hit on a probation slot protects it; above S - P protected slots the least recent protected one loses the flag (its
    stamp stays, so it is the next probation victim unless used again); a miss evicts the least recent probation slot."""
    P = max(1, int(S * prob_frac)); Q = S - P
    last, prot = {}, set(); m = 0; t = 0; F = []
    for e in seq:
        t += 1
        if flags: F.append(e not in last)
        if e in last:
            last[e] = t
            if e not in prot:
                prot.add(e)
                if len(prot) > Q:
                    d = min(prot, key=last.__getitem__); prot.discard(d)
        else:
            m += 1
            if len(last) >= S:
                cold = [k for k in last if k not in prot]
                v = min(cold if cold else last, key=last.__getitem__)
                del last[v]; prot.discard(v)
            last[e] = t
    return F if flags else m

def lfu_age(seq, S, flags=False, age=20000):
    cnt = defaultdict(float); c = OrderedDict(); m = 0; n = 0; F = []
    for e in seq:
        n += 1
        if flags: F.append(e not in c)
        if n % age == 0:
            for k in list(cnt): cnt[k] *= 0.5
        cnt[e] += 1
        if e in c: c.move_to_end(e)
        else:
            m += 1
            if len(c) >= S:
                v = min(c, key=lambda k: cnt[k])      # (ties: the oldest, the OrderedDict's order)
                del c[v]
            c[e] = 1
    return F if flags else m

def arc(seq, S, flags=False):
    t1, t2, b1, b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(); p = 0; m = 0; F = []
    def replace(e):
        nonlocal p
        if t1 and (len(t1) > p or (e in b2 and len(t1) == p)):
            k, _ = t1.popitem(last=False); b1[k] = 1
        else:
            k, _ = t2.popitem(last=False); b2[k] = 1
    for e in seq:
        if flags: F.append(e not in t1 and e not in t2)
        if e in t1: del t1[e]; t2[e] = 1
        elif e in t2: t2.move_to_end(e)
        elif e in b1:
            m += 1; p = min(S, p + max(1, len(b2) // max(1, len(b1)))); replace(e); del b1[e]; t2[e] = 1
        elif e in b2:
            m += 1; p = max(0, p - max(1, len(b1) // max(1, len(b2)))); replace(e); del b2[e]; t2[e] = 1
        else:
            m += 1
            if len(t1) + len(b1) == S:
                if len(t1) < S: b1.popitem(last=False); replace(e)
                else: t1.popitem(last=False)
            elif len(t1) + len(t2) + len(b1) + len(b2) >= S:
                if len(t1) + len(t2) + len(b1) + len(b2) >= 2 * S and b2: b2.popitem(last=False)
                if len(t1) + len(t2) >= S: replace(e)
            t1[e] = 1
    return F if flags else m

def belady(seq, S, flags=False):
    F = []
    n = len(seq); nxt = [0] * n; last = {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(seq[i], n + i); last[seq[i]] = i
    c = {}; h = []; m = 0
    for i, e in enumerate(seq):
        if flags: F.append(e not in c)
        if e in c: c[e] = nxt[i]; heapq.heappush(h, (-nxt[i], e))
        else:
            m += 1
            if len(c) >= S:
                while True:
                    d, v = heapq.heappop(h)
                    if v in c and c[v] == -d: del c[v]; break
            c[e] = nxt[i]; heapq.heappush(h, (-nxt[i], e))
    return F if flags else m

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prefix")
    ap.add_argument("--slots", default="155,221")
    ap.add_argument("--layers-step", type=int, default=1)
    ap.add_argument("--max-accesses", type=int, default=6000000, help="per trace file (the first N records)")
    ap.add_argument("--decode-rows", type=int, default=16, help="a forward of at most this many rows is decode / verify")
    a = ap.parse_args()
    files = sorted(glob.glob(a.prefix + ".layers*.u32"))
    if not files: sys.exit("no trace files at " + a.prefix + ".layers*.u32")
    per, kind = defaultdict(list), defaultdict(list); total = 0; marked = False
    for f in files:
        x = np.fromfile(f, dtype=np.uint32)[: a.max_accesses]
        mark = (x >> 31).astype(bool); lay = ((x >> 16) & 0x7FFF).astype(np.int32); low = (x & 0xFFFF).astype(np.int32)
        marked = marked or bool(mark.any())
        for L in np.unique(lay):
            sel = lay == L; m = mark[sel]; v = low[sel]
            # the rows of the forward each access belongs to: the last marker before it (no markers: 0 = unknown)
            rows = np.where(m, v, 0); idx = np.maximum.accumulate(np.where(m, np.arange(len(v)), -1))
            cur = np.where(idx >= 0, rows[np.maximum(idx, 0)], 0)
            acc = ~m
            per[int(L)] = v[acc].tolist()
            kind[int(L)] = ((cur[acc] > 0) & (cur[acc] <= a.decode_rows)).tolist()   # True = a decode / verify access
            total += int(acc.sum())
    layers = sorted(per)[:: a.layers_step]
    n = sum(len(per[L]) for L in layers); nd = sum(sum(kind[L]) for L in layers)
    print(f"{total} accesses in {len(files)} file(s); simulating {len(layers)} layer(s): {n} accesses, {nd} of them in decode / verify "
          f"forwards ({'markers present' if marked else 'NO markers: the split is unknown'})")
    pol = [("lru (today)", lru), ("slru", slru), ("slru-stamp", slru_stamp), ("lfu-age", lfu_age), ("arc", arc), ("belady (optimum)", belady)]
    for S in [int(s) for s in a.slots.split(",")]:
        print(f"\n{S} slots a layer:                all accesses        decode / verify accesses")
        base = based = None
        for name, fn in pol:
            miss = missd = 0
            for L in layers:
                flags = fn(per[L], S, True)
                miss += sum(flags); missd += sum(1 for f, k in zip(flags, kind[L]) if f and k)
            rate = miss / n; rated = missd / nd if nd else 0.0
            if base is None: base, based = rate, rated
            print(f"  {name:18s} {100 * rate:6.2f} % ({100 * (1 - rate / base):+5.1f} %)     {100 * rated:6.2f} % ({100 * (1 - rated / based) if based else 0:+5.1f} %)")
    print("\n(in brackets: misses fewer than lru; a positive number is better)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
