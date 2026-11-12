import sys, glob, os
from collections import defaultdict
import numpy as np
# admission-limited residency: a layer-step may upload (and so cache) only a share q of its misses; the rest run on the CPU leg
def parse(f):
    x = np.fromfile(f, dtype=np.uint32); steps = defaultdict(list); statics = defaultdict(set); cur = None; today = 0; n = 0
    for v in x:
        v = int(v)
        if v >> 16 == 0xFFFF: cur = None; continue
        L, e, tier, res = v >> 16, v & 0xFFF, (v >> 12) & 3, (v >> 14) & 1
        if cur is None or cur[0] != L: cur = (L, []); steps[L].append(cur[1])
        cur[1].append(e); n += 1
        if tier == 0: statics[L].add(e)
        today += not (tier == 0 or res)
    return steps, statics, today, n
def sim(steps, init, S, q, evict, decay=0.98, fixed=None):
    # fixed: a set that never leaves (today's static tier); the other slots are replaced by `evict` ("lru" | "freq")
    c = {e: 0 for e in init}; t = 0; cnt = defaultdict(float); miss = 0; up = 0
    while len(c) < S and fixed is None: c[-len(c) - 1] = -1            # the static slots the trace never used
    for st in steps:
        t += 1
        for k in list(cnt): cnt[k] *= decay
        for e in st: cnt[e] += 1
        M = [e for e in st if e not in c]
        for e in st:
            if e in c: c[e] = t
        miss += len(M)
        k = int(round(q * len(M)))
        for e in sorted(M, key=lambda e: -cnt[e])[:k]:
            cand = [x for x in c if c[x] != t and (fixed is None or x not in fixed)]
            if len(c) < S: c[e] = t; up += 1; continue
            if not cand: break
            v = min(cand, key=(lambda x: c[x]) if evict == "lru" else (lambda x: (cnt[x], c[x])))
            if evict == "freq" and cnt[v] >= cnt[e]: continue             # the newcomer is no better than the coldest resident
            del c[v]; c[e] = t; up += 1
    return miss, up
