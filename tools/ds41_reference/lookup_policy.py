#!/usr/bin/env python3
"""Phase 58: search prompt-lookup speculation policies offline, on real request traces (IE_DS41_LOOKUP_DUMP).

A pass drafts up to K tokens by copying what followed the most recent earlier occurrence of the context's last n
tokens, verifies 1 + K rows in one forward, and commits the matching drafts + one token; a pass with no draft is a
plain one-row step. For a greedy stream that is exact given the trace: the committed tokens ARE the trace.
Priced with measured step costs (docs/deepseek41/59): one row 79 ms in the shipping configuration (the CPU miss leg
serves one-row steps only), a T-row verify 131 / 171 / 292 ms at T = 2 / 3 / 6, ~41 ms per extra row.
  usage: lookup_policy.py <trace_dir> [--cost-plain 79] [--cost-row 41] [--cost-2 131]"""
import glob, os, struct, sys
import numpy as np

D = sys.argv[1]
arg = lambda k, d: float(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
PLAIN, ROW2, ROW = arg("--cost-plain", 79.0), arg("--cost-2", 131.0), arg("--cost-row", 41.0)
cost = lambda rows: PLAIN if rows == 1 else ROW2 + ROW * (rows - 2)

traces = []
for f in sorted(glob.glob(f"{D}/req_*.ids")):
    b = open(f, "rb").read(); P, N = struct.unpack("<2I", b[:8])
    ids = np.frombuffer(b[8:], "<i4"); traces.append((os.path.basename(f), ids[:P], ids[P:P + N]))

class Index:
    """the last OCC occurrences of every n-gram (n = 2..NMAX), maintained as the sequence grows; a draft copies from
    the most recent earlier occurrence (best=False) or from the one whose match extends furthest back (best=True)"""
    NMAX, OCC = 6, 16
    def __init__(self, seq, best=False):
        self.best = best; self.seq = []; self.occ = [dict() for _ in range(self.NMAX + 1)]
        for t in seq: self.push(t)
    def push(self, t):
        self.seq.append(int(t)); end = len(self.seq)
        for n in range(2, self.NMAX + 1):
            if end >= n:
                lst = self.occ[n].setdefault(tuple(self.seq[end - n:end]), [])
                lst.append(end)
                if len(lst) > self.OCC: del lst[0]
    def _m(self, e, end, n):
        m = n
        while m < 64 and end - m - 1 >= 0 and e - m - 1 >= 0 and self.seq[end - m - 1] == self.seq[e - m - 1]: m += 1
        return m
    def draft(self, K, nmin):
        """(draft, match length): the longest suffix n in [nmin, NMAX] with an EARLIER occurrence"""
        end = len(self.seq)
        for n in range(self.NMAX, nmin - 1, -1):
            if end < n + 1: continue
            lst = self.occ[n].get(tuple(self.seq[end - n:end]))
            if not lst or len(lst) < 2: continue          # lst[-1] is the suffix itself
            cands = lst[:-1] if self.best else lst[-2:-1]
            e, m = max(((c, self._m(c, end, n)) for c in cands), key=lambda x: (x[1], x[0]))
            return self.seq[e:e + K], m
        return [], 0

def simulate(policy, best=False):
    """policy(state) -> K for this pass; state carries the last pass's outcome"""
    tot_ms = tot_plain = tot_tok = 0.0; per = []
    for name, prompt, out in traces:
        idx = Index(prompt, best); g = 0; N = len(out); ms = 0.0; st = {"last_L": None, "last_K": 0, "fails": 0}
        while g < N:
            K, nmin = policy(st)
            d, m = idx.draft(K, nmin) if K > 0 else ([], 0)
            if d and not st.get("accept_match", lambda m: True)(m): d = []
            L = 0
            while L < len(d) and g + L < N and d[L] == out[g + L]: L += 1
            commit = min(L + 1, N - g)
            ms += cost(1 + len(d))
            st.update(last_L=L if d else None, last_K=len(d), m=m)
            if d: st["fails"] = 0 if L == len(d) else st["fails"] + (L == 0)
            for t in out[g:g + commit]: idx.push(t)
            g += commit
        tot_ms += ms; tot_plain += PLAIN * N; tot_tok += N; per.append((name, N, PLAIN * N / ms))
    return tot_plain / tot_ms, per

def fixed(K, nmin): return lambda st: (K, nmin)
def adaptive(kmax, nmin, grow=2, cool=2):
    def f(st):
        k = st.setdefault("k", 3)
        if st["last_L"] is not None:
            if st["last_L"] == st["last_K"]: k = min(kmax, k + grow)      # the whole draft landed: reach further
            elif st["last_L"] == 0: k = max(1, k // 2)                      # nothing landed: shrink
        st["k"] = k
        if st["fails"] >= cool: st["fails"] -= 1; return 0, nmin          # after repeated misses, sit out one pass
        return k, nmin
    return f
def matchlen(kmax, mmin):
    """draft only when the copy's matched length is at least mmin; draft length scales with it"""
    def f(st):
        st["accept_match"] = lambda m: m >= mmin
        return kmax, 2
    return f

print(f"{len(traces)} requests, {sum(len(o) for _, _, o in traces)} generated tokens; cost: plain {PLAIN} ms, "
      f"2 rows {ROW2} ms, +{ROW} ms per row")
results = []
for best in (False, True):
    for name, pol in [("fixed k3 n2", fixed(3, 2)), ("fixed k7 n3", fixed(7, 3)), ("adaptive k7 n3", adaptive(7, 3)),
                      ("match>=4 k7", matchlen(7, 4)), ("match>=6 k7", matchlen(7, 6)), ("match>=8 k7", matchlen(7, 8)),
                      ("match>=12 k7", matchlen(7, 12)), ("match>=6 k5", matchlen(5, 6)), ("match>=8 k5", matchlen(5, 8))]:
        sp, per = simulate(pol, best); tag = ("longest " if best else "recent  ") + name; results.append((sp, tag, per))
        worst = min(per, key=lambda x: x[2])
        print(f"{tag:24s} overall x{sp:.3f}   worst request x{worst[2]:.2f} ({worst[1]} tok)   "
              f"losing requests {sum(1 for p in per if p[2] < 0.99)} of {len(per)}")
best = max(results)
print(f"\nbest: {best[1]} x{best[0]:.3f}; per request (tokens, speed-up):")
for name, n, s in best[2]: print(f"  {name} {n:5d} x{s:.2f}")
