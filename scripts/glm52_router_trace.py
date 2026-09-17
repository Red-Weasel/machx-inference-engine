#!/usr/bin/env python3
"""Analyse a GLM-5.2 router trace (Phase 9).

Input: the file written by `IE_GLM52_TRACE=<path> ie-glm52-run`, one line per
(token, block):  `<pos> <block> <e0> <e1> ... <e7>`

Answers the questions that decide cache sizing and whether predictive prefetch is
worth building:
  * what fraction of requests the hottest N% of experts cover
  * reuse distance, i.e. how long until an expert is wanted again
  * the ACTUAL LRU hit-rate curve vs cache size — which is the number every
    tok/s projection so far has been substituting a synthetic zipf for
  * per-block skew, since a flat block wants a different policy than a peaky one

usage: glm52_router_trace.py <trace> [--experts 256] [--blocks 79]
"""
import argparse
import collections
import sys


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            rows.append((int(p[0]), int(p[1]), [int(x) for x in p[2:]]))
    return rows


def lru_hit_rate(keys, capacity):
    """Exact LRU hit rate over the request stream. O(n) with a dict as the
    recency order (py3.7+ dicts are insertion-ordered; move_to_end via
    OrderedDict keeps it honest)."""
    od = collections.OrderedDict()
    hits = 0
    for k in keys:
        if k in od:
            hits += 1
            od.move_to_end(k)
        else:
            od[k] = 1
            if len(od) > capacity:
                od.popitem(last=False)
    return hits / len(keys) if keys else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--expert-mib", type=float, default=9.60,
                    help="bytes of one expert's gate+up+down")
    args = ap.parse_args()

    rows = load(args.trace)
    if not rows:
        sys.exit("empty trace")
    tokens = len(set(r[0] for r in rows))
    blocks = sorted(set(r[1] for r in rows))
    topk = len(rows[0][2])
    print(f"trace: {len(rows)} (token,block) records | {tokens} tokens | "
          f"{len(blocks)} MoE blocks | top-{topk}")

    # request stream, keyed by (block, expert) — a given expert id in different
    # blocks is a DIFFERENT weight tensor and must never share a cache entry.
    keys = [(b, e) for _, b, es in rows for e in es]
    total = len(keys)
    freq = collections.Counter(keys)
    print(f"requests: {total} | distinct (block,expert): {len(freq)} "
          f"of {len(blocks)*args.experts} possible")

    print("\n=== hottest-N% coverage ===")
    ordered = [c for _, c in freq.most_common()]
    universe = len(blocks) * args.experts
    cum = 0
    marks = [0.05, 0.10, 0.20, 0.30, 0.50]
    mi = 0
    for i, c in enumerate(ordered, 1):
        cum += c
        while mi < len(marks) and i >= marks[mi] * universe:
            print(f"  top {marks[mi]*100:4.0f}% of experts -> {cum/total*100:5.1f}% of requests")
            mi += 1
    while mi < len(marks):
        print(f"  top {marks[mi]*100:4.0f}% of experts -> {cum/total*100:5.1f}% of requests "
              f"(only {len(ordered)} distinct seen)")
        mi += 1

    print("\n=== reuse distance (requests between consecutive uses) ===")
    last = {}
    dists = []
    for i, k in enumerate(keys):
        if k in last:
            dists.append(i - last[k])
        last[k] = i
    if dists:
        dists.sort()
        for p in (50, 90, 99):
            print(f"  p{p}: {dists[int(len(dists)*p/100)-1]:8d} requests")
        print(f"  mean: {sum(dists)/len(dists):.0f}   "
              f"({len(dists)}/{total} requests were repeats)")

    print("\n=== LRU hit rate vs cache size (THE number) ===")
    print(f"  {'slots':>7}  {'GiB':>7}  {'hit%':>6}  {'GiB/token H2D':>14}")
    per_tok_full = topk * len(blocks) * args.expert_mib / 1024.0
    for cap in (256, 512, 1024, 1550, 2048, 3072, 4096, 6144, 8192):
        if cap > len(freq) * 1.5:
            break
        hr = lru_hit_rate(keys, cap)
        print(f"  {cap:7d}  {cap*args.expert_mib/1024:7.2f}  {hr*100:6.1f}  "
              f"{per_tok_full*(1-hr):14.2f}")
    print(f"\n  (a full token touches {per_tok_full:.2f} GiB of routed experts)")

    print("\n=== per-block skew: share of that block's requests taken by its top 32 experts ===")
    per_block = collections.defaultdict(collections.Counter)
    for _, b, es in rows:
        for e in es:
            per_block[b][e] += 1
    shares = []
    for b in blocks:
        c = per_block[b]
        tot = sum(c.values())
        top = sum(v for _, v in c.most_common(32))
        shares.append((top / tot, b))
    shares.sort()
    print(f"  most uniform block  {shares[0][1]:3d}: {shares[0][0]*100:5.1f}%")
    print(f"  median block        {shares[len(shares)//2][1]:3d}: "
          f"{shares[len(shares)//2][0]*100:5.1f}%")
    print(f"  most skewed block   {shares[-1][1]:3d}: {shares[-1][0]*100:5.1f}%")


if __name__ == "__main__":
    main()
