import random, time
from solution import RollbackDSU

d = RollbackDSU(0); assert d.count() == 0
try: RollbackDSU(-1); raise AssertionError("negative n")
except ValueError: pass
d = RollbackDSU(5)
for bad in (-1, 5, 99):
    try: d.find(bad); raise AssertionError(f"find({bad})")
    except IndexError: pass
t0 = d.snapshot()
assert d.union(0, 1) and not d.union(1, 0) and d.connected(0, 1) and d.size(1) == 2 and d.count() == 4
t1 = d.snapshot(); d.union(2, 3); d.union(1, 3)
assert d.size(2) == 4 and d.count() == 2
d.rollback(t1); assert not d.connected(0, 2) and d.size(0) == 2 and d.size(3) == 1 and d.count() == 4
d.rollback(t1)  # no-op
d.union(3, 4); t2 = d.snapshot(); d.rollback(t0)
assert d.count() == 5 and all(d.size(i) == 1 for i in range(5))
# a token from a discarded branch stays invalid even at the same depth
d2 = RollbackDSU(4); s0 = d2.snapshot(); d2.union(0, 1); s1 = d2.snapshot(); d2.rollback(s0); d2.union(1, 2)
try: d2.rollback(s1); raise AssertionError("discarded-branch token accepted")
except ValueError: pass
assert d2.connected(1, 2) and not d2.connected(0, 1)
for bad in (t2, 99, -1):
    try: d.rollback(bad); raise AssertionError(f"rollback({bad!r}) accepted")
    except ValueError: pass
# randomized vs recompute-from-scratch model
rng = random.Random(4)
for trial in range(30):
    n = rng.randrange(1, 25); d = RollbackDSU(n); edges = []; snaps = []
    for _ in range(200):
        r = rng.random()
        if r < 0.55:
            a, b = rng.randrange(n), rng.randrange(n); d.union(a, b); edges.append((a, b))
        elif r < 0.7:
            snaps.append((d.snapshot(), len(edges)))
        elif r < 0.85 and snaps:
            tok, ne = snaps.pop(rng.randrange(len(snaps))); d.rollback(tok); del edges[ne:]
            snaps = [s for s in snaps if s[1] <= ne]
        comp = list(range(n))
        def f(x):
            while comp[x] != x: x = comp[x]
            return x
        for a, b in edges:
            ra, rb = f(a), f(b)
            if ra != rb: comp[ra] = rb
        groups = {}
        for i in range(n): groups.setdefault(f(i), []).append(i)
        assert d.count() == len(groups)
        for g in groups.values():
            assert all(d.size(i) == len(g) and d.connected(i, g[0]) for i in g)
# adversarial speed: always union a growing set with a singleton (bad union order without balancing)
n = 200_000; d = RollbackDSU(n); t = time.perf_counter()
for i in range(1, n): d.union(i, 0)
for i in range(n): d.find(i)
dt = time.perf_counter() - t
assert dt < 6.0, f"{dt:.2f} s"
print("OK")
