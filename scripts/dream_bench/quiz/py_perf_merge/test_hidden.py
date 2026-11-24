import random, time
from solution import merge_intervals as f

def ref(iv):
    cov = set()
    for s, e in iv: cov.update(range(s, e))
    out = []
    for x in sorted(cov):
        if out and out[-1][1] == x: out[-1] = (out[-1][0], x + 1)
        else: out.append((x, x + 1))
    return out
assert f([(1, 3), (3, 5)]) == [(1, 5)] and f([(1, 4), (2, 3)]) == [(1, 4)] and f([(4, 5), (1, 2)]) == [(1, 2), (4, 5)]
assert f([(-3, -1)]) == [(-3, -1)] and f([(-5, 0), (0, 2)]) == [(-5, 2)]
assert f([]) == [] and f([(3, 3)]) == [] and f([(1, 2), (2, 2), (2, 2)]) == [(1, 2)] and f([(5, 5), (1, 9)]) == [(1, 9)]
assert f(((i, i + 1) for i in range(0, 10, 2))) == [(i, i + 1) for i in range(0, 10, 2)]
inp = [[5, 6], [1, 2]]; f(inp); assert inp == [[5, 6], [1, 2]], "input modified"
assert all(type(x) is tuple for x in f([[1, 2], [0, 1]]))
try: f([(3, 1)]); raise AssertionError("start > end accepted")
except ValueError: pass
rng = random.Random(11)
for _ in range(2000):
    iv = []
    for _ in range(rng.randrange(0, 10)):
        s = rng.randrange(-30, 30); iv.append((s, s + rng.randrange(0, 6)))
    assert f(iv) == ref(iv), iv
big = [(s, s + rng.randrange(0, 50)) for s in (rng.randrange(0, 10**7) for _ in range(200_000))]
t = time.perf_counter(); f(big); dt = time.perf_counter() - t
assert dt < 3.0, f"{dt:.2f} s"
print("OK")
