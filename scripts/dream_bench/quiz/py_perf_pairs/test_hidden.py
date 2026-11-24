import random, time
from solution import count_pairs_divisible as f

def brute(a, k): return sum((a[i] + a[j]) % k == 0 for i in range(len(a)) for j in range(i + 1, len(a)))
assert f([1, 2, 3, 4, 5], 3) == 4 and f([-1, 1, 4], 5) == 2 and f([], 7) == 0 and f([5], 5) == 0
assert f([0, 0, 0], 1) == 3 and f([2, 2, 2, 2], 4) == 6
for bad in (0, -3):
    try: f([1, 2], bad); raise AssertionError("k < 1 accepted")
    except ValueError: pass
rng = random.Random(5)
for _ in range(1500):
    k = rng.randrange(1, 12); a = [rng.randrange(-30, 30) for _ in range(rng.randrange(0, 18))]
    assert f(a, k) == brute(a, k), (a, k)
mid = [rng.randrange(-10**6, 10**6) for _ in range(1500)]
for k in (3, 1000, 10**7): assert f(mid, k) == brute(mid, k), k
assert f([-1, 1], 3) == 1 and f([10**9, -10**9 + 6], 3) == 1
big = [rng.randrange(-10**9, 10**9) for _ in range(200_000)]
for k in (7, 1_000_003, 10**12):
    t = time.perf_counter(); f(big, k); dt = time.perf_counter() - t
    assert dt < 3.0, f"k={k}: {dt:.2f} s"
print("OK")
