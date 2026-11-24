import bisect, random, sys, time, types
# The functions must not use the bisect module (2026-11-21: a self-test under `if __name__ == "__main__"` that cross-checks
# against bisect is fine -- Opus 5.5 and Flash both wrote one and were failed by the old source-text check). While the
# solution is imported and while its functions run, `bisect` is a module whose every function raises.
_real = bisect
class _Poison(types.ModuleType):
    def __getattr__(self, name):
        def _no(*a, **k): raise AssertionError("must not use the bisect module")
        return _no
sys.modules["bisect"] = _Poison("bisect")
import solution
from solution import lower_bound, count_in_range
bisect = _real   # the checks below compare against the real one (the solution's own reference stays the poisoned module)
# (2026-11-21, the Codex review: reloading the poisoned module gave a solution the real functions back.) While the two
# functions run on probe inputs, every call into the bisect routines -- the C ones in _bisect or the Python ones in
# bisect.py -- is observed with the profiler hook; any such call fails the test, however the module was reached.
_hits = []
def _watch(frame, event, arg):
    if event == "c_call" and getattr(arg, "__module__", None) in ("_bisect", "bisect"): _hits.append(getattr(arg, "__name__", "?"))
    elif event == "call" and frame.f_code.co_filename.replace("\\", "/").endswith("/bisect.py"): _hits.append(frame.f_code.co_name)
_probe = [("a", 1), ("b", 3), ("c", 3), ("d", 7)]
sys.setprofile(_watch)
try:
    for _x in (-1, 0, 2, 3, 9):
        lower_bound([1, 2, 2, 3, 5], _x); lower_bound(_probe, _x, key=lambda q: q[1])
        count_in_range([1, 2, 2, 3, 5], _x, _x + 3); count_in_range(_probe, _x, _x + 3, key=lambda q: q[1])
finally:
    sys.setprofile(None)
assert not _hits, f"must not use the bisect module (called {sorted(set(_hits))})"
assert lower_bound([], 5) == 0
assert lower_bound([1, 2, 2, 2, 3], 2) == 1
assert lower_bound([1, 2, 3], 9) == 3 and lower_bound([1, 2, 3], -9) == 0 and lower_bound([1, 2, 3], 3) == 2
pairs = [("a", 1), ("b", 3), ("c", 3), ("d", 7)]
assert lower_bound(pairs, 3, key=lambda p: p[1]) == 1 and lower_bound(pairs, 4, key=lambda p: p[1]) == 3
assert count_in_range([("a", 1), ("b", 3), ("c", 9)], 1, 4, key=lambda p: p[1]) == 2
assert count_in_range([1, 2, 2, 3, 5], 2, 4) == 3 and count_in_range([1, 2], 5, 1) == 0 and count_in_range([], 0, 9) == 0
rng = random.Random(7)
for _ in range(3000):
    a = sorted(rng.randrange(20) for _ in range(rng.randrange(0, 15)))
    x = rng.randrange(-2, 23)
    assert lower_bound(a, x) == bisect.bisect_left(a, x), (a, x)
    lo, hi = rng.randrange(-2, 23), rng.randrange(-2, 23)
    assert count_in_range(a, lo, hi) == sum(lo <= v < hi for v in a), (a, lo, hi)
# O(log n): count comparisons via key
big = list(range(1_000_000)); calls = [0]
def k(v): calls[0] += 1; return v
lower_bound(big, 765_432, key=k)
assert calls[0] <= 25, f"{calls[0]} key calls: not O(log n)"
print("OK")
