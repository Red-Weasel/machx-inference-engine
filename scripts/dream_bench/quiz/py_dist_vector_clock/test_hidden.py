import random
from solution import increment, merge, compare

assert compare({"a": 1, "b": 0}, {"a": 1}) == "equal" and compare({}, {"x": 0}) == "equal"
assert compare({"a": 1}, {"a": 2}) == "before" and compare({"a": 2}, {"a": 1}) == "after"
assert compare({"a": 1}, {"b": 1}) == "concurrent" and compare({}, {"a": 1}) == "before"
assert compare({"a": 2, "b": 1}, {"a": 1, "b": 2}) == "concurrent"
c = {"a": 1}; d = increment(c, "b"); assert c == {"a": 1} and d == {"a": 1, "b": 1}
assert increment({"a": 5}, "a") == {"a": 6} and increment({"a": 5, "b": 2}, "b") == {"a": 5, "b": 3}
assert increment({}, "n") == {"n": 1} and increment({"n": 0}, "n") == {"n": 1}
a = {"a": 3, "b": 0}; b = {"a": 1, "c": 2}; m = merge(a, b)
assert m == {"a": 3, "c": 2} and a == {"a": 3, "b": 0} and b == {"a": 1, "c": 2}
assert merge({"z": 0}, {}) == {}
for bad in ({"a": -1}, {"a": 1.0}, {"a": True}, {"a": "1"}):
    for fn in (lambda: compare(bad, {}), lambda: compare({}, bad), lambda: merge(bad, {}), lambda: merge({}, bad), lambda: increment(bad, "q")):
        try: fn(); raise AssertionError(f"accepted {bad}")
        except ValueError: pass
rng = random.Random(2)
for _ in range(3000):
    x = {k: rng.randrange(3) for k in "abc" if rng.random() < 0.7}
    y = {k: rng.randrange(3) for k in "abcd" if rng.random() < 0.7}
    keys = set("abcd"); le = all(x.get(k, 0) <= y.get(k, 0) for k in keys); ge = all(x.get(k, 0) >= y.get(k, 0) for k in keys)
    want = "equal" if le and ge else "before" if le else "after" if ge else "concurrent"
    assert compare(x, y) == want, (x, y)
    mg = merge(x, y); assert compare(x, mg) in ("before", "equal") and compare(y, mg) in ("before", "equal") and 0 not in mg.values()
print("OK")
