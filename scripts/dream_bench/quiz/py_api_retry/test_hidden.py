import time
from solution import retry

slept = []
def flaky(n_fail, exc=ValueError):
    state = {"n": 0}
    def f(x, y=0):
        """doc of f"""
        state["n"] += 1
        if state["n"] <= n_fail: raise exc(f"fail {state['n']}")
        return (x, y, state["n"])
    return f, state
f, st = flaky(2); g = retry(3, sleep=slept.append)(f)
assert g(1, y=2) == (1, 2, 3) and slept == [0.1, 0.2] and st["n"] == 3
assert g.__name__ == "f" and g.__doc__ == "doc of f" and g.__wrapped__ is f
slept.clear(); f, st = flaky(10); g = retry(2, base_delay=1, factor=3, sleep=slept.append)(f)
try: g(0); raise AssertionError("should raise")
except ValueError as e: assert str(e) == "fail 3" and st["n"] == 3 and slept == [1, 3]
slept.clear(); f, st = flaky(1, KeyError); g = retry(5, exceptions=ValueError, sleep=slept.append)(f)
try: g(0); raise AssertionError("should raise")
except KeyError: assert st["n"] == 1 and slept == []
slept.clear(); f, st = flaky(1, KeyError); g = retry(5, exceptions=(ValueError, KeyError), sleep=slept.append)(f)
assert g(0)[2] == 2 and slept == [0.1]
slept.clear(); f, st = flaky(5); log = []
g = retry(5, base_delay=1, factor=10, max_delay=50, sleep=slept.append, on_retry=lambda k, e, d: log.append((k, str(e), d)))(f)
assert g(0)[2] == 6 and slept == [1, 10, 50, 50, 50] and log == [(1, "fail 1", 1), (2, "fail 2", 10), (3, "fail 3", 50), (4, "fail 4", 50), (5, "fail 5", 50)]
order = []; f, st = flaky(2)
g = retry(3, sleep=lambda d: order.append(("sleep", d)), on_retry=lambda k, e, d: order.append(("cb", k, d)))(f); g(0)
assert order == [("cb", 1, 0.1), ("sleep", 0.1), ("cb", 2, 0.2), ("sleep", 0.2)], order
slept.clear(); f, st = flaky(1); g = retry(0, sleep=slept.append)(f)
try: g(0); raise AssertionError("times=0 retried")
except ValueError: assert st["n"] == 1 and slept == []
err = RuntimeError("same")
def h(): raise err
try: retry(1, sleep=lambda d: None)(h)(); raise AssertionError
except RuntimeError as e: assert e is err
for bad in (-1, 1.5, True, "2"):
    try: retry(bad); raise AssertionError(f"times={bad!r}")
    except ValueError: pass
real = time.sleep; time.sleep = lambda d: (_ for _ in ()).throw(AssertionError("used time.sleep"))
try:
    f, st = flaky(1); assert retry(1, sleep=lambda d: None)(f)(0)[2] == 2
finally:
    time.sleep = real
print("OK")
