from solution import TokenBucket

class Clock:
    def __init__(self, t=100.0): self.t = t
    def __call__(self): return self.t
c = Clock(); b = TokenBucket(2, 4, clock=c)
assert all(b.try_acquire() for _ in range(4)) and not b.try_acquire()
assert abs(b.time_until() - 0.5) < 1e-9 and abs(b.time_until(4) - 2.0) < 1e-9
c.t += 0.25; assert not b.try_acquire() and abs(b.time_until() - 0.25) < 1e-9   # 0.5 token, nothing taken
c.t += 0.25; assert b.try_acquire() and not b.try_acquire()
c.t += 100; assert b.time_until(4) == 0.0 and b.try_acquire(4) and not b.try_acquire(0.5)   # capped at capacity
c.t += 0.25; assert b.try_acquire(0.5)
for bad in (0, -1, 5):
    for fn in (b.try_acquire, b.time_until):
        try: fn(bad); raise AssertionError(f"n={bad}")
        except ValueError: pass
for r, cap in ((0, 1), (1, 0), (-1, 5)):
    try: TokenBucket(r, cap, clock=c); raise AssertionError("bad config")
    except ValueError: pass
# backwards clock: no gain, and no double counting afterwards
c = Clock(10.0); b = TokenBucket(1, 10, clock=c); assert b.try_acquire(10)
c.t = 5.0; assert not b.try_acquire(1) and abs(b.time_until(1) - 6.0) < 1e-9, "token first available at t=11"
c.t = 10.5; assert not b.try_acquire(1), "time before 10.0 counted twice"
c.t = 11.0; assert b.try_acquire(1) and not b.try_acquire(0.5)
# clock is only read through the function given
import time as _t
real = _t.monotonic; _t.monotonic = lambda: (_ for _ in ()).throw(AssertionError("read time.monotonic"))
try:
    c = Clock(); b = TokenBucket(1, 1, clock=c); b.try_acquire(); b.time_until()
finally:
    _t.monotonic = real
print("OK")
