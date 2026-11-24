import time


class TokenBucket:
    def __init__(self, rate, capacity, clock=time.monotonic):
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be > 0")
        self.rate = float(rate)
        self.capacity = float(capacity)
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def _refill(self):
        now = self._clock()
        if now > self._last:
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now

    def _check(self, n):
        if not (0 < n <= self.capacity):
            raise ValueError(f"n must be in (0, capacity], got {n}")

    def try_acquire(self, n=1):
        self._check(n)
        self._refill()
        if self._tokens + 1e-12 >= n:
            self._tokens = max(0.0, self._tokens - n)
            return True
        return False

    def time_until(self, n=1):
        self._check(n)
        self._refill()
        behind = max(0.0, self._last - self._clock())   # clock went backwards: that stretch earns nothing
        return max(0.0, (n - self._tokens) / self.rate) + (behind if self._tokens < n else 0.0)
