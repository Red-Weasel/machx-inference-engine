Write `solution.py` with a class `TokenBucket(rate, capacity, clock=time.monotonic)`: a rate limiter.

- The bucket starts full with `capacity` tokens and refills continuously at `rate` tokens per second (fractions count), never above `capacity`.
- `clock` is a function returning the current time in seconds; tests pass a fake clock. Read it only through this function.
- `try_acquire(n=1) -> bool`: if at least `n` tokens are available now, take them and return `True`; otherwise take nothing and return `False`.
- `time_until(n=1) -> float`: seconds until `n` tokens will be available (0.0 if they are available now). Takes nothing.
- `rate <= 0` or `capacity <= 0` raises `ValueError`. `n <= 0` or `n > capacity` raises `ValueError` in both methods (such a request could never succeed).
- If the clock goes backwards, no tokens are added, and the bucket must not then hand out extra tokens when time moves forward again (time already counted is never counted twice).
