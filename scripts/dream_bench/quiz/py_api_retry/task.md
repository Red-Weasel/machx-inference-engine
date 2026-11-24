Write `solution.py` with a decorator factory:

`retry(times, exceptions=(Exception,), base_delay=0.1, factor=2.0, max_delay=None, sleep=time.sleep, on_retry=None)`

- `@retry(...)` wraps a (synchronous) function. When a call raises one of `exceptions` (a single exception class or a tuple of them), it is called again, up to `times` more times (so at most `times + 1` calls in total).
- Before retry number k (k = 1, 2, ...) it calls `sleep(d)` with `d = base_delay * factor ** (k - 1)`, capped at `max_delay` when that is not `None`. It must use the `sleep` it was given (tests pass a fake).
- If `on_retry` is given, call `on_retry(k, exc, d)` just before that sleep.
- An exception that is not in `exceptions` propagates at once, with no retry and no sleep. When the retries run out, the LAST exception is re-raised (the same exception object, not a wrapper).
- The return value of a successful call is returned unchanged. The wrapper keeps the wrapped function's `__name__`, `__doc__` and `__wrapped__`.
- `times` must be an int >= 0 (a bool is not accepted); otherwise raise `ValueError` when `retry(...)` is called.
