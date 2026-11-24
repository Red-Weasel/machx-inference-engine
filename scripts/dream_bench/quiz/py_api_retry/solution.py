import functools
import time


def retry(times, exceptions=(Exception,), base_delay=0.1, factor=2.0, max_delay=None, sleep=time.sleep, on_retry=None):
    if type(times) is not int or times < 0:
        raise ValueError("times must be an int >= 0")
    if isinstance(exceptions, type):
        exceptions = (exceptions,)
    exceptions = tuple(exceptions)

    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            for attempt in range(times + 1):
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    if attempt == times:
                        raise
                    d = base_delay * factor ** attempt
                    if max_delay is not None:
                        d = min(d, max_delay)
                    if on_retry is not None:
                        on_retry(attempt + 1, e, d)
                    sleep(d)
        return wrapper
    return deco
