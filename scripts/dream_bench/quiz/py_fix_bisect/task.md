`solution.py` below should give the leftmost insertion point in a sorted list (like `bisect.bisect_left`, with an optional key) and a range count built on it. Callers report that it sometimes never returns, that it returns the wrong index when the value is present several times, and that it is wrong for an empty list and for values larger than every element.

Fix it without using the `bisect` module. Keep both function names, signatures and docstring behaviour. It must stay O(log n) per call. Write the corrected `solution.py`.

```python
def lower_bound(a, x, key=None):
    """Index of the first element of the sorted list `a` that is NOT less than `x`
    (len(a) if every element is less). With `key`, elements are compared as key(element);
    `x` is already a key value. a must be sorted by that key."""
    lo, hi = 0, len(a) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        v = a[mid] if key is None else key(a[mid])
        if v <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo


def count_in_range(a, low, high, key=None):
    """How many elements e of the sorted list `a` have low <= key(e) < high."""
    return lower_bound(a, high, key) - lower_bound(a, low, key)
```
