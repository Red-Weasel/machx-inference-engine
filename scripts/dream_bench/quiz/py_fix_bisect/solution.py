def lower_bound(a, x, key=None):
    """Index of the first element of the sorted list `a` that is NOT less than `x`
    (len(a) if every element is less). With `key`, elements are compared as key(element);
    `x` is already a key value. a must be sorted by that key."""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        v = a[mid] if key is None else key(a[mid])
        if v < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def count_in_range(a, low, high, key=None):
    """How many elements e of the sorted list `a` have low <= key(e) < high."""
    if high <= low:
        return 0
    return lower_bound(a, high, key) - lower_bound(a, low, key)
