def longest_unique(s):
    """The longest substring of s with no repeated character. If several have the maximum
    length, the leftmost one. '' for ''."""
    last = {}
    lo = 0
    bi = bj = 0
    for j, c in enumerate(s):
        p = last.get(c)
        if p is not None and p >= lo:
            lo = p + 1
        last[c] = j
        if j + 1 - lo > bj - bi:
            bi, bj = lo, j + 1
    return s[bi:bj]
