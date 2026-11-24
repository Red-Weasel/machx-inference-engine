def longest_unique(s):
    """The longest substring of s with no repeated character. If several have the maximum
    length, the leftmost one. '' for ''."""
    best = (0, 0)
    for i in range(len(s)):
        seen = set()
        for j in range(i, len(s)):
            if s[j] in seen:
                break
            seen.add(s[j])
            if j + 1 - i > best[1] - best[0]:
                best = (i, j + 1)
    return s[best[0]:best[1]]
