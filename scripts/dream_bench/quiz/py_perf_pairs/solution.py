from collections import Counter


def count_pairs_divisible(nums, k):
    if k < 1:
        raise ValueError("k must be >= 1")
    c = Counter(x % k for x in nums)
    total = 0
    for r, n in c.items():
        s = (-r) % k
        if r < s:
            total += n * c.get(s, 0)
        elif r == s:
            total += n * (n - 1) // 2
    return total
