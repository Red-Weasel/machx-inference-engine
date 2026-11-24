import random, time
from solution import longest_unique

def slow(s):
    best = (0, 0)
    for i in range(len(s)):
        seen = set()
        for j in range(i, len(s)):
            if s[j] in seen: break
            seen.add(s[j])
            if j + 1 - i > best[1] - best[0]: best = (i, j + 1)
    return s[best[0]:best[1]]

for s, want in [("", ""), ("a", "a"), ("abcabcbb", "abc"), ("bbbb", "b"), ("pwwkew", "wke"), ("abba", "ab"), ("dvdf", "vdf"), ("tmmzuxt", "mzuxt")]:
    assert longest_unique(s) == want, (s, longest_unique(s), want)
rng = random.Random(3)
for _ in range(2000):
    s = "".join(rng.choice("abcde") for _ in range(rng.randrange(0, 25)))
    assert longest_unique(s) == slow(s), s
# random CJK text, then a long cyclic stretch whose windows are 2,500 characters wide (quadratic code takes minutes)
big = "".join(chr(0x4E00 + rng.randrange(4000)) for _ in range(150_000)) + "".join(chr(0x4E00 + (i % 2500)) for i in range(150_000))
t = time.perf_counter(); r = longest_unique(big); dt = time.perf_counter() - t
def fast(s):
    last = {}; lo = bi = bj = 0
    for j, ch in enumerate(s):
        if last.get(ch, -1) >= lo: lo = last[ch] + 1
        last[ch] = j
        if j + 1 - lo > bj - bi: bi, bj = lo, j + 1
    return s[bi:bj]
assert longest_unique("\u00e9") == "\u00e9" and longest_unique("\u00e9a\u00e9") == "\u00e9a"
assert r == fast(big), "wrong answer on the long string"
assert dt < 3.0, f"took {dt:.2f} s"
print("OK")
