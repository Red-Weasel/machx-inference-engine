Write `solution.py` with a function `count_pairs_divisible(nums, k)`.

It returns the number of index pairs `(i, j)` with `i < j` such that `(nums[i] + nums[j])` is divisible by `k`.

- `nums` is a list of integers, possibly negative, possibly with repeats; `k` is an integer >= 1. If `k < 1`, raise `ValueError`.
- Examples: `count_pairs_divisible([1, 2, 3, 4, 5], 3)` is 4 (1+2, 1+5, 2+4, 4+5). `count_pairs_divisible([-1, 1, 4], 5)` is 2 (-1+1 = 0 and 1+4 = 5 are both divisible by 5).
- It must handle `len(nums) == 200_000` with large `k` in well under a second.
