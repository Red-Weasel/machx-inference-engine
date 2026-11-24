Write `solution.py` with a function `merge_intervals(intervals)`.

`intervals` is an iterable of `(start, end)` pairs of integers describing half-open ranges `[start, end)`. Return the union as a list of `(start, end)` tuples, sorted by start, where no two returned ranges overlap or touch.

- Ranges that overlap or touch merge: `[(1, 3), (3, 5)]` -> `[(1, 5)]`; `[(1, 4), (2, 3)]` -> `[(1, 4)]`; `[(1, 2), (4, 5)]` stays as two ranges.
- Empty ranges (`start == end`) cover nothing and are dropped, even when they sit inside or next to another range.
- `start > end` is invalid: raise `ValueError`.
- Do not modify the input. The input may be in any order and may be a generator.
- 200,000 ranges must take well under a second.
