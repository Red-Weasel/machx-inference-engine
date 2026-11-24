A teammate wrote `parse_duration(s)` in `solution.py`. It turns a duration string into whole seconds:

- The string is one or more components `<digits><unit>`, units `d` (days), `h` (hours), `m` (minutes), `s` (seconds).
- Units must appear in that order (d, h, m, s), each at most once, with no spaces. Leading zeros are allowed (`"01h"`).
- `"1h30m"` -> 5400, `"2d"` -> 172800, `"0s"` -> 0, `"1d1s"` -> 86401.
- Anything else raises `ValueError`: the empty string, unknown units (`"5x"`), units out of order (`"30m1h"`), a repeated unit (`"1h1h"`), uppercase (`"1H"`), spaces, signs, or trailing junk (`"1h30"`).

Write `test_solution.py`: a test script that does `from solution import parse_duration`, checks the behaviour above with plain `assert` statements, and exits non-zero if anything is wrong (an uncaught `AssertionError` is enough). It must pass on a correct implementation and fail on implementations with bugs. Do not write the implementation itself.
