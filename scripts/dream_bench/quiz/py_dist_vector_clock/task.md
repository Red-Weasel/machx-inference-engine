Write `solution.py` with three functions for vector clocks. A clock is a dict mapping a node name (str) to a non-negative int counter; a node that is missing from the dict counts as 0, so `{"a": 1, "b": 0}` and `{"a": 1}` are the same clock.

- `increment(clock, node)`: return a NEW clock with `node`'s counter one higher. Do not modify the argument.
- `merge(a, b)`: return a NEW clock holding the element-wise maximum. Leave out entries whose value is 0.
- `compare(a, b)`: return `"equal"`, `"before"` (a happened before b: every counter of a <= b's and at least one is smaller), `"after"` (the reverse), or `"concurrent"` (neither).
- Any counter that is negative or not an int (bool counts as not an int) raises `ValueError`, in every function.
