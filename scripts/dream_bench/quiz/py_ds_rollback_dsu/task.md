Write `solution.py` with a class `RollbackDSU(n)`: a union-find (disjoint set) over the elements `0 .. n-1` that can undo unions.

- `find(x)`: the representative of x's set. `IndexError` if x is not in `0 .. n-1` (negative numbers too).
- `union(a, b)`: merge the two sets; return `True` if they were separate, `False` if already together.
- `connected(a, b)`, `size(x)` (the size of x's set), `count()` (the number of sets).
- `snapshot()`: return a token for the current state. `rollback(token)`: undo every union made since that snapshot, restoring `find` / `size` / `count` exactly. Rolling back to the current state is a no-op. A token that does not correspond to a reachable earlier state (for example one taken on a branch of unions that has since been rolled back, even if the same number of unions has been made again, or a made-up value such as `99`) raises `ValueError`.
- `n` may be 0; negative `n` raises `ValueError`.
- `rollback` costs O(1) per union it undoes; every other operation must be O(log n): 200,000 unions and finds on n = 200,000 in well under two seconds, even when unions are made in an adversarial order (for example always joining a big set into a single element).
