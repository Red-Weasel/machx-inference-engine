Write `solution.py` with a class `HashRing(nodes=(), vnodes=100)`: consistent hashing with virtual nodes, used to decide which server owns a key.

- `add(node)`: add a server (a string). `ValueError` if it is already present.
- `remove(node)`: remove it. `KeyError` if absent.
- `get(key)`: the server that owns `key` (any value; use `str(key)`). `LookupError` if the ring is empty.
- `nodes()`: the set of current servers.
- `vnodes < 1` raises `ValueError`.

Required properties:
- Deterministic: the same nodes give the same answer for a key, independent of the order they were added and across separate Python runs (so do not use the built-in `hash()`, which is randomized per process).
- Adding a server only moves keys TO the new server; removing it moves exactly those keys back, restoring the previous mapping.
- Balanced: with 5 servers and the default 100 virtual nodes each, every server owns between 60% and 140% of an even share of 20,000 keys. Adding a 6th server moves roughly 1/6 of the keys (between 8% and 28%).
- `get` must be O(log n) in the number of virtual nodes.
