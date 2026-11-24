Write `solution.py` with `topo_order(graph)` and an exception class `CycleError` (a subclass of `ValueError`) with an attribute `cycle`.

- `graph` maps each node to an iterable of the nodes it depends on. A node that appears only as a dependency is still a node. Duplicate dependencies are allowed.
- Return a list of all nodes where every node comes after all of its dependencies. When several nodes are ready at the same time, take the smallest first (nodes are mutually comparable, e.g. all strings), so the result is unique.
- If the dependencies contain a cycle, raise `CycleError`. Its `cycle` is a list of nodes `[n0, n1, ..., nk, n0]` where each node depends on the next one, the first and last entries are the same node, the other entries are distinct, and n0 is the smallest node of that cycle. A node that depends on itself gives `[n, n]`.
- 100,000 nodes in a long chain must work quickly and must not hit Python's recursion limit.
