import heapq


class CycleError(ValueError):
    def __init__(self, cycle):
        super().__init__("dependency cycle: " + " -> ".join(map(str, cycle)))
        self.cycle = cycle


def topo_order(graph):
    deps = {}
    for n, ds in graph.items():
        deps.setdefault(n, set()).update(ds)
        for d in list(deps[n]):
            deps.setdefault(d, set())
    users = {n: [] for n in deps}
    indeg = {n: len(ds) for n, ds in deps.items()}
    for n, ds in deps.items():
        for d in ds:
            users[d].append(n)
    heap = [n for n, k in indeg.items() if k == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        n = heapq.heappop(heap)
        out.append(n)
        for u in users[n]:
            indeg[u] -= 1
            if indeg[u] == 0:
                heapq.heappush(heap, u)
    if len(out) == len(deps):
        return out
    stuck = {n for n, k in indeg.items() if k > 0}
    n = min(stuck)
    path, pos = [], {}
    while n not in pos:
        pos[n] = len(path)
        path.append(n)
        n = min(d for d in deps[n] if d in stuck)
    cyc = path[pos[n]:]
    k = cyc.index(min(cyc))
    cyc = cyc[k:] + cyc[:k]
    raise CycleError(cyc + [cyc[0]])
