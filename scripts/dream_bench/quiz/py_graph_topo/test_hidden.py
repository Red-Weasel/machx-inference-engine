import random, time
from solution import topo_order, CycleError

assert topo_order({}) == [] and topo_order({"a": []}) == ["a"]
assert topo_order({"b": ["a"], "c": ["a"]}) == ["a", "b", "c"]
assert topo_order({"app": ["lib", "log"], "lib": ["log"], "test": []}) == ["log", "lib", "app", "test"]
assert topo_order({"x": ["y", "y"]}) == ["y", "x"] and topo_order({"z": iter(["a"])}) == ["a", "z"]
def cyc_of(g):
    try: topo_order(g)
    except CycleError as e: assert isinstance(e, ValueError); return e.cycle
    raise AssertionError("no CycleError")
assert cyc_of({"a": ["a"]}) == ["a", "a"]
assert cyc_of({"b": ["c"], "c": ["a"], "a": ["b"]}) == ["a", "b", "c", "a"]
def valid(g, c):
    assert len(c) >= 2 and c[0] == c[-1] and len(set(c[:-1])) == len(c) - 1 and c[0] == min(c[:-1]), c
    for u, v in zip(c, c[1:]): assert v in set(g.get(u, ())), (u, v, c)
g = {"m": ["n"], "n": ["m", "q"], "q": [], "z": ["m"]}; valid(g, cyc_of(g))
rng = random.Random(12)
for _ in range(1500):
    nodes = [f"n{i:02d}" for i in range(rng.randrange(1, 12))]
    g = {u: [rng.choice(nodes) for _ in range(rng.randrange(0, 3))] for u in nodes if rng.random() < 0.9}
    try:
        out = topo_order(g)
    except CycleError as e:
        valid(g, e.cycle); continue
    allnodes = set(g) | {d for ds in g.values() for d in ds}
    assert sorted(out) == sorted(allnodes) and len(out) == len(allnodes)
    where = {n: i for i, n in enumerate(out)}
    assert all(where[d] < where[u] for u, ds in g.items() for d in ds)
    # smallest-ready-first uniqueness check
    indeg = {n: 0 for n in allnodes}; users = {n: [] for n in allnodes}
    for u, ds in g.items():
        for d in set(ds): indeg[u] += 1; users[d].append(u)
    import heapq; h = [n for n in allnodes if indeg[n] == 0]; heapq.heapify(h); want = []
    while h:
        n = heapq.heappop(h); want.append(n)
        for u in users[n]:
            indeg[u] -= 1
            if indeg[u] == 0: heapq.heappush(h, u)
    assert out == want
N = 100_000; chain = {i: [i - 1] for i in range(1, N)}
t = time.perf_counter(); out = topo_order(chain); assert out == list(range(N)) and time.perf_counter() - t < 6.0
loop = {i: [i + 1] for i in range(N - 1)}; loop[N - 1] = [0]
c = cyc_of(loop); assert len(c) == N + 1 and c[0] == 0 and c[1] == 1
print("OK")
