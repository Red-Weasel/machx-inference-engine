import subprocess, sys, time
from collections import Counter
from solution import HashRing

r = HashRing()
try: r.get("x"); raise AssertionError("empty ring get")
except LookupError: pass
try: HashRing(vnodes=0); raise AssertionError("vnodes=0")
except ValueError: pass
names = [f"srv-{i}" for i in range(5)]
r = HashRing(names)
try: r.add("srv-0"); raise AssertionError("dup add")
except ValueError: pass
try: r.remove("nope"); raise AssertionError("bad remove")
except KeyError: pass
assert r.nodes() == set(names)
keys = [f"user:{i}" for i in range(20000)]
before = {k: r.get(k) for k in keys}
assert HashRing(list(reversed(names))).get("user:77") == before["user:77"], "depends on add order"
load = Counter(before.values()); fair = len(keys) / 5
assert all(0.6 * fair <= load[n] <= 1.4 * fair for n in names), load
r.add("srv-new"); after = {k: r.get(k) for k in keys}
moved = [k for k in keys if after[k] != before[k]]
assert all(after[k] == "srv-new" for k in moved), "keys moved between old servers"
assert 0.08 <= len(moved) / len(keys) <= 0.28, len(moved) / len(keys)
r.remove("srv-new"); assert all(r.get(k) == before[k] for k in keys), "remove did not restore"
assert r.get(12345) == r.get("12345")
# deterministic across processes (hash randomization)
code = "from solution import HashRing; r=HashRing(['a','b','c']); print([r.get(f'k{i}') for i in range(50)])"
outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONHASHSEED": s}).stdout for s in ("1", "2", "3")}
assert len(outs) == 1 and outs.pop().strip(), "differs between processes"
# get is fast
big = HashRing([f"n{i}" for i in range(100)]); t = time.perf_counter()
for i in range(50000): big.get(i)
assert time.perf_counter() - t < 4.5, "get too slow"
print("OK")
