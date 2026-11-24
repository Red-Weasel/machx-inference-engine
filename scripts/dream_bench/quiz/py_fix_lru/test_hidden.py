import random
from collections import OrderedDict
from solution import LRUCache

c = LRUCache(2); c.put("a", 1); c.put("b", 2)
assert c.get("a") == 1          # a becomes most recent
c.put("c", 3)                   # evicts b
assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3, "get must refresh recency"
c = LRUCache(2); c.put("a", 1); c.put("b", 2); c.put("a", 10)
assert len(c) == 2 and c.get("b") == 2 and c.get("a") == 10, "updating a key must not evict"
c.put("c", 3)                   # recency now b < a (the get calls above): b is evicted
assert len(c) == 2 and c.get("b") is None and c.get("a") == 10 and c.get("c") == 3
z = LRUCache(0); z.put("x", 1); assert len(z) == 0 and z.get("x", "dflt") == "dflt"
try:
    LRUCache(-1); raise AssertionError("negative capacity accepted")
except ValueError:
    pass
assert LRUCache(3).get("missing", 7) == 7
n = LRUCache(2); n.put("x", None); assert n.get("x", "miss") is None and len(n) == 1
# randomized against a model
rng = random.Random(1)
for cap in (1, 2, 3, 5):
    c = LRUCache(cap); ref = OrderedDict()
    for _ in range(3000):
        k = rng.randrange(8)
        if rng.random() < 0.5:
            v = rng.random(); c.put(k, v)
            if k in ref: ref.move_to_end(k)
            elif len(ref) >= cap: ref.popitem(last=False)
            ref[k] = v
        else:
            got = c.get(k)
            want = None
            if k in ref: ref.move_to_end(k); want = ref[k]
            assert got == want, (cap, k, got, want)
        assert len(c) == len(ref)
print("OK")
