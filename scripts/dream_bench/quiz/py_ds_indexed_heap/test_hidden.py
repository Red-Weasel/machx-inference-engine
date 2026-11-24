import random, time
from solution import IndexedMinHeap

class Obj:  # hashable, not orderable
    def __init__(self, n): self.n = n
h = IndexedMinHeap()
for exc, fn in ((IndexError, h.pop), (IndexError, h.peek), (KeyError, lambda: h.update("x", 1)), (KeyError, lambda: h.remove("x"))):
    try: fn(); raise AssertionError(f"expected {exc.__name__}")
    except exc: pass
a, b, c = Obj(1), Obj(2), Obj(3)
h.push(a, 5); h.push(b, 5); h.push(c, 5)
assert h.peek() == (a, 5) and len(h) == 3 and b in h
h.update(a, 5)               # same priority: a keeps its place (first pushed)
assert h.pop() == (a, 5) and h.pop() == (b, 5)
try: h.push(c, 1); raise AssertionError("duplicate accepted")
except ValueError: pass
h.push("z", 9); h.update("z", -1); assert h.peek() == ("z", -1); h.update("z", 100); assert h.pop() == (c, 5)
h.remove("z"); assert len(h) == 0 and "z" not in h
# randomized vs a model (min over (priority, push order))
rng = random.Random(9); h = IndexedMinHeap(); model = {}; order = {}; seq = 0
for step in range(20000):
    op = rng.random()
    if op < 0.4 or not model:
        k = rng.randrange(400)
        if k in model:
            try: h.push(k, 0); raise AssertionError("dup")
            except ValueError: continue
        p = rng.randrange(50); h.push(k, p); model[k] = p; order[k] = seq; seq += 1
    elif op < 0.6:
        k = min(model, key=lambda x: (model[x], order[x])); assert h.pop() == (k, model.pop(k)); del order[k]
    elif op < 0.8:
        k = rng.choice(list(model)); p = rng.randrange(50); h.update(k, p); model[k] = p
    else:
        k = rng.choice(list(model)); h.remove(k); del model[k], order[k]
    assert len(h) == len(model)
    if model: k = min(model, key=lambda x: (model[x], order[x])); assert h.peek() == (k, model[k]), step
# speed: O(log n) per op
h = IndexedMinHeap(); t = time.perf_counter()
for i in range(50000): h.push(i, rng.random())
for i in range(0, 50000, 2): h.update(i, rng.random())
for i in range(1, 50000, 4): h.remove(i)
while len(h): h.pop()
dt = time.perf_counter() - t
assert dt < 4.5, f"{dt:.2f} s"
print("OK")
