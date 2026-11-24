`solution.py` below holds an `LRUCache` class. Users report three problems:

1. An item that was just read with `get` is still the first one evicted when a new key is added.
2. Calling `put` with a key that is already in the cache sometimes evicts a different item, even though the number of items does not grow.
3. `LRUCache(0)` crashes with `KeyError` on the first `put` (it should simply store nothing).

Find and fix the bugs. Keep the class name, the method names and signatures, and the documented behaviour. Write the corrected `solution.py`.

```python
from collections import OrderedDict


class LRUCache:
    """Least-recently-used cache: holds at most `capacity` items; when full, putting a NEW key evicts
    the least recently used one. Both get (of a present key) and put count as a use."""

    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = capacity
        self._d = OrderedDict()

    def get(self, key, default=None):
        if key in self._d:
            return self._d[key]
        return default

    def put(self, key, value):
        if len(self._d) >= self.capacity:
            self._d.popitem(last=False)
        self._d[key] = value

    def __len__(self):
        return len(self._d)
```
