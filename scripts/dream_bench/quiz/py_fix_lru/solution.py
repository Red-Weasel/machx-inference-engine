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
            self._d.move_to_end(key)
            return self._d[key]
        return default

    def put(self, key, value):
        if self.capacity == 0:
            return
        if key in self._d:
            self._d.move_to_end(key)
            self._d[key] = value
            return
        if len(self._d) >= self.capacity:
            self._d.popitem(last=False)
        self._d[key] = value

    def __len__(self):
        return len(self._d)
