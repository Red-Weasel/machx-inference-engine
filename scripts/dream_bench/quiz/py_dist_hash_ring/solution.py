import bisect
import hashlib


def _h(s):
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


class HashRing:
    def __init__(self, nodes=(), vnodes=100):
        if vnodes < 1:
            raise ValueError("vnodes must be >= 1")
        self.vnodes = vnodes
        self._points = []       # sorted hash points
        self._owner = {}        # point -> node
        self._nodes = set()
        for n in nodes:
            self.add(n)

    def _vpoints(self, node):
        return [_h(f"{node}#{i}") for i in range(self.vnodes)]

    def add(self, node):
        if node in self._nodes:
            raise ValueError(f"node {node!r} already present")
        self._nodes.add(node)
        for p in self._vpoints(node):
            cur = self._owner.get(p)
            if cur is None:
                bisect.insort(self._points, p)
                self._owner[p] = node
            elif node < cur:            # collision (astronomically rare): smallest name wins, order-independent
                self._owner[p] = node

    def remove(self, node):
        if node not in self._nodes:
            raise KeyError(node)
        self._nodes.remove(node)
        for p in self._vpoints(node):
            if self._owner.get(p) == node:
                others = [n for n in self._nodes if p in self._vpoints(n)]
                if others:
                    self._owner[p] = min(others)
                else:
                    del self._owner[p]
                    self._points.pop(bisect.bisect_left(self._points, p))

    def get(self, key):
        if not self._points:
            raise LookupError("empty ring")
        i = bisect.bisect_right(self._points, _h(str(key)))
        return self._owner[self._points[i % len(self._points)]]

    def nodes(self):
        return set(self._nodes)
