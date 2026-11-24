class IndexedMinHeap:
    def __init__(self):
        self._h = []      # entries [priority, seq, key]
        self._pos = {}    # key -> index in _h
        self._seq = 0

    def __len__(self):
        return len(self._h)

    def __contains__(self, key):
        return key in self._pos

    def _less(self, i, j):
        a, b = self._h[i], self._h[j]
        return (a[0], a[1]) < (b[0], b[1])

    def _swap(self, i, j):
        h = self._h
        h[i], h[j] = h[j], h[i]
        self._pos[h[i][2]] = i
        self._pos[h[j][2]] = j

    def _up(self, i):
        while i > 0:
            p = (i - 1) // 2
            if not self._less(i, p):
                break
            self._swap(i, p)
            i = p

    def _down(self, i):
        n = len(self._h)
        while True:
            l = 2 * i + 1
            m = i
            if l < n and self._less(l, m):
                m = l
            if l + 1 < n and self._less(l + 1, m):
                m = l + 1
            if m == i:
                return
            self._swap(i, m)
            i = m

    def push(self, key, priority):
        if key in self._pos:
            raise ValueError(f"duplicate key {key!r}")
        self._h.append([priority, self._seq, key])
        self._seq += 1
        self._pos[key] = len(self._h) - 1
        self._up(len(self._h) - 1)

    def peek(self):
        if not self._h:
            raise IndexError("peek from an empty heap")
        return self._h[0][2], self._h[0][0]

    def pop(self):
        if not self._h:
            raise IndexError("pop from an empty heap")
        key, pri = self._h[0][2], self._h[0][0]
        self.remove(key)
        return key, pri

    def remove(self, key):
        i = self._pos.pop(key)
        last = self._h.pop()
        if i < len(self._h):
            self._h[i] = last
            self._pos[last[2]] = i
            self._up(i)
            self._down(self._pos[last[2]])

    def update(self, key, priority):
        i = self._pos[key]
        self._h[i][0] = priority
        self._up(i)
        self._down(self._pos[key])
