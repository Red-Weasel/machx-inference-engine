class RollbackDSU:
    def __init__(self, n):
        if n < 0:
            raise ValueError("n must be >= 0")
        self._p = list(range(n))
        self._sz = [1] * n
        self._ops = []          # (kept root, attached root, union id) per successful union
        self._comps = n
        self._next_id = 1

    def find(self, x):
        if not (0 <= x < len(self._p)):
            raise IndexError(x)
        while self._p[x] != x:
            x = self._p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self._sz[a] < self._sz[b]:
            a, b = b, a
        self._p[b] = a
        self._sz[a] += self._sz[b]
        self._ops.append((a, b, self._next_id))
        self._next_id += 1
        self._comps -= 1
        return True

    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def size(self, x):
        return self._sz[self.find(x)]

    def count(self):
        return self._comps

    def snapshot(self):
        # (depth, id of the union on top): a token from a discarded branch no longer matches
        return (len(self._ops), self._ops[-1][2] if self._ops else 0)

    def rollback(self, token):
        ok = (type(token) is tuple and len(token) == 2 and all(type(v) is int for v in token)
              and 0 <= token[0] <= len(self._ops)
              and (self._ops[token[0] - 1][2] if token[0] else 0) == token[1])
        if not ok:
            raise ValueError(f"invalid snapshot token {token!r}")
        depth = token[0]
        while len(self._ops) > depth:
            a, b, _ = self._ops.pop()
            self._p[b] = b
            self._sz[a] -= self._sz[b]
            self._comps += 1
