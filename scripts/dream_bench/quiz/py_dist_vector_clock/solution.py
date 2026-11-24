def _check(c):
    for k, v in c.items():
        if type(v) is not int or v < 0:
            raise ValueError(f"bad counter {k!r}: {v!r}")


def increment(clock, node):
    _check(clock)
    out = {k: v for k, v in clock.items() if v}
    out[node] = out.get(node, 0) + 1
    return out


def merge(a, b):
    _check(a); _check(b)
    out = {}
    for k in set(a) | set(b):
        v = max(a.get(k, 0), b.get(k, 0))
        if v:
            out[k] = v
    return out


def compare(a, b):
    _check(a); _check(b)
    less = greater = False
    for k in set(a) | set(b):
        x, y = a.get(k, 0), b.get(k, 0)
        if x < y: less = True
        elif x > y: greater = True
    if less and greater: return "concurrent"
    if less: return "before"
    if greater: return "after"
    return "equal"
