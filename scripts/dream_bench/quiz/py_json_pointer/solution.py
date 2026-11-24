import re


def _tokens(p):
    if p == "":
        return []
    if not isinstance(p, str) or not p.startswith("/"):
        raise ValueError(f"invalid JSON pointer {p!r}")
    out = []
    for t in p[1:].split("/"):
        if re.search(r"~(?![01])", t):
            raise ValueError(f"invalid escape in {t!r}")
        out.append(t.replace("~1", "/").replace("~0", "~"))
    return out


def _index(t, n, allow_end):
    if t == "-":
        if allow_end:
            return n
        raise IndexError("'-' points past the end of the list")
    if not re.fullmatch(r"0|[1-9][0-9]*", t):
        raise IndexError(f"invalid list index {t!r}")
    i = int(t)
    if i > n or (i == n and not allow_end):
        raise IndexError(f"list index {i} out of range")
    return i


def _step(cur, t):
    if isinstance(cur, dict):
        return cur[t]
    if isinstance(cur, list):
        return cur[_index(t, len(cur), False)]
    raise TypeError(f"cannot step into {type(cur).__name__} with {t!r}")


def get(doc, pointer):
    cur = doc
    for t in _tokens(pointer):
        cur = _step(cur, t)
    return cur


def set_(doc, pointer, value):
    toks = _tokens(pointer)
    if not toks:
        return value
    parent = doc
    for t in toks[:-1]:
        parent = _step(parent, t)
    last = toks[-1]
    if isinstance(parent, dict):
        parent[last] = value
    elif isinstance(parent, list):
        i = _index(last, len(parent), True)
        if i == len(parent):
            parent.append(value)
        else:
            parent[i] = value
    else:
        raise TypeError(f"cannot set inside {type(parent).__name__}")
    return doc
