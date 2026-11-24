import re
_U = {"d": 86400, "h": 3600, "m": 60, "s": 1}
def parse_duration(s):
    if not isinstance(s, str) or not re.fullmatch(r"(\d+d)*(\d+h)*(\d+m)*(\d+s)*", s) or not s: raise ValueError(s)
    return sum(int(n) * _U[u] for n, u in re.findall(r"(\d+)([dhms])", s))
