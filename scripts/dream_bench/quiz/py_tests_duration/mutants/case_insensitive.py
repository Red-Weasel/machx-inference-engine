import re
_RX = re.compile(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", re.I)
def parse_duration(s):
    if not isinstance(s, str) or not s: raise ValueError(s)
    m = _RX.fullmatch(s)
    if not m or not any(m.groups()): raise ValueError(s)
    d, h, mi, se = (int(g) if g else 0 for g in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + se
