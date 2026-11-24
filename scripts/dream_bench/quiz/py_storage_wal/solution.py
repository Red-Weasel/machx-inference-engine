import json
import struct
import zlib

MAX_PAYLOAD = 16 * 1024 * 1024


class CorruptLogError(Exception):
    def __init__(self, msg, offset):
        super().__init__(f"{msg} at offset {offset}")
        self.offset = offset


def _no_constant(name):
    raise ValueError(f"{name} is not valid JSON")


def encode_record(obj):
    p = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return struct.pack(">II", len(p), zlib.crc32(p)) + p


def replay(data):
    data = bytes(data)
    state = {}
    off = 0
    n = len(data)
    while off < n:
        if n - off < 8:
            break
        length, crc = struct.unpack_from(">II", data, off)
        if length > MAX_PAYLOAD:
            raise CorruptLogError("record too long", off)
        if n - off - 8 < length:
            break
        p = data[off + 8: off + 8 + length]
        if zlib.crc32(p) != crc:
            raise CorruptLogError("checksum mismatch", off)
        try:
            rec = json.loads(p.decode("utf-8"), parse_constant=_no_constant)
        except (UnicodeDecodeError, ValueError):
            raise CorruptLogError("bad payload", off) from None
        if not isinstance(rec, dict) or not isinstance(rec.get("k"), str):
            raise CorruptLogError("bad record", off)
        if rec.get("op") == "set" and "v" in rec:
            state[rec["k"]] = rec["v"]
        elif rec.get("op") == "del":
            state.pop(rec["k"], None)
        else:
            raise CorruptLogError("unknown record", off)
        off += 8 + length
    return state, off
