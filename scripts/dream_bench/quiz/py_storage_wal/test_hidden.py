import json, struct, zlib, random
from solution import encode_record, replay, CorruptLogError

def rec(obj): p = json.dumps(obj).encode(); return struct.pack(">II", len(p), zlib.crc32(p)) + p
def raw(p): return struct.pack(">II", len(p), zlib.crc32(p)) + p
e = encode_record({"op": "set", "k": "a", "v": 1}); L, c = struct.unpack(">II", e[:8])
assert L == len(e) - 8 and c == zlib.crc32(e[8:]) and json.loads(e[8:]) == {"op": "set", "k": "a", "v": 1}
log = rec({"op": "set", "k": "a", "v": 1}) + rec({"op": "set", "k": "b", "v": [1, {"x": None}]}) + rec({"op": "del", "k": "a"}) + rec({"op": "del", "k": "zz"})
assert replay(log) == ({"b": [1, {"x": None}]}, len(log)) and replay(b"") == ({}, 0)
nxt = rec({"op": "set", "k": "c", "v": "é"})
for cut in range(1, len(nxt)):   # every torn tail
    st, vb = replay(log + nxt[:cut]); assert st == {"b": [1, {"x": None}]} and vb == len(log), cut
assert replay(log + nxt) == ({"b": [1, {"x": None}], "c": "é"}, len(log + nxt))
def corrupt_at(data, off):
    try: replay(data)
    except CorruptLogError as ex: assert ex.offset == off, (ex.offset, off); return
    raise AssertionError("no CorruptLogError")
bad = bytearray(log); bad[len(rec({"op": "set", "k": "a", "v": 1})) + 9] ^= 0x40
corrupt_at(bytes(bad), len(rec({"op": "set", "k": "a", "v": 1})))
corrupt_at(log + raw(b"not json"), len(log))
corrupt_at(log + raw(b"\xff\xfe{}"), len(log))
corrupt_at(log + raw(b'{"op":"set","k":5,"v":1}'), len(log))
corrupt_at(log + raw(b'{"op":"put","k":"a","v":1}'), len(log))
corrupt_at(log + raw(b'{"op":"set","k":"a"}'), len(log))
corrupt_at(log + raw(b'[1,2]'), len(log))
corrupt_at(log + raw(b'{"op":"set","k":"a","v":NaN}'), len(log)); corrupt_at(log + raw(b'{"op":"set","k":"a","v":-Infinity}'), len(log))
u = raw(json.dumps({"op": "set", "k": "\u00e9t\u00e9", "v": "\u00fc"}, ensure_ascii=False).encode("utf-8"))
assert replay(u) == ({"\u00e9t\u00e9": "\u00fc"}, len(u)), "real multi-byte UTF-8 payload"
corrupt_at(log + struct.pack(">II", 16 * 1024 * 1024 + 1, 0) + b"xx", len(log))   # too long, even though truncated
assert replay(log + struct.pack(">II", 5, 0) + b"ab") == ({"b": [1, {"x": None}]}, len(log))  # torn, not corrupt
rng = random.Random(8)
for _ in range(300):
    st = {}; data = b""
    for _ in range(rng.randrange(0, 12)):
        k = rng.choice("abcd")
        if rng.random() < 0.7: v = rng.randrange(100); st[k] = v; data += encode_record({"op": "set", "k": k, "v": v})
        else: st.pop(k, None); data += encode_record({"op": "del", "k": k})
    assert replay(data) == (st, len(data))
print("OK")
