Write `solution.py` that replays a write-ahead log after a crash.

The log is a sequence of records. Each record is: a 4-byte big-endian unsigned payload length `L`, a 4-byte big-endian unsigned CRC-32 of the payload (`zlib.crc32`), then `L` bytes of payload. The payload is UTF-8 JSON, either `{"op": "set", "k": <string>, "v": <any JSON value>}` or `{"op": "del", "k": <string>}`.

Provide:
- `encode_record(obj) -> bytes`: one record for the dict `obj` (compact JSON).
- `class CorruptLogError(Exception)` with an attribute `offset`: the byte offset where the bad record starts.
- `replay(data: bytes) -> (state, valid_bytes)`: apply the records in order to an empty dict (`set` stores, `del` removes and is a no-op for a missing key). Return the final dict and the number of bytes of complete, valid records.

Crash rules:
- If the log ends in the middle of a record (too few bytes left for the 8-byte header, or for the payload the header announces), that record was being written when the machine crashed: ignore it and stop. `valid_bytes` is where that partial record starts.
- A complete record whose CRC does not match, whose payload is not valid UTF-8 JSON of one of the two forms above (including a `k` that is not a string), or whose header announces a length over 16 MiB (16 * 1024 * 1024 bytes, checked before looking for the payload) is corruption: raise `CorruptLogError` with that record's offset.
