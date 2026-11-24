Write `solution.py` with `get(doc, pointer)` and `set_(doc, pointer, value)` implementing JSON Pointer (RFC 6901) over Python dicts and lists.

- `""` points at the whole document: `get` returns `doc`, `set_` returns `value` (and leaves `doc` alone).
- Otherwise the pointer must start with `/`; it is split on `/` into reference tokens, and in each token `~1` means `/` and `~0` means `~` (decode `~1` before `~0`, so `"~01"` is the key `"~1"`). A `~` followed by anything other than `0` or `1`, or a non-empty pointer without a leading `/`, raises `ValueError`.
- In a dict the token is a key (any string, including `""` and `"-"`). In a list the token must be `0` or a positive integer without leading zeros; anything else raises `IndexError`.
- `get`: a missing dict key raises `KeyError`; a list index `>= len` raises `IndexError` (including `-`); stepping into anything that is not a dict or list raises `TypeError`.
- `set_(doc, pointer, value)` changes `doc` in place and returns it. The parent must exist (same errors as `get`). In a dict it adds or replaces the key. In a list, an index `< len` replaces, an index equal to `len` or `-` appends, a larger index raises `IndexError`.
