Write `solution.py` with a class `IndexedMinHeap`: a priority queue where every item has a unique key and its priority can be changed later.

- `push(key, priority)`: add a key. Raise `ValueError` if the key is already present.
- `pop()`: remove and return `(key, priority)` with the smallest priority. Ties go to the key that was pushed first (an `update` does not change a key's place in that order). `IndexError` if empty.
- `peek()`: the same pair without removing it. `IndexError` if empty.
- `update(key, priority)`: change a present key's priority, up or down. `KeyError` if absent.
- `remove(key)`: delete a present key. `KeyError` if absent.
- `len(h)` and `key in h` work.
- Keys are hashable but not necessarily comparable with each other; priorities are numbers.
- push, pop, update and remove must be O(log n) (100,000 mixed operations in well under a second). Do not scan the whole heap.
