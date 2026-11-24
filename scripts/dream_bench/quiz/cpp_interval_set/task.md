Write `solution.hpp` (a header, C++20, standard library only) defining a class `IntervalSet` in the global namespace: a set of `int64_t` points stored as half-open intervals `[lo, hi)`.

```cpp
class IntervalSet {
public:
    void add(int64_t lo, int64_t hi);      // add every point in [lo, hi)
    void remove(int64_t lo, int64_t hi);   // remove every point in [lo, hi)
    bool contains(int64_t x) const;
    int64_t covered() const;               // number of points in the set
    std::vector<std::pair<int64_t, int64_t>> intervals() const;
};
```

- `add` / `remove` with `lo >= hi` do nothing.
- `intervals()` returns the set as sorted, disjoint, non-empty intervals where no two touch: adding `[1,3)` and `[3,5)` gives `{[1,5)}`. Removing from the middle of an interval splits it.
- All values passed in are within ±10^15.
- `add` and `remove` must take O(log n + k) time, where n is the number of stored intervals and k the number of intervals they merge or remove; `contains` O(log n); `covered` O(1). The tests run 200,000 operations on sets of about 100,000 intervals.
