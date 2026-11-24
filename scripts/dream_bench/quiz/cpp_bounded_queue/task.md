Write `solution.hpp` (a header, C++20, standard library only) defining a thread-safe bounded FIFO queue, a class template `BoundedQueue<T>` in the global namespace:

```cpp
template <class T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity);  // throws std::invalid_argument if capacity is 0
    bool push(T v);                  // waits while full; false (and v not added) once the queue is closed
    std::optional<T> pop();          // waits while empty and open; std::nullopt once closed AND empty
    bool try_push(T v);              // never waits: false if full or closed
    std::optional<T> try_pop();      // never waits: std::nullopt if empty
    void close();                    // no more pushes; wakes every waiting thread
    size_t size() const;
};
```

- Any number of threads may call any member at the same time. The queue never holds more than `capacity` items.
- Items pushed before `close()` are still delivered by `pop()` / `try_pop()` after it; `pop()` returns `std::nullopt` only when the queue is closed and empty. A `push` that is waiting when `close()` is called returns `false`. Calling `close()` twice is harmless.
- `T` may be move-only (for example `std::unique_ptr<int>`).
- No busy-waiting: blocked calls must sleep until woken.
