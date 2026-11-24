Write `solution.rs` (a Rust module, standard library only) with a least-recently-used cache:

```rust
pub struct LruCache<K, V> { /* ... */ }

impl<K: std::hash::Hash + Eq + Clone, V> LruCache<K, V> {
    pub fn new(capacity: usize) -> Self;               // panics if capacity is 0
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn get(&mut self, k: &K) -> Option<&V>;        // a hit makes k the most recently used
    pub fn peek(&self, k: &K) -> Option<&V>;           // does not change the order
    pub fn put(&mut self, k: K, v: V) -> Option<(K, V)>;
    pub fn remove(&mut self, k: &K) -> Option<V>;
    pub fn keys(&self) -> Vec<K>;                       // most recently used first
}
```

- `put` with a key already present replaces its value, makes it the most recently used, and returns `None` (nothing is evicted).
- `put` with a new key when the cache holds `capacity` entries first evicts the least recently used entry and returns it as `Some((key, value))`; otherwise it returns `None`. The new key becomes the most recently used.
- `remove` takes the entry out (returning its value) without affecting the order of the others.
- `get`, `peek`, `put` and `remove` must take O(1) average time: the tests run hundreds of thousands of operations on a cache of 50,000 entries.
