`solution.rs` below is a fixed-capacity FIFO ring buffer used for recent-event history. Users report that it panics with "index out of bounds" after it has wrapped around, that once it is full the elements come out in the wrong order, and that `RingBuffer::new(0)` builds a buffer that misbehaves instead of being refused.

Fix it. The intended behaviour:

- `new(capacity)` panics if `capacity` is 0.
- `push(v)` appends `v` as the newest element. If the buffer is full, the oldest element is removed first and returned (`Some(old)`); otherwise it returns `None`.
- `pop()` removes and returns the oldest element (`None` when empty).
- `get(i)` returns the i-th element counting from the oldest (index 0), or `None` if `i >= len()`.
- `iter()` yields the elements oldest to newest. `len`, `is_empty` and `capacity` do what their names say.

Keep the public API (names and signatures) as it is.

```rust
/// Fixed-capacity FIFO ring buffer. When full, `push` overwrites the oldest element and returns it.
pub struct RingBuffer<T> {
    buf: Vec<Option<T>>,
    head: usize, // index of the oldest element
    len: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        RingBuffer { buf: (0..capacity).map(|_| None).collect(), head: 0, len: 0 }
    }

    pub fn capacity(&self) -> usize { self.buf.len() }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    pub fn push(&mut self, v: T) -> Option<T> {
        if self.len == self.buf.len() {
            return self.buf[self.head].replace(v);
        }
        let tail = self.head + self.len;
        self.buf[tail] = Some(v);
        self.len += 1;
        None
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 { return None; }
        let v = self.buf[self.head].take();
        self.head = (self.head + 1) % self.buf.len();
        self.len -= 1;
        v
    }

    /// The i-th element, counting from the oldest (0).
    pub fn get(&self, i: usize) -> Option<&T> {
        if i > self.len { return None; }
        self.buf[self.head + i].as_ref()
    }

    /// Oldest to newest.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        (0..self.len).map(move |i| self.get(i).unwrap())
    }
}
```
