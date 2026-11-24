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
