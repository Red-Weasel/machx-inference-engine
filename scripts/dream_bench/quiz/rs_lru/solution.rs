use std::collections::HashMap;
use std::hash::Hash;

const NIL: usize = usize::MAX;

struct Node<K, V> { k: K, v: V, prev: usize, next: usize }

pub struct LruCache<K, V> {
    cap: usize,
    map: HashMap<K, usize>,
    nodes: Vec<Option<Node<K, V>>>,
    free: Vec<usize>,
    head: usize, // most recently used
    tail: usize, // least recently used
}

impl<K: Hash + Eq + Clone, V> LruCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be positive");
        LruCache { cap: capacity, map: HashMap::new(), nodes: Vec::new(), free: Vec::new(), head: NIL, tail: NIL }
    }
    pub fn len(&self) -> usize { self.map.len() }
    pub fn is_empty(&self) -> bool { self.map.is_empty() }

    fn node(&self, i: usize) -> &Node<K, V> { self.nodes[i].as_ref().unwrap() }
    fn node_mut(&mut self, i: usize) -> &mut Node<K, V> { self.nodes[i].as_mut().unwrap() }

    fn unlink(&mut self, i: usize) {
        let (p, n) = { let x = self.node(i); (x.prev, x.next) };
        if p != NIL { self.node_mut(p).next = n } else { self.head = n }
        if n != NIL { self.node_mut(n).prev = p } else { self.tail = p }
    }
    fn push_front(&mut self, i: usize) {
        let h = self.head;
        { let x = self.node_mut(i); x.prev = NIL; x.next = h; }
        if h != NIL { self.node_mut(h).prev = i } else { self.tail = i }
        self.head = i;
    }

    pub fn get(&mut self, k: &K) -> Option<&V> {
        let i = *self.map.get(k)?;
        self.unlink(i);
        self.push_front(i);
        Some(&self.node(i).v)
    }
    pub fn peek(&self, k: &K) -> Option<&V> { self.map.get(k).map(|&i| &self.node(i).v) }

    pub fn put(&mut self, k: K, v: V) -> Option<(K, V)> {
        if let Some(&i) = self.map.get(&k) {
            self.node_mut(i).v = v;
            self.unlink(i);
            self.push_front(i);
            return None;
        }
        let mut evicted = None;
        if self.map.len() == self.cap {
            let t = self.tail;
            self.unlink(t);
            let n = self.nodes[t].take().unwrap();
            self.map.remove(&n.k);
            self.free.push(t);
            evicted = Some((n.k, n.v));
        }
        let node = Some(Node { k: k.clone(), v, prev: NIL, next: NIL });
        let i = match self.free.pop() {
            Some(i) => { self.nodes[i] = node; i }
            None => { self.nodes.push(node); self.nodes.len() - 1 }
        };
        self.map.insert(k, i);
        self.push_front(i);
        evicted
    }

    pub fn remove(&mut self, k: &K) -> Option<V> {
        let i = self.map.remove(k)?;
        self.unlink(i);
        let n = self.nodes[i].take().unwrap();
        self.free.push(i);
        Some(n.v)
    }

    pub fn keys(&self) -> Vec<K> {
        let mut out = Vec::with_capacity(self.map.len());
        let mut i = self.head;
        while i != NIL { let n = self.node(i); out.push(n.k.clone()); i = n.next; }
        out
    }
}
