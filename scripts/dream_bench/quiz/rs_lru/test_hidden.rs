mod solution;
use solution::LruCache;
use std::time::Instant;

// a slow but obviously-right model: Vec of (key, value), most recent first
struct Model { cap: usize, v: Vec<(u32, String)> }
impl Model {
    fn pos(&self, k: u32) -> Option<usize> { self.v.iter().position(|e| e.0 == k) }
    fn get(&mut self, k: u32) -> Option<String> { let p = self.pos(k)?; let e = self.v.remove(p); let r = e.1.clone(); self.v.insert(0, e); Some(r) }
    fn put(&mut self, k: u32, val: String) -> Option<(u32, String)> {
        if let Some(p) = self.pos(k) { self.v.remove(p); self.v.insert(0, (k, val)); return None; }
        let ev = if self.v.len() == self.cap { self.v.pop() } else { None };
        self.v.insert(0, (k, val)); ev
    }
    fn remove(&mut self, k: u32) -> Option<String> { let p = self.pos(k)?; Some(self.v.remove(p).1) }
}

fn main() {
    let mut seed: u64 = 99;
    let mut rnd = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as u32 };
    for cap in 1..=6usize {
        let mut c: LruCache<u32, String> = LruCache::new(cap);
        let mut m = Model { cap, v: Vec::new() };
        for step in 0..20000 {
            let k = rnd() % 10;
            match rnd() % 10 {
                0..=3 => { let val = format!("{}:{}", k, step); assert_eq!(c.put(k, val.clone()), m.put(k, val), "put cap {} step {}", cap, step); }
                4..=6 => assert_eq!(c.get(&k).cloned(), m.get(k), "get cap {} step {}", cap, step),
                7 => assert_eq!(c.peek(&k).cloned(), m.pos(k).map(|p| m.v[p].1.clone()), "peek"),
                _ => assert_eq!(c.remove(&k), m.remove(k), "remove cap {} step {}", cap, step),
            }
            assert_eq!(c.len(), m.v.len());
            assert_eq!(c.is_empty(), m.v.is_empty());
            if step % 5 == 0 { assert_eq!(c.keys(), m.v.iter().map(|e| e.0).collect::<Vec<_>>(), "keys cap {} step {}", cap, step); }
        }
    }
    // peek does not refresh; replacing a value in a full cache evicts nothing
    let mut c = LruCache::new(2);
    c.put("a", 1); c.put("b", 2); assert_eq!(c.peek(&"a"), Some(&1));
    assert_eq!(c.put("c", 3), Some(("a", 1)));
    assert_eq!(c.put("b", 20), None); assert_eq!(c.keys(), vec!["b", "c"]);
    assert!(std::panic::catch_unwind(|| LruCache::<u8, u8>::new(0)).is_err(), "new(0) must panic");
    // speed: 50,000 entries, 600,000 operations
    let t = Instant::now();
    let mut big: LruCache<u32, u64> = LruCache::new(50_000);
    let mut hits = 0u64;
    for i in 0..600_000u64 {
        let k = rnd() % 100_000;
        if i % 3 == 0 { big.put(k, i); } else if big.get(&k).is_some() { hits += 1; }
        if i % 97 == 0 { big.remove(&(rnd() % 100_000)); }
        if i % 5 == 0 && big.peek(&k).is_some() { hits += 1; }
    }
    assert!(big.len() <= 50_000 && hits > 0);
    let dt = t.elapsed().as_secs_f64();
    assert!(dt < 12.0, "600,000 operations took {:.1} s (not O(1))", dt);
    println!("{}", std::env::args().nth(1).unwrap_or_default());
}
