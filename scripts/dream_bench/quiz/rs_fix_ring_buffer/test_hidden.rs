mod solution;
use solution::RingBuffer;
use std::collections::VecDeque;
fn main() {
    let mut seed: u64 = 12345;
    let mut rnd = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as usize };
    for cap in 1..=6 {
        let mut rb = RingBuffer::new(cap);
        let mut m: VecDeque<String> = VecDeque::new();
        assert_eq!(rb.capacity(), cap);
        assert!(rb.is_empty() && rb.pop().is_none() && rb.get(0).is_none());
        for step in 0..20000 {
            if rnd() % 10 < 6 {
                let v = format!("v{}", step);
                let want = if m.len() == cap { m.pop_front() } else { None };
                m.push_back(v.clone());
                assert_eq!(rb.push(v), want, "push, capacity {} step {}", cap, step);
            } else {
                assert_eq!(rb.pop(), m.pop_front(), "pop, capacity {} step {}", cap, step);
            }
            assert_eq!(rb.len(), m.len());
            assert_eq!(rb.capacity(), cap);
            assert_eq!(rb.is_empty(), m.is_empty());
            let i = rnd() % (cap + 2);
            assert_eq!(rb.get(i), m.get(i), "get({}), capacity {} step {}", i, cap, step);
            if step % 7 == 0 {
                let got: Vec<&String> = rb.iter().collect();
                let want: Vec<&String> = m.iter().collect();
                assert_eq!(got, want, "iter, capacity {} step {}", cap, step);
            }
        }
    }
    let mut r = RingBuffer::new(3);
    for k in 0..5 { r.push(k); }
    assert_eq!(r.iter().copied().collect::<Vec<_>>(), vec![2, 3, 4]);
    assert_eq!((r.get(0), r.get(2), r.get(3)), (Some(&2), Some(&4), None));
    assert!(std::panic::catch_unwind(|| RingBuffer::<i32>::new(0)).is_err(), "new(0) must panic");
    println!("{}", std::env::args().nth(1).unwrap_or_default());
}
