import assert from 'node:assert/strict';
import { topKWords as f } from './solution.mjs';
assert.deepEqual(f('the cat and the hat and THE bat', 2), [['the', 3], ['and', 2]]);
assert.deepEqual(f("don't stop, DON'T", 3), [['don', 2], ['t', 2], ['stop', 1]]);
assert.deepEqual(f('b a c b a c', 10), [['a', 2], ['b', 2], ['c', 2]]);
assert.deepEqual(f('x1y2x café', 5), [['x', 2], ['caf', 1], ['y', 1]]);
assert.deepEqual(f('a b', 0), []); assert.deepEqual(f('a b', -2), []); assert.deepEqual(f('', 3), []);
assert.deepEqual(f('a a b', 1.9), [['a', 2]]); assert.deepEqual(f('a', NaN), []); assert.deepEqual(f('a b a', '2'), []); assert.deepEqual(f('a', null), []);
// big
let seed = 42; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const vocab = Array.from({ length: 50000 }, (_, i) => i.toString(26).replace(/./g, (c) => String.fromCharCode(97 + parseInt(c, 26))));
const parts = []; let len = 0;
while (len < 5_000_000) { const w = vocab[Math.floor(Math.pow(rnd(), 3) * vocab.length)]; parts.push(w); len += w.length + 1; }
const text = parts.join(rnd() < 0.5 ? ' ' : ',\n');
const t0 = performance.now(); const top = f(text, 10); const dt = performance.now() - t0;
const want = new Map(); for (const w of parts) want.set(w, (want.get(w) || 0) + 1);
const exp = [...want].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, 10);
assert.deepEqual(top, exp);
assert.ok(dt < 4500, `took ${dt.toFixed(0)} ms`);
console.log('OK');
