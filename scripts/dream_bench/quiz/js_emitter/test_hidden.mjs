import assert from 'node:assert/strict';
import { Emitter } from './solution.mjs';
let e = new Emitter(); const log = [];
assert.equal(e.emit('x'), false);
const f = function (a, b) { log.push(['f', a, b, this === e]); };
e.on('x', f); e.on('x', f); assert.equal(e.listenerCount('x'), 2);
assert.equal(e.emit('x', 1, 2), true); assert.deepEqual(log, [['f', 1, 2, true], ['f', 1, 2, true]]);
assert.equal(e.off('x', f), true); assert.equal(e.listenerCount('x'), 1); assert.equal(e.off('x', f), true); assert.equal(e.off('x', f), false);
assert.equal(e.off('nope', f), false);
// unsubscribe twice harmless, and removes only its own registration
log.length = 0; const u1 = e.on('y', () => log.push(1)); e.on('y', () => log.push(2)); u1(); u1(); e.emit('y'); assert.deepEqual(log, [2]);
// off removes the earliest active registration of fn
log.length = 0; e = new Emitter(); const g = () => log.push('g'); const u = e.on('z', g); e.once('z', g); u(); e.emit('z'); e.emit('z'); assert.deepEqual(log, ['g']);
// once: removed before call, re-entrant emit does not call it again
log.length = 0; e = new Emitter(); e.once('a', () => { log.push('once'); e.emit('a'); }); e.on('a', () => log.push('on'));
e.emit('a'); assert.deepEqual(log, ['once', 'on', 'on'], 'nested emit from inside a once listener runs the other listeners');
assert.equal(e.listenerCount('a'), 1);
// off removes the EARLIEST registration: f, g, f -> g, f
log.length = 0; e = new Emitter(); const ff = () => log.push('f'), gg = () => log.push('g');
e.on('o', ff); e.on('o', gg); e.on('o', ff); e.off('o', ff); e.emit('o'); assert.deepEqual(log, ['g', 'f']);
// added during emit: not called; removed during emit: not called
log.length = 0; e = new Emitter(); let late; const second = () => log.push('second');
e.on('b', () => { log.push('first'); e.on('b', (late = () => log.push('late'))); e.off('b', second); }); e.on('b', second);
e.emit('b'); assert.deepEqual(log, ['first']); log.length = 0; e.emit('b'); assert.deepEqual(log, ['first', 'late']);
// errors: all listeners run, first error rethrown
log.length = 0; e = new Emitter(); const e1 = new Error('one'), e2 = new Error('two');
e.on('c', () => { throw e1; }); e.on('c', () => log.push('ran')); e.on('c', () => { throw e2; });
assert.throws(() => e.emit('c'), (x) => x === e1); assert.deepEqual(log, ['ran']);
// error event
e = new Emitter(); const boom = new Error('boom'); assert.throws(() => e.emit('error', boom), (x) => x === boom);
assert.throws(() => e.emit('error', 'str'), Error); let got; e.on('error', (x) => (got = x)); assert.equal(e.emit('error', boom), true); assert.equal(got, boom);
// once with unsubscribe before firing
log.length = 0; e = new Emitter(); const uo = e.once('d', () => log.push('d')); uo(); assert.equal(e.emit('d'), false); assert.deepEqual(log, []);
assert.throws(() => e.on('x', 5), TypeError); assert.throws(() => e.once('x', null), TypeError);
console.log('OK');
