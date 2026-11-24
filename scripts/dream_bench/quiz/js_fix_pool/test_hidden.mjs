import assert from 'node:assert/strict';
import { runLimited } from './solution.mjs';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// order, once-each, concurrency cap
{
  let active = 0, peak = 0; const calls = new Array(10).fill(0);
  const delays = [30, 5, 20, 1, 15, 2, 25, 3, 10, 4];
  const tasks = delays.map((d, i) => async () => { calls[i]++; active++; peak = Math.max(peak, active); await sleep(d); active--; return i * 10; });
  const res = await runLimited(tasks, 3);
  assert.deepEqual(res, delays.map((_, i) => i * 10));
  assert.deepEqual(calls, new Array(10).fill(1), 'each task exactly once');
  assert.ok(peak <= 3 && peak >= 2, `peak concurrency ${peak}`);
}
// limit larger than tasks; empty
assert.deepEqual(await runLimited([async () => 1, async () => 2], 10), [1, 2]);
assert.deepEqual(await runLimited([], 2), []);
// failure: rejects with the error, nothing new starts afterwards
{
  const started = [];
  const boom = new Error('boom');
  const tasks = Array.from({ length: 8 }, (_, i) => async () => {
    started.push(i); await sleep(i === 1 ? 5 : 40);
    if (i === 1) throw boom; return i;
  });
  await assert.rejects(runLimited(tasks, 2), (e) => e === boom);
  await sleep(120);
  assert.deepEqual(started.sort((a, b) => a - b), [0, 1], `started after failure: ${started}`);
}
// first failure wins
{
  const e1 = new Error('first'), e2 = new Error('second');
  const tasks = [async () => { await sleep(30); throw e2; }, async () => { await sleep(5); throw e1; }];
  await assert.rejects(runLimited(tasks, 2), (e) => e === e1);
}
for (const bad of [0, -1, 1.5, NaN, '2'])
  await assert.rejects(runLimited([async () => 1], bad), RangeError, String(bad));
console.log('OK');
