`solution.mjs` below (an ES module) runs async tasks with a concurrency limit. Users report: some tasks run twice, results come back in the wrong order, and when one task fails, more tasks keep being started anyway.

Fix it so that:
- each task function is called exactly once; up to `limit` tasks run at the same time (it is a pool: while tasks remain, keep `limit` of them running, never more); the promise resolves to the results in the same order as `tasks`;
- if a task rejects, the returned promise rejects with that error (the first rejection to happen, if several fail), and once the pool has seen the rejection it starts no further task (tasks already running may finish);
- `limit` must be a positive integer, otherwise the returned promise rejects with a `RangeError`;
- an empty `tasks` array resolves to `[]`.

Keep the export name and signature. Write the corrected `solution.mjs`.

```javascript
// runLimited(tasks, limit): tasks is an array of functions that each return a promise.
// Runs them with at most `limit` running at the same time and resolves to their results
// in the SAME order as `tasks`.
export async function runLimited(tasks, limit) {
  const results = [];
  let i = 0;
  async function worker() {
    while (i < tasks.length) {
      const idx = i;
      results.push(await tasks[idx]());
      i++;
    }
  }
  const workers = [];
  for (let w = 0; w < limit; w++) workers.push(worker());
  await Promise.all(workers);
  return results;
}
```
