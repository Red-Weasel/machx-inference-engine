// runLimited(tasks, limit): tasks is an array of functions that each return a promise.
// Runs them with at most `limit` running at the same time and resolves to their results
// in the SAME order as `tasks`.
export async function runLimited(tasks, limit) {
  if (!Number.isInteger(limit) || limit < 1) throw new RangeError('limit must be a positive integer');
  const results = new Array(tasks.length);
  let next = 0;
  let failed = false;
  async function worker() {
    while (!failed && next < tasks.length) {
      const idx = next++;
      try {
        results[idx] = await tasks[idx]();
      } catch (e) {
        failed = true;
        throw e;
      }
    }
  }
  const n = Math.min(limit, tasks.length);
  await Promise.all(Array.from({ length: n }, worker));
  return results;
}
