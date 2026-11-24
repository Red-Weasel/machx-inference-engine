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
