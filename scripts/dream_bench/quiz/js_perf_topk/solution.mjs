export function topKWords(text, k) {
  if (typeof k !== 'number') return [];
  k = Math.floor(k);
  if (!(k > 0)) return [];
  const counts = new Map();
  for (const m of text.toLowerCase().matchAll(/[a-z]+/g)) counts.set(m[0], (counts.get(m[0]) || 0) + 1);
  const arr = [...counts];
  arr.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return arr.slice(0, k);
}
