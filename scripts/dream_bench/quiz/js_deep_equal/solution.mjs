export function deepEqual(a, b) {
  return eq(a, b, new Map());
}

function eq(a, b, seen) {
  if (Object.is(a, b) || (a === 0 && b === 0)) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false;
  if (Object.getPrototypeOf(a) !== Object.getPrototypeOf(b)) return false;
  let s = seen.get(a);
  if (s && s.has(b)) return true;
  if (!s) { s = new Set(); seen.set(a, s); }
  s.add(b);
  try {
    if (a instanceof Date) return Object.is(a.getTime(), b.getTime());
    if (a instanceof RegExp) return a.source === b.source && a.flags === b.flags;
    if (a instanceof Map) {
      if (a.size !== b.size) return false;
      for (const [k, v] of a) if (!b.has(k) || !eq(v, b.get(k), seen)) return false;
      return true;
    }
    if (a instanceof Set) {
      if (a.size !== b.size) return false;
      for (const v of a) if (!b.has(v)) return false;
      return true;
    }
    if (Array.isArray(a)) {
      if (a.length !== b.length) return false;
      for (let i = 0; i < a.length; i++) if (!eq(a[i], b[i], seen)) return false;
      return true;
    }
    const ka = Object.keys(a), kb = Object.keys(b);
    if (ka.length !== kb.length) return false;
    const kbs = new Set(kb);
    for (const k of ka) if (!kbs.has(k) || !eq(a[k], b[k], seen)) return false;
    return true;
  } finally {
    s.delete(b);
  }
}
