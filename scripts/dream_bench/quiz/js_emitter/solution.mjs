export class Emitter {
  #m = new Map();

  on(event, fn) { return this.#add(event, fn, false); }
  once(event, fn) { return this.#add(event, fn, true); }

  off(event, fn) {
    const l = this.#m.get(event);
    const rec = l && l.find((r) => r.fn === fn);
    if (!rec) return false;
    this.#remove(event, rec);
    return true;
  }

  emit(event, ...args) {
    const l = this.#m.get(event);
    if (!l || l.length === 0) {
      if (event === 'error') throw (args[0] instanceof Error ? args[0] : new Error('Unhandled error event'));
      return false;
    }
    let first, threw = false;
    for (const rec of [...l]) {
      if (rec.removed) continue;
      if (rec.once) this.#remove(event, rec);
      try { rec.fn.apply(this, args); } catch (e) { if (!threw) { threw = true; first = e; } }
    }
    if (threw) throw first;
    return true;
  }

  listenerCount(event) { return (this.#m.get(event) || []).length; }

  #add(event, fn, once) {
    if (typeof fn !== 'function') throw new TypeError('listener must be a function');
    const rec = { fn, once, removed: false };
    let l = this.#m.get(event);
    if (!l) { l = []; this.#m.set(event, l); }
    l.push(rec);
    return () => this.#remove(event, rec);
  }

  #remove(event, rec) {
    if (rec.removed) return;
    rec.removed = true;
    const l = this.#m.get(event);
    const i = l.indexOf(rec);
    if (i >= 0) l.splice(i, 1);
    if (l.length === 0) this.#m.delete(event);
  }
}
