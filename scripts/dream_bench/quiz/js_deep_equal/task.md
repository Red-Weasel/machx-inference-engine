Write `solution.mjs` (an ES module) exporting `deepEqual(a, b)`, a structural equality check:

- Primitives: equal if `Object.is` says so, except that `+0` and `-0` are equal. So `NaN` equals `NaN`.
- Functions are equal only if they are the same function.
- Two objects must have the same prototype (`Object.getPrototypeOf`), otherwise they differ (`{}` vs `Object.create(null)`, an array vs a plain object, a `Date` vs a plain object).
- Arrays: same length and equal elements at each index (a hole reads as `undefined`, so `[1, , 3]` equals `[1, undefined, 3]`).
- `Date`: equal time values (two invalid dates are equal). `RegExp`: same `source` and `flags`.
- `Map`: same size, and for every key of `a`, `b` has that same key (keys matched by identity, as `Map.has` does) with a deep-equal value. `Set`: same size and every element of `a` is in `b` (identity, as `Set.has` does).
- Other objects: the same own enumerable string keys (order does not matter) with deep-equal values. Symbol keys are ignored.
- Cyclic structures must not recurse forever: while a pair (a, b) is already being compared further up, treat it as equal.
