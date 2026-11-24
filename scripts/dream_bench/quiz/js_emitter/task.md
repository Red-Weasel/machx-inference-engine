Write `solution.mjs` (an ES module) exporting a class `Emitter`:

- `on(event, fn)`: register a listener; returns an unsubscribe function (calling it more than once is harmless). The same function may be registered several times; each registration is separate.
- `once(event, fn)`: like `on`, but the registration is removed right before its first call (so an `emit` from inside that listener does not call it again). Also returns an unsubscribe function.
- `off(event, fn)`: remove the earliest still-active registration of `fn` for that event; returns `true` if one was removed, `false` otherwise.
- `emit(event, ...args)`: call the listeners registered for that event in registration order, with `this` set to the emitter and the given arguments; return `true` if there were listeners, `false` otherwise.
  - Listeners added during an `emit` are not called by that same `emit`. Listeners removed during an `emit` that have not been called yet are not called.
  - If a listener throws, the remaining listeners still run; afterwards `emit` rethrows the first error thrown.
  - Emitting `"error"` with no listeners throws: the first argument if it is an `Error`, otherwise a new `Error`.
- `listenerCount(event)`: the number of active registrations.
- `on` / `once` with a non-function throw a `TypeError`.
