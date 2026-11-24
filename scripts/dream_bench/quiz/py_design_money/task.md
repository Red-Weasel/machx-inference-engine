Write `solution.py` with an immutable value class `Money(amount, currency)` and an exception `CurrencyMismatch` (a subclass of `ValueError`).

- `amount` is an `int`, a `str` like `"12.5"`, or a `decimal.Decimal`. A `float` (or `bool`) raises `TypeError`; a string that is not a number, NaN or infinity raises `ValueError`. Amounts are rounded to cents with round-half-to-even (`"0.125"` -> 0.12, `"0.135"` -> 0.14).
- `currency` is three uppercase ASCII letters (`"USD"`), otherwise `ValueError`.
- Properties `amount` (a `Decimal` with exactly two decimal places) and `currency`.
- Immutable: assigning or deleting any attribute raises `AttributeError`.
- Equal when currency and amount are equal; hashable consistently with that. A `Money` never equals a non-`Money`.
- `+` and `-` between two `Money` of the same currency; different currencies raise `CurrencyMismatch`. Unary `-`. `<` `<=` `>` `>=` between the same currency (`CurrencyMismatch` otherwise).
- `*` by an `int` or `Decimal` (either side), rounded half-to-even to cents. Multiplying by a float raises `TypeError`.
- `allocate(ratios)`: split into parts proportional to a list of non-negative int ratios (at least one positive, otherwise `ValueError`). The parts must add up to exactly the original amount. Each part first gets its share rounded toward zero to whole cents, and the leftover cents go one each to the parts in order, starting from the first part with a positive ratio and skipping zero ratios. Negative amounts split the same way as the positive amount, negated. `Money("1.00", "USD").allocate([1, 1, 1])` -> 0.34, 0.33, 0.33.
- `str(Money("12.5", "USD"))` is `"12.50 USD"`.
