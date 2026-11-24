from decimal import Decimal
from solution import Money, CurrencyMismatch

M = Money
assert M("12.5", "USD").amount == Decimal("12.50") and str(M("12.5", "USD")) == "12.50 USD" and str(M(3, "EUR")) == "3.00 EUR"
assert M("0.125", "USD").amount == Decimal("0.12") and M("0.135", "USD").amount == Decimal("0.14") and M(Decimal("-0.125"), "USD").amount == Decimal("-0.12")
assert str(M("0.1", "USD").amount) == "0.10"
assert M("0.014999999999999999999999999999", "USD").amount == Decimal("0.01") and M(10**26, "USD").amount == Decimal(10**26)
try: M([1], "USD"); raise AssertionError("list accepted")
except (TypeError, ValueError): pass
for a, exc in ((1.5, TypeError), (True, TypeError), ("abc", ValueError), ("NaN", ValueError), ("Infinity", ValueError)):
    try: M(a, "USD"); raise AssertionError(f"accepted {a!r}")
    except exc: pass
for c in ("usd", "US", "USDX", "U$D", 840):
    try: M(1, c); raise AssertionError(f"currency {c!r}")
    except ValueError: pass
m = M("1.00", "USD")
for fn in (lambda: setattr(m, "x", 1), lambda: setattr(m, "_cents", 5), lambda: delattr(m, "_cur")):
    try: fn(); raise AssertionError("mutable")
    except AttributeError: pass
assert m.amount == Decimal("1.00")
assert M("1", "USD") == M("1.00", "USD") and M("1", "USD") != M("1", "EUR") and M("1", "USD") != Decimal("1.00") and M("1", "USD") != 1
assert len({M("1", "USD"), M("1.00", "USD"), M("1", "EUR")}) == 2
assert M("1.10", "USD") + M("2.25", "USD") == M("3.35", "USD") and M(1, "USD") - M("0.01", "USD") == M("0.99", "USD") and -M(2, "USD") == M(-2, "USD")
assert M(1, "USD") < M(2, "USD") <= M(2, "USD") and M(3, "USD") > M(2, "USD") >= M(2, "USD")
for fn in (lambda: M(1, "USD") + M(1, "EUR"), lambda: M(1, "USD") - M(1, "EUR"), lambda: M(1, "USD") < M(1, "EUR")):
    try: fn(); raise AssertionError("mismatch allowed")
    except CurrencyMismatch as e: assert isinstance(e, ValueError)
assert M("0.05", "USD") * 3 == M("0.15", "USD") and 3 * M("0.05", "USD") == M("0.15", "USD")
assert M("1.00", "USD") * Decimal("0.125") == M("0.12", "USD") and M("1.00", "USD") * Decimal("0.135") == M("0.14", "USD")
try: M(1, "USD") * 1.5; raise AssertionError("float mul")
except TypeError: pass
al = lambda a, r: [str(x.amount) for x in M(a, "USD").allocate(r)]
assert al("1.00", [1, 1, 1]) == ["0.34", "0.33", "0.33"] and al("0.05", [3, 7]) == ["0.02", "0.03"]
assert al("1.00", [0, 1, 1]) == ["0.00", "0.50", "0.50"] and al("0.03", [0, 1, 1]) == ["0.00", "0.02", "0.01"]
assert al("-1.00", [1, 1, 1]) == ["-0.34", "-0.33", "-0.33"] and al("10", [1]) == ["10.00"]
for r in ([], [0, 0], [1, -1]):
    try: M(1, "USD").allocate(r); raise AssertionError(f"ratios {r}")
    except ValueError: pass
# (2026-11-21) a float ratio: the task says "int ratios" and elsewhere "a float raises TypeError": either error is right
try: M(1, "USD").allocate([1.5]); raise AssertionError("ratios [1.5]")
except (ValueError, TypeError): pass
import random
rng = random.Random(6)
for _ in range(2000):
    cents = rng.randrange(-100000, 100000); rat = [rng.randrange(0, 5) for _ in range(rng.randrange(1, 6))]
    if sum(rat) == 0: rat[0] = 1
    parts = M(Decimal(cents) / 100, "USD").allocate(rat)
    assert sum((p.amount for p in parts), Decimal(0)) == Decimal(cents) / 100
    for p, r in zip(parts, rat):
        if r == 0: assert p.amount == 0
print("OK")
