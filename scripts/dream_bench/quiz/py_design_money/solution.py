import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext


class CurrencyMismatch(ValueError):
    pass


_CENT = Decimal("0.01")


class Money:
    __slots__ = ("_cents", "_cur")

    def __init__(self, amount, currency):
        if isinstance(amount, (bool, float)):
            raise TypeError("amount must be int, str or Decimal, not " + type(amount).__name__)
        if isinstance(amount, int):
            d = Decimal(amount)
        elif isinstance(amount, (str, Decimal)):
            try:
                d = Decimal(amount)
            except InvalidOperation:
                raise ValueError(f"invalid amount {amount!r}") from None
        else:
            raise TypeError("amount must be int, str or Decimal, not " + type(amount).__name__)
        if not d.is_finite():
            raise ValueError("amount must be finite")
        if not (isinstance(currency, str) and re.fullmatch(r"[A-Z]{3}", currency)):
            raise ValueError(f"invalid currency {currency!r}")
        with localcontext() as ctx:
            ctx.prec = max(50, len(d.as_tuple().digits) + 10)   # exact: no rounding before the cent rounding
            cents = int((d * 100).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
        object.__setattr__(self, "_cents", cents)
        object.__setattr__(self, "_cur", currency)

    @classmethod
    def _of(cls, cents, cur):
        m = object.__new__(cls)
        object.__setattr__(m, "_cents", cents)
        object.__setattr__(m, "_cur", cur)
        return m

    def __setattr__(self, k, v):
        raise AttributeError("Money is immutable")

    def __delattr__(self, k):
        raise AttributeError("Money is immutable")

    @property
    def amount(self):
        with localcontext() as ctx:
            ctx.prec = max(50, len(str(abs(self._cents))) + 10)
            return (Decimal(self._cents) / 100).quantize(_CENT)

    @property
    def currency(self):
        return self._cur

    def _same(self, other):
        if not isinstance(other, Money):
            return False
        if other._cur != self._cur:
            raise CurrencyMismatch(f"{self._cur} vs {other._cur}")
        return True

    def __eq__(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        return self._cur == other._cur and self._cents == other._cents

    def __hash__(self):
        return hash((self._cur, self._cents))

    def __add__(self, other):
        if not self._same(other):
            return NotImplemented
        return Money._of(self._cents + other._cents, self._cur)

    def __sub__(self, other):
        if not self._same(other):
            return NotImplemented
        return Money._of(self._cents - other._cents, self._cur)

    def __neg__(self):
        return Money._of(-self._cents, self._cur)

    def __lt__(self, other):
        return self._cents < other._cents if self._same(other) else NotImplemented

    def __le__(self, other):
        return self._cents <= other._cents if self._same(other) else NotImplemented

    def __gt__(self, other):
        return self._cents > other._cents if self._same(other) else NotImplemented

    def __ge__(self, other):
        return self._cents >= other._cents if self._same(other) else NotImplemented

    def __mul__(self, k):
        if isinstance(k, (bool, float)) or not isinstance(k, (int, Decimal)):
            if isinstance(k, float):
                raise TypeError("cannot multiply Money by a float")
            return NotImplemented
        with localcontext() as ctx:
            ctx.prec = 100
            cents = int((Decimal(self._cents) * Decimal(k)).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
        return Money._of(cents, self._cur)

    __rmul__ = __mul__

    def allocate(self, ratios):
        ratios = list(ratios)
        if not ratios or any(type(r) is not int or r < 0 for r in ratios) or sum(ratios) == 0:
            raise ValueError("ratios must be non-negative ints with a positive sum")
        sign = -1 if self._cents < 0 else 1
        total = abs(self._cents)
        s = sum(ratios)
        parts = [total * r // s for r in ratios]
        left = total - sum(parts)
        for i, r in enumerate(ratios):
            if left == 0:
                break
            if r > 0:
                parts[i] += 1
                left -= 1
        return [Money._of(sign * p, self._cur) for p in parts]

    def __str__(self):
        return f"{self.amount} {self._cur}"

    def __repr__(self):
        return f"Money('{self.amount}', '{self._cur}')"
