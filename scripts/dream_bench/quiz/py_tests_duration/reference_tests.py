from solution import parse_duration

ok = {"1h30m": 5400, "2d": 172800, "0s": 0, "1d1s": 86401, "01h": 3600, "45m": 2700, "1d2h3m4s": 93784, "90s": 90}
for s, want in ok.items():
    assert parse_duration(s) == want, (s, parse_duration(s), want)
for bad in ["", "5x", "30m1h", "1h1h", "1H", "1h 30m", "-1h", "+1h", "1h30", "h", "1.5h", " 1h"]:
    try:
        parse_duration(bad)
    except ValueError:
        continue
    raise AssertionError(f"accepted {bad!r}")
print("OK")
