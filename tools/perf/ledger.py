#!/usr/bin/env python3
"""Performance ledger for the SYCL inference-engine optimization loop.

Append-only record of every kernel/optimization VARIANT we test, so optimization
is data-driven not vibes. Every entry captures whether the change was a real win
AND whether it held correctness.

HARD RULE (this engine): every change must hold perplexity. A "faster" kernel that
moves PPL past the gate, or regresses PPL beyond tolerance, is REJECTED.

Pure Python 3 standard library only. No third-party deps.

Verbs:
  add      append a row to PERF_LEDGER.md (computes speedup + accepted)
  summary  print all entries + aggregate stats (accepted count, best win/target)
  check    print ACCEPT/REJECT + reason, exit 0 (accept) / 1 (reject)

Acceptance gate:
  ACCEPT iff speedup > 1.0
            AND ppl_variant <= --ppl-gate (default 6.57)
            AND ppl_variant does not regress more than --ppl-tol (default 0.5%)
              over ppl_baseline (i.e. (ppl_variant - ppl_baseline)/ppl_baseline <= tol).
  An improvement in PPL (negative delta) never fails the gate.
"""

import argparse
import os
import sys

LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PERF_LEDGER.md")

DEFAULT_PPL_GATE = 6.57
DEFAULT_PPL_TOL = 0.5  # percent

# Markdown table column order. Keep in sync with HEADER below and parse_rows().
COLUMNS = [
    "date",
    "model",
    "phase",
    "target",
    "variant",
    "baseline",
    "variant_metric",
    "speedup",
    "ppl_baseline",
    "ppl_variant",
    "ppl_delta_pct",
    "accepted",
    "commit",
    "notes",
]

TABLE_HEADER = (
    "| date | model | phase | target | variant | baseline | variant | speedup "
    "| ppl_base | ppl_var | ppl_delta% | accepted | commit | notes |"
)
TABLE_SEP = (
    "|------|-------|-------|--------|---------|---------:|--------:|--------:"
    "|---------:|--------:|-----------:|:--------:|--------|-------|"
)

HEADER = """# Performance Ledger

Append-only record of every kernel/optimization **variant** we test on the SYCL
inference engine. The goal is data-driven optimization: every entry says whether
the change was a real speed **win** and whether it held **correctness**.

## Acceptance rule (the gate)

This engine has a hard rule: **every change must hold perplexity** (crown PPL
6.4527 / general gate **PPL ≤ 6.57**). A "faster" kernel that moves PPL is
**REJECTED**.

A variant is **ACCEPTED** only when ALL hold:

1. **Faster** — `speedup = variant_metric / baseline_metric > 1.0`.
2. **Under the gate** — `ppl_variant ≤ 6.57` (configurable via `--ppl-gate`).
3. **No PPL regression** — `(ppl_variant - ppl_baseline) / ppl_baseline ≤ 0.5%`
   (configurable via `--ppl-tol`). A PPL *improvement* (negative delta) never fails.

Otherwise the variant is **REJECTED** (worse/parity speed, or PPL moved).

## How to use

```
# Record a measured A/B (timestamp passed in — reproducible, no wall-clock call):
python3 tools/perf/ledger.py add --date "2026-06-22 14:30" \\
    --model 27b --phase decode --target gemv_q4k \\
    --variant "SoA int-dot GMEM" --baseline 15.55 --variant-metric 15.85 \\
    --ppl-baseline 5.60 --ppl-variant 5.61 --notes "occupancy-bound; parity" \\
    [--commit 6a3e752]

# See everything + aggregate stats:
python3 tools/perf/ledger.py summary

# Gate a single A/B in a script (exit 0 = accept, 1 = reject):
python3 tools/perf/ledger.py check --baseline 15.55 --variant 15.85 \\
    --ppl-baseline 5.60 --ppl-variant 5.61
```

`metric` is throughput (tok/s) — higher is better. `speedup` and `accepted` are
computed; never hand-edit them.

## Ledger

"""


def fmt_num(x, places=4):
    """Trim trailing zeros for compact, stable markdown."""
    s = f"{x:.{places}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def gate(baseline, variant, ppl_baseline, ppl_variant, ppl_gate, ppl_tol):
    """Return (accepted: bool, speedup: float, ppl_delta_pct: float, reason: str)."""
    if baseline <= 0:
        return False, 0.0, 0.0, "invalid baseline metric (<= 0)"
    speedup = variant / baseline
    if ppl_baseline > 0:
        ppl_delta_pct = (ppl_variant - ppl_baseline) / ppl_baseline * 100.0
    else:
        ppl_delta_pct = 0.0

    if speedup <= 1.0:
        return (
            False,
            speedup,
            ppl_delta_pct,
            f"not faster (speedup {speedup:.3f} <= 1.0)",
        )
    if ppl_variant > ppl_gate:
        return (
            False,
            speedup,
            ppl_delta_pct,
            f"PPL {ppl_variant:.4f} exceeds gate {ppl_gate:.4f}",
        )
    # Use a small epsilon so an exact-tolerance regression (e.g. exactly 0.5%)
    # is allowed despite binary floating-point rounding.
    if ppl_delta_pct > ppl_tol + 1e-9:
        return (
            False,
            speedup,
            ppl_delta_pct,
            f"PPL regressed {ppl_delta_pct:+.3f}% > tolerance {ppl_tol:.3f}%",
        )

    reason = f"faster {speedup:.3f}x; PPL {ppl_variant:.4f} (Δ{ppl_delta_pct:+.3f}%) within gate"
    return True, speedup, ppl_delta_pct, reason


def ensure_ledger():
    if not os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "w") as f:
            f.write(HEADER)
            f.write(TABLE_HEADER + "\n")
            f.write(TABLE_SEP + "\n")


def cell(s):
    """Escape a value for a markdown table cell."""
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def cmd_add(args):
    ensure_ledger()
    accepted, speedup, ppl_delta_pct, reason = gate(
        args.baseline,
        args.variant_metric,
        args.ppl_baseline,
        args.ppl_variant,
        args.ppl_gate,
        args.ppl_tol,
    )
    row = [
        cell(args.date),
        cell(args.model),
        cell(args.phase),
        cell(args.target),
        cell(args.variant),
        fmt_num(args.baseline),
        fmt_num(args.variant_metric),
        f"{speedup:.3f}x",
        fmt_num(args.ppl_baseline),
        fmt_num(args.ppl_variant),
        f"{ppl_delta_pct:+.3f}%",
        "yes" if accepted else "no",
        cell(args.commit or ""),
        cell(args.notes or ""),
    ]
    line = "| " + " | ".join(row) + " |\n"
    with open(LEDGER_PATH, "a") as f:
        f.write(line)
    verdict = "ACCEPT" if accepted else "REJECT"
    print(f"[{verdict}] {args.model}/{args.phase}/{args.target} :: {cell(args.variant)}")
    print(f"  speedup {speedup:.3f}x | ppl {fmt_num(args.ppl_variant)} (Δ{ppl_delta_pct:+.3f}%) | {reason}")
    print(f"  appended -> {LEDGER_PATH}")


def parse_rows():
    """Parse data rows from the markdown table. Returns list of dicts."""
    rows = []
    if not os.path.exists(LEDGER_PATH):
        return rows
    with open(LEDGER_PATH) as f:
        lines = f.readlines()
    in_table = False
    for ln in lines:
        s = ln.strip()
        if not s.startswith("|"):
            continue
        # The header row and separator row.
        if s.startswith("| date "):
            in_table = True
            continue
        if set(s) <= set("|-: "):
            continue
        if not in_table:
            continue
        cells = [c.strip() for c in s.strip().strip("|").split("|")]
        if len(cells) != len(COLUMNS):
            continue
        rows.append(dict(zip(COLUMNS, cells)))
    return rows


def cmd_summary(args):
    rows = parse_rows()
    print(TABLE_HEADER)
    print(TABLE_SEP)
    for r in rows:
        print("| " + " | ".join(r[c] for c in COLUMNS) + " |")
    print()

    total = len(rows)
    accepted = [r for r in rows if r["accepted"].lower() == "yes"]
    rejected = total - len(accepted)
    print("## Aggregate stats")
    print(f"- entries: {total}")
    print(f"- accepted: {len(accepted)}")
    print(f"- rejected: {rejected}")

    # Best win per target (highest speedup among ACCEPTED entries for that target).
    best = {}
    for r in accepted:
        try:
            sp = float(r["speedup"].rstrip("x"))
        except ValueError:
            continue
        tgt = r["target"]
        if tgt not in best or sp > best[tgt][0]:
            best[tgt] = (sp, r["variant"], r["model"], r["phase"])
    if best:
        print("- best accepted win per target:")
        for tgt in sorted(best):
            sp, variant, model, phase = best[tgt]
            print(f"    {tgt} ({model}/{phase}): {sp:.3f}x  [{variant}]")
    else:
        print("- best accepted win per target: (none accepted yet)")


def cmd_check(args):
    accepted, speedup, ppl_delta_pct, reason = gate(
        args.baseline,
        args.variant,
        args.ppl_baseline,
        args.ppl_variant,
        args.ppl_gate,
        args.ppl_tol,
    )
    verdict = "ACCEPT" if accepted else "REJECT"
    print(f"{verdict}: {reason}")
    print(
        f"  speedup={speedup:.3f}x  ppl_baseline={fmt_num(args.ppl_baseline)}  "
        f"ppl_variant={fmt_num(args.ppl_variant)}  ppl_delta={ppl_delta_pct:+.3f}%  "
        f"gate<={fmt_num(args.ppl_gate)}  tol<={fmt_num(args.ppl_tol)}%"
    )
    return 0 if accepted else 1


def build_parser():
    p = argparse.ArgumentParser(
        prog="ledger.py",
        description="Append-only perf ledger for the SYCL inference-engine optimization loop.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("add", help="append a measured A/B row to PERF_LEDGER.md")
    pa.add_argument("--date", required=True, help='timestamp "YYYY-MM-DD HH:MM" (passed in, not now())')
    pa.add_argument("--model", required=True, help='e.g. "27b", "crown", "coder30b"')
    pa.add_argument("--phase", required=True, help='"decode" | "prefill"')
    pa.add_argument("--target", required=True, help='kernel/area, e.g. "gemv_q4k"')
    pa.add_argument("--variant", required=True, help='short description, e.g. "SoA int-dot GMEM"')
    pa.add_argument("--baseline", type=float, required=True, help="baseline metric (tok/s)")
    pa.add_argument("--variant-metric", type=float, required=True, help="variant metric (tok/s)")
    pa.add_argument("--ppl-baseline", type=float, required=True, help="baseline perplexity")
    pa.add_argument("--ppl-variant", type=float, required=True, help="variant perplexity")
    pa.add_argument("--notes", default="", help="free-form notes")
    pa.add_argument("--commit", default="", help="git short hash (optional)")
    pa.add_argument("--ppl-gate", type=float, default=DEFAULT_PPL_GATE, help="max allowed PPL (default 6.57)")
    pa.add_argument("--ppl-tol", type=float, default=DEFAULT_PPL_TOL, help="max PPL regression %% (default 0.5)")
    pa.set_defaults(func=cmd_add)

    ps = sub.add_parser("summary", help="print all entries + aggregate stats")
    ps.set_defaults(func=cmd_summary)

    pc = sub.add_parser("check", help="gate one A/B; exit 0 accept / 1 reject")
    pc.add_argument("--baseline", type=float, required=True, help="baseline metric (tok/s)")
    pc.add_argument("--variant", type=float, required=True, help="variant metric (tok/s)")
    pc.add_argument("--ppl-baseline", type=float, required=True, help="baseline perplexity")
    pc.add_argument("--ppl-variant", type=float, required=True, help="variant perplexity")
    pc.add_argument("--ppl-gate", type=float, default=DEFAULT_PPL_GATE, help="max allowed PPL (default 6.57)")
    pc.add_argument("--ppl-tol", type=float, default=DEFAULT_PPL_TOL, help="max PPL regression %% (default 0.5)")
    pc.set_defaults(func=cmd_check)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
