#!/usr/bin/env python3
"""
viz.py — self-contained profile visualization for the SYCL inference-engine perf harness.

Reads a "capture" JSON (produced by the sibling roofline.py) and renders a single,
fully self-contained HTML file (inline SVG, inline CSS, no JS, no network/CDN refs).

Usage:
    python3 tools/perf/viz.py capture.json --out profile.html

Pure Python 3 standard library only — no pip / numpy / matplotlib.

Capture JSON schema (do NOT change — owned by roofline.py):
{
  "model": "Qwen3.6-27B-Q4_K_M", "phase": "decode", "device": "B70",
  "peak_bw_gbs": 608, "total_ms": 67.63,
  "run_metrics": {"xve_stall_pct": 84.9, "occupancy_pct": 27.6, "l3_bw_bound_pct": 0.0},
  "kernels": [
    {"name":"gemv_q4k","total_ms":34.91,"pct":51.6,"calls":248,
     "ms_per_call":0.1408,"bound":"latency","why":"..."}, ...
  ]
}
"""

import argparse
import html
import json
import sys


# Fixed bound-class -> color legend (kept in sync with the bar fills).
BOUND_COLORS = {
    "bandwidth": "#2f6fd6",   # blue
    "compute":   "#e8862a",   # orange
    "latency":   "#d6453b",   # red
    "launch":    "#7d4bbf",   # purple
    "unknown":   "#8a8f98",   # gray
}
DEFAULT_COLOR = BOUND_COLORS["unknown"]


def esc(value):
    """HTML/SVG-safe text (escapes &, <, >, and quotes)."""
    return html.escape("" if value is None else str(value), quote=True)


def color_for(bound):
    return BOUND_COLORS.get((bound or "unknown").lower(), DEFAULT_COLOR)


def fmt_ms(value):
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "?"


def fmt_pct(value):
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "?"


def load_capture(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("capture JSON top-level must be an object")
    data.setdefault("kernels", [])
    data.setdefault("run_metrics", {})
    return data


def kernel_weight(kern):
    """Sort/width key: prefer pct, fall back to total_ms."""
    for key in ("pct", "total_ms"):
        try:
            return float(kern.get(key))
        except (TypeError, ValueError):
            continue
    return 0.0


def pareto_count(kernels, threshold=80.0):
    """How many of the top kernels (by weight) cumulatively reach `threshold`% of time.

    Returns (count_for_threshold, total_kernels, cumulative_pct_at_count).
    Works off pct if present, otherwise off a total_ms-derived share.
    """
    total = len(kernels)
    if total == 0:
        return 0, 0, 0.0

    weights = [kernel_weight(k) for k in kernels]
    wsum = sum(weights)
    if wsum <= 0:
        return total, total, 100.0

    # kernels are already sorted descending by weight before this is called.
    cumulative = 0.0
    for idx, w in enumerate(weights, start=1):
        cumulative += w / wsum * 100.0
        if cumulative >= threshold:
            return idx, total, cumulative
    return total, total, cumulative


# ----------------------------------------------------------------------------- SVG / HTML


def gauge_svg(label, pct, color):
    """A compact horizontal gauge (0-100%) with a labeled fill."""
    pct_clamped = max(0.0, min(100.0, float(pct) if pct is not None else 0.0))
    width = 240
    height = 18
    fill_w = round(width * pct_clamped / 100.0, 2)
    return (
        f'<div class="gauge">'
        f'<div class="gauge-label">{esc(label)}'
        f'<span class="gauge-val">{fmt_pct(pct)}%</span></div>'
        f'<svg width="{width}" height="{height}" role="img" '
        f'aria-label="{esc(label)} {fmt_pct(pct)} percent">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="4" class="gauge-bg"/>'
        f'<rect x="0" y="0" width="{fill_w}" height="{height}" rx="4" fill="{color}"/>'
        f'</svg></div>'
    )


def metrics_block(run_metrics):
    """Render known run_metrics as gauges; tolerate missing/extra keys."""
    specs = [
        ("occupancy_pct", "Occupancy", BOUND_COLORS["bandwidth"]),
        ("xve_stall_pct", "XVE stall", BOUND_COLORS["latency"]),
        ("l3_bw_bound_pct", "L3-BW bound", BOUND_COLORS["compute"]),
    ]
    out = []
    seen = set()
    for key, label, color in specs:
        if key in run_metrics:
            out.append(gauge_svg(label, run_metrics.get(key), color))
            seen.add(key)
    # Any other numeric metrics roofline.py may add later.
    for key, val in run_metrics.items():
        if key in seen:
            continue
        if isinstance(val, (int, float)):
            out.append(gauge_svg(key, val, DEFAULT_COLOR))
    if not out:
        return '<div class="muted">no run_metrics</div>'
    return "".join(out)


def bars_svg(kernels):
    """Horizontal bar chart: one bar per kernel, width ∝ weight (pct or ms)."""
    if not kernels:
        return '<div class="muted">no kernels in capture</div>'

    weights = [kernel_weight(k) for k in kernels]
    wmax = max(weights) if weights else 0.0
    if wmax <= 0:
        wmax = 1.0

    row_h = 30
    gap = 8
    label_w = 220     # left gutter for kernel name
    bar_max_w = 560   # max drawable bar width
    right_pad = 130   # room for the "pct / ms" caption
    chart_w = label_w + bar_max_w + right_pad
    chart_h = len(kernels) * (row_h + gap)

    rows = []
    for i, kern in enumerate(kernels):
        y = i * (row_h + gap)
        cy = y + row_h / 2
        name = kern.get("name", "?")
        bound = (kern.get("bound") or "unknown").lower()
        color = color_for(bound)
        weight = weights[i]
        bar_w = round(bar_max_w * weight / wmax, 2)
        bar_w = max(bar_w, 1.0)

        pct = kern.get("pct")
        tot_ms = kern.get("total_ms")
        calls = kern.get("calls")
        mspc = kern.get("ms_per_call")
        why = kern.get("why", "")

        tip = (
            f"{name} [{bound}]\n"
            f"pct: {fmt_pct(pct)}%   total: {fmt_ms(tot_ms)} ms\n"
            f"calls: {calls}   ms/call: {mspc}\n"
            f"{why}"
        )
        caption = f"{fmt_pct(pct)}% · {fmt_ms(tot_ms)} ms"

        rows.append(
            f'<g class="krow">'
            f'<title>{esc(tip)}</title>'
            # name (truncates visually via CSS-less fixed gutter; SVG text just clips)
            f'<text x="{label_w - 8}" y="{cy}" class="kname" '
            f'text-anchor="end" dominant-baseline="middle">{esc(name)}</text>'
            # track + bar
            f'<rect x="{label_w}" y="{y + 4}" width="{bar_max_w}" height="{row_h - 8}" '
            f'rx="3" class="track"/>'
            f'<rect x="{label_w}" y="{y + 4}" width="{bar_w}" height="{row_h - 8}" '
            f'rx="3" fill="{color}"/>'
            # caption to the right of the bar
            f'<text x="{label_w + bar_max_w + 8}" y="{cy}" class="kcap" '
            f'dominant-baseline="middle">{esc(caption)}</text>'
            f'</g>'
        )

    return (
        f'<svg class="barchart" width="{chart_w}" height="{chart_h}" '
        f'viewBox="0 0 {chart_w} {chart_h}" role="img" '
        f'aria-label="per-kernel time breakdown">'
        + "".join(rows)
        + "</svg>"
    )


def legend_html():
    items = []
    for bound, color in BOUND_COLORS.items():
        items.append(
            f'<span class="legend-item">'
            f'<span class="swatch" style="background:{color}"></span>{esc(bound)}'
            f'</span>'
        )
    return '<div class="legend">' + "".join(items) + "</div>"


PAGE_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 28px 32px;
  background: #14171c; color: #e6e9ef;
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .06em;
     color: #9aa3b2; margin: 28px 0 12px; }
.muted { color: #7d8594; }
.card { background: #1b1f26; border: 1px solid #262b34; border-radius: 10px;
        padding: 18px 20px; }
.sub { color: #9aa3b2; font-size: 13px; }
.sub b { color: #e6e9ef; }
.headgrid { display: flex; flex-wrap: wrap; gap: 28px; align-items: flex-start; }
.metrics { display: flex; flex-wrap: wrap; gap: 18px 28px; }
.gauge { min-width: 240px; }
.gauge-label { font-size: 12px; color: #c2c8d2; margin-bottom: 4px;
               display: flex; justify-content: space-between; }
.gauge-val { color: #e6e9ef; font-variant-numeric: tabular-nums; }
.gauge-bg { fill: #2a2f38; }
.pareto { font-size: 15px; }
.pareto b { color: #ffce6b; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 10px 0 14px;
          font-size: 12px; color: #c2c8d2; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.barchart { max-width: 100%; }
.barchart .kname { fill: #d4d9e2; font-size: 12.5px; font-variant-numeric: tabular-nums; }
.barchart .kcap  { fill: #9aa3b2; font-size: 12px; font-variant-numeric: tabular-nums; }
.barchart .track { fill: #20252e; }
.barchart .krow:hover .track { fill: #283040; }
.barchart .krow { cursor: default; }
.foot { margin-top: 28px; color: #6c7480; font-size: 11px; }
"""


def render_html(cap):
    model = cap.get("model", "?")
    phase = cap.get("phase", "?")
    device = cap.get("device", "?")
    total_ms = cap.get("total_ms")
    peak_bw = cap.get("peak_bw_gbs")
    run_metrics = cap.get("run_metrics", {}) or {}

    # Sort kernels descending by weight (pct, else total_ms).
    kernels = sorted(cap.get("kernels", []), key=kernel_weight, reverse=True)

    p_count, p_total, p_cum = pareto_count(kernels, 80.0)
    pareto_txt = (
        f'<b>{p_count}</b> of <b>{p_total}</b> kernels = '
        f'{fmt_pct(p_cum)}% of the time (≥80% threshold)'
        if p_total else "no kernels"
    )

    head = (
        f'<div class="card">'
        f'<h1>{esc(model)}</h1>'
        f'<div class="sub"><b>{esc(phase)}</b> · device <b>{esc(device)}</b> · '
        f'total <b>{fmt_ms(total_ms)} ms</b>'
        + (f' · peak BW <b>{esc(peak_bw)} GB/s</b>' if peak_bw is not None else '')
        + '</div>'
        f'<h2>Run metrics</h2>'
        f'<div class="metrics">{metrics_block(run_metrics)}</div>'
        f'</div>'
    )

    pareto = (
        f'<div class="card" style="margin-top:18px"><div class="pareto">'
        f'Pareto: {pareto_txt}</div></div>'
    )

    chart = (
        f'<h2>Where the time goes</h2>'
        + legend_html()
        + f'<div class="card">{bars_svg(kernels)}</div>'
    )

    foot = (
        '<div class="foot">Generated by tools/perf/viz.py — self-contained '
        '(inline SVG, no JS, no network). Hover a bar for calls / ms-per-call / why.</div>'
    )

    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{esc(model)} — {esc(phase)} profile</title>'
        f'<style>{PAGE_CSS}</style></head><body>'
        f'{head}{pareto}{chart}{foot}'
        '</body></html>\n'
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render a self-contained HTML profile from a capture JSON."
    )
    parser.add_argument("capture", help="path to the capture JSON (from roofline.py)")
    parser.add_argument("--out", default="profile.html",
                        help="output HTML path (default: profile.html)")
    args = parser.parse_args(argv)

    try:
        cap = load_capture(args.capture)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"viz.py: failed to read capture '{args.capture}': {exc}", file=sys.stderr)
        return 1

    page = render_html(cap)
    try:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(page)
    except OSError as exc:
        print(f"viz.py: failed to write '{args.out}': {exc}", file=sys.stderr)
        return 1

    n = len(cap.get("kernels", []))
    print(f"viz.py: wrote {args.out} ({n} kernels, {len(page)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
