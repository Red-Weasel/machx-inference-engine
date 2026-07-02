#!/usr/bin/env python3
"""roofline.py -- bottleneck classifier for the SYCL inference-engine perf harness.

WHY THIS EXISTS
---------------
We just burned a build optimizing memory *coalescing* on a kernel that was
actually *occupancy* (latency) bound -- the coalescing change could never have
helped, because the XVEs were stalled waiting, not starved for bytes. This tool
fuses a per-kernel decode/prefill profile with run-level VTune gpu-hotspots
metrics and classifies each hot kernel into a bottleneck class, then names the
*matching* optimization lever. Optimize the right thing.

INPUTS
------
  --kprofile <file|->   : our kprofile text emitted by
                          `ie-bench --kprofile-decode` or
                          `ie-gemma4-gen ... profile-decode`.
                          Per-kernel columns: name total_ms pct% calls ms_per_call
                          plus a header line carrying "GPU total NN.NNN ms" and a
                          TOTAL row.
  --vtune <result_dir>  : (optional) a VTune gpu-hotspots result dir. We shell out
                          to `vtune -report summary -r <dir>` and parse RUN-LEVEL
                          metrics (this VTune version reports XVE-stall / occupancy /
                          L3-BW-bound per RUN, not per kernel):
                            "XVE Array Stalled/Idle: NN% of Elapsed ..."
                            "Occupancy: NN% of peak value"
                            "GPU L3 Bandwidth Bound: NN% of peak value"
                          The report contains multiple "Device Group" blocks (iGPU +
                          discrete B70). We attach the metrics from the block with the
                          HIGHEST occupancy -- i.e. the device actually doing the work.

OUTPUTS
-------
  - capture JSON (--out) with the exact schema the rest of the harness consumes
    (model/phase/device/peak_bw_gbs/total_ms/run_metrics/kernels[]).
  - a ranked, human-readable "levers" report to stdout.

CLASSIFICATION HEURISTIC  (run-level VTune metrics drive the call)
------------------------------------------------------------------
A roofline answers one question per kernel: is it limited by memory bytes, by
compute throughput, by latency/occupancy, or by launch overhead? On this VTune
version the architectural counters (L3-BW-bound, occupancy, XVE-stall) are
RUN-level, so they bias every hot kernel the same way; we still combine them with
the kernel's own shape (ms_per_call, calls) to peel off the launch-bound case.

  WITH VTune run metrics, evaluated top to bottom (first match wins):
    1. launch    : ms_per_call < 0.02 ms AND calls large (many tiny submits)
                   -> the cost is per-submit dispatch, not the kernel body.
    2. bandwidth : GPU L3 Bandwidth Bound > 60% of peak
                   -> saturating L3/VRAM bytes; classic GEMV-at-BW-ceiling.
    3. latency   : L3-BW-bound low AND occupancy < 40% AND XVE-stall > 60%
                   -> XVEs idle/stalled waiting (memory latency, dependency
                      chains, too few resident threads). THIS is our 27B decode:
                      not enough rows-per-subgroup to hide latency, so the array
                      stalls. Coalescing does NOTHING here.
    4. compute   : occupancy high (>=40%) and not bandwidth-bound
                   -> ALU/XMX throughput limited.
    5. unknown   : metrics present but no rule fired (rare).

  WITHOUT VTune metrics:
    every kernel is "unknown"; we still RANK by pct and flag the top-N as the
    candidates to profile with VTune next. We never guess a bound class from
    pct alone -- that guess is exactly the mistake this tool prevents.

LEVER MAP (bound class -> what to actually change)
    bandwidth -> SoA repack / coalesced wider loads / fewer bytes (lower-bit
                 quant, dedupe scales).
    latency   -> split-K, more rows-per-subgroup, async double-buffer / overlap,
                 raise occupancy (smaller regs, more WGs resident).
    compute   -> XMX/DPAS path, better register tiling, fuse epilogue.
    launch    -> kernel fusion / batch submits / persistent kernel.
    unknown   -> capture a VTune gpu-hotspots run, then re-run this tool.
"""

import argparse
import json
import os
import re
import subprocess
import sys

DEVICE = "B70"
PEAK_BW_GBS = 608          # B70 VRAM peak (sysfs-verified, docs/...landscape)
DEFAULT_TOP_N = 5          # how many hot kernels to flag when we have no VTune

# --- classification thresholds (documented above) ---
TH_L3_BW_BOUND = 60.0      # % of peak -> bandwidth-bound
TH_OCC_LOW = 40.0          # % occupancy below this contributes to latency-bound
TH_XVE_STALL = 60.0        # % stall above this contributes to latency-bound
TH_LAUNCH_MS = 0.02        # ms_per_call below this + many calls -> launch-bound
TH_LAUNCH_CALLS = 32       # "many" calls

LEVERS = {
    "bandwidth": "SoA repack / coalesced wider loads / fewer bytes (lower-bit quant, dedupe scales)",
    "latency":   "split-K + more rows-per-subgroup; async double-buffer/overlap; raise occupancy",
    "compute":   "XMX/DPAS path, better register tiling, fuse epilogue",
    "launch":    "kernel fusion / batch submits / persistent kernel",
    "unknown":   "capture a VTune gpu-hotspots run (--vtune) and re-run this tool",
}


# --------------------------------------------------------------------------
# kprofile parsing
# --------------------------------------------------------------------------
# Header line example:
#   --- DECODE kernel profile @ pos=1038 (T=1, one warm step; GPU total 67.630 ms) ---
_RE_GPU_TOTAL = re.compile(r"GPU total\s+([0-9]+\.?[0-9]*)\s*ms", re.IGNORECASE)
_RE_PHASE = re.compile(r"\b(DECODE|PREFILL)\b", re.IGNORECASE)
# Kernel row example:
#     gemv_q4k                      34.9086    51.6%     248      0.1408
_RE_ROW = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+"   # name
    r"([0-9]+\.?[0-9]*)\s+"              # total_ms
    r"([0-9]+\.?[0-9]*)%\s+"             # pct
    r"([0-9]+)\s+"                       # calls
    r"([0-9]+\.?[0-9]*)\s*$"             # ms_per_call
)


def parse_kprofile(text):
    """Return (kernels[list of dict], total_ms or None, phase or None)."""
    kernels = []
    total_ms = None
    phase = None
    for line in text.splitlines():
        if total_ms is None:
            m = _RE_GPU_TOTAL.search(line)
            if m:
                total_ms = float(m.group(1))
        if phase is None:
            mp = _RE_PHASE.search(line)
            # only trust the phase word on a header/profile line
            if mp and ("profile" in line.lower() or "kernel" in line.lower()):
                phase = mp.group(1).lower()
        stripped = line.strip()
        if stripped.upper().startswith("TOTAL"):
            # TOTAL row -- use it as a fallback for total_ms if header was absent.
            tm = re.search(r"TOTAL\s+([0-9]+\.?[0-9]*)", stripped, re.IGNORECASE)
            if tm and total_ms is None:
                total_ms = float(tm.group(1))
            continue
        m = _RE_ROW.match(line)
        if not m:
            continue
        name, t, pct, calls, mpc = m.groups()
        kernels.append({
            "name": name,
            "total_ms": round(float(t), 4),
            "pct": round(float(pct), 4),
            "calls": int(calls),
            "ms_per_call": round(float(mpc), 6),
        })
    return kernels, total_ms, phase


# --------------------------------------------------------------------------
# VTune parsing
# --------------------------------------------------------------------------
_RE_VT_STALL = re.compile(r"XVE Array Stalled/Idle:\s*([0-9]+\.?[0-9]*)%", re.IGNORECASE)
_RE_VT_OCC = re.compile(r"Occupancy:\s*([0-9]+\.?[0-9]*)%", re.IGNORECASE)
_RE_VT_L3 = re.compile(r"GPU L3 Bandwidth Bound:\s*([0-9]+\.?[0-9]*)%", re.IGNORECASE)
_RE_VT_GROUP = re.compile(r"Device Group\s*$")


def run_vtune_summary(result_dir):
    """Shell out to `vtune -report summary`. Returns stdout text or None."""
    vars_sh = "/opt/intel/oneapi/vtune/latest/vtune-vars.sh"
    if os.path.isfile(vars_sh):
        cmd = (". '%s' >/dev/null 2>&1; vtune -report summary -r '%s'"
               % (vars_sh, result_dir))
    else:
        cmd = "vtune -report summary -r '%s'" % result_dir
    try:
        proc = subprocess.run(["bash", "-lc", cmd], capture_output=True,
                              text=True, timeout=180)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        sys.stderr.write("[roofline] vtune report failed: %s\n" % e)
        return None
    if proc.returncode != 0 and not proc.stdout:
        sys.stderr.write("[roofline] vtune returned %d: %s\n"
                         % (proc.returncode, proc.stderr.strip()[:400]))
        return None
    return proc.stdout


def parse_vtune_summary(text):
    """Pick run-level metrics from the busiest Device Group block.

    The report carries one block per device group (iGPU + discrete B70). We split
    on 'Device Group' headers, read whatever metrics each block prints, then keep
    the block with the highest occupancy -- that is the device actually running
    the workload. A missing metric in the chosen block defaults to 0.0.
    """
    if not text:
        return None
    # split into blocks, each starting at a "... Device Group" header line
    lines = text.splitlines()
    blocks = []
    cur = []
    for line in lines:
        if _RE_VT_GROUP.search(line):
            if cur:
                blocks.append(cur)
            cur = [line]
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)
    if not blocks:
        # no group headers at all -- treat whole text as one block
        blocks = [lines]

    parsed = []
    for blk in blocks:
        btext = "\n".join(blk)
        stall = _RE_VT_STALL.search(btext)
        occ = _RE_VT_OCC.search(btext)
        l3 = _RE_VT_L3.search(btext)
        if stall is None and occ is None and l3 is None:
            continue
        parsed.append({
            "xve_stall_pct": float(stall.group(1)) if stall else None,
            "occupancy_pct": float(occ.group(1)) if occ else None,
            "l3_bw_bound_pct": float(l3.group(1)) if l3 else None,
        })
    if not parsed:
        return None
    # pick the block with the highest occupancy (the working device)
    best = max(parsed, key=lambda m: (m["occupancy_pct"] or -1.0))
    return {
        "xve_stall_pct": best["xve_stall_pct"] if best["xve_stall_pct"] is not None else 0.0,
        "occupancy_pct": best["occupancy_pct"] if best["occupancy_pct"] is not None else 0.0,
        "l3_bw_bound_pct": best["l3_bw_bound_pct"] if best["l3_bw_bound_pct"] is not None else 0.0,
    }


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------
def classify_kernel(k, rm):
    """Return (bound, why). rm = run_metrics dict or None."""
    mpc = k["ms_per_call"]
    calls = k["calls"]
    launch_like = (mpc < TH_LAUNCH_MS and calls >= TH_LAUNCH_CALLS)

    if rm is None:
        why = ("no VTune run metrics; ranked #%s by share (%.1f%%). "
               "Capture gpu-hotspots before optimizing." )
        return "unknown", why  # filled in by caller with rank

    l3 = rm.get("l3_bw_bound_pct", 0.0)
    occ = rm.get("occupancy_pct", 0.0)
    stall = rm.get("xve_stall_pct", 0.0)

    # 1. launch-bound: many tiny submits, dispatch dominates kernel body
    if launch_like:
        return ("launch",
                "ms/call %.4f over %d calls -> per-submit dispatch dominates"
                % (mpc, calls))
    # 2. bandwidth-bound
    if l3 > TH_L3_BW_BOUND:
        return ("bandwidth",
                "L3 BW bound %.1f%% of peak (>%.0f%%): saturating VRAM/L3 bytes"
                % (l3, TH_L3_BW_BOUND))
    # 3. latency / occupancy bound  (our 27B decode signature)
    if l3 <= TH_L3_BW_BOUND and occ < TH_OCC_LOW and stall > TH_XVE_STALL:
        return ("latency",
                "occupancy %.1f%% (<%.0f%%) + XVE stall %.1f%% (>%.0f%%) + L3 bound only %.1f%%: "
                "XVEs idle waiting -- not enough resident work to hide latency"
                % (occ, TH_OCC_LOW, stall, TH_XVE_STALL, l3))
    # 4. compute-bound
    if occ >= TH_OCC_LOW and l3 <= TH_L3_BW_BOUND:
        return ("compute",
                "occupancy %.1f%% (>=%.0f%%), L3 bound only %.1f%%: ALU/XMX throughput limited"
                % (occ, TH_OCC_LOW, l3))
    # 5. metrics present but ambiguous
    return ("unknown",
            "metrics present but no rule fired (L3 %.1f%%, occ %.1f%%, stall %.1f%%)"
            % (l3, occ, stall))


def classify_all(kernels, rm, top_n):
    """Classify in place; rank by pct desc. Returns sorted list."""
    ranked = sorted(kernels, key=lambda k: k["pct"], reverse=True)
    for i, k in enumerate(ranked):
        bound, why = classify_kernel(k, rm)
        if rm is None:
            hot = i < top_n
            why = ("no VTune run metrics; ranked #%d by share (%.1f%%). %s"
                   % (i + 1, k["pct"],
                      "CANDIDATE -- profile with VTune next." if hot
                      else "below top-%d cutoff." % top_n))
        k["bound"] = bound
        k["why"] = why
    return ranked


# --------------------------------------------------------------------------
# levers report
# --------------------------------------------------------------------------
def render_levers(model, phase, total_ms, rm, ranked, top_n):
    out = []
    out.append("=" * 78)
    out.append("ROOFLINE LEVERS  --  %s / %s  (device %s, %.0f GB/s peak)"
               % (model or "?", phase or "?", DEVICE, PEAK_BW_GBS))
    if total_ms is not None:
        out.append("GPU total: %.3f ms across %d kernels" % (total_ms, len(ranked)))
    if rm is not None:
        out.append("VTune run metrics: XVE-stall %.1f%% | occupancy %.1f%% | L3-BW-bound %.1f%%"
                   % (rm["xve_stall_pct"], rm["occupancy_pct"], rm["l3_bw_bound_pct"]))
    else:
        out.append("VTune run metrics: NONE (bound=unknown; top-%d flagged as candidates)" % top_n)
    out.append("=" * 78)
    hdr = "%-3s %-26s %7s %6s %9s  %-9s" % ("#", "kernel", "ms", "share", "ms/call", "bound")
    out.append(hdr)
    out.append("-" * 78)
    for i, k in enumerate(ranked):
        flag = "*" if (rm is None and i < top_n) else " "
        out.append("%-3d%s%-25s %7.3f %5.1f%% %9.4f  %-9s"
                   % (i + 1, flag, k["name"][:25], k["total_ms"], k["pct"],
                      k["ms_per_call"], k["bound"]))
    out.append("-" * 78)
    out.append("LEVERS (top %d hot kernels):" % min(top_n, len(ranked)))
    for i, k in enumerate(ranked[:top_n]):
        out.append("  %d. %s  [%s, %.1f%%]" % (i + 1, k["name"], k["bound"], k["pct"]))
        out.append("       why : %s" % k["why"])
        out.append("       lever: %s" % LEVERS.get(k["bound"], LEVERS["unknown"]))
    out.append("=" * 78)
    return "\n".join(out)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Roofline bottleneck classifier for the IE perf harness.")
    ap.add_argument("--kprofile", required=True,
                    help="kprofile text file, or '-' for stdin")
    ap.add_argument("--vtune", default=None,
                    help="VTune gpu-hotspots result dir (optional)")
    ap.add_argument("--model", default=None, help="model name for the capture")
    ap.add_argument("--phase", default=None, choices=["decode", "prefill"],
                    help="override phase (else inferred from kprofile header)")
    ap.add_argument("--out", default=None, help="write capture JSON here")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                    help="how many hot kernels to surface (default %d)" % DEFAULT_TOP_N)
    args = ap.parse_args(argv)

    if args.kprofile == "-":
        text = sys.stdin.read()
    else:
        with open(args.kprofile, "r") as f:
            text = f.read()

    kernels, total_ms, phase = parse_kprofile(text)
    if not kernels:
        sys.stderr.write("[roofline] no kernel rows parsed from kprofile -- "
                         "check the input format.\n")
        return 2
    phase = args.phase or phase or "decode"

    rm = None
    if args.vtune:
        if not os.path.isdir(args.vtune):
            sys.stderr.write("[roofline] --vtune dir not found: %s "
                             "(proceeding without run metrics)\n" % args.vtune)
        else:
            rm = parse_vtune_summary(run_vtune_summary(args.vtune))
            if rm is None:
                sys.stderr.write("[roofline] could not parse VTune metrics "
                                 "(proceeding without run metrics)\n")

    ranked = classify_all(kernels, rm, args.top_n)

    capture = {
        "model": args.model or "unknown",
        "phase": phase,
        "device": DEVICE,
        "peak_bw_gbs": PEAK_BW_GBS,
        "total_ms": round(total_ms, 4) if total_ms is not None else None,
        "run_metrics": rm if rm is not None else {
            "xve_stall_pct": None, "occupancy_pct": None, "l3_bw_bound_pct": None},
        "kernels": ranked,
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(capture, f, indent=2)
        sys.stderr.write("[roofline] wrote capture -> %s\n" % args.out)

    print(render_levers(args.model, phase, total_ms, rm, ranked, args.top_n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
