#!/usr/bin/env python3
"""benchscope_export.py — feed REAL ie-bench --kprofile timings into BenchScope.

Converts the per-kernel table that `ie-bench --kprofile` / `--kprofile-decode`
prints into `benchscope.trace.v1` events and POSTs them to a running BenchScope
server (default http://127.0.0.1:8787), so the dashboard shows ACTUAL B70
numbers instead of the synthetic demo trace.

The kprofile is an aggregate (summed GPU time per kernel for one warm step), not
a real per-call timeline, so kernels are laid end-to-end by their total duration:
the Gantt then reads as "share of the phase", which is exactly what the % column
means. `append`/KV kernels are tagged category=memory_kv so they show up in the
KV lane (where you can SEE that decode KV-append is a ~2 us sliver, not a delay).

Usage
-----
  # live: run a benchmark and pipe its kprofile straight in
  ./build/tools/ie-bench --gguf "$G" --prefill 16384 --decode 6 --warmup 4 \
      --kprofile-decode 2>/dev/null \
    | python3 tools/perf/benchscope_export.py --session real-coder-dec-16k \
        --model Coder-30B-A3B --arch qwen3moe --quant Q4_K_M --ctx 16384 \
        --flags IE_FA2_PREFILL_V2 --tok-s 24.2 --ppl 6.46

  # from a captured file instead of stdin:
  python3 tools/perf/benchscope_export.py --file cap.txt --session ... [...]

  # write JSONL instead of POSTing (for offline import):
  ... --jsonl out.jsonl
"""
from __future__ import annotations
import argparse, json, re, sys, time, urllib.request, urllib.error

HEADER_RE = re.compile(r"kernel profile", re.I)
CTX_RE    = re.compile(r"(?:pos=|T_pp=)(\d+)")
GPUTOT_RE = re.compile(r"GPU total\s+([\d.]+)\s*ms", re.I)
NUM_RE    = re.compile(r"[\d.]+")


def parse_kprofile(text: str):
    """Return (phase, ctx_tokens, gpu_total_ms, [ {name,total_ms,pct,calls,avg_ms} ])."""
    phase, ctx, gpu_total, rows, in_block = "unknown", 0, 0.0, [], False
    for line in text.splitlines():
        if HEADER_RE.search(line):
            phase = "decode" if re.search(r"\bDECODE\b", line) else \
                    ("prefill" if "prefill" in line.lower() else "decode")
            m = CTX_RE.search(line);    ctx = int(m.group(1)) if m else ctx
            m = GPUTOT_RE.search(line); gpu_total = float(m.group(1)) if m else gpu_total
            in_block = True
            continue
        if not in_block:
            continue
        s = line.strip()
        if not s:
            in_block = False
            continue
        tok = s.split()[0]
        if tok in ("TOTAL", "#") or s.startswith("---") or tok.startswith("==="):
            continue
        if not re.search(r"[A-Za-z]", tok):   # kernel names contain letters; skip stray numeric lines
            continue
        nums = NUM_RE.findall(s[len(tok):])
        if len(nums) < 3:            # need at least total_ms, pct, calls
            continue
        total_ms = float(nums[0]); pct = float(nums[1]); calls = int(float(nums[2]))
        if pct > 100.5 or calls <= 0:          # a %-of-phase >100 is a mis-parse, not a kernel
            continue
        avg_ms = float(nums[3]) if len(nums) > 3 else (total_ms / max(calls, 1))
        rows.append(dict(name=tok, total_ms=total_ms, pct=pct, calls=calls, avg_ms=avg_ms))
    return phase, ctx, gpu_total, rows


def categorize(name: str) -> str:
    n = name.lower()
    if "append" in n:                         return "memory_kv"     # the KV write
    if any(k in n for k in ("rms", "norm", "quant", "gather", "reduce")):
        return "kernel"
    return "kernel"


# Per-arch attention dims (overridable by --n-kv-heads / --head-dim). Used to turn
# the kprofile's (calls, ctx) into EXACT KV-cache byte traffic for the attention
# kernels — the bandwidth number the fa2_partial diagnosis hinges on.
ARCH_DIMS = {
    "qwen3moe":     {"n_kv": 4, "hd": 128},   # Coder-30B-A3B
    "qwen35_dense": {"n_kv": 8, "hd": 128},   # 27B dense
    "qwen35moe":    {"n_kv": 2, "hd": 256},   # crown 35B
}


def kernel_bytes(name: str, ctx: int, calls: int, n_kv: int, hd: int):
    """Return (bytes_read, bytes_written) for a kernel, or (0,0) if not derivable.

    Exact for the KV-attention kernels: a decode step calls them once per layer
    (calls == n_attn_layers), each reading the full causal KV (ctx positions).
    int8 KV = 1 byte/elem + a tiny fp16 per-row scale; fp16 = 2 bytes/elem.
    """
    n = name.lower()
    is_int8 = "int8" in n
    elem = 1 if is_int8 else 2
    # K and V both streamed; +scale rows for int8 (1 fp16 per (kv_head,pos)).
    per_pos = n_kv * (hd * elem + (2 if is_int8 else 0)) * 2
    if "fa2_partial" in n or n.startswith("attn_naive_compute"):
        return calls * ctx * per_pos, 0          # reads the whole KV per layer
    if "fa2_append" in n or "attn_naive_append" in n:
        # writes ONE token's K+V (this is the "KV append" op the dashboard shows)
        return 0, calls * (n_kv * hd * elem * 2)
    return 0, 0


def build_events(args, phase, ctx, gpu_total, rows):
    base = time.time_ns()
    path = {"engine": "Mach X Inference Engine", "arch": args.arch,
            "model": args.model, "quant": args.quant,
            "flags": args.flags.split(",") if args.flags else []}
    dims = dict(ARCH_DIMS.get(args.arch, {"n_kv": 0, "hd": 0}))
    if args.n_kv_heads: dims["n_kv"] = args.n_kv_heads
    if args.head_dim:   dims["hd"]   = args.head_dim
    weight_rows = [r for r in rows if r["name"].lower().startswith(("gemv", "moe", "gemm"))]
    weight_ms = sum(r["total_ms"] for r in weight_rows) or 1.0

    evs, cursor, sess_read, sess_written = [], base, 0, 0
    for r in rows:
        dur = int(round(r["total_ms"] * 1e6))
        b_read, b_write = kernel_bytes(r["name"], ctx, r["calls"], dims["n_kv"], dims["hd"])
        est = False
        # optional: attribute a known per-token weight-read budget across weight kernels
        if args.weight_gb and r in weight_rows and b_read == 0:
            b_read = int(args.weight_gb * 1e9 * r["total_ms"] / weight_ms); est = True
        secs = r["total_ms"] / 1000.0
        metrics = {"pct_of_phase": r["pct"], "calls": r["calls"],
                   "avg_ms": round(r["avg_ms"], 4), "total_ms": round(r["total_ms"], 4)}
        if b_read:
            gbs = b_read / secs / 1e9
            metrics.update(bytes_read=b_read, hbm_gb_s=round(gbs, 1),
                           pct_peak_bw=round(100 * gbs / args.peak_bw, 1),
                           bw_estimated=est)
            sess_read += b_read
        if b_write:
            metrics.update(bytes_written=b_write,
                           hbm_gb_s=round(b_write / secs / 1e9, 2))
            sess_written += b_write
        evs.append({
            "session": args.session, "type": "span", "category": categorize(r["name"]),
            "name": r["name"], "phase": phase, "ts_ns": cursor, "dur_ns": dur,
            "scope": {"model": args.model, "context_tokens": ctx, "calls": r["calls"],
                      "avg_ms": round(r["avg_ms"], 4)},
            "metrics": metrics, "path": path,
            "notes": "aggregate GPU time for this kernel across the step (kprofile)"
                     + ("; weight bytes ESTIMATED by time-share" if est else ""),
        })
        cursor += dur
    # session-level rollups
    if sess_read or sess_written:
        evs.append({"session": args.session, "type": "metric", "category": "metric",
                    "name": f"{phase}_hbm_read", "phase": phase, "ts_ns": base, "dur_ns": 0,
                    "metrics": {"bytes_read": sess_read, "bytes_written": sess_written,
                                "read_GB": round(sess_read / 1e9, 3)}, "path": path})
    if gpu_total > 0:
        evs.append({"session": args.session, "type": "metric", "category": "metric",
                    "name": f"{phase}_gpu_total_ms", "phase": phase, "ts_ns": base, "dur_ns": 0,
                    "metrics": {"gpu_total_ms": round(gpu_total, 3)}, "path": path})
    if args.tok_s is not None:
        evs.append({"session": args.session, "type": "metric", "category": "metric",
                    "name": f"{phase}_tok_s", "phase": phase, "ts_ns": base, "dur_ns": 0,
                    "metrics": {"tok_s": args.tok_s}, "path": path})
    if args.ppl is not None:
        evs.append({"session": args.session, "type": "instant", "category": "quality_gate",
                    "name": "ppl_gate", "phase": phase, "ts_ns": base, "dur_ns": 0,
                    "metrics": {"ppl": args.ppl, "gate": args.ppl_gate,
                                "pass": args.ppl <= args.ppl_gate}, "path": path})
    # --- the other lanes (so the dashboard isn't kernel-only) ---
    total_launches = sum(r["calls"] for r in rows)

    # HOST_SCHEDULER lane: launch count + (if VTune given) the GPU idle/occupancy
    # that explains a launch/dispatch-bound decode (tiny T=1 kernels + inter-launch
    # gaps leave the EUs idle). This is THE decode bottleneck signal.
    hs = {"launches": total_launches, "launches_per_token": total_launches}
    if args.gpu_idle is not None:  hs["gpu_idle_pct"] = args.gpu_idle
    if args.occupancy is not None: hs["occupancy_pct"] = args.occupancy
    note = ""
    if args.gpu_idle is not None and args.gpu_idle >= 25:
        note = (f"DISPATCH-BOUND: GPU ~{args.gpu_idle:.0f}% idle — {total_launches} tiny "
                f"launches/token don't fill the EUs (kernel speed irrelevant; lever = "
                f"command-graph/fusion to feed the GPU).")
    evs.append({"session": args.session, "type": "metric", "category": "host_scheduler",
                "name": f"{phase}_dispatch", "phase": phase, "ts_ns": base, "dur_ns": 0,
                "metrics": hs, "path": path, "notes": note})

    # WEIGHT_LAYOUT lane: group kernels by inferred quant/layout, report time share.
    def quant_of(n):
        n = n.lower()
        if "q6" in n:        return "Q6_K SoA int-dot"
        if "q4k" in n or "q4_k" in n or "gate_up" in n or "down_q4k" in n: return "Q4_K int-dot W4A8"
        if "q8" in n:        return "Q8 int-dot"
        if "gemv_fp16" in n or "fp16" in n: return "FP16 (expanded — int-dot candidate)"
        return None
    layups = {}
    for r in rows:
        ql = quant_of(r["name"])
        if ql: layups[ql] = layups.get(ql, 0.0) + r["total_ms"]
    for ql, ms in sorted(layups.items(), key=lambda kv: -kv[1]):
        evs.append({"session": args.session, "type": "metric", "category": "weight_layout",
                    "name": ql, "phase": phase, "ts_ns": base, "dur_ns": 0,
                    "metrics": {"total_ms": round(ms, 3),
                                "pct_of_phase": round(100 * ms / max(gpu_total, 0.001), 1)},
                    "path": path})

    # MEMORY_KV lane: KV cache mode + the KV read traffic (from the attention kernels).
    evs.append({"session": args.session, "type": "metric", "category": "memory_kv",
                "name": f"KV cache ({args.kv_mode})", "phase": phase, "ts_ns": base, "dur_ns": 0,
                "metrics": {"kv_mode": args.kv_mode, "kv_bytes_read": sess_read,
                            "kv_read_GB": round(sess_read / 1e9, 3),
                            "context_tokens": ctx}, "path": path})

    # ALGORITHM_PATH lane: the config + one event per active gate/flag.
    evs.append({"session": args.session, "type": "instant", "category": "algorithm_path",
                "name": f"{args.arch} {phase} config", "phase": phase, "ts_ns": base, "dur_ns": 0,
                "scope": {"model": args.model, "context_tokens": ctx}, "path": path})
    for fl in (path.get("flags") or []):
        evs.append({"session": args.session, "type": "instant", "category": "algorithm_path",
                    "name": f"flag: {fl}", "phase": phase, "ts_ns": base, "dur_ns": 0,
                    "path": path})
    return evs


def post(url, session, events):
    body = json.dumps({"session": session, "events": events}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="kprofile text file (default: stdin)")
    ap.add_argument("--session", required=True)
    ap.add_argument("--model", default="model"); ap.add_argument("--arch", default="qwen3moe")
    ap.add_argument("--quant", default="Q4_K_M"); ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--flags", default="")
    ap.add_argument("--n-kv-heads", type=int, default=0, dest="n_kv_heads",
                    help="override KV heads (else inferred from --arch) for exact KV byte traffic")
    ap.add_argument("--head-dim", type=int, default=0, dest="head_dim", help="override head_dim")
    ap.add_argument("--peak-bw", type=float, default=608.0, dest="peak_bw",
                    help="device peak VRAM BW in GB/s (B70 = 608) for %%-of-peak")
    ap.add_argument("--weight-gb", type=float, default=None, dest="weight_gb",
                    help="optional: per-token weight-read budget (GB) to attribute "
                         "across gemv/moe kernels by time-share (ESTIMATE, flagged)")
    ap.add_argument("--gpu-idle", type=float, default=None, dest="gpu_idle",
                    help="VTune XVE Array Idle %% (host_scheduler lane — dispatch-bound signal)")
    ap.add_argument("--occupancy", type=float, default=None, dest="occupancy",
                    help="VTune XVE Threads Occupancy %% (host_scheduler lane)")
    ap.add_argument("--kv-mode", default="fp16", dest="kv_mode",
                    help="KV cache mode for the memory_kv lane (fp16 | int8)")
    ap.add_argument("--tok-s", type=float, default=None, dest="tok_s")
    ap.add_argument("--ppl", type=float, default=None)
    ap.add_argument("--ppl-gate", type=float, default=6.57, dest="ppl_gate")
    ap.add_argument("--server", default="http://127.0.0.1:8787")
    ap.add_argument("--jsonl", help="write events to this JSONL file instead of POSTing")
    a = ap.parse_args()

    text = open(a.file).read() if a.file else sys.stdin.read()
    phase, ctx, gpu_total, rows = parse_kprofile(text)
    if a.ctx: ctx = a.ctx
    if not rows:
        print("benchscope_export: no kernel rows parsed — is this --kprofile output?", file=sys.stderr)
        sys.exit(2)
    events = build_events(a, phase, ctx, gpu_total, rows)

    if a.jsonl:
        with open(a.jsonl, "w") as f:
            for e in events: f.write(json.dumps(e) + "\n")
        print(f"wrote {len(events)} events -> {a.jsonl}")
        return
    try:
        resp = post(a.server + "/api/events", a.session, events)
        top = sorted(rows, key=lambda r: -r["total_ms"])[:3]
        print(f"POSTed {resp.get('accepted', len(events))} events to {a.server} "
              f"(session {a.session}, phase {phase}, ctx {ctx})")
        print("  top kernels: " + ", ".join(f"{r['name']} {r['pct']:.1f}%" for r in top))
    except urllib.error.URLError as e:
        print(f"benchscope_export: POST failed ({e}). Is the server up? "
              f"Falling back to --jsonl /tmp/benchscope_{a.session}.jsonl", file=sys.stderr)
        with open(f"/tmp/benchscope_{a.session}.jsonl", "w") as f:
            for ev in events: f.write(json.dumps(ev) + "\n")


if __name__ == "__main__":
    main()
