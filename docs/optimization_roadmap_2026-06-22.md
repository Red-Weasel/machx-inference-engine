# Optimization roadmap — 2026-06-22 (5-lens domain analysis synthesis)

Produced by 5 parallel domain agents (kernel / scheduler / KV / algorithm-path / weight-layout)
analyzing the clean-box profile + codebase. ~10 distinct levers found — several **bit-exact**
(zero PPL risk) and several **already built but switched off**. Roofline anchor for decode:
occupancy 27.6%, XVE-stall 84.9%, L3-BW-bound 0% → **latency/occupancy + launch bound, NOT
bandwidth bound** (this is why the SoA-coalescing attempt gave only parity).

## ⭐ Convergence signal
Two independent lenses (scheduler #3 + weight-layout #1) landed on the SAME lever:
**hoist `quantize_q8_1` out of the per-GEMV path** (the activation is re-quantized 3–5× per
layer; the fused `rms_norm_f32w_q8` kernel already exists and is unused on this path). When two
lenses converge, confidence is high.

## PHASE 1 — Bit-exact decode wins (NO PPL risk, fast A/B each)
| # | Lever | Lens | Est. | Notes |
|---|-------|------|------|-------|
| 1 | **Command-graph capture** of the fixed decode DAG (~1850 launches → 1 submission) | scheduler #1 | **→20–22 tok/s (closes llama gap alone)** | bit-exact; gating unknown = BMG Level-Zero graph support → **prototype 1 layer first** |
| 2 | **Hoist/fuse `quantize_q8_1`** — quant once/layer via existing `rms_norm_f32w_q8` | sched#3 + layout#1 (converged) | +5–12% | bit-identical; kernel already exists unused |
| 3 | **Free SLM activation staging** (GMEM variant) for Q4 decode GEMV | kernel #1 | +10–25% big-K | mirror existing `IE_QWEN35_Q6_SOA_GMEM` toggle |
| 4 | **N_PER_WG 32→16** for small-N shapes (more WGs → occupancy) | kernel #4 | low | one-line; A/B on N=4096 shapes |
| 5 | **Pre-fold scales at load** (super-scale fp16→fp32; `d·s_raw`,`dmin·m_raw` per sub-block) | layout #2+#3 | +1–4% | repack-time; trims hot-loop ALU |
| 6 | **Parallelize the fa2 combine pass** across head_dim | KV #3 | few % | bit-exact reduction reshape |

## PHASE 2 — Long-context PREFILL blocker (LOW risk — the publish-blocker, REFRAMED)
**The collapse is the O(ctx²) dense-attention re-read at a tiny chunk — NOT the DeltaNet cap.**
Proof: Coder-30B (no DeltaNet, T≤256) collapses WORST. The dense-attn chunk is a SEPARATE
lever with ZERO hang risk (the §1 hang lives only in the DeltaNet recurrence kernel).
| # | Lever | Risk | Notes |
|---|-------|------|-------|
| 7 | **Raise chunk for Coder/non-DeltaNet arches** | **near-zero (no DeltaNet)** | attacks the worst collapser; ships ~today; Phase-A test plan |
| 8 | **Decouple dense-attn chunk (large) from DeltaNet chunk (capped)** — `deltanet_recurrence` sub-chunks internally | low | the proper fix; recovers hybrids |

## PHASE 3 — PPL-gated wins (behind the gate)
| # | Lever | Lens | Notes |
|---|-------|------|-------|
| 9 | **Enable INT8 KV for the crown family** (already built + tuned, default-OFF) | KV #1 | ~2× KV bytes → big long-ctx DECODE win (16K 25→35); eval-gate K-quant-through-softmax |
| 10 | **Cross-WG split-K** for big-K/small-N GEMVs | kernel #2 | bigger occupancy ceiling; reorders FP adds → re-gate |

## PHASE 4 — Autotuner (#5)
Bank the geometry/variant wins (N_PER_WG, SLM-vs-GMEM, split-K factor) into a load-time
micro-bench that self-selects per (shape, device).

## DEFER — high-risk / low-value (the traps)
- **Raise the DeltaNet recurrence cap itself** — the box-freezer (§1 hang) AND low leverage
  (recurrence is O(ctx) not O(ctx²)). The scary lever is also the worthless one.
- **INT4 KV** — quality risk at exactly the long contexts we care about.
- **Chunked/blocked DeltaNet rewrite** — multi-day correctness milestone.

## MEASURED RESULTS (tournament round 1 — 2026-06-22)
Three levers tested, three rejected — but each sharpened the target (measurement killed
wrong levers cheaply; see `tools/perf/PERF_LEDGER.md`):
- **Q4 SoA decode** → parity (0.94–1.02×). Occupancy-bound, not BW — coalescing no headroom.
- **quantize_q8_1 hoist** → neutral (0.99×, PPL bit-identical). Removing ~150 launches/token
  did nothing → launch-COUNT isn't the lever; only the command-graph (1850→1, different
  mechanism) could be — but the neutral result is a yellow flag for it too.
- **Enable existing tiled FA2 prefill** → REGRESSION (0.72×, slower than naive at ALL ctx).
  The shelved kernel is a BAD impl (that's why it failed its pp512 gate); not the fix.

### ⭐ SHARP DIAGNOSIS (the real targets, now data-grounded)
- **Prefill blocker = `attn_naive_compute`** — naive O(T²) attention, **70.9% of prefill @4096**
  (measured). The fix is a GOOD flash-attention-2 prefill kernel (tiled, fused online-softmax,
  high occupancy). The existing tiled attempt is inadequate; this needs a real kernel, written
  well. Zero hang risk (attention, not DeltaNet). References: the decode `fa2_partial` kernels
  (good), llama's flash-attn (open source), the bad existing tiled (what to avoid). HIGHEST
  value (publish-blocker) + clearest target.
- **Decode gap = launch + occupancy bound.** Micro-levers neutral. Only structural fixes can
  move it: command-graph (big build, BMG-support unknown, now uncertain) + a higher-occupancy
  decode GEMV (split-K cross-WG). Lower certainty than the prefill fix.

**Verdict: the quick wins are exhausted; the real wins are bigger builds. The flash-attn
prefill kernel is the highest-value, most-certain next build (the publish-blocker).**

## Execution discipline
Each change: env-gated → CPU build → PPL gate (or full eval for #9) → order-controlled A/B
(new-old-new) → log to `tools/perf/PERF_LEDGER.md` → commit only if accepted. GPU strictly
serial, idle-checked before each load. Prefill cap tests (#7/#8) follow the agent's Phase-A/B
plan; any DeltaNet-cap test needs explicit user sign-off + `ie-bug-monitor`.
