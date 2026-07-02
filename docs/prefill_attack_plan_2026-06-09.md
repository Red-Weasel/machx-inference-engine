# Prefill Attack Plan — 2026-06-09

Goal: close pp512 from **202.9 tok/s → ≥443 tok/s** (50% of llama.cpp Vulkan
b8902's 885.9, the v1 ship gate), ideally further. Quality gate on every
change: **built-in 511-tok PPL ≤ 6.57** (baseline 6.54 ± 0.03), decode tok/s
not regressed (46.8 baseline, `ie-bench-suite`).

## Where the 2523 ms (T=512) actually goes

`unitrace --device-timing` (2026-06-09, /tmp/unitrace_pp512.txt) — GPU is
saturated (device-busy ≈ wall), so this is kernel efficiency, not launch gaps:

| kernel | device ms | % | effective BW |
|---|---:|---:|---|
| `gemm_q4_K_xmx` (6720 calls) | 1194 | 47% | ~53 GB/s |
| `moe_prefill_gate_up_silu_q4k` (40) | 458 | 18% | ~39 GB/s |
| `moe_prefill_down_packed_*_v2` (40) | 455 | 18% | ~20 GB/s |
| `gemm_q6_K_xmx` (1280) | 242 | 10% | similar |
| `gemm_q4_K` scalar (960) | 207 | 8% | — |
| recurrence + attention + router + rest | ~130 | 5% | — |

Achievable sustained BW on B70 is 425–486 GB/s → the quant-weight streaming
kernels run at 5–12% of that. Root cause (read in source): per-lane **scalar
byte loads** of `block_q4_K.qs[]`/`.scales[]`, adjacent lanes reading blocks
~1.1 KB apart (uncoalesced), plus `gemv_q_T` slicing T=512 into 32 launches
of M=16 (weight matrix re-streamed per slice; only L2 softens it).

## What changed externally (research sweep 2026-06-09)

- **Intel's empirical guidance for B70 prefill** (llama.cpp PR #22147,
  Intel-authored, validated on B70): *dequant-to-FP16 + big FP16 GEMM beats
  direct DPAS-on-Q4_K* (both their MMQ and fused dequant+GEMM lost to
  dequant + oneDNN). Matches our measurement exactly.
- llama.cpp PR #23142 (+70% MoE prefill on B60/Qwen3.6-35B): expert-contiguous
  row mapping + counting sort, no atomics. **We already do this** (host
  counting sort + packed stage1/stage2); their win came from removing a
  device-side scan we never had. Low expected value.
- llama.cpp #21893 (Battlemage DPAS non-determinism) **closed as a llama.cpp
  software bug** — see known_bugs.md correction.
- New NEO 26.18 / IGC 2.34.4 exist, but IGC #412 reports a register-allocator
  spill regression on high-GRF DPAS kernels on BMG (+33% slower prefill in
  reporter's case). **Do not upgrade toolchain mid-optimization.**

## Experiment queue (each gated on PPL + decode-unchanged)

### E1 — dequant-to-fp16 scratch + `gemm_fp16` for prefill projections  ✅ LANDED 2026-06-09

**Results:** PPL 6.5429 (= baseline 6.5433, gate passed). pp512 **309.4 tok/s**
(1654.7 ms) vs 166.9 same-session with `IE_NO_DEQ_FP16=1`, vs 202.9 morning
baseline → **+52%**. Decode: see suite re-run below (path fires only at T≥64).
Implementation: `dequant_q4_K_to_Bt` / `dequant_q6_K_to_Bt` (dequant_kernels.cpp)
+ fp16-GEMM branch at top of `gemv_q_T` (qwen36.cpp) with process-lifetime
grown-on-demand device scratch (max ~48 MB). Now at 35% of llama.cpp Vulkan's
885.9; remaining dominant cost is the MoE multiexpert pair (E2).
Replaces ~1643 ms (65%) of prefill: all `gemv_q_T` → `gemm_q4_K_xmx` /
`gemm_q6_K_xmx` / `gemm_q4_K` dispatches at T≥64.

Mechanics: per projection, `dequant_q4_K_to_Bt` (new kernel: Q4_K W[N,K] →
fp16 W^T[K,N] scratch, one block per WG, coalesced-ish writes), then existing
`gemm_fp16` (128×128 WG tile, 33.5 TFLOPS measured Phase 3), then fp32→fp16
cast of C (or fp16-store variant). Scratch: max 32 MB (qkv 2048×8192) + C
scratch 16 MB — VRAM headroom ~10 GB, fine.

Math: qkv/layer = 17.2 GFLOP → 0.52 ms at 33.5 TFLOPS + ~0.3 ms dequant
vs 10.3 ms today (~10×). Across all projections, projected prefill
2523 → ~1100 ms ≈ **460 tok/s pp512**. Precision: B was already rounded to
fp16 in the XMX path; accumulate stays fp32 → PPL should hold exactly.

Risk: gemm_fp16 is single-buffered SLM (18% peak) — even so it's ~6× the
effective throughput of the Q4_K path. Double-buffering it later is a
follow-on (E3).

### E2 — vectorized loads in MoE stage-1/stage-2  ✅ LANDED 2026-06-09

**Results:** PPL 6.5429 (bit-identical — per-accumulator add order preserved
by construction). pp512 **589.7 tok/s** (868.3 ms, was 1654.7 ms after E1)
→ **+91% over E1, +191% cumulative on the day**. Now at **67% of llama.cpp
Vulkan's 885.9 — the 50% v1 ship gate (443 tok/s) is PASSED.**
Implementation (all in `src/ops/moe_fused.cpp`): in
`moe_prefill_gate_up_silu_q4k`, `moe_prefill_down_packed_q4k_v2`, and
`moe_prefill_down_packed_q6k_v2`, weight bytes now load as one
`vec<uint32_t,4>` (Q4_K; 144 B block stride keeps qs[] 16 B-aligned) or
`uint16_t` pairs (Q6_K; 210 B stride is only 2 B-aligned), dequant to a
16-float register array once per block, and the per-block SLM activation
reads collapse from 16 scalar loads per row to two `vec<half,8>` loads.

### E2b — M-grid + vectorized scalar gemm_q4_K  ✅ LANDED 2026-06-09

**Results:** PPL 6.54 ✓, pp512 **754.5 tok/s** (678.6 ms, +28% over E2),
decode 46.8 = baseline ✓. **Now 85% of llama.cpp Vulkan.**
The N=32 alpha/beta projections were 209 ms across 960 launches of 2 WGs each
(M capped at 32/launch). `gemm_q4_K` (gemv_q4k.cpp) now tiles M via grid
dim 0 — one launch per projection — and its inner loop got the E2 vector-load
treatment. `gemv_q_T`'s scalar branch dispatches the full T in one call.

### E2c — SLM-staged Q6_K weight slabs in down_packed_q6k_v2  ✅ LANDED 2026-06-09

**Results:** PPL 6.54 ✓, pp512 **840.7 tok/s** (609.0 ms, +11% over E2b) →
**95% of llama.cpp Vulkan.** The WG's 32 columns × blocks_per_col Q6_K blocks
are one contiguous, 16 B-divisible slab; staging it into SLM with cooperative
vec4 loads once per WG sidesteps the 210 B-block alignment problem entirely
and stops per-M-tile global re-reads.

### E4 — vector header loads in Q4_K kernels  ⚪ NEUTRAL, REVERTED
840.74 vs 840.72 — the scalar header reads were already L1-served.
Diagnostic value: stage-1 gate_up is now COMPUTE-bound (~45% of the scalar
fp32 FMA roofline), not load-bound.

### E3 — BK=64 gemm_fp16 retile  ❌ REVERTED
pp512 840.7 → 664.7 (−21%). The 4× barrier reduction lost to the 4× SLM
growth (8→32 KiB/WG halves WG occupancy). PPL/decode fine; reverted on perf.
Lesson: gemm_fp16 wants double-buffering at SMALL BK, not a wider tile.

### E5 — pre-gathered expert-sorted x_packed for stage-1  ✅ LANDED 2026-06-09

**Results: PPL 6.54 ✓, pp512 = 899.7 tok/s (569.1 ms) — BEATS llama.cpp
Vulkan b8902's 885.9 (101.6%). Decode 46.8 unchanged. ENGINE NOW WINS BOTH
METRICS.** Every n-chunk WG of an expert was re-gathering the same activation
rows through `sorted_token_idx` — E_ffn/32 = 24× redundant scattered reads of
x per layer (~390 MB/layer). One `moe_gather_rows` pass per layer into the
new `ws_moe_xp_` buffer (max_T_moe × K_top × H fp16) makes every WG's A-tile
load contiguous and coalesced.

### v1.1/v1.2 follow-on session (2026-06-09, same day)
- **v1.1** `gemv_q6_K_slm` (SLM-slab staged Q6_K GEMV, lm_head + ssm_out):
  decode 46.8 → 52.4 at PPL 6.54. ✅
- **v1.2-B router rewrite** (vectorized logits + parallel top-8 via
  packed-key max-reduces; the old kernel ran a serial 8×256 scan on lane 0,
  ~4 ms/token): decode 52.4 → **66.2**, pp512 900 → 925.5. ✅ The single
  biggest decode win of the project.
- **v1.2-C** down-kernel M_TILE 8→16: pp512 → **938.6**. ✅
- **v1.2-A** fp16-packed (vec<half,2>) inner products in gate_up: −2.7%,
  REVERTED — IGC does not emit packed-rate fp16 FMAs from vec<half,2>;
  don't retry packed-half scalar-kernel math on this toolchain.
- **v1.2-D** gemm_fp16 double-buffer at BK=16: −9.3%, REVERTED. Together
  with E3: gemm_fp16's 4 KiB tiles are already latency-hidden by WG
  multithreading; leave the kernel alone.
- Final 2026-06-09 standings vs llama.cpp Vulkan b8902: prefill **106%**,
  decode **167%** (turbo GGUF: 175%), PPL 6.55.

### v1.3 — gate_up XMX rewrite  ❌ SETTLED NEGATIVE (commit 3cf1f03)
The full rewrite (E5 contiguous x_packed, E2 vectorized dequant, M_GROUPS=2,
fused SwiGLU) is CORRECT (PPL 6.55) but **3.4× slower** than scalar stage-1
(537 vs 938 pp512, warm JIT). Structural: ~32k strided 2 B SLM stores per
WG per K-block to feed row-major B-tiles + 80 KiB SLM ⇒ 1 WG/Xe-core.
Matches Intel's llama.cpp PR #22147 finding (DPAS-on-quant loses to
dequant+dense on B70). Kept as `IE_MOE_XMX=1` opt-in; revisit ONLY if the
toolchain gains working 2D block loads on BMG or cheap joint_matrix
B-packing. **gate_up scalar is the settled production kernel.**

### Remaining future headroom
- Parallel top-k/top-p sampler (`sample_softmax_topk_topp` is a single GPU
  thread over 248k vocab) — currently has NO callers; rewrite when the
  server/sampling path gets wired up. BLOCKER for Seal-app integration.
- dn_recurrence 46 ms is the floor (T≤256 chunking constraint).

### E3 — double-buffer gemm_fp16 SLM (Phase 3 carried debt, 18% → 35%+ peak)

### E4 — full-attn prefill path (290 ms) — only if E1–E3 land short of gate.

## Standing constraints
- No ESIMD / block2d / `Subgroup2DBlockLoad` / `SubgroupBlockRead` (DEVICE_LOST
  on BMG-G31 C0; repo rule).
- No toolchain upgrade during this work (IGC #412).
- `IE_ENABLE_GEMM_Q4K_ESIMD`, `IE_ENABLE_FA2_PREFILL_TILED` stay OFF.
- Prefill stays externally chunked at T≤256 in production paths for the DN
  determinism workaround; benches run T=512 single-call — fine for perf
  comparison, PPL tool uses production chunking.
