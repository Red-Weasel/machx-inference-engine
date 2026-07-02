# IE Engine — v1 RELEASE

**Date:** 2026-06-09
**Engine state:** commit `5088da1` + release-checklist fixes
**Hardware:** Intel Arc Pro B70 (BMG-G31, 32 GB GDDR6, Xe2-HPG, 608 GB/s)
**Model:** `Qwen3.6-35B-A3B-Q4_K_M.gguf` (19.7 GiB, hybrid DeltaNet+Attention MoE, 36B total / 3B active)
**Backend:** SYCL / Level Zero, IntelLLVM 2026.0.0, NEO 26.14.37833.4, IGC 2.32.7
**Comparison anchor:** llama.cpp Vulkan build b8902, same GGUF, same GPU,
`llama-bench -ngl 99 -sm none -mg 0`, measured the same hour as the engine numbers.

---

## Headline

**The engine beats llama.cpp Vulkan on both prefill and decode.**

> **2026-06-09 P1 amendment (docs/benchmark_matrix_2026-06-09.md):** the
> llama.cpp **SYCL** backend at master b9586 (post-#23142 + MMVQ) now
> measures 1092 pp512 / 81.1 tg128 on this hardware — ahead of this engine
> (86% / 82%). The Vulkan comparison below remains accurate (engine leads
> any Vulkan build decisively), but "fastest overall on B70" currently
> belongs to llama.cpp SYCL master. Gap-closing work (integer-dot decode
> GEMVs, oneDNN prefill GEMM) is specified in the matrix doc.
>
> **2026-06-10 RESOLUTION — TOTAL CROWN:** both gaps closed the next day.
> Decode crown v1.5 (84.1 turbo / 81.0 default vs SYCL master 81.31), then
> prefill crown v1.6: **pp512 1144 ± 5 vs SYCL master 1064 ± 8 same-hour
> (+7.6%)** via per-expert SoA weight repack + int-dot MoE prefill with
> full-K register lattices and split-half-sum corrections (default ON,
> PPL 6.52).  **The engine now leads llama.cpp's best backend on BOTH
> metrics on B70.**  See docs/benchmark_matrix_2026-06-09.md §v1.6.

| metric | engine | llama.cpp Vulkan b8902 | ratio |
|---|---:|---:|---:|
| **prefill pp512** | **938.6 tok/s** (v1.2) | 885.0 ± 5.5 | **106.1%** |
| **decode tg128** | **66.2 tok/s** (fp16 KV, 5-prompt suite, v1.2) | 39.75 ± 0.06 | **166.5%** |

**v1.2 (same day):** the MoE router was a hidden decode bottleneck — a serial
8×256 top-k scan on one lane (~4 ms/token) plus a scalar-load logit dot.
Vectorized logits + parallel top-8 (packed-key max-reduces): decode
52.4 → 66.2, pp512 +2.8%. Down-kernel M_TILE 8→16 added +1.4% pp512.
Tried and reverted with findings: fp16-packed inner products (−2.7%, IGC
doesn't emit packed-rate FMAs from vec<half,2>), gemm_fp16 double-buffer
(−9.3% — at 4 KB tiles, WG multithreading already hides latency).

Prefill went **202.9 → 899.7 tok/s (+343%) in the final optimization day**
(2026-06-09, E1–E5 in `docs/prefill_attack_plan_2026-06-09.md`), with PPL
held bit-for-bit at 6.54 through every step.

## v1 hard gates

| gate | target | measured 2026-06-09 | status |
|---|---|---|---|
| PP @ T=512 ≥ 50% of llama.cpp Vulkan | ≥ 443 tok/s | **899.7** (101.6% of llama.cpp itself) | ✅ **203% of gate** |
| TG ≥ 95% of llama.cpp Vulkan | ≥ 37.8 tok/s | **46.8** (117.7%) | ✅ |
| PPL stable on multi-token corpus | drift ≤ noise | 6.54 = historic baseline ± 0.03, reproduced 6× today | ✅ |
| INT8 KV cache validated | drift < 0.5% | 0.35% (`validate_int8_kv.sh`) | ✅ |
| Unit tests | all pass | 7/7 (`ctest`) — two stale Phase-5-era sub-tests updated to current kernel contracts | ✅ |
| Build clean | — | clean build; fresh-clone build verified (also fixed `.gitignore` hiding 4 src/core files) | ✅ |
| ≥50 tok/s decode @ small ctx (absolute) | 50 | **52.4 suite / 53.3 INT8 @ ctx=1** (v1.1 kernel, `35a85ea`) | ✅ |

**v1.1 decode update (same day):** the absolute-50 gate initially measured
48.5 (the historical 52.6 predated the bisect cleanup). Closed at full
quality by `gemv_q6_K_slm` — SLM-slab-staged Q6_K GEMV (no sub-group
shuffles, algebraic scale fold) on the lm_head and ssm_out shapes:
decode 46.8 → **52.4** suite, 48.5 → **53.3** INT8 @ ctx=1, PPL unchanged.

## Decode menu (v1.2)

| config | suite tok/s | INT8 @ ctx=1 | pp512 | PPL | vs llama.cpp tg128 |
|---|---:|---:|---:|---:|---:|
| **default** (Q6_K lm_head) | **66.2** | **67.7** | **938.6** | **6.55** | **167%** |
| turbo (`-lmhead-q4k.gguf`) | 69.5 | 73.4 | 947.1 | 6.64 | 175% |

fp16-KV 5-prompt suite (51–219-tok real prompts, 3-run median), ≤0.3% spread.
The turbo GGUF requires no engine changes — `gemv_q()` dispatches by tensor
dtype; pass `--gguf .../Qwen3.6-35B-A3B-Q4_K_M-lmhead-q4k.gguf`.

## Quality

- Built-in 511-tok PPL: **6.54** (fp16 KV), int8-KV prefill-chunked drift 0.35%.
- Every 2026-06-09 optimization was gated on PPL before being kept; E2/E2b
  preserve per-accumulator FP add order (bit-identical), E1/E2c/E5 reproduce
  6.54 exactly.
- Known constraint: DeltaNet recurrence non-determinism on BMG-G31 under long
  single-call prefill — production paths chunk at T≤256 where it never fires
  (`docs/known_bugs.md`; 28-step bisect concluded HW-level, vendor escalation
  pending).

## What made prefill fast (one day, 2026-06-09)

1. **E1** — T≥64 projections dequant Q4_K/Q6_K once to an fp16 scratch and run
   the dense `gemm_fp16` instead of fused quant-XMX (202.9 → 309.4).
2. **E2** — vectorized weight + SLM loads in the three MoE prefill kernels
   (309.4 → 589.7).
3. **E2b** — in-kernel M-tile grid for scalar `gemm_q4_K`; the N=32 alpha/beta
   path was 960 two-workgroup launches (589.7 → 754.5).
4. **E2c** — SLM-staged contiguous Q6_K weight slabs (754.5 → 840.7).
5. **E5** — expert-sorted activation pre-gather; killed 24× redundant scattered
   reads (840.7 → 899.7).

Tried and reverted on gates: E3 (BK=64 gemm_fp16 retile, −21% occupancy loss),
E4 (header vector loads, neutral).

## Deferred to v1.1+

- Recover absolute ≥50 tok/s decode @ small ctx (Q4_K lm_head variant or
  gemv_q6k_huge kernel work).
- ≥50 tok/s @ 32k ctx (algorithmic: paged attention / fp8 KV / MTP).
- gate_up stage-1 is now compute-bound: fp16-product math or XMX retry.
- gemm_fp16 double-buffer at BK=16.
- Server features: multi-turn KV reuse via prefix cache (primitives landed,
  PR #3), defrag, OpenAI-compatible endpoint (Phase 11).
- IGC vendor escalation for the DN recurrence issue (repro tools ready:
  `ie-bug-monitor`, `ie-bug-live`).
