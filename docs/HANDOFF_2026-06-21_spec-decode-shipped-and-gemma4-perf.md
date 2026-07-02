# Session handoff — 2026-06-21: MTP spec-decode SHIPPED + dense-parity settled + Gemma-4 perf target

Read `MASTER_DEV_PLAN.md` banner first. Strategic frame: **going public, Intel-facing** (memory
`reference_b70_competitive_landscape`). This session ran one long thread: make dense decode competitive →
proved the kernel's at the BW wall → shipped speculative decode → then scoped Gemma-4 perf.

## What shipped this session (all committed on `main`, crown 6.4527 bit-exact throughout, tree clean)

### 1. Dense-Q4 decode vs llama — SETTLED (honest)
Built a fresh **llama.cpp SYCL** build (`~/llama.cpp/build-sycl`, icpx/oneAPI, HEAD fdc3db9b6) — reuse it
for all future apples-to-apples. On the IDENTICAL 27B-Q4_K_M, single B70:
- **prefill: WE WIN 348.6 vs 287.7 = 1.21×.** decode: 15.2 vs llama 23.2.
- Chased decode with TWO gated kernel attempts, **both measured + reverted**: int-dot wire-up (neutral),
  SoA-Q4 repack (+6% decode / −24% prefill = net-negative). KEY: our AoS `gemv_q4_K` is **already
  389 GB/s = 64% of 608 = llama's level** ("46%" was overall-effective, incl F16-expansion bloat). **The
  Q4 GEMV is at the silicon wall — 90% isn't physical for a quantized GEMV (our fp16 GEMV tops at 82%).**
  Memory `dense_q4_parity_2026-06-20`. Don't re-tune it.

### 2. MTP self-speculative decode — SHIPPED as `ie run --spec` (the headline)
Decode is BW-bound + the kernel's maxed → only amortization beats it. The GGUF ships an unused native
**NextN/MTP head (`blk.64`)** = a built-in EAGLE self-draft. Built bottom-up (8 commits, all additive):
STEP 0 acceptance **0.5669** (`bac465e`) → 1a verify primitive all-pos-logits+pre-norm-hidden (`ba74240`)
→ 1b/2 lossless loop proven (`3638f65`) → 3 batched-T int-dot verify kernels, verify 1071→120 ms 8.8×
(`5046ac2`) → 4 per-position DeltaNet checkpointing, kills the rollback re-forward + FIXES the §1
non-determinism (`f8783f6`) → wired into Engine (`f7a46a7`).
**`ie run <gguf> --spec [--spec-k K]`:** in-engine LOSSLESS (token-identical to `ie run`, cyber+factual),
**cyber 15.5→17.4 = 1.13×** (tool 1.20×; engine adds tokenizer/UTF8/callback), default path byte-identical
(MTP head only loads with --spec), stops+streaming clean, single-GPU kQwen35Dense only. Memory
`project_mtp_spec_decode`. Realistic ceiling ~1.2-1.5× (verify W6A8 ALU-co-limit + MTP acceptance 2.6/verify;
K=4 optimal). Batched-T int-dot GEMV doubles as small-batch serving infra.

### 3. Gemma-4 26B-A4B — apples-to-apples measured, tune target localized (NOT yet built)
Same llama-SYCL, identical QAT Q4_0 GGUF, 1×B70: **prefill ours 114.4 vs llama 828.7 = 7.2× behind**
(down from ~15× pre-fused-MoE). llama decode 54.75; **our decode UNMEASURED** (ie-bench mis-detects gemma4
as qwen35moe; gemma4-gen times prefill only). **The 7.2× is the attention/dense projections, NOT the MoE**
(MoE is already fused int-dot; projections run fp16-dequant `gemm_q4_0` or a per-token `gemv_q4_0` fallback
on large-K o-proj — `gemma4.cpp:357-375`). Memory `project_gemma4_arch` (updated).

## NEXT (priority order)
1. **Gemma-4 int-dot W4A8 projections** (the 7.2× lever): generalize `moe_prefill_proj_q4_0_q8` to the
   dense/attn q/k/v/o + shared-FFN projections. AND instrument a real gemma4 decode bench (fix the
   tooling gap). Goal = CREDIBLE (~2-3× of llama), not fastest. Gemma is the BREADTH pillar, not the moat.
2. Wire **Gemma-4 31B dense** (downloaded, same forward −MoE) — cheap day-one-breadth win.
3. `ie serve --spec` passthrough (only `ie run` has it).
4. (optional, deferred) lower-ALU verify kernel for spec-decode (~1.2→~1.4×, hard).
5. Publish-prep: benchmark matrix + OpenVINO baseline + packaging (launch article drafted
   `docs/public/2026-06-15-mach-x-launch-article.md`).

## Scoreboard for publish
Win dense prefill 1.21× · at-parity dense decode + lossless 1.13× spec-decode on top · beat every
competitor on the DeltaNet arches (the moat) · Gemma-4 runs day-one on Arc (perf = credible-not-fastest,
the projection int-dot is the open lever).

## Method notes (carry forward)
- ONE GPU workload at a time — serialize every model load. (Double-loaded crown gates this session → one
  DEVICE_LOST exit-1; don't launch a 2nd GPU job before the first finishes.)
- llama apples-to-apples: `~/llama.cpp/build-sycl/bin/llama-bench -m <gguf> -p 512 -n 64 -ngl 99` with
  `ONEAPI_DEVICE_SELECTOR=level_zero:0` (single B70). It HAS gemma4 + qwen35.
- `ie-bench` mis-detects gemma4 (loads as qwen35moe) — use `ie-gemma4-gen ... bench` (prefill only).
- Measure before claiming: two dense kernel attempts looked obvious, measured neutral/negative, reverted.
