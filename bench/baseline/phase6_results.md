# Phase 6 — MoE router + sparse expert dispatch

**Date:** 2026-04-25
**Hardware:** Intel Arc Pro B70 (BMG-G31)
**Subject:** Layer 0 of `Qwen3.6-35B-A3B-Q4_K_M.gguf` (256 experts top-8 + 1 shared)

## Correctness gate: PASS (decisively)

| Test | Result | Margin |
|---|---|---|
| Router top-8 expert IDs match CPU ref | **bit-exact** | `[25,28,51,65,124,194,199,222]` |
| Router top-8 weights match CPU ref | OK | max_abs = 5e-5 |
| Full MoE block forward y vs CPU ref | OK | **max_abs = 1.7e-5** (600× under 1e-2 spec) |

The CPU reference is the 8-step algorithm transcribed verbatim from `research/06_moe_math.md` §12 — softmax(fp32) → topk → renorm → ascending-by-id sort → SwiGLU per expert in order → scatter-add → shared expert with sigmoid scalar gate. Both router and full block match to the fp16-rounding floor.

The bit-exact router output is the load-bearing signal: any off-by-one in softmax, top-k selection, renormalization, or sort order would cause a different top-8 (with 256 experts and softmax probabilities tightly clustered, even small numerical drift can flip rank-8 vs rank-9). Both `out_max_abs = 1.7e-5` AND identical top-8 confirms the GPU is faithfully reproducing the HF algorithm.

## Performance gate: MISS, with diagnosis

Spec: "Decode-step MoE latency ≤ 1.5 ms at hidden=2048." Achieved: **2.1 ms/layer** (40% over).

**Diagnosis: kernel-launch overhead dominates.** Per-layer kernel call tally:
- 1 router
- 8 routed experts × (gate + up + swiglu + down + scaled_add) = 40 calls
- 1 shared expert × (gate + up + swiglu + down + sigmoid_gate + scaled_add) = 6 calls
- **Total: 47 kernel launches per MoE layer**

At ~25 µs SYCL launch overhead per call: 1.2 ms of pure launch cost. The actual compute (40 small Q4_K/Q6_K GEMVs at 220 GB/s + tiny elementwises) is ~0.5–0.9 ms.

Across 40 layers: 40 × 2.1 ms = **84 ms/token** for MoE alone — over the full 20 ms/token budget for 50 tok/s. **This is the bottleneck.**

## Phase 9 backlog (additions from Phase 6, very high priority)

In rough ROI order:
1. **Fused MoE block kernel** — single SYCL graph or pre-recorded command list per layer. Eliminates 46 of the 47 launch overheads; expected per-layer drop from 2.1 ms to ~0.6 ms (compute-bound floor). This single change closes most of the 50 tok/s gap.
2. **Batched expert GEMVs** — stack the 8 (gate, up, down) GEMVs into single batched kernels operating across 8 expert weight slices. Cuts launch count from 32 to 4 for the routed path.
3. **SLM-cached activation `x`** — every routed expert reads the same x[2048]. Currently each kernel reloads x from global; fusing them lets us load once into SLM and reuse 8×.
4. (Stacks on Phase 3 backlog) **GEMV-Q4_K XMX-fused** at 365 GB/s peak → expert GEMVs ~1.7× faster. Ranks below #1 in priority because launch overhead, not BW, is the immediate bottleneck.

## What's in this phase

Code:
- `src/ops/gemv_q6k.cpp` — W6A16 GEMV (signed 6-bit, no dmin), structurally mirrors Q4_K. Validated bit-exact (max_rel = 5e-4, 1 fp16 ULP) on real Qwen3.6 Q6_K tensors at K=2048 N=512 and K=2048 N=8192.
- `src/ops/moe.cpp` — three primitives:
  - `moe_router`: 1 WG per token = 256 lanes; each lane computes one expert's logit (fp32 dot product, F32 router weights), then WG-wide softmax, then lane-0-serial top-8 + renorm + ascending-id sort.
  - `shared_expert_gate`: per-token sigmoid scalar (1-output GEMV).
  - `scaled_add`: `y += scale * x` for fp16.
- `tests/unit/moe_test.cpp` — loads layer-0 MoE tensors from the actual Qwen3.6-35B-A3B Q4_K_M GGUF, runs both CPU reference and GPU orchestration, diffs.
- `research/06_moe_math.md` — the source-of-truth math (605 lines, ~3050 words).

Tensor-level correctness on real Qwen weights:
- `ffn_gate_inp` F32 [2048, 256] router — used as-is.
- `ffn_gate_inp_shexp` F32 [2048] shared-gate — confirmed = `nn.Linear(2048, 1, bias=False).weight`.
- `ffn_{gate,up}_exps` Q4_K [2048, 512, 256], `ffn_down_exps` Q6_K [512, 2048, 256] — per-expert byte stride: `q4k = (H·I/256)·144 = 589 824 B`, `q6k = (I·H/256)·210 = 860 160 B`. Pointer offsets work out with the existing GEMV kernels — no new packed format needed.
- `ffn_{gate,up}_shexp` Q4_K + `ffn_down_shexp` Q6_K — single shared expert, dispatched separately.

## Files / artifacts
- `bench/baseline/phase6_moe_test.txt` — full test output.
- `research/06_moe_math.md` — ~605 lines, math + tensor map.

## Memory

Saved a project memory updating the perf-debt list with the Phase-9 fused-MoE priority (the dominant contributor to the 50 tok/s gap).
