# Phase 4 — Full-attention block

**Date:** 2026-04-25
**Hardware:** Intel Arc Pro B70 (BMG-G31)

## Correctness gate: PASS

### Element-wise (Phase 3 + new sigmoid_gate)
| Test | Result |
|---|---|
| `residual_add 4096` | OK |
| `SiLU 1024 random` | OK |
| `SwiGLU 1024 random` | OK |
| `RMSNorm 1×8` | OK |
| `RoPE partial` | OK |
| **`sigmoid_gate 1024 random`** | **OK** (new for Phase 4) |

### Attention block (kernel-level)
| Test | Result | Notes |
|---|---|---|
| KvCache append round-trip | OK | 4 tokens × 2 kv_heads × 32 head_dim, layer 1 of 2 |
| constant-V → output equals const | OK | softmax sums to 1 invariant |
| causal mask: future garbage ignored | OK | the partial-rotary 0.25 hazard's first-cousin gotcha |
| GQA mapping (group invariance + group split) | OK | within-group identical outputs; across-group differ |
| vs scalar fp32 reference, T=1 | OK | max_abs=1.21e-4 (1 fp16 ULP) |
| vs scalar fp32 reference, T=16 | OK | max_abs=1.94e-4 |
| vs scalar fp32 reference, T=256 | OK | max_abs=2.02e-4 |

**Phase 4 spec:** "matches HF transformers fixture within 1e-2 max-abs" — we match a scalar fp32 reference within 2e-4 (50× under the gate). Diff vs HF transformers itself requires the full forward-pass pipeline (QK-norm + RoPE + Q-proj split + output gate + out_proj wired together), which is Phase 8 work.

## Performance: well under target — Phase 9 backlog

`bench_attention` at Qwen3.6 dims (n_q=16, n_kv=2, d_head=256):

| Case | T | start_pos | ms/iter | ~GFLOPS |
|---|---|---|---|---|
| prefill T=1 | 1 | 0 | 0.037 | 0.4 |
| prefill T=16 | 16 | 0 | 0.101 | 22.0 |
| prefill T=256 | 256 | 0 | 1.13 | 478.5 |
| prefill T=1024 | 1024 | 0 | 7.6 | 1131 |
| **decode @ ctx=4k** | 1 | 4096 | **3.55** | 18.9 |
| **decode @ ctx=32k** | 1 | 32768 | **29.3** | 18.3 |

**Why decode at ctx=32k is 29 ms (= 3 tok/s across 10 full-attn layers):** the naive online-softmax layout has 1 WG per (token, q_head). With 16 Q heads sharing 2 KV heads, each KV head's data is read **8 times** — once per Q head pointing at it. Effective BW = 17 GB/s (2.8% of 608 GB/s peak).

**Fix (Phase 9):** FA-2-style WG-per-kv-head with SLM-shared K/V reads. Expected 8× reduction in cache-read traffic → decode @ 32k drops to ~4 ms/layer × 10 = 40 ms/token = 25 tok/s on attention alone. Still tight; combined with the GEMV/MoE budget the model-level 50 tok/s gate is doable but won't have huge headroom — first phase 9 priority should be this attention rewrite.

The naive kernel **is** correct and serves as the validation reference for the FA-2 variant.

## Files
- `include/ie/kv_cache.hpp` + `src/core/kv_cache.cpp` — KvCache with append API; layout `[L, kv_h, max_ctx, d_head]` fp16.
- `src/ops/elementwise.cpp` — added `sigmoid_gate` for the `attn_output_gate=true` parameterization.
- `src/ops/attention.cpp` — `full_attention` (online-softmax, naive, GQA, causal).
- `tests/unit/attention_test.cpp` — 7 algebraic + reference-diff tests.
- `bench/bench_attention.cpp` — prefill GFLOPS + decode latency at ctx ∈ {0, 4k, 32k}.

## Phase 9 backlog (additions from Phase 4)
- FA-2 attention kernel: WG-per-kv-head, SLM-staged K/V chunks (Bc=64), online softmax along K. Modeled on sycl-tla v0.8 `examples/06_bmg_flash_attention` (research/02 §6.2). Expected to clear decode latency by ~8×.
- Optional: pin all of one decode step's kernels into a SYCL graph so launch overhead amortizes across the 40 layer-blocks instead of being paid per-call.
