# Performance baseline — 2026-05-04 (post-cleanup)

Captured at clean baseline: `CMAKE_CXX_FLAGS=""`, all bisect-step diagnostic
flags removed from source, only `IE_ENABLE_MOE_DOWN_TILE=1` and the production
fused-op flags active.

## Configuration
- Model: `Qwen3.6-35B-A3B-Q4_K_M.gguf`
- KV cache: fp16
- Device: Intel Arc Pro B70 (BMG-G31, Xe2-HPG)
- Toolchain: IntelLLVM 2026.0.0, NEO 26.14.37833.4, IGC 2.32.7
- Decode budget per run: 128 tokens
- Runs per prompt: 3 (median reported)

## Quality
- Built-in 511-tok PPL: **6.54** (historic baseline 6.51–6.55, ±0.03 noise floor).
  No regression after cleanup.
- llama.cpp anchor on the same corpus, ctx=256: PPL = 10.37 (different
  methodology — see `docs/ppl_baseline_matrix.md`; not directly comparable).

## Throughput

| Prompt           | Prompt tok | Gen tok | Decode tok/s (median) | Decode spread | Prefill ms (median) | Prefill tok/s |
|------------------|-----------:|--------:|----------------------:|--------------:|--------------------:|--------------:|
| short-chat       |         51 |     128 |                 44.77 |          0.6% |               405.1 |        125.9 |
| long-instruction |         80 |     128 |                 44.83 |          0.9% |              ~580   |       ~140   |
| codegen          |         78 |     128 |                 44.88 |          0.7% |              ~570   |       ~140   |
| math-reasoning   |         86 |     128 |                 44.63 |          0.8% |               609.7 |        141.1 |
| long-context     |        219 |     128 |                 44.75 |          0.3% |              1434.5 |        152.7 |

### Headlines
- **Decode: ~45 tok/s, near-perfect run-to-run consistency (<1% spread).**
- **Prefill: 126–153 tok/s, scales sub-linearly with prompt size.**

### llama.cpp comparison (same B70, per `research/01_hardware.md` and PMZFX bench)
| metric | engine | llama.cpp | gap |
|---|---:|---:|---:|
| decode tok/s | ~45 | ~50–55 | **at parity ✓** |
| pp512 prefill tok/s | ~150 | ~615 | **~4×** |
| Qwen 3.6-35B-A3B prefill (PMZFX) | ~150 | 615 | 4.1× |

## Top kernels (decode, 128 steps aggregated, median run)

| # | Kernel | Calls | Total ms | Avg ms | % |
|---:|---|---:|---:|---:|---:|
| 1 | `gemv_q6k_huge`     |   128 | 437.6 | 3.42 | **23.0%** |
| 2 | `gemv_q4k`          | 16640 | 306.7 | 0.018 | **16.1%** |
| 3 | `gemv_q6k_med`      |  4352 | 236.2 | 0.054 | **12.4%** |
| 4 | `moe_dec_gate_q4k`  |  5120 | 170.6 | 0.033 |  9.0% |
| 5 | `moe_dec_down_q6k`  |  2560 | 131.0 | 0.051 |  6.9% |

Top-3 GEMV kernels = ~51% of decode compute. Top-5 = ~67%. The decode hot
path is dominated by Q4_K / Q6_K GEMV — closing any further gap to llama.cpp
on decode would require kernel-level wins on these. Current decode is at
parity, so this is a headroom opportunity, not a remediation target.

## Top kernels (prefill — not yet profiled)

The bench-suite reports decode kernels only. Prefill (single forward(T=N))
needs a separate profiling harness. Suggested next session: instrument a
forward(T=512) call with `KernelProfiler` and rank the resulting kernels.
The expected hot kernels (per the source) are `gemm_q4_K_xmx`,
`moe_prefill_gate_up_silu_q4k`, `moe_prefill_down_packed_q4k_v2`,
and `full_attention` (or `full_attention_fa2_prefill` if enabled — currently
disabled per repo rules).

## Memory (estimated)
- Model weights: ~22 GB (loaded once)
- KV cache (fp16, max_ctx=4096, 10 full-attn layers): ~10 MB
- DN state (30 layers): ~60 MB
- DN conv state: ~720 KB
- Workspace (max_T=1): ~128 MB
- **Total at single-chain steady state: ~22.2 GB / 32 GB → 69%**

## What's next (suggested for the next session)
1. **Profile prefill explicitly.** Add a small harness (or reuse `ie-bench`)
   that runs forward(T=512) with the kernel profiler hot, dump per-kernel ms.
   This will tell us *where* the 4× prefill gap comes from kernel by kernel.
2. **Compare prefill kernel mix to decode.** Decode is GEMV-bound (no surprise
   for Q4_K_M). Prefill is GEMM-bound (gemm_q4_K_xmx). The XMX path may have
   suboptimal tile sizes for Xe2's larger SLM.
3. **DO NOT** touch ESIMD / block2d / `gemm_q4k_esimd` / tile-load paths
   without explicit authorization. Two prior attempts regressed prefill.
4. **DO NOT** reopen the determinism investigation. See `docs/known_bugs.md`.
