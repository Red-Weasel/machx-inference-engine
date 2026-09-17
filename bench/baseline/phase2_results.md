# Phase 2 — Dequant kernels baseline

**Date:** 2026-04-25
**Hardware:** Intel Arc Pro B70 (BMG-G31)
**Driver:** compute-runtime 26.05.37020.3, IGC 2.30.1
**Compiler:** oneAPI 2026.0 DPC++

## Gate result: PASS (correctness + Q4_K throughput)

### Correctness — synthetic 1024 random super-blocks per format

| Format | max_abs | max_rel | bit-exact mismatches | Tolerance | Result |
|---|---|---|---|---|---|
| Q8_0 | 0     | 0     | 0 | 0 (require bit-exact fp16) | OK |
| Q4_K | 0     | 0     | 0 | 1 fp16 ULP (~5e-4 rel)     | OK |
| Q5_K | 0     | 0     | 0 | 1 fp16 ULP                 | OK |
| Q6_K | 0     | 0     | 0 | 1 fp16 ULP                 | OK |

The fp32 reference (transcribed from `ggml-quants.c`) is rounded through `sycl::half` rtne, then bit-compared to the device output. **Bit-exact agreement** across all formats.

### Correctness — real Qwen3.6-35B-A3B tensors

| Format / tensor | shape | mismatches |
|---|---|---|
| Q4_K / `blk.0.attn_gate.weight` | [2048, 4096] (8.4M elem) | 0 |
| Q4_K / `blk.0.ffn_gate_exps.weight` | [2048, 512, 256] (268M elem) | 0 |
| Q6_K / `blk.0.attn_qkv.weight` | [2048, 8192] (16.8M elem) | 0 |
| Q6_K / `blk.0.ffn_down_exps.weight` | [512, 2048, 256] (268M elem) | 0 |
| Q6_K / `blk.3.attn_v.weight` | [2048, 512] (1.0M elem) | 0 |
| Q6_K / `output.weight` | [2048, 248320] (508M elem) | 0 |

A combined ~1.07 billion model elements decoded, **zero mismatches**. Real Qwen weights tend to use the full dynamic range of K-quant scales/mins, which exercises the 6-bit packed-scale unpack and qh-bit lookup paths in ways uniform-random data does not — a much stronger signal than synthetic.

### Throughput — bench_dequant (256 MiB packed per format, 5 warmup + 20 iters)

| Format | ms/iter | input GB/s | output GB/s | Gate | Result |
|---|---|---|---|---|---|
| Q8_0 | 2.64 | 102 | 192 | (no gate) | — |
| **Q4_K** | **2.38** | **113** | **401.7** | **≥ 350 GB/s** | **PASS** (115%) |
| Q5_K | 3.07 | 87 | 255 | (no gate) | — |
| Q6_K | 1.93 | 139 | 339 | (no gate) | — (97% of 350) |

**Combined input + output BW for the gate kernel (Q4_K) = 514 GB/s ≈ 85% of B70 peak (608 GB/s).**

## Optimization journey (for the postmortem)

The first cut used a 1-element-per-work-item layout with WG = 256 items/block. It hit **148 GB/s** out for Q4_K — far from the gate. Two layout passes brought it to the gate without changing the dequant math:

| Q4_K layout | items/WG | outputs/item | qs bytes read/item | GB/s out | speedup |
|---|---|---|---|---|---|
| 1-elem (initial) | 256 | 1 | 1 (each byte read 2× in WG) | 148 | 1.0× |
| 2-elem            | 128 | 2 | 1 (no redundancy)            | 264 | 1.78× |
| **4-elem (final)** | **64**  | **4** | **2** (vec-loadable)         | **401.7** | **2.71×** |

Same idea applied to Q5_K (reading qh once per low/high pair) and Q6_K (4 outputs per item from 2 ql + 1 qh). Q5_K could likely hit ~350 GB/s with a 4-elem layout too, but no gate to chase.

**Why the win:** the 1-elem version made the GPU re-read each `qs` byte twice (once for low nibble, once for high nibble of the same byte, in different work-items). Moving to 2-elem killed the redundancy; 4-elem additionally vectorizes the byte read into a 16-bit chunk and amortizes scale-table lookup over more outputs. No special intrinsics, no SLM cleverness, no new compiler flags — just the right per-thread work granularity.

## Files

- `include/ie/quant_blocks.hpp` — packed-block layouts + fp16⇄fp32.
- `include/ie/dequant_ref.hpp` — host CPU reference (transcribed from `ggml-quants.c`).
- `include/ie/dequant.hpp` — public device dequant API.
- `src/ops/dequant_kernels.cpp` — SYCL kernels (Q8_0, Q4_K, Q5_K, Q6_K).
- `tests/unit/dequant_test.cpp` — synthetic + real-tensor diff harness.
- `bench/bench_dequant.cpp` — input/output GB/s gate.

## Note for Phase 3

Standalone dequant→fp16 was **not** going to be the production decode path. Phase 3's dequant-fused-into-XMX kernel reads packed bytes, decodes nibbles in registers, and feeds `joint_matrix` directly — there's no fp16 spill at all. These kernels remain as the trusted golden for the fused-path correctness diffs and as a fallback when XMX isn't available (e.g. for very small N where dispatch overhead dwarfs the matmul).
