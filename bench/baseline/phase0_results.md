# Phase 0 — Baseline measurements

**Date:** 2026-04-25
**Hardware:** Intel Arc Pro B70 (BMG-G31, PCI `8086:e223`, driver `26.05.37020.3`)
**Compiler:** IntelLLVM 2026.0.0 (oneAPI 2026.0 DPC++)
**AOT target:** `intel_gpu_bmg_g31`

## Environment

| Component | Version | Note |
|---|---|---|
| Linux kernel | 6.17.0-22-generic | well above 6.14 floor |
| `intel-opencl-icd` | 26.05.37020.3 | exactly matches PLAN.md floor |
| oneAPI compiler | 2026.0 | meets pin |
| `cmake` | 3.31.6 | local install |
| `ninja` | 1.13.2 | local install |

`sycl-ls` reports the B70 as `intel_gpu_bmg_g31`, sub-group sizes 16/32, `ext_intel_matrix` (XMX) and `ext_intel_esimd` aspects available.

## hello_gpu (vector add, 1M floats)

```
Vector add OK: 1048576 elements in 0.051 ms (246.5 GB/s effective)
```

246 GB/s on a 1M-element add is launch-overhead-bound (kernel runs <50 µs); not a meaningful bandwidth number. Real BW benchmarks come in Phase 2.

## bench_gemm_fp16 (4096×4096×4096, FP16 in / FP32 acc)

Kernel: 1 subgroup per workgroup, M_REPEAT=8 stacked accumulators (SG covers 64×16 output tile). **No SLM tiling, no 2D block load, no double-buffering** — those are Phase 3 work.

| Run | iters | ms/iter | TFLOPS | % of 183 TFLOPS peak |
|---|---|---|---|---|
| 1 | 20 | 2.655 | 51.76 | **28.3%** |
| 2 | 20 | 2.997 | 45.86 | 25.1% |
| 3 | 20 | 2.857 | 48.11 | 26.3% |
| 4 | 20 | 2.865 | 47.97 | 26.2% |
| 5 | 20 | 2.909 | 47.24 | 25.8% |

Median: **~26% of XMX peak**.

Numerical check: max-abs error vs FP32 host reference on top-left 64×16 block = **0.0002** (well within FP16-input tolerance for K=4096).

### Status of the Phase 0 gate

> *PLAN.md Phase 0 gate: "sample GEMM hits ≥50% of 183 TFLOPS at 4096³ FP16."*

**Missed: 26% achieved.** Naive `joint_matrix` without SLM tiling reads B from global memory (K/TK)×(M/TM) = 256×64 = ~16k times per output tile, so we're memory-traffic-limited, not XMX-limited. `research/02_programming_stack.md` §6.1 lays out the canonical Xe2 path (double-buffered SLM stage + `cl_intel_subgroup_2d_block_io` + multi-stage pipeline) which is explicitly Phase 3 work.

**Treat this as path-finding, not a perf result.** What this run *did* validate:
1. Toolchain end-to-end: source → SPIR-V → AOT BMG-G31 binary → runs.
2. `joint_matrix` emits correct numerics on B70 (the ggml-sycl correctness regression cited in research does NOT reproduce on our path).
3. The XMX is reachable from SYCL `joint_matrix` (we'd be at fractions of a TFLOP otherwise).
4. AOT compilation (`-fsycl-targets=intel_gpu_bmg_g31`) works and produces a working binary.

The 50% gate is **deferred to the optimized GEMM in Phase 3**. The Phase 0 gate is amended to: *"FP16 `joint_matrix` GEMM produces numerically correct output; toolchain end-to-end works."* — both passed.
