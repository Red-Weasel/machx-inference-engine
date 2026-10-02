# Intel Arc kernel performance and validation

The newest campaign is at the end: [v0.2.6, native integer-dot kernels for Q8_0](#v026-native-integer-dot-kernels-for-q8_0).
The sections before it describe the GLM-5.3-Flash campaign of September 2026.

This update tunes GLM-5.3-Flash and shared SYCL operators on two Intel Arc Pro
B70 GPUs through Level Zero, using IntelLLVM 2026.1.1. Kernel improvements are
reported separately from model throughput. Other devices and unmeasured shapes
retain the relevant fallback paths; they were not qualified by this B70 campaign.

## Implemented changes

- Shape-aware Q8 projection workgroups, exact small row counts, vector loads,
  and single-grid prefill batches. Returned events cover every tile on
  out-of-order queues; unaligned inputs retain bounded scalar loading.
- Smaller FP16 XMX tiles for measured small-M expert GEMM shapes. Output storage
  retains the existing requirement to round the row allocation up to eight.
- Ordered load-ahead for GLM router logits and MLA key/value projections.
  Sparse MLA shares latent tiles across heads and reuses softmax coefficients.
- Expert top-k specializes common expert counts and caches unbiased probabilities.
  Ragged expert counts, larger arrays, invalid counts, all-zero sigmoid
  normalization, and empty-work dependencies are handled explicitly.
- KDA recurrence uses SIMD16 with 256 GRFs on B70 Level Zero. Compiler assembly
  reports 3,136 spill bytes in the reference and no spill directive in the tuned
  kernel. Arithmetic and persistent-state layout remain unchanged.
- Indexer bitonic sort specializes measured padded extents and exchanges short
  distances through subgroup registers. Packed-key ordering, ties, causal masks,
  and padding remain exact. Generic, radix and scan fallbacks remain available.
- Causal convolution avoids history races on short chunks and invalid state
  access for a one-tap kernel.
- The GLM standalone runner defaults to pipelined two-card prefill and selects
  MTP pipedraft for eligible sustained generation. Warm overflowing expert
  batches use smaller waves to retain more cached weights.

## Measured kernel results

These latest three changes were measured against the retained Q8/router build
at the start of the campaign. They are not comparisons against an older GitHub
checkout or a different engine.

| Kernel | Isolated production-kernel speedup | Speedup inside GLM at 24K |
|---|---:|---:|
| Expert top-k | 1.49–1.68× | 1.61× |
| KDA recurrence | 1.93–2.62× | 1.33× |
| Attention indexer | 1.24–1.85× at decode shapes | 1.24× |

The in-model bucket comparison used 64 generated tokens with pipedraft disabled.
Q8 register forcing and parallel Sinkhorn were rejected because their primary
shapes did not improve. Neither experiment changes the shipped dispatch.

## Whole-model qualification

The original warm 8K comparison measured prefill **−0.47%** and decode **−2.55%**;
the 24K comparison measured **−1.87%** and **−0.71%**, respectively. An initial
combined-build first prefill chunk also incurred approximately 5.7 seconds of
extra first-use time. A later repeat did not show that cost. No compilation
trace was taken, so its cause is not asserted.

Regression isolation used 14 complete 8K-prompt / 512-token-generation runs:

| Configuration | Baseline decode | Combined decode | Interpretation |
|---|---:|---:|---|
| Original pinned configuration, two repeats each | 13.69 tok/s | 13.69 tok/s | No consistent combined regression reproduced |
| Larger pin-memory reserve, one pair | 9.26 tok/s | 9.38 tok/s | Rejected; worse overall performance and I/O |
| Unpinned banks, ABBA | 8.34 tok/s | 8.48 tok/s | Combined +1.69%, but the mode is slower overall |

The initial four-build order was baseline, router, router+KDA, combined, combined,
router+KDA, router, baseline. The same KDA binary ranged from 36.03 to 41.18 seconds
for decode. Slower runs showed increased expert staging, CPU work and physical
reads. Timers overlap and do not prove a unique cause. Nonlinear host/cache drift
also confounds aggregate prefill comparisons.

The larger-reserve sequence was stopped after its first pair because the
intervention worsened performance. No four-run confidence is claimed for it.
All 14 model runs exited successfully; all 13 comparisons to the first baseline
matched generated text, draft counts and expert-cache counts exactly. Keep the
original runtime settings. A small kernel-related effect remains unresolved
within the timing variation; **a combined whole-model speedup is not established**.

## Reproducing checks

After configuring with Intel's SYCL compiler, build the kernel gates:

```bash
cmake --build build --target g5_router_topk_test kda_recurrence_test \
  indexer_topk_test g5_router_test gemv_q8_soa_dual_test \
  gemm_fp16_shape_test conv1d_state_test -j2
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ctest --test-dir build \
  -R '^(g5_router_topk_test|kda_recurrence_test|indexer_topk_test|g5_router_test|gemv_q8_soa_dual_test|gemm_fp16_shape_test|conv1d_state_test)$' \
  --output-on-failure
```

The latest top-k, KDA and indexer gates cover 176, 102 and 240 configurations
across both B70s. KDA checks four dependent state updates per configuration.
Selection gates use frozen GPU references and independent CPU oracles; tests
exercise out-of-order dependencies, guards, ragged shapes and dispatch boundaries.

A matching full-model comparison requires the same GLM-5.3-Flash UD-Q4_K_XL
GGUF and WikiText corpus for both binaries. Example for each saved build:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
IE_G5_CPU_MISS=1 IE_G5_PIN_BANKS=1 IE_G5_PIN_MAX_GIB=95 \
./scripts/ie-run-guarded --mem 220G --timeout 900 \
  ./build/tools/ie-glm5next-run /path/to/model-00001-of-00006.gguf \
  --gpus 2 --ctx 16384 --ppl /path/to/wiki.test.raw \
  --prefill 8000 --chunk 1024 --ngen 512
```

Run one GPU workload at a time, avoid overlapping compiler work, record actual
pinned-layer allocation, keep slow runs, compare exact continuations and use
alternating controls. Separate cold/first-use observations from steady-state
claims. Host-resident expert workloads need repeated model measurements in
addition to isolated kernel timings.

<a id="v026-native-integer-dot-kernels-for-q8_0"></a>
## v0.2.6: native integer-dot kernels for Q8_0

Measured October 1, 2026 on two Arc Pro B70 cards (compute-runtime 26.35), on community fine-tunes of the
Qwen3.6-35B-A3B class and of Qwen3.8-27B (Q8_0) and on Qwen3.8-Flash. Kernel times are device time per launch on
card 0; model figures are single runs unless a line says A-B-A.

### What was wrong

The engine's portable integer-dot helper (`ie::dp4a_ss` over vector-loaded words) was lowered by the compiler through
byte extracts and inserts: 12 byte moves per operand word. Counted in the B70 code of the previous kernels, per
integer dot:

| Kernel (Q8_0) | Instructions per dot, before | Of them byte moves | After |
|---|---:|---:|---:|
| MoE prefill down | 33.8 | 24 | about 4.25 |
| MoE prefill gate+up | 9.7 | about two thirds | about 2.7 |
| MoE decode / lane rows, gate+up and down (dot blocks) | 20.4 and 27.4 | 18 and 24 | 2.5 and 3.2 |
| GEMV, one token (dot block) | 28.0 | 24 | 4.0 |
| GEMV, 2–16 rows (dot blocks) | 14.8–21.1 | 12.75–18 | 2.05–3.12 |

The new kernels use the compiler's native integer dot (`include/ie/dp4a_native.hpp`). The MoE kernels also keep a
tile's weights in registers and compute several output columns per subgroup; the GEMVs change only the dot. Each
kernel lives in its own source file beside the one it replaces, with a switch that selects the old one (`IE_Q8_MOE_DOWN_V2`, `IE_Q8_MOE_GATEUP_V2`, `IE_Q8_MOE_DECODE_V2`,
`IE_Q8_SOA_BATCHED_V2`, `IE_Q8_SOA_GEMV_ND`, `IE_Q8_SOA_GEMV_G_ND`; `=0` each). They keep the old kernels' per-lane
partial sums, reduction trees and final rounding, so their output is bit-identical.

### Measured kernel results

| Kernel | Shape | Before | After | Speedup |
|---|---|---:|---:|---:|
| MoE prefill down | 35B-A3B class, 2,048 rows | 17.56 ms | 3.15 ms | 5.57× |
| MoE prefill down | skewed expert load | 43.27 ms | 7.13 ms | 6.07× |
| MoE prefill gate+up | 35B-A3B class, 2,048 rows | 10.57 ms | 6.26 ms | 1.69× |
| MoE prefill gate+up | skewed expert load | 25.19 ms | 13.86 ms | 1.82× |
| MoE decode gate+up, 1 / 2 / 4 / 8 / 16 rows | 35B-A3B class | 16.6 / 50.8 / 111.7 / 215.2 / 423.5 µs | 10.3 / 45.6 / 107.8 / 212.7 / 415.8 µs | 1.61 / 1.11 / 1.04 / 1.01 / 1.02× |
| MoE decode down, 1 / 2 / 4 / 8 / 16 rows | 35B-A3B class | 15.5 / 33.8 / 69.4 / 147.3 / 288.8 µs | 7.8 / 19.8 / 39.3 / 96.7 / 181.4 µs | 1.99 / 1.71 / 1.77 / 1.52 / 1.59× |
| GEMV, 16 rows, single / dual launch | 27B, 35B-A3B and Qwen3.8-Flash shapes | — | — | 1.11–1.45× / 1.11–1.37× |
| GEMV, 8 rows | the same | — | — | 1.15–1.43× / 1.11–1.37× |
| GEMV, 4 rows | the same | — | — | 1.14–1.88× / 1.10–1.88× |
| GEMV, 2 rows | the same | — | — | 1.02–3.28× / 1.02–1.74× |
| GEMV, one token, large shapes | 27B feed-forward and attention-q, Qwen3.8-Flash LM head | — | — | 1.00–1.04× |
| GEMV, one token, small shapes | 27B k/v, 35B-A3B projections and shared expert, Qwen3.8-Flash linear attention | — | — | 1.16–1.58× |

Two estimates did not hold. The instruction count predicted about 3.5× for the 16-row GEMV and the device gave
1.11–1.45×: after the change that kernel streams weights at 116–223 GB/s, so it is bound by something other than
instructions or bandwidth (not profiled). The prefill gate+up was estimated at 2.5–3.5 ms and measures 6.26 ms, and
the decode gate+up gains nothing from 4 rows up. The large one-token GEMVs were already at 505–600 GB/s.

### Whole-model results

| Model | Workload | Before | After |
|---|---|---:|---:|
| 35B-A3B class | one request, cold prompt of 2,038 / 8,065 / 16,369 / 32,161 tokens, wall time, A-B-A | 2.20 / 9.04 / 22.72 / 60.86 s | 1.41 / 5.96 / 16.44 / 48.38 s |
| 35B-A3B class | the same with the down kernel only | — | 1.61 / 6.74 / 18.06 / 51.64 s |
| 35B-A3B class | `--parallel 16`, short prompts, 1 / 4 / 8 / 16 requests, A-B-A on the decode kernels | 80.8–81.1 / 230.0–230.8 / 315.2–316.7 / 390.6–391.3 tok/s | 81.9–82.4 / 236.1 / 331.9 / 414.1 tok/s |
| 35B-A3B class | agent replay (15 workers, 3 turns, then a 15-wide shared-prompt wave), MoE kernels | 214.9 s, 66.3 tok/s | 169.6 s, 84.2 tok/s |
| 35B-A3B class | the same replay, GEMV kernels | 144.8 s, 99.0 tok/s | 139.7 s, 101.2 tok/s |
| Qwen3.8-27B | `--ctx 8192 --parallel 16`, 1 / 4 / 8 / 16 requests, A-B-A on the GEMV kernels | 17.1–17.2 / 61.5 / 92.3 / 113.7–113.8 tok/s | 17.2–17.3 / 64.5 / 111.3 / 153.5 tok/s |
| Qwen3.8-Flash | `--ctx 8192 --parallel 16`, 16 requests, A-B-A on the GEMV kernels | 124.6–124.8 tok/s | 128.0 tok/s |

The down kernel carries about three quarters of the prefill gain. Single-request decode moves by less than 1 % with
the GEMV kernels and by 1.4 % with the MoE decode kernels; the gain is in the lanes' row steps, largest on the dense
27B.

### Validation

- Kernel tests compare each new kernel with the old one bit for bit, check that every output is written, and bound
  both against a host fp64 reference: 7 + 8 + 24 MoE cases and 156 + 38 GEMV cases passed on card 0. One MoE decode
  case first failed on a sample whose reference lies beyond the fp16 range, where both kernels store the same
  infinity; the test now accepts that, and the 24 cases pass.
- Greedy replies are identical with the switches on, off, and on the previous build: a five-prompt set at
  `--parallel 1` (one prompt of 32,979 tokens) and four prompts solo and batched at `--parallel 4`, on the 35B-A3B
  class for both kernel families and on Qwen3.8-27B and Qwen3.8-Flash for the GEMVs.
- Perplexity on the 35B-A3B class (a 12,288-token prefill, then 511 scored tokens of wikitext-2) is 2.3529, the same
  negative log-likelihood to six decimals in every configuration. Needles at 32K and 54K with the MoE kernels: 6 of
  6, the same replies as before.
- Not covered: the unit tests on card 1 and the speculative-verify paths. The opt-in `IE_GEMV_SMALLK` kernel is
  unchanged; on an OpenCL CPU device it leaves the valid columns of a partial last work-group unwritten when the
  column count is not a multiple of 8 (the model shapes it serves are multiples of 8).

### Reproducing

```bash
cmake --build build --target moe_q8_down_v2_test moe_q8_gate_up_v2_test moe_q8_decode_v2_test \
  gemv_q8_soa_batched_v2_test gemv_q8_soa_nd_test -j2
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/tests/moe_q8_down_v2_test            # correctness, old kernel == new kernel
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/tests/moe_q8_down_v2_test --bench    # device time per launch, old and new
```

The other four tests take the same `--bench` argument. For a model-level A/B, start the same build twice, once with
the switches at `=0`, and compare; the first request of a new build compiles the new kernels once, so do not time it.
