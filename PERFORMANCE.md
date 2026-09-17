# Intel Arc kernel performance and validation

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
