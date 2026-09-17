# DeepSeek kernel optimization pass, September 9, 2026

## Opportunities from the measured profile

Hardware: two Intel Arc Pro B70 GPUs. Checkpoint: DeepSeek V4 Flash
Vision-Exp Abliterated, native MXFP4 experts and F16/F32 remaining GGUF
weights. The text-only synthetic benchmark and original profile are saved in
`results/deepseek4-2026-09-09/`. GPU timings below sum both cards during a
separate sequential profiling pass. They are not concurrent model wall time.

| Priority | Work | Measured GPU cost | Next action |
|---|---|---|---|
| 1 | MXFP4 expert multiplication | 53.8% of prefill; 21.3% of decode | Investigate activation/weight reuse and the grouped down-projection dispatch. Prior vector-load and transposed-XMX experiments in campaign 72 were rejected; repeating those changes without a new mechanism is unlikely to help. |
| 2 | Dense projections | 41.9% of decode | Measure representative Q8 projection shapes and occupancy. Earlier launch fusion did not improve whole-model speed; start with the matrix-vector kernels themselves. |
| 3 | Router logits | 8.2% of prefill; 4.8% of decode | This pass: reuse both activations and weights across a small output tile while preserving every reduction. |
| 4 | Indexer scoring | 8.3% of prefill | Investigate reuse of query vectors across compressed keys. Any changed reduction needs exact score/top-k checks because near ties can change attention selection. |
| 5 | Attention | 14.8% of decode | Investigate loads and the per-key reduction/online-softmax dependency chain, with segmented fp16/fp32 KV and sparse-mask coverage. |

Router logits were selected for this bounded pass because their duplicated
loads can be reduced without changing numerical order. The higher-cost expert
and dense kernels remain the larger opportunities for a subsequent pass.

## Retained candidate

`ds4_router_logits_tiled` computes a two-token by two-expert output tile per
SIMD16 subgroup, with four subgroups per workgroup. Each activation is reused
for two experts and each weight for two tokens. Every output retains the
original lane-to-element assignment, four fp32 accumulation chains, tail
accumulation into chain zero, and final subgroup reduction.

Dispatch is limited to at least 64 tokens, hidden width 1024, and 64 experts.
Decode and smaller shapes keep the original scalar-token path. Partial token,
expert, hidden-dimension and workgroup tails are guarded. All four public
router variants share this logits path; bias, top-k and normalization are
unchanged. Qualification in this pass is on B70 only.

The production code is in `src/ops/deepseek4_ops.cpp`. The test is
`tests/unit/deepseek4_router_tile_test.cpp`, registered in CMake as
`deepseek4_router_tile_test`.

## Kernel experiments and correctness

| Experiment | Medium prefill | Production 2048-token chunk | Decision |
|---|---|---|---|
| Four tokens, one expert, 256-thread workgroup | About 1.25–1.30x | About 1.04x | Rejected for the production chunk size |
| Four tokens, one expert, 64-thread workgroup | About 1.25x | About 1.02x | Rejected; smaller workgroup did not resolve the limit |
| Two tokens, two experts, 64-thread workgroup | About 1.5–1.8x | About 1.83x | Retained after kernel and full-model checks |

The test was first built and run against the unchanged kernel. All 52 initial
correctness cases passed, while its opt-in 1.20x performance target failed,
with production/reference ratios near 1.00. The retained candidate passed that
target. Independent code review found no production defect and identified one
missing partial-final-workgroup case. That case and the geometry dispatch
boundaries were added. The final gate passed 84 cases across both B70s.

Correctness checks compare every logit bit-for-bit with a frozen GPU reference,
then compare expert indices and normalized weights against scalar-token
routing. Cases include tied expert rows, all four router APIs, image/text
selection, ragged hidden widths and expert counts, sentinel output guards,
out-of-order fill dependencies, and an in-order queue. A one-bit corruption
negative control confirms the exactness comparison detects changes. Timing
uses alternating reference/production order, three discarded repetitions, and
medians of ten measured repetitions. The performance assertion is opt-in;
ordinary CTest runs do not fail on timing variance.

## Full-model validation

All eight model runs completed successfully. The table reports medians of two
unprofiled runs per variant, in baseline/candidate/candidate/baseline order.
Decode measures 128 tokens after eight warm-up tokens. Prefill chunks are
capped at 2048 tokens; context capacity is 32768.

| Workload | Baseline tok/s | Optimized tok/s | Change |
|---|---:|---:|---:|
| Prefill, 128 tokens | 141.01 | 141.37 | +0.26% |
| Prefill, 512 tokens | 272.27 | 278.68 | +2.35% |
| Prefill, 4096 tokens | 499.41 | 517.43 | +3.61% |
| Prefill, 16384 tokens | 466.93 | 479.16 | +2.62% |
| Decode, 128-token starting context | 33.45 | 33.50 | +0.15% |
| Decode, 4096-token starting context | 27.81 | 27.87 | +0.21% |

Both candidate runs exceeded both baseline runs at 512, 4096 and 16384 prefill
tokens. At 4K, baseline runs were 497.21/501.60 tok/s and candidate runs were
513.98/520.88. At 16K, baseline runs were 467.22/466.64 and candidate runs were
479.61/478.72. The small short-prefill and decode differences are not claimed
as improvements. Decode uses the unchanged dispatch path.

Separate 4K profiling reduced summed router GPU time from 973.4180 ms to
546.7716 ms, a 1.78x kernel speedup (43.8% less time). Profiled throughput is
excluded from the table because profiling changes two-card scheduling.
All 18 recorded cache-hit, cache-miss, DMA-byte and first-token counters matched
across all four unprofiled runs.

The batch likelihood check used 512 tokens of the built-in prose corpus in
256-token chunks. All 511 saved per-token negative log-likelihood values
matched byte-for-byte; both runs had zero non-finite values, average NLL
1.699666 and perplexity 5.4721. This checks numerical preservation on this
corpus, not general model quality or vision accuracy. The common TSV SHA-256
is `8907d19d5927703a93a6522f62ba30569d2c5571ac12f715981af535f21deaf1`.

Baseline and candidate use the same saved `libie_core` archive, freshly
compiled benchmark/likelihood tools and compiler flags. The only differing
production object is `deepseek4_ops.cpp`. This avoids rebuilding or absorbing
unrelated working-tree GLM edits into the comparison. Build commands and
binary hashes are saved with the results. No compilation overlaps model runs.

The campaign performs a separate baseline profile/warm-up, unprofiled
baseline/candidate/candidate/baseline runs, a separate candidate profile, and
matched batch-mode per-token likelihood checks. Each model process uses the
single-flight guard, a 220 GiB memory cap, zero process swap, and a 1200-second
timeout. The machine was not rebooted; existing system swap usage is recorded
as a limitation. This pass does not qualify a clean-reboot competitive claim.

Working evidence: `results/deepseek4-opt-2026-09-09/`.
The machine-readable comparison is `summary.json`; each run has its own
command/environment manifest, binary hash and log.

## Integrated verification

The normal Release CMake build passed for `deepseek4_router_tile_test`,
`deepseek4_ops_gate_test`, `ie_ds4_bench` and `ie_ds4_ppl`. The guarded CTest
run passed both selected tests (2/2, 3.44 seconds). The dedicated router gate
previously passed all 84 exactness cases with its opt-in speedup assertion.
`git diff --check` passed. Build and CTest output are saved as
`integrated-build.log` and `integrated-ctest.log` in the evidence directory.

The optimization is local and has not been committed or pushed. Existing
unrelated working-tree changes were preserved.
