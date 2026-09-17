# DeepSeek indexer key tiling

This pass starts from the retained router and grouped MXFP4 expert optimizations.
It changes only indexer scoring in `src/ops/deepseek4_attn.cpp`. The previous
profile attributed 979.5 ms of summed 4K prefill GPU time to this scorer.

Each subgroup now scores two compressed keys and reuses four-element query
loads across them. Four FP32 dot-product chains, the final pairwise sum, head
order and subgroup reduction are preserved. Explicit FMA matches the original
compiler-contracted arithmetic. Both FP32 and FP16 key storage are supported.

Dispatch requires at least 64 query tokens, 16 heads, 64 dimensions and 32 keys;
the dimension must be divisible by four and both input bases vector-aligned.
Decode, smaller shapes, ragged dimensions and misaligned inputs retain the
original paths. Partial key pairs and ragged head counts are masked.

The first scalar-load tile differed by up to 2.79e-9 from the original kernel.
It was rejected. Explicit FMA restored exact outputs, but scalar-load tiling
made FP16 slower. Vector loading produced the useful candidate. Review then
identified a C++ object-lifetime problem with scalar-to-vector pointer casts.
The retained version uses SYCL vec::load from scalar pointers with alignment
hints justified by the dispatch conditions.

The final supported-load probe alternated four variants, discarded four
repetitions and reported the median of ten, using device allocations on both
B70s. Every tested output matched exactly. The selected variant uses two keys
per subgroup and sixteen subgroups per workgroup. One-key vector loading and
four-subgroup workgroups were also measured. These are synthetic kernel timings,
not model throughput. See `key-tile-load.log` for all shapes and both key types.

The regression test freezes the original score kernels as independent numerical
oracles. It covers both key types, dispatch boundaries, ragged heads/dimensions,
odd key counts, separate query/key misalignment, tied scores, causal top-k
selection, score/output guards, explicit out-of-order dependencies, and an
in-order queue. A deliberately flipped output bit checks the exact comparator.
The final candidate passed 136 cases on two cards. The normal Release build
passed; five selected CTest gates passed on each card: indexer tiling, indexer
top-k, expert GEMM, router tiling, and general DeepSeek operations.

## Full-model results

All ten campaign processes exited successfully. Medians of three unprofiled runs per variant:

| Workload | Baseline tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|
| Prefill, 128 tokens | 141.82 | 142.52 | +0.49% |
| Prefill, 512 tokens | 282.11 | 284.20 | +0.74% |
| Prefill, 4096 tokens | 536.89 | 547.22 | +1.92% |
| Prefill, 16384 tokens | 503.52 | 522.57 | +3.78% |
| Decode, starting context, 128 tokens | 33.16 | 33.40 | +0.73% |
| Decode, starting context, 4096 tokens | 27.62 | 27.79 | +0.64% |

| Workload | Baseline range | Candidate range |
|---|---:|---:|
| Prefill, 128 tokens | 140.747–142.548 | 142.306–143.014 |
| Prefill, 512 tokens | 279.731–291.221 | 276.843–286.242 |
| Prefill, 4096 tokens | 533.839–542.342 | 544.468–548.561 |
| Prefill, 16384 tokens | 502.802–504.169 | 520.361–527.925 |
| Decode, starting context, 128 tokens | 32.951–33.293 | 33.083–33.675 |
| Decode, starting context, 4096 tokens | 27.582–27.746 | 27.525–28.083 |

Every candidate exceeded every baseline at both 4K and 16K. Short-prefill and
decode ranges overlap; this pass claims a long-prefill improvement and no
material decode change. All 18 recorded cache-hit, cache-miss, DMA-byte and
first-token counters matched across all six throughput runs.

All 511 per-token NLL values matched byte-for-byte, with zero non-finite values,
average NLL 1.699666 and perplexity 5.4721. This establishes preservation on the
tested text corpus, not general model quality. The common NLL dump SHA-256 is
`8907d19d5927703a93a6522f62ba30569d2c5571ac12f715981af535f21deaf1`.

The separate 4K profile reduced indexer GPU time from 979.9243 to
682.9433 ms, a 1.43x speedup and
30.3% less time. Profile timings sum the two cards while profiling forces
sequential execution; these are not model wall times.

The two-key indexer tile is retained. Independent review's vector-load and
alignment-test findings were fixed and reviewed again; no actionable findings
remained. Production source hashes stayed unchanged throughout the campaign.


The matched binaries link the same saved library and tool objects, replacing
only the attention production object. The baseline already includes both earlier
optimizations. The campaign uses the installed Flash Vision-Exp native MXFP4
checkpoint, two B70s, no speculation, 2048-token prefill chunks, 128 generated
tokens for throughput, and A/B/B/A/A/B order. Separate profiled runs are excluded
from throughput medians. Likelihood checks use the built-in 512-token prose
corpus and compare all 511 per-token NLL values.

All model processes are serialized by `ie-run-guarded`, with a 220 GiB memory
cap, zero process swap, and no concurrent compilation. The host was not rebooted;
these are within-session comparisons. They do not establish general quality,
vision performance, clean-reboot competitive scores or multi-agent throughput.

Evidence: `results/deepseek4-indexer-opt-2026-09-09/`.

## Remaining opportunities

The matched baseline profile still spends 3108.6 ms in grouped MXFP4 expert
GEMM and 2191.6 ms in its XMX path. Those remain the largest measured kernel
targets. Hyper-connection (575.2 ms), attention XMX (559.2 ms) and HC mix
(443.6 ms) are smaller candidates for future profiling and exactness-gated
experiments. No changes to those kernels are part of this pass.
