# DeepSeek Flash indexer optimization, September 9, 2026

The retained two-key indexer tile reuses vector query loads while preserving
FP32 arithmetic and head reduction order. The baseline already contains the
router and eight-token expert optimizations from the earlier same-day passes.
The matched production binaries differ only in their attention object.

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

## Reproduction and limits

Same installed native F16/F32/MXFP4 Flash Vision-Exp derivative, two Arc Pro
B70s, text only, synthetic token prompts, no speculation. Throughput uses 128
generated tokens, 2048-token prefill chunks and A/B/B/A/A/B run order. Profiled
runs are separate. The host was not rebooted; these are within-session results.
Each model process uses the project guard, a 220 GiB cap and zero process swap.

The final scorer gate passed 136 cases across both cards, including FP32/FP16
keys, exact frozen-kernel comparisons, selected entries, ragged boundaries,
independent query/key misalignment, guards and out-of-order dependencies.
The normal Release build and all five selected CTest gates on each card passed.

Raw local evidence and reproducible build/campaign scripts are in
`results/deepseek4-indexer-opt-2026-09-09/`. Each run records its command,
selected environment, executable SHA-256, exit code and elapsed time.
[Machine-readable measurements](INDEXER_OPTIMIZATION.json) and
[implementation and experiment report](../../../docs/deepseek4/76_INDEXER_OPTIMIZATION_2026-09-09.md).
