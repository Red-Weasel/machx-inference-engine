# Seven DeepSeek kernel passes — September 9–10, 2026

Six changes retained; hyper-connection projection reuse was rejected after measurement. The conversion pass also rejected vector-only conversion before retaining SwiGLU fusion. Full-model measurements below compare against a frozen same-session baseline that already includes the earlier router, eight-token expert, and indexer improvements.

| Pass | Outcome | Evidence |
|---|---|---|
| 1. XMX tail work | Skip wholly unused eight-row accumulator blocks; live accumulation order unchanged | About 9–31% less GPU time in mixed-job microbenchmarks |
| 2. Grouped MXFP4 decode | Byte lookup decoding with WG256 | 4–12% less GPU time across 16 representative cases after correcting the pass-7 row loop |
| 3. HC projection reuse | Rejected; original retained | Two-row and four-row variants were approximately 2.4× and 4.3× slower at T2048/H4096 despite exact outputs |
| 4. HC residual mix | Reuse input across four output streams; fix exact-alias race | About 2.1× kernel speedup at T2048/H4096; 172 boundary cases |
| 5. SwiGLU conversion | Fuse float-input SwiGLU and final half conversion | About 40% less GPU time for large tensors; 204 boundary/dependency cases |
| 6. XMX descriptors | Protect host/device ring reuse, validate before staging, preserve empty dependencies | Original empty-submission overwrite reproduced; 24 queue/concurrency cases passed |
| 7. XMX boundaries | Typed float output, safe alignment handling, checked dispatch arithmetic and queue/thread ownership for oneDNN caches | Original output overwrite and unaligned numerical defects reproduced; see results/deepseek4-seven-passes/pass7/REPORT.md for final coverage |

## Full-model throughput

| Workload | Baseline median (range), tok/s | Candidate median (range), tok/s | Change |
|---|---:|---:|---:|
| pp 128 | 141.38 (141.24–142.24) | 143.25 (142.65–143.39) | +1.32% |
| pp 512 | 278.19 (269.61–279.45) | 279.95 (275.77–291.51) | +0.63% |
| pp 4,096 | 545.39 (541.37–549.84) | 562.91 (559.18–569.34) | +3.21% |
| pp 16,384 | 523.02 (521.35–524.57) | 540.65 (539.34–540.91) | +3.37% |
| tg 128 | 33.41 (33.39–33.42) | 33.46 (33.46–33.47) | +0.15% |
| tg 4,096 | 27.85 (27.79–27.87) | 27.90 (27.89–27.92) | +0.17% |

Three unprofiled runs per variant in A/B/B/A/A/B order, with eight decode warmup tokens. Two Arc Pro B70 cards use expert tensor parallelism, 2048-token prefill chunks, synthetic token prompts and 128 generated tokens. The checkpoint is DeepSeek-V4-Flash-Vision-Exp-Abliterated-NativePreserved-F16-F32-MXFP4.gguf (164,697,137,152 bytes); these are text-only measurements. No speculative decoding or clean reboot. Cache/DMA/first-token counters match across all six runs. Profiles were collected separately. Baseline quality/profile data are reused from the initial integrated campaign with the same baseline executable; all six throughput runs were rerun after correcting the row-loop regression and selecting the final decoder.

## Kernel profiles

Single instrumented 4K prefill run per variant; these are summed GPU event durations across both cards, not wall-clock speedups. Timing runs above have profiling disabled.

| Kernel bucket | Baseline ms | Candidate ms | Ratio |
|---|---:|---:|---:|
| Grouped MXFP4 | 3107.33 | 2986.56 | 1.04× |
| XMX MXFP4 | 2191.65 | 2009.62 | 1.09× |
| HC residual mix | 441.99 | 194.75 | 2.27× |
| Cast + SwiGLU aggregate | 410.97 | 375.11 | 1.10× |

The largest remaining measured kernel buckets are grouped MXFP4 experts, XMX experts and indexer scoring. Follow-up tuning should target representative expert dispatch shapes and indexer work, with the compiler/shape performance matrix below guarding against regressions. These profiles alone do not establish a wall-clock bottleneck because work can overlap.

The cast/SwiGLU row combines the original and fused buckets because conversion also serves other call sites. It is not an isolated fusion measurement.

## Numerical and safety gates

All 511 per-token NLL values are byte-identical; baseline and candidate perplexity are 5.4721 and 5.4721. Both report zero nonfinite values. NLL dump SHA-256: `8907d19d5927703a93a6522f62ba30569d2c5571ac12f715981af535f21deaf1`.

The affected CMake targets built successfully, and all 13 selected CTest entries passed separately on each card (26 total). The five expert entries were rerun on each card after the row-loop correction and final decoder selection; the final 13-entry suite was also rerun against the frozen candidate. Tests cover independent arithmetic oracles, exact frozen behavior, ragged rows, canaries, shifted pointers, queue dependencies, ring reuse, invalid requests, concurrent host submission and aliasing. S8 queue/thread checks executed on both cards without skips. An older dense-gate policy assertion was corrected after a frozen-baseline probe confirmed that F16/F32 support predated this campaign; no production requantisation policy changed. Raw logs and exact commands are retained in results/deepseek4-seven-passes; final source snapshots and source/binary hashes are in model-v3/source and model-v3/candidate-manifest.json. The quality corpus is small and does not establish broad model or vision quality.

## Regression caught during integration

The first combined candidate preserved exact model likelihoods but regressed long-prefill throughput. Grouped expert profile time increased from 3107.33 to 4044.63 ms. Initially suspected lookup decoding was removed, but time remained 4177.21 ms; that failed attribution is preserved in model/ and model-final/. A controlled probe isolated the pass-7 overflow-safe min-increment loop: it was 44–52% slower. A uint64 constant-stride loop retained overflow safety and matched original timing within -0.63%/+0.44% across 16 cases. With that fix, lookup decoding won all 16 representative cases and was restored before this final model campaign. A tile-count loop and grouped-only single-tile specialization were also measured; the wide constant-stride loop was the simplest successful shared implementation.

## Reproduction and scope

Scripts below are under `results/deepseek4-seven-passes`. `model-v3/build-comparison.py baseline|candidate` compiles each source set and its tool objects against matching headers, then links the same frozen library and dependencies. `model-v3/campaign.py` records arguments, environment, binary hashes, exits and timing for each run; `model-v3/summarize.py` validates run status, counters and exact likelihoods. `model-v3/final-gates.py` runs the affected suite under a 16 GiB guard. Model runs use a 220 GiB memory cap, no process swap and a 1200-second timeout; only one GPU workload runs at a time.

## Remaining maturity gaps

These passes improve measured performance and fix demonstrated defects; they do not establish CUDA-level maturity. Coverage remains limited to two B70 cards and this compiler/driver combination. Two initial rebuild attempts crashed in the installed host compiler optimizer; unchanged serial retries passed with the same flags (compiler-retry.md). The highest-priority follow-up is a persistent correctness/performance matrix across compiler versions and representative dispatch shapes: this campaign demonstrated that an arithmetic-safe loop rewrite can silently cost 44–52%. Next are randomized shape/alias/concurrency stress and long-running allocation/lifetime tests, followed by broader device coverage, larger model-quality corpora and device fault recovery. The XMX descriptor cache has process lifetime; oneDNN caches follow their host-thread/queue lifetime. The grouped wrapper retains an in-order, serialized-host contract; the lower-level XMX primitive supports dependency-ordered out-of-order/concurrent submissions. Generic in-place HC mixing and unaligned materialized fallback paths can wait on the host. The grouped XMX API now uses a distinct float-output descriptor; source callers must migrate.

Agent concurrency is still split between layers: a harness owns agent histories and tool workflows, while the engine controls request admission, batching, scheduling and memory. These kernel passes do not change the DeepSeek serving scheduler.

README and reports were updated locally. Nothing was pushed or published because the earlier public-remote/artifact approval block remains unresolved.
