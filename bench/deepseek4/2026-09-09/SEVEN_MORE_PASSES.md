# Seven further DeepSeek kernel passes, September 10, 2026

Five optimization passes retained; Q8 quantization grouping and Q8-to-half vector conversion were rejected after measurement. Both rejected passes retain new regression tests. This campaign uses the preceding seven-pass candidate as its baseline, with no reset to the earlier engine.

| Pass | Final behavior | Focused evidence |
|---|---|---|
| 1. XMX attention | Fuse float-to-half Q preparation into its existing shared-memory transpose for T>=512 and disjoint Q/output; retain staging for smaller or overlapping calls | 4.4–14.7% less combined preparation/attention wall time in device-USM cases; exact frozen outputs |
| 2. Q8 quantization | Existing kernel retained; reject 4/8/16 subgroups per workgroup | All grouping candidates slower; new finite-half, boundary, dependency and reuse regression |
| 3. Q8-to-half conversion | Existing kernel retained; reject vector widths 4/8/16 loads | Large aligned and shifted cases slower; new exhaustive half-scale, guard, dependency and reuse regression |
| 4. MoE gather | Reuse row mapping across two columns for H>=1024 and at most 2,097,152 output elements | About 1.64× at M=257/H=4096 and 1.76–1.80× at M=512/H=4096 |
| 5. Unweighted RMS | Keep each lane's two values across the original 512-wide loop/reduction | About 14% less time at 4,096 in-place rows and 4–5% at 65,536; exact output gate |
| 6. RoPE | Dispatch only rotary pairs for exact alias; validate live dimensions and byte spans | About 3.6× at T=2048/H=32/D=512/R=64; odd-width out-of-bounds write reproduced and fixed |
| 7. MoE scatter | Reuse packed index/weight across two columns, preserving ascending top-k accumulation | About 1.44–1.50× at T=257/H=4096/K=8; slower small/boundary shapes retain baseline |

Focused timings are shape-specific. Attention's first-row metric includes eliminated preparation and is host-wall time; other quoted kernel timings use GPU events. They are not interchangeable with full-model throughput.

## Full-model throughput

| Workload | Previous candidate median (range), tok/s | New candidate median (range), tok/s | Change |
|---|---:|---:|---:|
| pp 128 | 142.62 (141.77–143.04) | 143.32 (143.06–143.83) | +0.49% |
| pp 512 | 281.47 (275.82–283.11) | 288.48 (281.45–292.28) | +2.49% |
| pp 4,096 | 564.97 (558.28–572.54) | 573.96 (563.48–578.74) | +1.59% |
| pp 16,384 | 539.48 (534.09–539.82) | 546.60 (546.16–546.84) | +1.32% |
| tg 128 | 33.48 (33.13–33.55) | 33.34 (33.33–33.46) | -0.41% |
| tg 4,096 | 27.91 (27.57–27.93) | 27.86 (27.74–27.90) | -0.17% |

Decode median changes are -0.41% at 128 context and -0.17% at 4K. Their sample ranges overlap, as do the 4K prefill ranges. These three-run measurements do not establish statistical significance.

Three fresh unprofiled runs per variant in A/B/B/A/A/B order. Two B70 cards use expert tensor parallelism, 2,048-token prefill chunks, synthetic prompts, 128 generated tokens and eight decode warmup tokens. Loading is excluded. No speculative decoding or clean reboot. The checkpoint is DeepSeek-V4-Flash-Vision-Exp-Abliterated-NativePreserved-F16-F32-MXFP4.gguf, 164,697,137,152 bytes. These are text-only measurements; decode at 512 and 16K context was not rerun.

All 511 per-token NLL values match byte-for-byte; baseline and candidate perplexity are 5.4721 and 5.4721, with zero nonfinite values. NLL SHA-256: `8907d19d5927703a93a6522f62ba30569d2c5571ac12f715981af535f21deaf1`. All 18 selected cache/DMA/first-token counters match across the six throughput runs, and 13 selected profile counters match.

## Kernel profile

One fresh instrumented 4K prefill run per variant. Values are summed GPU event milliseconds across both cards, not wall time. Attention preparation and attention are aggregated because the retained fusion moves work between those buckets.

| Bucket | Baseline ms | Candidate ms | Baseline/candidate |
|---|---:|---:|---:|
| Attention + preparation | 666.56 | 587.14 | 1.14× |
| MoE gather | 115.01 | 115.11 | 1.00× |
| Unweighted RMS | 110.71 | 104.78 | 1.06× |
| RoPE | 105.25 | 28.31 | 3.72× |
| MoE scatter | 61.93 | 54.48 | 1.14× |
| Q8 quantization, unchanged | 178.47 | 179.05 | 1.00× |
| Q8-to-half, unchanged | 148.69 | 149.16 | 1.00× |
| Grouped MXFP4 | 2989.69 | 2991.94 | 1.00× |
| XMX MXFP4 | 2014.99 | 2016.09 | 1.00× |

## Correctness, regressions and scope

All 20 affected CTest entries passed independently on each B70 card, 40 successful executions. Six new test executables cover these seven passes. They combine frozen GPU oracles with independent host arithmetic bounds or exact conversions, output guards, shifted pointers, supported aliasing, shape boundaries, special values, ordered/out-of-order queues and buffer reuse. Q8 quantization covers all 63,488 finite half patterns; Q8 conversion covers all 65,536 half-scale bit patterns. Attention performs 1,025 mixed launches; RoPE performs 1,000 repetitions per 24 shape/alias/card cases; RMS performs 1,000 repetitions per 32 cases. Quantization/dequantization and MoE also exercise repeated fresh producers. These are short stress checks, not long-duration or broad-device reliability claims.

Negative controls prove the oracles reject a missing attention producer edge, corrupted quantization mapping/bytes, missing conversion/RMS/RoPE dependencies, wrong gather rows and reversed scatter accumulation. Delayed producers are drained after consumer readback before an expected failure can free their buffers.

Integration caught an RMS expression-form regression: an explicit two-load rewrite differed by 1–2 ULP although the private loop prototype was exact. That rewrite was rejected. The retained loop form passes 88 exact/guard/dependency cases, preserves generic-width code and passed the full model quality gate above. No compiler/ISA root cause is asserted. Review also excluded scatter's slower 65,536-element boundary range; retained scatter starts at 131,072 elements. Rejected candidates, failures and measurements remain in the evidence directory.

Hardware coverage remains two Intel Arc Pro B70 cards with oneAPI 2026.1 and the installed driver. This campaign adds no other hardware/compiler results and establishes no CUDA-level maturity claim. It does not change request admission, agent scheduling, model format support, arbitrary-overlap guarantees or invalid device-index handling. The quality corpus is small and text-only. Kernel event durations can overlap across queues/cards and must not be interpreted as additive wall-clock latency.

## Reproduction and evidence

Local evidence is under `results/deepseek4-seven-more-2026-09-10`. Each pass retains private candidates, frozen sources, timing logs, review findings and outcomes. `model/build-comparison.py` copies the preceding campaign's candidate binaries byte-for-byte as the new baseline, builds the current changed translation units and links the same frozen library/dependencies. `model/source-manifest.json`, `auxiliary-source-manifest.json` and `binary-manifest.json` record the build inputs. `model/source` snapshots headers, tool and new test sources; the changed production sources are frozen next to their objects.

`model/campaign.py` records exact commands, selected environment overrides, binary SHA-256, exit status and elapsed time. Baseline and candidate quality/profiles are fresh runs. Six unprofiled throughput runs follow A/B/B/A/A/B order. `model/summarize.py` checks all status codes, hashes, finite likelihoods, workload counters and exact NLL dumps. `model/final-gates.py` runs the 20-test suite separately on each card; `ctest-inventory.json` records its commands/properties and `test-binary-manifest.json` hashes the 18 distinct executables. GPU work is serialized through `scripts/ie-run-guarded`; focused tests use 16 GiB/240 seconds, model runs 220 GiB/1200 seconds and no process swap. No compilation runs while the model is loaded.

README and reports are local. The earlier automatic approval review blocked publication to the public GitHub remote pending repository/artifact scope confirmation; this campaign does not push or publish.
