# Seven more Qwen, GLM and shared kernel passes

All seven passes are integrated, with production kernel and paired model checks complete. This round starts from the completed September 10 Qwen/GLM candidate and preserves earlier DeepSeek and unrelated server changes.

Two Intel Arc Pro B70 GPUs, Intel oneAPI C++ compiler 2026.1.1, Linux 7.0.12-p2pwl. GPU workloads run serially under scripts/ie-run-guarded. No model is loaded while compiling. Exact source/library baselines, commands, timing distributions and model logs are in results/kernel-seven-2026-09-10/.

## Seven retained passes

These ratios compare the selected production function linked from ie_core against the frozen pre-round kernel. They describe the measured shape ranges, not whole-model throughput. Shapes outside each dispatch retain the original kernel. T is the token count, H the head count, D the head width, and K/V the flattened QKV slice widths.

| Pass | Kernel | Dispatch | Measured speedup |
|---|---|---|---:|
| 1 | DeltaNet L2 | B70, D128, 1–12,288 rows | 3.01–3.57× |
| 2 | Fused DeltaNet QKV preparation | B70, D128, source heads 16/24, T1–256 | 1.65–2.48× |
| 3 | Float/half DeltaNet gate preparation | B70, heads 32/48/64, T1–512 | 1.08–1.34× |
| 4 | QKV split conversion | T16–1024; K/V 1024/2048, 2048/4096 or 2048/6144 | 1.22–4.59× |
| 5 | Per-head Q/gate split | T1–1024; H16/D128 or H16/24/32 with D256 | 1.44–3.56× |
| 6 | Repeated heads | T1–1024; H16, D128/256, repeat 2/3, both layouts | 1.10–6.93× |
| 7 | GLM KDA L2 | D64/128, 1–32,768 rows | 1.14–4.33× |

Passes 1 and 7 reuse constant-width inputs in lane registers. They preserve their distinct math: DeltaNet adds epsilon inside native rsqrt; KDA divides by a square-root norm floored at epsilon. Pass 2 gives each source Q/K head an independent workgroup and reuses its normalized values for the two tiled copies. Pass 3 removes runtime head division through constant-head specialization. Passes 4–6 move four adjacent values per lane and reuse source addressing.

The g/beta eligibility cache now keeps an entry for every device visited by a host thread. Switching between the two engine queues no longer repeats driver name/vendor queries. Empty DeltaNet preparation calls preserve explicit dependencies through a no-op submission. L2 fallback row offsets now widen before multiplication. API comments document supported aliasing and correct the historical A_log parameter name: the input is already GGUF ssm_a=-exp(HF A_log).

## Correctness

All 11 affected CTest entries passed on each B70, 22 executions total. The new KDA test has 112 cases/card including finite exponent spread and values around the norm floor. DeltaNet preparation has 249 shape cases plus 9 empty-dependency cases/card. Movement tests cover accepted and fallback boundaries, odd widths, both head layouts, guards, shifted pointers, delayed out-of-order producers/consumers and identity aliases. The optimized QKV cast is compared with the frozen scalar cast using all 65,536 half encodings, including signaling NaNs and exact payload bits.

Exact frozen comparisons accompany independent arithmetic/indexing oracles. KDA nonfinite arithmetic compares NaN classification, while finite values and signed zeros compare bits. A negative control replacing explicit FMA with plain accumulation fails the norm test. This caught a floating-point contraction difference when constant loops unrolled; no relaxed math flags were introduced.

Source and independent review evidence is in implementation.patch, candidate-manifest.json and review.md. The final combined build initially failed in the movement-test compiler frontend while final test edits were in progress. The source validated as UTF-8, and a serial unchanged retry passed. The cause was not isolated; no compiler workaround was added. The first permission-service request for the CTest run timed out before execution; the allowed retry succeeded.

## Timing method and rejected candidates

Final logs are dn/linked-{l2,qkv,gb}-run{1,2,3}.csv, kda/final-linked-timings.csv and movement/run-linked.log. linked-summary.json contains all shape-level summaries, and linked-status.json records every successful command. Baseline and candidate use the same destination addresses and alternate execution order.

DeltaNet reports the median of three process medians. Each process takes the median of seven batch means, each averaging 120 dependency-chained event durations. The KDA table includes both in-place and disjoint output. KDA takes three warmed rounds after a discarded full matrix warmup, each with 50 event samples/variant; raw p10/median/p90 are retained. Movement uses seven rounds of 61 event samples/variant and saves each round's median/p10/p90. All measure device execution; host dispatch and model loading are evaluated separately.

Every retained movement shape improved across the aggregate medians. Across 1,386 paired rounds there were 1,385 wins and one tie. Unchanged movement fallback medians ranged 0.94–1.12× in the final linked run, with the largest slower difference only 0.104 microseconds; report these variations rather than claiming exact parity. DeltaNet fallback timings remain near 1×. First-case GPU clocks varied between processes, so L2's representative steady shapes are roughly 3× rather than relying on its largest startup ratio.

Rejected experiments remain in the evidence tree. Runtime-width row packing/caching slowed normalization. Plain constant-width accumulation changed output bits. Generic 32-bit gate indexing offered only about 1%, while constant-head specialization gave repeatable gains. QKV T512 and larger L2 shapes did not justify extending dispatch. The first private large-L2 timing used different destination allocations and gave a misleading regression; identical-address reruns corrected that measurement. QKV conversion's T1 prototype regressed, so decode retains the original path.

## Model checks

All main paired quality checks passed. Qwen 27B Q8_0 produced eight exact baseline/candidate continuations across A/B then B/A fresh-load ordering. Flash-Next UD-Q4_K_XL produced an exact 78,643,200-byte layer-output capture, with PPL 1.2900 / NLL 0.254680 over just 33 continuation predictions after a 128-token prefill. GLM 5.3 Flash UD-Q4_K_XL matched all 490 likelihood values, PPL 1.8533813656592886, and its 32-token generated continuation. GLM pinning, cache capacity, 7,014 hits / 8,717 misses and CPU-expert counts 1,076/820 also matched.

The corpus is the first 2,048 characters of the saved WikiText input. The full Qwen 27B prompt contains 514 tokens and GLM contains 491. These are regression checks, not a general language-model-quality benchmark. The Qwen tokenizer still warns that pre=qwen35 uses the qwen2-family fallback; exact A/B parity does not establish external-tokenizer parity.

Qwen uses the two-GPU Engine path, context 2048, chunk 128, temperature 0, 32 generated tokens, no prompt cache and no speculative decoding. Warm rows below aggregate runs 2 and 3 from each of the two fresh loads per variant.

| Qwen 27B metric | Baseline median [range] | Candidate median [range] | Change |
|---|---:|---:|---:|
| Prefill tokens/s | 604.051 [601.902, 604.659] | 607.850 [606.812, 608.520] | +0.63% |
| Decode tokens/s | 16.756 [16.751, 16.763] | 16.776 [16.765, 16.786] | +0.12% |

The first candidate request had a material startup delay: prefill 6,559.884 ms versus baseline 601.702 ms. In reverse-order fresh-load repeats, candidate returned to 601.715 ms and baseline 629.082 ms. The delay did not recur, but its cause was not isolated. Original logs remain alongside the repeats. Total process time including load and four requests was 32.16 s / 34.95 s for the first baseline/candidate pair and 29.24 s / 33.42 s in the reverse-order repeat; loading remains variable.

The standalone GLM cold check was slower: prefill 20.2→18.2 tokens/s and decode 12.54→11.97 tokens/s. Stage B CPU-worker time also changed 0.45→0.58 s. Exact cache/expert counts do not remove host-latency variation, and these cold timings are not evidence of an end-to-end speedup. A repeated-request Engine comparison ran in reverse order with fixed 64 GiB host pinning and 20 GiB expert cache. Both variants had identical residency: stage A 62.33 GiB pinned / 20.39 GiB mmap / 19.99 GiB GPU cache; stage B 61.17 / 28.85 / 19.99 GiB. The Engine profile also had a slower first candidate request: 9,388.141 ms of prefill versus baseline 6,291.419 ms. That startup difference is retained without assigning a cause. Its warmed runs 2 and 3 produced the following results.

| GLM fixed-memory Engine metric | Baseline median [range] | Candidate median [range] | Change |
|---|---:|---:|---:|
| Prefill tokens/s | 29.001 [28.865, 29.137] | 29.172 [29.046, 29.299] | +0.59% |
| Decode tokens/s | 11.108 [11.078, 11.138] | 11.252 [11.172, 11.332] | +1.30% |

All four paired Engine continuations matched. Within each variant, the first full-prompt continuation differed from subsequent repeats, while runs 2 and 3 matched each other. This reproduces the prior round's existing repeat-request variation. The kernel changes neither introduced nor resolved it in this check; its cause remains unisolated.

The warmed model gains are modest. GLM's prefill ranges overlap, and each fixed-memory variant has only two measured warm samples. Cold GLM performance declined while warm performance improved in a different memory profile; these results do not establish a general end-to-end speedup or isolate the cold slowdown.

Exact commands, statuses, per-request output and comparisons are in model/. All generations are plain greedy. Quality and throughput comparisons use identical settings within each profile; the standalone and repeated Engine GLM profiles use different memory caps and must not be compared as the same benchmark.

## Limits and reproduction

This is coverage on two B70 cards and one compiler/driver combination, not CUDA-level maturity or a sustained multi-device stress campaign. Earlier GLM repeat-request output variation remains unresolved; do not attribute it to these changes without evidence. Artifacts remain local and have not been pushed.

Build the three new tests and affected existing targets listed in final-build-retry.log. Run each card's eleven-test CTest regex from the recorded guarded commands. Run run-linked.py under a 16G/300-second guard with ONEAPI_DEVICE_SELECTOR=level_zero:gpu. After all compilation finishes, run model/run-paired.py under a 220G/1200-second guard, then model/compare.py. The additional model/run-q27-repeat.py and model/run-warm-glm.py scripts reproduce the reverse-order checks; run their matching compare scripts afterward. Model paths and exact flags are recorded in model/*-command.json. Never overlap model runs or compile while a model is loaded.
