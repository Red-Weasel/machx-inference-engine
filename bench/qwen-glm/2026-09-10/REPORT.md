# Qwen and GLM kernel pass, September 10, 2026

Three optimizations retained after four kernel investigations. The shared gated RMS variants were rejected. This round builds on the preceding DeepSeek work and preserves the existing GLM/server changes in the workspace.

| Kernel | Change retained | Production-linked timing evidence |
|---|---|---|
| Qwen partial and three-stream RoPE | In-place prefill launches only rotary pair owners when T>=16 and T×heads>=512. Small grids and disjoint outputs retain the full launch. | D256/R64/H32: 2.25–3.04× at T128–2048 across the two position layouts and both B70s. |
| GLM KDA gate | Two-dimensional row/channel launch removes per-element 64-bit modulo for 1–128 rows and width 8192. Other shapes retain the flat kernel. | Disjoint width 8192: 1.09–1.25× at T1/16/128. |
| GLM convolution plus cast | New `depthwise_conv1d_causal_f32` writes the half-rounded result directly to float. Three GLM KDA convolution/cast pairs use it. | C8192/K4: 1.04–1.38× at T128/512/1024; T1 measures 1.36–1.42×. T16 is mixed, 0.972–1.000×. |
| Shared gated RMS | Original kernel retained. Reject input caching and 4/8-subgroup grouping. | Cached variants regress small/mid-size cases; grouping alone gives no reliable overall benefit. |

These are microkernel measurements on two Intel Arc Pro B70 cards using icpx 2026.1.1. RoPE/KDA use GPU event duration. Convolution measures the device timeline from a preceding marker through completion of the whole operation, including the eliminated cast and inter-command gaps. The metrics do not establish full-model throughput gains. Production-linked raw logs and parsed values are under `results/qwen-glm-kernels-2026-09-10/`.

## Correctness and hardening

Empty convolution calls previously returned a default event and dropped input dependencies. A delayed-producer test reproduced that failure before the fix. Both output APIs now return a dependency-carrying event for zero tokens, channels or kernel size.

The fusion preserves the original FP16 rounding boundary. Both an ordinary `half` round trip and explicit vector RTE conversion produced unrounded FP32 output under this compiler. The probe caught both failures. A volatile private `uint16_t` store preserves the half bits before widening. The existing half API keeps its original store expression. State transitions and tap accumulation order are shared by both APIs.

Eight production-linked CTests passed independently on each B70, 16 executions total:

- `qwen_glm_kernel_test`: 72 cases per card, including both RoPE pair outputs, three divergent position streams, exact aliases, disjoint outputs, dispatch boundaries, zero shapes, shifted pointers, independent guards, finite arithmetic checks, infinities and delayed out-of-order producers.
- `conv1d_f32_test`: 96 cases run twice per card, including kernel sizes 1/2/4/7/65, channels 1/65/4096, chunks 1/2/3/4/8/17, complete streaming history, null history, half subnormals/infinities/NaNs, guards, delayed producers, buffer reuse and all empty dimensions.
- Existing elementwise, DeltaNet, KDA recurrence, convolution-state and interleaved-RoPE tests, plus the empty-convolution reproducer.

Finite outputs match frozen pre-round expressions exactly. Special-value checks require matching NaN classification, rather than claiming NaN payload identity. Independent host arithmetic and history checks supplement the frozen implementation. One compiler crash occurred while rebuilding the unchanged elementwise test; its unchanged serial retry passed. Both logs are retained.

Independent source review found no remaining important issues in the dispatch rules, convolution rounding, state transitions, GLM integration or test coverage. This is coverage of two B70s and one compiler release, not broad hardware/compiler certification.

## Model checks

Qwen 27B's four plain-greedy continuations match byte-for-byte, each 32 tokens after 129 or 514 prompt tokens. Prompt caching and speculation are disabled. After the initial short-prompt and first full-prompt warmups, the two repeated 514-token runs measure:

| Two-GPU Qwen 27B Q8_0 | Baseline median and range, tok/s | Candidate median and range, tok/s |
|---|---:|---:|
| Prefill | 602.99, 602.46–603.51 | 601.68, 600.86–602.49 |
| Decode | 16.764, 16.762–16.765 | 16.745, 16.732–16.757 |

This short two-sample comparison does not establish a whole-model improvement. Warmup times remain in the raw logs rather than being mixed into warmed throughput.

Flash-Next's captured mixer/FFN block outputs match exactly across 78,643,200 bytes, including the 128-token prefill and 32 subsequent decode inputs. SHA-256 is `e9bf989104590e5eebdafcceb2f77d421192a4956c8287a33bda51e76e4cb753`. Both runs report mean NLL 0.254680 and PPL 1.2900 across 33 continuation predictions. This is a short matched regression corpus, not a general model-quality assessment. The single-GPU runner uses its existing dense-Q8 mode and 132 expert-cache slots per layer.

GLM's 490 per-token NLL values match byte-for-byte, SHA-256 `b135aaedfa793f004657e0cbe71c4a1e1c563a7bd0e330030643a91a599d9e7b`, with zero nonfinite values and PPL 1.8533813656592886 in both runs. Its 32-token continuation after 128 prompt tokens also matches exactly. Cache and CPU-expert counts match. The standalone runner uses two GPUs, 128-token chunks and plain decode.

The single cold-start generation run measured 20.4→17.8 tok/s prefill and 12.50→12.27 tok/s decode. The candidate is slower in that observation. Stage-B CPU worker time also rose from 0.46s to 0.53s. These runs cannot establish a kernel cause.

To examine warmed behavior, a second comparison used the two-GPU Engine API, fixed 64GiB per-stage host pin caps, 20GiB expert-cache requests, 128-token chunks and repeated plain-greedy generations. Actual residency matched: 62.33/61.17GiB pinned host banks and 19.99GiB GPU cache per stage. The initial short prompt and first full prompt warmed the path; the last two 491-token requests measured:

| Warm GLM Engine, fixed memory profile | Baseline median and range, tok/s | Candidate median and range, tok/s |
|---|---:|---:|
| Prefill | 31.931, 31.866–31.996 | 32.111, 32.003–32.220 |
| Decode | 11.619, 11.314–11.925 | 11.612, 11.291–11.933 |

Prefill changes by +0.56% and decode by -0.06% in this small sample. This does not establish a general whole-model speedup. The fixed profile differs from the standalone runner, so compare variants within each table. The first requested 22.5GiB cache was rejected by the Engine's VRAM budget before weights loaded; the accepted 20GiB setting was used identically for both variants and does not change production defaults.

All four paired Engine continuations match exactly. However, within each variant the first full-prompt continuation differs from the later two, despite prompt caching being disabled. This behavior is present in the frozen baseline; its cause was not isolated. These results establish paired preservation, not invariance across repeated requests. The two JSON comparisons retain every run and its output hashes.

The initial single-GPU `ie-bench` diagnostic stalled in a Level Zero event wait and was terminated after 304 seconds. Its frozen pre-round executable was used, so this does not diagnose a regression from this round. The subsequent GPU numerical check passed. Qwen 27B validation was moved to the engine's two-GPU layer-split path, with 128-token chunks. Both runs completed successfully. The model's existing tokenizer fallback warning is retained in the log; matched runs test kernel changes, not tokenizer parity with an external implementation.

## Reproduction and artifacts

The local evidence directory is `results/qwen-glm-kernels-2026-09-10/`:

- `baseline-manifest.json`, `candidate-manifest.json`, `baseline/` and `round.patch` identify the source/library boundary without including unrelated workspace edits.
- `probe.log` records all private RoPE, RMS and KDA candidates. `linked-probe.log` records the retained production-linked dispatches.
- `conv-cast/diagnostic.log` and `explicit-round.log` record rejected rounding variants. `rejected-plain-round.hpp` and `rejected-vec-round.hpp` preserve their source.
- `conv-cast/production-cast.log` and `microkernel-summary.json` record the accepted fusion and aggregate kernel measurements. Review found that the earlier `conv-cast/linked-probe.log` used a simplified cast launch. That timing is superseded by the rerun using the actual `ie::cast_fp16_to_fp32` kernel; the earlier log remains as audit evidence.
- `ctest-card0.log`, `ctest-card1.log`, `baseline-empty-negative.log` and build logs record validation, including the compiler retry.
- `model/run.py` and each model's command/status JSON record model workloads and environment overrides. Standalone baseline executables were freshly built and copied before production edits. The replacement Qwen 27B engine probe was compiled against the frozen headers/library and the candidate headers/library separately.
- `model/comparison.json` and `model/warm-glm-comparison.json` record the paired quality, timing and residency checks. `model/run-warm-glm.py` reproduces the warmed Engine comparison.

Build and run the focused checks serially with the repository guard:

```bash
cmake --build build --target qwen_glm_kernel_test conv1d_f32_test conv1d_state_test rope_imrope3_test elementwise_test deltanet_test kda_recurrence_test -j1
scripts/ie-run-guarded --mem 16G --timeout 300 python3 results/qwen-glm-kernels-2026-09-10/run-tests.py
```

The guard serializes GPU use and caps memory. Never compile while a model is loaded. Large model runs use a 220GiB cap and run one process at a time. No public push was performed in this round.
