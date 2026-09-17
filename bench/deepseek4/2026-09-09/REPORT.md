# DeepSeek V4 Flash text throughput, September 9, 2026

The measured checkpoint is **DeepSeek-V4-Flash-Vision-Exp-Abliterated-NativePreserved-F16-F32-MXFP4.gguf**, a 164.70 GB Vision-Exp derivative. These measurements cover text inference only, not the unmodified Flash checkpoint or image processing.

## Measurements

| Phase | Prompt/context tokens | Run 1 tok/s | Run 2 tok/s | Median tok/s |
|---|---:|---:|---:|---:|
| Prefill | 128 | 141.12 | 141.19 | 141.16 |
| Prefill | 4,096 | 478.11 | 493.21 | 485.66 |
| Prefill | 16,384 | 464.96 | 466.14 | 465.55 |
| Decode | 128 | 33.40 | 33.50 | 33.45 |
| Decode | 4,096 | 27.76 | 27.72 | 27.74 |
| Decode | 16,384 | 29.99 | 29.95 | 29.97 |

Two timed process runs after one discarded warm-up process. The median of two runs is their midpoint. Each process also warms the first prefill shape internally. Load time is excluded. The table reports every measured context, without selecting a best run. All three throughput processes exited successfully and reported `run.status=ok` for all requested workloads.

This is a deterministic **synthetic-token throughput benchmark**, not natural-language task accuracy, a five-prompt serving suite, or a comparison against another engine. Prompts use the harness's fixed token-ID sequence; decode is greedy, single-stream, 128 timed tokens following eight untimed tokens at each depth. Prefill chunks are 2048 tokens; context capacity is 32768. Speculative decoding is not used.

## Machine and executable

- Two Intel Arc Pro B70 cards, 32 GiB each; 249 GiB usable system RAM.
- Level Zero, kernel `7.0.12-p2pwl`, Intel compute driver `26.22.38646.7`.
- Expert tensor parallelism across cards 0 and 1. Each card reports 81 streaming slots per layer; total pinned expert banks are 147.17 GB.
- Existing benchmark executable built September 3, copied before the runs. SHA-256: `4737584ccccc4a3a25fad033a5c5d0f1ec5121c8275afc1523225dadc832c158`.
- Workspace HEAD at measurement: `96771aa13a0b316142468d0414ac5e04e5f694a9`. The workspace had unrelated changes. The binary's exact source revision was not independently attested, so this is a result for the recorded binary, not certification of that HEAD or all current changes.
- **No reboot** before measurement: about 4 days 22 hours uptime, 4.7 GiB existing system swap use, 229 GiB available memory before loading, both GPUs at 400 MHz idle. Desktop applications remained open; no competing inference job was detected. System swap rose to about 8 GiB during the repeat loads while available RAM remained about 34 GiB. These are warm local measurements, not a clean-reboot headline under `docs/BENCHMARK_METHODOLOGY.md`.
- Every process used the single-flight guard, a 220 GiB memory limit, zero process swap, and a 1200-second timeout. No compilation overlapped this campaign.

## Reproduce

Set `DS4_GGUF` to the exact checkpoint above. Run this command three times sequentially; discard the first process and retain the next two. The archived executable hash identifies this campaign; rebuilding later may produce different results.

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
DS4_RUN_REAL=1 DS4_TP_GPUS=0,1 IE_DS4_RMS_EPS=1e-20 \
./scripts/ie-run-guarded --mem 220G --timeout 1200 \
  ./build/tools/ie-ds4-bench --gguf "$DS4_GGUF" \
  --pp 128,4096,16384 --ctx 128,4096,16384 \
  --decode 128 --warmup 8 --max-seq 2048 --max-ctx 32768
```

For the separate profile, use `--pp 4096 --ctx 4096 --decode 32`, add `--kprofile --kprofile-prefill`, and set `IE_QUEUE_PROFILING=1`. Keep the other options unchanged.

## Kernel profile

| Phase at 4K | Kernel group | GPU time, sum over cards | Share |
|---|---|---:|---:|
| pp | other | 8265.609 ms | 69.7% |
| pp | attention | 1714.006 ms | 14.5% |
| pp | hyper | 1022.109 ms | 8.6% |
| pp | moe_routed | 537.072 ms | 4.5% |
| pp | dense_materialise | 149.439 ms | 1.3% |
| pp | moe_shared | 90.607 ms | 0.8% |
| pp | moe_mixed | 71.292 ms | 0.6% |
| pp | lm_head | 1.972 ms | 0.0% |
| tg | attention | 20.878 ms | 53.0% |
| tg | other | 11.092 ms | 28.2% |
| tg | moe_shared | 2.596 ms | 6.6% |
| tg | hyper | 2.524 ms | 6.4% |
| tg | moe_routed | 1.051 ms | 2.7% |
| tg | lm_head | 0.952 ms | 2.4% |
| tg | moe_mixed | 0.261 ms | 0.7% |

Profiling is a separate process and is excluded from throughput medians. The process-global profiler runs the TP cards sequentially; these GPU times sum both cards and cannot be substituted for concurrent wall time. The full logs retain individual kernels, DMA, expert-cache statistics, and timing distributions. There is no speculative verify phase to profile.

## Archived Qwen measurements

The README's Qwen figures are from the August campaigns below, not September 9 reruns. They use different checkpoints, prompts, context capacities, and serving modes from this DeepSeek harness. They do not establish a controlled cross-model speed ratio. Flash's sampled agent runs measured token throughput but did not complete the requested task.

The excerpts below preserve the supporting records from workspace HEAD `96771aa13a0b316142468d0414ac5e04e5f694a9`; source paths identify the original campaign documents. The dense 27B source retains a historical Qwen3.6 title but explicitly identifies the August checkpoint as Qwen3.8-27B.

### docs/authority/qwen35-27b.md, lines 6–10

```text
> **Last verified:** 2026-08-15 evening (2-card split campaign on **Qwen3.8-27B**, two sessions —
> same `qwen35` arch tag, identical geometry + trailing NextN, Q8_0-only GGUF. Day totals:
> short-prompt tg 14.25→**22.3** (--spec K3, lossless 5/5, auto-cutoff >2K prompt), pp median
> 49.1→174; long-prompt pp: 2K **945** / 4K 851 / 9K **731** (pipelined prefill default-ON, +93%
> self); PPL stream 5.3567 bit-reproduced. See §2 ND warning, §3 secondary configs, §5 rows 16–23).
```

### docs/agent-serving-campaign.md, lines 165–185

```text
### Decode-speed matrix (2026-08-26, current binary, greedy, ctx 32768)

| config      | short (23-tok prompt) | deep (18.7K prompt) |
|-------------|----------------------|---------------------|
| split plain | 15.5 tok/s           | 12.9 tok/s          |
| split spec  | 22.0 tok/s           | 13.0 tok/s          |
| tp plain    | 18.2 tok/s           | 12.0 tok/s          |
| tp spec     | 24.5 tok/s           | 12.1 tok/s          |

Findings: spec engages correctly (+42% shallow) but pays nothing at depth (the
K+1-token verify re-reads the full attention window); the historical
"28.5-30 tok/s certified" does NOT reproduce on the current working tree —
best short-context today is 24.5 (tp+spec); at agent depth ALL configs
converge to ~12-13 tok/s (attention/KV-read bound — consistent with the
2026-08-15 campaign's ~11.5 cap finding). TP also prefills slower than the
split's pipelined path (364 vs 534 tok/s) and has no prompt-cache restore →
split remains the right agent-serving path. Post-Ω speed levers, in value
order: batched decode (aggregate ~13 → 35-45 across 4 slots) and int8-KV on
the serving path (halve the attention reads; needs slot-stash int8 support).
Best solo/chat preset today: split + --spec + --temp 0 (22 tok/s shallow,
keeps prompt cache).
```

### docs/qwen4exp-lossless-spec-plan.md, lines 688–701

```text
## ═══ CAMPAIGN CLOSE-OUT (2026-08-28 ~09:30 — qwen-flash optimization COMPLETE) ═══

Both founder mandates MET and certified:
- Prefill: 467-468 tok/s (run2, 4x1024 pipelined 2-GPU, bit-eq, warmed).
  Ladder: 175 → wave-flush → HC gemm prefill → per-piece mixed-bank →
  big-chunk 4096 → copy-queue prefetch → 467.
- Decode: 41.8 tok/s lossless on code text (35.0 chat), K=3, spec2-certified
  bit-identical to T=1. The single biggest lever: native half→float
  (`float(sycl::bit_cast<sycl::half>(h))`) across all hot dequant kernels
  (aaf0e9e) — exact, bit-identical, no determinism cost.
- Serving: warm prefill 290 tok/s (chunk-pipelined, 0205241); concurrency
  live via per-slot suspend/resume (--parallel 2, e3664c2) with in-process
  byte-exactness gate (ie-qwen4exp-slot-test PASSED).
- P2P: interstage wide handoff wired + bit-eq certified (IE_P2P=1, a3048b0);
```

### docs/agent-serving-campaign.md, lines 232–243

```text
## Phase Ω — Flash-Next attempts (2026-08-28, headless, 3 runs)

Backend: Flash-Next (qwen4exp) served 2-GPU, --parallel 2, model-spec sampling
(temp 1.0 / top_k 20 / top_p 0.95), ctx 32768. Serving stack after this
session's perf work: pipelined prefill (~290 tok/s warm), concurrency live,
decode ~17-22 tok/s in-agent.

Three headless Hermes runs (attempt 3 with narrowed toolset
`-t web,file,delegation` + action-forcing prompt). All three exited 0,
produced NO /tmp/omega-report.md, and share ONE failure shape: the model
reasons through the delegation plan correctly ("let me now call delegate_task
with the 3 tasks") and then ends the turn without emitting the tool block.
```

## Saved evidence

[Warm-up log](warmup.log), [run 1](run-1.log), [run 2](run-2.log), [profile](profile.log), [machine and binary metadata](metadata.json), and [machine-readable summary](summary.json). Log progress carriage returns are normalized to newlines and trailing whitespace is removed for Git; metric values are unchanged. Process exit records are included beside the logs. The model checksum in metadata is publisher-supplied and was not independently rehashed during this campaign. The multi-gigabyte model and executable are not included in Git.
