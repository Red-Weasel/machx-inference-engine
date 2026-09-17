# DeepSeek kernel follow-up, September 9, 2026

The retained change splits grouped MXFP4 expert prefill into independent
eight-token tiles. It preserves each output's arithmetic and leaves the
single-token decode specialization unchanged. The comparison baseline already
includes the earlier router tiling optimization.

## Measured result

Two Intel Arc Pro B70s, DeepSeek V4 Flash Vision-Exp Abliterated native MXFP4
GGUF, text-only synthetic prompts. Medians of three unprofiled runs per variant
in A/B/B/A/A/B order, 2048-token prefill chunks, 128 timed decode tokens after
eight warm-up tokens, 32768-token context capacity. The machine was not rebooted.
These within-session results are separate from the earlier binary measurements
in [REPORT.md](REPORT.md).

| Workload | Baseline tok/s | Optimized tok/s | Change |
|---|---:|---:|---:|
| Prefill, 128 tokens | 141.24 | 141.81 | +0.41% |
| Prefill, 512 tokens | 280.03 | 284.01 | +1.42% |
| Prefill, 4096 tokens | 513.95 | 537.16 | +4.52% |
| Prefill, 16384 tokens | 478.09 | 504.09 | +5.44% |
| Decode, 128-token starting context | 33.50 | 33.57 | +0.21% |
| Decode, 4096-token starting context | 27.91 | 27.92 | +0.02% |

Every optimized run exceeded every baseline at 4K and 16K. Shorter-prefill
ranges overlapped; no short-prefill improvement or decode gain is claimed.
Separate profiling reduced summed grouped-MXFP4 GPU time from 4166.83 to
3110.94 ms at 4K, a 1.34x kernel speedup. Profiled throughput is excluded.

The paired executables use identical saved library and tool objects; only the
expert production object differs. All 18 recorded cache/DMA/first-token
counters matched across the six throughput runs. Individual measurements are
preserved in [OPTIMIZATION.json](OPTIMIZATION.json).

## Validation and rejected experiments

- All 511 per-token likelihoods on a 512-token prose corpus matched exactly.
  Both variants had zero non-finite values, average NLL 1.699666 and perplexity
  5.4721. This checks preservation on that corpus, not general model quality.
- Expert, router and general DeepSeek operations tests passed on each B70.
  The expert gate covers uneven and empty jobs, partial tiles, descriptor
  splitting, ring reuse and guard rows against unchanged reference kernels.
- Independent review found no actionable correctness issue.
- Dense Q8 row tiling was rejected after cache-cold measurements showed
  regressions on important shapes. An MXFP4 lookup table helped less than
  token tiling and was not integrated.

## Concurrent agents

The harness coordinates agent tasks and tools. The engine controls inference
admission and scheduling. In the current source, `--parallel` accepts 1–4:

| Model path | Request scheduling |
|---|---|
| Qwen 27B split, eligible normal decode | Batched decode with separate state banks; deeper requests fall back to time slicing |
| Qwen Flash-Next | Time slicing through state stash/restore |
| DeepSeek | Whole-generation FIFO; multiple prompt-cache conversations do not provide batched decode |

Qwen speculative paths can bypass the normal batch scheduler. DeepSeek uses
both GPUs together for a generation; additional agents do not automatically
receive separate GPU slots. Continuous batching for DeepSeek requires engine
scheduler and per-sequence forward support as well as harness coordination.
