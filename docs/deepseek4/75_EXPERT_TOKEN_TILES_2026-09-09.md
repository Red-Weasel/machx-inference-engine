# Expert token tiling and concurrent agent serving

## Scope

This pass follows the router optimization in report 74. The baseline already
contains that router change. The production candidate changes only grouped
MXFP4 prefill scheduling in `src/ops/deepseek4_experts.cpp`.

## Experiments

Dense Q8 decode was tested with two or four output rows per subgroup to reuse
activation loads. Nine shapes were checked on each B70, including grouped
projections and ragged dimensions. Every output matched the shipped kernel
exactly. Timing rotated through at least 192 MiB of weights, alternated variant
order, discarded two repetitions and used the median of ten repetitions.

The candidates were rejected. For example, the 4096-by-512 projection rose
from 5.93 microseconds to 8.66 or more, and the 4096-by-64640 output head rose
from 478 to 488 or more. The 4096-by-4096 shapes gained at most about 2%.
Reducing activation loads did not produce a useful improvement across the
decode workload.

The next probe targeted grouped MXFP4 experts, using 32 distinct expert weight
banks with uneven token counts. Four variants were compared with alternating
order, four discarded repetitions and the median of ten repetitions:

| Variant | K=1024, N=4096, max M=48, card 0 | Decision |
|---|---:|---|
| Original serial token tiles | 2254 us | Baseline |
| Byte lookup table in local memory | 2114 us | Smaller gain; not integrated |
| Independent eight-token tiles | 1594 us | Candidate, 1.41x |
| Independent 32-token tiles | 2280 us | Rejected |

Card 1 measured 2298 to 1621 us for the eight-token candidate, also 1.42x.
All seven probe shapes on both devices matched exactly. These are synthetic
kernel results; full-model measurements are reported separately below.

## Production change

The grouped MXFP4 grid now indexes expert job, eight-token tile, and output
column group. Each workgroup runs the existing arithmetic helper on at most
eight rows. The host computes the largest token count within the actual
descriptor chunk; smaller jobs return before reading outside their row range.

This removes the serial loop over token tiles from each workgroup and allows
uneven expert jobs to spread across the grid. The weight format, quantization,
lane assignment, integer dot products and floating-point reduction order are
unchanged. All-single-token batches keep their existing decode specialization;
IQ3 and the single-expert reference launchers retain their existing paths.
Highly skewed jobs can create unused grid tiles, so synthetic speedups alone
do not establish a serving improvement.

The extended expert gate covers token counts 0, 1, 7, 8, 9, 15, 16, 17, 31,
32, 33, 48 and 95; 131 and 4096 output columns; descriptor capacities 7 and
64; repeated descriptor-ring reuse; and untouched guard rows between jobs.
It compares against the unchanged single-expert path. The original gate also
checks independent GEMV references, mixed dtypes and deliberate corruptions.

## Full-model validation

Both likelihood runs passed. All 511 per-token NLL values from the 512-token
built-in prose corpus matched byte-for-byte, with zero non-finite values,
average NLL 1.699666 and perplexity 5.4721. The common dump SHA-256 is
`8907d19d5927703a93a6522f62ba30569d2c5571ac12f715981af535f21deaf1`.
This checks preservation on the tested text corpus, not general model quality.

All six unprofiled throughput runs passed. The medians are:

| Workload | Baseline tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|
| Prefill, 128 tokens | 141.24 | 141.81 | +0.41% |
| Prefill, 512 tokens | 280.03 | 284.01 | +1.42% |
| Prefill, 4096 tokens | 513.95 | 537.16 | +4.52% |
| Prefill, 16384 tokens | 478.09 | 504.09 | +5.44% |
| Decode, 128-token starting context | 33.50 | 33.57 | +0.21% |
| Decode, 4096-token starting context | 27.91 | 27.92 | +0.02% |

Every candidate run exceeded every baseline at 4K and 16K. The 4K baseline
runs were 516.236, 513.950 and 513.594 tok/s; candidate runs were 526.905,
539.514 and 537.162. The 16K baseline runs were 479.686, 478.058 and 478.090;
candidate runs were 504.851, 503.027 and 504.088. Shorter-prefill results
overlap, so this pass claims a long-prefill gain and no material decode change.
All 18 recorded cache-hit, cache-miss, DMA-byte and first-token counters
matched across all six runs.

The separate 4K profile reduced grouped MXFP4 GPU time from 4166.8301 to
3110.9356 ms, a 1.34x speedup and 25.3% less time. These timings sum the two
cards during sequential profiling; they are not concurrent model wall time.
Profiled throughput is excluded from the table.

The normal Release build passed. The final CTest run passed all three selected
tests on each B70, six successful test executions total: expert GEMM, router
tiling, and the general DeepSeek operations gate. The strengthened expert test
includes the per-cap sentinel reset. `git diff --check` passed.

The eight-token expert tiling is retained. The dense Q8 and lookup-table
experiments remain evidence only; their production kernels were not changed.
Matched binaries use the same saved library and tool objects, with
only the expert production object replaced. The campaign checks per-token
likelihoods, separate profiles, and three unprofiled runs per variant in
A/B/B/A/A/B order. It uses the same installed DeepSeek Flash Vision-Exp native
MXFP4 checkpoint and two B70s as report 74. Each model run is serialized through
the project guard with a 220 GiB cap and zero process swap. The host has not
been rebooted; these are within-session comparisons.

Evidence and reproduction scripts:
`results/deepseek4-dense-opt-2026-09-09/`.

Independent review found no actionable correctness issue. It checked disjoint
tile writers, subgroup-uniform control flow, integer bounds, widened pointer
offsets, capped descriptor batches and preservation of the all-single-token
dispatch. The strengthened gate passed after the model campaign, with no
compilation competing with a loaded model for host RAM.

## What controls simultaneous agents

The harness owns agent tasks, conversation histories, tool execution and
coordination. The engine owns inference admission, scheduling and batching.
Agents can overlap tool work while their model requests wait, even if the
engine generates for only one request at a time.

In the current working tree:

- The HTTP server admits `Engine::parallel()` generations, configured with
  `--parallel 1..4`, default 1. The engine's FIFO gate controls GPU access.
- The Qwen 27B split backend allocates separate state banks and starts a joint
  decode stepper when parallelism is enabled. Eligible requests within
  `--slot-ctx` use batched decode; larger requests fall back to time slicing.
- The Qwen speculative path can return before reaching that batch stepper.
  Batched decode is therefore not a blanket promise for every 27B mode.
- Qwen Flash-Next has its own stash/restore path and can yield between prefill
  chunks and decode slices under contention. This is time slicing, not joint
  batched decode. Some comments describing all non-27B models as whole-turn
  FIFO are older than this implementation.
- DeepSeek currently takes whole-generation FIFO turns. Its host prompt-cache
  slots preserve multiple conversations but do not batch their decode steps.
  Increasing `--parallel` does not by itself increase DeepSeek decode capacity.

Relevant source: `include/ie/engine.hpp` for admission and state banks,
`src/engine/engine.cpp` for initialization and generation dispatch, and
`src/server/openai_server.cpp` for HTTP admission. DeepSeek's batching design
in report 60 describes the remaining per-sequence runtime and attention work;
its older server description is historical, not the current implementation.

For a DeepSeek agent fleet, the next substantial serving project is continuous
batching with isolated sequence state, per-sequence positions and masks,
chunked prefill, cancellation and backpressure. Kernel improvements help each
request but do not implement that scheduler.
