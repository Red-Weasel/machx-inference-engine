# Intel Arc kernel performance and validation

The newest campaign is at the end: [v0.2.8, lane scheduling and the prefill-attention threshold on the 35B-A3B class](#v028-lane-scheduling-and-the-prefill-attention-threshold),
after [v0.2.6, native integer-dot kernels for Q8_0](#v026-native-integer-dot-kernels-for-q8_0).
The sections before them describe the GLM-5.3-Flash campaign of September 2026.

This update tunes GLM-5.3-Flash and shared SYCL operators on two Intel Arc Pro
B70 GPUs through Level Zero, using IntelLLVM 2026.1.1. Kernel improvements are
reported separately from model throughput. Other devices and unmeasured shapes
retain the relevant fallback paths; they were not qualified by this B70 campaign.

## Implemented changes

- Shape-aware Q8 projection workgroups, exact small row counts, vector loads,
  and single-grid prefill batches. Returned events cover every tile on
  out-of-order queues; unaligned inputs retain bounded scalar loading.
- Smaller FP16 XMX tiles for measured small-M expert GEMM shapes. Output storage
  retains the existing requirement to round the row allocation up to eight.
- Ordered load-ahead for GLM router logits and MLA key/value projections.
  Sparse MLA shares latent tiles across heads and reuses softmax coefficients.
- Expert top-k specializes common expert counts and caches unbiased probabilities.
  Ragged expert counts, larger arrays, invalid counts, all-zero sigmoid
  normalization, and empty-work dependencies are handled explicitly.
- KDA recurrence uses SIMD16 with 256 GRFs on B70 Level Zero. Compiler assembly
  reports 3,136 spill bytes in the reference and no spill directive in the tuned
  kernel. Arithmetic and persistent-state layout remain unchanged.
- Indexer bitonic sort specializes measured padded extents and exchanges short
  distances through subgroup registers. Packed-key ordering, ties, causal masks,
  and padding remain exact. Generic, radix and scan fallbacks remain available.
- Causal convolution avoids history races on short chunks and invalid state
  access for a one-tap kernel.
- The GLM standalone runner defaults to pipelined two-card prefill and selects
  MTP pipedraft for eligible sustained generation. Warm overflowing expert
  batches use smaller waves to retain more cached weights.

## Measured kernel results

These latest three changes were measured against the retained Q8/router build
at the start of the campaign. They are not comparisons against an older GitHub
checkout or a different engine.

| Kernel | Isolated production-kernel speedup | Speedup inside GLM at 24K |
|---|---:|---:|
| Expert top-k | 1.49–1.68× | 1.61× |
| KDA recurrence | 1.93–2.62× | 1.33× |
| Attention indexer | 1.24–1.85× at decode shapes | 1.24× |

The in-model bucket comparison used 64 generated tokens with pipedraft disabled.
Q8 register forcing and parallel Sinkhorn were rejected because their primary
shapes did not improve. Neither experiment changes the shipped dispatch.

## Whole-model qualification

The original warm 8K comparison measured prefill **−0.47%** and decode **−2.55%**;
the 24K comparison measured **−1.87%** and **−0.71%**, respectively. An initial
combined-build first prefill chunk also incurred approximately 5.7 seconds of
extra first-use time. A later repeat did not show that cost. No compilation
trace was taken, so its cause is not asserted.

Regression isolation used 14 complete 8K-prompt / 512-token-generation runs:

| Configuration | Baseline decode | Combined decode | Interpretation |
|---|---:|---:|---|
| Original pinned configuration, two repeats each | 13.69 tok/s | 13.69 tok/s | No consistent combined regression reproduced |
| Larger pin-memory reserve, one pair | 9.26 tok/s | 9.38 tok/s | Rejected; worse overall performance and I/O |
| Unpinned banks, ABBA | 8.34 tok/s | 8.48 tok/s | Combined +1.69%, but the mode is slower overall |

The initial four-build order was baseline, router, router+KDA, combined, combined,
router+KDA, router, baseline. The same KDA binary ranged from 36.03 to 41.18 seconds
for decode. Slower runs showed increased expert staging, CPU work and physical
reads. Timers overlap and do not prove a unique cause. Nonlinear host/cache drift
also confounds aggregate prefill comparisons.

The larger-reserve sequence was stopped after its first pair because the
intervention worsened performance. No four-run confidence is claimed for it.
All 14 model runs exited successfully; all 13 comparisons to the first baseline
matched generated text, draft counts and expert-cache counts exactly. Keep the
original runtime settings. A small kernel-related effect remains unresolved
within the timing variation; **a combined whole-model speedup is not established**.

## Reproducing checks

After configuring with Intel's SYCL compiler, build the kernel gates:

```bash
cmake --build build --target g5_router_topk_test kda_recurrence_test \
  indexer_topk_test g5_router_test gemv_q8_soa_dual_test \
  gemm_fp16_shape_test conv1d_state_test -j2
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ctest --test-dir build \
  -R '^(g5_router_topk_test|kda_recurrence_test|indexer_topk_test|g5_router_test|gemv_q8_soa_dual_test|gemm_fp16_shape_test|conv1d_state_test)$' \
  --output-on-failure
```

The latest top-k, KDA and indexer gates cover 176, 102 and 240 configurations
across both B70s. KDA checks four dependent state updates per configuration.
Selection gates use frozen GPU references and independent CPU oracles; tests
exercise out-of-order dependencies, guards, ragged shapes and dispatch boundaries.

A matching full-model comparison requires the same GLM-5.3-Flash UD-Q4_K_XL
GGUF and WikiText corpus for both binaries. Example for each saved build:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
IE_G5_CPU_MISS=1 IE_G5_PIN_BANKS=1 IE_G5_PIN_MAX_GIB=95 \
./scripts/ie-run-guarded --mem 220G --timeout 900 \
  ./build/tools/ie-glm5next-run /path/to/model-00001-of-00006.gguf \
  --gpus 2 --ctx 16384 --ppl /path/to/wiki.test.raw \
  --prefill 8000 --chunk 1024 --ngen 512
```

Run one GPU workload at a time, avoid overlapping compiler work, record actual
pinned-layer allocation, keep slow runs, compare exact continuations and use
alternating controls. Separate cold/first-use observations from steady-state
claims. Host-resident expert workloads need repeated model measurements in
addition to isolated kernel timings.

<a id="v026-native-integer-dot-kernels-for-q8_0"></a>
## v0.2.6: native integer-dot kernels for Q8_0

Measured October 1, 2026 on two Arc Pro B70 cards (compute-runtime 26.35), on community fine-tunes of the
Qwen3.6-35B-A3B class and of Qwen3.8-27B (Q8_0) and on Qwen3.8-Flash. Kernel times are device time per launch on
card 0; model figures are single runs unless a line says A-B-A.

### What was wrong

The engine's portable integer-dot helper (`ie::dp4a_ss` over vector-loaded words) was lowered by the compiler through
byte extracts and inserts: 12 byte moves per operand word. Counted in the B70 code of the previous kernels, per
integer dot:

| Kernel (Q8_0) | Instructions per dot, before | Of them byte moves | After |
|---|---:|---:|---:|
| MoE prefill down | 33.8 | 24 | about 4.25 |
| MoE prefill gate+up | 9.7 | about two thirds | about 2.7 |
| MoE decode / lane rows, gate+up and down (dot blocks) | 20.4 and 27.4 | 18 and 24 | 2.5 and 3.2 |
| GEMV, one token (dot block) | 28.0 | 24 | 4.0 |
| GEMV, 2–16 rows (dot blocks) | 14.8–21.1 | 12.75–18 | 2.05–3.12 |

The new kernels use the compiler's native integer dot (`include/ie/dp4a_native.hpp`). The MoE kernels also keep a
tile's weights in registers and compute several output columns per subgroup; the GEMVs change only the dot. Each
kernel lives in its own source file beside the one it replaces, with a switch that selects the old one (`IE_Q8_MOE_DOWN_V2`, `IE_Q8_MOE_GATEUP_V2`, `IE_Q8_MOE_DECODE_V2`,
`IE_Q8_SOA_BATCHED_V2`, `IE_Q8_SOA_GEMV_ND`, `IE_Q8_SOA_GEMV_G_ND`; `=0` each). They keep the old kernels' per-lane
partial sums, reduction trees and final rounding, so their output is bit-identical.

### Measured kernel results

| Kernel | Shape | Before | After | Speedup |
|---|---|---:|---:|---:|
| MoE prefill down | 35B-A3B class, 2,048 rows | 17.56 ms | 3.15 ms | 5.57× |
| MoE prefill down | skewed expert load | 43.27 ms | 7.13 ms | 6.07× |
| MoE prefill gate+up | 35B-A3B class, 2,048 rows | 10.57 ms | 6.26 ms | 1.69× |
| MoE prefill gate+up | skewed expert load | 25.19 ms | 13.86 ms | 1.82× |
| MoE decode gate+up, 1 / 2 / 4 / 8 / 16 rows | 35B-A3B class | 16.6 / 50.8 / 111.7 / 215.2 / 423.5 µs | 10.3 / 45.6 / 107.8 / 212.7 / 415.8 µs | 1.61 / 1.11 / 1.04 / 1.01 / 1.02× |
| MoE decode down, 1 / 2 / 4 / 8 / 16 rows | 35B-A3B class | 15.5 / 33.8 / 69.4 / 147.3 / 288.8 µs | 7.8 / 19.8 / 39.3 / 96.7 / 181.4 µs | 1.99 / 1.71 / 1.77 / 1.52 / 1.59× |
| GEMV, 16 rows, single / dual launch | 27B, 35B-A3B and Qwen3.8-Flash shapes | — | — | 1.11–1.45× / 1.11–1.37× |
| GEMV, 8 rows | the same | — | — | 1.15–1.43× / 1.11–1.37× |
| GEMV, 4 rows | the same | — | — | 1.14–1.88× / 1.10–1.88× |
| GEMV, 2 rows | the same | — | — | 1.02–3.28× / 1.02–1.74× |
| GEMV, one token, large shapes | 27B feed-forward and attention-q, Qwen3.8-Flash LM head | — | — | 1.00–1.04× |
| GEMV, one token, small shapes | 27B k/v, 35B-A3B projections and shared expert, Qwen3.8-Flash linear attention | — | — | 1.16–1.58× |

Two estimates did not hold. The instruction count predicted about 3.5× for the 16-row GEMV and the device gave
1.11–1.45×: after the change that kernel streams weights at 116–223 GB/s, so it is bound by something other than
instructions or bandwidth (not profiled). The prefill gate+up was estimated at 2.5–3.5 ms and measures 6.26 ms, and
the decode gate+up gains nothing from 4 rows up. The large one-token GEMVs were already at 505–600 GB/s.

### Whole-model results

| Model | Workload | Before | After |
|---|---|---:|---:|
| 35B-A3B class | one request, cold prompt of 2,038 / 8,065 / 16,369 / 32,161 tokens, wall time, A-B-A | 2.20 / 9.04 / 22.72 / 60.86 s | 1.41 / 5.96 / 16.44 / 48.38 s |
| 35B-A3B class | the same with the down kernel only | — | 1.61 / 6.74 / 18.06 / 51.64 s |
| 35B-A3B class | `--parallel 16`, short prompts, 1 / 4 / 8 / 16 requests, A-B-A on the decode kernels | 80.8–81.1 / 230.0–230.8 / 315.2–316.7 / 390.6–391.3 tok/s | 81.9–82.4 / 236.1 / 331.9 / 414.1 tok/s |
| 35B-A3B class | agent replay (15 workers, 3 turns, then a 15-wide shared-prompt wave), MoE kernels | 214.9 s, 66.3 tok/s | 169.6 s, 84.2 tok/s |
| 35B-A3B class | the same replay, GEMV kernels | 144.8 s, 99.0 tok/s | 139.7 s, 101.2 tok/s |
| Qwen3.8-27B | `--ctx 8192 --parallel 16`, 1 / 4 / 8 / 16 requests, A-B-A on the GEMV kernels | 17.1–17.2 / 61.5 / 92.3 / 113.7–113.8 tok/s | 17.2–17.3 / 64.5 / 111.3 / 153.5 tok/s |
| Qwen3.8-Flash | `--ctx 8192 --parallel 16`, 16 requests, A-B-A on the GEMV kernels | 124.6–124.8 tok/s | 128.0 tok/s |

The down kernel carries about three quarters of the prefill gain. Single-request decode moves by less than 1 % with
the GEMV kernels and by 1.4 % with the MoE decode kernels; the gain is in the lanes' row steps, largest on the dense
27B.

### Validation

- Kernel tests compare each new kernel with the old one bit for bit, check that every output is written, and bound
  both against a host fp64 reference: 7 + 8 + 24 MoE cases and 156 + 38 GEMV cases passed on card 0. One MoE decode
  case first failed on a sample whose reference lies beyond the fp16 range, where both kernels store the same
  infinity; the test now accepts that, and the 24 cases pass.
- Greedy replies are identical with the switches on, off, and on the previous build: a five-prompt set at
  `--parallel 1` (one prompt of 32,979 tokens) and four prompts solo and batched at `--parallel 4`, on the 35B-A3B
  class for both kernel families and on Qwen3.8-27B and Qwen3.8-Flash for the GEMVs.
- Perplexity on the 35B-A3B class (a 12,288-token prefill, then 511 scored tokens of wikitext-2) is 2.3529, the same
  negative log-likelihood to six decimals in every configuration. Needles at 32K and 54K with the MoE kernels: 6 of
  6, the same replies as before.
- Not covered: the unit tests on card 1 and the speculative-verify paths. The opt-in `IE_GEMV_SMALLK` kernel is
  unchanged; on an OpenCL CPU device it leaves the valid columns of a partial last work-group unwritten when the
  column count is not a multiple of 8 (the model shapes it serves are multiples of 8).

### Reproducing

```bash
cmake --build build --target moe_q8_down_v2_test moe_q8_gate_up_v2_test moe_q8_decode_v2_test \
  gemv_q8_soa_batched_v2_test gemv_q8_soa_nd_test -j2
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/tests/moe_q8_down_v2_test            # correctness, old kernel == new kernel
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/tests/moe_q8_down_v2_test --bench    # device time per launch, old and new
```

The other four tests take the same `--bench` argument. For a model-level A/B, start the same build twice, once with
the switches at `=0`, and compare; the first request of a new build compiles the new kernels once, so do not time it.

<a id="v028-lane-scheduling-and-the-prefill-attention-threshold"></a>
## v0.2.8: lane scheduling and the prefill-attention threshold on the 35B-A3B class

Measured October 2, 2026 on two Arc Pro B70 cards (compute-runtime 26.35), on a community fine-tune of the
Qwen3.6-35B-A3B class (Q8_0), with v0.2.6's agent replay: 15 worker conversations for 3 turns each, then a wave of 15
conversations that share one system prompt, 75 requests against
`ie serve --gpus 2 --ctx 262144 --parallel 16 --max-queue 16 --thinking on`. No kernel was written for this release:
two changes are to the order in which the two-card lane pipe runs its steps, one is the position from which an existing
kernel is used. Step times are read from the server's trace (`IE_LANES_TRACE=1`, one log line per card step) and are
medians on card 0; card 1 runs 1–3 % slower. Each build was measured against the build before it, two runs each.

### What the trace showed

**The paused turns.** In v0.2.6 every serial turn (a cached prefix's restore, the shared-prefix mark, the prompt's end)
paused the pipe until every step in flight had landed. One replay on v0.2.6:

| turn | pauses | paused time | share |
|---|---:|---:|---:|
| prompt end | 74 | 14,725 ms | 89 % |
| restore, mark | 10 | 1,749 ms | 11 % |
| all | 84 | 16,474 ms | |

66 of the pauses waited only for decode groups (5,596 ms in all); 18 waited for another lane's prefill piece
(10,878 ms, 200–2,175 ms each). The prompt-end turn paused for one reason: it ran the prompt's last 3–6 rows as a
full two-card forward on the cards' shared workspace. With those rows sent through the pipe as a piece and the turns'
copies addressed by lane, the turns run beside the pipe:

| workers phase of the replay | v0.2.6 | turns beside the pipe |
|---|---:|---:|
| pauses, paused time over the replay | 84–86, 13.3–13.4 s | 0, 0.7 s |
| card 0 idle | 19.9–20.5 s (24.5–25.0 %) | 10.7–10.8 s (13.7–13.8 %) |
| card 0 idle in gaps of 50 ms or more before a decode step; the longest gap | 15.9–16.6 s; 2,068–2,070 ms | 5.7–5.8 s; 291–292 ms |
| card 1 idle | 18.1–18.7 s | 8.4–8.5 s |

**Decode groups.** A group of decoding lanes is one card step. Its time by size (workers phase, the same in both
builds): 1 row 5.7–5.8 ms, 2 rows 7.7, 4 rows 11.3, 6 rows 15.1, 8 rows 17.5–17.7 ms, so a row costs 5.7 ms alone and
2.2 ms in a group of 8. Without the pauses, whose releases had resubmitted every parked lane at once, the groups
stayed as the lanes' arrival aligned them; the regroup (card 0 waits up to 20 ms for the decode group landing on
card 1 before forming a group that would leave three or more rotating) restores the formation:

| workers phase, card 0 | decode steps | rows per step | 1-row steps | decode card time |
|---|---:|---:|---:|---:|
| v0.2.6 | 1,595–1,599 | 4.53–4.54 | 164–184 | 19.6 s |
| turns beside the pipe | 2,184–2,196 | 3.30–3.31 | 792–819 | 21.9–22.4 s |
| with the regroup | 1,586–1,587 | 4.56 | 175–177 | 19.6–19.7 s |
| with the attention threshold at 512 (v0.2.8) | 1,654–1,655 | 4.37 | 285–288 | 19.9 s |

The builds decode 7,235–7,236 rows in this phase; all of them in 8-row steps would be 904 steps, about 16 s. The regroup's waits
held card 0 for 636–650 ms over the replay (37–46 waits, 33–38 of them brought lanes, 4–8 timed out).

**Prefill pieces by depth.** A 512-row prefill piece (the size a prompt is read in beside running lanes) by the
position it starts at, before and after the tiled attention kernel's threshold moved from 6,144 positions to 512:

| start of the piece | 0–2K | 2–4K | 4–6K | 6–8K | 8–16K | fit below 6,144 | fit from 6,144 |
|---|---:|---:|---:|---:|---:|---|---|
| threshold 6,144 (v0.2.6) | 208.0 ms | 247.1 | 295.8 | 276.6 | 366.1 | 155.3 + 30.44 ms per 1,000 positions | 159.0 + 18.06 |
| threshold 512 (v0.2.8) | 190.1 ms | 215.0 | 248.8 | 276.0 | 366.5 | 159.0 + 18.17 | 158.8 + 18.08 |
| pieces in the bucket | 30 | 49 | 46 | 9 | 22 | 115 | 41 |

Below 6,144 the previous kernel cost 30 ms per 1,000 positions of depth where the tile costs 18, and the tile has no
overhead at depth 0 (a 512-row piece over 512 keys: 156.5 ms against 158.0), so the old threshold made a piece at 4–6K
slower than one at 6–8K. On v0.2.6 with the environment knob, one run per value, a threshold of 1,024 measured the
same speed as 512 (replay 123.5 s both, workers phase 75.6 and 75.3 s) and 2,048 less (workers phase 77.4 s, against
83.2 s at 6,144). 512 was chosen because at 1,024 or 2,048 the first rows of a long prompt still take a different
kernel depending on how the prompt was cut.

### Whole-model results

| Change, each against the build before it | Replay | Workers phase | Workers' decode per request, median / 5th percentile | Workers' first token, median |
|---|---:|---:|---:|---:|
| turns beside the pipe | 132.4 / 131.3 → 127.7 / 127.7 s | 82.0 / 82.8 → 78.8 / 78.5 s | 8.65 / 6.88, 8.80 / 6.50 → 8.59 / 7.66, 8.61 / 7.65 tok/s | 3.79 / 4.79 → 4.16 / 4.00 s |
| decode regroup, a lone prompt's re-cut, the tail piece first | 127.8 / 128.9 → 122.2 / 122.2 s | 78.5 / 79.8 → 75.9 / 75.8 s | 8.76 / 7.68, 8.61 / 7.24 → 8.85 / 7.87, 8.93 / 7.84 | 4.15 / 4.15 → 4.06 / 4.01 s |
| attention threshold 6,144 → 512 | 122.2 / 122.2 → 116.3 / 116.5 s | 75.9 / 75.8 → 70.3 / 70.3 s | 8.85 / 7.87, 8.93 / 7.84 → 9.83 / 7.12, 9.86 / 7.13 | 4.06 / 4.01 → 2.83 / 2.58 s |

With the threshold at 512 the workers' prefill goes from 1,085–1,086 to 1,171–1,172 tok/s together and the card-0
prefill time of the workers phase from 45.1 to 39.8 s. One request alone (`--parallel 1 --ctx 40960`, wall time for the
prompt plus one token, measured with the threshold set by environment on the v0.2.6 build): 2,038 tokens 1.41 → 1.28 s;
8,065, 16,369 and 32,161 tokens unchanged (5.95 → 5.94, 16.45 → 16.43, 48.45 → 48.48 s), because a prompt alone is read
in 8,192-row pieces that reach 6,144 either way.

Three expectations did not hold. Moving the turns beside the pipe was expected to raise the median per-request decode
toward the 25.9 tok/s of 16 lanes that only write; it stayed at 8.6, because the smaller groups took the time the
pauses gave back. The regroup was expected to reach 6–8 rows per step; it reaches 4.56, v0.2.6's formation, because
the 15 lanes are rarely all decoding at once (62 eight-row steps against 413 six-row steps). And smaller prefill
pieces do not help the agents: with 384- and 256-row pieces instead of 512 the median decode fell from 8.6 to 8.45
and 8.34 tok/s and the replay took 3.0 and 5.7 s longer.

The first prompt of the shared-prompt wave gets its first token later than on v0.2.6: 11.29–11.31 s against 8.56–8.57 s
(v0.2.6 when the leader interleaved with its followers) or 10.22 s (when it prefilled alone first). The cause is not
established. Its prefill takes 8.9 s, as v0.2.6's alone path (8.8 s); the time from its first sampled token to the
client's first byte is 2.30–2.42 s against 1.42–1.43 s on v0.2.6. On the build before the threshold change 23 other
pieces (2.03 s of card 0) ran between its tail piece and its first decode step, but a test build that ran that step
first (1 piece, 84 ms) gave the same first token (11.21–11.23 s), so the decode queue is not the cause. The 14
followers' first tokens come at 11.29–11.31 s, against 11.28–11.36 or 12.93–13.00 s on v0.2.6.

### Validation

- Greedy replies after the first change equal v0.2.6's on the 35B-A3B class, Qwen3.8-27B, Qwen3.8-Flash,
  MiMo-V2.6-Flash and DeepSeek-V4.1-Flash: 156 checks, 0 differences. With `IE_Q35MOE_TURN_DRAIN=1` the replay's 75
  replies equal v0.2.6's. After the regroup: 124 checks, 0 differences (35B-A3B class and Qwen3.8-27B), and the
  replay's 75 replies equal the build before. Each build's two replay runs gave the same 75 replies.
- The release build with `IE_Q35MOE_FA2_TILE_MINCTX=6144` equals v0.2.6's records byte for byte on the check's short
  prompts, tool call and batch of four; at its default, `--parallel 16` and `--parallel 1` agree on every greedy
  request, a 1,060-token and a 32,979-token prompt included, and a batch of four short prompts equals the same prompts
  alone (70 checks in all, 0 differences). Between the two thresholds the 1,060-token prompt's reply differs, by
  design; the replies to prompts of 23–68 tokens and to the 32,979-token prompt do not. The release build's 75 replay
  replies equal those of v0.2.6 with the knob at 512.
- Perplexity (wikitext-2, one prefill chunk at position 0, then 511 scored tokens), threshold 512 against 6,144:

  | prefill chunk | kernel at 512 / at 6,144 | perplexity at 512 | at 6,144 | change |
  |---|---|---:|---:|---:|
  | 12,288 rows | tile / tile | 2.3529 | 2.3529 | 0 |
  | 4,096 | tile / previous | 9.3172 | 9.3232 | −0.064 % |
  | 2,048 | tile / previous | 5.3012 | 5.2915 | +0.183 % |
  | 1,024 | tile / previous | 3.4606 | 3.4779 | −0.497 % |
  | 512 | tile / previous | 3.9006 | 3.8995 | +0.028 % |

  Each row scores a different 511-token window, so only the pair in a row compares. The sign changes between windows.
- Needles at 10 / 50 / 90 % depth in 2,043- and 3,580-token prompts: 6 of 6 found, replies byte-identical at both
  thresholds.
- 0 GPU fault lines after each of the 40 GPU jobs of the three verification runs; every server stopped through
  `POST /admin/shutdown`.
- Not covered: only the first change was measured with both builds alternating in one session; for the other two the
  reference is the previous build's runs of earlier the same day (at most three hours before, the same prompts);
  the one-request 2,038-token prefill was not re-measured on the release build; the lead-and-six-workers shape was not re-run after the regroup or the threshold change; Qwen3.8-Flash,
  MiMo-V2.6-Flash and DeepSeek-V4.1-Flash were not re-run after the first change; needles at 32K and 54K were not
  re-run (every piece of a prompt that long reaches 6,144 at either threshold); a perplexity span longer than 511
  tokens was not scored, and the 1,024-row figure sits 0.003 % inside the 0.5 % limit the check used.

### Reproducing

```bash
IE_LANES_TRACE=1 ./build/src/ie serve <35B-A3B-Q8_0.gguf> --gpus 2 --ctx 262144 --parallel 16 --max-queue 16 --thinking on
python3 tools/swarm_replay.py --port 11435 --out <dir> --scenario both --lanes 15 --turns 3 \
  --append-min 1000 --append-max 3000 --wave-tokens 12000 --wave-turns 1
./build/tools/ie-perplexity --gguf <35B-A3B-Q8_0.gguf> --gpus 2 --ctx 16384 \
  --prefill-chunk 1024 --max-tokens 1536 --text /path/to/wiki.test.raw
```

Run the server once per setting (`IE_Q35MOE_TURN_DRAIN=1`, `IE_Q35MOE_REGROUP=0`, `IE_Q35MOE_FA2_TILE_MINCTX=6144`)
and compare; the trace's `[lanes card]` lines give each card step's rows and time. The replay builds its prompts from
the checkout's `src/` and `include/`, so compare two runs only on the same tree. For perplexity, `--max-tokens` is the
chunk plus 512.
