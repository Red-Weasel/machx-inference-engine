# GLM-5.3-Flash serve speed — 2026-10-19

Goal: make GLM-5.3-Flash faster in real use (`ie serve`, Dream's agent loop). The model had to be
re-downloaded to the main drive because it streams its experts from disk and host RAM. Evidence lives in
`results/glm53-2026-10-19/`: per-leg `*.serve.log`, `*.turns.json`, `gate_all.log`, `chunk_sweep.log`, `pp_ab.log`,
and `summarize.py` builds the table.

## What was wrong in serve

| gap | effect in an agent conversation |
|---|---|
| no prompt cache | every turn re-read the whole conversation: TTFT 243 / 273 / 311 s on turns 1-3 (9.6K-12.4K tokens) |
| serial two-card prefill | the standalone runner's pipelined prefill (stage A on chunk k+1 while stage B runs chunk k) was never wired into serve |
| no MTP draft | serve decoded 10.8-11.5 tok/s; the runner's MTP pipelined draft was opt-in and runner-only |
| 256-token prefill chunk | prefill streams the host expert bank once per chunk; 256 is a Qwen bound and GLM inherited it |

## The re-download changed the architecture name

Unsloth rewrote the GGUF metadata on 2026-10-05: `general.architecture` became `glm5-next` (llama.cpp's name), and
every `glm5next.` key became `glm5-next.`. Shard 1 holds only metadata (9.4 MB, 73 keys); tensor names are unchanged,
and all 52 tensor names the loader binds are present. One new key, `attention.indexer.index_share_mtp = True`, is not
read by the engine. The live v0.2.25 binary refuses the file with "unsupported architecture: glm5-next".

- The engine now accepts both names: `detect_arch`, and `read_glm5next_config` reads keys under the file's own prefix.
  `ie capabilities` reports the canonical `glm5next`, since Dream's launcher keys on it.
  Test: `tests/unit/glm5_config_test.cpp` (both names, a mismatched prefix, and the real file via `IE_GLM53_GGUF`).
- For the old binary: `~/models/GLM-5.3-Flash-GGUF/compat-glm5next/` holds shard 1 with the keys renamed back
  (`results/glm53-2026-10-19/gguf_rename_arch.py`), plus symlinks to shards 2-6.
- Dream reads `general.architecture` from the GGUF itself (`dream/local/model_defaults.py:243`). For the new file
  it sees `glm5-next`, so its GLM recommendations are skipped until Dream learns the alias. Loading the compat
  directory avoids this.

## Gate (2026-10-19, 17:07-17:52)

Configuration: `ie serve --gpus 2 --ctx 200000 --threads 8`, under `ie-run-guarded`.

Conversation (`agent_turns.py`):
- A ~9.6K-token system prompt (SERVER_CONTROLS.md) plus three tools.
- Each later turn feeds back a ~1.4K-token tool result.
- Speed legs: model-card sampling (temperature 1.0, top_p 0.95, top_k 20), seed 7, 128 max tokens.
- Old binary: `build/src/ie`, v0.2.25, 45504970, reading the compat shard. New binary: e23711f4, reading the
  original file.

| leg | turn 1 TTFT (9.6K cold) | turn 2 TTFT | turn 3 TTFT | decode tok/s |
|---|---|---|---|---|
| S1 old (no cache, serial, chunk 256) | 243.2 s (39 tok/s) | 273.3 s (0 cached) | 310.8 s (0 cached) | 10.8 / 11.5 / 10.9 |
| S2 new, chunk 256 | 129.1 s (74 tok/s) | 22.2 s (9,590 cached) | — (harness) | 11.5 / 10.0 |
| S3 new, chunk 1024 | **65.3 s (148 tok/s)** | **16.3 s** (9,590 cached) | **16.1 s** (11,000 cached) | 10.9 / 11.5 / 10.9 |
| S4 new, chunk 256 + MTP draft | 135.1 s (71 tok/s) | 23.2 s | 23.5 s | **13.4 / 13.6 / 13.2** (drafts taken 80 / 73 / 54 %) |

- **S2 turn 3:** turn 2's reply hit the 128-token cap partway through a tool call. The client then sent that cut-off
  call back as history, and the server correctly refused it ("incomplete assistant tool history"). The client now
  strips a length-truncated tool call.
- **S3 replies are byte-identical to S1's on all three turns** (md5 ba460e7b, 5876d318, f0d6048e). The cached-turn
  restore produced exactly what the old binary's full re-prefill produced, even with the q* CPU expert leg on.

Exactness legs (greedy, `IE_G5_CPU_MISS=0`, which makes continuations independent of the expert-cache schedule):

- cache restore: re-sending turn 2 restored 10,995 of 10,997 tokens, answered in 0.5 s, and gave the identical
  reply: **PASS**
- pipelined prefill == serial prefill (`IE_G5_NO_PIPELINE=1`), both turns: **PASS**
- MTP pipelined draft == plain decode, both turns: **PASS**

## Why prefill is slow, and the chunk lever

- Each B70 sits on PCIe 5.0 **x8** (the root ports at 00:06.0 and 00:06.3 run 32 GT/s x8), about 25 GB/s usable.
- A stage's host expert bank is about 88 GB. Prefill touches nearly every expert in every chunk, so about 0.2% of
  them hit the expert cache (CAMPAIGN_2026-09-03). The Belady-style victim choice keeps the ~68 cached slots per
  layer across chunks, so about 76% of the bank streams per chunk: roughly 73 GB, about 2.9 s per stage at x8.
- Measured per chunk at 1024: about 6.9 s per stage. The other ~4 s is per-expert compute and launch overhead
  (gemv_q4_K_rows, about five launches per expert).
- Fewer chunks means fewer bank streams. Chunk 256 → 1024 doubled prefill (74 → 148 tok/s) and cut a 1.4K-token
  turn's TTFT from 22 s to 16 s, with no decode loss: the expert cache shrank about 1 GiB per card.

### Chunk sweep (MTP draft on, same conversation; `chunk_sweep.log`)

| chunk | cold 9.6K prefill | per-turn prefill (1.4K new) | decode tok/s | expert cache (card 0 / card 1) |
|---|---|---|---|---|
| 256 (gate S4) | 71 tok/s | 60-62 tok/s | 13.2-13.6 | 19.4 / 21.4 GiB |
| **1024** | **146 tok/s** (66.2 s) | **86-88 tok/s** (16.0 s) | **13.9-14.2** | 18.4 / 20.4 GiB |
| 2048 | 147 tok/s (65.3 s) | 86 tok/s (16.4 s) | 13.0-13.9 | 17.1 / 19.0 GiB |
| 4096 | every request fails | — | — | 14.4 / 16.3 GiB |

- Past 1024 prefill gets no faster. Fitting the three working points as
  pipelined-time = (chunks + 1) × (fixed + per-token × rows) gives about 2.4 s fixed (streaming the misses) plus
  about **3.4 ms per token per stage**. So above 1024, prefill is bound by per-token compute: the per-expert multi-row
  GEMVs (`gemv_q4_K_rows`, not XMX) run at about 2.6 TFLOP/s effective. At 2048 the smaller expert cache also costs
  decode 3-7%.
- 4096 loads, but every forward throws "Provided range and/or offset does not fit in int": a kernel's 32-bit SYCL
  range overflows. The likely culprit is the sparse attention, rows × 64 heads × 2051 selected × 4, which passes 2³¹
  at 4096. GLM's load used to accept 1..8192; it now refuses anything above 2048, the largest measured
  (`kGlm5MaxPrefillChunk`).
- **Default then: 1024** (`kGlm5PrefillChunk`). The MTP draft costs about 1% of prefill at 1024; at 256 it cost
  4-5%. Superseded by [Prefill on XMX](#prefill-on-xmx-1815-1910): with the compute cut, the default is 2048.

### Runner prefill bench vs serve

`ie-glm5next-run --ppbench` keeps the expert banks on mmap unless `IE_G5_PIN_BANKS=1`, while serve pins them by
default. Unpinned, the CPU copies every miss into the pinned ring: `pool_copy` took 55-60 s of a 101 s, 10-chunk
run, giving 101 tok/s against serve's 146. Prefill A/Bs therefore need `IE_G5_PIN_BANKS=1` to measure the served
path.

## Prefill on XMX (18:15-19:10)

A per-kernel profile of the pinned runner at chunk 1024 (`pp_prof1024.log`, `IE_QUEUE_PROFILING=1`: the GPU buckets
are trustworthy, the wall is inflated) showed where the per-token compute goes:

| kernel | GPU time, 4 chunks x 2 stages | what it is |
|---|---|---|
| `gemv_q4k_rows` | 14.3 s (186,566 calls) | routed experts, Q4_K, multi-row GEMV |
| `gemv_q8_soa_f16_T` | 8.1 s (1,060 calls) | dense Q8_0 projections (attention, shared expert), T-row int-dot GEMV |
| `gemv_q5k_rows` | 6.4 s | routed experts, Q5_K |
| `gemv_q8_soa_f16_Td` | 5.3 s | dense Q8_0 dual projections |
| `g5_attend_sel` | 2.5 s | DSA sparse attention |
| `g5_kabsorb` + `g5_vabsorb` | 2.9 s | MLA absorb, a plain per-element dot kernel |

None of the four top buckets used XMX. A standalone bench (`tests/unit/glm5_moe_rows_vs_xmx_bench.cpp`, card 1,
`moe_rows_vs_xmx_bench.txt`) put one expert's gate/up GEMM (K 4096, N 2048) through the shipped rows GEMV and through
dequant-to-fp16 + `gemm_fp16` (joint_matrix):

| rows | 16 | 32 | 48 | 64 | 96 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| rows GEMV ms | 0.062 | 0.123 | 0.188 | 0.249 | 0.373 | 0.495 | 0.998 |
| dequant + XMX ms | 0.244 | 0.244 | 0.243 | 0.189 | 0.247 | 0.200 | 0.313 |

The XMX chain costs about the same at any row count: it is the dequant (about 0.2-0.27 ms per matrix). The crossover
is about 60 rows. The outputs agree to rel 4e-4 (fp32 accumulation either way; the GEMV rounds to fp16 at the end).

Two switches in `Glm5NextModel` (`init_runtime`, `compute_e`, `mm`):

- `IE_G5_XMX_ROWS` (default **64**, 0 = off): a routed expert with at least that many rows in this chunk takes
  dequant + XMX. Decode never does (it gives an expert at most a few rows).
- `IE_G5_Q8_GEMM_ROWS` (default **32**, 0 = off): a dense Q8_0 projection with at least that many rows is dequantized
  into a reusable fp16 buffer (`q8_w16_`, the largest dense matrix, reserved in `glm5_runtime_reserve`) and runs
  `gemm_fp16`. Decode (one row, or a few with the draft) keeps the int-dot GEMV.

Pinned runner, 10,240 tokens of wikitext (`pp_pin.log`, `pp_pin2.log`, `pp_pin3.log`; expert hit rate is the same
with and without the switches, so the gain is compute):

| chunk | baseline | experts >= 64 rows on XMX | + dense Q8 on XMX |
|---|---|---|---|
| 1024 | 152.9 tok/s | 162.9 | **214.3** |
| 2048 | 147.5 | 181.4 | **278.2** (x1.89) |

With the compute cut, the chunk lever works again: 2048 streams the bank half as often. The GLM default chunk is
now **2048** (`kGlm5PrefillChunk`; the cap stays 2048).

`IE_G5_PP_TILES=1` (the runner's prefill tiling) with pinned banks faulted card 1 (GPU page fault, DEVICE_LOST,
`pp_pin.log` 18:18:57). Card 1 passed the bench right after, and no fault has been logged since. Tiles stay
opt-in; the fault is not investigated.

### PPL gate (wikitext, ctx 8192, chunk 2048, banks pinned; `ppl_ab.log`)

Runner 7a633f6b (the switches still opt-in in that build), each leg in its own process; scored by the chunked
prefill path, so every expert >= 64 rows and every dense projection ran on XMX in the second leg:

| leg | PPL | NLL | predictions |
|---|---|---|---|
| rows GEMV (`IE_G5_XMX_ROWS=0 IE_G5_Q8_GEMM_ROWS=0`) | 1.8118 | 0.59433 | 8,191 |
| XMX (`IE_G5_XMX_ROWS=64 IE_G5_Q8_GEMM_ROWS=32`) | **1.8097** | 0.59318 | 8,191 |

**PASS**: −0.12 %, no worse. The q* CPU expert leg was on in both legs (the served configuration), so part of the
difference is that leg's run-to-run variation; the claim is "not worse", not "better". Whole leg, load included:
3 min 11 s rows, 2 min 28 s XMX.

### Final serve check (19:00-19:09; `final_serve.log`, ie 256452d1, no GLM flags)

Same conversation and sampling as the gate (model-card sampling, seed 7, 128 max tokens), `ie serve` with the
defaults: prompt cache, pipelined prefill, MTP draft, chunk 2048, prefill on XMX.

| leg | turn 1 TTFT (9.6K cold) | turn 2 TTFT | turn 3 TTFT | decode tok/s | expert cache card 0 / card 1 |
|---|---|---|---|---|---|
| S1 old binary (v0.2.25) | 243.2 s (39 tok/s) | 273.3 s | 310.8 s | 10.8 / 11.5 / 10.9 | — |
| gate, chunk 1024 + draft (no XMX) | 66.2 s (146 tok/s) | 16.3 s | 16.0 s | 14.0 / 14.2 / 13.9 | 18.41 / 20.36 GiB |
| **final defaults (chunk 2048 + XMX)** | **38.7 s (251 tok/s)** | **10.8 s** | **10.7 s** | 13.4 / 13.6 / 12.3 | 16.94 / 18.90 GiB |
| final build, `--prefill-chunk 1024` | 51.8 s (187 tok/s) | 12.9 s | 12.7 s | 13.8 / 13.4 / 13.5 | 18.28 / 20.24 GiB |

- Turn 1's reply is byte-identical to the old binary's (md5 ba460e7b) under sampling. Turns 2-3 differ: the XMX
  prefill changes the arithmetic (PPL above: not worse).
- **The chunk default's trade.** Turns 1-2 of the gate's chunk-1024 leg and of the final defaults wrote the same 61-
  and 79-token replies, so their decode compares directly: 14.0 / 14.2 vs 13.4 / 13.6 tok/s, about **4 % slower
  decode at 2048**. Card 1's expert cache drops from 65 to 60 slots per layer. In exchange, a 1.4K-token agent turn
  starts 2.1 s sooner and a 9.6K cold prompt 13 s sooner. At 14 tok/s, 4 % of decode costs about 0.3 s per 100
  tokens, so 2048 wins on turns whose replies are shorter than about 700 tokens. `--prefill-chunk 1024` gives the
  decode back.
- vs the old binary: cold TTFT ÷6.3, follow-up turns ÷25-29, decode ×1.2-1.25.

Exactness on the final defaults (`final_exact.log`, 19:10-19:19; greedy, `IE_G5_CPU_MISS=0`, 2 turns × 48 tokens):

- cache restore: re-sending turn 2 restored 10,989 of 10,991 tokens, answered in 0.58 s, same reply (444df4bf): **PASS**
- pipelined == serial prefill (`IE_G5_NO_PIPELINE=1`), both turns: **PASS**
- plain decode (`IE_G5_PIPEDRAFT=0`) == the default MTP draft, both turns: **PASS**

`final_serve.sh`'s own exactness leg (E_final) hit a harness bug, not an engine one. Its 48-token greedy turn 1 was cut
off inside a tool call, which streamed as raw `<tool_call>` text in `content`. The client stripped only `tool_calls`,
so the server refused the history ("incomplete assistant tool history"). `agent_turns.py` now also cuts the raw call
text. The broken leg's logs are kept as `E_final_harnessbug.*`.

## What is left: profile at the new defaults (19:20, `pp_prof2048.log`)

Pinned runner 8da42b6a, chunk 2048 × 3 chunks, `IE_QUEUE_PROFILING=1` (GPU buckets trustworthy, wall inflated),
both stages:

| kernel | GPU time | calls | note |
|---|---|---|---|
| `gemm_fp16` | 10.6 s | 31,986 | the XMX path: experts >= 64 rows + dense Q8 projections, ~0.33 ms per call |
| `gemv_q4k_rows` | 4.7 s | 82,500 | experts below 64 rows |
| `g5_attend_sel` | 3.9 s | 1,056 | DSA sparse attention |
| `gemv_q5k_rows` | 2.4 s | 41,250 | experts below 64 rows, Q5_K |
| `g5_kabsorb` + `g5_vabsorb` | 4.2 s | 33 + 33 | 64 ms per call, ~0.5 TFLOP/s |
| `dequant_q4k_bt` + `dequant_q5k_bt` | 2.5 s | 28,702 | the dequant ahead of every expert XMX GEMM |
| `gemv_q8_soa_f16_T` | 0.7 s | 306 | was 8.1 s at chunk 1024 before the Q8 switch (dense projections below 32 rows remain) |

Ranked next levers:

1. **MLA absorbs on XMX.** About 13% of GPU time. `IE_G5_ABSORB_XMX_ROWS`: rounds 4-5 below, default ON after the
   20:05 gate.
2. **A fused dequant-in-GEMM for Q4_K/Q5_K experts.** It would remove the 2.5 s of dequant passes and move the rows/XMX
   crossover below 64 rows (the 4.7 + 2.4 s of rows GEMV). This is a new kernel.
3. **`g5_attend_sel`.** 3.9 s. Not yet studied.

### MLA absorbs on XMX: default ON since the 20:05 gate (rounds 4-5, 19:20-19:37)

`IE_G5_ABSORB_XMX_ROWS=N` (default **64**, 0 = off; it was opt-in through round 5): a prefill call of at least N rows (clamped above the spec-verify window of 8)
runs the k and v absorbs in `absorb_xmx` instead of the per-output-element kernels. Per MLA layer and chunk, the
weight is transposed into the Q8 gemm scratch and the input is packed per head. Each head then runs `gemm_fp16` into
`mmscr32_`, and the result is scattered back. The packed input borrows buffers that are dead at that point
(`mla_o16_` / `mla_att_` for k, `mla_qlat_` for v), so no new VRAM is used. If the switch can't run, a stderr line
says so. A read-only review found no defects (layouts, capacity, in-order lifetimes, T == 1 exclusion, range sizes).

| variant (chunk 2048, pinned runner, 10,240 tokens) | prefill | PPL (wikitext, ctx 8192) |
|---|---|---|
| defaults (round 4; reproduces round 3 to the digit) | 278.0 tok/s | **1.8097** |
| absorbs, input rounded to fp16, one GEMM per head | 308.1 (+10.8 %) | 1.8118 |
| absorbs, input split fp16 hi + lo, two GEMMs per head (kept in the code) | 303.4 (+9.1 %) | 1.8113 |

The PPL runs are deterministic: the defaults gave 1.8097 twice. So +0.0016 is a real numerics change, not noise.
Splitting the input recovers only a quarter of it. The rest is the change of summation order, which this model
amplifies: rows → XMX for the experts and dense projections moved PPL by −0.0021, the same size in the other
direction. **Perturbation band (19:39-19:44, `ppl_ab.log`).** Moving only the expert rows/XMX threshold reroutes some experts
between two paths that both accumulate in fp32. With every other setting at the defaults:

| `IE_G5_XMX_ROWS` | 48 | **64 (default)** | 96 |
|---|---|---|---|
| PPL | 1.8117 | 1.8097 | 1.8115 |

Benign arithmetic changes move this PPL by about 0.002, and the default's 1.8097 sits at the low edge of that band.
The absorbs' 1.8113 (+0.0016) is inside it. Round 3's 1.8118 → 1.8097 for the XMX switches is inside it too, which is
why it is reported as "not worse", not "better".

**Gate with the absorbs ON by default (20:05-20:21, `gate_absorb.log`;
ie a5a5caf3, runner 11ba4bc1, no GLM env or flags).** A stderr line confirms the switch on both stages in every leg.

| check | absorbs off (19:00-19:19) | **absorbs on by default** |
|---|---|---|
| runner ppbench, chunk 2048 × 5 | 278.0 tok/s | **306.8 tok/s** (+10.4 %) |
| PPL (wikitext, ctx 8192) | 1.8097 | 1.8113 (= the opt-in run; inside the band) |
| serve: cold 9.6K TTFT | 38.7 s (251 tok/s) | **36.6 s (265 tok/s)** |
| serve: turns 2 / 3 TTFT | 10.8 / 10.7 s | **10.0 / 9.9 s** (1.4K new tokens at 141-145 tok/s) |
| serve: decode | 13.4 / 13.6 / 12.3 | 13.7 / 13.9 / 12.9 (turns 1-2 the same replies, ba460e7b / a7d80ca3) |
| cache restore, greedy | PASS | **PASS** (10,989 of 10,991 restored, 0.57 s, 444df4bf) |
| pipelined == serial prefill | PASS | **PASS** |
| plain decode == MTP draft | PASS | **PASS** |

The greedy exactness replies are byte-identical to the absorbs-off build's (42aae9a9, 444df4bf).

## 2026-10-20: the expert GEMMs (rounds 6-9; branch p4-glm-serve after c46dada)

An agent turn prefills ~1.4K new tokens in one chunk, which gives an expert ~39 rows on average. At the 64-row cutoff
most experts therefore take the rows GEMV. Each card's GPU is busy 93 % of its window (`pp_prof2048.log`: 16.7 of
17.9 s), so prefill is bound by kernel time; the expert streaming overlaps it.

**Cold clocks fooled the first microbench.** `moe_rows_vs_xmx_bench.txt` (19:xx yesterday) timed the dequant at
0.272 ms. Warm, the same kernel takes 0.036-0.048 ms (`results/glm53-2026-10-20/moe_rows_vs_xmx_warm.txt`). A
bench must warm the B70 first, as `dequant_bt_bench` and `gemm_splitk_bench` now do.

Negative results before split-K (bit-identical kernels; patch and benches in `results/glm53-2026-10-20/discarded/`;
the tiled dequant came back in round 10, below):

| change | kernel, warm | end to end (pinned runner) |
|---|---|---|
| tiled SLM-staged Q4_K/Q5_K → Bᵀ dequant | x1.36-1.39 Q4_K, x1.11-1.14 Q5_K | 303.1 vs 306.8 (2048 × 5), 236.0 vs 233.6 (1408 × 3) |
| + rows GEMV with the per-row loop unrolled behind a guard | x1.1-1.2 at 16-64 rows | 297.0 / 230.4 |
| the 64-row cutoff moved (24 / 32 / 48 / 96) without split-K | — | flat: 223.5-236.0 (1408), 298.4-308.6 (2048) |

A single run varies by ±2-3 %, so these gains are below what one run can resolve.

**The real limit: the small-M GEMM ran ~32 work-groups.** At ≤ ~300 rows, every `gemm_fp16` tile variant puts about
32 work-groups on the 32 Xe cores, and each walks K=4096 in 256 single-buffered steps. Warm, a 16-48-row gate/up
GEMM takes 0.20 ms flat (~1.3 TFLOP/s). `gemm_fp16_splitk` (src/ops/gemm_fp16.cpp) runs four K-quarter groups of
sub-groups inside each work-group with 16-byte loads and sums their partials in SLM in a fixed order
(`gemm_splitk_bench.txt`):

| rows | gate/up K 4096 N 2048 | down K 2048 N 4096 |
|---|---|---|
| 16-48 | 0.20 → 0.04 ms (x5.0-5.3) | 0.107 → 0.041 ms (x2.5-2.7) |
| 64 | x3.5 | x1.8 |
| 96-192 | x1.9-2.5 | x1.1-1.3 |
| 256 | x1.5 | x0.88 (so: gemm_fp16 from 256 rows) |

Its error against an fp64 reference is 3-4x smaller than gemm_fp16's (2.4e-6 → 0.6e-6), because the accumulation
chains are shorter. GLM's XMX experts use it below 256 rows (`IE_G5_SPLITK`, default on; 0 = off).

Round 9 (`pp_r9.log`; the control is split-K off):

| cutoff `IE_G5_XMX_ROWS` | 1408 × 3 | 2048 × 5 |
|---|---|---|
| 64, split-K off (control) | 231.1 | (306.8 committed) |
| 64 | 239.0 | 323.8 |
| **32** | 241.9 | **330.1** |
| 16 | 242.1 | 328.8 |

The new default cutoff is 32.

**Gate (`gate_splitk.log`, 12:21-12:38, ie 5628c075 / runner 870b8d76, no GLM env or flags):**

| check | committed c46dada | split-K, cutoff 32 |
|---|---|---|
| runner, chunk 2048 × 5 | 306.8 tok/s | **331.5** (+8.1 %) |
| runner, chunk 1408 × 3 | 233.6 | **244.3** (+4.6 %) |
| serve: cold 9.6K TTFT | 36.6 s (265 tok/s) | **32.4 s (300 tok/s)** |
| serve: turns 2 / 3 TTFT | 10.0 / 9.9 s | **9.7 / 9.5 s** (1.3-1.4K new at 146-152 tok/s) |
| serve: decode | 13.7 / 13.9 / 12.9 | 14.0 / 13.5 / 14.3 (different sampled replies) |
| cache restore / pipelined == serial / plain == draft | PASS | **PASS** (greedy replies byte-identical, 42aae9a9 / 444df4bf) |

PPL: at ctx 8192 the three split-K samples (cutoff 32 / 48 / 64: 1.8121 / 1.8142 / 1.8126) sat ~0.0017 above the
split-K-off samples (1.8097-1.8118). That is about 2σ of those few samples, so it got a lower-noise check: twice the
text (ctx 16384, 16,383 predictions), split-K off vs on at cutoff 32: **1.9469 vs 1.9461**. Split-K is no worse
(`ppl_ab_skband.log`, `ppl16k_x32_sk0.log` / `ppl16k_x32_sk1.log`), and at the kernel level it is more accurate. Committed (d50907d).

**Round 10 (12:54-13:16): the tiled dequant again, on top of split-K.** Split-K made the expert GEMMs smaller, so the
dequant before them is a larger share. Warm, the tiled kernel takes a GLM expert matrix from 0.041 to 0.030 ms (Q4_K)
and 0.036 to 0.032 ms (Q5_K), 0 differing halves (`results/glm53-2026-10-20/dequant_bt_bench.txt`, now with a 500 ms
warmup). Interleaved against runner 870b8d76 (d50907d), same session (`pp_r10_*.log`):

| | d50907d, rep a / b | + tiled dequant, rep a / b | delta |
|---|---|---|---|
| chunk 2048 × 5 | 331.7 / 331.9 | 335.4 / 335.2 | **+1.1 %** |
| chunk 1408 × 3 | 245.1 / 245.1 | 246.4 / 246.3 | **+0.5 %** |

Interleaved reps in one session agree to 0.1 %, so this small gain can be measured; the ±2-3 % noise is between
sessions. PPL 1.8121 (NLL 0.59447), the split-K gate's to the last digit. `ie serve` greedy (`round10_exact.log`):
replies 42aae9a9 / 444df4bf, byte-identical to the split-K gate's, restore included. Kept.

## What is left: profile at 6ce76b3 (13:17, `pp_prof2048_6ce76b3.log`)

Chunk 2048 × 3, `IE_QUEUE_PROFILING=1`. The GPU-busy buckets are reliable. Wall time and the busy fraction are not,
because profiling adds host time per submit and this build submits more kernels. Instrumented GPU time fell from
32.7 s to 23.3 s for the same 6,144 tokens (absorbs on XMX, split-K, tiled dequant). Per card: busy 11.7 of a
15.0 s window (was 16.7 of 17.9).

| bucket | 10-19 defaults | now | share now |
|---|---|---|---|
| gemm_fp16 (dense, absorbs, experts ≥ 256 rows) | 10.6 s | 4.87 s | 21 % |
| gemm_fp16_splitk (experts 32-255 rows) | — | 4.46 s | 19 % |
| g5_attend_sel (DSA sparse attention) | 3.93 s | 3.89 s | 17 % |
| dequant Q4_K + Q5_K → Bᵀ | 2.54 s | 3.32 s (more experts on XMX) | 14 % |
| rows GEMV Q4_K + Q5_K (experts < 32 rows) | 7.10 s | 2.41 s | 10 % |
| g5_kabsorb + g5_vabsorb (sub-group kernels) | 4.17 s | gone (XMX) | — |

Next targets, by expected gain:

1. **Dequantize inside the split-K GEMM.** Load the B tile from the Q4_K / Q5_K blocks straight into SLM. That
   removes the 3.3 s of dequant and the round trip of the fp16 Bᵀ through VRAM, about 14 % of GPU time.
2. **g5_attend_sel**, 3.7 ms per call. It is the largest non-GEMM kernel and it grows with context.
3. **The busy fraction.** A clean wall-time run (no profiling) per stage would show whether the cards now wait on
   expert streaming. The fraction fell from 93 % to 78 % here, but profiling overhead also grew with the kernel
   count, so this run cannot tell.

## Shipped on branch p4-glm-serve (c46dada, d50907d, 6ce76b3; not installed)

- Prompt cache for GLM (one conversation). Per stage, the KDA scan + conv state and the DSA open-pool rolls are
  saved at the stable conversation depth to pinned host RAM; the latent KV and pool keys stay in place on the GPU.
  `--no-prompt-cache` turns it off.
- Two-card pipelined prefill. `IE_G5_NO_PIPELINE=1` = serial.
- MTP pipelined draft, ON by default on two cards. `IE_G5_PIPEDRAFT=0` = off.
- GLM prefill chunk default 2048 when `--prefill-chunk` is not given, and at most 2048. `IE_G5_PREFILL_CHUNK`
  overrides; `ie capabilities` reports the default.
- Prefill MoE experts with >= 64 rows and dense Q8_0 projections with >= 32 rows on XMX (`IE_G5_XMX_ROWS`,
  `IE_G5_Q8_GEMM_ROWS`; 0 = off). PPL gate passed.
- Prefill MLA absorbs of >= 64 rows on XMX, fp16 hi + lo input (`IE_G5_ABSORB_XMX_ROWS`; 0 = off). Gated 20:21.
- 2026-10-20: the XMX experts' GEMMs split K below 256 rows (`IE_G5_SPLITK`; 0 = off), and the expert rows/XMX
  cutoff is 32 (was 64). Gated 12:38.
- 2026-10-20: tiled Q4_K / Q5_K → Bᵀ dequant, bit-identical (+1.1 % prefill at chunk 2048). Round 10.
- Both arch names accepted.

The Codex adversarial review of this delta (2026-10-19) found four issues; three are fixed:

1. `vocab_size` was still read under `glm5next.`, so a renamed file skipped the vocab check. Fixed, with a test
   under both names.
2. With a chunk below 8, the pipelined forward width was clamped to 8 while the serial path used the given chunk.
   Fixed: the width is now the chunk itself; buffers stay sized max(chunk, 8).
3. A load that failed after the draft set `IE_G5_MTP` / `IE_G5_MTP_STAGE` left them set for the process. Fixed:
   the process's values are put back on every exit from the GLM load. `IE_G5_MTP` set from outside is admitted
   only on a two-card draft load.
4. **Not fixed: `IE_G5_QSTAR_A=0.5`, which the draft load sets, is process-wide.** Glm5NextModel reads q* into
   function statics at its first MoE call, so a second load in the same process inherits it. `ie serve` loads
   once per process. Tools that load twice need per-instance routing parameters, a refactor of the q* statics.
