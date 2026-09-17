# V4.1 — Phase 43: chunked prefill with the two cards as pipeline stages (criteria before the result)

## The measurement that set it (handoff item 1, "the DMA/compute overlap")

`IE_DS41_STAGES=1` on the long test (8,192 tokens in 2,048-token chunks, ctx 33,792, int-dot route, warmed expert
file; `ds41_work/p24/p43_stages_8k.log`), one continuation chunk at NC 4-8k:

| term per chunk | ms |
|---|---|
| wall | 11,262 |
| layers: attention | 2,235 |
| layers: MoE call | 8,289 = groups 8,137 + mmap group 131 + tail 16 |
| .. mmap fill (reader thread, concurrent with the groups) | 3,605 |
| experts served | static 2,113 + pinned 8,846 (stream hits **0**) + mmap 1,355 = ~309 of 384 per layer |
| **pinned -> VRAM bytes** | **154.9 GiB** |
| **mmap -> VRAM bytes** | **23.7 GiB** |

An expert is 17.9 MiB. A 2,048-token chunk occupies nearly every expert of every layer, and with 8 stream slots per
layer nothing persists between chunks, so **~179 GiB of expert weights cross PCIe per chunk** -- ~7.5 s at a
24 GB/s link, more than all the chunk's compute (the fetch pipeline can only overlap group g+1's bytes with group
g's GEMMs). Prefill is link-bound, not kernel-bound. And the forward runs card 0's layers, then card 1's: **each
card's link is idle while the other card works**.

## The change

`Ds41Forward::forward_pipelined(ids, chunks, logits)`: card 0 runs chunk k+1 on the calling thread while card 1
runs chunk k on its own thread, fed through a one-slot mailbox carrying exactly forward()'s host boundary state
(the hc stream, the ffn_pre mix, the row count / offset, the bounced source caches). `forward()` becomes
`forward_impl(..., stage)`: a stage runs its card range only; a later stage skips the host prep (engram, embed --
refused if an engram layer sits on it), never resets or reads `all_ids_`, and only the last stage commits `n_pos`.
Shared caches the kernels use are per device and mutex-guarded (`ds4_hc_scratch`, `ds4_w16/a16_scratch`) or
thread-local per queue (oneDNN). Opt-in: `IE_DS41_PIPE_PREFILL=1` (generator and long test). Refused with expert
parallel, the drafter capture, or one card.

## Criteria

1. **Bit-identical**: the long test's prefill logits (FNV over the fp32 bytes) and its 16 greedy decode ids equal
   the sequential run's on the same build (8k and 32k); the 24k held-out `ie-ds41-run` text identical to the
   sequential stream. Each card issues the same launches in the same order, so anything else is a defect.
2. **Faster**: 32k prefill rate above the sequential 165 tok/s (int-dot) by >= 25 % (two links at once, minus the
   host-DRAM contention of two DMA streams plus two reader pools -- expected well short of 2x).
3. **Safe**: host MemAvailable does not fall below its sequential level by more than the carries (~0.3 GiB); no new
   pinned allocation; no device error across the gate runs.
4. **Nothing else moves**: decode, multi, rollback and dspark paths untouched (forward() is the one-stage call);
   the decode test and multi test re-run on the build.

## Results (build of 07:47, `ds41_work/p43/`)

**Criterion 1 -- bit-identical: MET.**

| run, same build | sequential (`IE_DS41_PIPE_PREFILL=0`) | pipelined |
|---|---|---|
| long test 8k: prefill logits FNV / 16 decode ids FNV | `28671defa195b5d5` / `a99cb39e2a4232db` | identical |
| long test 32k: prefill logits FNV / 16 decode ids FNV | `148e9b3e94d32d84` / `32d61289ca003384` | identical |
| 24k held-out `ie-ds41-run`, 120 greedy tokens | 629 bytes, needle retrieved | **byte-identical**, needle retrieved |

**Criterion 2 -- faster: MET (+68 % at 32k against a +25 % bar).**

| | sequential | pipelined | |
|---|---|---|---|
| long test 8k prefill (4 chunks) | 187.0 tok/s | **269.6** | +44 % |
| long test 32k prefill (16 chunks) | 165.1 tok/s (198.5 s) | **277.9** (117.9 s) | **+68 %** |
| 24k held-out text prefill (12 chunks) | 183 tok/s (132.1 s) | **298** (81.3 s) | **+63 %** |
| decode after it (not on this path) | 13.91 (32k, 16 tok) / 10.57 (24k text) | 14.35 / 10.21 | noise, single short runs |

**Criterion 3 -- safe: MET.** Minimum host MemAvailable during the 32k runs: 41 GiB pipelined, 42 GiB sequential
(the pinned arena's normal floor); no device errors in any run.

**Criterion 4 -- nothing else moves: MET.** On the same build: decode test 28/28, multi 38/38, rollback 74/74,
dspark 153/153, forward 5/5, cont 15/15.

**Verdict: DEFAULT ON** in the generator (`ie-ds41-run`, and so every generator user) and the long test;
`IE_DS41_PIPE_PREFILL=0` restores the serial loop. The long test keeps the serial walk when `IE_QUEUE_PROFILING` or
`IE_DS41_STAGES` is set, since those attribute time per card.

## After the gate (build of 08:10): two follow-ups measured

**Continuation chunks no longer keep their scratch in VRAM.** The persistent per-step-length scratch lists
(Phase 18) now serve decode-sized steps only (T <= 8); a continuation chunk allocates and frees its own. Same bytes,
same arithmetic: the 32k pipelined prefill logits FNV is `148e9b3e94d32d84` again (`c2048_pipe.log`).

**Where a pipelined chunk's time goes, per card (`IE_DS41_PIPE_TRACE=1`, 32k long test):**

| chunk | stage 0 (card 0) busy | stage 1 (card 1) busy | card 1 waiting for input | prefill |
|---|---|---|---|---|
| 2,048 x 16 | 106.3 s (6.6 s/chunk) | 108.9 s (6.8 s/chunk) | 9.0 s | **277.9 tok/s** |
| 4,096 x 8 (`IE_DS41_MAX_FWD` via the test's chunk) | 102.0 s (12.7 s/chunk) | 89.0 s | 25.7 s | 285.8 tok/s (+3 %) |

**A larger chunk is NOT a lever at 32k**: per-token card time falls only 4 % from 2,048 to 4,096, so the per-chunk
expert bytes (which barely grow with T) are not what binds a card here -- its kernels are (Phase 42's profile: 10.3 s
of named kernels in a 12.8 s serial chunk, the O(NC) attention 3.2 s and the int-dot expert GEMM 4.8 s of it). The
stages are balanced at 2,048 (6.6 vs 6.8 s), and their busy sum (215 s) exceeds the serial wall (198.5 s) by 8 %:
the contention cost of two cards streaming and computing at once. At 8k, where the kernels are smaller, the chunk
was link-bound instead (the stage table at the top). Chunk 4,096 also lowered the run's MemAvailable floor to
36 GiB (41 at 2,048). The decode ids after either chunking are the same 64 tokens.

**So the next prefill terms are the two kernels**, per card now: the expert GEMM (the XMX route, decision 1, whose
perplexity gate this phase's `ie-ds41-ppl` exists to run) and `ds4_attention_xmx`'s O(NC) block walk.

## Handoff decision 1, measured: perplexity of the XMX prefill expert route (`ie-ds41-ppl`, 08:32-08:36)

16,384 tokens of wikitext-2 test (raw, no template), chunks of 1,024 through the exact path (bounded replay off,
full logits), 16,383 predictions, the held-out ranking, pairwise on the same tokens (`ppl_intdot.bin`,
`ppl_xmx.bin`):

| route | PPL | mean NLL |
|---|---|---|
| int-dot W4A8 (the default) | **1.88567** | 0.634286 |
| XMX W4A16 (`IE_DS41_PREFILL_XMX=1`) | **1.88988** (+0.22 %) | 0.636515 |
| paired difference, XMX - int-dot | **+0.00223 nats/token, block SE 0.00123 over 16 blocks (t = 1.81)** | top-1 agreement 98.44 % |

**No difference detectable at this sample (t = 1.81 < 2)**, with the point estimate slightly WORSE for the XMX
route -- unlike V4, where the same route measured better than int-dot. This is the measurement decision 1 asked
for; the default is unchanged (the founder's call): +8-14 % prefill against a quality delta indistinguishable from
zero here and a W4A16 route whose routing follows fp16 numerics at near-ties.

**Re-measured on the Phase 45 stack (09:35, `xmx32k.log`):** 32k pipelined prefill 333.5 tok/s with the XMX route
against 307.7-318.4 default (+5-8 %), 64 decode ids identical to the default's.
