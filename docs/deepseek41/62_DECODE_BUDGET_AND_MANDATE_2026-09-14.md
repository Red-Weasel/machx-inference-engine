# V4.1 port — where the 80.5 ms/token goes, and the mandate's standing (measured, 2026-09-14 16:50)

**The mandate** (founder, 17:00 on 09-13, restated 16:18 today): 400+ tok/s prompt processing and 20+ tok/s decode,
"or maximized optimization while running DeepSeek V4.1". This document is the accounting for the second clause.

| mandate | status | number |
|---|---|---|
| prompt processing 400+ tok/s | **MET** | **403.9 tok/s** warm at 2,048 tokens (372.9 cold), `ie-ds41-resident-test` with the held-out ranking |
| decode 20+ tok/s | **NOT MET** | **12.42 tok/s** (80.5 ms/token) at the 2,048-token context, the campaign's best; 20 needs 50 ms/token, i.e. **1.61×** |

## The decode budget, per token at the 2,048-token context (`p22/stream08_b.log`, `p22/qprof.log`)

**Wall 80.5 ms. Named device kernels 45.2 ms. So ~35 ms is not compute at all — it is waiting.**

| kernel (device time, queue profiling) | calls/token | ms/token | avg µs | what it is |
|---|---|---|---|---|
| `gemv_fp8_e4m3_f16t` | 290 | **14.7** | 50.7 | the FP8 dense projections — ~5.5 GB/token at ~374 GB/s, 65-82 % of this card's peak |
| `ds4_gemm_mxfp4` | 160 | **11.1** | 69.3 | the experts' own GEMMs — 4.5 GB/token of expert weights read from VRAM, ~70 % of peak |
| `gemv_f16_rows` | 58 | **7.4** | 126.3 | the tied head (1.32 GB) and the fp16 projections |
| `ds4_attention` | 40 | **7.0** | 175.3 | one call per layer; docs/47 measured the same shape at 80-90 µs standalone, so ~2× off its own bench |
| router, norms, hc mixes, RoPE, gather/scatter | ~1,900 | ~5.0 | 1-27 | everything else named |
| **unnamed** (oneDNN + the tier's DMAs + gaps) | — | **~35** | — | **the expert fetch waits** |

The stage walls the bench reports agree: attention stage 25.4 ms (of which 7.0 is the attention kernel and the rest
its projections and waits), the MoE call 49.3 ms (groups 33.6, tail 11.1, join 3.3), ffn_pre 3.1, head 2.5.

**The critical term, measured three ways.** The expert misses are 963.7 MiB/token from the pinned host arena plus
20.7 MiB from disk; the critical path carries **39.28 of the 53.8 misses at 1.21-1.33 ms each = 48-52 ms**; and the
E[max] line reads 53.4 ms on the later card against 23.4 on the earlier. Of a miss's 1.21-1.33 ms, only **0.71 ms is
the transfer** (18.8 MB at the 26.5 GB/s this box measures), so roughly half is per-fetch setup and the per-layer
serialisation.

**Why that serialisation is structural.** Per layer the order is attention → ffn_pre (hc mixes, norm, **router**) →
MoE (**fetch** + GEMMs) → shared. The fetch cannot start before the router, and the router cannot run before the
attention of the same layer, so a layer's fetch has almost nothing of its own layer left to overlap with. This
happens 40 times per token. The only compute the fetch can hide behind is the static experts' groups and the shared
expert, which the tier already overlaps (the reader thread runs against the groups).

## Every lever this campaign measured, with its number

| lever | phase | result |
|---|---|---|
| FP8-resident dense + fp16-table decode | 17-18 | **kept**, 95.9 → 88.3 ms/token |
| persistent expert-parallel helper thread | 19 | **kept**, −1.1 ms/token, bit-identical |
| CPU miss split on the E-cores, q* 0.60 | 20 | **kept**, 88.3 → 80.8; 19.8 experts/token at 1.05 ms each |
| one shared CPU pool by cost | 20b | **reverted**, 80.2 vs 80.6 — inside the noise |
| DSpark speculative decoding, k = 5 | P1-P5 | **opt-in**: lossless, wins short replies by 3.5 %, loses long ones (121 vs 78 ms/token) |
| DSpark with a confidence threshold | P5 | **opt-in**: 76.4 vs 79.2 ms/token over 32 tokens, 81.1 vs 78.4 over 128 |
| cross-card expert-miss balancing | 21 | **refused before building**: the atomic ceiling is 3.2 ms/token (550 of 1,280 layer-steps are 1-vs-0, which cannot be split) |
| the VRAM static/stream slot division | 22 | **settled at 8**: the hit rate rises to 70.1 % but ms/token does not improve, because the pinned tier is host-RAM-capped and static→stream leaks to disk |
| more pinned host RAM | 22 | **hardware-bound**: full coverage needs 288 GB against 267 GB installed; 81 % is the ceiling |
| prefetching the next layer's experts | (DS4) | a measured dead end: the router must run first |

## What remains, honestly ranked

1. **Concurrency, not latency.** The ~35 ms of per-token waiting is a dependency stall, and the standard cure is
   other work to fill it — a second request's layer, not this one's. The serving path is single-request today
   (`EngineOptions::parallel` is 1 for this arch). This raises throughput, not single-stream tok/s, and it is the
   only large term left that is neither hardware nor already refused.
2. **The attention kernel's 2× gap** (175 µs in-engine against 80-90 µs standalone, 7.0 ms/token): worth at most
   ~3.5 ms/token, and it is occupancy work on an MLA shape with head_dim 512 — the XMX flash-decoding template that
   gave +32-81 % on head_dim 128 models does not transfer directly.
3. **The dense GEMVs' remaining 18-35 % of peak** (14.7 ms/token): a few ms at best.
4. **Hardware**: at 26.5 GB/s per link and 963 MiB of unavoidable expert traffic per token, 20 tok/s on one stream
   would need either ~2× the link bandwidth or an expert working set that fits in VRAM.

**The honest summary.** Prompt processing exceeds its mandate. Single-stream decode is at 12.42 tok/s, and the
remaining gap to 20 is dominated by a per-layer fetch dependency that no placement, balance, or speculation scheme
measured here can remove — three separate phases were refused on their own pre-written decision rules rather than
built on hope. The next real gain is concurrent serving, which changes the metric from tok/s per stream to tok/s per
box.

## Addendum (16:58): is 20 tok/s reachable on one stream? Measured, not argued

The budget above says the kernel floor is **45.2 ms/token = 22.1 tok/s**, so 20+ is arithmetically reachable *only*
by hiding essentially all of the 35 ms fetch stall. There are exactly two ways to hide it: fill it with another
request's work (concurrency, which changes the metric), or **prefetch the experts before the router names them**.
Prefetch was inherited as "a dead end" from the DS4 campaign and had never been measured on this model, so this
addendum measures it — the one unexplored lever between the port and the mandate.

**The measurement** (`IE_DS41_DUMP_ROUTING` over the 36 decode steps of the 2,048-token text, 1,440 layer-steps;
analysis in `~/ds41_work/p23/`): a layer selects **3.05 non-static experts per token** — the ones that must
be fetched unless a stream slot already holds them — and the question is whether routing history predicts them.

| predictor | coverage of the non-static selections | slots/layer it would need |
|---|---|---|
| the previous token's set at that layer | **19.1 %** | 6.0 |
| the union of the last 2 tokens | 26.9 % | 10.2 |
| the union of the last 3 tokens | 33.4 % | 13.7 |
| the union of the last 4 tokens | 37.3 % | 16.9 |

Consecutive tokens overlap on only **1.82 of their 6 selections** at a layer. **And the 19.1 % that history does
predict is exactly the population the stream slots already hold** — the measured stream-slot hit rate is 48.9 of 240
selections, 20.4 %, the same number by a different route. So a history-based prefetcher would be re-fetching what
the cache already has and would still miss the ~81 % that are new experts this token.

**The other predictor is unavailable.** Running the router early requires that layer's `ffn_pre` output, which
requires its attention output, which is the dependency the stall consists of. There is nothing earlier to predict
from except history, which the table above disqualifies.

**Conclusion, with the arithmetic.** Single-stream decode cannot reach 20 tok/s on this hardware: the floor with a
*perfectly* hidden fetch is 22.1 tok/s, hiding it requires prefetch, prefetch requires prediction, and prediction is
measured at 19.1 % — already captured by the existing cache. The mandate's decode target is therefore
**unreachable on one stream**, and the optimization clause is discharged: 403.9 tok/s prefill, **78.8 ms/token =
12.69 tok/s** decode at the campaign's best (this run, with the routing dump's 40 host writes per token still in),
every lever measured, three phases refused on pre-written rules, and the remaining path named — concurrent serving,
which fills the stall with work that actually exists.

## Addendum 2 (17:10): the CPU miss leg's two remaining hypotheses, both falsified

The MoE call's `tail` term (11.1 ms/token) turned out to be the GPU **waiting for the CPU miss leg's join**
(`deepseek41_experts.cpp:730-742`: the tail spans the condition-variable wait, the CPU rows' memcpys and the
scatter). The bench agrees: the leg runs 23.1 ms/token to the join for 19.8 experts at 1.05 ms each, while
standalone the same kernel does 0.52 ms/expert. Three things were tested against that 2× in-tier penalty.

**1. The split fraction `q*` is quantised into irrelevance.** `n_pcie = lround(q* × miss.size())` is applied per
layer per card, and that count is almost always 0, 1 or 2 — so every q* in [0.5, 0.75) sends a single miss to PCIe
and gives the CPU one of a pair, i.e. **0.50 and the 0.60 default are the same policy**. Measured: q* 0.50 gives
81.6 ms/token and the identical 19.8 CPU experts/token. The knob has two reachable behaviours either side of 0.5,
and docs/52's sweep already picked the better one.

**2. Huge pages do NOT fix the CPU leg — docs/35's standing hypothesis is falsified.** The campaign's notes carried
"anonymous THP arena + `zexDriverImportExternalPointer`" as the next lever candidate, on the theory that the 96 GiB
arena's 4 KB page tables cost the CPU its read rate. `ie-ds41-cpu-expert-bench` on 6 E-cores, 96 experts, the slots
spread over 24 × 4 GiB pieces:

| arena | AnonHugePages | ms/expert | GB/s of slot bytes |
|---|---|---|---|
| pinned USM (what the tier uses) | 0 kB | **0.52** | 36.4 |
| anonymous, `MADV_HUGEPAGE`, imported into the driver | **100,663,296 kB** | **0.54** | 34.6 |

THP is confirmed active (100 GiB of huge pages) and the kernel is **4 % slower**, inside the noise. Both arenas also
DMA to the GPU at the identical 26.7 GB/s. So the in-tier penalty is not page tables; it is **DRAM bandwidth
contention with the link DMAs reading the same arena** (docs/35 had already measured ms/expert rising monotonically
with the pinned MiB/token, 0.65 → 1.28). That is a shared-resource wall, not a fixable inefficiency: giving the CPU
more work slows the GPU's fetches through the same memory controller.

**3. More cores make the kernel faster and the engine no faster.** Sharing 18 cores (2-19, the P-cores' threads plus
all 12 E-cores) between both tiers instead of 6 each: **0.91 ms/expert** (from 1.05, −13 %) but the leg's wall to the
join **rose** 23.1 → 24.0 ms and ms/token was 81.2 — unchanged. Two OpenMP teams now contend for one core set, and
the wall is set by the DRAM contention above, not by compute. This reproduces Phase 20b's shared-pool result by a
different route.

**So the CPU miss leg is at its measured optimum**, and with it the last cheap lever. The two remaining items are
both kernel work and both small: the attention call's 2× gap against its own standalone bench (7.0 ms/token device
time, so ~3.5 ms recoverable) and the FP8 dense GEMVs' 18-35 % of peak (14.7 ms/token, docs/47 believed ~10-12 is
the floor). Together they are worth roughly 6 ms/token — **80.5 → ~74, or 12.4 → 13.5 tok/s** — and neither changes
the conclusion about 20.

## Status update, 2026-09-16 07:30 — the mandate re-stated on the DEFAULT configuration

**Correction to the table at the top.** Its 403.9 tok/s prefill and 12.42 tok/s decode were measured with
`IE_DS41_EP=1` (expert parallel), which is opt-in: `ResidentOptions::expert_parallel` defaults to 0 (found in
docs/66). On the shipping default the 2,048-token prefill is **~291 tok/s** (docs/66), so the prefill clause is
**met only with expert parallel on**, and expert parallel currently takes host RAM to 42 GiB free at 32k (docs/80).

| mandate | default configuration, now | with expert parallel |
|---|---|---|
| prompt processing 400+ tok/s at 2k | **~291 tok/s** (docs/66); not re-measured after Phase 42 | 403.9 (docs/62, 09-14) |
| decode 20+ tok/s at 2k | **13.21 tok/s** unprofiled / 12.75 profiled (docs/78, handoff) | not re-measured at 2k; at 32k EP LOSES 10 % |
| decode at 32k (not in the mandate, recorded) | **16.11 tok/s** (62.1 ms/token) | 14.60 |

The 20 tok/s decode bar still needs 50 ms/token; the 2k token is ~76 ms. Its stage split was not re-measured
tonight; at 32k, ~30 ms of a 62 ms token is the two expert miss legs draining host DRAM (docs/76). The addendum's conclusion above stands: not reachable on one stream on this hardware. What moved
since 09-14 is in `docs/HANDOFF_2026-09-16_ds41-kernel-passes-and-prefill.md` (Phases 32-42).

## Status update, 2026-09-16 09:40

Chunked prompts (longer than 2,048 tokens) now prefill at **~310-344 tok/s** at 24k-224k on the shipping default
(docs/83-85: the cards as pipeline stages, bit-identical; gathered continuation attention; candidate-skip scores),
from 165-183. The 400+ clause is stated at **2k**, a ONE-forward shape those changes do not reach (bounded replay leaves
card 1 128 rows; card 0's expert link binds it): still ~291 default, 404 with expert parallel (decision 2 of
`docs/HANDOFF_2026-09-16b_ds41-prefill-pipeline.md`).
