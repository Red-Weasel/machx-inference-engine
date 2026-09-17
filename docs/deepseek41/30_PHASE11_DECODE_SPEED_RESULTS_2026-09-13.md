# V4.1 port — Phase 11 results: decode speed

**Criteria:** `29_PHASE11_DECODE_SPEED_CRITERIA_2026-09-13.md` (amended 13:10 from the peer
session's pre-read, before any measurement). **Test:** `ie-ds41-decode-test` (the Phase 9 criteria
unchanged, then the Phase 11 measurement section). **Build before:** 8468b69 + the step-1
instrumentation. Two cards, held-out placement, 6 GiB reserve, replay ON.

## Step 1 — the decode step measured, BEFORE (13:38, `scratchpad/p11/decode_before.log`)

Every intra-layer host wait was still in place, so the columns bound their stages and sum.

| | after the 12-token golden prompt | after the replayed 2048-token pp text |
|---|---|---|
| steps 4-35, ms/token (tok/s) | **590.2 (1.69)**, min 453, max 742 | **344.6 (2.90)**, min 278, max 419 |
| step 0-3 total | 475 / 447 / 462 / 519 | 370 / 297 / 321 / 314 |
| host prep (hashes, gather) | 0.5-0.9 | 0.7-13.5 (engram rows faulted in from the 189 GB table) |
| engram (layers 1, 14) | 1.2-2.0 | 1.2 |
| attention (hc site + projections + attention + o) | 79-83 | 84-89 |
| ffn-pre (hc site + norm + router) | 54-55 | 55-57 |
| MoE call | 295-370 | 139-207 |
| — groups (pinned fetch + GEMMs) | 66-101 | 62-110 |
| — join (the mmap reader: disk) | 209-295 | 57-95 |
| — mmap group + tail + prep + spawn | ~7 | ~3 |
| shared expert | 8.1 | 7.5 |
| head | 2.3 | 2.4 |
| setup (per card: rope tables, scratch) | 3.4-4.5 | 3.9-5.2 |
| experts static / pinned / mmap per token | 18.0 / 149.1 / 72.9 | 104.7 / 119.8 / 15.6 |
| stream-slot hits per token | 80.8 | 42.8 |
| **decode hit rate** (static + stream hits) / 240 | **41.2%** | **61.5%** |
| pinned MiB per token (actual H2D) | 1224 | 1379 |
| disk MiB per token | 1308 | 279 |
| pinned bytes over the groups' wall | 22.4 GB/s | 22.2 GB/s |

Launches per layer (traced run, `launches_before.txt`, step 0 after the golden prompt): median
**54 kernels + 11 copies per layer** (min 55, max 104 enqueues at layers 0 and 20, which carry
their card's setup), 4 host drains per layer by the adapter's `releaseSubmittedKernels` count,
2,707 enqueues per step. The report's Reuse-mode layer is 11 kernels.

**What the numbers say.** The dense body (attention + ffn-pre + shared) costs ~148 ms per token
= 3.7 ms per layer. Its fp16 weights are ~324 MB per layer (q_a 1280×5120, q_b 32768×1280,
kv 512×5120, o_a 8192×4096, o_b 5120×8192, shared 3×2304×5120), which at the card's ~500 GB/s
is 0.65 ms per layer, ~29 ms per token with the head. So ~120 ms per token is launch, wait and
latency overhead, not bandwidth — the stop rule of step 3 does NOT trigger. The MoE at the
2048-token context is 140-200 ms: the pinned fetch (1.38 GiB at 22 GB/s = 62 ms of pure
transfer) plus the disk reads of the mmap experts (15.6 per token, ~6 ms each on the reader's
leg) plus the per-group drains.

**The bound as measured (criterion 4), at the 2048-token context:** 1379 MiB per token from the
pinned tier at 22.2 GB/s = **65 ms per token = 15.4 tok/s** if nothing else cost anything; the
mmap tier's 279 MiB per token at its measured ~4 GB/s effective (one 17.9 MiB read per
missing expert, serial on the reader's leg) adds ~70 ms → **~7.4 tok/s** at this residency.
Above that needs more static slots (the FP8-resident dense set, docs/26), the mmap experts
read in parallel slices, or a batched verify (DSpark). The prefill's 45% (docs/22) was the wrong
number for decode: the stream slots' temporal locality gives 41% on the short prompt and 61.5%
at the 2048-token context.

## Step 2/3 — the dense body at M = 1

### What was built

1. **V4's decode GEMV toolkit at T = 1 — built, measured, REVERTED.** Every fp16 projection of
   the dense body through `ds4_decode_gemv_f16` / `_grouped_f16` / `_multi` (fp32 activation in,
   no cast launches, a site's projections of one activation in one launch; the GEMVs ran at
   560-580 GB/s, launches per layer 54 → 48). Isolated with a kill switch on the same build
   (`decode_iso_*.log`, waits in place, the 2048-token context): the oneDNN route 298.6 ms/token
   vs the GEMV route 304.4 with the original mixes kernel; 188.6 vs 202.0 with the new one. The
   route buys nothing and reads slightly slower on both pairs, and it was the change that carried
   the phase's numerics risk (below). Out, under the same stop rule as the waits; the oneDNN
   projections at M = 1 are as fast as a purpose-built GEMV here because the per-call cost they
   were meant to save (~1-4 µs each) is not where the step's time is.
2. **The intra-layer host waits gated off at decode — measured, then REVERTED.** Waits off vs on:
   278.1 vs 277.2 ms/token on the interim build, 191.4 vs 198.1 on the final one, both within
   run noise (min/max spreads of ±60 ms). The host was never the bound, and a barrier removal
   that buys nothing leaves an ordering argument to maintain (V4's H1 was the same result and
   was reverted the same way). Every wait is back where it was; the stage columns always sum.
3. **`ds41_hc_mixes` at T = 1** — the finding of this phase. The kernel gives each of the 24 mix
   rows to ONE lane that walks all 20,480 elements serially, in one work-group for the whole
   GPU: **1,240 µs per call, two calls per layer, 99 ms per token = 68% of every named kernel's
   device time** before this phase (`decode_after_kprof_hcoff.log`). A prefill's thousands of
   work-groups hide the same shape completely, which is why nothing in Phases 2-10 saw it. A
   first rewrite spread each row over the work-group's 256 lanes and still measured 692 µs — one
   work-group cannot pull 2 MB of mix weights at any useful rate. The shipped shape: 200
   work-groups each reduce one row's chunk (plus the RMS row), and a one-work-group finish sums
   the chunks and runs the same tail. T = 1 only: the prefill's summation order is untouched.
4. **The mmap tier's reads sliced at decode**: a layer has one or two mmap experts at decode, so
   one reader thread walked each 17.9 MiB expert alone (~6 ms on the join leg per expert) while
   seven readers idled. A batch smaller than the reader count is now split into slices (8 for one
   expert, 4 for two) and every thread reads and permutes its part; the tier test checks 8 and
   3 slices reproduce the whole-plane pack byte for byte. Isolated (`decode_iso_old.log`: the old
   numerics on the final build, same routing and bytes as the pre-phase run): 344.6 → 298.6 ms
   per token at the 2048-token context, **46 ms from the slices alone**.

### Measured, in order (the 2048-token context; the short prompt in parentheses)

| build (ms/token and tok/s from untraced runs; device ms from `IE_QUEUE_PROFILING` runs) | ms/token | tok/s | named device ms/token | of which hc_mixes |
|---|---|---|---|---|
| pre-phase: 8468b69 + the step-1 instrumentation (`decode_before.log`) | 344.6 (590.2) | 2.90 (1.69) | not measured: the `ie::ps` names on the V4.1 ops came with step 2's build, and the oneDNN projections are unnamed anyway | — |
| interim build, mixes decode shape OFF (`IE_DS41_HCMIX_T1=0`): the GEMV route + the ORIGINAL mixes kernel (`decode_after_kprof_hcoff.log`) | 333.5 (576.7) | 3.00 (1.73) | 151.1 (146.7) | **99.3** (1,240 µs per call) |
| interim build: the GEMV route + the one-work-group mixes rewrite (`decode_after.log`, `_kprof.log`) | 278.1 (453.7) | 3.60 (2.20) | 107.4 (102.7) | 55.4 (692 µs) |
| GEMV route + two-launch mixes + sliced mmap reads, waits off (`decode_after2*.log`) | 191.4 (387.1) | 5.22 (2.58) | 52.0 (48.6) | 1.05 (13 µs per pair) |
| same, waits restored (`decode_iso_final.log`) | 202.0 (406.0) | 4.95 (2.46) | — | — |
| oneDNN route + two-launch mixes + sliced reads, waits restored (`decode_iso_mixes.log`) | **188.6** (291.5†) | **5.30** (3.43†) | — | — |
| **SHIPPED** (the row above, rebuilt without the GEMV code; `decode_ship*.log`) | **194.9** (289.3†) | **5.13** (3.46†) | 52.0 (48.5), a different set: the oneDNN projections are named in this column | **1.07** (6.4 + 7.0 µs per pair) |

† the short prompt's greedy continuation changed with the mixes' summation order (43 mmap experts
per token instead of 73), so its ms/token is a different sequence and is not compared.

The named-kernel SET is identical in the three GEMV-build profiled columns (the same build
lineage, the same `ie::ps` names; only the mixes kernel's implementation differs), so their totals
are comparable with each other; the pre-phase row has no comparable column. The SHIPPED build's
column covers a DIFFERENT set — its projections are oneDNN kernels, named for the profiler by
pushing their events (`ds41.onednn.*`) where the GEMV build had `ds41.dec.*` — so its 52.0 is
compared with nothing above (that it equals the GEMV build's 52.0 is the point below, not a
coincidence to lean on). 99.3 / 146.7 = 68%.

**Why the GEMV route had nothing to win, in one number each** (`decode_ship_kprof.log`, the
2048-token context, per call): oneDNN q_b 145 µs for 84 MB = **578 GB/s**, o_b 143 µs (84 MB,
586 GB/s), o_a 118 µs (67 MB, 568 GB/s), shared gate + up 103 µs (47 MB), shared down 43 µs,
head 2.22 ms (1.32 GB, 596 GB/s) — against the GEMV route's 148 / 150 / 117 / 107 / 44 µs and
2.29 ms on the same shapes. oneDNN's M = 1 kernels were already at the card's ~590 GB/s
streaming rate; the cast launches it saved were ~4 µs each and the multi-launch's saving ~1.4 µs
per projection, and the isolation says the route was net slower — where the ~13 ms went is not
resolved by the kernel table (its device time is equal) and was not pursued once the decision
was made. Shipped kernel table, 2048-token context, 32 steps: expert GEMMs (`ds4_gemm_mxfp4`)
384 ms = 12.0 ms/token, attention 235 ms = 7.3 (183 µs per call), the twelve oneDNN projections
+ head 875 ms = 27.3, router 1.6, norms 0.9, mixes 1.07, everything else under 1. The named
total is 52.0 ms of a 194.9 ms step; the unnamed rest is the expert DMAs, the disk reads and the
host. (The per-card "busy" figure excludes the oneDNN events, which carry no queue: 12-15% is a
floor for the GPU's share of its own window, not the number.)

**A superseded attribution, recorded because it is the third of its kind this campaign:** the
earlier "the GEMV route buys ~11 ms" (pre-phase 344.6 → the GEMV build with the original mixes
333.5) was a cross-build comparison confounded by the sliced mmap reads riding in the same
build. The kill-switch isolation on ONE build (`decode_iso_*.log`) has the sliced reads at 46 ms
on their own and the GEMV route COSTING 5.8 ms with the original mixes and 13-14 ms with the
two-launch mixes. The 19.4% of Phase 10 and the Q8-route kill-switch story of 2026-09-11 were the
same error class; the fix is the same each time — one build, switches, nothing quoted across
builds.

**2048-token context: 344.6 → 194.9 ms per token, 2.90 → 5.13 tok/s (1.77x)** on the shipped
build (`decode_ship.log`, untraced, steps 4-35; the gate's independent run of the same binary:
194.3 ms = 5.15 tok/s, a same-binary spread of 0.6 ms — the 188.6 of the isolation run was a
DIFFERENT binary, the GEMV code compiled in and switched off, and is not a noise sample). **The
bytes and the routing are identical before and after** — 1378.9 MiB pinned / 279.0 disk /
61.5% hit before, 1378.3 / 275.1 / 61.6% after — so the 150 ms removed is work, not an easier
token; the step-1 rows account for it completely: attention site 85.2 → 33.3 and ffn-pre
55.3 → 3.9 (the mixes kernel, 103 ms), join 57.5 → 29.2 (the sliced reads, 28 ms), the groups
column unmoved (79.5 → 80.0: the fetch untouched) — 131 of the 132 ms of step 1. The 12-token
prompt's greedy continuation is a different sequence on the new summation order (43 mmap
experts per token instead of 73), so its 590.2 → 289.3 ms is not a like-for-like figure and is
not claimed. The step at the 2048-token context (steps 0-3): 165-208 ms = MoE 110-154 (groups
64-109: the pinned fetch at 22 GB/s; join 26-46: the mmap experts' disk reads, from 57-95 before
the slices) + attention site 33 + ffn-pre 4 + shared 7.5 + setup 4-6 + head 2.4 + prep 0.6-7.5.
Bytes per token unchanged by construction: 1378 MiB pinned, 275 MiB disk, decode hit rate 61.6%.

### The breakdown that sums (criterion 2; `decode_ship.log`, steps 0-3)

| ms per token, 2048-token context | before | shipped | short prompt: before → shipped (different sequence, see †) |
|---|---|---|---|
| host prep | 0.7-13.5 | 0.6-7.5 | 0.5-0.9 → 0.5-3.1 |
| engram (layers 1, 14) | 1.2 | 1.2-1.3 | 1.2-2.0 → 1.2-1.4 |
| attention site (hc mixes + projections + attention + o) | 84-89 | **32.7-33.8** | 79-83 → 29.2-32.0 |
| ffn-pre (hc mixes + norm + router) | 55-57 | **3.8-3.9** | 54-55 → 4.2-5.1 |
| MoE call | 139-207 | 110-154 | 295-370 → 229-287 |
| — groups (pinned fetch + GEMMs) | 62-110 | 64-109 | 66-101 → 66-106 |
| — join (mmap reader: disk) | 57-95 | **26-46** | 209-295 → 138-208 |
| shared expert | 7.5 | 7.3-7.6 | 8.1 → 8-13 |
| setup | 3.9-5.2 | 4.0-5.6 | 3.4-4.5 → 3.6-4.1 |
| head | 2.4 | 2.4 | 2.3 → 2.4 |
| **total, step 0-3** | 297-370 | **165-208** | 447-519 → 279-342 |

Each row sums to its total (step 1 at 2048: 0.6 + 1.3 + 33.3 + 3.9 + 111.8 + 7.4 + 4.3 + 2.4 =
165.0). Source: `decode_ship.log`; the waits are the original ones, so every run's columns sum.

### Launches per layer (criterion 5; traced runs, step 0 after the golden prompt)

Before: median 54 kernels + 11 copies per layer (2,707 enqueues per step, `launches_before.txt`).
Shipped: median **56 kernels** + 12 copies (2787 enqueues per step,
2317 kernels, `launches_ship.txt`; min/max 57/106 enqueues, layers 0 and 20 carry their card's
setup). The two-launch mixes add two per layer; nothing was removed. (The reverted GEMV route's
48 was the multi-projection launches, and is gone with it.) The report's 11 for a Reuse-mode
layer is a fusion target this phase did not attempt (docs/26's Mega-mHC item); at ~20 µs of
launch gap each, the ~56 are worth at most the dense body's ~10 ms over its bandwidth floor
(the stop rule of step 3).

### Correctness (criterion 1, the same bars as docs/25, before → shipped)

| check | before (run 4) | after (final) | bar |
|---|---|---|---|
| state after the prefill, layers 0 / 2 / 20 | unchanged (the prefill path is untouched) | unchanged | 3e-3 / 3e-3 / 1.2e-2 |
| forced step 0, all 40 layers, worst card 0 / card 1 | 9.45e-3 / 1.59e-2 | **9.36e-3 / 1.49e-2** | 1.2e-2 / 3.5e-2 |
| forced step 0, engram out at 1 / 14 | 4.03e-4 / 2.68e-3 | 4.04e-4 / 2.55e-3 | 1.2e-2 |
| forced step 0, logits vs the golden | 7.51e-3 | 7.74e-3 | 5e-2 |
| own router, judged to the first flip (layer 10), worst | 7.72e-3 | 7.54e-3 | 1.2e-2 |
| step 0 token / top-5 | 16; set equal, order 16 603 339 14170 1 | 16; set equal, order 16 603 14170 339 1 (two neighbours swapped) | token + set |
| greedy steps 1-3 (margins 2.95 / 0.35 / 4.18) | 16 455 6102 | 16 455 6102 | tokens |
| hashes (sequence check) | exact | exact | exact |
| negative control (RoPE off by one) | 40 layers over, worst 0.825 | 40 over, worst 0.803 | must FAIL |

(The GEMV route's digits, for the record: forced worst 1.02e-2 / 1.48e-2 on card 0 / card 1 —
the move toward card 0's bar that the isolation above explains, and that is no longer shipped.)

## The bound, restated from the final measurement (criterion 4), and what selects the next phase

At the 2048-token context the step is 195 ms: the pinned fetch is 65 ms of it (1378 MiB at
22.1 GB/s), the mmap disk reads 26-46, the expert GEMMs ~12, the dense body ~45 ms (its fp16
weights are ~29 ms at bandwidth), setup ~5, head 2.4. The commit message's "fetch-bound" is
too strong (gate 11 finding 4): the fetch is a third of the step, and the disk reads and the
dense body together exceed it — three comparable terms, and the cheapest win is not necessarily
the first of them. Three levers, in the order of bytes:

1. **More static slots** (docs/26's FP8-resident dense set: ~7 GiB back per two cards = ~380
   more expert slots per card). The held-out profile's cumulative selection mass decides how
   much of the 120 pinned experts per token that buys; the profile file has the answer without a
   run. Every expert that moves from pinned to static saves 17.9 MiB of PCIe per selection.
2. **Fewer bytes per accepted token: a batched verify** (DSpark, the report's §2.4.3, or a
   prompt-lookup drafter with no weights): k drafted tokens verified in one forward fetch the
   experts once for k rows. Needs T > 1 at pos0 > 0 (today refused) and the M > 1 decode kernels
   (V4's grouped GEMM already handles them). This is the lever that beats the fetch bound
   instead of moving it.
3. **The mmap tier**: 15.6 experts per token from disk at the 2048-token context, 73 on the short
   prompt. Pinning more of the arena (RAM: 54 GiB was free during the runs) or a better profile
   for short prompts. The sliced reads halved the join; the rest is the NVMe.

The dense body's remaining ~40 ms is ~50 kernels per layer at ~20 µs of launch gap each; the
Mega-mHC-style fusion (docs/26) is the fourth lever and worth at most that.

## Files

What 0111bf1 carries (gate 11 finding 1 corrected an earlier draft of this list that named
reverted symbols): `src/ops/deepseek41_ops.cpp` (the two-launch decode shape of
`ds41_hc_mixes`, the ops submitted through `ie::ps`, the `IE_DS41_HCMIX_T1=0` kill switch),
`include/ie/deepseek41_upload.hpp` (the `scratch` parameter, `ds41_hc_mixes_scratch_floats`),
`src/model/deepseek41_forward.cpp` (the `hcs` scratch passed to the mixes at decode, the inline
decode kernels named through `ie::ps`, the oneDNN projections named through `pe()` — the
projections themselves are the pre-phase oneDNN calls; the two new stats fields copied from
the tier), `include/ie/deepseek41_forward.hpp` (`bytes_pinned`, `experts_stream_hit`),
`src/model/deepseek41_experts.cpp` + `include/ie/deepseek41_experts.hpp` (the cache-counter
bytes and stream hits, the sliced `ds41_slot_pack_pread`), `tools/ds41_decode_test.cpp` (the
Phase 11 section, `IE_DS41_BENCH_ONLY`, `IE_DS41_LAUNCH_MARKS`, `IE_QUEUE_PROFILING`,
`IE_DS41_DUMP_DIR`), `tools/ds41_tier_test.cpp` (the sliced pack cross-check),
`tools/ds41_reference/ur_launch_count.py`. Not in the tree: `ds4_decode_gemv` routes,
`IE_DS41_DECGEMV`, `set_stage_sync`, `IE_DS41_STAGE_SYNC` (built, measured, reverted — above).
Logs in `scratchpad/p11/`: the shipped build's `decode_ship{,_kprof,_traced}.log`,
`launches_ship.txt`, `ship_md5.txt`; the isolation's `decode_iso_*.log`, `isolation.txt`; the
pre-phase `decode_before.log`, `launches_before.txt`; the reverted builds' `decode_after*.log`;
the gate's own runs in `scratchpad/gate11/`.

### Gate 11 (the peer session, 15:40): PASS

All six criteria re-derived from the raw logs and re-run: the shipped md5s reproduced from a
clean build of 0111bf1; every decode digit bit-identical (the decode path is deterministic run
to run); 194.3 ms/token on its own run; the breakdown sums checked by hand on both builds; the
launch delta exactly 80 kernels = 2 per layer with copies and host waits unchanged, which
independently confirms the wait revert; unload 268.4 / 148.4 MiB residue; prefill unchanged
(pp2048 278.6 / 282.1). Findings, all taken in the commit after the gate: the Files list above
(F1), the spread sentence (F2), a 16-byte-block guard in the sliced read (F3: the slices are
16-byte aligned by construction and a plane that was not would lose its tail silently), the
"fetch-bound" framing (F4), and the in-order-queue note on the mixes API (F5, verified correct
as written).
