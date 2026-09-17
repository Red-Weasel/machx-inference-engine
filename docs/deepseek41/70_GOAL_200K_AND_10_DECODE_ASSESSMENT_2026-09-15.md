# V4.1 — the founder's goal "200k ctx and 10+ decode", assessed against measurement

## Verdict

| bar | verdict | evidence |
|---|---|---|
| **200,000-token context** | **MET** | **223,237 real tokens** prefilled in 2,200 s (101 tok/s) through the generator's own chunking, and the needle planted at **50 % depth was retrieved** from a HELD-OUT document. Synthetically, 262,144 tokens prefill. |
| **10+ tok/s decode** | **MET to ~24k, NOT MET at 200k** | 24,193 tokens, real text, held-out ranking: **10.15 tok/s**. 223,237 tokens, same ranking: **4.31 tok/s** (232 ms/token against the 100 ms that 10 tok/s needs). |

The ranking work still pays at length — 3.16 -> 4.31 tok/s at 223k, 1.36x — but it does not close a 132 ms gap.

## Why 10 tok/s at 200k is not reachable with the levers this campaign has identified

Itemised from measured per-kernel and per-stage numbers, scaled on each term's measured NC slope:

| term | ~ms/token at 200k | status |
|---|---|---|
| `ds4_indexer_topk` | ~30 | exactly linear in NC (4.00x measured on NC x4); a designed fix exists (select within the 16,384-entry candidate pool at the 4 consumer layers, ~16x on ~62 % of it) |
| MoE `tail` | ~35 | = CPU leg minus groups; the leg is the critical path |
| attention pick-scan residual | ~21 | each slice scans all `n_picks` for its own ~`n_picks / SP`; bucketing is a further ~32x |
| attention base, MoE groups, dense GEMVs | ~60 | NOT NC-linear; this is the short-context cost |
| **measured total** | **232** | |

**Removing every NC-linear term above still lands near 147 ms/token = 6.8 tok/s.** So the gap is not closeable by
the identified kernel work, and saying otherwise would be false.

## What actually binds, measured on REAL text (and the synthetic benches hid it)

`IE_DS41_STAGES=1` on 24,193 tokens of real text with the held-out ranking, last decode step:

```
layers 127.5 = attn 32.0 + ffn_pre 3.1 + moe 91.2 + shared 0.4
moe 91.2 = groups 25.5 + join 55.4 + mmap_group 1.0 + tail 8.1   | cpu leg 74.6 span, 36.3 compute
experts static 136 pinned 51 mmap 11 cpu 42 of 240, stream hits 31 -> hit rate 69.6% | pinned 358.6 MiB, disk 197.2 MiB
```

**`join` -- the wait for expert fills -- is the largest single term, and disk traffic is 197.2 MiB/token.** On the
synthetic prompt with a synthetic ranking the same term read **6.5 MiB/token**, which is precisely why the
synthetic benchmark flattered every residency change: a repeated block routes to a narrow expert set that fits the
fast tiers, and real text does not.

**And that is a HARDWARE bound, by this engine's own arithmetic.** The runtime allocates **47 static + 266 pinned
= 313 of the 384 experts a card owns per layer = 81.5 %**, which is exactly the ceiling docs/61 derived
independently (full coverage needs 288 GB of pinned host RAM against 267 GB installed). The remaining ~18.5 % is
disk-resident by construction, real-text routing reaches it, and the fills are what `join` waits on. Raising
`IE_DS41_PIN_CAP_GIB` toward full coverage would need ~229 GiB of the 249 GiB total, leaving ~20 GiB for the
operating system -- and a pinned-bank RAM livelock has already frozen this box once (2026-08-27). **That is the
founder's call, not a change to make, and the arithmetic says it would not reach 10 tok/s at 200k anyway.**

## What this campaign did deliver against the goal

* **Long context works and is CORRECT**: 262,144 tokens prefill; 223,237 real tokens with retrieval at 50 % depth
  from a held-out document; three hard walls removed (context-scaled scratch, the 16,384-latent indexer refusal,
  single-call prefill) each behind a bit-identity or reference-equality gate.
* **Decode at long context improved 2.4x-2.7x** by fixing residency, with the mechanism confirmed on four
  independent counters and the ranking rule established and corrected: valid only for the (phase, length,
  WORKLOAD) profiled, distribution dominating length by ~3x.
* **10+ tok/s is real at ~24k on real text, held out** (10.15), against 5.28 for the shipped ranking.
* Two changes **removed for failing their own pre-written bars**, and two of my own conclusions **retracted after
  measurement** (the dense-mask theory, and the synthetic ranking).

## The honest next step, if the goal is to be pursued

The binding term is expert residency against host RAM, not kernels. The three NC-linear kernel items are worth
~85 ms and are the only work with a designed fix. Beyond them the options are hardware (more host RAM, or a
faster path to the expert file) or a change of metric (concurrent serving raises tok/s per box, not per stream --
docs/62's standing conclusion). **A ranking profiled on the founder's actual traffic is the one cheap thing left
that could move real-workload decode further, and it is now one command: `--profile-out`.**

## Addendum: the page cache IS a large free win, and two more hypotheses died on measurement

**1. Warming the expert file into page cache — free, and the biggest single lever found.** The mmap tier's whole
backing store, `ie_experts_tail.ieslot`, is only **30.8 GiB**, and a run leaves ~48 GiB of page cache after the
186.8 GiB pinned arena. `cat`-ing the file takes **5.8 s** at 5.3 GB/s. Same real-text workload, same ranking,
last decode step:

| term | cold cache | warmed |
|---|---|---|
| layers | 127.5 | **80.6** |
| MoE call | 91.2 | **44.6** |
| .. **join** (waiting for expert fills) | **55.4** | **21.9** |
| decode hit rate | 69.6 % | **89.6 %** |
| **disk MiB/token** | **197.2** | **71.7** |

**-46 ms of MoE for a five-second file read.** The 300-step average moves less (9.09 -> 9.52 tok/s) because the
early steps are cold and the cache decays under the pinned arena's pressure during a run, which is itself worth
knowing: **the page-cache state is a hidden variable in every tier measurement in this campaign**, and it explains
run-to-run MoE swings of 51.6 vs 104.6 ms on identical configurations.

**2. The radix top-k's sub-group rotation is NOT its cost — falsified.** `ds4_indexer_topk` is exactly linear in
n_keys (4.00x on n_keys x4) and worth ~26-34 ms/token at 200k-262k, so its atomic-free histogram rotation (kSG
sequential steps and kSG sub-group barriers per WG keys, 8 passes deep) was the obvious suspect. A 4-bit,
per-lane-row variant removing the rotation **entirely** measured **4.31 ms/token against 4.26** at ctx 32,768 —
*nothing*, because 16 passes double the key reads by as much as the rotation saved. It was bit-identical (28/28,
every value equal) and it was **removed rather than kept behind a flag**. The kernel is **memory-latency bound
with ONE work-group per query row** at decode; the lever is more work-groups over the key range (a two-level
histogram), which is a larger change than this campaign has evidence to justify.

**3. The attention ballot, earlier in this phase** — also bit-identical, also removed, also for failing its bar.

## Where this leaves the goal, with the arithmetic

At 223k, warmed, the estimate is ~186 ms/token. The remaining fixable terms and their honest sizes:

| term | ~ms at 200k | fix | status |
|---|---|---|---|
| `ds4_indexer_topk` | ~29 | multi-work-group histogram | not built; rotation fix FALSIFIED |
| attention kernel pick-scan | ~10-20 | bucket picks by slice | designed, not built |
| MoE / tier | ~70 warmed | more pinned host RAM | **hardware**: 81.5 % coverage is the ceiling |
| not NC-linear (attn base, groups, GEMVs) | ~60 | — | this is the floor |

**Even building both unbuilt kernel fixes lands near 136 ms/token = 7.3 tok/s at 223k.** So **10+ tok/s at 200k is
not reachable on this box**, and the wall is the expert working set against 267 GB of host RAM — the same wall
docs/61 derived and docs/62 reached from the other direction for 20 tok/s at 2k.

**The founder's decision, since there is no defensible default:** accept 10+ at conversational lengths (~24k, where
it is measured at 9.5-10.2 tok/s on real text, held out) and 4.3-5.8 tok/s at 200k+; or add host RAM so the expert
working set fits; or change the metric to throughput per box (concurrent serving), which docs/62 already names as
the only large term left. **Warming the expert file should be done regardless — it is free and worth ~46 ms.**

## FINAL: measured in the best available configuration, and the mechanism identified

`ie-ds41-run` on the **held-out** document at **223,237 tokens**, with the page cache warmed AND a real-text
decode ranking profiled at 214,594 tokens — every lever this campaign found, applied at once:

```
223237 prompt tokens, prefill 2155.54 s = 104 tok/s | 300 generated in 66.80 s = 4.49 tok/s
layers 346.3 = attn 59.1 + ffn_pre 3.3 + moe 282.6 + shared 0.5
moe 282.6 = groups 36.8 + join 179.6 + mmap_group 2.1 + tail 61.0 | cpu leg 245.2 span, 161.1 compute
experts static 80 pinned 75 mmap 17 cpu 68 of 240, stream hits 36 -> hit rate 48.3% | pinned 699.3 MiB, disk 304.8 MiB
```

**4.49 tok/s**, against 4.31 with a cold cache and a 23k ranking. The levers that were worth 2.4x at 24k are worth
4 % here, and the reason is now measured rather than guessed.

**My "the MoE cannot scale with NC" argument was WRONG, and here is what actually happens.** A token does make 240
expert selections at any context — but *which* experts changes:

| | ctx 41k, real text, warm | ctx 224k, real text, warm | |
|---|---|---|---|
| **selections served from static VRAM** | **207 / 240** | **80 / 240** | |
| static slots available per layer per card | 52 | **47** | only -10 % |
| decode hit rate | 89.6 % | **48.3 %** | |
| expert bytes fetched per token | 126 MiB | **1,004 MiB** | **8x** |
| CPU leg per expert | 1.1 ms | **3.6 ms** | the DRAM contention docs/62 measured |
| MoE call | 44.6 ms | **282.6 ms** | |

**VRAM displacement is ruled out by the slot counts themselves** — 52 against 47, a 10 % change that cannot produce
a 2.6x change in static hits. What collapses is **ROUTING CONCENTRATION**: the top 47 experts of 384 per layer
capture **86 %** of a token's selections at 41k and only **33 %** at 224k. A token at position 223k integrates a
far longer and more varied history, so the router spreads its choices much more widely. This is consistent with
every other measurement in this campaign (docs/62's prefetch study found consecutive tokens share only 1.82 of 6
selections even at 2k) and it is why the earlier control was right: reserving KV capacity costs 2.8 %, because the
allocation was never the problem.

**So the wall is arithmetic, and it is hardware.** At 224k the engine must move **~1 GB of expert weights per
token**. The links carry 26.5 GB/s, which is **38 ms/token at perfect efficiency** and 180 ms measured once the
CPU miss leg contends for the same DRAM. To reach 100 ms/token the traffic would have to fall to roughly 250
MiB/token, i.e. a static hit rate near 85 % at 224k — and the static tier is 47 of 384 experts per layer, sized by
32 GiB of VRAM per card against a 475 GiB checkpoint. **No scheduling, ranking, kernel or cache change reaches
that**; it needs either VRAM enough to hold a diffuse working set or a different expert-caching architecture.

**Verdict, final:** `200k ctx` is **MET** (223,237 real tokens, needle retrieved at 50 % depth from a held-out
document). `10+ decode` is **MET to ~41k** (9.5-10.2 tok/s on real text, held out) and **NOT MET at 200k**
(4.49 tok/s), and it is **not reachable on this hardware**. The two unbuilt kernel fixes (`indexer_topk`
multi-work-group, attention pick bucketing) are worth ~40 ms of a 246 ms gap — they would take 223k from 4.49 to
about 5.4 tok/s, which is worth having and is not the goal.

**What the founder can decide between, with numbers:**
1. **Accept the measured shape**: 10+ tok/s to ~41k, 4.5 at 223k. Warm the expert file (free, 46 ms) and profile a
   ranking on real traffic (one command, 2.4x at conversational lengths).
2. **More VRAM per card** — the only change that addresses the binding term, because the static tier is what
   captures routing and it is VRAM-sized.
3. **Concurrent serving** — raises tok/s per box rather than per stream, which docs/62 already identified as the
   one large term left; it does not make a single 200k stream faster.

## Addendum 2: the CPU split at long context (FALSIFIED), and the number that finally decomposes the goal

**The CPU miss split is still a win at 224k.** docs/62 measured the split's per-fetch penalty as DRAM contention
with the link DMAs, and at 224k the CPU leg runs 245 ms of span and 161 ms of compute against the same memory
controller — so turning it off should have let every DMA reach link speed. Measured, same held-out document, same
ranking, warm cache: **4.26 tok/s with the split OFF against 4.49 ON.** Slightly WORSE. The contention does not
invert at this scale; the split is earning its keep here as it did at 2,048 tokens.

**But that run produced the number this whole phase needed.** Its last decode step happened to land on a highly
concentrated routing pattern:

```
layers 128.8 = attn 63.1 + ffn_pre 3.3 + moe 61.0 + shared 0.6
moe 61.0 = groups 26.1 + join 33.8 + mmap_group 0.3 + tail 0.3
experts static 208 pinned 30 mmap 2 cpu 0 of 240, stream hits 14 -> hit rate 92.5% | pinned 286.9 MiB, disk 35.9 MiB
```

**A 92.5 % hit rate at a 224,000-token context gives 128.8 ms of layer time = 7.4 tok/s, NOT 10.** Residency
essentially solved, and the target is still missed — because **attention is 63.1 ms** at that context. (The two
runs' last steps are not comparable with each other: the CPU and GPU expert paths are not bit-identical, so by
step 300 the two runs are generating different text with different routing concentration. Only the 300-step
average compares, and it says the split wins.)

So the goal decomposes, finally, into two terms of known size:

| term at 224k | measured | needed for 100 ms/token |
|---|---|---|
| MoE at a 92.5 % hit rate | 61 | ~50 (its PCIe floor is 26) |
| **attention** | **60-63** | **~30** |
| .. `ds4_indexer_topk` (exactly linear in NC) | ~30 | **~5 — BUILDABLE** |
| .. the attention kernel itself | ~30 | ~25 |
| everything else | ~8 | ~8 |
| **total** | **~129 = 7.4 tok/s** | **~100 = 10 tok/s** |

**One of the two is software and one is hardware.** `ds4_indexer_topk` is latency-bound with ONE work-group per
query row — proved by removing its sub-group rotation entirely for no change (4.31 vs 4.26 ms) — so splitting the
key range across work-groups with a two-level histogram is worth ~25 ms of the ~29 ms gap. A sustained 90 % hit
rate is the other half, and docs/70 prices it at **1.9x the static tier's VRAM** (89 slots/layer against 47).

**Every lever measured at long context, with its value:**

| lever | at 223-224k |
|---|---|
| warm the expert file into page cache | small on average, large on the concentrated tail |
| a matched real-text decode ranking | 4.31 -> 4.49 tok/s |
| the CPU miss split OFF | 4.49 -> 4.26 (**worse**) |
| the attention kernel 44 % faster (pick partition) | ~0 on the wall (hidden behind the DMA wait) |
| non-uniform static slots across layers | 0.1 % |
| the radix top-k's rotation removed | 0 |
| heads packed per work-group | 0 |
| extending the expert file over the whole mmap range | ~36 ms, NOT YET BUILT |
| `indexer_topk` across work-groups | ~25 ms, NOT YET BUILT |

## Status update, 2026-09-16 07:30 — what Phases 32-42 did to the goal

| bar | 2026-09-15 | 2026-09-16 | note |
|---|---|---|---|
| 200k context | MET (223,237 tokens, needle retrieved) | MET, unchanged | prefill not re-run at 223k |
| 10+ decode at ~24k | 10.15 tok/s | **10.26 tok/s** (24,193 held-out tokens, `p24/p39_text.log`) | met, thin margin |
| 10+ decode at 32k | 11.43 (profiled, `p24/bal32k_1.log`) | **16.11 tok/s** unprofiled (`p24/p39_32k.log`) | met |
| 10+ decode at 223k | 4.49 tok/s | **not re-measured** | see below |

The two "NOT YET BUILT" rows of the lever table above: **`indexer_topk` across work-groups is BUILT** (Phase 39,
`b92b928`, bit-identical; a 114,688-key row 1,226 -> 191 us per call in the unit check, ~8 ms/token over 8 index
layers at T = 1 by that arithmetic, against this doc's ~25 ms estimate). **Extending the expert file over the whole
mmap range is still unbuilt.** The decode attention is now XMX gathered flash-decoding (Phase 34), context-independent
at ~2 ms/token, so it no longer grows with NC. None of this touches the binding term at 223k -- routing
concentration against VRAM, ~1 GB of expert bytes per token -- so the verdict above stands: 10+ at 223k is a
hardware question. Re-measuring 223k decode with Phases 34-39 in is item 3 of the handoff's next-work list; the
expected result is a few ms/token better than 4.49 tok/s, not a different verdict (estimate, not measured).

## Status update, 2026-09-16 09:40 — Phases 43-45 and the 223k re-measurement

| bar | 2026-09-16 07:30 | 2026-09-16 09:30 | log |
|---|---|---|---|
| 200k context | MET (prefill 104 tok/s, 2,155 s) | MET, **prefill 319 tok/s (700 s)**, needle retrieved | `ds41_work/p43/r223k_b.log` |
| 10+ decode at 223k | 4.49 (not re-measured) | **4.71 tok/s** -- NOT MET | same |

The re-measurement found a defect first: Phase 39's split top-k aborted every decode-shaped call past 131,072 keys
(fixed, docs/85). Decode at 223k is still bound by expert residency (last step: moe 177 of 215 ms, hit rate 70.8 %),
so the verdict above stands. Prefill at 224k now runs at the 32k rate: the card pipeline (docs/83), the gathered
continuation attention (docs/84) and the candidate-skip scores (docs/85). Entry point:
`docs/HANDOFF_2026-09-16b_ds41-prefill-pipeline.md`.
