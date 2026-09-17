# V4.1 port — Phase 21 criteria: the per-layer E[max] term, and moving misses across the cards (written BEFORE the build)

**The term.** Under expert parallel each card owns a share of every layer's experts (ranking positions ≡ part mod
n_parts, `deepseek41_experts.cpp:278`), fetches its own share's misses over its own link, and the MoE call cannot
finish until BOTH cards have finished — so every layer pays the LATER card. Measured directly (docs/51, and the
decode bench's `ep E[max]` line on the current default build): per token, summed over layers, the later side is
**57.1 ms** and the earlier **24.0 ms**; the link misses divide **39.3 / 14.5** the same way. The cause is not a slow
card and not the helper thread (docs/51 falsified the join hypothesis): it is that a layer has only ~1.3 link misses
and they land 2-0 or 1-0 on the parity halves far more often than 1-1.  docs/53 established that the CPU miss split
cannot absorb it (the E-cores serve T = 1 misses from the arena at ~20 GB/s under the links' DMA; the E[max] line did
not move: 53.4 / 24.3 with the shared pool against 53.7 / 24.2 without).

**Why it cannot be fixed by a better static split.** The split already interleaves by measured hotness, so it is
balanced in expectation; what costs is the per-layer VARIANCE of which experts a token actually routes to. Balancing
therefore has to happen per layer at run time, which requires a miss to be servable by EITHER card.

**The blocker, verified in the code.** A pinned expert exists only in its owner's arena, and that arena is SYCL host
USM allocated in one queue's context (`Ds4HostArena`'s segments are `sycl::malloc_host(bytes, q)`,
`src/core/expert_stream.cpp:1314`). `expert_stream.hpp:375` states the rule this repo already follows: "a USM
allocation belongs to the context it was made in, and handing card 1 a pointer that only card 0's context knows is
undefined behaviour" — and since the multi-device-context mirror fix the engine uses single-device contexts, so the
two arenas are genuinely private. **The escape hatch already exists in this file**: the expert file's staging slots
are anonymous (huge-page) memory imported into the Level Zero driver with `zexDriverImportExternalPointer`
(`deepseek41_experts.cpp:293-307`) and the driver DMAs them. Both cards are enumerated from ONE L0 platform, so the
driver handle both tiers obtain should be the same — **an assumption, and step 1's first checkpoint.**

## The phase, in three bounded steps, with a decision rule fixed before the measurement

**Step 0 — measure the ceiling before building anything** (a diagnostic only: a histogram, no path change). The
bench sums `max` and `min` per layer but never the DISTRIBUTION, and the achievable saving depends on it, because a
miss is atomic — one expert is one 18.8 MB fetch, so a layer at (1,0) or (2,1) cannot be improved at all while (2,0)
and (3,0) can. Add to the decode bench a per-layer histogram of (slower, faster) miss counts over the measured
steps, and from it the **atomic-balanced ideal**: Σ_layers ceil((m_s + m_f) / 2) against today's Σ_layers m_s, in
misses and in ms (scaled by the measured per-miss link time).

> **DECISION RULE, fixed now:** if the atomic-balanced ideal saves **< 6 ms/token** at the 2,048-token context, Phase
> 21 stops at step 0 and the doc records the term as measured-and-refused, because the refactor below cannot beat its
> own ceiling. 6 ms is ~7.5 % of the 79.9 ms/token default — below that, the lever docs/59's closing list ranks next
> (the decode hit rate) comes first.

**Step 1 — one shared arena both cards can DMA** (only if step 0 clears the rule). The pinned arena becomes anonymous
huge-page memory imported into the L0 driver once and handed to both tiers; each tier still OWNS and FILLS exactly
its own parity share, and the slot map covers the union so either tier can READ any pinned expert. Nothing about
which card computes which expert changes, so this step is **bit-identical by construction** and buys no speed by
itself — it buys the capability step 2 needs.

**Step 2 — balance each layer's misses across the cards** (its own criteria doc and gate). For a layer whose miss
set divides m_s / m_f, move misses from the fuller side until the counts differ by at most one, and let the receiving
card fetch those experts from the shared arena and compute them. **This changes the grouping of the MoE sum, so the
digits change and multi-row bit-identity is forfeited while it is on** — stated here, before the build, not
discovered by the gate: the per-card partial sums are added, so moving an expert between cards changes the rounding,
and a balance that depends on the current call's miss set makes a T-row step's grouping differ from a one-row step's
(the property `ie-ds41-multi-test` certifies for the opt-in speculative path docs/59 just measured as a net loss).

## Pass criteria for THIS phase (steps 0 and 1; `ie-ds41-decode-test` with the ranking, `IE_DS41_EP=1 IE_DS41_EXPERT_FILE`)

1. **Step 0's histogram and ideal, reported** at the 2,048-token context over ≥ 32 measured steps: the (slower,
   faster) pair counts, the mean misses per layer, Σ m_s, the atomic-balanced Σ ceil((m_s+m_f)/2), the per-miss link
   ms implied by the measured bytes and time, and the saving in ms/token and tok/s. The decision rule above is then
   applied in writing.
2. **The driver-handle assumption settled**: the two cards' `ze_driver_handle_t` printed and compared. If they
   differ, the arena must be imported once per driver, and the phase says so rather than assuming one import covers
   both.
3. **Step 1 is bit-identical**: the decode test's forced-routing digits and its 200 per-step dumps equal the current
   build's exactly (the same comparison Phase 19 used for the persistent helper), and `ie-ds41-multi-test` still
   passes 38 / 38 with `IE_DS41_CPU_MISS=0`.
4. **Step 1 costs nothing**: ms/token at the 2,048-token context within the run-to-run spread of the current default
   (79.9, spread 78.6-81.3 over the P5 samples), pinned RAM within 1 % of today's, and the load time within 5 s.
5. **The foreign-read capability proven, not assumed**: a test in which card B reads a pinned expert that card A owns
   and compares the bytes with the pack path's own output for that expert (the same byte-for-byte check the expert
   file's validation already does for one slot). Without this, step 2 has no foundation.
6. **A kill switch**: `IE_DS41_EP_SHARED_ARENA=0` restores the per-card arenas, and the bit-identity of criterion 3
   holds in both arms.
7. Prefill, the resident test, the replay test and the generate test unchanged (the arena is a decode-path store,
   but it is allocated at init and the prefill reads it too).

## Explicitly NOT in this phase

Step 2's balancing (its own doc, gate and digits argument), any change to the ranking or the static/pinned counts,
the CPU miss split's q*, and DSpark (closed by docs/59: opt-in, a net loss).

## Step 0 results (16:05, `ie-ds41-decode-test` with the held-out ranking; log `~/ds41_work/p21/step0.log`)

**PHASE 21 IS REFUSED BY ITS OWN DECISION RULE. The achievable saving is 3.2 ms/token, not the ~20 the E[max] line
suggested.** The rule fixed above required ≥ 6 ms/token at the 2,048-token context; the measurement says 3.2, so
steps 1 and 2 are not built and the term is recorded as measured-and-refused.

At the 2,048-token context (32 measured steps, decode hit rate 68.9 %, 82.7 ms/token):

| a layer's link misses (fuller, emptier) | count of layer-steps | what an atomic balance would change |
|---|---|---|
| (0,0) | 196 | nothing |
| (1,0) | **550** | **nothing — one expert is one atomic 18.8 MB fetch and cannot be halved** |
| (1,1) | 366 | nothing — already balanced |
| (2,0) | **72** | the only real win: 2 → 1 on the critical path |
| (2,1) | 87 | nothing — ceil(3/2) = 2 is already the max |
| (2,2) | 4 | nothing |
| (3,0) / (3,1) | 3 / 2 | 3 → 2 on the critical path |

Today the critical path carries **39.28 misses per token**; an atomic per-layer balance would carry **36.88**, i.e.
**2.41 fewer**. At the measured **1.33 ms per critical miss** (the E[max] gap 56.3 − 23.2 divided by the miss gap
39.3 − 14.5) that is **3.2 ms/token: 82.7 → 79.5, 12.09 → 12.58 tok/s.** Under the replayed short-prompt state
(hit rate 56 %, 147.6 ms/token) the same computation gives 9.7 ms/token, but that regime is disk-bound and is not
the one the mandate is measured in.

**What the histogram corrects, and it is the finding of this phase.** docs/51 read the E[max] gap as "~20 ms/token
above an even split", and Phase 20/20b and this phase were all written against that reading. The gap is real, but it
is not mostly a *balance* problem: **550 of 1,280 layer-steps are (1,0)** — one card has a single miss and the other
has none. That is not imbalance that can be moved, it is the irreducible cost of a layer having any miss at all
while its partner has none, and no cross-card scheme can divide one atomic expert fetch. Only the 5.6 % of
layer-steps at (2,0) and the 0.4 % at (3,0)/(3,1) are movable. "An even split" was never the achievable baseline;
the atomic-balanced split is, and it is 3.2 ms/token away.

**So the shared arena and the runtime balance are not built**, and the ~0.7 GiB of extra RAM, the import refactor,
the forfeited multi-row bit-identity and the digit changes are all not spent. Criteria 2-7 are moot; they are left
above as written so the refusal is auditable against them.

**Where the time actually is, from the same run.** 39.28 critical misses × 1.33 ms = **52 ms of the 82.7 ms/token**
is critical-path miss service. The lever is therefore **fewer misses, not better-placed ones** — the decode hit rate
(68.9 % here: 116.4 static + 102.7 pinned + 1.2 mmap of 240 selections per token, 963.7 MiB/token still crossing the
links). That is docs/59's item (b) and it is now the measured front-runner: each point of hit rate removes ~14 MiB
and ~0.19 ms from every token's critical path. A phase on residency policy (more pinned RAM, a better ranking, or
admitting the 8 stream slots' evictions differently) is the next criteria doc.
