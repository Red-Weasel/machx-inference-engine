# V4.1 port — Phase 8 results: the resident runtime on two cards

**Criteria:** `21_PHASE8_RESIDENCY_CRITERIA_2026-09-12.md` (with its disclosed amendments).
**Test:** `ie-ds41-resident-test <model> <golden> <golden>/model [profile_out] [ranking_in]`.
**Logs:** `scratchpad/golden/resident_run{2,3}.log`, `tier` test in the tool's own output.

## The three runs

| run | cards | ranking | mmap tier path | load | pp512 cold / warm | pp2048 cold / warm | whole run |
|---|---|---|---|---:|---:|---:|---:|
| 1 (`cb1f42d`) | 1 | index order | SYCL memcpy from the pageable mmap | 515 s | 3.2 / 3.3 | 7.9 / 8.1 | 1,387 s |
| 2 | 1 | measured (run 1's profile) | same | 531 s | 9.0 / 8.9 | 11.9 / 12.4 | 1,387 s |
| 3 | **2** | measured (run 2's profile) | host pack into pinned staging + one DMA per expert; fetch pipeline; whole-mapping `MADV_NORMAL` | **136 s** | **47.8 / 84.5** | **61.9 / 59.2** | **226 s** |

tok/s throughout. Run 3's per-card placement, derived from each card's free VRAM after its
dense set and a 3 GiB reserve, and from its share of the 160 GiB host pin cap:

```
card 0: layers 0..19,  dense 6.90 GiB, experts 21.71 GiB VRAM (54 static + 8 stream / layer), 79.8 GiB pinned; VRAM free 31.86 -> 1.94
card 1: layers 20..39, dense 7.54 GiB, experts 21.01 GiB VRAM (52 static + 8 stream / layer), 79.8 GiB pinned; VRAM free 31.85 -> 1.99
```

Experts by tier per forward (UNIQUE OCCUPIED experts per layer, summed over the 40 layers —
not selections; the by-selection hit rates are in the fix-round section below):

| forward | static | pinned | mmap | mmap -> VRAM | MoE time / pass |
|---|---:|---:|---:|---:|---:|
| golden prompt, 12 tok | 174 | 951 | 101 | 1.77 GiB | (forward 2.15 s cold, 1.22 s warm) |
| pp512 | 1,943 | 5,840 | 315 | 5.5 GiB | 6.8 s of 10.7 |
| pp2048 | 2,120 | 8,982 | 1,321 | 23.1 GiB | 18.9 s of 33.1 |

Run 2 at pp512, for the same prompt on one card with 12 static slots: 460 / 6,670 / 968 and
16.9 GiB; run 1 (index order): 243 / 4,824 / 3,031 and 53.1 GiB.

## What changed between run 2 and run 3, and what is NOT separated

Four things went in at once, so run 3 attributes nothing to any one of them:

1. **Two cards, pipeline by layer.** `Ds41Forward` holds one `Card` per queue (its dense cache,
   its expert tiers, its layer range); the layer body is the same code, run on whichever card
   owns the layer. At the boundary the hyper-connection stream (T x 4 x 5120 fp32), the previous
   layer's `ffn_pre` mix and the shared attention caches are copied to the host and back: 160 MiB
   at pp2048, a few ms. Layer 20 happens to rebuild the shared caches (it is both a kv-source
   and an index-source layer), so nothing of them actually crosses at this split, but they are
   carried anyway so the split point is not a hidden dependency. The dense set per card halves
   (6.9 / 7.5 GiB) and the static tier goes from 12 slots per layer to 54 / 52.
2. **The mmap tier packs on the host into pinned staging** (8 slots, one DMA per expert), the
   same path the arena uses, instead of a SYCL memcpy from the pageable mapping that the runtime
   served one page fault at a time (~350 MB/s, state D). The tier test shows the effect in
   isolation: 9 mmap experts, 161 MiB, inside a 141 ms MoE call.
3. **The whole-mapping `MADV_NORMAL`** through `SafetensorsModel::advise`, once per shard, in
   `Ds41ExpertTier::init` — this IS the fix commit `cb1f42d`'s message wrongly claimed; see the
   correction in docs/21. Load: 515 s -> 136 s for the same 159.7 GiB pinned.
4. **V4's fetch pipeline** in the tier's group loop (`acquire_pipelined`, group g+1 issued
   before group g's GEMMs). Not separately measured here; V4 measured 1.8-1.9x on this box.

The tier test (`ie-ds41-tier-test`) re-ran on the final code: two splits bit-identical, rel
3.056e-3 against the reference as before, VRAM returned.

## Criteria, one by one

1. **Correctness unchanged (as amended).** Own router on the golden prompt, Q8 grouped path:
   worst 6.124e-3 through layer 6, first flip at layer 7 at golden margin 1.932e-3 (a near-tie),
   top-1 " Berlin". Identical to runs 1 and 2 to the printed digit — the split moved bytes, not
   arithmetic. Warm second forward bit-identical. **PASS.**
2. **Bytes moved per forward.** Dense: 0 on every forward (resident). Experts: the mmap tier's
   bytes are exact (above); the pinned tier's H2D bytes are reported as an upper bound (every
   pinned expert in a group holds a stream slot, hits included), not measured at the copy.
   **PASS on dense, upper-bound on pinned** — the exact pinned-tier byte count is a named gap.
3. **Priority measured, not assumed.** The profile is real (own-router selections, written in
   V4's format, read back by the runtime) and the control comparison exists at equal slots on one
   card: pp512 static hits 243 (index order) -> 460 (measured), 3.0% -> 5.7% of the occupied
   (layer, expert) pairs — not of selections — and the mmap tier 3,031 -> 968 experts. Two caveats the criterion asked about and this run does
   not meet: the profile was gathered on the SAME prompts it is measured on (golden x2, pp512
   x2, pp2048 x2 — about 5K tokens, not the >= 64K held-out corpus), so the pp512/pp2048 hit
   rates are in-sample; and the test does not assert "the VRAM set IS the top of the ranking"
   (the tier's `init` places rank [0, n_static) by construction and the tier test checks the
   placement, but the resident test has no such assertion). **Measured; corpus and assertion
   open.**
4. **Two cards, no mirror.** Both queues on explicit single-device contexts. MemAvailable
   dropped 160.80 GiB for 159.68 GiB pinned: a mirror of the 57 GiB of VRAM would have shown.
   **PASS.**
5. **Pinned tier accounted.** 160.80 vs 159.68 GiB, within 5% + 2 GiB. **PASS** (in-process
   this time; run 1's 191 GiB drop was the old mmap path's RSS on top).
6. **Prefill measured on the real path.** pp512 47.8 cold / 84.5 warm, pp2048 61.9 / 59.2, on
   the same build and process that re-proved correctness. Against DS4-Flash's 686 at 2048 this
   is 11x below. Why, from this run's own numbers: at pp2048 the MoE is 18.9 s of 33.1 s and
   the other 14.2 s is the dense body of 40 layers — the Phase 7 correctness loop, fp16 oneDNN
   GEMMs with a wait after nearly every op, fp32 T x T masks and scores, no XMX attention, no
   fused norms. Both halves are untuned; neither has been profiled yet. **Measured.**
7. **Unload returns everything.** VRAM: 197 MiB / 191 MiB unreturned on cards 0 / 1, under
   256 MiB. RAM, judged at the process boundary per amendment 2: MemAvailable 213.9 GiB before
   launch, 224.9 GiB after exit. The in-process figure the test prints (67.6 GiB "not returned"
   while the process was alive) is the Level Zero runtime holding freed pinned allocations plus
   the mmap tier's touched pages — it is why the criterion moved to the boundary. **PASS at the
   boundary.** The test's in-process unload check is now an informational print, not a
   `[FAIL]`: it measured something the amended criterion no longer asks, and it failed on every
   run for that reason. The load-side check (criterion 5) stays a check.
8. **The mmap tier.** Built, on the OS page cache over the NVMe; cost reported: 23.1 GiB per
   pp2048 forward inside an 18.9 s MoE total. The tier does not yet time its phases (pack, DMA,
   GEMM) separately, so the mmap tier's share of that 18.9 s is not isolated. **Named, built,
   cost bounded above, not separated.**

Warm vs cold: pp512 warm is 1.8x cold because the 8 stream slots per layer keep the previous
pass's pinned-tier hits; at pp2048 the working set (nearly all 384 experts per layer) exceeds
the slots and warm equals cold.

## Where the time goes now, and the next lever

At pp2048, 33.1 s = 18.9 s MoE + 14.2 s dense body. Inside the MoE, the mmap tier moves
23.1 GiB per forward through `ds41_slot_pack` — a single-threaded per-byte nibble permutation
of every expert (17.9 MiB each, 1,321 of them) — before the DMA. If that loop runs at 1-2 GB/s,
it alone is 12-23 s: it would be most of the MoE time. That is a hypothesis from arithmetic,
not a measurement; the next step instruments the tier (pack / DMA / GEMM ms per call) and, if
confirmed, replaces the host permutation with a plain memcpy into pinned staging plus Phase 4's
in-place device repack kernel (`ds41_expert_bank_upload` already has it). The other half, the
dense body, is Phase 7's correctness loop and has never been profiled; DS4-Flash's tuned dense
path is the template.

Decode is Phase 9 (criteria in docs/19; golden in `scratchpad/golden/decode/`).

## Gate verdict on 3ea4b87: FAIL — and the fix round (2026-09-12, 20:35)

The Phase 8 gate (a fresh evaluator that rebuilt and re-ran every test itself) returned FAIL
with two blocking findings and six others. What it found, and what changed:

1. **BLOCKING — HEAD did not compile.** `include/ie/deepseek41_experts.hpp` had two
   uncommitted lines (`mm_stage_`, `mm_stage_slots_`) that 3ea4b87's `.cpp` uses; the measured
   binaries came from the working tree, not from any commit — the same class of error as the
   cb1f42d correction above. Committed as `524d454`. My status check had listed the forward
   header and not this one.
2. **BLOCKING — criterion 3, the profile was in-sample.** The gate's own two-card index-order
   control measured pp2048 at 33.0 tok/s against 62.3 with the in-sample ranking: the headline
   carried a 1.9x in-sample advantage. Fixed by a held-out profile: 65,536 tokens of the
   engine's own docs (180 files, 0 shared 64-token windows with the pp text) prefilled in 32
   chunks of 2048 through the own router, written by `write_profile`, and the test now prints
   the hit rate BY SELECTIONS (the gate's finding 7: the earlier "hit rates" counted unique
   occupied experts) for three rankings at once — the one in use, index order, the held-out
   profile — plus an assertion that the placement IS the ranking (0 misplaced pairs).
   Run A, ranking in use = the in-sample profile, corpus profiled in 1,782 s (36.8 tok/s):

   | pass | in use (in-sample) VRAM/pinned/mmap % | index order | held-out profile |
   |---|---|---|---|
   | pp512 | 72.1 / 27.6 / 0.3 | 14.5 / 59.7 / 25.8 | 28.8 / 55.3 / 15.8 |
   | pp2048 | 66.4 / 33.0 / 0.5 | 14.2 / 60.2 / 25.6 | 45.0 / 46.8 / 8.2 |

   So a real profile roughly doubles-to-triples the VRAM hit rate over index order, and the
   in-sample figure overstated it by another 1.5-2.5x. **Run B — the held-out ranking in use,
   the honest pp numbers — has not been run yet** (the machine was needed); it is the next
   launch: `ie-ds41-resident-test <model> <G> <G>/model "" <G>/ds41_profile_heldout.txt`.
3. **Finding 3 (card 1 unjudged).** A FORCED-routing pass now judges all 40 layers on the
   resident two-card path: card 0 worst 9.02e-3 at layer 18, card 1 worst 2.76e-2 at layer 38,
   no jump at the boundary (8.36e-3 -> 8.20e-3), top-5 ORDER the golden's, logits rel 4.39e-2.
   The 5e-2 logits bar and the 1.5x boundary factor are regression bars set after the gate's
   probe measured those values, and the test says so.
4. **Finding 4.** "dense moved: 0" is now summed from the layer stats and checked.
5. **Finding 5 / criterion 8 — the mmap tier's cost, isolated.** The tier times its phases:

   | pass | MoE ms | prep | mmap total | of which host pack | groups (fetch + GEMM + scatter) |
   |---|---:|---:|---:|---:|---:|
   | pp512 | 7,096 | 1 | 2,936 | 2,850 | 4,149 |
   | pp2048 | 23,117 | 3 | 14,658 | 13,647 | 8,444 |

   At pp2048 the single-threaded nibble pack is 13.6 s of the 35.6 s pass — 59% of the MoE and
   1.7 GB/s over 23.1 GiB, the rate the gate had inferred from its two runs. Confirmed, not
   hypothesised: docs/23's step 2 (parallel pack, then device repack if the host copy binds)
   is the next lever. The groups' 8.4 s is the second.
6. **Finding 6.** `tests/unit/ds41_residency_plan_test.cpp` was compiled with `-DNDEBUG`, so
   its `assert`s were compiled out and it could not fail; now a `CHECK` macro that fails
   regardless. The planner itself remains an offline tool: the runtime budgets from each card's
   measured free VRAM (`init_resident`), and the test's hit-rate table asks the runtime where
   each expert actually is rather than the planner.
7. Amendments judged by the gate: A1 (Q8 floor) legitimate; A2 (process boundary) legitimate;
   A3 (control = run 1) a WEAKENING — accepted, and replaced by the held-out profile above;
   the criterion-7 test change legitimate but the RAM half is enforced only by the launcher.
   For this run: MemAvailable 220.2 GiB before launch, 219.8 GiB after exit (in-process it
   printed ~60 GiB "not yet returned", as on every run).

### Run B (2026-09-13, 06:43) — the held-out ranking in use: the honest Phase 8 number

Same build (107ea36), `ranking_in = ds41_profile_heldout.txt`, no corpus pass. PASS on every
check with the same correctness digits (own router 6.124e-3 to the flip at 7, " Berlin",
forced pass card 0 9.02e-3 / card 1 2.76e-2, top-5 order, logits 4.39e-2). Placement
asserted (0 misplaced). Load 140 s. VRAM unreturned 19 / 18 MiB.

| pass | cold / warm tok/s | occupied static / pinned / mmap | mmap -> VRAM | MoE ms = prep + mmap (host pack) + groups | selections VRAM/pinned/mmap % (in use vs index order) |
|---|---|---|---:|---|---|
| pp512 | **24.7** / 34.3 | 1,375 / 5,293 / 1,430 | 25.0 GiB | 16,563 = 2 + 12,772 (12,454) + 3,774 | 28.8/55.3/15.8 vs 14.5/59.7/25.8 |
| pp2048 | **40.1** / 42.1 | 2,047 / 8,085 / 2,291 | 40.1 GiB | 30,057 = 4 + 23,076 (19,330) + 6,964 | 45.0/46.8/8.2 vs 14.2/60.2/25.6 |

Against the in-sample runs (47.1 / 57.6 cold) the held-out placement gives 24.7 / 40.1: the
in-sample headline overstated pp512 by 1.9x and pp2048 by 1.4x, which is what the gate's
control predicted. The host pack is 60% of the pp512 pass and 38% of the pp2048 pass; the
whole mmap tier 61% / 45%. Phase 8's deliverable — a measured number with correctness
re-proved on the same build — is these two rows.

Run A's other numbers, same build: load 140 s; own-router worst 6.124e-3 to the flip at 7,
" Berlin", warm bit-identical 1.22 s; pp512 47.1 cold / 85.3 warm; pp2048 57.6 cold / 84.7
warm (run 3's warm was 59.2 — the warm pass varies between runs by more than the cold pass
does, unexplained); VRAM unreturned 63 / 65 MiB; whole run 2,006 s of which 1,782 s the corpus.
Not re-presented to the gate yet.

## Files

`include/ie/deepseek41_forward.hpp`, `src/model/deepseek41_forward.cpp` (cards),
`src/model/deepseek41_experts.cpp` (advise, staged mmap tier, fetch pipeline),
`include/ie/safetensors.hpp`, `src/loaders/safetensors_reader.cpp` (`advise`),
`tools/ds41_resident_test.cpp` (all Level Zero Arc cards, per-card report, boundary note),
`tools/ds41_forward_test.cpp` (probe takes the owning queue).
