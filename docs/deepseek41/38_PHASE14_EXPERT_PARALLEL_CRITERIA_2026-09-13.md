# V4.1 port — Phase 14 gate criteria: expert parallel across both cards (written BEFORE the build)

**The founder's goal:** 400+ tok/s prefill, 20+ decode. **Where prefill goes today** (the resident
test, 4520911 + the block time of 6485afa not yet in that measurement; `reg_resident_off.log`):
pp2048 cold 7.41 s = 276 tok/s, of which the MoE call is 5,823 ms and **the groups — the pinned
experts streamed over the owning card's ONE PCIe link — 5,505 ms** (5,701 pinned experts per pass
× 17.9 MiB = 100 GiB at ~18 GB/s); attention 700, ffn-pre 444, the mmap group 272 with the disk
reader's fill leg 4,411 ms overlapped. The link is 78% of the pass and the other card's link sits
idle while it runs, because each card owns a contiguous layer range and every expert of a layer is
served by the card that owns the layer. **At decode** the same link carries 1,378 MiB per token
(~60 ms of the ~158) with the other link idle.

**The lever:** split every layer's routed experts between the two cards by the held-out ranking's
parity — the card that owns the layer serves one half from its tiers, the other card serves the
other half from its own tiers over its own link and computes them on its own Xe cores — and add
the two partials. Each card then holds a tier over ALL 40 layers with half the experts per layer:
the VRAM (static slots) and the pinned RAM per card are unchanged in total (45 × 20 → ~22 × 40;
228 × 20 → 114 × 40); the stream slots per card per layer halve (4 + 4 = 8 as today) so their VRAM
does not double. The dense body stays layer-split (attention, hc, router on the owning card).

**Step 0, measured before the build — the disk.** The mmap tier reads 28.5 GiB per pp2048 pass
and the NVMe's O_DIRECT rate is **~12 GB/s whatever the reader count**: one `dd` 11.7 GB/s, two
concurrent 5.9 + 5.9, four 4 × 3.0, six 6 × 1.8 (16 MiB and 4 MiB blocks). So two readers do not
double the fill; the fill leg's floor is ~2.4 s of reads per pass plus the permute, which does
split across two teams (1,542 → ~770 ms each).

**The arithmetic, stated before the build.** pp2048: the groups 5,505 → ~2,750 ms per pass with
the two links streaming in parallel; the fill leg ~3.2 s (reads 2.4 at the disk's ceiling + the
halved permute), overlapped with the groups as today, so the MoE call lands near max(2.75, 3.2)
+ the mmap group ≈ 3.4 s and the disk is the co-bottleneck; the cross-card traffic is x (T × H
fp32 = 40 MiB at 2,048 tokens) out and the partial y back, through pinned host staging (D2H,
a host copy between the two cards' pinned buffers, H2D), ~8 ms per layer = ~0.32 s per pass,
later P2P. Projected pass ~4.9 s → **~400-450 tok/s**, right at the goal, with lever C (more
pinned RAM = fewer disk experts) the next term. Decode (the peer's pre-read, P1): the cost per
layer is the SLOWER card's, max(k, 6 − k) over the parity split, not the mean — E[max] = 3.94
of 6 selections, and over the ~1.9 link misses per layer E[max] ≈ 1.5 of 2 = 75% — so the
link's ~60 ms/token falls to ~39-45, the transfers add ~8 ms (20 KB each, ~0.2 ms per layer),
and the projection is **~145-151 ms/token**, barely distinguishable from the 149.6 the CPU split
already delivers; the post-EP miss distribution differs (22 + 4 slots per layer for ~3
selections), so only the measurement decides, and criterion 4 reports the per-layer max/min
expert and miss split so this can be confirmed or refuted. Prefill is the prize; decode is
measured, not promised. The number is what it is.

## Scope, in order; each step verified before the next

1. **The tier takes a subset.** `Ds41ExpertTier::init` gains a per-layer expert subset (the ranking
   positions ≡ c mod 2 for card c); experts outside it are tier 3 ("not mine"): never static,
   pinned or mmap, skipped by the partition, and their packed rows carry weight 0 and a zeroed row
   so the scatter adds nothing (the CPU path's mechanism). The full tier (no subset) is byte-for-
   byte today's. Verified by `ie-ds41-tier-test`: the sum of two complementary subset tiers'
   outputs equals the full tier's within **1e-6** relative — the expected difference is fp32
   reassociation of ~6 terms per token (~1e-7, the floor: the grouping is not removable, so
   exact equality is not available even with a fixed accumulation order), the bar 10x it, and
   the value printed; a second call bit-identical; the two halves' expert counts add up to the
   full tier's.
2. **The forward in EP mode** (`ResidentOptions::expert_parallel`, env `IE_DS41_EP=1` for the A/B):
   every card's tier is initialised over all layers with its subset and its halved budgets; per
   layer, the owner stages x to the other card (pinned host staging: D2H, a host copy, H2D), runs
   its own tier's `moe` on the host thread and the other tier's on a helper thread (its own reader
   thread and OpenMP block time), joins, and merges. **Amended during the build (the reason
   stated):** the first protocol merged a fp32 PARTIAL SUM per card, and the dumps showed why that
   cannot ship — the scatter accumulates a token's six products in fixed k-order in fp32, so two
   partials differ from the whole by one reassociation (1e-8 at layer 0), the RMS norm rescales
   that to 4e-5 on ordinary channels, and the Q8 activation quantisation of the next layer turns
   it into a different rounding trajectory (1e-3 per layer, 40% L2 by layer 39 in the prefill,
   the forced bar at layer 38 tripped by 16%, the generate continuity at T = 512 by 20%). The
   shipped protocol exchanges ROWS instead: the other tier runs in export mode (its packed fp16
   rows, as its GEMMs wrote them, no scatter), the owner imports them into the same packed
   positions of its own workspace before its one scatter, the alien weights stay real, and the
   accumulation is the whole tier's, bit for bit. Stats: the two
   tiers' expert/byte counts summed per layer, the two `moe` walls kept separately (`moe_ms` =
   the owner's, `moe_remote_ms` = the other's), the transfer time counted. `expert_tier(L, e)`
   answers from the card that holds e.
   **The control arm** (the peer's pre-read, P2 — the analogue of Phase 13's q* = 1.0):
   `IE_DS41_EP=control` runs the same machinery with no parallelism — the owner's tier is
   today's full tier over its own layers, and every other card runs an EMPTY tier (a subset of
   no expert; the tier convention `part == n_parts`) for the layer on the helper thread, with
   the same staging, joins and add. Machinery cost = control − off; parallelism gain = EP −
   control. Without it a null decode result is unattributable and a positive one overcredited.
3. **Correctness, the whole set with the env on — and stronger than the bars (amended with the
   row protocol): EP is BIT-IDENTICAL to off.** The decode test's dumps (`IE_DS41_DUMP_DIR`: the
   prefill's per-layer MoE inputs, MoE outputs and layer outputs, and the forced step's) differ
   by exactly 0 at all 40 layers, and every digit the tests print equals the off arm's. Then the
   set: forward test, resident test, replay test,
   decode test (the four greedy tokens, the forced digits at all 40 layers under their bars),
   generate test (the continuity at four lengths, the 210 ms bar — expected well under), tier
   test, tokenizer/prompt tests unaffected. Unload leaves no thread and no device memory
   (free_resident on both cards).
4. **Measured on one build, three arms** (off / control / EP; the block time in all): pp512 and
   pp2048 cold/warm from the resident test; decode steps 4-35 at the 2,048-token context from the
   decode bench with, per layer, the max/min of the two cards' expert and link-miss counts (the
   E[max] question); the load time; the per-card VRAM/pinned accounting printed at init
   (card_info) — both cards' static + stream + pinned per layer over 40 layers.
5. **The founder-facing statement:** the prefill and decode numbers against 400+ / 20+, the
   corrected lever table (docs/35's two lines plus this phase's), and what remains (P2P instead of
   host staging, lever C, attention at T = 1, FP8 dense).

## Pass criteria

1-3 above hold; 4 is reported whatever it says; the default (env off) is byte-for-byte today's
path; a refusal — not a fallback — on one card (`init_resident` with one queue and the env on is
the ordinary single-card path, stated) and on the CPU miss split's cores when both tiers would
run workers on the same E-cores (this phase: the split with EP is refused at init with the reason;
the two-worker core partition is the follow-up).

## Explicitly NOT in this phase

P2P transfers (host staging first, measured, P2P when the single-device contexts allow it);
prefix-cache reuse across turns (docs/33); the CPU split under EP; FP8-resident dense (Phase 15).
