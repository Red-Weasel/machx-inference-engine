# V4.1 port — Phase 14 results: expert parallel across both cards

**Criteria:** `38_PHASE14_EXPERT_PARALLEL_CRITERIA_2026-09-13.md` (with the peer's pre-read P1-P5 and
the mid-build amendment to the row protocol, both recorded there). **Starting line:** 878a0b8 —
pp512 109.5 / pp2048 279.9 tok/s (the resident test, cold), decode 158-161 ms/token at the
2,048-token context with the CPU split off, 149.6 with it on (docs/35).

## Step 0 — the disk (measured before the build)

The NVMe delivers ~12 GB/s of O_DIRECT reads whatever the reader count (one `dd` 11.7 GB/s; two
concurrent 5.9 + 5.9; four 4 × 3.0; six 6 × 1.8). The mmap tier's 28.5 GiB per pp2048 pass has a
2.4 s floor; two readers do not double the fill, only the permute splits.

## Step 1 — the tier takes a subset (`Ds41ExpertTier::init(..., part, n_parts)`)

Experts at ranking positions ≡ part (mod n_parts) are the tier's; the others are tier 3, skipped
by the partition, their packed rows zeroed and — without an importer — given weight 0. `part ==
n_parts` is the empty subset (the control arm). `ie-ds41-tier-test`: two complementary halves'
partials sum to the full tier's output at **rel 9.4e-8** (layer 0, the golden routing, T = 8 —
an earlier draft said 0.00e+00: a six-decimal `to_string` of that number, gate 14 finding 3),
**4.7e-8** and **8.2e-8** (layers 1 and 14, random routing, T = 12; bar 1e-6, floor ~1e-7 from
fp32 reassociation); the halves' expert counts add up (42 of 42); a second call bit-identical.

## Step 2 — the forward: two protocols, one shipped

**The first protocol (fp32 partial sums per card) could not ship, and the dumps say why.** The
scatter accumulates a token's six products in fixed k-order in fp32; two partials differ from the
whole by one reassociation — 2.8e-8 at layer 0 (max-relative) — the RMS norm rescales that to
4.2e-5 on ordinary channels, and the next layer's Q8 activation quantisation turns it into a
different rounding trajectory: 1.1e-3 at layer 1, growing every layer, 40% L2 by layer 39 in the
prefill. Against the golden: the forced bar at layer 38 tripped (4.06e-2 vs 3.5e-2), the forced
logits 2.1e-2 vs the off arm's 7.7e-3 (bar 5e-2), the generate continuity at T = 512 6.0e-2 vs
5e-2, the near-tie flip moved from layer 10 to 12. Ruled out on the way: a race (the remote tier
inline, `IE_DS41_EP_SEQ=1`: identical digits), the halved stream pipeline (EP at 8 slots:
identical digits), the machinery (the control arm: digits identical to off), the tier itself
(exact at layers 0/1/14). The instrument that found it: `IE_DS41_DUMP_DIR` now dumps the
prefill's per-layer MoE inputs (`PX`), MoE outputs (`PM`) and layer outputs (`P`) beside the
forced step's (`M`, `L`).

**The shipped protocol exchanges rows, not sums.** The other card's tier runs in export mode
(`set_ep_export`): everything as usual, no scatter; its packed fp16 rows stay in its workspace,
`ep_rows()` lists which packed rows are its experts'. The owner's tier runs with an import hook
(`set_ep_import`) called on its queue after its own rows are written and before its one scatter:
the hook joins the helper, copies the other tier's rows (gathered contiguously on that card, D2H,
a host copy between the two cards' pinned buffers, H2D) into the same packed positions of the
owner's workspace — the packing is the same counting sort of the same routing on both cards —
and the alien weights stay real. The one scatter then accumulates the six products in the whole
tier's order, bit for bit. **Verified: the dumps differ by exactly 0.0 at all 40 layers** (PX,
PM, P, M, L), the decode test's digits are the off arm's (card 0 worst 9.355e-3, card 1
1.486e-2, the four greedy tokens, hashes), the resident test's are the off arm's (9.024e-3 at
layer 18, 2.755e-2 at layer 38, forced logits 4.195e-3).

The rest of the machinery: each card's tier over all 40 layers with its parity share — at the
shipped default of 8 stream slots per card per layer, 18 + 8 / 17 + 8 slots and 114 pinned per
layer per card: the pinned RAM is the same in total as 228 over 20 layers (79.8 GiB per card),
and the VRAM trades 400 static experts for 320 more stream slots against off's 45 + 8 / 44 + 8
(hit rate 66.0% vs 61.6%, 3.5 more disk experts per token: gate 14 finding 5); `card_info`'s
"experts N GiB VRAM" omits the EP staging (~160 MiB per card; the "VRAM free" line is the truth:
finding 6) — x staged through the two cards' pinned buffers, a
helper thread per remote card per layer with its own OpenMP block time, the control arm
(`IE_DS41_EP=control`: the same machinery, the other card's tier empty — its tier now holds no
VRAM slot and is allocated after the budget, so the owner's static count is off's; on the build
gate 14 measured, it sat inside the budget and cost 5 static slots per layer, which made that
arm SLOWER than off (decode 168 vs 157-162, pp2048 270 vs 277) and unable to attribute anything
— finding 1; the corrected arm is re-measured below. The lesson, in the words the next control
arm in this campaign will be designed from: a control must not only do no work, it must also
CONSUME NO RESOURCES — an empty tier still inside the VRAM budget cost five static slots per
layer and made the arm slower than the thing it was controlling for), `IE_DS41_EP_STREAM=N`
for the pipeline-depth experiment, the CPU split refused with EP (the two-worker core partition
is a follow-up), the EP allocations placed BEFORE the VRAM budget query (after it, the control
arm's warm pp2048 pass read 11.6 s against 7.3 — Phase 9's eviction signature).

## Measured (criterion 4), one build, idle machine

`scratchpad/p13/ep_chain.sh` (off / control), `ep_diag.sh`, `ep_final.sh` (the row protocol); the
2,048-token context for decode, the resident test's cold pass for prefill; the block time
unconditional in every arm.

**Prefill** (the resident test):

| arm | stream slots per card per layer | static per layer (card 0 / 1) | pp512 cold / warm | pp2048 cold / warm | pp2048 MoE: groups / fill leg / tail |
|---|---|---|---|---|---|
| off | 8 | 45 / 44 | 109.5 / 112.5 | 279.9 / 281.6 | 5,483 / 4,261 / 8 |
| control (machinery, no parallelism) | 8 | 45 / 44 | 108.7 / 110.5 | 262.4 / 175.9 † | 5,494 / 4,264 / 9 |
| EP, rows, 4 + 4 | 4 | 22 / 21 | 129.1 / 133.1 | 299.6 / 301.3 | 4,516 / 3,644 / 594 |
| **EP, rows, 8 + 8 (the default)** | 8 | 18 / 17 | **123.2 / 128.3** | **307.9 / 319.3** | 4,138 / 3,993 / 449 |
| (EP, partial sums, 8 + 8 — not shipped) | 8 | 18 / 17 | 124.7 / 131.4 | 308.2 / 324.5 | 4,057 / 3,941 / 9 |

(† the control arm measured before the EP allocations moved ahead of the VRAM budget: its warm
pass shows the eviction; its cold pass says the machinery costs ~0.5 s per pp2048 pass.)

**pp2048 279.9 → 308-319 tok/s (+10-14%), pp512 109.5 → 123-129 (+13-18%)** — not the ~400-450
of the projection, and the breakdown says why: the owner's groups fell 5,483 → 4,138 ms for HALF
the experts, i.e. each link streamed ~12.5 GB/s during the pass against 26 + 26 GB/s when the two
links copy alone (`scratchpad/p13/dma_pair`: 52.4 GB/s aggregate, expert-sized copies included).
The links are not the wall; the host memory they read from is, shared with the two readers'
disk DMA, their permute (a read + write pass over 29-32 GiB) and the pread copies — during the
pass the host moves ~45 GB/s on average and more at the peaks. Gate 14 demonstrated it rather
than inferred it: under a 66 GB/s host-memory load (six 1 GiB copy loops) the two links fall
from 52.4 to **15.5 GB/s aggregate** (7.8 each), expert-sized copies included. The fill leg (3.6-4.0 s) now sits
beside the groups (4.1-4.5 s) rather than hidden under 5.5 s of link time, so both walls are the
host side of the mmap tier, not the links: fewer disk experts (lever C: more pinned RAM) and a
permute-free on-disk layout are the terms that move prefill from here, before P2P's 0.3 s.

**Decode** (the decode bench, steps 4-35, 2,048-token context; the split off in every arm):

| arm | ms/token | tok/s | link MiB/token | disk MiB/token | hit rate | per layer: the slower card's selections / link misses |
|---|---|---|---|---|---|---|
| off | 161.1 / 158.3 (builder); 156.6 / 162.0 (gate 14) | 6.2-6.4 | 1,378 | 275 | 61.6% | — |
| control (the arm gate 14 measured: 40 / 38 static, confounded) | 168.2 (gate 14); 156.1 (20:03, pre-fix build) | — | 1,378 | 275 | 61.6% | — |
| control, corrected (an empty tier consuming no resources) | 155.8 / 162.2 | 6.3 | 1,383 | 282 | 61.3% | — |
| EP, rows, 4 + 4 | 151.4 | 6.6 | 1,418 (709 per link) | 285 | 60.4% | 3.9 of 6 / 1.55 of 1.98 |
| **EP, rows, 8 + 8 (the default)** | **149.4 / 156.7 (builder); 156.1 / 150.9 (gate 14)** | **6.4-6.7** | 1,127 (563 per link) | 337 | 66.0% | 3.9 of 6 / 1.27 of 1.57 |
| (EP, partial sums, 4 + 4) | 144.9 | 6.9 | 1,416 | 277 | 60.7% | — |

**Decode: ~5 ms/token, likely real, not robust** (gate 14 finding 2, replacing an earlier
"~161 → ~150" that paired the best EP sample with the worst off sample): on means of two runs
per arm EP 153.5 vs off 159.3 (~3.5%), the spread inside each arm ~5 ms, the sign consistent
across the four bench samples — and in the full decode test's post-replay steps, a different
cache state, EP read SLOWER (194 vs 176 on the gate's runs, 180 vs 174 on the builder's). The
geometry is exactly the pre-read's E[max]: the slower card serves 3.9 of a token's 6 selections
and carries 4x the faster card's link misses (62 of 79 per token at 4 + 4), so the link time per
layer is the slower half's, not the mean's; the machinery-vs-parallelism split of that 5 ms is
what the corrected control arm measures: nothing at decode (155.8 / 162.2 against off 157-159), ~1% of a prefill pass (273.6 / 277.2 vs 276.9 / 281.2) — the gain is the parallelism (docs/41). Prefill is this phase's result; decode is a
small, state-dependent side effect and is reported as such. The row protocol costs ~5-7 ms/token of decode against the partial sums (144.9 at 4 + 4
against 151.4; 149.4 at 8 + 8: the owner's scatter now waits for the other card's rows instead
of overlapping them) and buys bit-identity with the single-tier path — the property this
campaign's whole regression method rests on. The trade is deliberate and is recorded here so no
later phase quietly reclaims the 5 ms by going back to partial sums and loses the property. The
staging + import costs 1.0 ms/token. The load is unchanged (37-38 s).

**Correctness (criterion 3), all with `IE_DS41_EP=1`:** resident PASS (digits the off arm's),
decode PASS (digits the off arm's, the dumps 0.0 at 40 layers — gate 14: 200 of 200 files
byte-identical by `cmp`), generate PASS (the continuity 4.539e-2 / 2.628e-2 / 4.739e-2 /
1.991e-2 at 12 / 130 / 512 / 1030 — the off arm's to the digit; 161 ms/token under the 210 bar),
replay PASS (24 of 24 prefixes — an earlier draft said 20/20: finding 4), tier PASS; forward test PASS (the non-resident path,
unaffected). Unload returns both cards' VRAM and joins the helpers.


## The founder-facing statement (criterion 5)

Against the founder's 400+ / 20+:

| | before Phase 13 | now (Phase 14, `IE_DS41_EP=1`) | the goal | what stands between |
|---|---|---|---|---|
| prefill pp2048 | 277-280 tok/s | **308-319** (gate 14: 307.8 / 311.4) | 400+ | the host side of the mmap tier: 29-32 GiB per pass read from disk, copied, permuted and DMA'd beside the two links' 100 GiB; lever C (pinned RAM: every expert moved off the disk is 17.9 MiB less of all of that), a permute-free expert file, then P2P (0.3 s) |
| prefill pp512 | 110 | **123-129** | — | the same |
| decode at 2,048 ctx | 200 ms (5.0) | **~150-157 ms (6.4-6.7): ~5 ms from EP, near the noise and state-dependent; 149.6 with the CPU split alone, the two not yet combined** | 50 ms (20) | the dense body alone is ~46 ms (attention 30); the expert misses cost the slower card's share per layer; no single-stream path to 20 on this hardware — the honest ceiling with the levers left is ~9-11 tok/s |

Expert parallel is correct (bit-identical to the single-tier path) and worth +10-14% prefill and
~3-5% decode on its own; it moved the prefill wall from the one link to the host memory the disk
tier consumes, which is where the next phase goes.

