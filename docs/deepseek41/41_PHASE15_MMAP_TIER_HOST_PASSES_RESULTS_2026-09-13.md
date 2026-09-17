# V4.1 port — Phase 15 results: the mmap tier's host-memory passes (C0 permute, C1 live cap, C2 expert file)

**Criteria:** `40_PHASE15_MMAP_TIER_HOST_PASSES_CRITERIA_2026-09-13.md` (with the peer's pre-reads and
the step-0 results recorded there). **Starting line:** edcf991 + the gate 14 fixes — pp2048 308-319
tok/s, pp512 123-129, decode ~150-157 ms/token with `IE_DS41_EP=1` at the 160 GiB pinned cap.

## Step 0 (docs/40): the disk ~12 GB/s whatever the reader count; O_DIRECT into pinned host USM refused
(EFAULT), into imported anonymous memory accepted at 11 GB/s with the DMA intact; the permute
instruction-bound (3.95 ms per slot on one core vs memcpy 0.87).

## C0 — the vectorised permute (`permute_plane`, SSE2; the scalar kept as the reference)

Byte-identical to the reference on 9 random planes (`ds41_permute_plane_selftest`, run by the tier
test). Micro-bench: one thread 3.95 → **1.72 ms per slot**, eight threads 8.23 → **6.82** (44 GB/s of
traffic against memcpy's 64). In the tier: the reader's permute per pp2048 pass **1.13 → 0.82 s per
reader** (EP on, 8 readers). No numerics changed: every digit equals the previous build's.

## C1 — the live pinned cap with a 40 GiB free-after-pin floor

The rule gave 196.8 GiB (MemTotal 267.38 GB, MemAvailable 230.24 GB); the floor bound it at
**174.4 GiB** (MemAvailable − 40 GiB), i.e. **124 pinned per layer per card** (from 114), 86.8 GiB
per card, 173.7 GiB in total; **MemAvailable during the run 38.9 GiB** (the peer's prediction ~37:
the floor holds at the pin and the engine's own unpinned footprint follows). With `IE_DS41_EP=1`,
one run each (the resident test / the decode bench):

| | 160 GiB cap (docs/39) | live cap 174.4 GiB |
|---|---|---|
| mmap experts / bytes per pp2048 pass | 1,824 / 31.9 GiB | **1,416 / 24.8 GiB** |
| the reader's fill leg per pass (reads + permute per reader) | 3,993 ms (2,070 + 1,129) | **3,354 ms (1,706 + 818)** |
| pp2048 cold / warm | 307.9 / 319.3 tok/s | 276.1 / **329.9** |
| pp512 cold / warm | 123.2 / 128.3 | 114.8 / 133.4 |
| decode, 2,048 ctx, steps 4-35 | 149-157 ms/token | **138.5 ms/token (7.22 tok/s)**, disk 337 → 220 MiB/token |

The warm prefill and the decode improve as the arithmetic said; the COLD pass got slower (7.42 s
against a 6.21 s warm pass, a gap that was 0.24 s at the 160 cap) — something the first pass
pays at the tighter memory, not yet understood ("Pineapple": a page-cache reclaim under the
39 GiB available is the candidate; a second run at the cap and the pass breakdown decide).
The load-then-build survival check of the criteria: not run (49-51 GiB available in every run at the cap on the fresh boot; the case is open for the gate).

## The interruption, and the re-baseline

At 22:29 VS Code's own process died ("OOM error in V8: JavaScript heap out of memory" — its
JavaScript heap, not the machine: no oomd, kernel or GPU fault, no engine run alive), the founder
rebooted at 22:43, and the reboot wiped the /tmp scratchpad that held every golden and every log
of the day. The goldens were regenerated from the reference scripts into a persistent place
beside the model (`ie_golden/`; `golden_model` 111 s), and the pp / corpus token files — which
never had a generator in the repository — are now made by `ie-ds41-text-ids` from the engine's
own docs (pp2048 from two V4.1 docs, the 65,536-token corpus from 43 other docs, disjoint). The
pp text therefore CHANGED at 22:49: every prefill and decode number from here on is on the new
text and is compared only with numbers measured on it (the `rebaseline` chain: off and EP at
the 160 GiB cap first, then C1, then C2). The numbers above (C0 in the tier, C1) were measured
on the old text before the crash and are kept as such.

## C2 — the expert file (the tail window in slot layout, one direct pread into imported staging)

All on the new pp text, one build (fdeafad + the text-ids tool), the re-baseline chain
(`~/ds41_work/rebaseline.sh`, logs beside it), each arm one run; the block time in all;
EP = `IE_DS41_EP=1`; the file = `IE_DS41_EXPERT_FILE=<model>/ie_experts_tail.ieslot`, written by
`ie-ds41-expert-file --cutoff 297 --end 341 --confirm` in **9.6 s** (33.1 GB, 1,760 slots, 3.45
GB/s; the system volume went 141 → 108 GB free, the floor is 50). At init the tier reports
**"covers 800 of this tier's 1640 / 1680 mmap experts"** (EP: each tier's half of the tail) and
**"one slot verified byte for byte"**; every forced digit and the decode test's digits equal the off
path's (card 0 9.355e-3 / card 1 1.486e-2; resident 9.024e-3 / 2.755e-2 / logits 4.195e-3) — the
bytes are the same bytes.

**Prefill, pp2048 cold / warm tok/s** (the resident test) and the cold pass's MoE breakdown:

| arm | pinned per layer per card | mmap experts / GiB per pass | groups | fill leg (reads + permute per reader) | pp512 | **pp2048** |
|---|---|---|---|---|---|---|
| off, 160 GiB | 228 (one tier) | 1,296 / 22.7 | 5,535 | 3,570 (1,833 + 1,090) | 102.3 / 104.2 | 285.1 / 285.9 |
| EP, 160 GiB | 114 | 1,499 / 26.2 | 4,111 | 3,591 (1,915 + 879) | 127.4 / 132.0 | 324.6 / 324.1 |
| EP, live cap (C1) | **133** | 830 / 14.5 | 3,665 | 2,119 (1,010 + 492) | 142.4 / 147.3 | 358.7 / 364.0 |
| **EP, live cap + the file (C1 + C2)** | 133 | 830 / 14.5 | **3,292** | **1,970 (808 + 125)** | 137.9 / **158.7** | 340.1 / **388.9** |
| off, live cap + the file | 266 (one tier) | 682 / 11.9 | 5,544 | 1,928 (970 + 143) | 103.2 / 105.3 | 286.3 / 289.1 |

What the rows say. C1 moved 4.7 GiB per pass off the disk and cut the fill leg 3.6 → 2.1 s; C2
took the permute from 492 → **125 ms per reader** (the file serves the hot half of the tail) and
the reads 1,010 → 808, and the groups — the two links' stream — fell 3,665 → 3,292 ms as the host
memory freed: the mechanism docs/40 named (fewer host passes → faster links), measured. The
warm pass lands at **388.9 tok/s** against the 400 goal, the cold at 340: the first pass after
load pays ~0.75 s with the file that it does not pay without it (6.02 vs 5.71 cold; the warm
passes 5.27 vs 5.63) — not understood ("Pineapple": the imported staging's first DMA mappings, or
the file's first O_DIRECT extents; a second cold pass decides) — gate 15 ran it twice: cold = warm, 390.9 / 394.3 and 390.9 / 395.1; the 6.02 s pass was a one-run outlier. Without EP the file changes nothing
(286 → 286): the single-tier wall is the one link, not the mmap tier, as docs/39 said.

**Decode, 2,048-token context, steps 4-35** (the decode bench; this text routes to far fewer disk
experts per token than the old one — 4.9 vs 15.3 at the 160 cap — so its decode is faster
everywhere and is compared only within this table):

| arm | ms/token | tok/s | link MiB/token | disk MiB/token (experts) |
|---|---|---|---|---|
| off, 160 GiB | 129.5 | 7.72 | 1,451 | 88 (4.9) |
| EP, 160 GiB | 115.7 | 8.65 | 1,301 | 110 (6.2) |
| off, live cap | 127.2 | 7.86 | 1,517 | 24 (1.3) |
| **EP, live cap** | **111.3** | **8.99** | 1,373 | 36 (2.0) |
| EP, live cap + the file | 111.5 | 8.97 | 1,373 | 36 (2.0) |

EP is worth 11-12% at decode on this text (the slower card serves 154 of 240 selections and 59
of 77 link misses per token — the E[max] geometry again), C1 another 4%, and the file nothing at
decode here (two disk experts per token leave it nothing to accelerate).

**Correctness with the file — gate 15's verdict on criterion 3 was FAIL, and the fault was the
re-baseline, not the engine:** the tier test, the resident test and the full decode test PASS with
the file on (EP and off) with the off path's digits, and the replay test PASS (KL mean 3.3e-3, max
1.7e-2 under 0.02 / 0.2; next-token agreement 22 of 24 prefixes on this text against 24 of 24 on
the old — finding 11), but the generate test FAILED in every arm on this build: it carried the OLD
text's continuation (369 2619 1683 16) as a hard-coded expectation, and its continuity check's
borrowed 5e-2 max-relative bar was exceeded at T = 130 (5.86e-2, the same digit in the Phase 14
arm and the Phase 15 arm — text-dependent, not introduced here). Fixed exactly as named: the
replay test now writes the exact path's four tokens (`replay_pp2048_greedy4.i32`) and the
generate test reads them; the continuity check is the replay test's KL criterion (< 0.02, the
same argmax) with the max-relative number printed beside it; re-presented to the same gate. The
gate also ran the survival check the builder had not: a 10 GiB consumer started after the pin at
the live cap (MemAvailable 51 → ~40 GiB), the run completed, oomd pressure 0.00; and the floor
branch — 30 GiB held before the load: cap 168.2 GiB with the floor clause printed, 120 per layer,
MemAvailable 208 → 38.0, PASS (the builder's fresh-boot runs never bound the floor: finding 7).


## The corrected control arm (gate 14 finding 1, closed here)

The empty tier the control arm runs for the other card now consumes no resources (no cache,
arena, workspace or staging; `moe()` returns at once) — the arm itself still carries EP's staging
buffers (160 MiB per card before the budget), which tip one static slot on card 1 (45 / 43
against off's 45 / 44: gate 15 finding 4), a cost EP = 1 pays too, so the control remains valid: decode 155.8 / 162.2 ms/token
against off 157-159 on the same build, pp2048 273.6 / 277.2 tok/s against off's 276.9 / 281.2,
no eviction (5.0 GiB of VRAM free) — those are the pre-crash numbers whose logs did not survive;
gate 15's post-reboot reproduction on the same text: pp2048 282.0 / 284.6 vs off 286.1 / 287.0,
decode 129.0 vs 129.2 ms/token. So the machinery costs ~1% of a prefill pass and nothing
measurable at decode, and EP's gain is the parallelism. The two earlier shapes are recorded in
docs/39: inside the budget (5 static slots per layer lost) and after it with one slot per layer
(0.7 GiB of the reserve, eviction in the warm pass) — a control must consume no resources.


## The founder-facing statement

Against 400+ prefill / 20+ decode, on the new pp text, EP + the live cap + the expert file:

| | Phase 14 (EP at 160, this text) | now (C1 + C2, `IE_DS41_EP=1 IE_DS41_EXPERT_FILE=...`) | the goal | what remains |
|---|---|---|---|---|
| prefill pp2048 warm / cold | 324 / 325 | **389-395 / 340-395; gate 15's re-check on 09b11ef: 409 / 407** | 400+ | the cold pass's 0.75 s was a one-run outlier (gate 15 reproduced cold = warm twice); the tail window covers half the mmap experts (a bigger window when the volume allows, or the C1 cap moved by the founder's RAM choice); P2P for the 0.3-0.4 s of staging; then the links' own rate under the remaining host traffic |
| prefill pp512 | 127 / 132 | **158 / 138** | — | the same |
| decode at 2,048 ctx | 115.7 ms (8.65) | **111.3 ms (8.99)** | 50 ms (20) | the dense body (~46 ms: attention 30); the slower card's share per layer; the CPU split not yet combined with EP (its cores); no single-stream path to 20 |

Levers now on by default: the OpenMP block time (Phase 13), the live pinned cap with the 40 GiB
floor (C1), the vectorised permute (C0). Opt-in, measured: expert parallel (`IE_DS41_EP=1`,
bit-identical), the expert file (`IE_DS41_EXPERT_FILE`), the CPU miss split
(`IE_DS41_CPU_MISS=1`, not with EP). The decision on defaults for EP and the file is the
founder's: EP costs nothing in correctness and 12-14% of prefill / 11% of decode is on the
table; the file costs 33 GB of the system volume.

