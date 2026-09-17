# V4.1 port — Phase 22 criteria: the decode hit rate, and how the VRAM slots are divided (written BEFORE the build)

**The term, measured in docs/60 step 0.** At the 2,048-token context the critical path carries **39.28 expert misses
per token at 1.33 ms each = 52 ms of the 82.7 ms/token**. Phase 21 established that *moving* those misses between the
cards can recover only 3.2 ms; what is left is to *not have them*. The decode hit rate is **68.9 %**: of the 240
expert selections a token makes (40 layers × top-6), 116.4 are served by a statically VRAM-resident expert, 48.9 by
an expert already sitting in a VRAM stream slot from an earlier token, and the remaining ~53.8 are fetched over the
links (963.7 MiB/token from the pinned host arena, 20.7 MiB from disk).

**The lever, and why it is free to test.** The VRAM budget is already an explicit zero-sum split
(`deepseek41_forward.cpp:143-156`): after the dense path, the state and the reserve, the card has `slots` expert
slots per layer, of which `stream` are evictable stream slots and **`n_static = slots − stream`** are permanent. One
more stream slot therefore costs exactly one static slot. The knob exists: **`IE_DS41_EP_STREAM=N`** under expert
parallel. So step 0 is a pure measurement — no code, no risk.

**Why the answer is not obvious, and what the numbers say a priori.** Per slot per layer-token, averaged over the
current placement: the 44 static slots (22 per card) yield 2.91 hits, i.e. **0.066 hits per slot**; the 16 stream
slots (8 per card) yield 1.22 hits, i.e. **0.076 hits per slot**. Stream slots look *better* at the average — but a
static slot holds a permanently hot expert, so its MARGINAL slot (the 22nd hottest) is worth much less than its
average, and the same is true of stream slots as the reuse window lengthens. docs/48's P0 table measures the reuse
that stream slots live on: the union of selections over 1 / 2 / 3 / 4 consecutive steps is 6.00 / 10.18 / 13.92 /
17.05 per layer, so roughly 7 of 24 selections over four steps are repeats — locality exists and a longer window
should capture more of it. The prior is therefore "more stream, less static", and the measurement decides.

**What was measured before, and why it does not settle this.** docs/39 compared 4 + 4 against 8 stream slots and
found 8 better on both axes (pp2048 308-319 against 300, decode 149 against 151 ms/token). That was the *other*
direction, at a build whose decode was 150 ms/token with a different static count and no CPU miss split. It says
nothing about 12, 16 or 24.

## Step 0 — the sweep (no code change)

`IE_DS41_EP_STREAM` ∈ **{8 (today's default), 12, 16, 24}**, each with `ie-ds41-decode-test` and the held-out
ranking at the 2,048-token context (`IE_DS41_BENCH_ONLY=long` so a run is one bench block), ≥ 2 samples per setting.
Per setting, report from the bench's own lines: ms/token and tok/s, the static / pinned / mmap selection split, the
stream-slot hits, the decode hit rate, MiB/token from the pinned tier and from disk, the critical misses per token
and the ms per critical miss, and the resulting static slots per layer (`card_info`'s `n_static`).

> **DECISION RULE, fixed now.** A setting replaces the default only if it (a) beats the 8-slot default by **≥ 3
> ms/token** at the 2,048-token context on ≥ 2 samples with non-overlapping spreads, AND (b) does not cost more than
> **2 % of pp2048** — the founder's 400+ prefill mandate is currently met at 403.9 tok/s warm and must stay met, and
> stream slots are what the prefill's fills land in. If (a) holds but (b) fails, the setting is reported as a
> decode-only option behind the existing env knob and the default does not move. If nothing clears (a), Phase 22
> stops at step 0 and records the slot division as measured-and-settled.

## Step 1 — only if step 0 clears the rule

Change the default `stream_slots` (a one-line change in `ResidentOptions`, `deepseek41_forward.hpp:100`), keep
`IE_DS41_EP_STREAM` as the kill switch, and re-run the regression set. No other change is in scope.

## Pass criteria for THIS phase

1. **The sweep reported** as above, with the decision rule applied in writing.
2. **Bit-identity across the sweep**: the slot division changes only WHICH experts are cached where, never any
   arithmetic, so the decode test's forced-routing digits must be identical at every setting (they are the same
   comparison Phase 19 used). A setting whose digits move is a defect, not a result.
3. **pp2048 for any setting that clears (a)**, from `ie-ds41-resident-test` with the ranking, against the current
   403.9 tok/s warm / 372.9 cold.
4. **The hit-rate arithmetic checked, not assumed**: static hits + stream hits + fetches = 240 selections per token
   at every setting, and the reported hit rate equals (static + stream hits) / 240.
5. **If the default moves**: the regression set on the new default — decode, resident, replay, forward, generate,
   multi (`IE_DS41_CPU_MISS=0`), rollback, dspark — and the kill switch restoring today's numbers.
6. **Reported, not judged**: what the sweep implies about the ceiling. If N stream slots per layer capture the whole
   4-step reuse window, the remaining misses are cold first-touches and the hit rate has a hard roof that this
   division cannot pass; say where that roof is from the measured union table.

## Explicitly NOT in this phase

More pinned host RAM (the disk term is already 20.7 MiB/token — there is nothing there to win), a different ranking,
prefetch (a measured dead end on DS4), the expert file's layout, DSpark (closed), and cross-card miss movement
(refused in docs/60).

## Step 0 results (16:20-16:45, `ie-ds41-decode-test` with the held-out ranking, `IE_DS41_BENCH_ONLY=long`; logs `~/ds41_work/p22/`)

**PHASE 22 STOPS AT STEP 0: the slot division is settled at 8. Nothing came within 3 ms/token of the default, and
the sweep's own numbers explain why the hit rate is the wrong objective.**

| `IE_DS41_EP_STREAM` | static slots /layer/card | ms/token (samples) | tok/s | static / pinned / mmap selections | stream hits | hit rate | pinned MiB/token | **disk MiB/token** | critical misses |
|---|---|---|---|---|---|---|---|---|---|
| **8 (default)** | 22 | **80.5, 81.6** | **12.42, 12.25** | 116.4 / 102.7 / 1.2 | 48.9 | 68.9 % | 963.7 | **20.7** | 39.28 |
| 12 | 18 | 81.7 | 12.24 | 105.2 / 114.0 / 2.1 | 60.6 | 69.4 % | 957.0 | 37.5 | — |
| 16 | 14 | 81.2, 81.2 | 12.31, 12.31 | 92.1 / 127.1 / 2.5 | 75.5 | 69.8 % | 925.1 | 44.8 | 37.94 |
| 24 | 9 | 86.0 | 11.62 | 61.2 / 157.9 / 4.2 | 107.0 | 70.1 % | 912.7 | **75.1** | 37.62 |

**The decision rule, applied.** The best non-default setting (16) reads 81.2 twice against the default's 80.5 and
81.6 — overlapping, and 0.4 ms apart at best, against a 3 ms bar. 24 is 5.4 ms WORSE. Nothing clears (a), so
criterion 3's prefill check was never triggered and the default `stream_slots = 8` stands with no code change.

**Why the hit rate rose while the speed did not — the real finding.** The hit rate is monotone in stream slots
(68.9 → 69.4 → 69.8 → 70.1 %) and the critical misses do fall (39.28 → 37.94 → 37.62), yet ms/token is flat then
worse. The cause is visible in the last two columns: **the pinned tier is capped by host RAM, not by the static
count.** `n_pinned = min(experts_here − n_static, host_cap / per_slot_all_layers)`
(`deepseek41_forward.cpp:157`) and the second term binds — 200.5 GB of pinned cap ÷ 2 cards ÷ (40 layers × 18.8 MB)
= **133 slots per layer per card**, exactly what the runtime reports. So giving up a static slot does NOT promote an
expert into the pinned tier; it drops one out of the fast tiers entirely and onto **disk**, which is why disk traffic
more than tripled (20.7 → 75.1 MiB/token) and the marginal cost per critical miss rose with it (1.28 → 1.32 → 1.49
ms). Static and pinned together cover 22 + 133 = 155 of the 192 experts a card owns per layer; the other 37 are cold
by construction.

**The roof, for criterion 6.** Full coverage would need 192 experts × 40 layers × 18.8 MB × 2 cards = **288 GB** of
pinned host RAM against 267 GB installed (200.5 GB usable under the live rule), so **81 % coverage is the hardware's
ceiling** and no slot division reaches past it. Stream slots do capture reuse — 8 slots hold 48.9 of the 240
selections and 24 slots hold 107.0, tracking docs/48's P0 union table — but every slot they gain is taken from the
tier that has no host copy to fall back on.

**Criterion 2 is FALSIFIED, and the mechanism is identified.** I predicted the division "changes only WHICH experts
are cached where, never any arithmetic". It does change the arithmetic: the forced-routing logits read **5.043e-3**
at 8 slots and **8.047e-3** at 16 (both far under the 5e-2 bar, both PASS 28 / 28). The cause is not the CPU miss
split — at 8 slots, turning the split off moves the same digit only 5.043e-3 → 4.926e-3 (gate_p2's two runs), a
25× smaller effect. It is the **mmap tier's separate group**: mmap experts are repacked into a transient bank and
their GEMMs run after the resident groups, so a different accumulation grouping, and their share grows exactly as
static shrinks (1.2 → 2.1 → 2.5 → 4.2 selections per token). A slot division is therefore a numerics change, not a
pure caching change — which is a rule worth carrying into any future residency work.

**The campaign's best decode number, from this sweep:** **80.5 ms/token = 12.42 tok/s** at the 2,048-token context on
the shipping default (`p22/stream08_b.log`).

**What the measurement says to look at next, and it is not residency.** Of the 1.28-1.49 ms a critical miss costs,
only **0.71 ms is the transfer itself** (18.8 MB at the 26.5 GB/s this box measures), so **roughly 0.6 ms per miss —
about 23 ms/token — is not bandwidth**: per-fetch setup, the per-layer serialisation, and the fact that a layer's
fetch cannot start before its router has run. That is now the largest identified term that is neither hardware nor
already refused, and it is a latency/overlap question rather than a placement one.
