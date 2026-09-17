# V4.1 port — Phase 25 criteria: a residency ranking for the phase AND THE LENGTH it is judged at (written BEFORE the build)

**What docs/66 measured.** Decode degrades with context and 85 % of it is one term: the MoE call grows 49.2 ->
101.0 ms/token between ctx 8,192 and 32,768, and inside it the `join` — the wait for the CPU miss leg — grows 9.2
-> 46.5. The cause is residency: **expert selections served from static VRAM halve (136.6 -> 76.4), the decode hit
rate falls 63.7 % -> 43.7 %, and disk traffic rises 6.2x (35.9 -> 222.3 MiB/token).** Two candidate causes were
eliminated by control: the KV allocation (the same 8,192 tokens under a 8,256 vs 262,208 capacity costs 91.4 vs
94.0 ms/token, 2.8 %) and my synthetic repeated prompt (23,320 tokens of real text decode at 5.68 tok/s, slightly
worse). What is left is that a longer context changes WHICH experts the router picks.

**And the shipped ranking cannot know that, by its own provenance.** `ie_ranking_heldout.txt` line 2 reads
`V4.1 held-out own-router profile: ... corpus_ids_65536.i32, 65536 tokens` — it was built from a **prefill** sweep.
`Ds41Forward::profile_` counts every own-router forward into one histogram with **no phase split**
(`deepseek41_forward.cpp:1159`), so a profile taken over a prefill-dominated run IS a prefill histogram.
`expert_stream.hpp:707-717` already records what that costs, measured on this engine one level shallower:

> "a `--ctx 4096 --decode 512` run records 1,155,840 prefill selections against 132,612 decode ones, so a blended
> histogram is 90 % a prefill histogram. MEASURED consequence on the real model: the ranking derived from the
> blended counts scores 31.16 % static hit on decode, against 31.47 % for plain INDEX ORDER. **It is worse than
> doing nothing**, and nothing about the file it is written to says so."

**So this phase is that same lesson one level deeper: a ranking must match the PHASE and the LENGTH it will serve.**
Nothing in the file format or the loader records either, which is why this was invisible.

## The build, and it needs no engine change

`reset_profile()` already zeroes the counts and `write_profile()` already emits the per-layer
`ds4-expert-priority 1` format. So a pure decode-phase, long-context ranking is: prefill to N, **reset the
profile**, run many decode steps, write. The only change is to `ie-ds41-long-test`: an optional profile-out path
and a decode-step count large enough to rank with.

**Sample-size arithmetic, stated before the run so the result is not over-read.** Each decode step makes
`40 layers x 6` = 240 selections, so S steps give `6 x S` selections per layer against 384 experts. docs' own
finding is that each layer's top ~80 carries 83.8-95.8 % of its decode selections, so the quantity that must be
resolved is the top ~80 of 384, not all 384. At **S = 512** that is 3,072 selections per layer, ~38 per top-80
expert — thin but enough to separate the head from the tail. Anything below S = 256 is reported as
under-sampled rather than trusted.

## The measurement

Three arms, same binary, same build, at **ctx 32,768** (cheap: ~200 s prefill + ~80 s decode each) and reported
with the docs/66 stage breakdown so the MECHANISM is visible and not just the wall:

| arm | ranking |
|---|---|
| **A** | today's `ie_ranking_heldout.txt` (prefill-derived, 65,536-token sweep) |
| **B** | a decode-phase ranking profiled at ctx 32,768 |
| **C** | **INDEX ORDER** (no ranking file) — the control `expert_stream.hpp` demands, because a bad ranking measured worse than none |

> **DECISION RULE, fixed now.** B replaces the default only if it (a) improves decode ms/token at ctx 32,768 by
> **>= 10 %** against arm A on **>= 2 samples with non-overlapping spreads**, (b) **beats arm C** — if index order
> is as good, the ranking machinery is not earning its keep at this length and that is the finding, and (c) costs
> no more than **2 % of pp2048**, since the 400+ prefill mandate is stated there. If B improves the decode hit rate
> but NOT ms/token, that is reported as a falsification of the causal chain in docs/66, not buried.

## Pass criteria for THIS phase

1. **The three arms measured** as above, decision rule applied in writing, with the stage breakdown (MoE call,
   `join`, static/pinned/mmap/cpu selections, hit rate, pinned and disk MiB/token) for each.
2. **The mechanism confirmed or denied numerically**: if B works, the static selections and the hit rate must rise
   and `join` and disk MiB/token must fall. A speed-up without those moving means the cause is something else and
   the improvement is luck.
3. **Sample size reported**, per the arithmetic above, and any arm under S = 256 flagged as under-sampled.
4. **Provenance written into the file**: the new ranking's header must name the phase (decode) and the context it
   was profiled at, so the next person cannot repeat this mistake by reading the file.
5. **Correctness untouched**: a ranking changes only WHICH experts sit in which tier. Per docs/61 that is NOT
   arithmetic-neutral (the mmap tier's separate group changes the accumulation grouping — 5.043e-3 vs 8.047e-3
   there), so the bar is the existing one: the decode test's 28 criteria pass with the forced-routing logits under
   5e-2, not bit-identity.
6. **If B wins at 32,768, it is then profiled and re-measured at 262,144** — the length the founder actually asked
   for — because this phase's whole point is that a ranking does not transfer across lengths, and it would be
   absurd to prove that and then ship a 32,768-token ranking for a 250k context.
7. **The GPU survives**: zero `xe ... Timedout job` / reset lines across the phase.

## Explicitly NOT in this phase

The `ds4_indexer_topk` barrier cost (measured, exactly linear in NC, and only 3.5 ms of a 53 ms regression — it is
real and it is small, so it is a later phase), gather attention, concurrency, and the slot division (docs/61
settled it at 8 for short context; whether it moves at long context is a separate question this phase does not
touch, so that the ranking is the only variable).

## Results at ctx 32,768 (512 decode steps per arm, same binary, same build): the ranking WAS the bottleneck

| | **C** index order | **A** shipped (65,536-token PREFILL sweep) | **B** decode-phase profiled at 32,768 |
|---|---|---|---|
| **decode ms/token** | **519.9** | **225.3** | **83.7 / 83.3** (two samples) |
| **tok/s** | **1.92** | **4.44** | **11.95 / 12.01** |
| layers: attention | 37.4 | 37.9 | 35.3 |
| **layers: MoE call** | **472.9** | **178.1** | **39.9** |
| .. groups | 36.8 | 38.3 | 27.0 |
| .. **join** (the wait for the CPU miss leg) | **370.7** | **102.2** | **3.4** |
| .. tail | 55.6 | 34.1 | 8.6 |
| CPU leg to the join | 457.0 | 172.8 | 35.4 |
| **expert selections from static VRAM** | 33.4 | 57.8 | **162.5** |
| .. from pinned host | 85.8 | 84.8 | 30.8 |
| .. from **disk** | 39.3 | 14.8 | **0.4** |
| .. on the CPU leg | 81.6 | 82.7 | 46.4 |
| **decode hit rate** | **32.2 %** | **41.7 %** | **71.5 %** |
| pinned MiB/token | 751.2 | 763.4 | **388.8** |
| **disk MiB/token** | **703.8** | **264.6** | **6.5** |
| prefill tok/s (same run) | 144.8 | ~150 | **165.5** |

**The decision rule, applied.** (a) B improves decode by **62.9 %** against A, on two samples reading 83.3 and
83.7 ms/token against A's 225.3 — spreads nowhere near overlapping, against a 10 % bar. (b) B beats index order by
**6.2x**, so the ranking machinery earns its keep at this length; note also that A beats C by 2.0x, so the shipped
prefill ranking was helping — it was simply leaving most of the win on the table. (c) prefill did not cost
anything: it IMPROVED, 144.8 -> 165.5 tok/s, because the same residency serves both phases here.
**B passes on every clause.**

**Criterion 2 satisfied — the mechanism moved, all four terms, in the predicted direction:** static selections
57.8 -> 162.5 (2.8x), hit rate 41.7 -> 71.5 %, `join` 102.2 -> 3.4 ms (30x less), disk 264.6 -> 6.5 MiB/token
(40x less). So the causal chain in docs/66 is confirmed rather than merely consistent: residency -> misses ->
disk -> the CPU leg's join -> the wall.

**Criterion 3, sample size:** 512 decode steps = 3,072 selections per layer, above the 256-step floor docs/67 set
before the run. The first attempt produced only 64 steps — the capacity was `target + 64`, leaving no room to
decode — and the tool FLAGGED it as under-sampled in both the log and the ranking's own header rather than
silently writing a thin profile. That guard did its job on its first outing.

**Criterion 4, provenance:** the new ranking's header reads
`V4.1 DECODE-PHASE profile at a 32768-token context: 512 decode steps, 122880 selections, 3072 per layer`, so the
phase and the length are recorded in the file. The shipped ranking's header says only
`corpus_ids_65536.i32, 65536 tokens` with no phase and no decode context — which is exactly how a prefill
histogram came to be chosen for decode residency without anyone noticing.

**And the founder's bar is MET at this length: 11.95-12.01 tok/s at a 32,768-token context**, against "500k ctx and
10+ decode is at least useable". At 2,048 tokens the campaign's best is 12.42 tok/s, so long-context decode is now
within 4 % of short-context decode where it was 3.7x worse.

### The mechanism confirmed WITHOUT a stopwatch: the two rankings disagree about 79 % of VRAM

`p24/rankdiff.py` compares two per-layer rankings at the only place residency cares about — the head. Each layer
gets 22 static VRAM slots per card here, so what decides the decode hit rate is which experts land in a layer's
top-22, not the rest of the permutation. Shipped (prefill, 65,536-token sweep) against decode-phase at 32,768,
over all 40 layers:

| top-N | mean overlap | min | max |
|---|---|---|---|
| 8 | 18.1 % | **0.0 %** | 62.5 % |
| **22 — the static slots per layer per card** | **20.5 %** | **0.0 %** | 59.1 % |
| 44 | 24.9 % | 6.8 % | 63.6 % |
| 80 | 36.5 % | 16.2 % | 68.8 % |
| 133 | 52.1 % | 30.8 % | 78.9 % |

**Roughly four fifths of the experts held in VRAM were the wrong ones for decode, and on some layers none of them
overlapped at all.** This is independent evidence for the same conclusion as the timing: it is measured from the
files alone, with no run involved, so the 41.7 -> 71.5 % hit-rate change is explained by WHAT is resident rather
than by anything incidental to a particular run. The same tool gives a cheap transfer predictor: comparing a
32,768-token ranking against a 262,144-token one at top-22 says whether a single ranking can serve both lengths
before spending 46 minutes finding out.
