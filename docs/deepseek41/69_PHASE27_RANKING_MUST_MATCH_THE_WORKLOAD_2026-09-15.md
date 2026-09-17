# V4.1 port — Phase 27: a residency ranking must match the WORKLOAD, not just the phase and the length

## This document corrects docs/67, and the correction matters more than the original result

docs/67 reported that a decode-phase ranking profiled at ctx 32,768 took decode there from 4.44 to 12.01 tok/s,
and I wrote that "the ranking WAS the long-context bottleneck". **That measurement was real and its mechanism was
real, but it was taken on a SYNTHETIC workload and it does not generalise.** The prompt was the golden 2,048-token
block repeated to length, decoded greedily — which produces a highly repetitive token stream that routes to a
narrow set of experts. A ranking profiled on that trajectory is excellent for that trajectory and **a regression
on real text.**

Measured on one identical real-text workload (`needle_32k_long.txt`, 23,355 prompt tokens, **300** decode tokens,
`ie-ds41-run --temp 0`):

| the ranking was profiled on | decode tok/s | against the shipped ranking |
|---|---|---|
| the prefill corpus (shipped `ie_ranking_heldout.txt`) | **5.46** | — |
| **synthetic** decode at ctx 32,768 (docs/67's arm B) | **3.45** | **0.63x — WORSE THAN DOING NOTHING** |
| **real text** decode at ctx 23,355 | **13.17** | **2.41x** |

So the *machinery* is worth 2.41x and the conclusion "residency is the long-context bottleneck" stands. What was
wrong was the ranking I built, and the claim that docs/67's 12.01 tok/s described anything a user would see.

**Why all three differ so much, measured from the files alone with no run involved** (`p24/rankdiff.py`, mean
overlap of each layer's top-22 — the static VRAM slots per layer per card — over 40 layers):

| pair | overlap at top-22 | min |
|---|---|---|
| shipped (prefill) vs synthetic decode | 20.5 % | 0.0 % |
| **real-text decode vs synthetic decode** | **13.6 %** | 0.0 % |
| **real-text decode vs shipped (prefill)** | **11.5 %** | 0.0 % |

The three rankings are very nearly disjoint where it counts. They are not three approximations of one truth; they
describe three different workloads.

## The rule this establishes, and why nothing caught it

`expert_stream.hpp:707-717` already records the PHASE half of this, measured on this engine: a blended histogram
is 90 % a prefill histogram, and a ranking built from it scored 31.16 % static hit against 31.47 % for plain index
order — "worse than doing nothing". docs/67 added the LENGTH half: a 32,768-token ranking used at ctx 8,192 reads
**185.5 ms/token against the shipped ranking's 106.1**, its `join` term exploding to 85.9 ms, so the mismatch hurts
in BOTH directions. This phase adds the third and largest axis: **the token DISTRIBUTION.**

**A residency ranking is only valid for the (phase, context length, workload) it was profiled on**, and the
`ds4-expert-priority` file format records **none of the three**. That is why a prefill-corpus ranking could be
shipped for decode, why I could then ship a synthetic one, and why neither the loader nor any test complained.
Every ranking this engine writes now names its phase, its context and its token count in its header — necessary,
but a convention, not a guard.

## What was built

`ie-ds41-run --profile-out FILE` writes a decode-phase ranking from whatever real prompt it was given, using
`Ds41Generator::set_profile_decode_only(true)` to reset the routing profile at the exact prefill/decode boundary
(`deepseek41_generate.cpp`, where `st.prefill_s` is taken). No engine change beyond that switch: `reset_profile()`
and `write_profile()` already existed. So the recipe for a correct ranking is now one command on a representative
prompt, and the header says what it is valid for.

## The honest scoreboard for the founder's goal (200k ctx, 10+ decode)

| context | workload | decode | ranking used |
|---|---|---|---|
| 2,048 | golden | 12.42 tok/s | shipped |
| 23,355 | **real text, 300 steps** | **13.17 tok/s** | **real-text, 23k** |
| 23,355 | real text, 300 steps | 5.46 tok/s | shipped |
| 32,768 | synthetic | 11.83 tok/s | synthetic 32k |
| 262,144 | synthetic | 5.80 tok/s | synthetic 262k + gathered attention |

**The decode bar is met at 23,355 tokens on real text.** The context bar (200k) is met for PREFILL — 262,144
tokens prefill and the needle at 50 % depth is retrieved. What is not yet measured is decode at ~200k with a
ranking profiled there on real text, which is the run in flight: a 209,349-token real-text prompt, profiled with
the shipped ranking, then re-measured with its own.

## The three axes, separated and weighed — at the TRUE head size

The overlap table above used top-22, which was docs/61's static-slot count measured with **expert parallel ON**.
The runs in this phase have EP off (the shipping default, docs/66), and the runtime prints what it actually
allocated: **47 static + 8 stream slots per layer per card, 266 pinned**. So 47 is the head that decides the
static hit rate, and the comparison belongs there. Re-measured over 40 layers:

| what differs between the two rankings | overlap at top-47 | min | so the static set moves by |
|---|---|---|---|
| **workload** — real text vs the shipped prefill corpus | **22.3 %** | 12.8 % | ~78 % |
| **workload** — real text vs the synthetic repeated block | **21.1 %** | 6.4 % | ~79 % |
| **length only** — synthetic at 32,768 vs synthetic at 262,144 (8x) | **73.7 %** | 46.8 % | **~26 %** |

**The distribution axis dominates the length axis by about 3x.** Changing what the model is reading replaces four
fifths of the experts that belong in VRAM; changing the context length eightfold replaces one quarter. That is the
quantitative form of this phase's correction, and it also predicts something useful: **a ranking profiled on
representative text at ONE length should largely transfer to another length**, so the practical recipe is to
profile on real traffic rather than to profile per length.

It also re-reads docs/67's length finding correctly. A 32,768-token ranking used at ctx 8,192 measured 185.5
ms/token against the shipped ranking's 106.1 — I attributed that to the length mismatch, but both of those
rankings differ from the 8,192-token truth on the WORKLOAD axis too (one is synthetic, one is a prefill corpus),
and the table above says that axis is the bigger one. **The clean length-only experiment is the 73.7 % row**, and
it says length alone is the mild axis.

## HELD OUT, and the in-sample number was indeed inflated

The 13.17 tok/s above was profiled AND measured on the same prompt, which is in-sample and flatters itself — the
exact trap `ie_ranking_heldout.txt` is named after. `make_needle2.py` generates a second document sharing nothing
with the first but its shape: a different domain (a survey vessel's log, not a maintenance archive), different
sentence templates, a different planted fact (`QX-8814-teal`) and a different question. Profiling on document A
and measuring on document B is the honest test.

**Held out, 24,193 prompt tokens, 300 decode steps, both arms retrieving their needle:**

| the ranking was profiled on | decode on the HELD-OUT document |
|---|---|
| the prefill corpus (shipped) | **5.28 tok/s** |
| **real text — a DIFFERENT document** | **10.15 tok/s — 1.92x** |

So the in-sample figure overstated it by about a quarter (13.17 -> 10.15), which is worth knowing and is why the
test was run; but **the effect is real, generalises across documents, and 10.15 tok/s clears the founder's 10+
bar at 24k.** The corrected scoreboard for the ranking axis, all on real text:

| ranking | decode | note |
|---|---|---|
| synthetic repeated block | 3.45 | **worse than the shipped ranking** |
| shipped (prefill corpus) | 5.28-5.46 | the honest baseline |
| real text, in-sample | 13.17 | an upper bound, not a deliverable |
| **real text, HELD OUT** | **10.15** | **the number to quote** |

**And length transfers, predicted before it was measured.** The real-text rankings profiled at 23,355 and at
214,594 tokens overlap **83.2 %** at top-47 (min 74.5 %), against 20.9 % between the 214,594-token ranking and the
shipped one. So the residency a long context wants is nearly the residency a short one wants, PROVIDED both are
profiled on real text — which makes the practical recipe cheap: **profile once on a representative prompt, at any
convenient length, and use it everywhere.** Profiling per length is not necessary; profiling on representative
TEXT is.

## Criterion 4 of the founder's goal, at 200k on real text

`ie-ds41-run --prompt-file` on a 214,594-token real-text prompt, chunked by the generator itself: **the needle at
50 % depth was retrieved** and prefill ran 2000.8 s = 107 tok/s. Decode there with the SHIPPED ranking is **3.16
tok/s** — the baseline the goal has to beat. The decisive measurement now in flight is the strongest form
available: a ranking profiled on document A at 23k, measured on document **B** at ~194k, so it is held out on
BOTH the document and the length at once.
