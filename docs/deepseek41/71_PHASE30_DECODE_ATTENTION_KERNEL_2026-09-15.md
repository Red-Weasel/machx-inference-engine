# V4.1 — Phase 30: the decode attention kernel, 13.12 -> 7.29 ms/token, and what that does NOT buy

## The win: partition segment b by PICK INDEX

The Phase 26 gathered path had each of the SP slices scan the WHOLE pick list to find the ~`n_picks / SP`
entries on its own `i % SP == s` stride — SP redundant scans of 512 entries to do ~16 columns of work each.
Giving each slice a CONTIGUOUS RUN of the pick list needs no scan at all.

| `ds4_attention` at ctx 32,768, decode | ms/token | avg µs |
|---|---|---|
| dense stride scan (pre-Phase 26) | 13.27 | 331 |
| gathered, per-slice SCAN of all picks | 13.12 | 328 |
| **gathered, partitioned by pick index** | **7.29** | **182** |

**-5.8 ms/token, 44 % of the kernel**, and named kernels 53.4 -> 47.6. Behaviourally neutral where it
engages: 120 real decode tokens at 24,193 tokens of held-out text produce **identical output text**, needle
included, with it on and off. (It is not bit-identical by construction — it regroups the per-slice partials —
so identical text over 120 steps is the gate, not a bit comparison. The decode test's 28 criteria are also
value-identical, but that prompt's NC is too small to engage the gathered path, so it proves nothing here.)

## FOUR hypotheses about this kernel, three of them mine, all falsified by measurement

| hypothesis | prediction | measured | verdict |
|---|---|---|---|
| the dense `[rows, NC]` mask's BANDWIDTH is the quadratic term | large | mask kernels are **< 0.12 ms/token** | dead (docs/66) |
| the radix top-k's sub-group ROTATION is its cost | ~8x on 26-34 ms | **4.31 vs 4.26 ms** | dead (docs/68) |
| K is re-read per head, so it is BANDWIDTH-bound | 64x too many bytes | the repo had already measured a shared SLM tile: **5x fewer K reads, 25 % SLOWER** | dead |
| 2,560 work-groups/token is launch/occupancy overhead | most of 13.12 ms | packing 8/16/32 heads per work-group: **13.15 ms — nothing** | dead |
| **the per-slice pick SCAN is the cost** | — | **13.12 -> 7.29** | **confirmed** |

Head packing is a pure geometry change (each sub-group already owns one head outright, so it is
bit-identical at any value) and it measured **zero**, so it was removed rather than kept behind a flag; the
finding is recorded in the kernel's own comment. Every one of these was cheap to test and plausible to
argue, which is the point: on this engine the arithmetic of a hypothesis has been a poor predictor and the
A/B has been decisive in minutes.

## What the win does NOT buy, and this is the important part

| | named kernels | wall |
|---|---|---|
| pick scan | 53.4-53.5 | 96.3-96.7 |
| **pick partition** | **47.5-47.6** | **86.7-93.7** |

So ~6 ms of device time became up to ~9 ms of wall **on the synthetic prompt with a matched ranking**, where
the expert tier is fast. On **real** text at 24,193 tokens with a real-text ranking the same change moved the
wall from 109.9 to 109.1 ms/token — **0.8 %**. The reason is docs/62's standing measurement: the GPU executes
only 26-27 % of its own window at decode, and the rest is waiting for expert bytes. **Shortening a kernel
that was already hidden behind a DMA wait converts compute time into idle time, not throughput.**

**The conclusion for "make the kernels faster":** the attention kernel is now 44 % faster and it is kept
because it is free and can only help — but decode is not compute-bound, so kernel work cannot move the
founder's 200k goal. The named kernels total 47.6 ms of an 86.7-109 ms wall; even taking every named kernel
to ZERO would not reach 10 tok/s at 200k, where the wall is 232 ms and 283 of the MoE's ms are expert
fetches. The binding term remains expert residency against host RAM and VRAM, priced in docs/70 at **1.9x
the static tier** (89 slots/layer against today's 47) to reach the ~85 % hit rate that 10 tok/s needs.

## Also settled here, from the selection counts (docs/70's tooling)

At 224k, with the raw per-(layer, expert) decode counts:
* the top-47 per layer carry **67.6 %** of that document's selections — but the ranking actually loaded (from
  a DIFFERENT document, profiled at 214k) captures only **49.0 %**, and the engine measured ~33 % static on
  its last step. So long-context rankings are **document-specific**: the top-47 sets overlap only **53.5 %**
  between two real documents at the same length, against 25.7 % for the shipped prefill ranking.
* **Non-uniform slot allocation across layers is dead**: a greedy reallocation of the identical budget by
  marginal value captures **67.7 % against uniform's 67.6 %** at 224k, and 78.5 % against 78.4 % at 41k. The
  per-layer concentration spread (47.6-79.3 % at 224k) is simply not wide enough to exploit.
