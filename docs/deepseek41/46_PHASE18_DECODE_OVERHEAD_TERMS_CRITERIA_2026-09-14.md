# V4.1 port — Phase 18 gate criteria: the decode step's overhead terms (written BEFORE the build)

**Where things stand (docs/44, docs/45; the 2,048-token context, EP + the live cap + the expert file
+ `IE_DS41_DENSE_FP8=1 IE_DS41_FP8_PACKED=1`, gate 17 pending):** decode **95.9 ms/token = 10.4 tok/s**;
pp2048 390-409 tok/s. The step's columns (steps 1-3 of `p17b_prof_lut.log`, ms): attention 25.0,
MoE 45-74 (groups 20-39 = the owner tier's pinned fetch; tail 21-24 = the owner waiting for the
other card's tier, the E[max] geometry of docs/39), shared expert 5.4, ffn_pre 3.1, **setup 2.7-3.2**,
head 2.3, prep + engram 1.1. The named kernels are 45.8 ms/token of device time (FP8 GEMV 14.6,
expert GEMMs 12.1, attention 7.0, o_a 4.7, head 2.2, router 1.5, the rest < 1 each); each card
executes ~20% of its own window. The attention site's wall (25) now matches its device time
(~23: the four FP8 projections ~10.8, attention 7.0, o_a 4.7, mixes / rope / norm ~1) — the host
gap Phase 11 chased there is gone; the terms this phase names are elsewhere.

## The three terms, and the arithmetic stated before the build

1. **The shared expert runs AFTER the routed MoE and is not hidden (5.4 ms/token).** Its three
   projections (`sh_gate`, `sh_up`, `sh_down`, ~0.12 ms of GEMV per layer) and its SwiGLU read only
   `xfn` — the same input the routed experts take — and nothing in the tier's `moe()` touches
   `x16 / t16 / shg / shu / shh`. Enqueued on the owner's in-order queue right before the tier call
   (after the EP staging's D2H wait, so the remote card's start is not delayed), they execute while
   the first group's DMAs are in flight, where today's queue is idle; the add `moe += shh` and the
   hc mix stay where they are, so the sum is formed in the same order — **bit-identical**. Gain:
   up to the shared column where the fetch is the layer's longer leg (it is in nearly every layer at
   this hit rate): ~4-5 ms/token. The risk the measurement decides: if the tier's first pageable
   H2D copies block the host thread behind the shared kernels, the first group's DMA issue slips by
   ~0.13 ms per layer and the gain nets to zero; the groups column shows it, and the fallback is a
   pre-groups hook in the tier (also bounded). Kill switch `IE_DS41_SHARED_EARLY=0`.
2. **The per-step setup (2.7-3.2 ms/token) is allocation and host round trips, all avoidable.** Per
   card per step today: ~48 `sycl::malloc_device` scratch allocations and their frees (the S32 /
   S16 / SI lambdas; a Level Zero allocation is tens of µs); five `rope_tables` calls, each two
   H2D copies, two `wait()`s and a free (10 host syncs per card for four positions' worth of
   cos/sin); the stream copies with waits; and the **source-cache bounce**: card 0's last kv-source
   state (layer 14's `nc` latents + index keys, 5.2 MB at 2,048 positions, growing linearly with
   the context) copied D2H at the end of card 0 and H2D at the start of card 1 on every step — dead
   at this split, because card 1's first layer (20) is both a kv source and an index source and
   overwrites `cur` and `sh_topk` before any consumer reads them. The change: (i) the decode
   scratch persistent per card (allocated on the first decode step at the T = 1 shapes, reused,
   freed with the state; a prefill keeps today's per-call allocation); (ii) the RoPE tables at decode
   from per-card resident inv_freq (uploaded once) and the step's positions in one copy, the same
   `ds4_rope_cos_sin` kernel enqueued without waits (same inputs → **bit-identical**); (iii) the bounce
   skipped when the next card's first layer re-derives both caches (the dumps at 0.0 prove it is
   never read). Gain: setup ~3 → < 1 ms/token. Kill switch `IE_DS41_SETUP_FAST=0`.
3. **The FP8 GEMV's table decode is 14.6 ms/token against a ~10-12 floor (docs/45).** Two one-line
   micro-steps, each measured on its own: (a) the local table as fp32 instead of fp16 (the same 256
   values, no half → float conversion per weight: **bit-identical** by construction); (b) the
   activation block converted to fp32 once per 32-weight block instead of once per FMA (the same
   conversions, hoisted: **bit-identical**). The unit test's bit-identity check (docs/45 criterion 1)
   now compares the shipped decode against the scalar decode, whichever variant ships. Gain: 14.6 →
   ~12 ms/token if the per-weight ALU was the cost; the kernel table says.

## Step 0 — measure before building (no GPU code changes)

(a) **The GPU clock during decode.** The split attention kernel's own bench (deepseek4_attn.cpp: SP =
64, T = 1, 640 live columns) is 0.08-0.09 ms per call; the decode profile shows 175 µs per call, and
the FP8 GEMV moves ~400 GB/s where the fp16 oneDNN GEMV at the same site moved 578 (docs/30). Both
fit a core clock below 2,800 MHz during a step whose cards are busy ~20% of the time (`act_freq`
reads 0 between kernels: the cards power-gate). A 10 ms sampler of both cards' `act_freq` /
`cur_freq` (sysfs, read-only) during the decode bench's steps 4-35 gives the distribution; if the
clock sits below the maximum for a material share of the busy time, the A/B is `min_freq` = 2,800
on both cards vs the default 400 on the same build (a reversible sysfs write that needs root — the
founder's, listed as such; nothing in this phase changes system state). Reported whatever it says.
(b) **The setup term's composition**: the allocation count per card per step and the bounce bytes,
from the code and one traced step, so the change's expected gain is written down before the change.

## Scope, in order

1. Term 1 (the shared expert ahead of the tier), one build, kill switch, bit-identical, measured.
2. Term 2 (the setup: persistent decode scratch, RoPE without waits, the dead bounce), one build,
   kill switch, bit-identical, measured.
3. Term 3 (the GEMV micro-steps), each on its own build or switch, bit-identical, the kernel table.
4. The founder table updated; defaults: a bit-identical, measured win is ON with its kill switch
   (the block time, C0, C1 precedent); anything that is not bit-identical or not a measured win is
   reverted, not switched off.

## Pass criteria

1. **Bit-identity**, each term with its switch on against the Phase 17b path (all switches off on
   the same build): the decode test's forced digits and logits identical to the digit, the four
   greedy tokens and the hashes the same, and `IE_DS41_DUMP_DIR` dumps at exactly 0.0 at all 40
   layers (M, L, and the prefill's PX / PM / P where the term touches prefill — terms 1 and 2 are
   decode-only by construction; term 3 is the T = 1 GEMV only). The FP8 unit test's bit-identity
   check PASS on all seven shapes for the shipped decode.
2. **Measured on one build** (`IE_DS41_EP=1 IE_DS41_EXPERT_FILE IE_DS41_DENSE_FP8=1 IE_DS41_FP8_PACKED=1`
   at the live cap, the 2,048-token context, steps 4-35): the decode bench with all switches on,
   and each switch off in turn; the shared / setup columns move as the arithmetic above says (or
   the doc says why not); the kernel table's GEMV row for term 3; a second sample of the all-on arm
   to bound the spread. Every arm's bytes and routing identical (pinned MiB, disk MiB, hit rate).
3. **Prefill unchanged**: the resident test's pp512 / pp2048 within the run-to-run spread of docs/44
   (390-409 warm) and its digits the Phase 17b path's.
4. **The whole test set PASS** with the defaults: decode, resident, generate, replay, tier, forward,
   the FP8 unit test; unload clean (both cards' VRAM returned).
5. Step 0 reported whatever it says; no system setting changed by the build.

## Explicitly NOT in this phase

`o_a` in FP8 (a block-diagonal FP8 GEMV, ~2.3 ms and ~1.3 GB of VRAM — Phase 19, judged by the bars);
the CPU miss split under EP (docs/42 scope 3, Phase 20; the E[max] lever); the head in Q8 (a
precision decision); the T = 1 attention beyond step 0's clock question; DSpark (the checkpoint
carries the three MTP layers, `dspark_block_size` 5 — the structural decode lever, a founder
decision and a multi-phase build); any P2P staging.
