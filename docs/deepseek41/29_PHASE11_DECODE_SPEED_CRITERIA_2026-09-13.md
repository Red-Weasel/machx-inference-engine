# V4.1 port — Phase 11 gate criteria: decode speed (written BEFORE the build)

**Starting line:** decode is correct (Phase 9, gate PASS) and costs ~0.48 s per step at position
12 on the resident runtime (docs/25) — the Q8 grouped prefill path at M = 1 row per expert,
every routed expert not in VRAM fetched from the pinned arena or read from disk, every op in
the dense body launched with a host wait, and nothing measured. Prefill is at 292 tok/s
(Phase 10); at 2 tok/s of generation the engine is not usable interactively, so this phase
comes before any further prefill work.

## The arithmetic that bounds it (to be confirmed by the measurement, not assumed)

A token routes to 6 of 384 experts in each of 40 layers: 240 expert slots of 17.9 MiB, 4.2 GiB
per token if none were resident. With the held-out PREFILL profile 45% of selections are in
VRAM (docs/22), so ~2.3 GiB per token would cross PCIe from the pinned arena at ~20 GB/s:
**~115 ms per token from the fetch alone, ~8.5 tok/s**, before any kernel runs — docs/06's
projection of 6-9 tok/s, now with measured inputs. BUT that is a prefill number carried into
decode (the peer review of 2026-09-13 13:07 caught it): a prefill chunk sweeps every expert
once and the stream slots have no reuse to exploit, while decode has temporal locality — V4's
cache measured 0.48 at pp128 and 0.77-0.82 at tg4096 in one run. The decode hit rate is
therefore MEASURED by step 1 (static hits + stream-slot hits over the 240 selections, from the
cache's own counters) and the bound in criterion 4 is derived from it; the 45% stays as the
prefill contrast only. Anything faster than the measured bound needs fewer bytes per token: more
experts resident (the dense set is 14.45 GiB as fp16 per two cards; held as FP8 with an
on-the-fly dequant it would free ~7 GiB, ~380 more expert slots per card), a better profile,
or DSpark's drafter (the report's §2.4.3), which does not reduce bytes per accepted token by
itself but batches the verify.

## Scope, in order; each step measured before the next

1. **Instrument the step.** The decode test prints the per-stage breakdown at every step
   (the resident test's stage timers work at T = 1): host prep (hashes, gather), attention,
   ffn-pre, the MoE chain (prep / spawn / groups / join / mmap group / tail, with the bytes
   ACTUALLY fetched from the pinned tier — the cache's miss counter, not experts x slot bytes —
   and read from disk, and the selections served without a transfer: static hits and stream-slot
   hits), shared, head, and the host-side launch count per layer. Steps 0-3 after the golden
   prompt AND after the replayed 2048-token pp text (the realistic case: a long context, 128-row
   rings). The launch count comes from a run under SYCL_UR_TRACE=2 (which serialises and prints
   every submission); every timing comes from an UNTRACED run; neither number is quoted from the
   other's log (the Phase 10 headline was +19.4% until it turned out to compare an unprofiled run
   against a profiled baseline; like for like it was +16.6%).
2. **The MoE at M = 1.** V4's decode expert path (`ds4_expert_gemv`, docs/deepseek4) instead of
   the Q8 grouped GEMM with one row per expert; the fetch pipeline at decode sizes (V4 measured
   its fetch/compute overlap there). The mmap tier at decode: six experts per layer read from
   disk at ~10 GB/s is ~11 ms per layer if all missed — the profile decides how often.
3. **The dense body at M = 1.** Fewer launches: V4's decode GEMV toolkit for the fp16
   projections (fp32 activation in, no cast launches, a site's projections of one activation in
   one launch), then the hyper-connection chain (mixes, collapse, norm) as one kernel per site
   where the arithmetic allows (the report's Mega-mHC shape), no host wait inside a layer except
   where a host-side decision needs the data (the router's indices). Target: the launch count
   per layer in decode near the report's 11 for a Reuse-mode layer. **Stop rule:** if step 1's
   breakdown shows the fetch dominating and the launch + host time small, step 3's ceiling IS
   that small number; the phase says so and stops there rather than spend a day on 4% (V4's H1
   removed the decode host waits, was correct on every dump, and measured NULL because the link
   was the bound).
4. **Bytes per token.** Report the fetched bytes per token by tier and the resulting bound at
   the measured PCIe rate; if the fetch is the bound after 2 and 3, the FP8-resident dense set
   (more slots) is the next phase, not this one.

## Pass criteria

1. **Correctness under the same bars as docs/25**, with the before/after digits reported:
   `ie-ds41-decode-test` PASS — the state dumps under their bars (3e-3 at layers 0 and 2,
   1.2e-2 at 20), the forced step 0 at all 40 layers under 1.2e-2 (card 0) / 3.5e-2 (card 1),
   the own-router step judged to the first flip with the near-tie rule, the four greedy tokens
   the golden's, the engram hashes exact (a SEQUENCE check — they come from token ids — so it
   stays strict through any kernel swap); the forward, resident and replay tests PASS. The
   digits themselves WILL move when a decode kernel changes its accumulation (every state dump
   is downstream of an earlier layer's MoE output), so "same digits" is not the criterion; the
   bars are, and the before/after digits go in the results table.
2. **The breakdown is measured and sums** for steps 0-3, both prompts, before and after.
3. **tok/s at decode, measured:** steps 4-35 after the golden prompt and after the replayed
   2048-token prompt, reported as ms/token and tok/s with the bytes per token and the decode hit
   rate beside them, from an untraced run. Stated as a choice, before the build: THIS PHASE GATES
   ON MEASUREMENT COMPLETENESS AND CORRECTNESS, NOT ON A SPEED NUMBER. The steps' before/after
   deltas are reported, and the speed target belongs to the phase that criterion 4's bound
   selects (a number chosen after the data is what criterion 7 of Phase 10 fell into).
4. **The bound is stated from the measurement**, with the tier bytes per token, the measured
   DECODE hit rate (static + stream-slot hits over the selections), and the PCIe rate the fetch
   actually achieved (bytes / the groups' wall), so the next phase's lever is a computed choice.
5. **Launches per layer at decode** counted and reported before and after step 3, from traced
   runs (`tools/ds41_reference/ur_launch_count.py` over `SYCL_UR_TRACE=2` with the decode test's
   `IE_DS41_LAUNCH_MARKS=1` layer markers).
6. **Unload unchanged** (the resident test's two-cycle check).

## Explicitly NOT in this phase

DSpark (needs the drafter's weights bound and its own golden), the FP8-resident dense set,
the FP4/FP8 caches, sampling beyond greedy, the serving loop, tokenisation in the engine.

## Amendments before the build (2026-09-13 13:10, from the peer session's pre-read)

Five points, all taken: the bound's hit rate was a prefill number (P3: now measured at decode);
counts and timings must come from different runs (P2); "same digits" was unsatisfiable once a
kernel changes (P1: bars, with the digits reported); criterion 3 had no threshold (P4: stated as
the phase's choice — measurement completeness and correctness — not left implicit); step 3 gets
a stop rule (P5). Verified by the peer without action: `bytes_pinned` is the cache's MISS-path
counter (expert_stream.cpp:1759) so a stream-slot hit adds nothing, and the V4.1 tier never
speculates, so `spec_bytes` cannot hide bytes from it — if speculation is ever enabled, report both.

## Amendment 2 (14:35, from the peer session's second read, BEFORE the numerics isolation run)

**The forced-step bar measures distance to a bf16 reference, so a change that makes the engine
more accurate can move its digit UP.** docs/25 records 1.2e-2 as a Phase 8 regression bar against
the golden; the golden (`golden_decode.py`) holds the weights in bf16, and V4's decode GEMV keeps
the activation fp32 where the oneDNN route rounded it to fp16 (2^-11 relative). The bar stays as
written (it is what the test asserts and it still holds: 1.02e-2 on the final build), but a move
of the digit is not accepted as "because of the activation precision" on the strength of the
story. The pre-registered check: the forced step 0 dumped with (a) the old numerics (GEMV route
off, mixes decode shape off), (b) the GEMV route on alone, (c) both on; per layer, each against
the golden AND old-vs-new directly. The explanation stands only if old-vs-new differs on the
activation-rounding scale (~1e-3 relative or below) while the distance to the golden moves by
about that much; a larger old-vs-new difference means something other than precision moved and
is a finding, not a disclosure. Two numerics changes landed together (the GEMVs and the mixes
reduction order), so the attribution is isolated, not assumed.

**The opening arithmetic (~115 ms, ~8.5 tok/s) is superseded** by the measured bound: 1382 MiB per
token at 22 GB/s = 65 ms at the 2048-token context, because the decode hit rate is 61%, not the
prefill profile's 45%. The step at 344.6 ms before this phase had the fetch at about a fifth of
it; by criterion 4's own rule the next lever was therefore NOT more residency but the other four
fifths, which the breakdown named (the hyper-connection mixes kernel, docs/30).

**The host-wait removal (step 3's first item) is reverted** under the stop rule: measured worth
nothing on two builds (waits on and off within run noise), and a barrier removal that buys
nothing leaves an ordering argument to maintain. The measurement is the result; the code is not
shipped.
