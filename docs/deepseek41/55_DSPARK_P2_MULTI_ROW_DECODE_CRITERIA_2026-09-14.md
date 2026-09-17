# V4.1 port — DSpark P2 gate criteria: the backbone's decode forward at T > 1, bit-identical to T one-row steps (written BEFORE the build)

**The contract** is docs/48 section E, P2, taken verbatim; this page fixes the names. The verify pass of
speculative decoding needs one forward over 1 + k rows at `pos0 = n_pos`, today refused (`forward: at pos0 > 0
only single-token steps are admitted`). docs/48 C.1 enumerated the 22 sites that assume T = 1 at decode; the
build changes the twelve it names (1 admission, 2 the engram hash rows, 3-4 the RoPE tables per row and per
emitted latent, 6 the ratio-1 compressor over T rows, 7 the ratio-2 groups closed by a T-row step around the
held half, 8 the index keys' per-latent RoPE, 10 the real-position table for the top-k, 12-13 the decode masks
as [T, NKV] with per-row ring causality, 16 the hyper-connection mixes' chunked shape for T rows, 18 the
persistent scratch keyed by T), and nothing else. **Opt-in `IE_DS41_DECODE_MULTI=1`** (up to 8 rows) until the
gate; at T = 1 every path is the one Phases 18-20 measured (the same slots, the same five RoPE entries, the same
one-row mask).

## Pass criteria (docs/48 P2, with the test that judges each)

1. **Bit-identity** — `ie-ds41-multi-test <model> <golden dir>`: for T ∈ {2, 3, 6}, at an even and an odd pos0
   (the ratio-2 parity), on the 12-token golden prompt (pos0 12 / 13) and past the ring's wrap on the 2,048-token
   pp text (pos0 2048 / 2049): the T-row step's logits rows equal the T one-row greedy steps' bit for bit, every
   layer's state (`read_state`: the ring slot by slot, the latents, the index keys, nc) identical, the engram
   hashes of the T rows identical. If a term cannot be made bit-identical the test names it (the first layer
   and which of ring / latents / index keys / nc) and the doc states a bar with the measured residual — never
   silence.
2. **The negative control** — the same test with the per-row ring causality off
   (`set_multi_nocausal_diagnostic(true)`, a diagnostic setter) at T = 6 on each scenario's even pos0: the
   T-row step must differ.
3. **The item-8 latent RoPE** — a T = 2 ratio-1 step through the old single-row table would rotate both
   latents at one position; criterion 1's index-key comparison at T ≥ 2 on the pp text covers it (layers 20+
   are ratio 1 and index sources).
4. **Prefill unchanged** — `ie-ds41-forward-test`, `ie-ds41-resident-test` (digits and pp512 / pp2048),
   `ie-ds41-replay-test` on the build; the decode path at T = 1 unchanged: the full decode test's dumps 0 of
   200 differ against the previous build (Phase 20's) with the switch unset AND set.
5. **Measured** — the T = 6 step's ms and its column split on the decode shapes, against docs/48 D.3's
   central row (332 ms); which of the two known traps (the FP8 dense dequantised per call at T > 1 —
   5 bytes per weight; the mixes' general kernel, now replaced by the T-row chunked shape) is the larger
   remaining term. Not a pass/fail bar: the number the next phase (the small-M FP8 GEMV) is written from.

## Explicitly NOT in this phase

The rollback (P3), the loop (P4), the drafter (P1, whose golden is being prepared), the small-M FP8 GEMV at
T > 1 (the perf step after criterion 5's number), the CPU split at T > 1 (docs/48 D.4.2).

## Results (09:29, `ie-ds41-multi-test`, build `3d803150…`; logs `~/ds41_work/p2/`)

**Criteria 1-3: PASS, 38 of 38** (12 cases × 3 checks + 2 controls; gate P2 finding 2) — for T ∈ {2, 3, 6} at pos0 12 / 13 (the golden prompt) and 2048 / 2049 (the pp
text, the ring wrapped): every logits row bit-identical to the T one-row greedy steps' (0 of 258,560 / 387,840 /
775,680 differ), every layer's ring, latents, index keys and nc identical, the engram hashes identical; the
negative control (the ring causality off) differs by 775,680 logits on both scenarios. Item 8's latent RoPE is
covered by the pp-text cases (layers 20-39 are ratio 1 and index sources; T ≥ 2 emits several latents).

**What it took beyond docs/48's twelve items — three terms the brief did not name, each found by the row-0
per-layer probe the test carries:**

1. **Every dense GEMM at decode is now a row-identical kernel** (`gemv_fp8_rows`, `gemv_f16_rows`, src/ops/gemv_fp8.cpp;
   `tests/unit/gemv_rows_test.cpp` 48 of 48): at T > 1 the FP8 sites went through the dequant + oneDNN GEMM and the
   fp16 sites (o_a, the compressor, the indexer, the head) through oneDNN at M = T — each a different summation order
   from the M = 1 kernel, and a ~1e-7 difference at layer 0 cascades through the Q8 activation path (docs/39) to
   routing flips (row 0 differed by 1.4e-4 at layer 0 and 7e-2 by layer 27). The rows kernels read the weights once
   for M ≤ 8 rows with the M = 1 arithmetic per row; the FP8 one at M = 1 is bit-identical to the shipped table
   decode. The T = 1 path's fp16 sites moved with it (oneDNN → the rows kernel): the forced digits went 1.055e-2 /
   1.373e-2 / 6.008e-3 → **8.885e-3 / 1.304e-2 / 5.043e-3** (bars 1.2e-2 / 3.5e-2 / 5e-2), the bench 80.8 → 81.7
   ms/token (inside the spread), the resident test PASS (pp2048 390.6 / 394.9, the low end of the day's 390-414;
   the kernels are decode-only). Criterion 4's "dumps 0 of 200 vs the previous build" is therefore NOT met at T = 1 —
   stated here as the price: the T = 1 decode arithmetic changed at five oneDNN sites.
2. **The expert grouped GEMM's arithmetic depends on the rows per expert**: two rows of a step selecting the same
   expert form a 2-row job whose M-tile sums differently from a 1-row job (layer 3 differed with an identical MoE
   input). At 2 ≤ T ≤ 8 an expert's rows are now one job each (the slot read once per row from VRAM: decode-only).
3. **The attention's split factor depends on T** (64 slices at T = 1, 32 at 2, 16 at ≤ 16) and the tree combine's
   order with it — rows with more than 16 live columns differed. `g_ds4_attn_split_fixed64`, set by the V4.1
   runtime, keeps 64 slices for T ≤ 8 (the V4 path untouched).
4. **docs/48 C.1 item 5 ("the ring write is already correct for any T") was wrong for the wrapped ring**: writing
   the step's T rows before attention evicts the T − 1 oldest keys the earlier rows still see (row 0 at 2048 needs
   1921, whose slot now held 2049). The attention now reads this step's rows from `kvwn` at their slots' column
   indices (`Ds4KvSegs::a_new`, the one-row step's layout, so the combine order is the one-row step's) and the old
   ring elsewhere, with the per-row window rule in the mask; the ring is written after the attention. At T = 1
   the same path (row 0 at its own slot) — bit-identical to the write-first order.

**Criterion 5 (the T = 6 step, `IE_DS41_CPU_MISS=0` in the test):** at pos0 2048 / 2049: T = 2 **158.8 / 147.0 ms**,
T = 3 **207.0 / 200.8**, T = 6 **344.1 / 312.3** — 52-57 ms per row at T = 6 against 82 at T = 1, on docs/48 D.3's
central row (332). The FP8 dequant trap is gone (the rows kernels); the remaining terms of a T = 6 pass are the
union's link bytes, the attention at 64 slices × 6 rows (~12 ms/pass over the 16-slice ladder) and the expert
jobs read once per row. Not tuned here.

**Gate P2 (10:08): PASS** — 38 of 38 reproduced (EP + split off, and the single-tier path), the split-on arm measured
(24 of 38 FAIL, max |logit diff| 0.2-3.1: the split's fp32 rows cascading), the T = 1 digits confirmed, the bench
not reproduced under a loaded box (89-91 vs 81.7; environment), findings taken: the count above, the overlay guarded
in the attention's tile / XMX branch, P4 to run with `IE_DS41_CPU_MISS=0` for exact losslessness (or state a bar).

**Not bit-identical by design:** the CPU miss split serves misses at T = 1 only (its rows are fp32-computed), so a
T-row step and T one-row steps agree bit for bit only with the split off (`IE_DS41_CPU_MISS=0`, as the test runs) —
the P4 loop's losslessness against plain greedy holds exactly with the split off and up to the split's own
fp32-vs-Q8 differences with it on; docs/48 D.4.2's split at T > 1 is a later phase.
