# V4.1 port — Phase 17b gate criteria: the FP8 GEMV at the bandwidth floor (written BEFORE the build)

**The finding (docs/44):** `gemv_fp8_e4m3_f16` moves ~325 GB/s where the fp16 GEMM it replaced moved
~580 — half the bytes bought 2 of the 10 ms the phase saved; the other 8 came from the freed VRAM.
The kernel is ALU-bound: its per-byte decode (`ds41_e4m3`: three field extractions, a subnormal
table lookup, a select, then int-to-float) costs ~8 ops per weight beside the one fp32 FMA,
where the Q8 leaf it was copied from spends ~2. The projections' 17.9 ms per token at the
bandwidth floor would be ~10.

**The change, bounded to the kernel:** decode two E4M3 bytes at once into one fp16 pair by bit
arithmetic — for a normal code b, fp16 bits = (b & 0x80) << 8 | ((b & 0x7F) << 7) + 0x2000
(E4M3's bias 7 to fp16's 15 is +8 in the exponent, the 3 mantissa bits shift up 7), applied to
both bytes of a 16-bit lane pair in one 32-bit word; the seven subnormal codes (e = 0, m ≠ 0: the
value m × 2^-9, a NORMAL fp16 number the formula does not produce) fixed with one masked select
per pair; NaN codes (0x7F / 0xFF) are not in a checkpoint. Then the two halves convert to fp32
and enter the same fp32 FMAs as today — the accumulation is unchanged, so the unit test's 1e-5
bar stays and the forced digits do not move at all against Phase 17's (bit-identical to
`gemv_fp8_e4m3_f16` as shipped is the target: the same products in the same order). ~3 ops per
weight.

## Pass criteria

1. `gemv_fp8_test` unchanged and PASS — and a new check: the packed kernel is **bit-identical** to
   the Phase 17 kernel on every output of the seven shapes (same weights, same order, same fp32
   sums); if not bit-identical, the difference is explained (a decode edge case) and the 1e-5
   bar still holds.
2. Measured in the decode bench's kernel table (`IE_QUEUE_PROFILING=1`, EP + file + FP8 at the live
   cap): the GEMV's device ms per token from 17.9 toward the floor (≥ 500 GB/s of FP8 bytes, i.e.
   ≤ ~12 ms); the decode bench's ms/token off/on the packed kernel on one build (`IE_DS41_FP8_PACKED=0`
   the kill switch for the A/B); the decode test's digits identical to Phase 17's if bit-identical.
3. No other change; prefill untouched (the T > 1 path is the scratch + oneDNN).

## Explicitly NOT in this phase

fp16 accumulation (a precision change); the head in Q8; the o_a block-diagonal kernel in FP8.

## Results (00:27, build cde448b; `IE_DS41_EP=1 IE_DS41_EXPERT_FILE IE_DS41_DENSE_FP8=1` at the live cap)

**The pair decode failed, the table decode works.** The field-arithmetic pair decode measured 2x
SLOWER than the scalar decode — 118 ms/token and **33.4 ms** of GEMV device time against 17.9 —
because its per-lane masks, shifts and the subnormal select chain cost more than they saved; it is
gone. What decodes cheaply is a 256-entry fp16 table in local memory (every E4M3 value is exact in
fp16, so the products and sums are unchanged): one byte extraction and one local load per weight.

| decode (`IE_DS41_FP8_PACKED`) | unit test | GEMV device ms / token (32 profiled steps) | GB/s of FP8 bytes | decode bench, steps 4-35 |
|---|---|---|---|---|
| scalar `ds41_e4m3` (Phase 17) | 1e-7 vs double | 17.9 (61.8 µs per call) | ~325 | 99.6, 103.2 ms/token |
| the pair decode (removed) | bit-identical | 33.4 | ~175 | 118.1 |
| **the table decode (`=1`)** | **bit-identical to the scalar on all seven shapes** | **14.6 (50.4 µs)** | ~400 | **95.9 ms/token = 10.4 tok/s** |

Criterion 1 holds (bit-identical, the 1e-5 bar untouched); criterion 2 is met in direction and
not in magnitude: 14.6 ms against the ≤ ~12 the floor implies — the remaining cost is the local
loads plus the per-weight byte extraction and half → float conversion (the next two micro-steps:
a float table (no conversion) and the activation staged once per block as fp32; each a one-line
change measured on its own). The decode step: **110.7 (fp16) → 99.6-103 (FP8) → 95.9 ms/token**.
Opt-in until gate 17 (`IE_DS41_FP8_PACKED=1` selects it).

**Gate 17 (07:36, docs/47): PASS with a finding on criterion 2(a).** The true FP8 bytes per token are 5.49 GB (290 calls),
so 17.9 → 14.7 ms/token is 307 → 374 GB/s (the ~325 / ~400 above are ~6% high) and the ≤ ~12 ms bar is NOT met: the
kernel is not at the bandwidth floor and this doc's title does not claim it was reached. Phase 18 term 3 falsified the
premise (an ALU-bound per-weight decode): the fp32 table and the staged activations move no device time (50.4 vs 50.8 µs
per call), so the floor question needs a different instrument (docs/47). The table decode is the default since the gate.
