# V4.1 port — Phase 17 gate criteria: the dense projections resident in their own FP8 (written BEFORE the build)

**The term.** Phase 11's kernel table (docs/30, the 2,048-token context, T = 1): the twelve
oneDNN dense projections + the head are **27.3 ms of every token's device time**, the expert GEMMs
12.0, attention proper 7.3, router / norms / mixes ~1 each. They are bandwidth-bound on the
dense weights: the cache holds them as fp16 (14.45 GiB across the two cards; 27 ms × ~530 GB/s ≈
14.5 GB) — but the checkpoint stores every attention / index / compressor / output / shared-expert
matrix as **FP8 E4M3 with one E8M0 scale per 32 × 32 block** (`layers.L.attn.wq_b.weight` F8_E4M3
[32768, 1280], `.scale` F8_E8M0 [1024, 40]; the head and embedding BF16), and the engine's load-time
dequant (`ds41_dense_dequant_f16`, bit-exact against the host) doubles those bytes for every decode
step that follows. The deployment (docs/26) runs FP8. Keeping the dense matrices in their native
FP8 and reading them with an FP8-weight GEMV at T = 1 halves the projections' bytes with NO change
to the weight values (every fp16 weight today IS e4m3 × 2^(e−127), exactly representable) and frees
~3.3 GiB of VRAM per card for static expert slots.

**The arithmetic, stated before the build.** Bytes per token 14.5 → ~7.3 GB (1 byte per weight +
1/1024 of scales; the BF16 head stays, 1.32 GB on card 1) → the projections 27.3 → ~14 ms →
decode ~111 → **~98 ms/token (~10.2 tok/s)** at the 2,048 context with EP + the live cap; the
freed VRAM ~180 static slots per card (~4.5 per layer) → a few % more decode hit rate on top.
Prefill: oneDNN's GEMMs need fp16, so each layer's matrices are dequantised into a per-card fp16
scratch per pass with the existing kernel (~361 MB per layer → ~1.1 GB of device traffic per
layer ≈ 2 ms → ~80 ms per pp2048 pass ≈ 1.5%, within the run-to-run spread); "the number is what
it is". Numerics: the GEMV accumulates in fp32 over the same weight values; its summation order
differs from oneDNN's, so the outputs differ at the fp32-reassociation level and, through the Q8
activation path, the forced digits MOVE (Phase 14's mechanism) — the bars judge (1.2e-2 / 3.5e-2),
the before/after digits are reported, bit-identity with the fp16 path is NOT claimed (EP vs off
under the new path stays bit-identical to itself).

## Scope, in order

0. **Confirm the term on the current build and text** (Phase 16 step 0(a)'s profile, queued
   behind gate 15): the projections' device ms per token before anything is built.
1. **The kernel**, `gemv_fp8_e4m3_f16(q, x_f16, w_fp8 [N, K], scale_e8m0 [N/32, K/32], y_f16, K,
   N)` (`src/ops/gemv_fp8.cpp`): the Q8_0 SoA f16 leaf's shape — one 16-lane subgroup per output
   column, each lane 32-weight blocks strided by 16, two 128-bit weight loads per block, 32 FMAs
   against the fp16 activation, the block's E8M0 scale applied once per block, fp32 accumulation,
   a subgroup reduction, fp16 store — with the E4M3 → float decode identical to
   `ds41_dense_dequant_f16`'s (a 256-entry table). `tests/unit/gemv_fp8_test.cpp`: on the model's
   seven shapes with random bytes and scales, the kernel equals a double-precision host dot over
   the same decoded weights within **1e-5** relative before the fp16 store (the store rounds at
   ~5e-4, checked separately at 1e-3), and equals `ds41_dense_dequant_f16` + `gemm_nt_f16_onednn`
   within 2e-3 (the two paths' orders differ); a T = 4 rows variant is NOT in this phase (decode
   is T = 1).
2. **The cache** (`Ds41DenseCache`): with `IE_DS41_DENSE_FP8=1` (opt-in for the A/B, the default
   decided by the measurement), the GEMM-consumed FP8 matrices stay FP8 + scales on the device
   (no fp16 copy) and a per-card fp16 scratch of the largest matrix serves prefill: `f16(q,
   matrix)` dequantises into it (in-order on the queue) and returns the pointer; the layer's
   `bytes` and `card_info` report the true footprint. The BF16 head / embedding / router and the
   fp32 norms are untouched.
3. **The forward**: at T = 1 every projection site calls the FP8 GEMV; at T > 1 the scratch path.
   The KernelProfiler names the new kernel; the launch count per layer does not rise.
4. **Measured on one build, the env off / on**: the decode bench (EP + live cap + file: the
   headline), the kernel table at T = 1 (the projections' device ms), the resident test (prefill
   within the spread; the static count per card up by the freed VRAM), the forced digits and the
   logits vs the golden with the bars, the whole test set (decode, resident, generate, replay, tier,
   forward — the forward test's non-resident path is unaffected), unload clean.
5. **The founder statement**: the table with this phase's row and the honest distance.

## Step 1 result (23:52, before the measurements)

`gemv_fp8_test` on the seven shapes: the kernel is within **~1.1-1.4e-7** of the double reference over
the decoded weights (bar 1e-5) and within **1.5-3.2e-7** of `ds41_dense_dequant_f16` + the oneDNN fp16
GEMM at M = 1 (bar 2e-3): the two paths agree to fp32 precision — the oneDNN GEMM accumulates in
fp32 too, so the "different summation order" of the arithmetic above is a ~1e-7 effect, and the
forced digits are expected to move only through the Q8-activation trajectory (Phase 14's
mechanism), if at all. `ds41_e4m3` equals the E4M3 formula on all 254 finite codes.

## Pass criteria

1-4 as stated; 4 reported whatever it says, every bar under its limit or the change reverted; the
default with the env off is byte-for-byte today's; no test weakened.

## Explicitly NOT in this phase

The BF16 head as Q8 (a precision decision — the logits' argmax on near-ties); the expert GEMMs
(already MXFP4); a T > 1 FP8 GEMM (prefill keeps oneDNN fp16); the KV / index caches in FP8 (docs/26
decision 1).
