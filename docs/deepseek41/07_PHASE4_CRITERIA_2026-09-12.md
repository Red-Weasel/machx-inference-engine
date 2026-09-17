# V4.1 port — Phase 4 gate criteria (written BEFORE the build)

**Scope:** the two device-side primitives the whole forward pass rests on — get V4.1's routed
experts onto the GPU, and dequantise its dense FP8 weights on the GPU — each verified on device
against the host reference Phase 2 already proved bit-exact. First GPU work in this port.

Artifacts: `ds41_expert_bank_upload` and `ds41_dense_dequant_f16` (in
`src/ops/deepseek41_upload.cpp`, declared in `include/ie/deepseek41.hpp`), exercised by
`tools/ds41_device_test.cpp` → `ie-ds41-device-test`.

## The claim this phase tests

The engine's `DS4ExpertBank` MXFP4 planes are documented as
`mx_qs[n*(K/2) + b*16 + j]` and `mx_e[n*(K/32) + b]` — i.e. `[N, K/2]` and `[N, K/32]`, both
row-major. V4.1 ships `w1.weight [2304, 2560]` with `w1.scale [2304, 160]`, which for
K = 5120, N = 2304 is *the same two arrays*. **If that is true, 268.95 GiB of routed experts
upload with a straight memcpy and no repack at all**, and the engine's existing, month-tuned
MXFP4 expert kernels run on them unmodified. This phase either demonstrates that on the device
or finds where it is false.

## Pass criteria

1. **GPU clearance is checked and reported before any allocation**, and the test refuses to run
   if a device is busy or short of memory. (Founder hard rule 2.)

2. **Expert upload is a copy, not a repack.**

   > **FALSIFIED BY THE BUILD — and this is the phase's main result.** The premise was wrong.
   > The engine's MXFP4 block is gpt-oss **interleaved** (byte j holds element j and element
   > j+16); V4.1's is **sequential** (byte b holds 2b and 2b+1). Criterion 3's one-hot check
   > found it: `k0=0` and `k0=5119` agreed while `k0=1` and `k0=37` did not, and elements 0 and
   > 31 are precisely the permutation's only fixed points. Every byte-level and shape-level
   > check in Phases 2 and 3 passed regardless, because the bytes and shapes really are the
   > same — only the order differs.
   >
   > The criterion is replaced by the stronger one the situation demands: **the device FP4
   > planes must equal a host recomputation of the exact permutation**, per expert, and the
   > E8M0 scale planes (which genuinely do cross unchanged) must still be byte-identical to the
   > source.

3. **The existing expert GEMV reads them correctly — proved exactly, not approximately.**
   Running `ds4_expert_gemv` on the uploaded bank with a **one-hot** activation `x = e_{k0}`
   must return `y[n] == fp16(W[n, k0])` for every `n`, where `W[n, k0]` is the host decode from
   Phase 2. One-hot makes the sum a single term, so there is no accumulation error to hide
   behind: the comparison is **exact equality in fp16**, not a tolerance. Tested at multiple
   `k0` including an **even and an odd** index, so a nibble-order flip fails.

4. **Dense FP8 dequant on device is bit-exact against the host.**
   `ds41_dense_dequant_f16(w, scale, N, K)` must produce, for every one of the N*K elements,
   exactly `fp16(e4m3_to_f32(byte) * e8m0_to_f32(scale_byte))` with the 32x32 block mapping —
   compared as **fp16 bit patterns**, so a sign-of-zero or a one-ulp rounding difference fails.
   Run on a real tensor (`layers.0.attn.wq_b`, 41.9 M elements), not a synthetic one.

5. **The denormal trap does not recur on device.**

   > **AMENDED BEFORE BUILDING (disclosed).** As first written this criterion said to run the
   > fp16 dequant on a block whose E8M0 scale byte is 0 and check it does not return zeros.
   > That test cannot work and would have been a fake pass either way: 2^-127 times the largest
   > E4M3 value (448) is about 2^-118, and fp16's smallest subnormal is 2^-24, so **every**
   > correct implementation returns zero there. The bug is unobservable through an fp16 output.
   > (Which is itself worth recording: for an fp16 dense dequant, a 2^-127 block scale is zero
   > regardless, so the host bug this phase inherited was harmless in *this* kernel — but not in
   > any fp32 path, and not in the E8M0 decode itself.)
   >
   > The criterion is therefore moved down to the decode, where the bug actually lives, and made
   > stronger: **the device decoders `ds41_e4m3` / `ds41_e8m0` must agree with the host
   > `ie::e4m3_to_f32` / `ie::e8m0_to_f32` on all 256 codes each, compared as fp32 BIT PATTERNS,
   > evaluated inside a real SYCL kernel under the production build flags.** That executes the
   > 2^-127 subnormal path on device at fp32, which is exactly where icpx's default
   > `-ffp-model=fast` flushed it on the host, and it also catches sign-of-zero and the NaN
   > encodings. The two decoders were moved into `deepseek41_upload.hpp` so a test kernel can
   > call the production functions rather than a transcription of them.

6. **Real memory accounting.** The test reports device bytes actually allocated for one layer's
   384 experts and compares against the predicted `384 * 17.93 MiB = 6.72 GiB`, within the
   allocator's granularity. A silent 2x from an unnoticed conversion would show here.

7. **Clean teardown.** Every device allocation is freed; the test reports free VRAM before and
   after and they match within allocator slack.

8. **No regression.** The 20 host-only tests green at the end of Phase 3 are still green, and
   nothing outside the new files plus the two CMakeLists changes.

## Explicitly NOT in this phase

Tiering across VRAM/RAM/NVMe, the residency planner, attention, routing, mHC, the forward pass.
Phase 4 answers: *do V4.1's bytes land on the GPU correctly, and does existing engine code read
them?*
