# V4.1 port — Phase 5a result: the MoE block matches the reference (2026-09-12)

`ie-ds41-moe-test` against `tools/ds41_reference/golden_moe.py`, layer 0, 8 deterministic tokens.
The golden is DeepSeek's **own** `MoE(0)` — their Gate, their Expert, their routing, their
clamps — with only the leaf GEMMs replaced (`docs/deepseek41/09`).

```
=== gate (host recompute vs the reference's routing) ===
[ ok ] top-6 expert indices match the reference exactly  (0 of 48 differ)
[ ok ] routing weights match the reference  (max |diff| 0.000000)

=== routed experts on device (reference's own routing) ===
       uploaded 3 x 384 experts, 6.72 GiB
       fp16 activations   max|diff| 1.277e-06   rel 2.044e-04   rms 2.207e-07
       Q8_1 activations   max|diff| 2.047e-05   rel 3.276e-03   rms 4.730e-06
MoE TEST: PASS                                                    (2.97 s)
```

## Why the two halves are tested separately

Routing and expert arithmetic are checked independently, and the expert half is fed the
**reference's own** indices and weights rather than the engine's. If they were tested together a
routing error and an expert error could cancel into one plausible end-to-end number, and a
single figure would not say which was wrong. This is the same lesson Phase 4 paid for: a check
has to distinguish *which* thing is wrong, not merely that something is.

## What the numbers mean

- **Routing is exact** in selection: all 48 top-6 indices identical; weights within 1e-5 (the
  test's bar — an earlier draft said "bit-equal", which a 6-decimal print cannot establish).
  That confirms the whole gate is understood — `sqrtsoftplus` scoring, the correction bias
  steering *selection only* while the weights come from the unbiased scores, the `+1e-20`
  renormalisation (not `norm_eps`), and `route_scale` 1.5.
- **2.0e-4 relative at fp16 activations** is the arithmetic of the block itself: MXFP4 weights,
  fp16 activations, fp32 accumulation, against an fp32 golden.
- **3.3e-3 at Q8_1 activations** is the engine's fast decode path. The engine's own header
  predicts "the Q8_1 activation quantiser costs 0.4%" for V4; 0.33% here is the same regime,
  measured independently on a different model.

## One numerical difference from the reference, by design

V4.1's `Expert.forward` applies the routing weight to `h` **before** `w2`; the engine's
`ds4_experts_forward` applies it **after** `down`. Since `down` is linear these are
mathematically identical, but not bit-identical in finite precision, and in the reference the
weight is also applied before the activation quantisation that `w2` would do. This is inside the
2.0e-4 already measured; it is recorded because it is the kind of difference that gets
rediscovered later as a mystery.

## Where this leaves the port

The MoE block — **57% of the model's weights, 268.95 GiB** — is verified end to end against the
reference on device. Combined with Phase 4, what now works: experts upload correctly, dense FP8
dequantises correctly, routing is exact, and the routed-expert block reproduces the reference.

Remaining for a running layer: the attention block (RoPE and mHC are reuse — see
`03_ARCH_FROM_THE_CODE` addendum — but the block-diagonal `wo_a` and the dense FP8 GEMMs are
new), then the residency planner, then the layer loop.

## Gate findings applied (2026-09-12)

The Phase 5 gate passed the phase but showed two things about this test that needed fixing
before it becomes a regression guard:

1. **A w1/w3 swap in the expert bank passed at rel 1.90e-2 against the old 2e-2 bar.** Two
   causes: the bar had ~100x of headroom over the measured 2.0e-4, and the golden fed
   `randn*0.02` with no norm in front, where the real post-`ffn_norm` input is std 0.127 —
   and SwiGLU's gate/up asymmetry, the only thing a swap perturbs, shrinks with the
   activations. Now: input at std 0.13 and bars of 2e-3 / 8e-3. Measured at the new scale:
   fp16 **3.2e-4**, Q8_1 **3.1e-3**; a swap lands ~2e-2, an order of magnitude outside.
2. **The routing check was a host recompute**, proving the gate semantics were understood but
   never running the engine's own device router. `ds4_router_topk` (sqrtsoftplus, noaux_tc)
   now runs on the same input: **top-6 indices match exactly as sets (0 of 48 differ), weights
   within 1e-4 (measured 0.000000)**. That is the kernel the layer loop will call.
