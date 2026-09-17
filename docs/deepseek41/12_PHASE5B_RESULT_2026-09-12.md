# V4.1 port — Phase 5b result: attention matches, with zero new kernels (2026-09-12)

`ie-ds41-attn-test` vs `golden_block.py`, layer 0, T=8, on device.

```
Intel(R) Arc(TM) Pro B70 Graphics — layer 0 attention, T=8, 64 heads x 512 (rope 64), o_groups 8 x 1024
[ ok ] qr (q_norm)      rel 2.196e-04   (tol 3e-03)
[ ok ] q (roped)        rel 2.383e-04   (tol 3e-03)
[ ok ] kv (roped)       rel 2.386e-04   (tol 3e-03)
[ ok ] o (attention)    rel 2.371e-04   (tol 3e-03)
[ ok ] o (unroped)      rel 2.372e-04   (tol 3e-03)
[ ok ] wo_a (grouped)   rel 2.674e-04   (tol 5e-03)
[ ok ] attn out         rel 3.063e-04   (tol 5e-03)
worst stage 3.063e-04 — ATTENTION TEST: PASS
```

Error accumulates from 2.2e-4 to 3.1e-4 across seven stages and four GEMMs — the gentle growth
fp16 activations against an fp32 golden should produce, with no stage stepping out of line.

## Criterion 6 held: no new kernel was written

The whole attention path runs on code the V4 port already shipped:

| stage | kernel | new? |
|---|---|---|
| `wq_a`, `wq_b`, `wkv`, `wo_b` | `gemm_nt_f16_onednn` (row-major `[N, K]` weight = safetensors `[out, in]`) | no |
| `q_norm`, `kv_norm` | `ds4_rms_norm` | no |
| RoPE on q and kv | `ds4_rope_apply`, `sin_sign = +1` | no |
| attention | `ds4_attention` — one latent KV head read as both K and V, per-head sinks, head_dim 512 | no |
| window mask | `ds4_sliding_causal_mask` | no |
| conjugate rotation on the output | `ds4_rope_apply`, `sin_sign = -1` | no |
| **block-diagonal `wo_a`** | `gemm_bmm_nt_f16_onednn` | no |
| FP8 → fp16 weights | `ds41_dense_dequant_f16` | Phase 4 |

`git diff --stat` for this phase is the test tool and one CMakeLists line. The claim that V4.1's
attention is reuse is not an argument — it is what the diff shows.

## Criterion 4: the conjugate rotation is genuinely tested

`o (attention)` and `o (unroped)` came back at 2.371e-4 and 2.372e-4 — near-identical, which is
the shape of a vacuous check, so it was checked rather than assumed. The two goldens differ by
**max 0.5146 on a 0.8321 scale** — 62% — and the difference lives **entirely** in the trailing 64
rope channels: the first 448 "nope" channels differ by **exactly 0.0**. That is precisely how a
partial rotation on the trailing `rope_head_dim` should behave.

The near-identical *relative* errors have a different cause: rotation is norm-preserving, so it
turns the error vector without changing its magnitude. Omitting the `sin_sign = -1` call would
leave the engine's output unrotated and miss `attn_o_unroped` by ~62% — about 200x the 3e-3
tolerance. The trap the V4 header warns about ("omitting it silently corrupts long context")
cannot be passed through here.

## What now works end to end against the reference

- routing: **exact** (Phase 5a)
- routed experts: 2.0e-4 (fp16 act) / 3.3e-3 (Q8_1 act) (Phase 5a)
- attention: **3.1e-4** through all seven stages (this phase)
- expert upload, dense FP8 dequant: bit-exact (Phase 4)

Remaining for a full layer: wiring mHC (`ds4_hc_mix` — reuse, see `03` addendum) around the two
sub-blocks and comparing `blk_out`. Then the residency planner and the layer loop.
