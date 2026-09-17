# V4.1 port — Phase 5b gate criteria (attention block)

> **PROCESS NOTE, disclosed rather than hidden.** The alpha-omega loop says pass criteria are
> written **before** the phase is built. That held for Phases 2, 3 and 4. It did **not** hold
> here: `tools/ds41_attn_test.cpp` was written first and these criteria second, though both were
> written before the test was ever **run**, so no number below was chosen to fit a result that
> had already been seen. The tolerances are justified from the arithmetic in each case, not from
> observed error. Recording this because a criteria set written after the code is exactly the
> kind of thing that silently becomes a rubber stamp, and the gate should weigh it accordingly.

**Scope:** layer 0's attention block on device, checked against DeepSeek's own `Attention(0)`
stage by stage. No compressor, no indexer (layer 0 is `compress_ratio == 0`), no KV cache reuse
across steps, no decode path.

## The claim this phase tests

That V4.1's attention needs **essentially no new kernels**. Every piece is asserted to be
already present from the V4 port:

| V4.1 needs | engine already has |
|---|---|
| RMSNorm, `y = w * x * rsqrt(mean(x²)+eps)` | `ds4_rms_norm` (fp32 in/out) |
| adjacent-pair partial RoPE on the trailing `rope_dim` | `ds4_rope_apply` |
| the **conjugate** rotation on the attention output | the same, `sin_sign = -1` |
| attention over ONE latent KV head, read as both K and V, with per-head sinks | `ds4_attention` (head_dim ≤ 512; V4.1's is 512) |
| the sliding-window causal mask | `ds4_sliding_causal_mask` |
| dense projections against a row-major `[N, K]` weight | `gemm_nt_f16_onednn` |
| **block-diagonal `wo_a` over `o_groups`** | `gemm_bmm_nt_f16_onednn` |

If that is right, the only genuinely new code in the attention path is the FP8→fp16 dequant
already built and verified in Phase 4.

## Pass criteria

1. **Every stage is compared separately**, against a golden dumped at that boundary:
   `qr` (after `q_norm`), `q` (after RoPE), `kv` (after RoPE), `o` (after attention),
   `o` (after the conjugate rotation), `wo_a` output, and the block output. A single end-to-end
   number is not acceptable — it would say something is wrong without saying where.

2. **The golden's stage decomposition is proven equal to the reference's own `forward`.**
   `golden_block.py` asserts `torch.allclose(stage_recompute, Attention.forward(x), atol=0,
   rtol=0)` — exact — so the per-stage goldens are not a parallel reimplementation that could
   drift from the thing being modelled.

3. **Tolerances, justified before the run.** The engine computes in fp16 activations with fp32
   accumulation against an fp32 golden. fp16 carries ~11 bits of mantissa, so a single GEMM
   contributes ~5e-4 relative, and error compounds down a 7-stage chain with two more GEMMs
   after the attention. So: **3e-3 for the first four stages** (one or two GEMMs deep), **5e-3
   for the last two** (three and four GEMMs deep). Anything an order of magnitude above these
   indicates a structural error, not precision, which is what the criterion is for. A stage that
   lands near zero while its neighbours do not is equally suspect and must be explained.

4. **The conjugate rotation is actually tested.** `o (attention)` and `o (unroped)` are separate
   stages, so omitting the `sin_sign = -1` call must fail the second while the first still
   passes. This is the trap the V4 header warns about ("omitting it silently corrupts long
   context") and it must not be possible to pass this phase without it.

5. **`wo_a` is exercised as block-diagonal, not dense.** The grouped stage is compared on its
   own, so treating `[8192, 4096]` as one dense matmul — which has the right output shape — must
   fail at that stage.

6. **No new kernel is added.** `git diff --stat` for this phase must show only the new test tool
   and the CMakeLists line. If the attention required a new kernel, the claim above is false and
   the phase should say so rather than quietly add one.

7. **Clean teardown**: every device allocation freed.

## Explicitly NOT in this phase

The compressor, the indexer, the KV cache across steps, decode, mHC wiring, the layer loop.
Phase 5b answers one question: *does the attention block reproduce the reference, using only
kernels that already existed?*
