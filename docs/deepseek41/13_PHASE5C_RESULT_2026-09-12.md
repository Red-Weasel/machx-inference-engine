# V4.1 port — Phase 5c result: a whole layer runs (2026-09-12)

`ie-ds41-block-test` vs `golden_block.py`, layer 0, T=8, on device. The golden is DeepSeek's own
`Block(0)`, and `golden_block.py` asserts its stage decomposition reproduces `Block.forward`
**exactly** (`atol=0, rtol=0`), so the per-stage goldens cannot have drifted from it.

```
Intel(R) Arc(TM) Pro B70 Graphics — layer 0 FULL BLOCK, T=8, hc_mult=4
[ ok ] hc attn_pre      rel 1.312e-07
[ ok ] hc attn_post     rel 1.470e-07
[ ok ] hc attn_comb     rel 1.877e-07
[ ok ] attn_norm in     rel 1.318e-07
[ ok ] attn out         rel 2.691e-04
[ ok ] hc ffn_pre       rel 2.731e-05
[ ok ] ffn_norm in      rel 2.908e-04
[ ok ] moe out          rel 2.479e-04
[ ok ] BLOCK OUT        rel 2.046e-04
worst stage 2.908e-04 — BLOCK TEST: PASS
```

The mHC coefficients land at **1e-7** — fp32 against fp32, no quantised weights in that path.
Everything downstream of a GEMM sits at 2-3e-4, the fp16-activation floor established in
Phase 5b. Nothing steps out of line.

## A vacuous check, caught here rather than by a gate

The first run of this test reported `moe out rel 9.845e+01` and **passed**, because the
tolerance was `1e10` and the line was annotated "reported only". Two things were wrong:

1. It compared the block's MoE output against `golden_moe.py`'s `moe_out`, which is computed
   from a *different, synthetic input*. The comparison was meaningless regardless of tolerance.
2. A tolerance of 1e10 is not a check. Phase 4's gate failed this build twice for exactly this
   shape of thing — a line that looks like verification and cannot fail.

Fixed properly: `golden_block.py` now recomputes the FFN half with the block's own submodules
and dumps `blk_ffn_in`, `blk_moe_out` and `blk_st1`, with an assertion that the recomposition
equals `Block.forward` exactly. The MoE stage is now a real comparison against the MoE *inside
the block*, and it reads **2.479e-04**.

Worth recording that the end-to-end number was right (2.046e-04) the whole time: the block
output matched because the engine's own MoE output fed it. A green end-to-end number sat on top
of a broken intermediate check, which is precisely why every phase here compares stages.

## The one new kernel in the whole forward path

`ds41_hc_mixes` + `ds41_hc_collapse` (`src/ops/deepseek41_ops.cpp`). V4 and V4.1 compute
identical mHC coefficients, but consume `pre` at different times:

- **V4** fuses the mixes and the stream collapse into one call; the collapse uses the `pre` that
  same call produced (`deepseek4.cpp:3399, 4076`).
- **V4.1** uses each sublayer's coefficients in the *next* one (`Block.forward`, model.py:968):
  attention collapses with the `pre_mix` handed in from the previous layer's FFN, and the FFN
  collapses with the `pre` attention produced.

That is the tech report's **"Single-Pass mHC, a revision of V4's mHC"** pinned to a concrete
line of code. V4.1 therefore needs `pre` as an output and a collapse that takes one from
elsewhere, which `ds4_hyper_connection` — four dispatch paths, collapse fused — does not offer.

**The duplicated Sinkhorn is guarded, not hoped at.** `tests/unit/ds41_hc_parity_test.cpp`
requires the V4.1 copy to reproduce `ds4_hyper_connection`'s `post` and `comb` on identical
random inputs at the real shape, and additionally pins `pre` (which the V4 kernel never exposes)
by requiring `ds41_hc_collapse` with that `pre` to reproduce the V4 kernel's fused collapse:

```
ds41_hc_parity_test: post 1.208e-06  comb 1.823e-06  collapse(pre) 9.531e-07
```

Bit-exactness is deliberately not the bar — the two kernels reduce the RMS sum over
`hc_mult*hidden` in different work-group tilings and fp32 addition is not associative. ~1e-6 is
reduction-order noise.

## Everything that now runs, against the reference

| piece | result |
|---|---|
| read the checkpoint's bytes | bit-exact (Phase 2) |
| bind 96,085 tensors | 0 unclaimed (Phase 3) |
| expert upload, dense FP8 dequant | bit-exact on device (Phase 4) |
| MoE routing | **exact** (Phase 5a) |
| routed experts | 2.0e-4 fp16 act / 3.3e-3 Q8_1 act (Phase 5a) |
| attention, 7 stages | 3.1e-4 (Phase 5b) |
| **a whole layer** | **2.0e-4** (this phase) |

Remaining before tokens come out: layers 1-39 (which add the compressor, the indexer and engram
— layer 0 has none of them), the KV cache across steps, the residency planner for 268.95 GiB of
experts, embed/lm_head, and the decode loop.

## Phase 5 gate: PASS, seven non-blocking findings, all acted on (2026-09-12)

1. **MoE test admitted a w1/w3 swap** (1.90e-2 vs a 2e-2 bar; golden input at std 0.02 where
   the real post-norm input is 0.127). Fixed: realistic input scale, bars 2e-3 / 8e-3.
2. **The parity test could not see Sinkhorn defects** — its random input was already doubly
   stochastic before the Sinkhorn ran, so 19/20/21 iterations, dropped `+eps` and the wrong
   pass order all agreed to 1e-6. Fixed: real layer-0 hc tensors and a post-layer-scale
   stream, and the test now runs 19 and 21 iterations itself and **fails unless they differ
   from the reference at 20** — measured **3.671e-3 and 3.262e-3**, ~300x the 1e-5 bar.
   > **CORRECTED by the Phase 6 gate.** This entry originally read "measured 0.79 and 0.81".
   > Those were **use-after-free artifacts**: the self-validation block had been inserted after
   > the `sycl::free` loop and read six freed USM pointers. The printed values were
   > nondeterministic across runs, and — decisively — on the trivial input the guard exists to
   > reject, the freed-memory arrangement could exit 0 while the correct one exits 1. The
   > frees now come last; three consecutive runs give identical values.
3. Stale "BIT-FOR-BIT" in `deepseek41_ops.cpp`'s header. Corrected to what the test does.
4. **5a's routing check was a host recompute**; the engine's device router had never run
   against V4.1. `ds4_router_topk` now runs in the MoE test: indices exact, weights 0.000000.
   The "bit-equal" overclaim in `10_PHASE5A_RESULT` corrected.
5. The `-1e30` running-max floor in `sparse_attn` is not exercised at T=8 (no fully-invalid
   row exists). **Carried as a named risk into decode**, where a still-filling ring buffer
   produces exactly that row.
6. FFN-site `post`/`comb` were dumped but never compared; the block test freed almost
   nothing. Fixed: both stages pinned (1.5e-4, 1.1e-5); every allocation freed.
7. At T=8 the 128-window is never narrower than the sequence, so 5b exercised only the causal
   half of the mask. Now covered by 6b at T=1,536, where the window is genuinely sliding.
