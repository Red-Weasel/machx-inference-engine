# Prefill crown — executable plan (EXECUTED 2026-06-10 ✅ CROWN TAKEN)

> **Outcome:** pp512 **1144 ± 5 vs llama.cpp SYCL master 1064 ± 8 (+7.6%)**,
> same-hour alternating runs, PPL 6.52 at production defaults, decode crown
> held.  Steps 1, 2, 4, 5 executed as written; step 3 (register batching)
> was the decisive move and went further than planned: full-K register
> lattices for BOTH stages + `block_q8_1s` split half-sums that eliminate
> every isum dp4a.  IE_MOE_Q8 is now default-ON (IE_NO_MOE_Q8=1 opts out);
> the SoA repack is default-ON (IE_NO_MOE_SOA=1 opts out).  Detail:
> docs/benchmark_matrix_2026-06-09.md §v1.6.  Historical plan below.


Standing: engine ~946 pp512 (thermally cool: ~964), llama.cpp SYCL master
1088.  Decode crown HELD (84.1 turbo / 81.0 default vs their 81.31).

## Intelligence (from their source + unitrace, all verified)
- Their dp4a == plain byte math (dpct::dp4a) — identical to ie/dp4a.hpp.
  Lowering is NOT a differentiator.
- Their "reorder" (ggml-sycl.cpp reorder_qw_q4_k) is a struct-of-arrays
  split per tensor: [qs × nblocks | scales × nblocks | half2 dm × nblocks].
  Removes the 16 B header interrupting every 144 B of quant stream.
- Their MoE prefill runs mul_mat_vec_q4_K_q8_1_ncols (token-batched in
  REGISTERS, ncols ≤ 8) + quantize_row_q8_1: ~123 ms expert compute where
  ours spends ~326 ms (fp16) / ~310 ms (current int-dot opt-in).

## Engine state (all committed)
- moe_prefill_gate_up_silu_q4k_q8 (opt-in IE_MOE_Q8=1): int-dot stage 1,
  PPL 6.52 correct, 876 vs 919 fp16 — loses 5% on GGUF AoS layout.
- quantize_q8_1 of E5 expert-sorted x_packed wired (ws_moe_xp_q8_).

## Execution steps (next session, fresh GPU)
1. Load-time SoA repack for ffn_gate_exps / ffn_up_exps / ffn_down_exps:
   per-EXPERT region (keep expert_stride_bytes semantics; only the intra-
   region layout changes: qs stream | scales stream | dm stream).  One
   repack kernel at load; PPL-free by construction (same bits moved).
   Gate: PPL identical + loader unit check.
2. Point the q8 stage-1/stage-2 kernels at the SoA streams (qs loads become
   clean contiguous uint4; scales/dm from small side arrays).
3. If still short: ncols-style register batching (their decomposition) for
   the expert kernels.
4. IGC asm dump sanity (IGC_ShaderDumpEnable) that dp4a lowers to idp4a.
5. Order-controlled showdown (new-old-new) + same-hour llama.cpp rerun;
   on win: flip IE_MOE_Q8 default, update RELEASE.md + matrix, then the
   public writeup.

## Measurement rules (hard-learned)
- GPU swings ±40 tok/s when heat-soaked: A/Bs MUST be order-controlled
  or fresh-boot.  First run after a rebuild pays JIT inside the timed
  region — always discard run 1.
