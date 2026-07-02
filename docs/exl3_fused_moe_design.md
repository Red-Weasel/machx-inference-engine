# Fused EXL3 MoE kernel — design (perf pass for the 80B qwen3next EXL3)

**Status (2026-06-20):** designed, not yet built. The 80B EXL3 forward is CORRECT but the MoE
uses a slow per-expert loop (`src/model/qwen3next.cpp`, the `gate_dt==kEXL3` branch): per token,
per active expert (K=10), 3× `gemv_exl3_forward` (gate/up/down) + swiglu + scaled_add →
~2400 kernel launches/token at decode across 48 layers. This fuses that to ~96 (2/layer = **25×**).

**Validation oracle = the existing slow per-expert loop** (it produces correct output —
"Paris.", Rayleigh answer). Build the fused kernels behind an env flag, diff their output vs the
slow loop on the SAME model; flip default once bit-close. No numpy oracle needed.

## Reuse map (exact)
- **Q4_K fused MoE (the structure to mirror):** `src/ops/moe_fused.cpp`
  - Stage 1 gate+up+swiglu decode: `moe_decode_gate_up_silu_q4k_impl<SOA>` (lines 82–197).
    Grid `nd_range<2>({K_top, wgs_n*WG_ITEMS},{1,WG_ITEMS})`; group(0)=expert k, group(1)=output
    chunk; SG_SIZE 16, N_PER_WG 32, WG_ITEMS 512; cooperative x[H]→SLM (116–117); per-expert weight
    view via stride (121–126); SG-reduce gate/up → `silu(g)*u` (188–194).
  - Stage 2 down+reduce: `moe_decode_down_q4k_impl<SOA>` (225–315). Grid `nd_range<1>(n_wgs*WG_ITEMS)`,
    one WG per 32-col H chunk; **K_top loop INSIDE the kernel** (272), h preloaded to SLM (258–260),
    accumulate `topk_w[k]*down`. This is the batching pattern to copy.
- **EXL3 decode primitive:** `src/ops/gemv_exl3.cpp`
  - `gemv_exl3` (51–139): one WG per tile-column (TN=N/16), 16 SG × 16 lane, tile=16×16=256 weights,
    `psz=bits*256/32` u32/tile, trellis addr `codes_u32[(ki*TN+ni)*psz + woff]`. Inverse tensor-core
    tile permutation built in SLM (80–96), per-lane `my_tt` loop-invariant in ki. 8-way ILP decode
    loop (120–129) — **copy the bit-extraction + `decode_cb0_dev` line-for-line (107–112)**.
  - `gemv_exl3_forward` (164–185): `had128(A⊙suh) → gemv_exl3 → had128(acc)⊙svh`.
  - Hadamard `src/ops/hadamard.cpp` (23–66): one WG/128-block, 7 radix-2 butterflies, pre=suh/post=svh.

## Key constraint
**suh is PER-EXPERT** (`gate_suh + e*K`, etc.) → the input Hadamard `had128(x⊙suh)` CANNOT be hoisted/
shared across a token's experts. Mitigation: keep Hadamard as separate calls (Option A — cheap,
orthogonal, easy to verify) rather than inlining it into the trellis kernel. So per stage:
hadamard(x⊙suh_e) per expert (still 1 call vs K gemvs), then the fused trellis-decode kernel.

## The 2 kernels (decode; new file `src/ops/moe_exl3.cpp` + `include/ie/ops.hpp` decls)
1. `moe_decode_gate_up_silu_exl3(x[T,H], topk_idx/w, gate_{trellis,suh,svh}, up_{...}, h_out[T*K,E_ffn],
   T,H,E_ffn,K_top, gate_bits,up_bits, expert_stride)`. Grid ~ (T, K_top, E_ffn/32). Per WG: token t,
   expert e=topk_idx[t,k]; load x→SLM; gate trellis-decode (16 cols/SG) + up; SG-reduce; `silu(g)*u`
   → h_out[t*K+k, n]. Template on bits (1–8) or runtime-branch.
2. `moe_decode_down_exl3(h_in[T*K,E_ffn], topk_idx/w, down_{trellis,suh,svh}, y_out[T,H], ...)`.
   Grid (T, H/32). Per WG: token t, 32 H-cols; K_top loop inside; `acc += topk_w[k]*gemv_exl3(h[k],down_e)`;
   write y_out[t,n:n+32] (memset y first, as the slow path does). SLM for h ≈ K*E_ffn halves (~2.5 KiB) ✓.

Prefill (T>1) variant later — decode is the win first. Mind per-(layer,proj) variable bits (gate/up/down
may differ; layer 3 attn was 4-bit, experts 5-bit) → bits come from `lw.{gate,up,down}_bits`.

## Blockers (all manageable)
- Trellis bit-extraction / tensor-core perm: copy gemv_exl3.cpp:80–112 EXACTLY; diff vs slow loop.
- SLM budget fine (K*E_ffn halves). Variable bits → template or runtime branch.

## Wire-in
`src/model/qwen3next.cpp` EXL3 MoE branch: replace the per-(t,e) loop with the 2 kernel calls (keep the
slow loop behind `IE_EXL3_MOE_SLOW=1` as the diff oracle). Then shared expert/router unchanged.
