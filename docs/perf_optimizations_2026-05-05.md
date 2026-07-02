# Performance Optimizations — 2026-05-05 session

Working baseline: 6.54 PPL on built-in 511-tok corpus, 143.5 prefill tok/s @ T=512, 44.7 decode tok/s. Save-state at `docs/save_states/2026-05-05_baseline/`.

## Optimization #1 — XMX M_TILE: 8 → 16 (M_GROUPS=2)
**Status: KEPT.** PPL unchanged 6.54. Prefill +26.2 %.

### Change
`src/ops/gemm_q4k_xmx.cpp` and the `gemv_q_T` dispatcher in
`src/model/qwen36.cpp`:

- `gemm_q4_K_xmx`, `gemm_q6_K_xmx`: `M_TILE_MAX` raised 8 → 16 by stacking
  `M_GROUPS_MAX = 2` joint_matrix accumulators per SG (each TM=8). Both
  groups share one `B_smem` dequant per K-tile, so per-launch weight reuse
  doubles. SLM: A_smem 4→8 KiB, C_scratch 1→4 KiB, total per WG ~44 KiB.
- `gemv_q_T` (Q4_K and Q6_K branches): caller M_TILE 8 → 16 when the XMX
  kernel is selected (still 8 when the scalar fallback is used).
  Halves the per-projection launch count (T=512: 64 → 32).

### Measurement (`./build/tools/ie-bench --ctx 1024 --prefill 512 --decode 0 --profile`)

| metric | baseline | new | Δ |
|---|---:|---:|---:|
| **prefill total** | 3567.95 ms / 143.5 tok/s | 2827.98 ms / **181.0 tok/s** | **+26.2 %** |
| attn_dn_block | 1401.4 ms | 1029.8 ms | −26.5 % |
| moe_routed_multiexpert | 1059.9 ms | 984.3 ms | −7.1 % |
| moe_shared | 553.3 ms | 490.5 ms | −11.4 % |
| dn_qkv_proj | 490.9 ms | 336.3 ms | **−31.5 %** |
| dn_alpha_beta_gate | 424.6 ms | 360.8 ms | −15.0 % |
| attn_full_block | 453.4 ms | 315.4 ms | **−30.4 %** |
| dn_ssm_out | 409.9 ms | 277.2 ms | **−32.4 %** |
| dn_recurrence | 50.0 ms | 45.7 ms | −8.6 % (noise) |

PPL on built-in 511-tok: **6.54** (baseline 6.54). No math regression.

### Decode check (full bench-suite still pending)
Quick smoke at decode=16 didn't show regression (45.1 tok/s vs 44.7 baseline — within noise floor).

### Risk notes carried
- M_GROUPS_MAX=2 fits comfortably in SLM (44 KiB ≤ 192 KiB Xe-core budget). Future bump to M_GROUPS=4 (M_TILE=32) is feasible from SLM alone but may exceed register/occupancy thresholds; tested and regressed — register pressure is the hard wall.
- If a future kernel calls `gemm_q4_K_xmx` / `gemm_q6_K_xmx` with M ≤ 8, it still works correctly — the second M-group's accumulator is computed but not stored (mask check at writeback).

## Optimization #2 — Shared-expert prefill batching (T>1)
**Status: KEPT.** PPL unchanged 6.55. Prefill +1.7 % (181.0 → 183.7 tok/s).

### Change
`src/model/qwen36.cpp`: shared-expert path at T>1 used to D2H-copy the per-token sigmoid gate scalar to the host, then loop `for (t = 0..T-1)` calling 4 single-row ops per iter (gemv gate / gemv up / swiglu / gemv down / scaled_add).  Replaced by one batched call to `gemv_q_T(...T)` for each of gate/up/down (which now uses the M_TILE=16 XMX kernel) + a single `swiglu(T*E_ffn)` + a new `scaled_add_per_token_row` kernel.

`src/ops/moe.cpp`: added `scaled_add_per_token_row` kernel — y[t,h] += scale_per_tok[t] * x[t,h] in a single launch.

`include/ie/ops.hpp`: declared `scaled_add_per_token_row`.

### Measurement
- moe_shared section: 490.5 ms → 406.7 ms (−17 %)
- prefill total: 183.7 tok/s (was 181.0)

## Optimization #3 — Scalar `moe_prefill_gate_up_silu_q4k`: M_TILE 8 → 16
**Status: KEPT.** PPL unchanged 6.54. Prefill +1.8 % (183.7 → 187.7 tok/s).

### Change
`src/ops/moe_fused.cpp` `moe_prefill_gate_up_silu_q4k`: M_TILE 8 → 16. SLM
A_slm grows from 32 KiB/WG to 64 KiB/WG (still ≤ 192 KiB Xe-core budget).
Each WG processes one expert's tokens; for T=512 with K_top=8 across E=256
experts, average n_tok/expert ≈ 16 — M_TILE=16 covers a typical expert
in one outer iteration (was 2 before), halving the weight-read amortization
window.

Tried 32 — regressed (occupancy bound). Tried the same on
`moe_prefill_down_packed_q4k_v2` and `q6k_v2` (M_TILE 8 → 16) — wash, slight
regression. Reverted those two; only the gate/up kernel kept at 16.

## Optimization #4 — Scalar `gemm_q4_K`: M_TILE 8 → 32
**Status: KEPT.** PPL unchanged 6.54. Prefill +1.5 % (187.7 → 191.2 tok/s).

### Change
`src/ops/gemv_q4k.cpp` `gemm_q4_K` (the SLM-tiled scalar Q4_K GEMM, used as
fallback when the XMX shape constraints aren't met — N % 64 ≠ 0 or K % 256 ≠
0): M_TILE bumped from the historic 8 (with comment "tried 16 — slower")
to **32**. Caller-side `M_TILE_SCALAR` in `gemv_q_T` (qwen36.cpp) raised to
match.  SLM A_slm = M × K × 2 = 32 × 2048 × 2 = 128 KiB/WG (well within
192 KiB Xe-core budget when only one or two such WGs are concurrent).

The historic M_TILE=16 regression no longer applies in the current state —
likely because the bigger XMX wins absorbed enough of the prefill load
that scalar-fallback occupancy became less critical.

This kernel is hot for the DN alpha (N=32) and beta (N=32) projections
that don't satisfy the XMX `N % 64 == 0` constraint.

### Measurement (ie-bench T=512 --decode 0)
3-run median: **2677 ms / 191.2 tok/s** — final state.

## Cumulative scoreboard (end-of-session bench-suite, post-Opt #4)

| metric | baseline | current | Δ |
|---|---:|---:|---:|
| **prefill T=512 (ie-bench)** | 143.5 tok/s | **191.2 tok/s** (3-run median) | **+33.2 %** |
| decode short-chat            | 44.84 tok/s | 45.36 tok/s | +1.2 % ✓ |
| decode long-instruction      | 44.83 tok/s | 45.37 tok/s | +1.2 % ✓ |
| decode codegen               | 44.88 tok/s | 45.28 tok/s | +0.9 % ✓ |
| decode math-reasoning        | 44.63 tok/s | 45.25 tok/s | +1.4 % ✓ |
| decode long-context          | 44.75 tok/s | 45.14 tok/s | +0.9 % ✓ |
| **built-in 511-tok PPL**     | **6.54**    | **6.54**    | identical ✓ |

Quality + decode parity preserved end-to-end. Decode actually improved
slightly (~+1 %) — likely because shared-expert path simplification reduced
some launch latency at T=1 too (the `gemv_q4_K_dual_ffn` path is unchanged
but the surrounding flow saw fewer host roundtrips).

### Per-section impact (--profile, T=512)

| section | baseline ms | current ms | Δ |
|---|---:|---:|---:|
| attn_dn_block | 1401.4 | 1019.1 | **−27.3 %** |
| moe_routed_multiexpert | 1059.9 | 987.8 | −6.8 % |
| moe_shared | 553.3 | 384.4 | **−30.5 %** |
| dn_alpha_beta_gate | 424.6 | 341.2 | −19.6 % |
| dn_qkv_proj | 490.9 | 324.9 | **−33.8 %** |
| attn_full_block | 453.4 | 303.8 | **−33.0 %** |
| dn_ssm_out | 409.9 | 295.9 | **−27.8 %** |
| dn_recurrence | 50.0 | 47.8 | −4.4 % |
| **profile-instrumented total** | 3567.95 | 2706.44 | **−24.1 %** |

## Failed attempts (documented for future sessions)

| attempt | result | likely cause |
|---|---|---|
| `gemm_q4_K_xmx` M_GROUPS=4 / M_TILE=32 | 181 → 157 tok/s | register pressure — 4 acc tiles per SG |
| `moe_prefill_gate_up_silu_q4k_xmx` (XMX MoE) M_TILE 8→16 | 181 → 150 tok/s | 8 SGs × 2 acc per SG = 16 acc tiles per WG; existing comment says XMX MoE is already 10–15 % slower than scalar version |
| `moe_prefill_gate_up_silu_q4k` (scalar) M_TILE 8→32 | 187 → 162 tok/s | occupancy / SLM pressure (128 KiB) |
| `moe_prefill_down_packed_{q4k,q6k}_v2` M_TILE 8→16 | 187.0 → 186.4 tok/s | wash — expert n_tok≈16 already covered by 2 chunks; reverted |
