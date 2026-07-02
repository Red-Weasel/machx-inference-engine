# Crown — Qwen3.6-35B-A3B (`qwen35moe`) — Living Authority Doc

> Single source of truth for the Crown's optimization state. Living: update at every milestone;
> if reality and this doc disagree, fixing the doc is part of the change. Cross-arch roadmap stays
> in `MASTER_DEV_PLAN.md` (§7 in-flight, §2 current state); this is the per-arch deep state.
>
> **Last substantive update:** 2026-06-26 (created). Numbers carry their own dates; treat any
> figure without a date as historic/uncertain.

---

## 0. Identity

| Field | Value |
|---|---|
| Family | Qwen3.6-35B-A3B — "the Crown", the locked daily-driver / perf-flagship |
| Arch tag | `qwen35moe` → Engine dispatch `kQwen35Moe` (auto-detected by `ie run/serve`) |
| Shape | Hybrid **gated-DeltaNet (linear/recurrent) + full-attention + 256-expert MoE** |
| Layers | 40, pattern `[L,L,L,F]×10` → **30 Gated-DeltaNet + 10 full-attention** (`full_attn_interval=4`) |
| MoE | 256 experts, top-8 + 1 shared; **3B active / 36B total** params per token |
| Dims | `hidden=2048`, `n_q_heads=16`, `n_kv_heads=2` (GQA), `head_dim=256`, `vocab=248320`, `max_pos=262144` |
| Attn quirks | partial rotary 0.25 (`rope_dim=64`), QK-Norm, attn **output gate** (sigmoid), no QKV bias |
| DeltaNet | `ssm_inner=4096` (32 v-heads × 128 `ssm_head_dim`), conv1d, gated recurrence |
| Forward path | `src/model/qwen36.cpp` (`QwenModel::forward`); config `include/ie/qwen36.hpp` (`QwenConfig`) |
| Main GGUF | `~/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf` → **symlink** into `~/models/lmstudio-community/Qwen3.6-35B-A3B-GGUF/` (21,166,757,728 B). Experts Q4_K, ffn-down mixed Q4_K/Q6_K per layer, lm_head/output Q6_K |
| MTP head GGUF | `~/models/.../Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (unsloth) — **head blk.40 only**; main quant incompatible with loader (see §2) |
| **One-line standing** | **Prefill WIN 1.14× + decode PARITY** (re-verified 2026-06-25 — the old 1.10× decode "win" is dead; llama-SYCL caught up). The structural DeltaNet moat holds (no competitor runs this arch on B70); the *speed* claim is prefill + day-one-on-new-arch, not decode. |

**Symlink landmine:** the lmstudio dir is the *live target*, not a deletable duplicate. Deleting it
dangles the gate symlink and breaks `gemv_q4k_test`, `moe_test`, `tokenizer_test`, `ie-perplexity`'s
default, and the §6 quickstart. The `-lmhead-q4k` variant (21,035,644,768 B) scores **6.54 — NOT a
substitute** for the 6.4527 gate. (`reference_crown_gguf_symlink`.)

---

## 1. Envelope (the benchmark contract)

| Axis | Contract |
|---|---|
| Hardware | 1×Intel Arc Pro B70 (Battlemage / BMG-G31, 32 GB GDDR6, **608 GB/s**, 367 INT8 TOPS, ~183 FP16 TFLOPS XMX). PCIe Gen4 x8 |
| Software | oneAPI 2026.0 / NEO 26.18, kernel 6.17.0-35 (clean-box numbers below). AOT `-fsycl-targets=intel_gpu_bmg_g31` |
| Model+quant | Qwen3.6-35B-A3B **Q4_K_M** (the lmstudio crown GGUF) — production default |
| Target modes | single-stream **decode** (short/mid ctx) + **short/mid prefill** (≤~1–2K). Long-ctx prefill is a *known weakness* (§3/§8), not in the win envelope |
| Competitor | **llama.cpp SYCL master** (icpx, `-DGGML_SYCL=ON`), same GGUF / same card. SYCL is llama's fastest B70 backend on this arch (Vulkan regressed to ~775/35 and cannot run the DeltaNet path well) |
| Quality gate | PPL ≤ **6.57** (hard). Production default **6.4527** |
| A/B method | `ours = ie-bench --prefill 512 --decode 128` ≡ `llama = llama-bench -p 512 -n 128`. Discard cold/JIT first run for both (ours `--warmup 4`); order-controlled (new-old-new); GPU idle-clocked ≤600 MHz + swap≈0 before each load; one workload at a time |

---

## 2. Correctness

**PPL gate.** `./build/tools/ie-perplexity` (defaults to the crown symlink) — production NLL
**1.864495 / PPL 6.4527**. On the 2026-06-22 clean corpus (post oneAPI-2026.0) it reads **5.4163**,
**bit-identical to pre-update** (the oneAPI update changed no numerics). Streaming PPL is
deterministic → bit-exact A/Bs are possible. `--prefill-chunk 256/512` exercises the MoE-prefill
kernels but carries DeltaNet HW noise ±0.005 nats run-to-run (band comparisons only, see landmine 3).

**Reference oracle.** Per-layer residual cosine vs llama.cpp dumps = 1.000000 on the dense sibling;
crown rides the same validated forward. Deterministic greedy gen is bit-exact (NLL constant fixed).

**Silent-bug landmines (every one degrades quality invisibly if reverted):**
1. **`rope_partial` cross-work-item race** (`34e4c01`) — was *latent in the crown* (immunity was
   subgroup-lockstep scheduling luck, not correctness); exposed by dense head shape. Fix = one-item-
   owns-pair. Touching the partial-RoPE work-mapping re-arms it.
2. **qwen2 pre-tokenizer contraction alternative** (`f508c01`) — missing it scored PPL 6.52; fix →
   **6.45**. Found by Seal dogfooding, not a unit test.
3. **DeltaNet recurrence non-determinism on BMG-G31** (`docs/known_bugs.md §1`) — single-byte state
   divergence that amplifies to PPL ~74 collapse if the recurrence runs too many successive steps in
   one call. **Software-irreducible (HW-level FMA pipeline non-determinism).** Production workaround =
   external prefill chunking at **T≤512** (raised 256→512 on 2026-06-20: bug **not reproducible** on
   NEO 26.14/26.18 + kernel 6.17.0-35; `ie-bug-monitor` 24/24 clean chains, PPL `--prefill-chunk 512`
   ×3 bit-identical). **Do NOT re-investigate the root cause** (exhaustively bisected, Steps 18–27).
   `IE_QWEN35_PREFILL_CHUNK` reverts the cap.
4. **The 12 Qwen3.6 arch quirks** (`project_qwen36_quirks`) — partial-RoPE global vs local, QK-norm,
   output gate, weightless paths, etc. Each silently degrades quality if dropped.
5. **MoE accumulation order** (qwen3next sibling lesson, `1ed07e0`) — the shared-expert add path must
   not read a stale `moe_y`; crown's contract differs from qwen3moe/qwen3next.
6. **Symlink** (§0) — infra-care, not numeric, but breaks the gate battery.

Default path is **byte-identical** through every optimization below (all MTP / experimental code is
additive and only reached via separate tools). Gate battery: ctest 30/30 + crown PPL 6.4527 bit-exact.

---

## 3. Pareto frontier (measured matrix)

**Clean box, fresh both engines, same GGUF / same single card.** Primary source:
`docs/benchmark_results_2026-06-22.md` (2026-06-22, oneAPI 2026.0 / NEO 26.18). Historic TOTAL-CROWN
row from 2026-06-10 kept for the turbo ceiling.

### Decode (tok/s, `tg128`)

| Regime | Ours | llama SYCL | Verdict | Bottleneck class |
|---|---:|---:|:---:|---|
| **short ctx, RE-VERIFIED 2026-06-25** | **78.1** | **76.9** | **PARITY** ⚠ | weight-BW (both at the same memory wall) |
| short ctx (depth 0), 2026-06-22 *(superseded)* | 79.6 | 72.5 | ~~1.10×~~ | llama-SYCL has since caught up |
| short ctx, 2026-06-10 **turbo** (q4k lm_head, PPL 6.61) | 84.1 | 81.31 | turbo ceiling | user-selectable |
| at depth (512→16K) | 81.3 → 45.9 | — | graceful | normal KV growth |

> ⚠ **CORRECTION (`docs/COMPETITIVE_SCORECARD_2026-06-25.md §4`):** the strict apples-to-apples decode on the
> identical Q4_K_M file is now **PARITY (78.1 vs 76.9)**, NOT a win — current llama-SYCL caught up. The prior
> 1.10× (06-22) and the "84 vs 55" framing are **dead** for the decode claim. Decode is bandwidth-bound (GEMVs
> ~70%, DeltaNet recurrence only 3.4%) → both engines hit the same memory wall. **The DeltaNet win that HOLDS is
> Crown PREFILL 1.14×** (below); decode leadership would require MTP, which currently nets a slowdown (§ MTP).

### Prefill (tok/s, `pp` at T)

| T_pp | Ours | llama SYCL | Verdict | Bottleneck class |
|---:|---:|---:|:---:|---|
| 512 | **1116–1123** | 1017–1023 | **WIN 1.09×** | quant GEMM (in-L2 regime), int-dot MoE |
| 1024 | **1160** | 1011 | **WIN 1.15×** | same |
| 4096 | 293 | **962** | **LOSS 0.30×** | DeltaNet chunk-cap + MoE GEMM (naive; <gate) |
| 8192 | 285→**348** | **996** | LOSS 0.35× (was 0.29×) | **tile lever** (attn) — residual MoE-bound |
| 16384 | 201→**316** | **902** | LOSS 0.35× (was 0.22×) | **tile lever** (attn) — residual MoE-bound |

> **Long-ctx tile lever (2026-06-26, `IE_QWEN36_NO_FA2_TILE` opt-out, gate ctx≥6144):** ported the same
> Gemma hd256 wide-tile kernel that flipped the 27B 16K to a win. Crown 6.4527 bit-exact (decode-path),
> coherent, ctest 30/30. **Self: pp8192 +22%, pp16384 +57%** (201→316-class recovery, attention
> portion). **BUT crown long-ctx prefill REMAINS a LOSS (~0.35×)** — unlike the dense 27B (attention-
> bound → tile flipped it to a WIN), the crown's long-ctx collapse is **MoE-GEMM-bound**: llama's MoE
> prefill (coopmat) does 902–996 tok/s at 8–16K; our int-dot MoE prefill is the dominant residual cost.
> The tile recovers attention only. **To win crown long-ctx prefill needs the MoE-prefill GEMM lever**
> (oneDNN/coopmat for the crown MoE) — the real next frontier, NOT attention.

Historic same-hour 2026-06-10 (TOTAL CROWN): pp512 **1144 ± 5 vs 1064 ± 8 (+7.6%)**.

### MTP self-spec-decode (the differentiator axis)

| Mode | net | lossless | accept | verdict |
|---|---:|:---:|---:|---|
| Crown MTP (K=4), 2026-06-25 | **0.36×** | **YES** (token-for-token == greedy) | **1.88/round** | capability shipped, **NOT a speed win** (248K-vocab verify overhead; decode is PARITY so MTP *would* be the lever to win it but nets a slowdown here — re-verified 0.81× even on 27B) |

**Bottom line (RE-VERIFIED `COMPETITIVE_SCORECARD_2026-06-25.md`):** Crown is a **prefill WIN (1.14×
re-verified; 1.09–1.15× ≤1K) + decode PARITY** — NOT the "both-axes win" earlier memory claimed (llama-SYCL
caught up on decode). The defensible speed claim is **prefill leadership + the structural DeltaNet moat** (no
competitor runs this arch on B70 — Vulkan lacks shaders, OpenVINO preview-only). Decode parity → MTP is the only
path to a decode win and it isn't there yet (0.36× net). Win envelope: short/mid-ctx prefill; long-ctx prefill is a real loss.

---

## 4. Bottleneck map

**Decode** (GPU tot ~11.6 ms/tok; profiler-relative ranking, 2026-06-22):

| Kernel | % wall | Role | Class |
|---|---:|---|---|
| `gemv_q4k_q8` | 13.9% | MoE gate/up int-dot decode | weight-BW / ALU |
| `gemv_q6k_huge_q8` | 11.3% | **lm_head** (248K-vocab, 1 call) | weight-BW |
| `moe_dec_gate_q4k` | 10.5% | MoE gate decode | weight-BW |
| `gemv_q6k_med` | 9.2% | mid GEMV (ssm_out etc.) | ALU |
| `fa2_partial` | 7.9% | FA-2 split-K decode attn | KV-BW / latency |

Decode is a **long tail of small latency-bound GEMVs + the one huge lm_head** — not launch-gaps
(census: ~14 ms kernels + ~1 ms gaps). The fp16 GEMVs already run **238–350 GB/s effective (near
ceiling)** → **ALU was never the decode limit**; the win came from killing the tail (fusion +
int-dot on the few ALU-bound Q6_K kernels + Q8 riding the norm kernels with zero quantize launches).
Roofline class: **weight-BW-bound with a launch/occupancy long tail**, pushed near the 608 GB/s
ceiling by the SoA repack + register lattices.

**Prefill** (historic kprofile): MoE expert FFNs dominate — int-dot stage-1 q8 **157 ms (37%)**,
`gemm_fp16`+dequant ~68 ms, `dn_recurrence` ~47 ms. In the short-ctx regime the data is L2-resident
→ we win. **Long ctx:** the DeltaNet recurrence is inherently sequential across `T≤512` chunks; at
16K that is 32 serial chunks while llama's attention prefill parallelizes → the 0.20–0.30× collapse.

---

## 5. Hypothesis ledger

Every lever tried. **KEEP** = shipped/default. **REVERT** = banked negative, do-not-retry. **SPECIALIZE**
= wins one regime only. **DEFER** = correct but low-ROI now.

### KEEP (shipped, default-on unless noted)

| Lever | Mechanism | Result | Decision | Gate/commit |
|---|---|---|---|---|
| Per-expert **SoA repack** at load (qs‖scales‖dm) | aligned contiguous int32 quant reads; each scale/min touched once | neutral *alone* but the enabler for int-dot | KEEP | `IE_NO_MOE_SOA` opt-out; `quant_soa.hpp` |
| `block_q8_1s` activations + **full-K register lattices** int-dot MoE prefill | s0/s1 split half-sums give exact K-quant corrections w/ zero isum dp4a; weights nib-masked into regs once per (expert,col) | stage-2 down 204→101 ms, stage-1 189→157 ms; **pp512 +7.6%** | KEEP (default-on) | `IE_NO_MOE_Q8` opt-out; needs `H%512==0` + `E_ffn==512` |
| **Two-stage T=1 router** | single-WG occupancy starvation 2.6 ms→~0.1 ms | **+13% decode** | KEEP | v1.4 |
| **Parallel MoE router** (packed-key max-reduce) | serial 8×256 top-k scan on one lane was ~4 ms/tok | decode **52.4→66.2 (+27%)** | KEEP | `0705f7a`/`31065f8` |
| v1.4 **fusion campaign** | qkv split/repeat/L2norm 5→1; shexp 5→2; conv1d T==1 in-kernel; alpha/beta+g_beta single-WG | decode **66.2→79.4** | KEEP | `a1e309e`→`f25840b`; `IE_FUSE_*` |
| v1.5 **Q8 rides the norm kernels** + int-dot on ALU-bound Q6_K GEMVs | `rms_norm_f32w_q8`/`gated_rms_norm_q8` = zero quantize launches; `ie/dp4a.hpp` on lm_head/ssm_out | decode → **84.1 turbo / 81.0 default** | KEEP (default-on) | `IE_NO_Q8_DECODE` opt-out; `57f055a`/`b2dbec6` |
| `gemv_q6_K_slm` (SLM-slab Q6_K GEMV) | no shuffles, algebraic fold | closed the 50 tok/s gate at full quality | KEEP | `IE_NO_Q6K_SLM` kill |
| down-kernel `M_TILE` 8→16 | amortize weight read | pp512 925→**938** | KEEP | — |
| dequant→fp16 + `gemm_fp16` for T≥64 projections | E1 prefill | prefill arc 203→900 (+343%) | KEEP | `IE_NO_DEQ_FP16` opt-out |
| DeltaNet chunk cap **256→512** | bug not reproducible on current stack | long-prefill 1.08–1.15× on >256-tok | KEEP (conditional) | `IE_QWEN35_PREFILL_CHUNK` revert |

### REVERT (negative — do NOT retry)

| Lever | Why it failed | Note |
|---|---|---|
| `vec<half,2>` packed fp16 math | IGC won't emit packed-rate FMAs | −2.7% |
| `gemm_fp16` double-buffer / BK=64 (E3) | tiles already latency-hidden by WG multithreading | −9.3% / −21% |
| **gate_up XMX** full rewrite | correct PPL but strided SLM B-tile feed + occupancy | **3.4× slower**; `IE_MOE_XMX=1` opt-in only; do NOT retry without working 2D block loads |
| Q4_K projection int-dot | already BW-bound (238–350 GB/s) | perf-neutral |
| `gemv_q4_K` A-bypass | q6k-bypass lesson does NOT transfer | −19% |
| cross-layer residual+norm fusion | measurement vetoed the kernel-doc prediction | −1.5 decode / −31 pp512 |
| header vector loads (E4) | neutral | banked |
| naive dp4a MoE port into tiles | needs the SoA repack *first* | −9% (superseded by the SoA+lattice KEEP) |
| **oneDNN matmul** (`IE_ONEDNN`) | our `gemm_fp16` already matches Intel's lib at E1 shapes | neutral — **the prefill gap is NOT the GEMM library**; kept opt-in |
| Q8 decode activations *alone* | ~90 quantize launches/tok ate the savings | perf-neutral (66.0 vs 66.2) → only won once **fused into norms** (became a KEEP) |
| **Long-ctx chunk-cap lift T≥1024** | TESTED + FALSIFIED 2026-06-26: +6.7%@1024 / 0.40× REGRESSION@2048; "fix" is timing-luck + DEVICE_LOST risk (Triton #6658) | do-NOT-retry; keep default 512 (§8) |

### SPECIALIZE

- **Multi-expert single-launch MoE prefill** dispatched only for **64 ≤ T < 2048**: T<64 setup
  (sort + 4×H2D + mostly-empty WGs) > work; T≥2048 fused-per-token wins on data locality (a token's
  working set stays cached vs touching all 256 experts spread across DRAM). See §6.
- `IE_ENABLE_MOE_DOWN_TILE` / `moe_prefill_down_packed_*_v2` — down-proj tile variant.

### DEFER

- **MTP spec-decode as a speedup** — lossless **proven** (token-for-token, accept 1.88), but net
  **0.36×**: the 248K-vocab lm_head paid K× in the draft + `verify(T=K)` can't be amortized → the worst
  MTP case. Decode is PARITY (not a win), so MTP *would* be the lever to take the decode lead, but at
  0.36× net (re-verified 0.81× even on 27B) it does not help yet. Capability shipped; speed path
  **deferred** until the verify is cheaper. Perf-win path if ever needed: (1) head
  lm_head reuse main int-dot Q6_K `output_`; (2) batched int-dot verify (`gemv_q4_K_q8s_batched`,
  `96394d7`); (3) host-argmax → GPU. Commits `14596da`→`f534cff`→`d15bbcd`; `project_crown_mtp_port`.

---

## 6. Shape dispatch

| T / regime | Path | dtype guard | env gate |
|---|---|---|---|
| **T==1 (decode)** | `gemv_q` (T=1) int-dot/Q8; fused MoE `moe_decode_gate_up_silu_q4k` → `moe_decode_down_q4k`/`_q6k`; FA-2 split-K `fa2_partial` (optional INT8-KV); Q8 norms; int-dot lm_head `gemv_q6_K_q8` | `can_fuse_moe`: gate+up Q4_K, down Q4_K or Q6_K | `IE_NO_Q8_DECODE`, `IE_NO_MOE_SOA` |
| **64 ≤ T < 2048 (prefill sweet spot)** | counting-sort + `moe_gather_rows` + `quantize_q8_1s` → `moe_prefill_gate_up_silu_q4k_q8` (stage 1) → `moe_prefill_down_packed_q4k/q6k_q8` (stage 2); `gemm_fp16` projections (T≥64); tiled FA-2 prefill | q8 lattice needs `H%512==0` (stage 1) + `E_ffn==512` (stage 2); else fp16 fallback | `IE_NO_MOE_Q8`, `IE_ENABLE_FA2_PREFILL_TILED`, `IE_ENABLE_MOE_DOWN_TILE`, `IE_MOE_DN_PK4K/PK6K` |
| **T ≥ 2048** | **fused-per-token** MoE (data locality) | same | — |
| **prefill chunking** | external chunk **T≤512** (DeltaNet recurrence cap, §2 landmine 3) | — | `IE_QWEN35_PREFILL_CHUNK` |
| **dtype mix** | per-layer ffn-down is Q4_K *or* Q6_K → branched; lm_head Q6_K (Q4_K turbo variant = PPL 6.61) | `w.ffn_down_exps.dt` branch | — |

Full env-gate set (`qwen36.cpp` / `moe_fused.cpp`): `IE_NO_MOE_Q8`, `IE_NO_MOE_SOA`, `IE_MOE_Q8`,
`IE_MOE_XMX`, `IE_NO_Q8_DECODE`, `IE_Q8_DECODE`, `IE_NO_DEQ_FP16`, `IE_ONEDNN`, `IE_NO_XMX`,
`IE_ENABLE_FA2_PREFILL_TILED`, `IE_ENABLE_MOE_DOWN_TILE`, `IE_MOE_DN_PK4K`, `IE_MOE_DN_PK6K`,
`IE_FUSE_SSM_AB`, `IE_FUSE_RES_RMS`, `IE_FUSE_FFN_GATE_UP`, `IE_QWEN35_PREFILL_CHUNK`,
`IE_NO_Q6K_SLM`, `IE_ENABLE_GEMM_Q4K_ESIMD` (do-not-use, ESIMD ban), `IE_MAX_LAYER`/`IE_TRACE` (debug).

---

## 7. Layout / scheduler state

- **Weights:** load-time per-expert **SoA repack** of the 3 expert tensors (qs‖scales‖dm streams),
  `expert_stride` preserved, all 7 MoE kernels layout-templated on a runtime `soa` flag
  (`quant_soa.hpp`; `repack_moe_q6k_soa_host` for Q6_K down). Opt-out `IE_NO_MOE_SOA`.
- **Activations:** `block_q8_1s` (48 B: `d,s0,s1,pad,qs[32]`) → exact Q4_K min / Q6_K per-16 scale
  corrections via s0/s1, zero isum dp4a; `quantize_q8_1s` once per layer.
- **Fusions:** qkv split/repeat/L2norm 5→1 launch; shexp chain 5→2 (SiLU + gate-scalar + scaled-accum
  folded into GEMVs); conv1d T==1 state writeback in-kernel; alpha/beta dual + g_beta single-WG;
  residual+RMS (`IE_FUSE_RES_RMS`); ssm a/b (`IE_FUSE_SSM_AB`); ffn gate/up (`IE_FUSE_FFN_GATE_UP`).
- **Scheduler:** in-order queue; H2D writes ride the queue **without per-call `.wait()`** — only the
  top-k host read blocks (then 4 H2D queued + one wait). `enable_profiling` is **opt-in**
  (`IE_QUEUE_PROFILING`, default OFF — always-on regressed nothing real but is profiler-only).
- **KV cache:** fp16 default; INT8-KV supported (shipped/validated on Coder, available here).
- **MTP head (additive, tool-only):** blk.40 hybrid — attn/eh_proj/lm_head **fp16**, MoE experts stay
  quantized; `w_lm_head` = model `output_` shared dequant; per-position DeltaNet checkpoint for
  rollback-free commit.

---

## 8. Open frontier

**Standing verdict.** Crown **short/mid-ctx PREFILL is WON (1.14×)**; **decode is PARITY** (re-verified
2026-06-25: 78.1 vs 76.9 — llama-SYCL caught up) and at/near the BW ceiling (kernels run 238–350 GB/s
effective on a 608 GB/s part; recurrence only 3.4%). The int-dot MoE + SoA repack + norm-fusion campaign
took decode 44.7→84.1 (turbo) in two days — a huge self-improvement, but llama matched it → now a TIE, not
a win. **Further decode micro-efficiency = commoditized grind** (defend, don't chase); the moat is the
*structural* "no competitor runs DeltaNet on B70" + prefill leadership, not raw decode tok/s.

**#1 — LONG-CONTEXT PREFILL LOSS → ✅ REOPENED + WON 2026-06-26 eve (oneDNN large-M MoE, opt-in, 1.46×).**
The "small-M-bound, no kernel can amortize, oneDNN is a DEAD END" verdict below was **WRONG.** It assumed the
MoE ran per-512-chunk (M=16). It does NOT — the MoE is computed **full-T per `forward()` call**; the 512 is the
ENGINE feeding `forward()` in 512-token slices (for §1 DeltaNet safety). The real blocker was a **`T<2048`
fused-MoE dispatch cliff** (`qwen36.cpp`): above it the code serialized **per-token (M=1)** → THAT is what made
the chunk-cap "2048 regression" (not a fundamental limit), and it kept the large-M regime (M≥256) **unreachable.**
THE FIX (validated): feed crown a **big engine prefill chunk (8192 → M=256)**, made §1-safe by **internally
sub-chunking the DeltaNet recurrence to ≤512 launches** (bit-identical sequential-scan split; no-op at T≤512;
strictly safer than the falsified chunk-cap lift), + raise the fused dispatch ceiling/scratch, + a per-expert
**oneDNN MoE** (`dequant_q4_K/q6_K_to_Bt` → fp16 Bt + `gemm_fp16_onednn`). MEASURED (REAL ~12K wikitext prompt,
real routing, `ie-prompt-bench`): big-chunk + int-dot MoE **REGRESSES** 763→334 (int-dot re-streams weights ~per
row); big-chunk + **oneDNN** MoE = 726.9→**1058.2 = 1.46× WIN.** vs llama-SYCL `llama-bench` crown pp8192 =
**910** (synthetic tokens = dummy routing, llama's BEST case) → our **REAL-routing 1058 still WINS 1.16×**; the
gap is larger in reality because dummy routing (only ~8 experts hit) hides the MoE cost, and llama's real-routing
long-ctx prefill is far worse (`llama-cli -f` CPU-stalled ~20 min on crown's DeltaNet — couldn't complete a clean
real-prompt run). Needle-recall correctness PASS; crown 6.4527
bit-exact (decode untouched); ctest 30/30. Gated **opt-in** `IE_QWEN36_MOE_ONEDNN` (+ `IE_QWEN36_MOE_PREFILL_MAXT`,
`IE_QWEN36_MOE_FUSED_CEIL`, `IE_QWEN36_DN_RECUR_CHUNK`).
**✅ SoA PRODUCTION PATH DONE (decode-safe):** new additive kernel `dequant_q4_moe_soa_to_Bt` (crown's packed-scale
SoA Q4_K → fp16 Bt; mirrors `dequant_q6_K_soa_to_Bt` + `dequant_q4_K_to_Bt`) + the oneDNN branch is now SoA-aware
(`dq_bt`: gate/up→`dequant_q4_moe_soa_to_Bt`, down→`dequant_q6_K_soa_to_Bt`; AoS fall-through). So oneDNN prefill
runs on the **DEFAULT SoA experts** — no `IE_NO_MOE_SOA` needed → **decode stays SoA, zero regression.** MEASURED
(SoA, real 12K prompt, `ie-prompt-bench`): **prefill 727→1079 = 1.48× WIN, decode 56.55 (vs AoS 52.8 — penalty
GONE)**, needle-recall PASS. **✅ SHIPPED DEFAULT-ON (2026-06-26, lead-approved):** on by default for long-ctx
crown configs (`max_ctx≥8192`); opt-out `IE_QWEN36_NO_MOE_ONEDNN`; short-ctx (`max_ctx<8192`) allocates nothing
and is unchanged. The engine feeds `pf_chunk=8192` and sizes the workspace to `min(8192,max_ctx)` at LOAD
(engine.cpp) — the latter is the fix for the chunk-vs-ctx gating bug (`forward()` self-`ensure_workspace(T)`
decided `moe_onednn_on` from the per-forward CHUNK, so a [4096,8192) prompt missed the lever + hit the
per-token cliff). VALIDATED `ie run --ctx 8192` no-env: **1167 tok/s** (vs opt-out int-dot 890 = 1.31× on a
4986-tok prompt), needle-correct, 6.4527 + ctest green. Long-ctx PPL Δ+0.15% (decode/gate bit-exact). The
historical "FALSIFIED" analysis below is kept for the record (right about int-dot, wrong about oneDNN at large-M).

- *Observation (updated 2026-06-26):* pp8192 0.35×, pp16384 0.35× vs llama (was 0.20–0.30×). The
  **attention portion is now fixed** by the hd256 wide-tile lever (`IE_QWEN36_NO_FA2_TILE` opt-out,
  +57% @16K, §3) — but crown long-ctx prefill is STILL a loss because the **dominant residual cost is
  the MoE prefill GEMM**: llama does 902–996 tok/s @8–16K (coopmat MoE), our int-dot MoE prefill trails.
- *Mechanism (REFINED 2026-06-26 — the oneDNN-port lever is FALSIFIED by the math):* the MoE residual is
  **small-M-bound, NOT kernel-choice-bound.** Crown = **256 experts, top-8**, and prefill runs at the
  **T≤512 DeltaNet chunk** → rows/expert = 512·8/256 = **16**. At 16 rows/expert the expert weights are
  read once and reused only 16× → weight-BW-bound; **no GEMM kernel (int-dot OR oneDNN/coopmat) can
  amortize** (this is exactly the Gemma "T≈512 small-M starvation" trap). llama hits **256 rows/expert**
  because it batches the MoE over the FULL 8K prefill (M=8192·8/256). **So a straight oneDNN MoE port is
  a DEAD END at the current chunk size** — do not build it. The real lever is making the MoE GEMM
  **large-M**, which requires EITHER (i) **lift the DeltaNet chunk cap** (T≤512→≥2048; bigger M + fewer
  serial recurrence steps — but the §1 HW-bug area, box-freeze risk, **user-gated**), OR (ii)
  **decouple MoE batching from the recurrence chunk** (accumulate post-attn-norm hidden across chunks →
  one large-M MoE GEMM per layer; an L-effort prefill-loop restructure on the sacred file). THEN oneDNN
  /coopmat wins. Both are real but non-trivial; the oneDNN kernel alone is not the lever.
- *Chunk-cap lift — TESTED + FALSIFIED 2026-06-26 (user-authorized "push through"):*
  - **SAFETY (clean on this box):** `ie-perplexity --prefill-chunk 1024 ×3` and `2048 ×3` =
    **bit-identical each, no PPL collapse (5.71 / 6.37, not ~74), no crash, no `xe`/GuC reset in dmesg.**
    The §1 bug is NOT firing at 1024/2048 on the current stack — consistent with "no longer reproducible."
  - **PERF (the killer):** crown long-ctx prefill A/B (Engine path, ~10.5K-tok prompt, tile-on, only the
    chunk varies): **512 → 798.7 tok/s · 1024 → 852.2 (+6.7%) · 2048 → 316.6 (0.40×, big REGRESSION).**
    The "4× fewer MoE weight-reads → big win" hypothesis is **FALSIFIED** — best case is +6.7% (1024), and
    2048 falls off a cliff (O(T²) attn per larger chunk + the 4K MoE-scratch cap). **The cap-bump is NOT
    the lever.**
  - **RESEARCH (why even +6.7% isn't worth it):** the "no longer reproducible" is **timing-luck, NOT a
    fix** (didn't fire on the old NEO 26.14 either; NEO notes ship no determinism fix; only IGC codegen
    moved, which shifts the *timing* the bug couples to). Plus a SEPARATE documented hazard — **Intel
    Triton XPU #6658: the gated-delta-rule recurrence → DEVICE_LOST on this exact BMG-G31/B70 silicon**
    from long single-launch recurrences (timeout). Risk grows with chunk length; CR 26.14 itself has
    regressions (compute-runtime #922, multi-rank). **VERDICT: dead end — keep `IE_QWEN35_PREFILL_CHUNK`
    default 512; do NOT lift it. +6.7% does not justify a latent-HW-bug + DEVICE_LOST risk.**
- *The ONLY remaining real lever for crown long-ctx prefill = (ii) MoE-batch-decouple* (layer-major
  prefill: keep the DeltaNet recurrence chunked at 512 but batch the MoE GEMM over the full T → large-M →
  oneDNN/coopmat wins). L-effort restructure of qwen36.cpp, uncertain ROI for a narrow >6K-prompt use
  case. Not started. Everything else (chunk-cap, straight oneDNN port) is falsified above.

**#2 — MTP spec-decode perf path.** DEFER. Verdict: **low-ROI right now** — 248K vocab + `verify(T=K)`
overhead make it 0.36× net (0.81× even on 27B). Since decode is PARITY (not a win), MTP *is* the natural
lever to take the decode lead — but only once the verify path is cheaper (batched int-dot verify + a
cheaper head lm_head). Until then it ships as a lossless capability, not a speedup.

**Honest one-liner:** the Crown is the publishable flagship and its win is *defended*, not *open*;
the single genuinely open optimization is long-context prefill, and it is gated behind a hardware-
hang risk that only the owner can clear — everything else here is regression-defense and grind.
