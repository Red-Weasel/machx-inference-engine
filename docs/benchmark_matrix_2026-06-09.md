# Multi-backend benchmark matrix — 2026-06-09 (P1)

Same GGUF (`Qwen3.6-35B-A3B-Q4_K_M.gguf`), same Arc Pro B70, same flags
(`llama-bench -ngl 99 -p 512 -n 128 -r 3 -sm none -mg 0`; SYCL pinned via
`ONEAPI_DEVICE_SELECTOR=level_zero:0`), all measured 2026-06-09 evening.

| backend | pp512 tok/s | tg128 tok/s | notes |
|---|---:|---:|---|
| **this engine (default)** | **938.6** | **66.2** | PPL 6.55; turbo GGUF: 947.1 / 69.5 |
| llama.cpp SYCL **master b9586** | **1092.4 ± 7.4** | **81.1 ± 0.1** | post-#23142 (+70% MoE PP) + MMVQ int-dot decode |
| llama.cpp Vulkan master b9586 | 774.9 ± 37.8 | 35.1 ± 0.3 | REGRESSED vs b8902 on this model |
| llama.cpp Vulkan b8902 (2026-05) | 885.0 ± 5.5 | 39.8 ± 0.1 | the engine's original comparison anchor |
| Intel llm-scaler-vllm 1.4 (prod B70 stack) | — | — | Qwen3.6 not in validated model list (Qwen3-30B-A3B only) |
| OpenVINO GenAI | — | — | not installed; Qwen3.6 hybrid-DeltaNet support unconfirmed |

## Honest standings

- **vs llama.cpp Vulkan (any version): engine wins decisively** — 121–106%
  prefill, 166–188% decode.
- **vs llama.cpp SYCL master: engine is BEHIND** — 86% prefill, 82% decode.
  The SYCL backend improved enormously between b8902 and b9586:
  PR #23142 (expert-contiguous MoE prefill, +70% on this exact model class),
  multi-column MMVQ (#21845), and the reorder-layout integer-dot GEMV path.
- Public claims must say: *fastest vs llama.cpp Vulkan; currently 82–86% of
  llama.cpp SYCL master; gap-closing work identified below.*

## Why they're ahead, technique by technique (both are adoptable)

1. **Decode (their 81 vs our 66): MMVQ integer dot products.** llama.cpp
   quantizes activations to Q8_1 once per token and computes GEMV dots in
   INT8 (dp4a-class instructions) against the quantized weights directly —
   no per-element dequant in the inner loop, ~4× fewer instructions per
   element than our fp16/fp32 dequant-and-FMA GEMVs. Xe2 has int-dot
   (`int dot: 1` in their device line). This is the single biggest lever.
   PPL note: Q8_1 activation quantization is the standard llama.cpp CUDA/SYCL
   decode path — quality impact is established as negligible, but OUR gate
   stays: PPL ≤ 6.57 or it reverts.
2. **Prefill (their 1092 vs our 939): better dense GEMM.** Their SYCL prefill
   dequants to fp16 and runs oneDNN matmul (Intel's tuned library) — our E1
   does the same dance into our own `gemm_fp16` (33.5 TFLOPS, 18% of peak).
   oneDNN was already in this project's locked tech stack as the "production
   GEMM fallback" (PLAN.md) and never wired. Swapping the E1 GEMM call to
   oneDNN should close most of the 154 tok/s gap; our DN/MoE structure work
   (E2–E5) stays.

## v1.4 fusion campaign update (same night)

Decode 66.2 → **79.4** (97.9% of SYCL master's 81.1), pp512 → 956, PPL 6.54:
two-stage T=1 router (+13% alone — single-WG occupancy starvation was the
real cost, not the algorithm), qkv split-norm 5→1, shexp chain 5→2, conv1d
state fused, alpha/beta+g_beta fused.  P1a int-dot GEMVs: quality-perfect,
perf-neutral (fp16 GEMVs already at 238–350 GB/s); kept opt-in pending
rms_norm+quantize fusion.  Census correction: decode is long-tail
kernel-bound (~14 ms kernels / ~1 ms gaps), not launch-bound.

## v1.5 — DECODE CROWN TAKEN (2026-06-10, same-hour paired runs)

| config | tg128 | PPL | vs llama.cpp SYCL master (81.31 ± 0.21 same-hour) |
|---|---:|---:|---|
| default (Q6_K lm_head) | 81.0 | **6.52** (project-best) | statistical tie |
| **turbo (Q4_K lm_head)** | **84.1** | 6.61 | **+3.5% — CLEAR LEAD** |

How: Q8_1 activation quant rides the norm kernels (rms_norm_f32w_q8,
gated_rms_norm_q8, zero standalone launches); the ALU-bound staged-Q6_K
GEMVs (lm_head 1.68 ms → ~1.0 ms floor; ssm_out) plus the turbo Q4_K
lm_head run hardware integer dots (ie/dp4a.hpp).  Decode two-day arc:
44.7 → 84.1 (+88%).  Negative results banked: Q4_K projection int-dot
neutral (bandwidth-bound), gemv_q4_K A-bypass −19%, cross-layer
residual+norm fusion −1.5.  Prefill remains 955 vs 1088 (88%) → P1b.

## Action plan — STATUS (2026-06-10 final)

P1a: DONE (decode crown taken via norm-fused Q8 + int-dot lm_head/ssm_out).
P1b oneDNN: DONE — NEUTRAL (our gemm_fp16 already matches; opt-in IE_ONEDNN=1).
Prefill crown: see docs/prefill_crown_plan.md (SoA repack + int-dot MoE,
verified from their source; int-dot stage-1 landed opt-in at −5%).

## Original action plan (historical)

- **P1a — MMVQ-class integer-dot decode GEMVs** (gemv_q4_K, gemv_q6_K_slm,
  moe_decode_*): Q8_1 activation quant kernel (per-32 block scale) + int-dot
  inner loops. Target: decode 66 → 85+.
- **P1b — oneDNN matmul behind the E1 prefill path** (replace gemm_fp16 call;
  keep dequant_to_Bt). Target: pp512 939 → 1100+.
- Re-run this matrix after each; the public writeup waits until the engine
  leads the BEST llama.cpp backend, not the most convenient one.

## Reproduction

- Engine: `./build/tools/ie-bench --ctx 1024 --prefill 512 --decode 0`,
  `./build/tools/ie-bench-suite`.
- llama.cpp master: worktree at origin/master (76da2450a), built
  `-DGGML_VULKAN=ON` and `-DGGML_SYCL=ON -DCMAKE_CXX_COMPILER=icpx`.
- Drivers: NEO 26.14.37833.4, IGC 2.32.7, oneAPI 2025.3 (SYCL build env).

## v1.6 — PREFILL CROWN TAKEN → TOTAL CROWN (2026-06-10, same-hour paired runs)

Alternating ours/theirs, same GGUF, same B70, minutes apart:

| run | engine (defaults) | llama.cpp SYCL master 76da2450a |
|---|---:|---:|
| 1 | **1147.3** | 1064.25 ± 7.66 |
| 2 | **1138.9** | 1063.03 ± 9.49 |
| 3 | **1146.7** | — |

**pp512 ~1144 vs ~1064 = +7.6% — the engine now leads llama.cpp's best
backend on BOTH metrics** (decode crown from v1.5 held: 84.1 turbo / 81.0
default vs 81.31).  PPL 6.52 at production defaults (gate ≤ 6.57).

How (docs/prefill_crown_plan.md, executed):
1. Load-time per-expert SoA repack of ffn_{gate,up,down}_exps
   (`ie/quant_soa.hpp`, llama.cpp reorder layout per-expert) — neutral alone,
   but the enabler; all 7 MoE kernels layout-templated, `IE_NO_MOE_SOA=1`
   kill switch, decode guard A/B held (81.25/80.72/81.84).
2. `block_q8_1s` activations (48 B, split half-sums s0/s1) — exact K-quant
   min/offset corrections with ZERO isum dp4a (Q4_K uses s0+s1; Q6_K's
   per-16 scales use s0/s1 separately).
3. Full-K register lattices: stage-2 int-dot down kernels (one SG = whole
   K=512; weights register-resident per column across all routed rows;
   204 → 101 ms) and the same shape for stage-1 (189 → 157 ms).
4. dp4a → hardware idp4a verified via IGC_ShaderDumpEnable (128×/kernel,
   no spills) — plan step 4.

Kernel ledger (pp512, 40 layers): moe_pfl_gate_q8 157.3 · down_q8_6k 53.7 ·
down_q8_4k 48.9 · gemm_fp16 57.6 · dn_recurrence 47.2 ms.
Remaining headroom: stage-1 still 37% of prefill; gemm_fp16+dequant ~68 ms;
dn_recurrence 47 ms.

## P2 dense baseline — qwen3-8b (2026-06-10 night, T8 close)

First dense-architecture perf ledger entry. Same B70, `~/.seal/models/Qwen3-8B-Q4_K_M.gguf`
(Ollama blob, `general.architecture=qwen3`), engine at P2-complete HEAD
(rope_partial race fixed `34e4c01`). Engine: `ie-bench --ctx 1024 --prefill 512
--decode 128 --warmup 8`, three runs. llama.cpp: SYCL master 76da2450a (b9586),
`llama-bench -ngl 99 -sm none -mg 0`, run same hour (21:42).

| run | engine pp512 | engine tg128 |
|---|---:|---:|
| 1 | 1184.99 | 43.62 |
| 2 | 1195.85 | 43.67 |
| 3 | 1190.43 | 43.69 |
| **mean ± sd** | **1190.4 ± 5.4** | **43.66 ± 0.04** |

Same-hour llama.cpp SYCL pair: **pp512 1035.85 ± 3.77, tg128 77.74 ± 0.13**.

Standings (ledger entries, not public claims — claim stays crown-only):
- **prefill: engine leads +14.9%** (1190 vs 1036) — the crown's prefill
  machinery (oneDNN-class gemm_fp16, fused norms) carries over to dense.
- **decode: engine at 43.7 vs their 77.7 (56%) — UNOPTIMIZED, honestly so.**
  The crown's decode crown came from int-dot Q8 GEMVs riding fused norm
  kernels; none of that has been ported/tuned for the dense shapes
  (gemv_fp16 attn paths, dense lm_head). Dense decode optimization is
  explicitly OUT of P2 scope (model breadth, no perf work); it is P3+
  backlog material.

Quality at this baseline: dense PPL bit-exact across runs (avg NLL 2.940491
at `--max-tokens 511`, 2.937037 at the default invocation — invocation-bound
constants, see `docs/ppl_baseline_matrix.md`); layer parity cosine 1.000000
and greedy 64/64 vs llama.cpp (`scripts/p2_parity_qwen3.sh` ALL GREEN).
Crown invariants re-verified same session: PPL avg NLL 1.864495 exact (2/2),
pp512 1158.57, tg 81.92 (bands ≥ 1080 / ≥ 79 held through the rope fix).

---

## §P3b — Dense decode per-kernel profile (optimization baseline)

**2026-06-10** · Qwen3-8B Q4_K_M · B70 [0xe223] · tool
`ie-bench --kprofile-decode` (one fully-warm T=1 step; 3 stable runs).
Full analysis + roofline + hypothesis verdicts:
`docs/dense_decode_profile_2026-06-10.md`.

Measured decode step ≈ 22.17 ms = 44.4 tok/s (matches the 43.7 ledger).
Top buckets (ms/token, % of step, eff GB/s vs 450 roofline):

| kernel | ms | % | calls | eff GB/s |
|---|---:|---:|---:|---:|
| gemv_q4k_q8 (all Q4_K proj) | 10.32 | 46.6 | 198 | 321 (71%) |
| **gemm_q6k (18× Q6_K ffn_down)** | 5.31 | 24.0 | 18 | **140 (31% — THE CLIFF)** |
| gemv_q6k_huge_q8 (lm_head) | 2.28 | 10.3 | 1 | 226 |
| fa2_partial_fp16 (KV read) | 2.19 | 9.9 | 36 | latency-bound |
| gemv_fp16 (attn_v) | 0.62 | 2.8 | 36 | 487 (above roofline) |

**Target ranking (measured, reordered):** (1) gemm_q6k cliff 140→350+ GB/s
≈ +3.2 ms → +~9 tok/s → **Task 1 as written**; (2) NEW: gemv_q4k_q8 321→450
GB/s ≈ +2.95 ms (not in the original 3 suspects — fresh desk pass);
(3) fa2_partial launch/latency; (DROP) attn_v Q8_0 — already 487 GB/s,
Suspect #2 REFUTED. Recommended next dispatch: Task 1 (K-tiled Q6_K GEMV).

---

## §qwen3moe — Qwen3-Coder-30B-A3B fused MoE perf ledger (2026-06-12)

First `qwen3moe` arch perf entry. Same B70 [0xe223], `~/models/Qwen3-Coder-30B-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
(E=128 experts, K=8 used, E_ffn=768, 48 layers; gate/up Q4_K, down Q6_K).
Fused MoE landed (gather→gate_up_silu→down→reduce prefill chain + 2-launch
decode pair) calling the crown's runtime-parameterized `moe_fused.cpp` ops —
**zero crown-file edits**. Order-controlled new-old-new, 3 alternating same-hour
rounds (ours↔llama), first post-rebuild ie-bench run discarded (JIT). Engine:
`ie-bench --ctx 768 --prefill 512 --decode 128 --warmup 4`. llama.cpp:
Vulkan `build-vk` (master `fdc3db9b6`, b9598), `llama-bench -ngl 99 -sm none
-p 512 -n 128 -r 3` (internal warmup+repeat). Single GPU each (engine is
single-GPU). NOTE: both engines run on the SAME B70 (you swap engines to A/B);
the 18.6 GB model lives in the card's 32 GB VRAM for both, so system RAM is NOT
on the steady-state prefill compute path. The early "round 1/2/3" pp512 numbers
below are cold/first-forward and carry warm-up noise; the warm-plateau Tier-2
profile (further down) is the load-bearing prefill number. The gap is a GPU
KERNEL gap, not a swap gap (see Tier-2).

| round | ours fused pp512 | ours fused tg128 | ours unfused pp512 | ours unfused tg128 | llama pp512 | llama tg128 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 61.0 | 19.3 | 60.8 | 18.9 | 989.6 | 58.9 |
| 2 | 61.4 | 19.3 | 54.8 | 17.9 | 983.8 | 58.1 |
| 3 | 60.7 | 19.8 | —    | —    | 983.2 | 59.0 |
| **median** | **~61.0** | **~19.3** | **~58** | **~18.4** | **~984** | **~58.6** |

| metric | engine (fused) | engine (unfused) | llama.cpp Vulkan | gap |
|---|---:|---:|---:|---:|
| pp512 prefill | ~61.0 | ~58 | ~984 | 0.062× (cold/first-forward; warm plateau is ~88 → ~10.4× — see Tier-2) |
| tg128 decode  | ~19.3 | ~18.4 | ~58.6 | 0.33× (llama 3× faster) |
| 512-tok PPL   | 11.99 (fused) | 11.99 (unfused) | oracle (windowed 20.19) | oracle-consistent, Δfused−unfused = 0.013 |

**Honest standings (NOT a "beats llama" entry — crown stays the only such claim):**
- **Fused MoE is correct**, not faster end-to-end here: fused pp512 ≈ unfused
  pp512 and fused tg128 only marginally > unfused. The MoE FFN was never the
  qwen3moe bottleneck on this box.
- **Decode profile proves it.** `ie-bench --kprofile-decode` (one warm T=1 step,
  GPU total 37.1 ms ≈ 27 tok/s GPU-only): `attn_naive_compute` **71.3%** of the
  step; the fused MoE decode kernels (`moe_dec_gate_q4k` + `moe_dec_down_q6k`)
  are only **~13%** combined. qwen3moe decode is **naive-attention-bound** — it
  has no FlashAttention/split-K path (`ensure_attn_partials` is a no-op for this
  arch). That, not MoE, is the lever.
- **Prefill** runs the UNOPTIMIZED fp16 MoE expert GEMM path. The crown beats
  llama via a bespoke int-dot Q8 MoE kernel that hard-requires `E_ffn==512`;
  Qwen3-Coder's `E_ffn=768` falls back to the generic fp16 MoE expert GEMM, and
  top-8 routing (vs the crown's top-2) is ~4× the expert work. ~10× behind llama
  is the honest gap and it is entirely engine-side KERNEL efficiency (NOT swap —
  the model is VRAM-resident on the same B70 for both engines). See Tier-2 below
  for the warm kernel profile that proves it.
- **Value delivered:** fused MoE = the *correct, publishable, parity-proven*
  baseline (PPL 11.99 == unfused to fp16 floor) and the right kernel structure
  for when qwen3moe gets FA-2 + multi-GPU. The headline number for Coder waits
  on FA-2 for the dense-attention path.

**Next levers (ranked, measured):** (1) FlashAttention/split-K for the qwen3moe
dense-attention decode (71% of decode) → biggest win; (2) oneDNN prefill GEMM +
de-serialize per-layer host routing; (3) qwen3moe multi-GPU → Qwen3-Next-80B.

### FA-2 decode wired — lever (1) LANDED (2026-06-12, commit a325cd1)

qwen3moe DECODE (T==1) now calls the existing ESIMD-safe
`full_attention_fa2_decode` + an `ensure_attn_partials` scratch
(`ws_attn_partials_`, sized `n_chunks_max × n_q_heads × (head_dim+2)` FP32,
Bc=64), mirroring dense/qwen35/crown. PREFILL (T>1) stays on naive
`full_attention` (oneDNN prefill = separate Tier-2). **Two-file change**
(`qwen3moe.cpp` + `qwen3moe.hpp`), zero new kernels, zero crown edits.

| metric | before (naive attn) | after (FA-2 decode) | result |
|---|---:|---:|---|
| tg128 decode | ~19.3 | **~37.4** | **1.94×** (ie-bench `--ctx 768 --prefill 512 --decode 128 --warmup 4`, order-controlled: A=38.0 discarded JIT; B=37.41, C=37.38 heat-soaked) |
| pp512 prefill | ~61.0 | ~61.2 | unchanged (prefill path untouched) |
| 512-tok PPL | 11.99 | **11.98** | held to fp16 floor (Δ −0.01; FA-2 is a different reduction order, not bit-exact, but within < 0.1 gate) |
| crown PPL (untouched) | 6.45 / NLL 1.864495 | 6.45 / NLL 1.864495 | **bit-exact** |

Greedy coherence: "Write a function to reverse a string" → coherent
`def reverse_string(s)` (standalone test tool, which lazily sizes partials in
`forward`, reported 40–46 tok/s decode on short ctx — FA-2's win grows with ctx
length vs the 512-ctx ie-bench number). The ~3× llama decode gap is now mostly
closed; the residual is the shared Q4_K/Q6_K GEMV ceiling (Tier-3, P3b dead-end),
not attention. Remaining levers: (2) oneDNN prefill + device routing,
(3) qwen3moe multi-GPU → Qwen3-Next-80B.

### Tier-3 prefill DONE — GPU router + generalized int-dot down kernel: 8.1× (2026-06-12)

Same B70 [0xe223], Qwen3-Coder-30B-A3B Q4_K_M (18.6 GB). Two additive,
qwen3moe-owned levers (crown `qwen36.cpp`/`moe_fused.cpp` untouched).

**Lever 1 — GPU-gemm router.** The Tier-2 "routing 0.3%" was GPU-kprofile-only.
`IE_QWEN3MOE_PROFILE_HOST=1` showed the host `route_token` dot loop = **4859 ms
cumulative = ~66% of the 6354 ms pp512 wall** (GPU total only 2501 ms). Moved
logits to one `gemv_q_T`/`gemm_fp16` (F16 router weight, [T,H]@[H,E]); host keeps
only softmax/top-k. Order-controlled A/B: **pp512 host 59 → GPU 196 = 3.3×**.

**Lever 2 — generalized int-dot W4A8 down** (`src/ops/moe_qwen3.cpp`, NEW file).
Crown int-dot down is `E_ffn==512`-locked; `_gen` kernels generalize to any
`E_ffn%256==0` (lane walks q8 blocks {lane,lane+16,…}<E_ffn/32). + int-dot gate/up.

| pp512 (warm, order-controlled) | int-dot (default) | fp16 (NO_Q8) | llama Vulkan |
|---|---|---|---|
| tok/s | **651 ± 5** | 197 ± 2 | ~984 |

**kprofile @ pp512 (warm, GPU-total 747 ms, was 2501):** down_q8_6k_gen 190 ms
(was 1720, 9×) · down_q8_4k_gen 149 ms (was 354) · gate_q8 253 ms · attn 80 ms.
**Total: pp512 80.6 (orig host-router fp16 baseline) → 651 = 8.1×; 12× → 1.5×
behind llama.** Gates: qwen3moe PPL **11.98 / NLL 2.4834 bit-stable** (== fp16 =
authoritative correctness), Coder greedy coherent, crown **6.45 bit-exact**, ctest
28/28. Opt-outs: `IE_QWEN3MOE_NO_Q8=1`, `IE_QWEN3MOE_HOST_ROUTER=1`.

### Tier-2 prefill investigated — oneDNN is NOT the qwen3moe lever (2026-06-12)

Mirrored the 27B's `dense::prefer_onednn()` lever onto qwen3moe and A/B'd it.
**Result: no win** — and the kernel profile says exactly why. Same B70 [0xe223],
Coder-30B-A3B Q4_K_M. Both engines run on the SAME B70 with the 18.6 GB model
resident in the card's 32 GB VRAM — system RAM is NOT on the steady-state
prefill path. Measured at the **warm plateau** of a 6-point single-process sweep
(weights GPU-resident, first-forward warm-up discarded).

**pp512 prefill kernel profile (warm, GPU-total 2302 ms):**

| kernel | ms | % | role |
|---|---:|---:|---|
| `moe_pfl_down_pk6k` | 1596 | **69.3%** | MoE expert **down**-proj, **Q6_K** experts |
| `moe_pfl_down_pk4k` | 340 | 14.8% | MoE expert down-proj, Q4_K experts |
| `moe_pfl_gate_q4k` | 277 | 12.0% | MoE expert gate/up GEMMs |
| `attn_naive_compute` | 72 | **3.1%** | attention (the only oneDNN-routable term ≈ nil) |
| `moe_gather`+`moe_pfl_reduce` | 6.5 | 0.3% | host-routing pack/scatter — **negligible** |
| dequant/rms/rope/residual | ~12 | tail | |

**Interpretation:** ~96% of qwen3moe prefill is in **GPU MoE-GEMM kernels**
(`moe_pfl_down_pk6k` 69% + Q4_K down 15% + gate 12%); host-routing pack/scatter
is **0.3%**, attention 3.1%. This is a **KERNEL gap, not a swap gap** — the model
is VRAM-resident on the same B70 for both engines. The dominant `moe_*` ops are
the generic **fp16 MoE expert GEMM** fallback: the crown's bespoke int-dot Q8 MoE
kernel hard-requires `E_ffn==512`, so Qwen3-Coder's `E_ffn=768` drops to the
unoptimized fp16 path, and top-8 routing (vs the crown's top-2) is ~4× the expert
work. oneDNN — the 27B's 1.65× lever at H=5120 — has nothing to bite on here
(`prefer_onednn()` only routes `dense::gemv_q_T` = attn q/k/v/o + lm_head, <5% of
prefill; qwen3moe H=2048 so those GEMMs are tiny). The real lever is a tuned
int-dot MoE-expert kernel for `E_ffn≠512` / `top-k≠2`.

| pp512 (warm plateau) | in-house gemm (default) | oneDNN ON | llama.cpp Vulkan |
|---|---:|---:|---:|
| tok/s | **~88** (87.5–87.9 stable) | ~89–92 (87.4–92.5) | **939 ± 36** |

oneDNN ≈ in-house within ±noise → **not enabled by default** (kept behind opt-in
`IE_QWEN3MOE_ONEDNN=1` for reproducible A/B; default path bit-identical to the
prior commit, PPL **11.98** / NLL 2.483450 unchanged either way). Crown
**6.45 / NLL 1.864495 bit-exact** (no crown files touched). We are **~10.4×**
behind llama on pp512 — entirely the **engine-side fp16 MoE expert GEMM** (no
int-dot at `E_ffn=768`, top-8 = ~4× the crown's expert work). NOT swap: both
engines run the VRAM-resident model on the same B70 (the earlier "16×" was the
cold/first-forward number before the warm plateau, not a RAM-swap effect).

**Part B (device-side routing) DEFERRED** — the per-layer host-router pack/scatter
is only ~0.3% of prefill (`moe_gather`+`reduce` above); moving it on-device would
buy nothing. The real qwen3moe prefill lever is a **tuned int-dot MoE-expert
kernel** generalizing the crown's int-dot Q8 win to `E_ffn≠512` / `top-k≠2`
(SoA repack / split-K Q6_K down-proj), which is Tier-3 and must be an **additive
qwen3moe-owned** kernel — the crown `moe_fused.cpp` is FORBIDDEN, so a new file/op,
not an edit. ESIMD-safe constraint applies (1D block reads only). It closes the
~10× prefill gap AND unlocks Qwen3-Next-80B. Next: that additive int-dot
MoE-expert kernel, or qwen3moe multi-GPU → Qwen3-Next-80B.

## §Wave-1 — per-family GPU validation gate (2026-06-12)

Wave-1 proves each model *family* runs correctly end-to-end (arch routing +
tokenizer + chat template + cosine + finite PPL + coherent chat), NOT a perf
crown. First gate landed: **Mistral-Small-3.2-24B-Instruct-2506** — validates
the whole **tekken-tokenizer + `[INST]`-template + Mistral/Devstral/Codestral/Nemo**
cluster on the shared DenseModel (`kLlama3`) path. Single B70 [0xe223],
`~/models/lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-GGUF/...Q4_K_M.gguf`
(14.3 GB; mmproj vision tower ignored — text only). Oracle: llama.cpp `build-vk`
master `fdc3db9b6`. No engine source edits (read/validate task).

| model | arch route | tokenizer | SWA guard | min per-layer cos vs oracle | PPL (builtin, det.) | chat `[INST]` | crown regression |
|---|---|---|---|---:|---:|:---:|---|
| **Mistral-Small-3.2-24B** | `general.architecture=llama` → **kLlama3** (DenseModel: RMSNorm+SwiGLU+GQA 32/8+NEOX rope θ1e9, no QK-norm/bias) | **tekken** (`pre=tekken`); encode ids **identical** to `llama-tokenize` (incl. BOS=1 + multibyte `café`→35858) | **CLEAN** (no `attention.sliding_window` KV — v3.2 dropped global SWA; load-guard correctly does not fire) | **0.99954** (L01–L39, ~1.0; L40/L41 "DIVERGED" is the shared final-norm/lm_head dump-shape artifact — engine dumps all 6 tokens, oracle dumps only the last → tail-misalign, same as validated Llama-3.x) | **7.42** / NLL 2.004174 (bit-exact across 2 runs) | ✅ coherent (reverse-string answer, 4 idiomatic variants), renders `[INST] {user} [/INST]`, stops clean at `</s>` (no free-continuation) | **6.45 / NLL 1.864495 bit-exact** (shared template/tokenizer untouched the crown) |

Config: 40 layers, hidden 5120, ffn 32768, 32 q / 8 kv heads, head_dim 128,
vocab 131072, rope θ=1e9, ctx_train 131072, rms_eps 1e-5. `chat_template` carries
`[INST]`…`[/INST]` → classifies **kMistral** → `build_mistral_prompt` + `</s>`
stop. System-prompt path (folds into first user turn / `[SYSTEM_PROMPT]` for v3+)
is code-confirmed; not exercised via interactive `ie run` (no `--system` flag —
that path is server-only). **Unlocks the Devstral/Codestral/Mistral-Nemo cluster**
(same tekken + `[INST]` + kLlama3 path).

**Cluster fill — Devstral-Small-2507 VALIDATE-ON-DEMAND GREEN (2026-06-13).** The
newest agentic-coding Mistral (`~/models/lmstudio-community/Devstral-Small-2507-GGUF/
...Q4_K_M.gguf`, 14 GB). Same shape as Mistral-Small-24B (arch `llama`→kLlama3,
**tekken**, 40L hidden 5120 ffn 32768 32q/8kv, **head_dim 128 via
`attention.key_length`** — NOT 5120/32=160; the explicit-head_dim landmine §6
resolves correctly). **ZERO engine edits.** `wave1_gate.sh` ALL GREEN: per-layer
cosine min **0.999016** (L01–L39 ≥0.99964; L40/L41 the usual final-norm tail
artifact, still ≥0.999 here), PPL **8.11** / NLL 2.093483, chat → "The capital of
France is Paris." (renders `[INST]`/`[SYSTEM_PROMPT]`, clean stop). Crown untouched
(no source change). Remaining cluster items are validate-on-demand: **Codestral**
(SP/SentencePiece track — the one genuinely different tokenizer path) and
**Mistral-Nemo** (tekken, redundant with Devstral on the head_dim path).

### Wave-1 Gate 2/9 — DeepSeek-R1-Distill-Qwen-1.5B (2026-06-12)

Validates the **DeepSeek-R1-Distill `<｜User｜>`/`<｜Assistant｜>` sentinel template**
(`kDeepSeek` → `build_deepseek_prompt`, wired `c5b12ca`, first real-GGUF test) +
`<think>`-on reasoning + `<｜end▁of▁sentence｜>` stop. The model is a **Qwen2
fine-tune** → routes to the already-validated `kQwen3Dense` path; the NEW thing
proven is the template + special tokens. Single B70,
`~/models/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/...Q4_K_M.gguf` (1.1 GB, kept).
Oracle: llama.cpp `build-vk`. **One harness-only edit** (wave1_gate stage-0 ANSI
strip — see below); zero engine source edits.

| model | arch route | tokenizer | template / `<think>` / stop | min per-layer cos vs oracle | PPL (builtin, det.) | chat smoke | crown regression |
|---|---|---|---|---:|---:|:---:|---|
| **DeepSeek-R1-Distill-Qwen-1.5B** | `general.architecture=qwen2` → **kQwen3Dense** (Qwen2 fine-tune; 28L, hidden 1536, ffn 8960, 12q/2kv, hd 128, θ1e4, eps1e-6, vocab 151936) | **qwen2 BPE**; encode ids **identical** to `llama-tokenize` (BOS 151646 + multibyte `café`). DeepSeek sentinels are **single ids**: User=151644, Assistant=151645, eos `<｜end▁of▁sentence｜>`=151643 | `classify_template_family`→**kDeepSeek** (standalone-verified, not grep). Render: `enable_thinking=true`→`<｜User｜>{u}<｜Assistant｜><think>\n` (**think ON**); `false`→`<think>\n\n</think>\n\n`. Stops at `<｜end▁of▁sentence｜>` | **0.999404** (proper-shaped layers, ~1.0; argmax `ĠParis` correct. `**DIVERGED**` labels = rel_fro≥0.05 Q4_K-vs-oracle quant gap + L28 dump-shape tail artifact, NOT a forward bug) | **81.24** / NLL 4.397384 (finite/det.; high but expected for a 1.5B Q4_K_M distill — property gate) | ✅ "12*13 step by step" (serve, think ON) → reasoning trace + **156** (`\boxed{156}`), **finish_reason=stop** at eos, 264 tok, no run-on. `ie run` (think off) → 156 + empty-think | **6.45 / NLL 1.864495 bit-exact** |

The `<think>` gate on the DeepSeek path is its own `enable_thinking` flag (the
template carries no literal `<think>`, so it does NOT route through the ChatML
`model_has_think` gate from `8c568a6` — `build_deepseek_prompt` injects `<think>`
unconditionally when thinking is on). **R1-Distill-Llama is the same template on
the llama path** (`general.architecture=llama` → kLlama3, same `<｜Assistant｜>`
sentinels → kDeepSeek) → this gate proves the whole R1-Distill family.

**Harness fix (`scripts/wave1_gate.sh` stage-0):** `ie-inspect` colorises output
(no `--no-color`); the old arch parse read the ANSI reset as the value (`0m`,
spurious FAIL). Now strips `\x1b[…m` and extracts the **quoted** value on the
`general.architecture` row → parses `qwen2` cleanly. Reusable for the rest of
Wave-1. (The `deepseek` marker check still SKIPs because ie-inspect doesn't print
the `chat_template` string value — confirmed by hand from the GGUF bytes.)

### Wave-1 Gate — DeepSeek-R1-Distill-Llama-8B (2026-06-12)

Closes the R1-Distill family: the **same DeepSeek sentinel template on the LLAMA
path**. Where Gate 2/9 proved `kDeepSeek` → `build_deepseek_prompt` on the qwen2
fine-tune (`kQwen3Dense`), this proves it on a **plain Llama-3 fine-tune** →
`general.architecture=llama` → **kLlama3** (the validated DenseModel path: Q/K
un-permute + `rope_freqs` + llama-bpe). This is also the **zero-code llama-clone
tier anchor** — Yi / Nemotron / InternLM / Baichuan ride this identical kLlama3
path (validate-on-demand). Single B70,
`~/models/DeepSeek-R1-Distill-Llama-8B-GGUF/...Q4_K_M.gguf` (4.6 GB). **Zero
engine source edits, zero harness edits** (wave1_gate reused as-is).

| model | arch route | tokenizer | template / `<think>` / stop | min per-layer cos vs oracle | PPL (builtin, det.) | chat smoke | crown regression |
|---|---|---|---|---:|---:|:---:|---|
| **DeepSeek-R1-Distill-Llama-8B** | `general.architecture=llama` → **kLlama3** (DenseModel; 32L, hidden 4096, ffn 14336, 32q/8kv, hd 128, θ5e5, eps1e-5, vocab 128256; **no `sliding_window`**) | **llama-bpe** (NOT qwen2); engine encode ids **byte-identical** to `llama-tokenize` (`128000 9906 1917 13 711 282 2120 1680 471 865 9 17 220 674 220 4513 19 53050`, incl. multibyte `café`). DeepSeek sentinels are **single ids**: User=128011, Assistant=128012, eos `<｜end▁of▁sentence｜>`=128001, bos=128000, `<think>`=128013 `</think>`=128014 | `<｜Assistant｜>` sentinel in GGUF chat_template → `classify_template_family`→**kDeepSeek** (dispatched in `engine.cpp` BEFORE the kLlama3 branch, so the llama-arch model takes `build_deepseek_prompt`, not `build_llama3_prompt`). Render: `enable_thinking=true`→`<｜User｜>{u}<｜Assistant｜><think>\n` (**think ON**); `false`→`<think>\n\n</think>\n\n`. Stops at `<｜end▁of▁sentence｜>` (128001) | **0.999929** (L01–L31 all ≥0.99992; L32 = 0.998990 is the final-norm/logits **dump-shape tail artifact** — llama.cpp slices it to last-token via `inp_out_ids` so the compare crosses 24576-vs-4096 widths; rel_fro 0.045 WARN not DIVERGED. argmax greedy correct) | **25.66** / NLL 3.244969 (finite + **bit-exact deterministic** across 2 runs; high but expected for an 8B Q4_K_M distill on the short builtin corpus — property gate) | ✅ "What is 12*13? Think step by step." (`ie run`, think off) → coherent FOIL derivation → **156** (`\boxed{156}`), stops at eos (`> ` prompt returns, no run-on) | **6.4527 / NLL 1.864495 bit-exact** |

**The DeepSeek template is now proven on BOTH the qwen2 and llama paths** — the
classification + `build_deepseek_prompt` + `<think>` `enable_thinking` flag +
`<｜end▁of▁sentence｜>` stop are arch-independent (they live above the DenseModel
forward). The free **llama-clone tier** (Yi / Nemotron / InternLM / Baichuan,
all `general.architecture=llama` Llama-3 derivatives) rides this exact validated
kLlama3 path and is **validate-on-demand** (no new code expected). One non-fatal
gate quirk: wave1_gate stage-1 (encode-parity) SKIPs because the bundled
`ie-dense-dump` rejects `-n 0`; re-run with `-n 1` emits the `tokens=` line and
ids match `llama-tokenize` exactly (done by hand here). Stage-2's hard
`cos≥0.999` FAILs only on the L32 final-norm tail-compare shape artifact — not a
forward bug (every real transformer layer is ≥0.99992).

### Wave-1 Gate 3/9 — Microsoft Phi family — **Phi-4 GREEN** / Phi-3.5-mini still blocked (updated 2026-06-12)

**Status: Phi-4 VALIDATED (GREEN).** The `phi3`-arch blockers are cured by three
additive, presence/dtype/`pre`-gated levers (commits `aaab17b`, `aa37fc9`,
`27eaa24`, `e917e6d`): (1) **Q5_K/Q8_0 dense GEMV** via `dequant_q5_K_to_Bt`
(reusable — unblocks *any* Q5_K dense GGUF); (2) a **fused-tensor splitter** at
load that slices `attn_qkv`→Q/K/V and `ffn_up`→gate/up row-spans (NEOX natural
order, Q5_K Q/K slice via dequant); (3) the **`pre=dbrx`** tokenizer flag (dbrx
pretokenizes identically to llama-bpe). Arch routing was already correct
(`general.architecture=phi3` → `kLlama3`). Single B70 [0xe223], oracle llama.cpp
`build-vk` (`ie-llama-dump`). **Phi-3.5-mini stays BLOCKED** — it additionally
needs LongRoPE + a non-power-of-two head_dim 96 (separate milestone).

| model | arch route | tensors (fused vs split) | tokenizer (`pre`) | rope / head_dim | status | crown regression |
|---|---|---|---|---|---|---|
| **Phi-4 (~15B)** | `phi3` → **kLlama3** (40L, hidden 5120, 40q/10kv GQA, hd 128, ffn 17920, θ2.5e5, eps1e-5, vocab 100352) | FUSED `attn_qkv`[5120,7680]=**Q5_K** + FUSED `ffn_up`[5120,35840]=Q4_K → **split at load** (Q5120+K1280+V1280; gate17920+up17920) | **`pre=dbrx`** → folded into the llama-bpe flags (digits_1to3 + ignore_merges); **encode-parity vs `llama-tokenize` PASS** (10 golden + 4 round-trip; digit-run 1234567→`4513 10961 22`). No add_bos default (Phi-4 carries none) | full rope (hd 128, θ2.5e5); no LongRoPE | **GREEN** — loads; **per-layer cosine ≥ 0.999999 L01..L41** vs oracle; greedy=` Paris` (logit 18.44 vs 18.45 oracle, top-5 identical); **PPL 8.2475 / NLL 2.109913** (deterministic ×2); **chat coherent, stops at `<|im_end|>`** (ChatML stop set wired, `<|im_sep|>` render note below) | **6.45 / NLL 1.864495 bit-exact**; dense NLL 2.940491 bit-exact |
| **Phi-3.5-mini (~3.8B)** | `phi3` → **kLlama3** (32L, hidden 3072, 32q/32kv MHA, **hd 96**, ffn 8192, vocab 32064) | **`pre=default`**, model=`llama` (SPM) — friendlier | **LongRoPE** (`rope_factors_long`/`_short`, SU-scaled by seq-len; dense path only does single `rope_freqs` linear scale); **hd 96 is NOT power-of-two** → tripped by the FA-2 load gate (`head_dim must be power-of-two ≥16`) | NOT DOWNLOADED — header-read showed same **fused Q5_K `attn_qkv` + fused Q4_K `ffn_up`**, PLUS LongRoPE + hd-96 gate → strictly harder than Phi-4 | **BLOCKED** — fused-split now solved, but LongRoPE + head_dim-96 power-of-two gate remain (NEXT milestone) | — |

**Why both Phi candidates are blocked (the `phi3` convert convention):**
llama.cpp's phi3 GGUF convert keeps Microsoft's *fused* `Wqkv` and `gate_up_proj`
projections fused (one `attn_qkv` + one `ffn_up` per layer, gate‖up concatenated),
unlike the Llama/Qwen converts which pre-split. Our DenseModel loader has no
fused-tensor splitter, and the fused `attn_qkv` is quantized **Q5_K**, which the
dense GEMV path does not support (only Q4_K/Q6_K/F16; the crown/qwen3moe paths
*do* have a `dequant_q5_K_to_Bt` kernel, but it is not wired into
`upload_quant_dense`). The byte-split itself is feasible (row-major output rows →
contiguous superblock spans, same span logic as `upload_quant_dense_permuted`),
but Q5_K dense GEMV + (Phi-3.5) LongRoPE + the hd-96 FA-2 gate make this a real
feature, not a one-line fallback.

**Follow-up to unblock Phi (precise):**
1. **Fused-tensor splitter** in `dense_dispatch.hpp` — slice `attn_qkv`[N=Nq+2·Nkv]
   into Q/K/V row-spans and `ffn_up`[N=2F] into gate/up row-spans at load
   (contiguous per-row superblock byte spans; gate the llama un-permute on the
   Q/K slices only). Triggered when `attn_q.weight` is absent but `attn_qkv.weight`
   is present.
2. **Q5_K in the dense GEMV path** — wire `dequant_q5_K_to_Bt` (already in
   `qwen35_dense.cpp`/`qwen3moe.cpp`) into `upload_quant_dense`, or accept Q5_K in
   the dense GEMV dispatch. Needed for *any* Q5_K dense GGUF, not just Phi.
3. **`dbrx` pre-tokenizer** (Phi-4 only) — add a `pre=="dbrx"` branch:
   tiktoken/cl100k regex `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
   with `ignore_merges=true` (matches `LLAMA_VOCAB_PRE_TYPE_DBRX`). Phi-3.5-mini
   uses `pre=default`/SPM so it needs no new pre-tokenizer.
4. **LongRoPE + non-pow2 head_dim** (Phi-3.5-mini only) — SU short/long factor
   selection by sequence length, and relaxing the FA-2 power-of-two head_dim gate
   for hd 96. Phi-4 (hd 128, plain rope) needs neither.

**chat_template (Phi-4 validated):** Phi-4 = ChatML-with-`<|im_sep|>`
(`<|im_start|>{role}<|im_sep|>{content}<|im_end|>`, bos=100257 eos=`<|im_end|>`=100265,
second EOG `<|endoftext|>`=100257) → classifies **kChatML** (`<|im_start|>`
present). Chat smoke (`echo "Write a Python function to reverse a string" | ie run
phi-4`) returns a clean, correct `s[::-1]` function with an explanation and
**stops at `<|im_end|>`** (no free-continuation). The engine renders ChatML via
`build_chatml_prompt`, which uses a `\n` role separator, NOT Phi-4's `<|im_sep|>`
— the model is robust to this (reply is coherent and high quality), so a byte-exact
`<|im_sep|>`-separator variant is a **noted follow-up, not a blocker**. Stop set:
a `tf==kChatML && arch!=kQwen3Dense` branch (engine.cpp `e917e6d`) sets
`<|im_end|>`+`<|endoftext|>` explicitly (Qwen3-dense EXCLUDED to keep its prior
stop set; verified crown/dense bit-exact). Phi-3.5-mini's classic-Phi
`<|system|>`/`<|user|>`/`<|assistant|>`+`<|end|>` template is a later concern
(model still gated on LongRoPE + hd-96).

### Wave-1 Gate — IBM Granite-3.3-8B-Instruct (2026-06-13)

Validates the **`granite` arch dense tier** — the kLlama3 path + the four Granite
scalar multipliers (the only forward edit in the breadth sprint) + the new
`kGranite` chat template. Single B70, `~/models/lmstudio-community/
granite-3.3-8b-instruct-GGUF/...Q4_K_M.gguf` (4.7 GB). Oracle: llama.cpp `fdc3db9b6`.
Commits 34b0049 (forward) + 13f2fe7 (template).

| model | arch route | the 4 scalars (verified) | min per-layer cos vs oracle | PPL (builtin, det.) | chat | crown/dense regression |
|---|---|---|---:|---:|:---:|---|
| **Granite-3.3-8B** | `general.architecture=granite` → **kLlama3** (40L, hidden 4096, ffn 12800, 32q/8kv, hd 128, vocab 49159, θ1e7, eps1e-5) | embedding_scale **12.0** ×embd; residual_scale **0.22** ×each sub-block; **attention.scale 0.0078125** (DOTTED key — fixed from `attention_scale`; SDPA softmax scale via Q pre-scale by ·√HD, crown kernels untouched); logit_scale **16.0** ÷logits | **0.9995–0.99999** (L01–L39); L40/L41 <0.999 = final-norm/lm_head dump-shape tail artifact; `rel_fro~0.07` "DIVERGED" tags = fp16-residual precision on one massive-activation channel @3433 (cos ~1.0) | **10.30** / NLL 2.332057 | ✅ kGranite `<|start_of_role|>` turns → coherent Python (reverse-string slicing + docstring + unittest), stops clean at `<|end_of_text|>` | crown **6.4527 / NLL 1.864495** + dense qwen3-8b **NLL 2.940491** bit-exact; ctest 28/28 |

Forward proof: greedy "The capital of France is" → " **Paris.**" (argmax id 2716
'ĠPar' logit 16.4 — without `attention.scale` the scores are ~11× too large →
garbage). All edits value-gated (`!= default` → every non-Granite model byte-
identical; `dense::scale_inplace` launches zero kernels off-path). **Unlocks the
Granite-3.x dense tier** (3.1/3.2/3.3 instruct). Granite-4.0 = hybrid-Mamba2, a
separate SSM arch → Wave-2.

## §qwen3next — Qwen3-Next-80B-A3B (first-mover GPU on Intel Arc, 2026-06-13)

Hybrid gated-DeltaNet + gated full-attn (NEOX) + 512-expert MoE + shared expert,
46 GB Q4_K_M, layer-split across 2×B70 (`ie run/serve --gpus 2`). Forward correct
(greedy " Paris. The capital of Italy is Rome"); cosine sweep all 48 layers
0.998–1.007 vs the fp32 oracle; PPL 4.7282 (256 tok, `ie-qwen3next-ppl`). int-dot
W4A8 down DEFAULT.

**Head-to-head vs llama.cpp SYCL** (built `-DGGML_SYCL=ON` icpx 2026.0 @ `fdc3db9b6`;
its SYCL backend HAS gated_delta_net/ssm_conv/ssm_scan GPU ops → runs qwen3next ON
GPU, unlike Vulkan→CPU). Same 46 GB GGUF, both 2×B70 `-sm layer`, same session:

| metric            | engine (ours) | llama.cpp SYCL | ratio          |
|-------------------|---------------|----------------|----------------|
| **decode tg128**  | **51.8 tok/s**| 37.1 ± 0.2     | **1.40× ours** |
| prefill pp256     | 513 (@208 tok)| 473.9 ± 10     | ~ahead         |
| prefill pp512     | 566 (chunked) | 622.2 ± 7      | 0.91×          |
| prefill pp128     | 336 (@~83 tok)| 334.4 ± 19     | ~parity        |

Decode win is clean (text-independent, same metric). Prefill is parity/ahead up to
~256 tok and ~10% behind at 512 — the **DeltaNet T≤256 prefill chunking** (hardware
workaround, `docs/known_bugs.md`) caps long-prefill throughput vs llama's single
512-prefill. int-dot down A/B: pp512 566 vs fp16 224 = **2.5×**.

**5-prompt suite** (`ie-qwen3next-bench`, same protocol as ie-bench-suite; 2×B70,
int-dot, decode=128, runs=3): short-chat 53t pp198/tg52.0 · long-instruction 83t
pp332/tg51.8 · codegen 78t pp336/tg51.8 · math-reasoning 89t pp340/tg51.8 ·
long-context 208t pp513/tg51.7 — median **pp 336 / tg 51.8 tok/s**.

Tools: `ie-qwen3next-{gen,ppl,bench}`. Crown PPL 6.4527 bit-exact throughout.
