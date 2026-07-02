# Qwen3.6-27B (`qwen35` dense-hybrid) — Living Authority Doc

> Single source of truth for the 27B optimization state. Living doc — update at every
> 27B milestone; if reality and this file disagree, fixing the file is part of the change.
> Cross-arch roadmap lives in `MASTER_DEV_PLAN.md §7`; this is the per-arch deep state.
> **Last verified:** 2026-06-26 (prefill win RE-VERIFIED ~2× vs current llama-SYCL `fdc3db9b6`; decode roofline settled `81b9394`).

---

## 0. Identity

- **Family / size:** Qwen3.6-27B — the crown DeltaNet family at dense scale. `model_type qwen3_5`
  (`Qwen3_5ForConditionalGeneration`), text-only decode of a multimodal checkpoint.
- **Geometry** (`include/ie/qwen35_dense.hpp:19-22`, verified): hidden 5120, **64 transformer
  layers = 48 gated-DeltaNet (linear) + 16 gated full-attention** (`full_attn_interval=4`) **+ 1
  trailing NextN/MTP layer (blk.64)**. FFN 17408 (dense SwiGLU, **no MoE**). Full-attn: 24 q-heads
  / 4 kv-heads, **head_dim 256**, gqa=6, partial RoPE n_rot=64 (`partial_rotary_factor 0.25`),
  rope_theta 1e7, fp16 KV. DeltaNet: 16 k-heads / 48 v-heads × 128, ssm_inner 6144,
  conv_channels 10240 (**NOT** SI·2 — the crown identity is false at this geometry), conv_kernel 4.
  vocab 248320 (NOT tied — separate Q6_K `output.weight` lm_head).
- **GGUF files + role:**
  - Daily driver / benchmark file: `~/models/bartowski/Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_M.gguf`
    (mixed per-layer dtypes: Q4_K FFN/qkv, Q6_K some down/attn_q, Q5_K attn_k/attn_output, Q8_0 ssm_out).
  - 1-card Q6_K variant (env-gated SoA decode lane).
  - `~/models/local/Qwen3.6-27B-obliterated-cyber/...Q8_0.gguf` (~27 GB, **2×B70 layer-split only**).
- **Forward path:** `src/model/qwen35_dense.cpp` (single-GPU, `Qwen35DenseModel`). Multi-GPU:
  `src/model/qwen35_split.cpp` (`Qwen35SplitModel`, layer-split, native Q8_0-SoA). Tensor-parallel:
  `src/model/qwen35_tp.cpp` (`Qwen35TpModel`, FFN-slice — measured NO-GO, see §5).
- **Engine dispatch tag:** `kQwen35Dense`. `--gpus 1` → `Qwen35DenseModel`; `--gpus 2` →
  `Qwen35SplitModel` (oneDNN forced OFF); `IE_QWEN35_TP=1 --gpus 2` → `Qwen35TpModel`.
  `qwen36.cpp` (crown MoE) is NEVER edited — this is additive new code reusing `src/ops/*` leaves.
- **One-line standing:** **WINS prefill at EVERY length (1.96–3.08× vs current llama-SYCL `fdc3db9b6`,
  RE-VERIFIED 2026-06-26: pp1024 1.96× / 4096 2.07× / 8192 3.08× / 16384 2.13×), LOSE decode (~16 vs
  llama 21.6 = 0.74×).** The 16K prefill collapse is FIXED (long-ctx tile lever — was 0.52× loss).
  Decode is the one genuine weak spot, on the **commoditized dense-GEMV-BW axis** (not the DeltaNet
  moat). Publish on the prefill moat (now a clean sweep, all lengths) + 27B lossless spec; never claim decode.

---

## 1. Envelope (benchmark contract)

| Axis | Contract |
|---|---|
| Hardware | 1×B70 (608 GB/s HBM, ~32 GB) for decode + short/mid prefill; 2×B70 only for the Q8 cyber model (28.6 GB > 0.7× single-card). Clean box: GPU idle-clocked ≤600 MHz, swap≈0, one workload. |
| Model + quant | `Qwen_Qwen3.6-27B-Q4_K_M.gguf` is THE head-to-head file (matches llama's). Q6_K (1-card) and Q8_0 (2-card cyber) are secondary. |
| Target modes | Decode short-ctx (tok/s) · prefill 512–4K (tok/s) · long-ctx prefill 16K (weak) · long-ctx decode (attn-bound, untuned). |
| Competitor build | llama.cpp **SYCL** (`~/llama.cpp/build-sycl`, icpx/oneAPI 2026.0, HEAD `fdc3db9b6`), `-ngl 99`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`. SYCL (not Vulkan) is llama's fastest qwen35 backend on B70. |
| Method | ours = `ie-bench --prefill 512 --decode 128 --warmup 20` ≡ llama `llama-bench -p512 -n128`, same GGUF/card. Discard cold/JIT first run. **`ie run` is NOT a perf tool** (JIT-noisy). |
| Quality gate | Engine-wide crown `ie-perplexity` ≤ 6.57 (bit-exact 6.4527) on every change; per-arch 27B PPL bit-stable (see §2). |

✅ **Prefill win RE-VERIFIED 2026-06-26 (apples-to-apples, same GGUF, single B70 pinned, current
llama-SYCL build `fdc3db9b6`/9598):** ours **pp1024 572 vs llama 292.3 = WIN 1.96×**; **pp4096 577 vs
278.9 = WIN 2.07×**. The "public llama ≈718 may erase it" suspicion is **REFUTED** — current
llama-SYCL on this box does 292/279 (the old H2H 287 was accurate, not a slow config). llama decode
tg32 = 21.62 on-GPU (sane, not CPU-fallback) → our decode ~16 still LOSES 0.74× (the known
commodity-axis loss — never claimed). **The 27B prefill ~2× win is now a DEFENSIBLE public claim.**

---

## 2. Correctness

- **PPL gate (engine-wide):** `./build/tools/ie-perplexity` crown ≤ 6.57 (held bit-exact **6.4527 /
  NLL 1.864495** across every 27B commit — all 27B code is additive, crown path untouched).
- **27B PPL (arch-specific, `ie-perplexity --gguf <27B file>`):** bit-stable, quant/corpus-dependent:
  - Q4_K_M: **5.6022** (2026-06-22 clean box, full-context stream); earlier **5.34 / NLL 1.675056**
    and **5.3391/5.4163** (different window/build — direction consistent, treat 5.34–5.60 as the band).
  - Q6_K: **5.5578 SoA vs 5.5577 AoS-oracle** = Δ0.00002% (int-dot SoA numerically indistinguishable).
  - oneAPI 2026.0 update changed **no** numerics (re-verified bit-identical pre/post).
- **Reference oracle:** llama.cpp master `fdc3db9b6` CPU build (`build-cpu/bin/llama-perplexity` +
  `llama-eval-callback`; NOT `llama-cli`). Full op-by-op dataflow captured in
  `docs/qwen35_27b_oracle_dataflow.md`. **Per-layer cosine ≥0.9995 on all 64 layers**, exact greedy
  argmax ' Paris', top-5 identical (commit `698156c`). rel_fro ~0.02 is inherent fp16-residual
  precision (we fp16 vs llama fp32) × 64 layers + DeltaNet — proven oneDNN-invariant (`IE_QWEN35_NO_ONEDNN`).
- **Deterministic-gen check:** greedy "The capital of France is" → "Paris." on single-GPU, 2-card
  split, and TP paths.
- **Silent-bug landmines:**
  1. `conv_channels = d_inner + 2·k_heads·ssm_state = 10240`, computed DIRECTLY — never SI·2.
  2. attn_q is **joint Q|gate** `{5120,12288}` — split per-head into Q and the σ-gate
     (`split_q_gate_per_head`), post-attn sigmoid gate before output proj.
  3. DeltaNet q/k repeat is **TILE** (16→48, `repeat_interleave_heads` default interleave=**false**) —
     the opposite of the 80B's interleave (would silently corrupt if flipped).
  4. FFN residual adds to the **pre-post-norm** tensor (qwen35 residual order).
  5. Mixed per-layer quant: `ssm_out` is Q8_0/Q4_K/Q6_K depending on layer — dispatch on the actual
     `dtype`, never assume (unconditional Q6_K-down dispatch NaN'd the 80B sibling).
  6. blk.64 (NextN/MTP) is SKIPPED for text decode (but IS the spec-decode draft head, §5).
  7. tokens 248068/248069 = `<think>`/`</think>` (USER_DEFINED, legit reasoning markers — do NOT
     suppress as "garbage"; the "chat bug" was a misdiagnosis, `7766c80`).

---

## 3. Pareto frontier (measured matrix)

**Decode — Q4_K_M, 1×B70, clean box, vs llama SYCL (same file):**

| Date | ours tg | llama tg | verdict | note |
|---|---:|---:|:---:|---|
| 2026-06-20 | 15.2 | 23.2 | **LOSE 0.65×** | first clean H2H (`build-sycl fdc3db9b6`) |
| 2026-06-22 | 15.1 | 22.0 | **LOSE 0.69×** | post oneAPI 2026.0 (pp512/tg128 line) |
| 2026-06-22 eve | 16.3 | 22.65 | **LOSE 0.72×** | 5-lens fan-out, same box |
| 2026-06-25 | **~17.0** (16.86–17.14) | ~22–23 | **LOSE ~0.75×** | `IE_QWEN35_Q4K_REORDER` opt-in (+4–7% over ~16 baseline) |

**Prefill — Q4_K_M, 1×B70, vs llama SYCL:**

| T_pp | ours | llama | verdict | bottleneck class |
|---:|---:|---:|:---:|---|
| 1024 | **572** | 292.3 | **WIN 1.96×** ✅ | naive attn (≤gate), parallel chunks |
| 4096 | **577** | 278.9 | **WIN 2.07×** ✅ | naive attn (≤gate) |
| 8192 | **722** | 234.8 | **WIN 3.08×** ✅ | **TILE** (fa2 wide hd256) |
| 16384 | **499** | 233.9 | **WIN 2.13×** ✅ | **TILE** — was 0.52× collapse w/ naive (115) |

> ✅ **RE-VERIFIED 2026-06-26 — apples-to-apples, same GGUF (16.74 GiB), single B70 pinned
> (`level_zero:0`), current llama-SYCL build `fdc3db9b6`/9598, both on-GPU** (llama tg32 21.62 = sane).
> **27B now WINS prefill at EVERY length (1.96–3.08×).** The 8192/16384 wins are the new **long-ctx tile
> lever** (2026-06-26): the head_dim-256 full-attn layers were on naive O(T²) attention that re-read the
> whole KV T× → 16K collapsed to 115 tok/s (0.52× loss). Routing them through the proven Gemma wide-tile
> kernel (`full_attention_fa2_prefill_tile_gemma`, KV once per query-tile, gated ctx≥6144 so the verified
> ≤4K naive win is untouched) → pp8192 272→722 (**2.65× self**), pp16384 115→499 (**4.35× self**) →
> **flips 16K from 0.52× LOSS to 2.13× WIN.** Coherence-verified (identical capitals vs naive); the
> kernel is argmax-bit-identical to naive at hd256 (Gemma proof). Opt-out `IE_QWEN35_NO_FA2_TILE`,
> tune `IE_QWEN35_FA2_TILE_MINCTX`. The "public llama ≈718 flips it" worry is REFUTED (llama 234–293 flat).

**Secondary configs:**

| Config | decode | note |
|---|---:|---|
| Q8_0 cyber, 2×B70 split | ~11 | dense reads 28.6 GB/tok; BW ceiling ~40, gap = layer-split serial tax + ~1600 launches/tok + prototype `gemv_q8_0_soa_q8` |
| Q8_0, 1×B70 (`IE_QWEN35_Q8=1`) | 0.6 | **non-viable** — 28.56/32 = 0.89× VRAM → host-spill thrash |
| Q6_K, 1×B70 | 12.8 | SoA-Q6 decode lane (was 5.0 pre-kernel) |
| TP FFN-slice (`IE_QWEN35_TP=1`) | 12.6 | ≤ 13 split baseline → **NO-GO** (no-P2P host-bounce ≈ the FFN it splits) |

**Net standing:** prefill WIN at EVERY length (1.96–3.08× RE-VERIFIED 2026-06-26 vs current
llama-SYCL — DEFENSIBLE public claim; 16K collapse FIXED via the long-ctx tile lever → 2.13× WIN); decode LOSE ~0.74× on the commoditized axis. Per
`docs/COMPETITIVE_SCORECARD_2026-06-25.md:161`: **dense-27B decode and Coder-16K are LOSSES — never claim them.**

---

## 4. Bottleneck map

**Decode (Q4_K_M, 1×B70):**

| Profile (date) | wall/tok | dominant kernels | class |
|---|---:|---|---|
| 2026-06-22 | 67.6 ms | gemv_q4k 51.6% · gemv_q6_soa 27.2% · gemv_q8_0_soa 5.6% · fa2_partial 5.0% | 79% in two GEMVs |
| 2026-06-25 (reorder ON, `81b9394`) | 57 ms | Q4_K FFN gate/up reordered **40.8%** · Q6_K FFN-down SoA **31.5%** · **DeltaNet recurrence only 1.4%** | **~87% GEMV-BW-bound** |

- **Roofline:** 16.74 GiB/tok ÷ 608 GB/s ≈ **36 tok/s ceiling**. The two big GEMVs run **~42–46% BW**
  (reorder lifted Q4_K from ~42→~46%) vs **llama ~60–64%**. Q6_K FFN-down already on a strong
  ~80%-BW SoA lane; small projections stay AoS (reorder regresses them).
- **The critical nuance (VTune, 2026-06-22):** 27B decode is **occupancy/latency-bound — 27%
  occupancy, 85% XVE-stalled, L3-BW-bound 0%** — NOT bandwidth-saturated. This is WHY pure
  coalescing/SoA fixes are flat: the limiter is too few subgroups in flight, not byte traffic.
  The "87% BW-bound" (06-25) and "occupancy-bound" (06-22) are consistent: GEMVs dominate wall but
  don't saturate BW because occupancy is low → the lever is **occupancy/WG/dp4a micro-tuning**, not layout.
- **DeltaNet recurrence hypothesis FALSIFIED:** 1.4% of decode, FLAT across ctx (O(1)/token recurrent
  state, not KV). `gemv_q8_0_soa` (ssm_out) is a projection, not a state read. Not worth touching.

**Long-ctx decode attention** (`project_27b_decode_attn_kv_lens.md`, 2026-06-25): 16 full-attn layers,
gqa=6, head_dim=256, fp16 KV. At 16K: KV read = 16×16384×4×256×2(KV)×2B = **4.29 GB/token**; v1
`full_attention_fa2_decode` = 29.8 ms → **144 GB/s = 24% peak**. Root cause = per-KV-position
`reduce_over_group` ON the serial online-softmax dependency chain (`attention.cpp:643-662`), NOT GQA
re-read (L2-absorbed) nor scalar-half load. Short-ctx attn is only ~2.7–5% — the long-ctx lever only.

**Prefill:** WIN range is oneDNN matmul + batched alpha/beta; 16K was a naive-O(T²) hd256 full-attn
collapse, FIXED 2026-06-26 by routing those layers through the Gemma wide-tile kernel (→ 2.13× WIN,
§3); the residual DeltaNet T≤512 chunking (32 serial chunks) is launch-bound but no longer a net loss.

---

## 5. Hypothesis ledger (every lever — KEEP / REVERT / SPECIALIZE / DEFER)

| # | Lever | Observation → mechanism | Result | Commit / gate |
|---|---|---|---|---|
| 1 | Batch ssm_alpha/beta (N=48→pad 64) | 49k serial-gemv launches = 38% of prefill; pad to 64 → batched gemm_fp16 | **KEEP** prefill 185.8→296 | `cc52fb9` |
| 2 | oneDNN matmul default (qwen35 prefill) | ~1.65× at these shapes | **KEEP** 296→521 | `44febe3`, `dense::prefer_onednn()`; opt-out `IE_QWEN35_NO_ONEDNN`. **Single-card only** (multi-card `ctx_for` static-engine → DEVICE_LOST) |
| 3 | Q6_K SoA decode GEMV (`gemv_q6_soa`) | AoS scalar Q6 = 140 GB/s; SoA int-dot W6A8 | **KEEP** 27B-Q6 5.0→12.8 (2.5×) | `6292b4c`/`233a55d`; default-ON, opt-out `IE_QWEN35_NO_Q6_SOA` |
| 4 | Q8_0 SoA fast lane (ssm_out, 1-GPU wire) | F16-expand = 2× bytes; native Q8_0-SoA int-dot ~80% BW | **KEEP** | `upload_q8_soa` |
| 5 | Q5_K→Q8_0 requant SoA (attn_k/attn_output) | Q5_K read as F16 = ~3× native bytes; requant to Q8 SoA ~80% BW, near-lossless | **KEEP** | `upload_requant_q5k_q8_soa` |
| 6 | int-dot Q4_K wire-up (`gemv_q4i8_T`) | "plain `gemv_q4_K` is slow, wire int-dot like Q6" | **REVERT** neutral ~+1%; kernels same speed (0.1273 vs 0.1289 ms) | `IE_QWEN35_NO_Q4_I8` (2026-06-20) |
| 7 | SoA-Q4 decode repack (`gemv_q4_soa`) | port llama reorder-MMVQ AoS→SoA | **SPECIALIZE/REVERT** decode +6% but **prefill −24%**; VTune says occupancy-bound not BW (baseline already ~64% on big FFN). Q4 GEMV is NOT the bottleneck | opt-in `IE_QWEN35_Q4_SOA`, default OFF (2026-06-20/22) |
| 8 | **Q4_K reorder** (llama 3-region global SoA, `gemv_q4_K_reorder_q8`, `src/ops/gemv_q8dot.cpp:241`) | reorder layout = the real BW lever where dp4a/SoA were flat | **SPECIALIZE/opt-in** clean A/B **+4–7%** (~16→17.0), full FFN gate+up+down. NPWG knob `IE_Q4K_REORDER_NPWG` default 32 (32/16 tie best; npwg=1 catastrophic 6.65 — 17KB SLM kills occupancy). Still loses to llama | `60aaa49`/`15c7dda`/`15e67a3`; opt-in `IE_QWEN35_Q4K_REORDER` |
| 9 | Quantize-hoist (one Q8_1/norm reused by gate+up) | save redundant quant launches | **REVERT** neutral (launch-count not the lever) | opt-in `IE_QWEN35_QUANT_HOIST`, default OFF |
| 10 | MTP spec-decode (`ie run --spec`, native blk.64 head) | self-EAGLE; verify(T=K) amortizes weight BW | **SHIPPED but prompt-dependent**: ~1.0× prose / ~1.2× code; **net-NEGATIVE on prose at K>2**; latest in-engine A/B cyber 1.13× / short-factual 0.77× / scorecard 13.19 vs 16.33 = LOSS. **THE WALL:** llama base 1.6× ours → spec is a multiplier on OUR base, can't close the gap. **LOSSLESS** (token-identical, per-pos DeltaNet checkpoint `Qwen35SpecCheckpoint` killed rollback re-forward 76→0.6ms) | `--spec`/`--spec-k` (default K=2, `engine.hpp:64`); productionized in `Engine::generate` |
| 11 | Hybrid tensor-parallel (FFN-slice, `Qwen35TpModel`) | both cards on one token | **REVERT/NO-GO** 12.6 ≤ 13 split baseline; 1 host-bounce all-reduce/layer ≈ the whole FFN it splits (no P2P on 2×B70) | `IE_QWEN35_TP=1`; Phase-0 NO-GO |
| 12 | Q8 single-GPU (`IE_QWEN35_Q8`) | fit 28.56 GB on one 32 GB card | **REVERT (this box)** 0.6 tok/s host-spill thrash; 2-GPU split (14.3/card) is correct | kept gated for >32 GB cards |
| 13 | no-SLM gemv A/B | remove SLM staging | **REVERT** regressed ~10× | — |
| 14 | DeltaNet recurrence as decode bottleneck | "recurrence is the launch-bound tax" | **FALSIFIED** 1.4% of decode, FLAT across ctx | `81b9394` |
| 15 | Wire vec/v2/int8 `fa2_decode` into 27B | long-ctx attn 24% BW @16K; vec is the lane=D-slice root-cause fix | **DEFER** — exists ONLY in qwen3moe; **`--int8-kv` is a silent no-op for 27B** (`qwen35_dense` never branches `kv.is_int8()`). vec landmine: `q_vals[32]` OK at head_dim=256 ONLY at NTH_KQ=8; overflows at NTH_KQ=4 → must bump to [64] before exposing the knob | not wired |

---

## 6. Shape dispatch (path per T / ctx / dtype + env gates)

**Decode (T==1):** per-weight by GGUF dtype —
- Q6_K → `gemv_q6soa_T` (SoA int-dot W6A8, default-on; opt-out `IE_QWEN35_NO_Q6_SOA`).
- Q8_0 (ssm_out) → `gemv_q8soa_T` (SoA int-dot W8A8 ~80% BW).
- Q5_K (attn_k/attn_output) → requant→Q8-SoA `gemv_q8soa_T`.
- Q4_K → AoS `dense::gemv_q_T` (default); `IE_QWEN35_Q4K_REORDER` → `gemv_q4_K_reorder_q8` (+4–7%, opt-in);
  `IE_QWEN35_Q4_SOA` → `gemv_q4soa_T` (opt-in, default-OFF — prefill-regressing).
- Full-attn SDPA → `full_attention_fa2_decode` (v1, KV-stationary split-K), `qwen35_dense.cpp:1488`.

**Prefill (T≥2):** oneDNN `gemm_fp16_onednn` matmul (default; `IE_QWEN35_NO_ONEDNN` forces bit-exact
`gemm_fp16` for cosine oracle). SoA weights → `dequant_*_soa_to_Bt` → gemm. Attn → `full_attention`
(naive) — DeltaNet layers chunk at **T≤512** (`IE_QWEN35_PREFILL_CHUNK` override; §1 HW-hang cap).

**Spec-verify (2 ≤ T ≤ 16):** batched int-dot `gemv_q6_soa_q8_batched` / `gemv_q4_K_q8s_batched`
(`96394d7`, isum-elim + T-bucket, verify −9.8%); attn = **loop** v1 `fa2_decode` over T positions
(tiled prefill kernel was 4× slower at T=4 — GPU-starved). Lossless-by-construction. Opt-out
`IE_QWEN35_NO_BATCHED_VERIFY` / `IE_QWEN35_NO_FA2_PREFILL`.

**Multi-GPU:** `--gpus 2` → `Qwen35SplitModel` (native Q8_0-SoA packed, 14.28 GB/card, oneDNN OFF).
`IE_QWEN35_TP=1` → `Qwen35TpModel` (NO-GO, kept for reference).

**Default forward is byte-identical** to the AoS/oracle path: all fast lanes that touch the crown PPL
gate are either default-on AND PPL-verified (Q6/Q8 SoA) or opt-in pending GPU validation (Q4 reorder/SoA).

---

## 7. Layout / scheduler state

- **Reordered / SoA / fused:** Q6_K weights → SoA bit-planes (`Q6SoaW` lo/hi/sc/d) at load.
  Q8_0 → SoA (`Q8SoaW` qs+d, bit-exact de-interleave). Q5_K → device-requant to Q8-SoA.
  Q4_K → AoS by default, optional 3-region global reorder (`upload_q4_reorder`, same bytes as AoS).
  ssm_alpha/beta F32 → transposed+N-padded(48→64) fp16 for batched gemm. ssm_conv1d/ssm_norm cast fp16.
- **Fusion:** `residual_add_rms_norm_fused` (pre-FFN); optional quant-hoist (one Q8_1/norm feeds gate+up,
  decode T==1, opt-in). split_q_gate_per_head + sigmoid_gate fused around full-attn.
- **Host-stall / launch state:** decode is ~100% GPU-busy (NOT host-launch-bound after the
  engine-wide `enable_profiling` opt-in fix `a36c138`). Prefill 16K is launch-bound (32 serial chunks).
- **KV mode:** fp16, 16 full-attn layers only (DeltaNet layers = recurrent state, no KV). int8-KV
  plumbs through engine.cpp but is a **no-op for 27B** (not wired into `qwen35_dense`).

---

## 8. Open frontier

**The honest verdict:** 27B is the **WIN-prefill / LOSE-decode** arch. Prefill now **WINS at EVERY
length (1.96–3.08×) vs current llama-SYCL `fdc3db9b6`** (§3) — a DEFENSIBLE public claim; the old
"≈718 flips it" worry is refuted, AND **the long-ctx prefill collapse is FIXED 2026-06-26** (the
head_dim-256 full-attn layers now use the Gemma wide-tile kernel for ctx≥6144 → 16K 0.52× LOSS →
2.13× WIN; the one prefill weakness is gone — see §3 + the long-ctx tile note). **The same lever was
ported to crown (`bf73093`, +57%@16K, 6.4527 bit-exact) and 80B (`442af21`, +12.5%@8.4K, bit-identical)
this session — but BOTH crown and 80B REMAIN long-ctx-prefill losses (crown ~0.35× vs llama), because
their long-ctx bottleneck is MoE-GEMM-bound, not the attention the tile lever fixes. The straight oneDNN
MoE-prefill port is also FALSIFIED by small-M (`c96909a`: 256 experts/top-8 @ T≤512 chunk = 16
rows/expert).** Decode is
the one genuine weak spot, and it lives on the **commoditized dense-GEMV-BW axis** where llama's
mature SYCL MMVQ leads — **not** the DeltaNet moat we actually win. The 06-25 roofline settled it:
**no single layout lever closes 17→23** (reorder maxed at +7%, SoA/dp4a flat, recurrence falsified
at 1.4%). Closing it is occupancy/WG/dp4a micro-tuning — a grind, not a moat.

**Ranked next bottlenecks (justified + falsifier):**

1. **Short-ctx decode 17→~20 (occupancy lever).** Observation: two big GEMVs at ~46% BW vs llama
   ~60%, but VTune says **occupancy-bound (27%, 85% XVE-stalled)**, not BW-saturated. Mechanism: a
   **no-SLM MMV Q4_K variant** (q8 from global like llama, 1 row/WG, more subgroups in flight) raises
   occupancy where the 17KB-SLM reorder kernel can't (npwg=1 already proved SLM is the occupancy
   killer). **Falsifier:** if a no-SLM MMV Q4_K closes 46→~60% BW it's the one remaining real lever;
   if it's flat (matches the SoA/reorder ceiling) then 27B decode is genuinely at its occupancy wall
   and should be abandoned for byte-reduction. **Verdict: commoditized grind, modest ROI.**

2. **Long-ctx decode attention (16K, 24% BW).** Observation: v1 `fa2_decode` reduce-on-softmax-chain
   = 29.8 ms/tok @16K. Mechanism: **wire the `fa2_decode_vec` port** (lane=D-slice, defer reduce) +
   **wire int8-kv** (halves the 4.29 GB/tok KV read) into `qwen35_dense` — both exist but only in
   qwen3moe. Est 29.8→~8–10 ms; int8-kv gave Coder 1.29×@8K. **Falsifier:** vec must be GPU-validated
   at head_dim=256 (bump `q_vals[]` to [64], keep NTH_KQ=8). **Verdict: a real lever for the long-ctx
   regime** (separable + stackable), but short-ctx-neutral (attn only ~3%).

3. **Byte-reduction (EXL3-3bit).** The real decode win per the ledger: 3.0 vs 4.5 bpw = ~33% fewer
   bytes = ~33% faster decode for the single-card regime. Runtime shipped (`project_exl3`); needs a
   3bpw 27B model. **Verdict: the cleanest path past the BW wall** — zero new kernel, but model-gated.

4. **[CLOSED] Long-ctx prefill collapse — FIXED 2026-06-26** via the hd256 tile lever (16K 0.52× LOSS
   → 2.13× WIN, §3). The DeltaNet T≤512 chunk-cap lift was separately TESTED + FALSIFIED this session
   (`0a293c0`): safe on box but +6.7%@1024 / REGRESSES@2048 — research shows the "fix" is timing-luck
   with a DEVICE_LOST hazard (Triton #6658). Keep `IE_QWEN35_PREFILL_CHUNK` default 512; do not
   re-investigate.

**De-prioritized / closed:** dense Q4 GEMV grinding (negative twice — int-dot neutral, SoA
net-negative); spec-decode as a 27B decode win (prompt-dependent, net-negative on prose, can't beat
llama's 1.6× base); hybrid TP (NO-GO no-P2P); single-GPU Q8 (host-spill). The strategic call (commits
`81b9394`, `d15bbcd`): **publish on the prefill moat + lossless spec, not the commoditized dense decode.**
