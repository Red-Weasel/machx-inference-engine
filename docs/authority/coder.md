# Coder — Qwen3-Coder-30B-A3B (`qwen3moe`) — Living Authority Doc

> Per-arch single source of truth (framework: `docs/authority/README.md`). `MASTER_DEV_PLAN.md`
> stays the cross-arch roadmap; this is the deep optimization state for the Coder / `qwen3moe` arch.
> **Update this doc at every Coder milestone.** If reality and this doc disagree, fixing the doc is part of the change.

---

## 0. Identity

| Field | Value |
|---|---|
| Family | `qwen3moe` — plain dense transformer (QK-norm attn + partial RoPE + GQA) with a **top-k MoE FFN**. NO DeltaNet, NO shared expert. The 6th arch family. |
| Model / sizes | Qwen3-Coder-30B-A3B-Instruct (30B total, ~3B active). Config: 48 layers, hidden 2048, head_dim 128, **32 q / 4 kv heads (GQA=8)**, **128 experts / top-8 used**, expert_ffn 768, vocab 151936. |
| GGUF | `~/models/Qwen3-Coder-30B-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (~18.6 GB, lmstudio-community). Per-layer **mixed dtype**: attn projections Q4_K/Q6_K/Q5_K/Q8_0; MoE `gate/up_exps` Q4_K, `down_exps` Q6_K (some layers Q4_K). |
| Forward path | `src/model/qwen3moe.cpp` (`Qwen3MoeModel`), header `include/ie/qwen3moe.hpp`. Owned kernels `src/ops/moe_qwen3.cpp` (generalized int-dot down), `src/ops/moe.cpp` (router + on-device pack), `src/ops/attention.cpp` (tile FA). **Additive — the crown's `qwen36.cpp`/`moe_fused.cpp` are UNTOUCHED.** |
| Engine dispatch tag | `ModelArch::kQwen3Moe`. Loaded/forwarded via `ie run/serve`; chat = plain ChatML, eos `<|im_end|>` (151645), no BOS, NON-reasoning (no `<think>` injection — gated on chat_template, commit `8c568a6`). |
| One-line standing (2026-06-26) | **The COMMODITIZED axis — a plain MoE transformer where llama's mature SYCL kernels are the bar, no DeltaNet moat.** After a multi-session prefill campaign Coder now **WINS prefill @4K (~1.28×) and @8K (~1.08×)**, loses @16K (~0.88×); **decode LOSES 49.5 vs 71.5 = 0.69×** — the **launch-fusion line is now CLOSED** (on-device routing = perf-neutral `4eff28a`; `res_rms` fusion = +2-3% but not bit-exact, shipped opt-in `IE_QWEN3MOE_RES_RMS_FUSED`): the residual gap is **commodity intra-kernel occupancy/efficiency** (MoE GEMVs ~42-46% BW vs llama ~60%) — **no single bounded lever closes it**; not a sprint priority. |

---

## 1. Envelope (benchmark contract)

- **Hardware:** 1×B70 (Battlemage BMG-G31, 32 GB VRAM, 608 GB/s, PCIe Gen4 x8). Single-card — the model fits one card and the winning prefill levers (oneDNN / MoE-XMX) bind context to card 0, so Coder is a **1-GPU story** (multi-card oneDNN = `DEVICE_LOST`).
- **Software:** oneAPI 2026.0 / NEO 26.18 / kernel 6.17.0-35.
- **Model + quant:** the single Q4_K_M GGUF above (same file both engines — apples-to-apples requires byte-identical weights).
- **Target modes (priority):** (1) interactive **decode** (T=1, short/mid ctx) — the open loss; (2) **short/mid prefill** (≤8K, where real prompts live) — now winning; (3) long prefill (16K) — secondary, still behind.
- **Competitor:** `llama.cpp` **SYCL** backend (icpx build, `level_zero:0` pinned to ONE card — NOT `level_zero:*` which silently uses both B70s + iGPU and invalidates the compare). Protocol: `ours = ie-bench --prefill P --decode N` vs `llama = llama-bench -pP -nN`; GPU verified idle (≤600 MHz, swap baseline 0) before each load; discard cold/JIT first run (`--warmup 4`); one model at a time.
- **Quality gate (two-part, both required):** (a) **crown PPL 6.4527 bit-exact** — the additive-correctness proof every Coder change must hold (`./build/tools/ie-perplexity`, ≤6.57); (b) **Coder's own PPL** must hold (see §2). PPL is necessary, not sufficient — also greedy-coherence + factual prompt.

---

## 2. Correctness

- **PPL gate (Coder):** full-causal **PPL ≈ 11.98 / NLL 2.4834** (`ie-qwen3moe-test` / `ie-perplexity` 4th branch, full ctx). NB: the 2026-06-22 clean-sweep harness reports **7.8113** for Coder on a different windowing — both are self-consistent within their method; track deltas within one harness, not across.
  - ⚠ **The default `ie-perplexity` runs T=1 streaming, which BYPASSES every T>1 prefill kernel** (tile/v2/XMX all give bit-identical, meaningless PPL). To gate a **prefill-attention** change you MUST force T>1:
    `./build/tools/ie-perplexity --gguf <coder> --prefill-chunk 256`
    (Lesson from the tile ship: a coherence-only default-on regressed France→"London"; the `--prefill-chunk 256` PPL caught it; two precision fixes — store Q unscaled fp16 + apply 1/√d post-dot in fp32; softmax weights fp16→fp32 — brought tile to **+0.39% PPL / +0.12% NLL** within gate, France→"Paris".)
- **Reference oracle:** llama.cpp on the SAME GGUF+corpus (windowed `-c 256`) scores **~20.19**; ours 11.99 → the SAME directional ratio (~0.6) as 27B (5.34/8.38) and 8B (18.9/24.5) → forward correct + oracle-consistent.
- **Deterministic-gen check:** greedy "The capital of France is Paris. The capital of Germany is Berlin…" coherent + on-topic; codegen prompts answer correctly (plain ChatML, no `<think>`).
- **Fused == unfused gate:** fused MoE PPL **11.9981** == unfused **11.9856** (Δ0.013) is the authoritative fused-correctness check (per-token greedy is dominated by the prefill fp16 floor so PPL equality is the right gate). Int-dot W4A8 path == fp16 PPL bit-stable (11.98/NLL 2.4834).
- **Silent-bug landmines (Coder-specific):**
  1. **`<think>` injection derails Coder.** Qwen3-Coder HAS `<think>`/`</think>` in vocab (151667/8) but is a NON-reasoning Instruct model. Gate `model_has_think` on the GGUF **chat_template** containing `<think>` (NOT token presence), else the engine injects `<think>\n\n</think>` → free-continuation garbage. (`8c568a6`)
  2. **T==1 decode reduce must memset `ws_ffn_y_` first** (`qwen3moe.cpp:587`/`527`). `moe_prefill_reduce` ACCUMULATES (`acc = y; acc += …`); without the zeroing the decode reduce adds onto stale state (≈5.37 vs correct ≈−0.017) → residual explosion → garbage. **T>1 prefill is left untouched** (its reduce overwrites every token slot each layer — the validated path; memsetting it would perturb the validated PPL).
  3. **Mixed expert dtype per layer** — `down_exps` is Q6_K on most layers but **Q4_K on some**; dispatch on `w.down_dt`, never assume Q6_K (an unconditional Q6_K down dispatch NaN'd on the analogous 80B GGUF).
  4. **Prefill-attention kernels must be `--prefill-chunk` PPL-gated, never streaming** (see above).

---

## 3. Pareto frontier (measured, clean-box, dated)

**Decode (tg128), 1×B70, same Q4_K_M, `ie-bench` vs `llama-bench`:**

| Regime | Ours | llama SYCL | Verdict | Bottleneck class | Date |
|---|---:|---:|:---:|---|---|
| Decode tg128, short ctx | **49.5** | **71.5** | **LOSE 0.69×** | launch / host-stall (NOT BW: 16.2% of 608 GB/s ceiling) | 2026-06-26 |
| Decode @8K (int8-KV opt-in) | 41.1 | 84.8 | LOSE | KV-BW partly; attn fa2_partial 62% latency-bound | 2026-06-22 |
| Decode @16K | ~22–31 (collapsing) | **84.8 (flat)** | LOSE 3.85× | fa2_partial decode-attn occupancy (~11% BW, latency-bound) | 2026-06-23 |

**Prefill (tok/s), 1×B70, single-card `level_zero:0`, current default stack (tile + wide-tile + MoE-XMX/oneDNN):**

| T (prefill) | Ours | llama SYCL | Verdict | Bottleneck class | Date |
|---:|---:|---:|:---:|---|---|
| 4096 | **~1303** | ~1016 | **WIN ~1.28×** | MoE-GEMM (oneDNN/XMX) + attn tile | 2026-06-24 (`2255497`) |
| 8192 | **~919** | ~849 | **WIN ~1.08×** | attn tile + MoE-GEMM | 2026-06-24 (`2255497`) |
| 16384 | ~547 | ~624 | LOSE ~0.88× | **attention O(T²) at SG16 tile ceiling** (61% of wall) | 2026-06-24 |

> **Standing has IMPROVED dramatically over the 2026-06-22 snapshot** (which read prefill `lose 0.55×` / decode `lose 0.66×`, table below). That row is now **STALE for prefill** — superseded by the tile→wide-tile→MoE-XMX/oneDNN campaign. Decode is essentially unchanged (49.5 vs 71.5).

**(STALE, kept for arc) 2026-06-22 clean sweep — `docs/benchmark_results_2026-06-22.md`:**

| T_pp | ours pp | llama pp | ratio | | tg128 ours | tg128 llama | ratio |
|---:|---:|---:|:---:|---|---:|---:|:---:|
| 512 | 646–667 | 1179–1196 | 0.55× | | 49.5–49.7 | 71.5–75.5 | 0.66–0.69× |
| 4096 | 240 | 991 | 0.24× | | | | |
| 16384 | 50.6 | 601 | 0.08× | | | | |

(The 16K collapse there was BEFORE the prefill tile/XMX work; the residual long-ctx weakness is now the attention tile ceiling, not unchunked O(T²).)

**Net:** Coder is **prefill-competitive (WIN @4–8K, the interactive range)** and **decode-behind (0.69×)**. It is the commoditized axis — there is no DeltaNet moat here (contrast: Crown wins short-ctx prefill + holds decode parity, 80B/27B have the DeltaNet hybrid moat).

---

## 4. Bottleneck map

**Decode (T=1) — `--kprofile-decode`, GPU-tot ~15.3–15.7 ms/token (vs ~18.9 ms wall → ~3.6 ms / 19% non-kernel bubbles):**

| Kernel | % of decode | Class |
|---|---:|---|
| MoE experts total | **~43.5%** | `moe_pfl_gate_q8` 17.6–17.9 + `down_q8_6k_gen` 13.1–13.3 + `down_q8_4k_gen` 12.0–12.3 |
| dense attn `fa2_partial` | ~15.5–16.3% | latency/occupancy-bound (~11% BW @ long ctx) |
| **`gemv_fp16`** (attn proj, fp16-expanded) | ~14.1–14.3% | secondary lever (Q4_K→fp16 dequant path) |
| `gemv_q4k` (small projections) | ~7.1–7.2% | AoS, reorder REGRESSES |
| lm_head | ~6.6% | 1 call |

- **Roofline (2026-06-26):** decode = **1.995 GB/tok**, 608 GB/s ceiling = 304 tok/s → we run at **16.2% of ceiling**, llama at **23.5%** (~5× headroom on both). **DECODE IS NOT BANDWIDTH-BOUND** — it is **launch / occupancy / host-stall bound.** Per decode MoE layer we: memcpy router logits D→H + host `route_from_logits` + memcpy packing H→D + a **per-layer `q.wait()` drain** (`qwen3moe.cpp:444/466/874`) = **~144 GPU bubbles/token**. llama's whole ~45% edge = dispatch amortization + on-device router ids + NO host wait (its MoE experts are STANDARD `block_q4_K`, explicitly NOT reordered — `should_reorder_tensor` requires `OP_MUL_MAT`, `MUL_MAT_ID` bails).
- ⛔ **Reorder/SoA on MoE experts ≈ 0% gain at decode (REJECTED)** — not BW-bound, so a faster layout buys nothing. Do NOT chase it like the 27B.

**Prefill (T>1) — historical kprofile @pp512 warm:** ~84% of prefill = MoE EXPERT GEMMs (`down` ~69% + `gate` ~12%), attention ~3% at 512. The picture **inverts at long ctx**: @16K **attention is ~61% of wall** (O(T²)), MoE-GEMM the rest. So the two prefill levers are MoE-GEMM (short/mid) and attention-tile (long). The attention tile kernel is near its **practical SG16 ceiling** (register-block KQ spilled, GQA head-packing a wash — both measured, §5).

---

## 5. Hypothesis ledger (every lever; KEEP / REVERT / SPECIALIZE / DEFER)

### Decode levers

| # | Lever | Observation → mechanism | Result | Commit / gate |
|---|---|---|---|---|
| D1 | **FA-2 decode attention** | Coder was the only model still on naive `full_attention` at decode → wire `full_attention_fa2_decode` + `ensure_attn_partials`. | **KEEP** — decode ~19.3→37.4 (1.94× vs the then-naive baseline; that "1.94× vs llama" framing was later found BOGUS/stale-llama). PPL 11.98 held. | `a325cd1` |
| D2 | **Per-expert SoA-Q6 repack (decode)** | Q6_K `down_exps` read AoS (interleaved ql/qh → uncoalesced) = 25.7% of decode; crown already SoA-repacks. Scaffolded in qwen3moe (`soa` arg) but never wired (`soa=false` hardcoded). | **SPECIALIZE (Q6_K only)** — decode 43.5→49.5 (+13.8%), prefill +2.6%. **DTYPE-CONDITIONAL: Q6_K SoA = 2.38× (0.204→0.086 ms/call); Q4_K SoA REGRESSES** (gate +38%, Q4_K AoS already coalesced) → Q4_K stays AoS. | `c8dd418`, opt-out `IE_NO_MOE_SOA` |
| D3 | **T==1 decode → int-dot fused `_gen` path** | Crown's fp16 `moe_ffn_decode` was tuned for top-2; pack the single token's K=8 experts as 8 rows → gather + `gate_up_silu_q8` + `down_q8_gen` + reduce packs top-8 far better. | **KEEP** — +28% decode (37.5→48), PPL-identical. (Unlocked by the `ws_ffn_y_` memset — landmine §2.2.) | task #6, default-on, opt-out `IE_QWEN3MOE_FP16_DECODE` |
| D4 | **INT8-KV decode** | Built-but-unwired `full_attention_fa2_decode_int8` (crown/dense only) → mirror into qwen3moe per-layer (`qwen3moe.cpp:577`). | **KEEP (opt-in)** — decode @8K 31.8→41.1 = **1.29×** (crown 1.26×). Coder PPL 11.98 holds; crown 6.4527→6.46 (+0.1%, ≤6.57). **Tapers to 1.06× @16K.** | `d70065e`, opt-in `--int8-kv` |
| D5 | **On-device decode routing** | Decode looked launch/host-stall bound (§4). Move router softmax+top-k+pack ON-DEVICE from the already-on-device `ws_router_logits_` (`moe_router` E=256→E=128) + on-device T=1 pack-build → **delete the per-layer `q.wait()`** so all 48 layers enqueue/run-ahead. | **DEFER (foundation) — FALSIFIED `4eff28a`.** Built (2 kernels `moe_topk_from_logits` `src/ops/moe.cpp:349` + `moe_build_pack_decode` `:406`). MEASURED: PPL 11.98→11.97 LOSSLESS ✓, decode ~50.9→~51.1 **PERF-NEUTRAL** — host round-trip was NOT the bottleneck (FALSIFIED). Decode is bound by the **launch COUNT/occupancy**; the real lever is kernel FUSION. Kept as its lossless prerequisite. | env `IE_QWEN3MOE_ONDEVICE_ROUTE`, **default-OFF** |
| D6 | **command-graph** (1850→1 launches) | Hypothesis: dispatch amortization. | **REVERT/DEAD** — llama wins WITH its own graph default-OFF (`GGML_SYCL_DISABLE_GRAPH=1`); we're already 97% GPU-busy (single in-order queue, 1 wait/token); ≤3% to gain. | not pursued |
| D7 | **W4A8 / dtype change** | Hypothesis: our quant is the gap. | **DEAD** — llama's Q4_K decode is ALSO W4A8 dp4a (`vec_dot_q4_K_q8_1`). Dtype is not the gap. | — |
| D8 | **WG-geometry (llama MMV_Y=1, 1 sg/WG)** | 4 lenses PREDICTED 1.3–1.5×. | **FALSIFIED BY MEASUREMENT** — 0.95× (AoS 16.3 > SoA-MMV 15.4 > SoA-fat 14.8). | gated-off `IE_QWEN35_Q4_SOA_MMV` |
| D9 | **Q4_K SoA int-dot decode** | Coalescing hypothesis. | **REVERT** — parity (occupancy-bound, coalescing no headroom). | gated-off `IE_QWEN35_Q4_SOA` |
| D10 | **quantize_q8_1 hoist** (−150 launches/tok) | Launch-count hypothesis. | **REVERT** — neutral 0.99× → launch-COUNT is not the lever (launch-LATENCY/host-stall is, → D5). | gated-off `IE_QWEN35_QUANT_HOIST` |
| D11 | **`fa2_partial` decode-attn restructure** (long ctx) | `fa2_partial` = 62% of decode @16K @ ~11% BW → latency/occupancy-bound; tried TARGET_SUPER occupancy, reduce-blocking, softmax-defer, no-SLM, manual butterfly-shuffle, tile-size, half8-vectorize, AND a faithful llama `fattn-vec` port. | **DEAD (7+ approaches flat in clean throughput)** — gap is STRUCTURAL (our split-K+SLM+combine+GQA-redundancy carries ~4× more GPU overhead than llama's lean `flash_attn_ext_vec`). Needs a ground-up cool-box port; **de-prioritized vs the prefill win** (commodity axis). ⚠ kprofile per-kernel "wins" were PROFILING ARTIFACTS (profiling disables auto-vec). | discard `IE_FA2_DECODE_V2` scaffold |
| D12 | **gemv_fp16 attn-proj → int-dot** | 14.4% of decode (3/layer). GGUF-verified: this is **k-proj (Q8_0) + o-proj (Q5_K) + router (F16)** fp16-expanded (no int-dot *small* GEMV for Q8_0/Q5_K), NOT "Q4_K-expanded" (q-proj Q4_K & v-proj already ride int-dot gemv_q4k/gemv_q6k_small). | **DEFER** — per-call ≈ gemv_q4k (0.0153 vs 0.0151ms despite 3.5× bytes) ⇒ occupancy-floored, not BW; a dtype change alone won't help (only fusion would, → D14). | — |
| D13 | **`res_rms` fusion** (attn-out residual_add + ffn_norm → one pass) | Launch + HBM-roundtrip count is lightly on the decode critical path; the `residual_add_rms_norm_fused` kernel already exists (used by all dense models). | **SHIP OPT-IN** — decode **+2-3%** (55.3 vs 54.2, new-old-new) ✓, but fused 2-pass fp32 ≠ split numerics → **Coder PPL 11.98→12.02 (+0.33%)**, not bit-exact (`round_residual`-through-half did not recover it). Default = lossless split; crown 6.45 / dense bit-exact (shared kernel untouched), ctest 30/30. | env `IE_QWEN3MOE_RES_RMS_FUSED`, **default-OFF** |
| D14 | **Multi-bank QKV fusion** (q+k+v share attn_norm → 1 GEMV, full grid) | The genuine launch+occupancy lever (would beat llama, which never fuses QKV); attacks the occupancy-floored tiny GQA k/v (NK=512). | **NOT BUILT (L-effort)** — blocked by 3 heterogeneous dtypes (Q4_K/Q8_0/Q6_K) ⇒ needs a mixed-dtype multi-bank kernel (gemma's is uniform-Q4_0). Projected ~+4-7%, lossless if native dtypes read. Fund only if Coder decode becomes the priority. | — |

### Prefill levers

| # | Lever | Observation → mechanism | Result | Commit / gate |
|---|---|---|---|---|
| P1 | **GPU-gemm router** | Host `route_token` dot loop (E=128×H=2048 per token × T × 48L) = **~66% of pp512 wall** (~4859 ms), INVISIBLE to `--kprofile` (GPU-only). Move logits to one `gemv_q_T` (F16 router weight), keep only softmax/top-k on host. | **KEEP** — pp512 host 59→GPU 196 = 3.3×, PPL bit-stable. | Tier-3, opt-out `IE_QWEN3MOE_HOST_ROUTER` |
| P2 | **Generalized int-dot W4A8 down** | Crown's int-dot down is HARD-LOCKED to `E_ffn==512`; `moe_prefill_down_q{6,4}k_q8_gen` (`src/ops/moe_qwen3.cpp`) generalize to any `E_ffn%256==0` (Coder 768): each lane walks q8 blocks `{lane,lane+16,…}`, **bit-identical at E_ffn=512**. | **KEEP** — `down_q8_6k_gen` 1720→190 ms (9×); pp512 197→651 = 3.3× (8.1× vs orig 80.6). Also unlocks 80B MoE shapes. | Tier-3, opt-out `IE_QWEN3MOE_NO_Q8` |
| P3 | **FA-2 prefill v2 (query-row-block)** | Pure full-attn → unchunked naive O(T²) `attn_naive_compute` = 73% of prefill, 16K=317s. Keep naive's split-head-dim+reduce inner loop but 1 q-row/subgroup, Br=16 rows/WG sharing one K/V SLM tile → cut redundant KV HBM ~Br×. | **SPECIALIZE (T≥2048)** — 512 0.95× / 4K 1.04× / **16K 2.05×** (317→146s). KEY: naive is **L2-bound (not HBM) ≤4K** → tiling parity there; only past L2 does redundancy hit HBM. | `430ab08`, opt-out `IE_NO_FA2_PREFILL_V2` |
| P4 | **FA-2 prefill TILE (faithful llama port)** | The 16K gap was a MISFRAME: **llama-SYCL uses NO XMX/joint_matrix for attention** ("Todo: use XMX" in `fattn.cpp`; the "tensor cores" line was the CUDA backend). Port llama's tile INNER LOOP: 16 sg × 1 q-row, each lane does 4 COMPLETE register-resident Q·K dots, full head_dim as half2 in registers, distributed softmax/VKQ, ONE width-16 reduce per TILE (not per key). | **KEEP (default Coder prefill)** — 16K 105→324 (3.08×), 4K 2.0× / 8K 2.4×; llama gap 5.9×→1.9×. A high-level-structure-only port was 0.6× — **the perf is the inner-loop thread/register mapping, NOT WG geometry.** | `cfbed29`, opt-out `IE_NO_FA2_PREFILL_TILE`→v2 |
| P5 | **MoE-GEMM on XMX / oneDNN** | llama switches quantized MoE GEMM to fp16-on-XMX above batch 32 (`MMQ_MAX_BATCH_SIZE=32`); we ran int-dot dp4a at batch ~256 on a compute-bound (AI~680) matrix-matrix shape = wrong engine. Per-expert dequant W4/6_K→fp16 + `gemm_fp16`/`gemm_fp16_onednn` over the expert-sorted slices (`moe_xmx_prefill`). | **SPECIALIZE (T≥4096)** — XMX/oneDNN: 4K 1150→1303 ahead of llama; session @16K arc v2 105→tile 352→+MoE-XMX 481→4.6×. oneDNN ~1.65× hand-rolled `gemm_fp16`. Small-batch (2048, ~30 rows/expert) REGRESSES (dequant-per-forward) → **T-gated, int-dot below**. SoA-aware Q6_K dequant keeps decode SoA-Q6K. | `d0cff68`, opt-out `IE_QWEN3MOE_NO_MOE_XMX`, `IE_QWEN3MOE_MOE_XMX_MINT` |
| P6 | **Wide tile (REGDOT + SMALLBC) spread from Gemma** | Gemma's occupancy/regdot attention levers generalized to HD=128. | **KEEP (default)** — Coder prefill **+6%** (4K 1248→1303, 8K 867→919, 16K 535→547); **FLIPS 8K vs llama parity→WIN (919 vs 849 = 1.08×)**, 4K WIN 1.28×. crown PPL 6.45 UNCHANGED (numerically equiv at HD=128). | `2255497`, opt-out `IE_QWEN3MOE_NO_WIDE_TILE` |
| P7 | **GQA head-packing tile** | Share ONE K/V SLM tile across G=4 GQA-sibling q-heads → kill ~gqa× redundant K/V reads. | **DEAD (WASH)** — attention is ALU-bound not KV-bound; B70 L2 absorbs the redundant reads. | gated-off `IE_FA2_TILE_GQA` |
| P8 | **Register-block KQ dot (`_regtile`)** | Bounded chunked staging (CPE=4 + 2 parity accs) to break the single-acc carried dependency. | **DEAD at HD=128 (REGRESSED, spill)** — the 64-iter hd128 dot is too short; WINS only at Gemma hd256/512. | gated-off `IE_FA2_TILE_REGTILE` |
| P9 | **hd128 Bc=16 (tinybc)** | Smaller Bc for more occupancy. | **REVERT (banked negative)** — REGRESSES Coder. | `e7c60ad`, gated-off |
| P10 | **oneDNN on attn projections only** (Tier-2) | Mirror 27B's `prefer_onednn()`. | **REVERT (opt-in only)** — NO win: only routes attn q/k/v/o + lm_head (<5%, H=2048 tiny GEMMs), never the MoE expert path. | opt-in `IE_QWEN3MOE_ONEDNN` |
| P11 | **gemm-orchestration prefill attn** (Q·Kᵀ via gemm_fp16) | `bench_gemm_fp16` hit 26.7 TFLOPS at attn shapes; orchestrate attn as gemm+softmax+gemm. | **DEAD (DELETED)** — query-blocks → ~5000 launches/layer + materialized S = launch-bound → 0.68× v2 (the "13–27×" was shape-misleading). Also was the sole caller hitting `gemm_fp16`'s ragged-N store corruption (fixed + path removed). | DELETED |
| P12 | **Fused XMX flash-attn-2 prefill** | Naive-fused / multi-subgroup XMX kernel. | **DEAD** — multi-sg re-tiling reached ~parity v2 @8K / 0.86× @16K; residual = SLM-staged softmax round-trip needs register-resident softmax. Superseded by P4 tile. | gated-off `IE_FA2_PREFILL_XMX` |
| P13 | **Fused MoE FFN (host counting-sort packer)** | Replace unfused per-token K×4 gemv loop with packed fused ops. | **KEEP** — CORRECTNESS lever not perf (pp512 ~61 vs ~58); fused PPL 11.9981 == unfused 11.9856. Unfused kept as parity oracle. | `60b9d45` family, oracle `IE_QWEN3MOE_UNFUSED` |
| P14 | **host-scheduler: remove ~30k per-op `.wait()`s (prefill)** | Hypothesis: host gaps. | **DEAD (banked)** — only +5% (in-order queue already overlaps GPU work). | — |

---

## 6. Shape dispatch (path per T / ctx / dtype + env gates)

**Attention (`qwen3moe.cpp:803–865`):**
- **T==1 decode:** `full_attention_fa2_decode` (+ INT8 variant if `--int8-kv`). [D1/D4]
- **T>1 prefill (default):** `full_attention_fa2_prefill_tile_gemma` (wide tile, REGDOT+SMALLBC) at HD=128 [P6]. Opt-out chain: `IE_QWEN3MOE_NO_WIDE_TILE` → `full_attention_fa2_prefill_tile` (hd128 tile, P4) → `IE_NO_FA2_PREFILL_TILE` → v2 (`IE_NO_FA2_PREFILL_V2` → naive). Force XMX A/B `IE_FA2_PREFILL_XMX`; A/B variants `IE_FA2_TILE_GQA` / `IE_FA2_TILE_REGTILE` (both gated-off, dead at hd128).

**MoE FFN:**
- **T==1 decode (default):** int-dot fused `_gen` path (`moe_ffn_fused_prefill` with T=1) [D3]; SoA-Q6 banks [D2]. Opt-out `IE_QWEN3MOE_FP16_DECODE` → crown fp16 `moe_ffn_decode`. Oracle `IE_QWEN3MOE_UNFUSED`.
- **Routing (decode):** default = GPU router GEMV + host softmax/top-k + host pack + per-layer `q.wait()`. `IE_QWEN3MOE_ONDEVICE_ROUTE` (T==1, **default-off**) = fully on-device softmax-top-k + pack-build, NO host round-trip, NO `q.wait()` — LOSSLESS but **perf-neutral** (D5 falsified; kept as the kernel-fusion prerequisite).
- **T≥4096 prefill:** MoE-XMX/oneDNN fp16-on-XMX GEMM [P5] when banks admissible (`bank_ok`: Q6_K any layout, Q4_K AoS only; H%256==EF%256==0). Below `IE_QWEN3MOE_MOE_XMX_MINT` (default 4096) → int-dot W4A8 [P2]. Opt-out `IE_QWEN3MOE_NO_MOE_XMX`.
- **int-dot path** active when `IE_QWEN3MOE_NO_Q8` unset AND `H%512==0`; down `_gen` kernels gated on `EF%256==0`.
- **Routing (prefill):** GPU-gemm router [P1] default; `IE_QWEN3MOE_HOST_ROUTER` restores legacy host dot loop. Profiler `IE_QWEN3MOE_PROFILE_HOST`.

**KV:** fp16 default; INT8 shadow per-layer when `--int8-kv` (opt-in) [D4].

---

## 7. Layout / sched state

- **Expert banks:** raw stacked `[E, *, */256]`, per-expert stride = `nbytes/E`. **SoA-repacked ONLY for Q6_K** (`upload_expert_soa` → `repack_moe_q{4,6}k_soa_host`, intra-expert reorder, PPL-free by construction); Q4_K stays AoS (SoA regresses it). Default-on, opt-out `IE_NO_MOE_SOA`. [D2]
- **Router weight:** F32 GGUF → device F16 `[H,E]` transposed at load (for the GPU-gemm router). [P1]
- **Mixed-dtype attn projections:** Q4_K/Q6_K/F16 ride the dense GEMV path; Q5_K/Q8_0 are dequanted to F16 `[K,N]` at load (`upload_weight_auto`). Decode attn-proj currently fp16-expanded (`gemv_fp16`, 14% of decode — D12 defer to int-dot).
- **MoE-XMX scratch:** lazily allocated, default-OFF until T≥minT (round_up(TK,8)-row per-expert GEMM scratch; pad rows absorb tile row-tail spill). [P5]
- **Host-stall / launch state (decode):** the ~19% non-kernel wall was HYPOTHESIZED as per-layer D→H/H→D + `q.wait()` host-stalls (~144 bubbles/token) — but **D5 deleting the per-layer wait left decode FLAT** (perf-neutral), so those bubbles were already overlapped (in-order single queue, ~97% GPU-busy). The real non-kernel cost is the **launch count itself** (~720 launches/token) → the fix is kernel FUSION, not sync removal. [§4, D5]
- **Multi-GPU:** N/A — Coder is single-card (fits one B70; oneDNN/MoE-XMX bind ctx to card 0).

---

## 8. Open frontier

**#1 (CLOSED 2026-06-26) — Decode launch-fusion line: small win, NOT cleanly lossless → gap is commodity intra-kernel efficiency.**
Two experiments now bound this line, and together they settle it:
- *(a) On-device routing [D5], `4eff28a`:* removed the per-layer D→H route + H→D pack + `q.wait()` drain. **PPL 11.98→11.97 LOSSLESS, decode PERF-NEUTRAL** — the host bubbles were already overlapped by the in-order queue. Falsified "host-stall bound."
- *(b) `res_rms` fusion [D13], 2026-06-26:* fused the attn-out `residual_add` + `ffn_norm` into the one-pass `residual_add_rms_norm_fused` (1 fewer launch + one fewer HBM write/read-back of `ws_x`/layer). **Decode +2–3% (55.3 vs 54.2, new-old-new) — a REAL but small win**, confirming launch/HBM-roundtrip count *is* lightly on the critical path (more than D5's wait-removal). BUT the fused 2-pass fp32 accumulation is **not bit-identical** to the split path (FMA/rounding differences inherent to the single-kernel form — `round_residual`-through-half did NOT recover it; both fused variants give 12.02) → **Coder PPL 11.98→12.02 (+0.33%)**. Shipped **opt-in** `IE_QWEN3MOE_RES_RMS_FUSED` (default = lossless split); crown 6.45 / dense bit-exact (shared kernel untouched, ctest 30/30).
- *Conclusion:* the cheap launch-fusion levers yield only **single-digit-% additive** gains and trade a little PPL; reaching llama's 71.5 (a +37% gap, 0.69×) needs **either** the L-effort multi-bank QKV fusion (Card 2 — would beat llama on the attn axis but Coder's Q/K/V are 3 heterogeneous dtypes Q4_K/Q8_0/Q6_K, so it needs a mixed-dtype kernel, ~+4-7%) **or** broad **per-GEMV occupancy / dp4a-loop efficiency** (the MoE experts run ~42-46% effective BW vs llama ~60%) — i.e. **commodity intra-kernel grind**, OR amortization via MTP spec-decode (blocked: Coder GGUF has no NextN head). **No single bounded lever closes it.** Ceiling math: GPU-busy 15.34ms vs wall 19.3ms → even fully eliminating the ~4ms launch overhead floors at 65 tok/s < 71.5, so kernel efficiency is also required. **Do NOT pour the sprint into this tail** — bank the prefill win + the opt-in fusion, defend the moat elsewhere.

**#2 — Long-ctx (16K) prefill attention (the ONLY remaining prefill loss across all arches, 0.88×).**
- 16K is ~61% attention at the **SG16 tile ceiling** (register-block spilled P8, GQA wash P7, faithful port done P4).
- ⛔ **ncols/`nbatch_K` lever = NO-GO (analyzed 2026-06-26, 4-agent workflow).** The premise ("cut redundant causal-KV HBM") is **falsified by P7**: GQA head-packing tested the identical KV-reuse mechanism and WASHED — *attention is ALU/FMA-bound, NOT KV-bound; B70 L2 absorbs the redundant reads*. Raising `ncols` attacks the same L2-absorbed HBM term and leaves the FMA count invariant (same T²/2 dots). Worse, `cpw=2` (Br 16→32) grows SLM 22→28 KB → 5→4 WG/Xe-core = an occupancy LOSS on an occupancy-sensitive kernel. And `nbatch_K` has no home: it's only needed at hd256/512 (= Gemma, which already WINS 16K **1.58×**), not at hd128 (= Coder, the loss, where `vkq[2][8]` doesn't spill). Predicted outcome: wash-to-slight-regression. **Do not build it.**
- ✅ **SG32 IMPLEMENTED + VALIDATED 2026-06-26 (opt-in `IE_FA2_TILE_SG32`).** llama's edge was SIMD32 issue efficiency (warp_size=32, no XMX). Rewrote the wide-tile FA kernel to subgroup-size 32 via a **shadow template param** (`fa2_tile_wide_impl<HD,Bc,REGDOT,SG=SG_SIZE>` + `constexpr SG_SIZE=SG` shadow + `WG_ITEMS=Br*SG_SIZE` → the whole body auto-retargets, 4 additive/gated edits). Register-safe (design-vetted: hot path 16→9 live fp32/lane, no spill; same 16 HW threads/WG). **IGC ACCEPTS reqd_sub_group_size(32) + occupies on BMG-G31** (the one empirical unknown — RESOLVED). **MEASURED (Coder, warm A/B, hd128):** 4K 1268 vs 1286 (−1.4%, attention-light), **8K 918 vs 875 (+4.9%), 16K 572 vs 535 (+6.9%)**. Quality: Coder PPL **2.3686 == 2.3686 SG16 (IDENTICAL** to 4 dp — the 32-lane reduce reassoc is sub-NLL-precision); coherence perfect; crown 6.4527 bit-exact (default SG=16, + crown is hd256 so untouched); ctest 30/30. **HONEST: it does NOT flip 16K to a win** (0.88→~0.92× vs llama) — a real but modest +5-7% on attention-heavy long-ctx, exactly the design's "parity-to-marginal-win, not a crush." Shipped **opt-in** (not bit-exact; −1.4% at 4K).
- ✅ **EXTENDED TO hd256 2026-06-26 (same opt-in flag).** Hoisted `sg32` to function scope + gated the hd256
  dispatcher block (`<256,32/64,REGDOT,32>`; register-safe per design: Bc=32→KVPL=1/DPL=8 = 13 fp32/lane). Covers
  Gemma's SWA layers (hd256, ~5/6 of gemma attention) + the crown/27B/80B long-ctx tile. **Gemma-31B (ie-prompt-bench,
  ~5.5K prompt): prefill 117.6→120.6 = +2.5% overall** — consistent with the mechanism (the SWA layers' +5-7%
  DILUTED by the hd512 GLOBAL layers, which are NOT SG32'd, + MLP). ⚠ Box was warm (decode 2.2 tg = thrashed
  signature; treat the +2.5% as a noisy-positive, clean-box re-bench pending). **Coherence-confirmed** (gemma +
  SG32 → correct Rayleigh-scattering answer); crown 6.4527 bit-exact (default), ctest 30/30.
- ✅ **EXTENDED TO hd512 2026-06-26 (same opt-in flag) — ALL gemma attention layers now SG32-capable.** SG32
  forces Bc=32 at hd512 (the smallbc default Bc=16 → KVPL=0). Register-safe per design (DPL=16/KVPL=1 = 21
  fp32/lane < spill cliff). hd512 = Gemma's GLOBAL layers, which do **FULL-causal O(T²)** attention → at 16K
  they DOMINATE gemma's attention cost (the windowed SWA hd256 is cheaper), so this is the **bigger** gemma-16K
  lever, not the smaller 1/6. CORRECT (gemma all-layers SG32 → correct prime-number answer), crown 6.4527
  bit-exact (default), ctest 30/30. ⚠ **GEMMA PERF UNCONFIRMABLE on this bench (within noise).** Cleaner-box
  re-bench (new-old-new, ~5.5K, box recovered to load 1.32/GPU-idle): SG16 115.5 → SG32 120.4 → SG16 120.1 — the
  two identical SG16 runs differ 4% (warmup drift) and SG32 sits INSIDE that spread, so the expected +2.5-5%
  gemma gain is below the bench noise floor → **unresolvable** (the gemma-31B prefill is inherently too noisy
  here; decode also persistently 2.2 tg = a gemma-path behavior, not box thrash). **The CONFIRMED SG32 number
  stays hd128 +5-7% (Coder, cleanly resolvable); the hd256/hd512 gemma extension is correct + mechanism-sound but
  its perf is unconfirmed.** **REMAINING (low-ROI):** a `T≥6144` gate + default-on decision; a gemma 16K sweep
  on a truly clean box (where the O(T²) global layers would show the biggest SG32 effect, if resolvable).

**#3 (deferred) — `fa2_partial` long-ctx decode-attn.**
- 62% of decode @16K at ~11% BW; 7+ kernel restructurings + a `fattn-vec` port ALL flat → STRUCTURAL, needs a ground-up cool-box port. De-prioritized.

**#4 (deferred) — `gemv_fp16` attn-proj → native int-dot. [D12]** 14% of decode; secondary to #1.

---

### The honest verdict

**Coder is the COMMODITIZED axis — a plain MoE transformer with no DeltaNet moat, where llama's mature SYCL kernels are the bar.** Every "win" here is a hard-fought port of llama's own technique (tile FA inner loop, MoE-on-XMX batch-32 switch, oneDNN GEMM), not a structural advantage. **Prefill is now a genuine competitive story (WIN @4–8K), bought by a multi-session campaign.** **Decode (0.69×) — the launch-fusion line is now CLOSED (2026-06-26).** Two experiments bound it: on-device routing was perf-neutral (host bubbles already overlapped), and the cheap `res_rms` fusion gave a real but small **+2-3%** that isn't cleanly lossless (+0.33% PPL → opt-in). So launch/HBM-count is only *lightly* on the critical path. The residual gap is **commodity intra-kernel occupancy/efficiency** (MoE GEMVs at ~42-46% effective BW vs llama's ~60%) plus a launch-overhead floor that, even if fully removed, caps at ~65 < 71.5 tok/s. The only paths past it are the **L-effort multi-bank QKV fusion** (Card 2/D14 — blocked on a mixed-dtype kernel for Q4_K/Q8_0/Q6_K; ~+4-7% and would beat llama on the attn axis) or **MTP spec-decode amortization** (blocked: Coder GGUF has no NextN head). **Strategic framing: bank the prefill win + the opt-in fusion; do NOT pour the sprint into this commodity decode tail. The moat lives on Crown PREFILL + the DeltaNet hybrids' structural day-one advantage, not here.**

---

*Sources: `MASTER_DEV_PLAN.md` §7 (top banner + IN-FLIGHT 2026-06-26) / §2; `docs/benchmark_results_2026-06-22.md`; memory `project_qwen3moe`, `perf_harness_and_prefill_blocker_2026-06-22`, `decode_fanout_int8kv_2026-06-22`, `project_prefill_tile_win_2026-06-23`; `src/model/qwen3moe.cpp`, `src/ops/moe.cpp`, `src/ops/moe_qwen3.cpp`. Commits: `b65fccc`(arch) `8c568a6`(chat) `60b9d45`(fused) `a325cd1`(FA-2 decode) `c8dd418`(SoA-Q6 +13.8%) `d70065e`(INT8-KV) `430ab08`(FA-2 prefill v2) `cfbed29`(prefill tile) `d0cff68`(MoE-XMX/oneDNN) `2255497`(wide tile) `e7c60ad`(tinybc negative); on-device routing SHIPPED default-OFF, FALSIFIED perf-neutral `4eff28a` (`src/ops/moe.cpp:349/406`, env `IE_QWEN3MOE_ONDEVICE_ROUTE`); `11b0a74`(res_rms fusion opt-in + launch-fusion line CLOSED).*
