# Hand-off — Qwen3.6 Inference Engine

> **Read `MASTER_DEV_PLAN.md` first** (repo root) — it is the build's
> single source of truth. This file is the deep engine-internals
> appendix: kernel history, perf ledger, measurement rules, tooling.

**Last updated**: 2026-06-10 late (**TOTAL CROWN — both metrics lead llama.cpp SYCL master same-hour**)
**Repo**: `/home/weezy/00 - Inference Engine` on branch `main` (everything committed)
**Status**: v1 SHIPPED (RELEASE.md). **TOTAL CROWN 2026-06-10: prefill pp512 1144 ± 5 vs llama.cpp SYCL master 1064 ± 8 (+7.6%, alternating same-hour runs 1147/1064/1139/1063/1147); decode turbo 84.1 vs 81.31 (+3.5%), default 81.0 @ PPL 6.52.**  Prefill crown won by executing `docs/prefill_crown_plan.md`: load-time per-expert SoA weight repack (`ie/quant_soa.hpp`, kill switch `IE_NO_MOE_SOA=1`) + int-dot MoE prefill **default ON** (`IE_NO_MOE_Q8=1` opts out): `block_q8_1s` activations (48 B, split half-sums s0/s1 → zero isum dp4a), full-K register lattices for stage-1 gate/up (189→157 ms) and stage-2 down (204→101 ms).  PPL 6.52 at production defaults; unit tests green (incl. new `quant_soa_test`); decode guard A/B held through the repack (81.25/80.72/81.84).  Detail: `docs/benchmark_matrix_2026-06-09.md` §v1.6.  All Vulkan llama.cpp builds beaten outright on everything.  The `deltanet_recurrence` non-determinism investigation is **complete** (HW-level; chunk prefill at T≤256); see `docs/known_bugs.md`. Do **not** re-investigate. **Measurement rule: GPU swings ±40 tok/s heat-soaked — all A/Bs order-controlled (new-old-new); discard first run after rebuild (JIT).**  Remaining headroom (not needed for the win): stage-1 q8 still 37% of prefill; gemm_fp16+dequant ~68 ms; dn_recurrence 47 ms.

**P1 COMPLETE (2026-06-10)**: product surface shipped — Engine API, OpenAI-compatible server `ie serve` with tools support, and the `ie` CLI; Seal runs on the engine via openai-compat with real tool calls verified (commits 4f4339c..6fcc613). Closing gate battery all green: fresh build 0 errors; ctest 12/12; PPL 6.52 (gate ≤ 6.57, both runs); crown pp512 1170.45 / tg 83.33 (bands ≥ 1080 / ≥ 79); `server_smoke: all OK`; 1h soak via `scripts/soak_1h.sh` — 5402 completions in 3600 s, zero failures, RSS 21,040,632 → 21,040,932 kB (+0.0014%, no leak).

**✅ P3a Llama-3.1-8B COMPLETE (2026-06-12)**: Llama-3.x runs on the DenseModel
path (`general.architecture=llama` → `ModelArch::kLlama3`, routed via `is_dense_arch`).
Deltas: load-time Q/K inverse row-permute (`llama_qk_unpermute_rows` + `upload_quant_dense_permuted`,
inverts convert_hf_to_gguf NORM-pairing so our NEOX rope is right), `rope_partial_ff`
(rope_freqs scaling; null→bit-identical to rope_partial), llama-bpe tokenizer
(digit triplets + ignore_merges, keyed on `tokenizer.ggml.pre`), `build_llama3_prompt`
+ per-arch `stop_ids_`, add_bos default-true for llama-bpe (bartowski 8B omits the KV).
QK-norm was already presence-gated. **VALIDATED:** per-layer cosine **0.99998–1.000000
all 32 layers** vs `ie-llama-dump` (un-permute/rope_ff correct — would blow up at L01);
coherent gen; greedy near-tie (' a'/' Paris' within 0.1 logit, fp16 precedent); **PPL
10.79** deterministic. qwen3 parity ALL GREEN + crown PPL 6.45 = zero regression from
the shared-file edits. Gate `scripts/p3_parity_llama3.sh`. New host test
`dense_unpermute_test`. Commits a60f967→fa9fe0e.

**P3e AWQ+GPTQ IMPORT COMPLETE (2026-06-11)**: `ie import` loads **AWQ and GPTQ**
4-bit safetensors directly (formats llama.cpp can't) → native GGUF the loader runs,
across **Qwen3 AND Qwen2.5 dense**. Pipeline in `src/loaders/`: `safetensors_reader`
(+ sharded `SafetensorsModel`), `awq.{cpp,hpp}` (awq+gptq dequant), `gguf_writer`,
`quantize_q4k`+`quantize_q6k` (ggml-faithful fp32→Qx_K encoders), `hf_import`
(driver, dtype-aware). Qwen2 added: bias + QK-norm gated on tensor presence in
`dense_transformer.cpp` (Qwen3 byte-identical, verified PPL 2.937037 EXACT); new
`add_bias` op (`elementwise.cpp`); `read_dense_config_auto`. Verified on real
Qwen3-4B AWQ/GPTQ + Qwen2.5-3B/7B AWQ (7B 2-shard, PPL 15.48 + correct codegen).
Detail `docs/p3e_awq_import_validation.md`, memory `project_awq_gptq_import.md`.

**✅ P3d Qwen3.6-27B COMPLETE (2026-06-12)**: forward validated to per-layer cosine.
`tools/qwen35_dump.cpp` (→`ie-qwen35-dump`, engine side) + rebuilt `ie-llama-dump`
(oracle, vs llama master `fdc3db9b6`, build-vk libs) + `tools/diff_layers.sh` over
"The capital of France is": **cosine ≥0.9995 on all 64 layers** (embedding 1.000000;
min L53 0.999543), **exact greedy argmax id=11751 ' Paris'** (16.03 vs oracle 16.15),
top-5 identical. The `rel_fro ~0.02` (vs dense path's 0.0014) = the **fp16 residual
stream** (we keep fp16, llama fp32) × 64 layers + DeltaNet/M-RoPE — PROVEN inherent,
not a defect: `IE_QWEN35_NO_ONEDNN=1` (bit-exact gemm_fp16) yields the SAME cosine
(oneDNN-invariant), and exact argmax + PPL 5.34 confirm correct output; worst always on
massive-activation dim 3994, smooth depth-accumulation. Crown PPL 6.45 held. Run dumps
SERIALLY (one model on GPU). Build history below.
**P3d Qwen3.6-27B (history, 2026-06-11)**: `qwen35` dense-hybrid (crown's
gated-DeltaNet + full-attn MINUS MoE + dense MLP; 64 transformer +1 NextN/MTP,
hidden 5120, ffn 17408, head_dim 256, ssm inner 6144 / 48 v / 16 k / state 128,
full_attn_interval 4, vocab 248320). **DONE:** arch detect + `read_qwen35_config`
(`55039da`), tensor layout (`docs/qwen35_27b_gguf_layout.md`), tokenizer-by-reuse,
+ **`dequant_q5_K_to_Bt`/`dequant_q8_0_to_Bt`** GPU kernels (the 27B has Q5_K
attn_k/output + Q8_0 ssm_out; `tests/unit/dequant_bt_test.cpp`). **✅ ORACLE RESOLVED +
Task 2.5 DONE (cycle-2):** old llama.cpp HEAD `dcad77cc3` just predated the MTP/NextN
support series; updated to master `fdc3db9b6` (+665 commits), CPU-rebuilt at
`~/llama.cpp/build-cpu/` → loads the 27B cleanly (`blk.64` dense via
`LLM_KV_NEXTN_PREDICT_LAYERS`), generates coherent text. Bit-exact oracle =
`llama-perplexity` + `llama-eval-callback` (NOT `llama-cli`, now interactive-only).
Dumped per-op tensors → **verified the entire qwen35 dataflow vs ground truth**:
`docs/qwen35_27b_oracle_dataflow.md` (conv_channels=10240 NOT SI2=SI*2; 16k/48v×128;
full-attn `attn_q{5120,12288}`=joint Q\|gate; FFN residual on pre-post-norm). **Op
inventory CONFIRMED — Task 3 needs zero new kernels.** **REMAINING:** `Qwen35DenseModel`
loader + the ~600-line unfused DeltaNet/gated-attn forward (re-derive from leaf ops,
conv_channels=10240, FORBID `dn_qkv_split_norm_fused`) + GPU parity. Fits one B70.
Plan `docs/superpowers/plans/2026-06-11-p3d-*.md`. P3b PAUSED.

**✅ TENSOR-PARALLEL DECODE (TP-2/TP-3) DONE (2026-06-12)**: `DenseModelTP` (new
`include/ie/dense_tp.hpp` + `src/model/dense_tp.cpp`; additive — crown/dense/split/27B untouched).
Megatron column(q/k/v,gate-up)-then-row(o-proj,ffn-down) per-layer split across both B70s, 2
host-bounce all-reduces/layer, concurrent in-order queues. Re-pack helpers: `up_col` (Q4_K/Q6_K
contiguous row-slice + F16 strided transpose-slice), `up_row` (256-superblock Q4_K/Q6_K K-slice),
`up_bias`. Validated `ie-multi-gpu-tp-test` qwen3-8b: prefill cosine 0.999999, argmax match, greedy
16/16 identical, coherent. **72B decode 10.4 tok/s TP vs 7.2 layer-split = 1.44×** (`ie-multi-gpu-run
--tp`). Crown PPL 6.45 bit-exact (regression green). NOT bit-exact by design. Next ~2×: Level-Zero
P2P all-reduce. See MASTER_DEV_PLAN §5.7.

**✅ 72B PPL VALIDATED (2026-06-12)**: Qwen2.5-72B-Instruct-AWQ → streamed-import GGUF →
**engine PPL 8.97** (avg NLL 2.193958, 511 tokens, full causal context, fp16 KV) via the
layer-split path across 2×B70, new tool `ie-multi-gpu-ppl` (additive; mirrors `ie-perplexity`
streaming-T=1 NLL + built-in corpus over `DenseModelSplit::forward`). Correctness via
cross-model ordering: same-tokenizer Qwen3-8B NLL 2.94 (PPL 18.9) >> 72B NLL 2.19 — bigger
model decisively better, as required; gross import mismap would be PPL >50. ⚠ llama same-file
anchor BLOCKED: imported GGUF declares `vocab_size=151936` but `token_embd` is [8192,152064]
(Qwen2.5 +128 pad) → llama `check_tensor_dims` rejects; our engine handles it (PPL unaffected,
pad rows unused). Writer fix = declare vocab_size = embedding rows. See MASTER_DEV_PLAN §5.7.

**P2 COMPLETE (2026-06-10 night)**: second architecture — **Qwen3-dense** (`qwen3` GGUF arch, e.g. Qwen3-8B Q4_K_M). Dense path internals: `src/model/dense_transformer.cpp` + `include/ie/dense_transformer.hpp` (DenseModel forward — QK-norm, partial rope, GQA dims from metadata) with dispatch helpers in `src/model/dense_dispatch.hpp` — **zero edits to `qwen36.cpp`**; arch detection keys loader/tokenizer/template/Engine. Validated bit-exact vs llama.cpp after the `rope_partial` in-place race fix (`34e4c01`, bisect: `docs/dense_nondeterminism_2026-06-10.md`): layer parity cosine 1.000000, greedy 64/64, PPL avg-NLL exact-equality gate (constants + the invocation-binding caveat live in `docs/ppl_baseline_matrix.md`). Per-arch gate: `scripts/p2_parity_qwen3.sh`. Dense baseline ledgered (`docs/benchmark_matrix_2026-06-09.md` §P2): pp512 1190 ± 5 vs llama.cpp SYCL same-hour 1036 ± 4 (**+14.9%**); tg 43.7 vs their 77.7 (dense decode unoptimized — crown's int-dot Q8 decode GEMVs not ported to dense shapes; out of P2 scope).

---

## 1. Project at a glance

From-scratch C++ inference engine for **Qwen3.6-35B-A3B-Q4_K_M** on **Intel Arc Pro B70 (BMG-G31, Xe2-HPG)** using SYCL/Level Zero + IGC.

- **Targets**: ≥50 tok/s decode ✓ (66.2 at full quality, 167% of llama.cpp Vulkan), ≥800 tok/s pp512 prefill ✓ (938.6, 106% of llama.cpp Vulkan — **both targets EXCEEDED 2026-06-09**)
- **Architecture**: hybrid MoE — 40 layers (30 DeltaNet linear-attention + 10 full-attention), 256 experts top-2 + 1 shared, head_dim=256, n_q_heads=16, n_kv_heads=2 (GQA=8)
- **Quantization**: Q4_K weights (144 B/block), Q6_K some weights (210 B/block), fp16 activations, optional INT8 KV cache
- **Toolchain**: IntelLLVM 2026.0.0, NEO 26.14.37833.4, IGC 2.32.7, GuC 70.60.0
- **Model file**: `/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf`

### Performance baseline (verified head-to-head 2026-06-09)

Same GGUF, same B70, minutes apart. llama.cpp Vulkan b8902 via `llama-bench -ngl 99 -sm none -mg 0` (pinning to B70 made no difference vs default). See `docs/perf_baseline_2026-06-09.md`.

| metric | engine | llama.cpp Vulkan b8902 | gap |
|---|---:|---:|---:|
| **decode tg128 (turbo GGUF)** | **84.1** @ PPL 6.61 | SYCL master 81.31 ± 0.21 (same-hour) | **👑 +3.5% CROWN** |
| **decode tg128 (default)** | **81.0** @ PPL 6.52 (project-best) | SYCL master 81.31 | tie at better quality |
| **pp512 prefill tok/s** | **1144 ± 5 (defaults)** | SYCL master 1064 ± 8 (same-hour) / Vulkan-any 775-885 | **👑 +7.6% CROWN** |

**E1–E5 landed 2026-06-09 (+343% pp512 in one day, PPL 6.54 held at every
step, decode unchanged). THE ENGINE NOW BEATS LLAMA.CPP VULKAN ON BOTH
METRICS: decode 46.8 vs 39.75 (118%), prefill 899.7 vs 885.0 (101.6%).**
- E1: prefill projections at T≥64 dequant Q4_K/Q6_K to fp16 scratch + dense
  `gemm_fp16` (kill switch `IE_NO_DEQ_FP16=1`). 202.9 → 309.4.
- E2: vectorized weight + SLM loads in the three MoE prefill kernels
  (`moe_fused.cpp`), bit-identical math. 309.4 → 589.7.
- E2b: scalar `gemm_q4_K` in-kernel M-tile grid + vectorization. 589.7 → 754.5.
- E2c: SLM-staged Q6_K weight slabs in down_packed_q6k_v2. 754.5 → 840.7.
- E5: pre-gathered expert-sorted x_packed for stage-1 (kills 24× redundant
  scattered activation reads). 840.7 → 899.7.
- Tried + reverted: E3 (BK=64 gemm_fp16 retile, −21%), E4 (header vector
  loads, neutral — proved stage-1 is compute-bound now).
Full story: `docs/prefill_attack_plan_2026-06-09.md`. Future headroom (not
needed for the win): fp16-product math or XMX retry in gate_up, gemm_fp16
double-buffer at BK=16, E2c-style slab staging for the Q4_K down kernel.
| built-in 511-tok PPL | 6.54 baseline (±0.03 floor) | 10.37 (different methodology) | gate: every perf change must hold ≤6.57 |

NOTE: the "~50 tok/s parity" figures in older docs referenced llama.cpp **SYCL** backend reports, not a measured Vulkan run. Measured Vulkan TG is 39.6 — the engine wins decode outright.

See `docs/perf_optimizations_2026-05-05.md` for the four-change set:
1. XMX M_TILE=16 in `gemm_q4_K_xmx` + `gemm_q6_K_xmx` (+26 % — the big win)
2. Shared-expert prefill batching (T>1 path) (+1.7 %)
3. Scalar `moe_prefill_gate_up_silu_q4k` M_TILE 8→16 (+1.8 %)
4. Scalar `gemm_q4_K` M_TILE 8→32 (+1.5 %)

Combined: **+33.2 %** prefill tok/s, decode and PPL preserved. Save state
at `docs/save_states/2026-05-05_baseline/`.

### Qwen3.6-27B (`qwen35` dense-hybrid) perf ledger (2026-06-11, P3d)

Second perf-validated model. `Qwen35DenseModel` (new files; crown untouched). Same
B70, same GGUF, order-controlled vs **llama.cpp Vulkan** (build-vk from master
`fdc3db9b6` + 1-line `eMesaHoneykrisp` patch — it HAS the GPU DeltaNet shaders).

| metric | engine | llama.cpp Vulkan | gap |
|---|---:|---:|---:|
| **pp512 prefill** | **577** (was 186) | 303 ± 5 | **👑 1.9× FASTER** |
| **tg128 decode** | 10.0 | 9.72 | parity |
| 512-tok PPL (builtin) | 5.34 (NLL 1.675056) | 8.38 (llama `-c 256`, methodology differs) | oracle-consistent |

Three stacked prefill opts (profiling-driven via `ie-bench --kprofile`; PPL
bit-identical 5.3391 at every step; all crown-safe via the **qwen35-only
`dense::prefer_onednn()`** flag, set in `Qwen35DenseModel::load`):
1. Batch `ssm_alpha/beta` (N=48 → pad 64) — killed 49k serial gemv launches
   (38% of prefill). 186 → 296.
2. oneDNN on the Q4_K/Q6_K dequant→gemm path. 296 → 521.
3. oneDNN on the F16 branch too (attn_k/v/o, ssm_out, alpha/beta). 521 → 577.
- A/B'd + REJECTED: fused-XMX prefill (39.8 — restreams quant blocks at big-K);
  int-dot q8 decode (gemv_q4_K_q8 slower than scalar at the big-N FFN).
- Remaining prefill cost is the SHARED dequant kernel (69% of captured) + the
  DeltaNet scan (15%) — both off-limits for quick wins (crown-shared / correctness).
- Decode is Q6_K-GEMV-bound (gemm_q6k ffn_down K=17408 fallback + attn) — the
  known-hard P3b dead-end; left at parity.

### Qwen3-Coder-30B-A3B (`qwen3moe`) fused-MoE perf ledger (2026-06-12, P3 breadth)

Sixth arch family, FUSED MoE landed. `Qwen3MoeModel` calls the crown's
runtime-parameterized `moe_fused.cpp` ops (gather→gate_up_silu→down→reduce
prefill chain + 2-launch decode pair) from `qwen3moe.cpp` only — **ZERO edits to
`qwen36.cpp`/`moe_fused.cpp`**; the unfused per-expert loop is the parity oracle
behind `IE_QWEN3MOE_UNFUSED=1`. Same B70, `~/models/Qwen3-Coder-30B-GGUF/*Q4_K_M.gguf`
(E=128, K=8, E_ffn=768, no shared; gate/up Q4_K, down Q6_K). Order-controlled
new-old-new vs **llama.cpp Vulkan** build-vk (master `fdc3db9b6`). `ie-bench` taught
the arch (additive branch). Full table: `docs/benchmark_matrix_2026-06-09.md` §qwen3moe.

| metric | engine fused | engine unfused | llama.cpp Vulkan | gap |
|---|---:|---:|---:|---:|
| pp512 prefill | ~61 | ~58 | ~984 | 0.062× (llama 16×) |
| tg128 decode  | ~19.3 | ~18.4 | ~58.6 | 0.33× (llama 3×) |
| 512-tok PPL   | 11.9981 | 11.9856 | oracle (windowed 20.19) | Δ 0.013, oracle-consistent |

**HONEST: NOT a "beats llama" entry** (crown stays the only such headline). The
fused MoE is *correct, parity-proven, publishable* — but it is **not** the qwen3moe
perf lever. `ie-bench --kprofile-decode` (one warm T=1 step, 37.1 ms): `attn_naive_compute`
= **71.3%**; fused MoE decode kernels only **~13%**. qwen3moe is **naive-attention-bound**
(`ensure_attn_partials` is a no-op for this arch — no FA-2/split-K yet). Prefill is also
RAM-swap-bound on the 32 GB box + serialized per-layer host routing. **Next levers (ranked):**
(1) FA-2/split-K for the qwen3moe dense-attention decode (the 71% lever); (2) oneDNN
prefill GEMM + de-serialize host routing; (3) qwen3moe multi-GPU → Qwen3-Next-80B.

### Phi-4 (`phi3` → `kLlama3` dense) breadth gate (2026-06-12, Wave-1 Gate 3/9)

`phi3` family unblocked for the **Phi-4** tier via three additive levers — no new
kernels, no perf crown (a correctness gate). The reusable win is **Q5_K/Q8_0 dense
GEMV** (`dense_dispatch.hpp` `upload_quant_dense_auto` → existing `dequant_q5_K_to_Bt`),
which unblocks ANY Q5_K dense GGUF. Plus a **fused-tensor splitter at load**
(`attn_qkv`→Q/K/V, `ffn_up`→gate/up row-span slices, NEOX natural order, Q5_K Q/K via
dequant) and the **`pre=dbrx`** tokenizer flag (folds into llama-bpe: digits_1to3 +
ignore_merges; host encode-parity test `tests/unit/dbrx_pretok_test.cpp`). Gate
results: per-layer cosine ≥ 0.999999 (L01..L41) vs `ie-llama-dump`; greedy=` Paris`;
**PPL 8.2475 / NLL 2.109913** (deterministic ×2); ChatML chat coherent + stops at
`<|im_end|>` (engine kChatML stop set, Qwen3-dense excluded). Crown 6.45 / NLL
1.864495 + dense NLL 2.940491 bit-exact after every shared-file edit. Commits
`aaab17b`/`aa37fc9`/`27eaa24`/`e917e6d`. **NEXT (phi3):** Phi-3.5-mini needs LongRoPE
(`rope_factors_long`/`_short`) + relaxing the FA-2 hd power-of-two gate for head_dim 96.

✅ **27B chat bug — RESOLVED as a MISDIAGNOSIS (2026-06-12).** The alleged temp-0.7 chat
garbage does NOT reproduce (114 clean gens: thinking ON/OFF, 100+ seeds, varied prompts).
Tokens **248068/248069 = `<think>`/`</think>`** (USER_DEFINED, verified in the GGUF
token_type) — the model's legitimate reasoning markers, NOT garbage; they decode to ''
under skip-special, so the raw-generate's `<think>` argmax merely *looked* like an invisible
special-token bug. The "order-dependent state leak" is refuted: `forward_step` is
temperature-independent (so "greedy works" + "sampler picks rank-0 garbage" was internally
contradictory), `DeltaNetState::reset()` zeros state+conv, the production `deltanet_recurrence`
holds no static/persistent scratch (statics are only under default-OFF `IE_DN_RECURRENCE_REWRITE`),
and `kv_.reset()`+`start_pos=0` make stale KV unreadable. The 27B chats correctly at default
temp 0.7 (`ie run` + `ie serve`); engine.cpp UNCHANGED — no code fix warranted, and the
"obvious" special-token suppression would have BROKEN `<think>`/`</think>`. Regression harness:
`tools/qwen35_load_test.cpp <gguf> <prompt> <base_seed> <n_seeds>` sweep mode. See MASTER_DEV_PLAN §7.

---

## 2. The deltanet recurrence non-determinism — closed

**Status**: 28-step bisect complete. Bug is HW-level FMA pipeline non-determinism on Xe2 BMG-G31, software-irreducible from SYCL.

**Reading**:
- `docs/known_bugs.md` — what the bug is, what it isn't, how to reproduce, what NOT to do.
- `docs/bisect_step25_26_summary.md` — full bisect summary.

**Production workaround**: chunk prefill externally at T≤256. Bug doesn't fire on short prefill or single-token streaming-decode. Built-in 511-tok PPL = 6.55.

**Reproduction tools**:
- `./build/tools/ie-bug-monitor` — 8-chain live dashboard with sparklines + per-buffer divergence localization.
- `./build/tools/ie-bug-live` — simple A vs B streaming demo with first-divergence callout.

---

## 3. Source state after cleanup (2026-05-04)

Removed: 14 default-OFF bisect-step macros and their dead code paths (Steps 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28 + supporting helpers and the failed Step 19 GR variant). Removed `state_fp16_quantize_dequantize_inplace` (Step 26). Removed `clear/dump_dn_debug_buf` and `clear/dump_dn_upstream_trace_buf` (Steps 27/28). Removed `IE_FA_FORWARD_CHUNK_T` (Step 10) and `IE_FA_ZERO_ATTN_OUT` (Step 8) from `qwen36.cpp`.

Line-count delta:
| file | before | after | Δ |
|---|---:|---:|---:|
| `src/ops/deltanet.cpp` | 859 | 359 | **−500 (−58%)** |
| `src/model/qwen36.cpp` | 1594 | 1309 | **−285 (−18%)** |
| `src/core/deltanet_state.cpp` | 97 | 71 | −26 |
| `tools/ie_validate_chunking.cpp` | 1276 | 1243 | −33 (Step 27/28 compare blocks gated under `#if 0`) |
| **total** | **3826** | **2982** | **−844** |

Kept (still load-bearing or dormant-but-validated):
- `IE_DN_RECURRENCE_REWRITE` — clean alternative kernel (DMA-based double buffer); default OFF. Validated to preserve baseline math; does NOT fix the bug; kept as a structurally-clean alternative.
- `IE_VALIDATE_C_ONLY` — validator mode that runs only Path C1+C2 (skips A/B/A2). Useful for fast determinism-only runs.
- `IE_ENABLE_GEMM_Q4K_ESIMD` (Step 1) and `IE_ENABLE_FA2_PREFILL_TILED` (Step 2) — dormant perf scaffolds. Default OFF. **Do not touch per repo rules.**
- `IE_ENABLE_MOE_DOWN_TILE = 1` — production-validated. Stays on.
- `IE_FUSE_SSM_AB`, `IE_FUSE_RES_RMS` — production fused-op flags.

PPL after cleanup: 6.54 (= 6.55 historic ± 0.03 noise floor). No regression.

---

## 4. Tools available

| Tool | Purpose |
|---|---|
| `./build/tools/ie-perplexity` | Engine PPL on built-in 511-tok or `--text <file>`. Smoke test for math correctness. |
| `./build/tools/ie-validate-chunking` | Five-path validator (A/B/A2/C1/C2) with hash-checkpoint determinism check. |
| `./build/tools/ie-bug-live` | Simple two-chain (A,B) streaming demo. Streams iter-by-iter status, prints colored alert on first divergence. |
| `./build/tools/ie-bug-monitor` | 8-chain live dashboard. Per-iter persistent-state hashing per chain, per-layer divergence localization, sparkline graphs for latency + every top kernel, memory tracker, auto-focus recommendations. |
| `./build/tools/ie-bench` | Per-kernel microbenchmarks. |
| `./build/tools/ie-bench-suite` | Wider perf suite. |
| `./build/tools/ie-attn-profile` | Attention-specific profiling. |
| `./build/tools/ie-kernel-monitor` | Per-kernel timing monitor. |
| `./build/tools/ie-kv-scale` | KV cache scaling tests. |

External anchor: `/home/weezy/llama.cpp-vulkan/llama-b8902/llama-perplexity` for cross-engine PPL anchor (different methodology — see `docs/ppl_baseline_matrix.md`).

---

## 5. Open work (in priority order)

### #0: DONE 2026-06-09 — both perf targets exceeded (see RELEASE.md)
Prefill 202.9 → 900.0 tok/s (E1–E5) and decode 46.8 → 52.4 (staged Q6_K
GEMV) in one session. The items below are v1.2+ headroom, not gaps:
- gate_up stage-1 is compute-bound: fp16-product math or XMX retry.
- gemm_fp16 double-buffer at BK=16 (E3 showed wider BK loses to occupancy).
- E2c-style SLM slab staging for the Q4_K down kernel.

### #1 (was: close the prefill perf gap — CLOSED 2026-06-09)
Constraints that still apply to any future kernel work:
- Must NOT touch ESIMD / block2d / `Subgroup2DBlockLoad` / `SubgroupBlockRead` / `gemm_q4k_esimd` / tile-load smoke tests.
- Two existing dormant scaffolds (`IE_ENABLE_GEMM_Q4K_ESIMD`, `IE_ENABLE_FA2_PREFILL_TILED`) regressed prefill in the past and remain default-OFF.

Investigation paths (none exercised yet):
- Confirm `-O3 -DNDEBUG` reaches all per-target compilation. Per `research/...`, missing `-DNDEBUG` alone costs 50–180% prefill on llama.cpp's path.
- F16 accumulation in fp16 reduction paths.
- Tile-size re-tuning for Xe2's 256 KB unified L1+SLM.
- Persistent zero-gap MoE kernel with atomic counters for dynamic expert load balancing (Intel vLLM blog reports ~15% gain).
- Profile a `forward(T=512)` call under `unitrace --device-timing` and rank kernels by total time. The dashboards show per-iter decode kernels; prefill kernels (gemm_q4_K_xmx, moe_prefill_*) need their own profile.

### #2: Vendor escalation on the determinism bug
File a minimal repro against IGC at https://github.com/intel/intel-graphics-compiler/issues. Builds from `tools/ie_bug_live.cpp`. ETA 4–8 h to prepare.

### #3: Long-context MoE optimizations
The shared-expert path and routing decisions could likely be fused further; current decode profile shows MoE kernels collectively at ~25–30% of compute.

---

## 6. Quick path to current state

```bash
cd "/home/weezy/00 - Inference Engine/build"
cmake --build . -j                       # builds all targets clean
./build/tools/ie-perplexity              # → PPL 6.54 ± 0.03 baseline
./build/tools/ie-bug-monitor --max-iters 100   # 8-chain dashboard
                                          # → bug typically fires within
                                          # the first 100 iters
```

`CMAKE_CXX_FLAGS` is empty. No diagnostic macros are enabled. The engine runs at production-default flags.

---

## 7. Auto-memory pointers

`/home/weezy/.claude/projects/-home-weezy-00---Inference-Engine/memory/MEMORY.md`:

- `feedback_research_first.md` — user wants exhaustive pre-work before coding
- `feedback_plan_md_source_of_truth.md` — keep PLAN.md current
- `project_inference_engine.md` — overall pointer
- `project_qwen36_quirks.md` — 12 silent-bug arch details
- `project_qwen36_gguf_layout.md` — verified GGUF tensor layout
- `project_v2_phase1_freeze_diagnosis.md` — IGC BMG page-fault (closed; use 1D block reads)
- `project_dn_recurrence_bug_dead_end.md` — the 28-step bisect dead end (this file's subject)
- `reference_research_files.md` — research/ pointers
