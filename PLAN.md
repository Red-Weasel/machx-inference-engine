Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


# Native LLM Inference Engine — Master Plan

> **AUTHORITY MOVED (2026-06-10):** the single source of truth for the
> build is now **`MASTER_DEV_PLAN.md`** at the repo root. This file
> remains the historical/technical plan for the engine's kernel phases
> and research index — keep using it for that detail, but state, roadmap,
> and decisions live in MASTER_DEV_PLAN.md.

**Target hardware:** Intel Arc Pro B70 (BMG-G31, Xe2-HPG, 32 GB GDDR6, 608 GB/s, 367 INT8 TOPS, ~183 FP16 TFLOPS, $949).
**Daily-driver model:** `Qwen/Qwen3.6-35B-A3B` (released 2026-04-16, hybrid Mamba+Attention MoE, 36B total / 3B active).
**Performance target (v1, revised 2026-04-26 after consultant review):**
- TG ≥ 95% of llama.cpp Vulkan on the same hardware/model, all measured ctx (1, 1k, 4k, 16k, 32k). **✅ EXCEEDED** — verified head-to-head 2026-06-09: engine 46.8 tok/s vs llama.cpp Vulkan b8902 39.6 tg128 = **118%** (see `docs/perf_baseline_2026-06-09.md`).
- PP ≥ 50% of llama.cpp Vulkan @ T=512.  **✅ EXCEEDED 2026-06-09 at 101.6%** — engine 899.7 tok/s vs llama.cpp Vulkan b8902 885.0 ± 5.5, same-hour head-to-head (was 202.9 = 23% that morning).  **The engine now BEATS llama.cpp on BOTH decode (118%) and prefill (102%).**  Won via E1 dequant-to-fp16 prefill GEMM + E2/E2b/E2c vectorized & SLM-staged MoE kernels + E5 pre-gathered expert-sorted activations — NOT via ESIMD/2D-block-I/O (still HW-blocked).  See `docs/prefill_attack_plan_2026-06-09.md`.  **All v1 hard gates now pass.**
- Long-ctx (32k) functional and correct.  **✅ ACHIEVED.**

**2026-06-10 — TOTAL CROWN: the engine beats llama.cpp's BEST backend (SYCL master) on both metrics, same-hour head-to-head.**
- **Prefill pp512: 1144 ± 5 vs SYCL master 1064 ± 8 = +7.6%** (alternating runs 1147/1064/1139/1063/1147). Won via load-time per-expert SoA weight repack (`ie/quant_soa.hpp`) + int-dot MoE prefill (default ON; `IE_NO_MOE_Q8=1` opts out): full-K register lattices (one q8 activation block per lane, weights nib-masked into registers once per column) with exact K-quant corrections through precomputed q8 block half-sums (s0/s1, `block_q8_1s`) — zero isum dp4a. See `docs/prefill_crown_plan.md` (EXECUTED).
- **Decode tg128: 84.1 turbo / 81.0 default vs SYCL master 81.31** (2026-06-10) — held through the SoA repack (guard A/B 81.25/80.72/81.84).
- PPL gate 6.52 (≤ 6.57) at production defaults; all unit tests green.

**Original absolute targets (deferred to v1.5):** ≥50 / ≥100 / ≥150 tok/s decode @ 32k.  Consultant review (2026-04-26) flagged that hitting 50 tok/s @ 32k means BEATING llama.cpp Vulkan by ~30% on the same hardware — a research target, not engineering.  Reframed v1 around relative-to-best-open-source parity; absolute numbers remain milestones for v1.5.

Research backing this plan lives in `research/{01_hardware,02_programming_stack,03_quant_formats,04_qwen_arch}.md`. Re-read those before each phase — they hold the byte-level facts this document only summarizes.

---

## 1. Why this is achievable (roofline)

Decode is bandwidth-bound. Per decode token at Q4_K_M, single user, no batching:

| Component | Bytes streamed/token |
|---|---|
| Active weights (3B at ~4.85 bits) | **~1.84 GB** |
| KV cache reads, 10 full-attn layers × 10 KiB/token × T | 40 MB @ 4k, 320 MB @ 32k, 1.28 GB @ 128k |
| DeltaNet recurrent state (30 layers, fixed 60 MB total, write-back only) | ~negligible per token |

Effective B70 sustained BW assumption: **70–80% of 608 GB/s ≈ 425–486 GB/s** (XeTLA / sycl-tla benchmarks land here on Xe2 with 2D block load).

| Context | Per-token bytes | Ceiling @ 80% BW | Ceiling @ 70% BW |
|---|---|---|---|
| 4k    | ~1.88 GB | **258 tok/s** | 226 tok/s |
| 32k   | ~2.16 GB | **225 tok/s** | 197 tok/s |
| 128k  | ~3.12 GB | **156 tok/s** | 137 tok/s |

**50 tok/s = 22% of the 32k ceiling.** Even a moderately tuned implementation will clear it. The model's hybrid arch (only 10 of 40 layers carry KV) is what makes long-context viable on this card.

Prefill is compute-bound and dominated by 8 active expert FFNs + 10 full-attention QKV/Out projections. At ~183 FP16 TFLOPS, prefill of 4k tokens in <1 s is realistic (~25% of peak).

---

## 2. Tech-stack decisions (locked)

| Concern | Decision | Rationale |
|---|---|---|
| Host language | **C++20** | RAII, std::span, concepts; no runtime deps |
| Device language | **SYCL (DPC++) + ESIMD escape hatch** | `joint_matrix` is the cleanest XMX path; ESIMD `xmx::dpas` for hot inner loops |
| Toolchain | **upstream `intel/llvm` nightly** (not gated oneAPI release) | newest BMG fixes; sycl-tla v0.8 already validates BMG-G31 |
| AOT target flag | `-fsycl-targets=intel_gpu_bmg_g31` | confirmed in sycl-tla examples |
| Driver baseline | compute-runtime ≥ 26.05.37020.3, IGC ≥ 2.28.4 (current stable: 26.14.37833.4 / IGC 2.32.7) | needed for working BMG-G31 codegen |
| Runtime layer | **Level Zero** (under SYCL) | lowest launch overhead; multi-queue scheduling later |
| Production GEMM/SDPA | **oneDNN 3.8** as escape hatch (**not** a Q4_K_M drop-in — requires repack to single-level groupwise INT4; only viable if accepting format change + PPL regression; see item 13 option (b)) | shipping Xe2 paths + INT4/INT8 weight decompression |
| Reference codebase to mine | **`intel/sycl-tla`** v0.8 BMG examples, **oneDNN** matmul, **llama.cpp `ggml-quants.c`** for dequant | avoid `ggml-sycl` — broken on B70 (issues #21893, #21517) |
| Build system | **CMake** + Ninja | aligns with sycl-tla / oneDNN |
| Tokenizer | port `llama.cpp/src/llama-vocab.cpp` (MIT) | proven Qwen2/Qwen3 BPE + Unicode regex |
| Server (later) | C++ HTTP via cpp-httplib, OpenAI-compatible `/v1/chat/completions` | minimal deps |
| Testing | GoogleTest unit + Python diff-against-`transformers` for model fidelity | Python only as a test-time dep |

**Explicit non-goals for v1:** multi-GPU, batching for many users, training, vision encoder, speculative MTP head, FP8 weights. All carried as phase-2 backlog.

---

## 3. Architecture (components)

```
┌──────────────────────────────────────────────────────────────────┐
│  CLI / HTTP server (later)                                        │
├──────────────────────────────────────────────────────────────────┤
│  Engine API:  load_model() · encode() · prefill() · decode_step()│
├──────────────────────────────────────────────────────────────────┤
│  Sampler   │  Tokenizer (BPE) │ Chat-template (ChatML, <think>)  │
├──────────────────────────────────────────────────────────────────┤
│  Model graph (Qwen3.5-MoE arch, hybrid layer pattern)            │
│    · Embedding  · 40× Block  · FinalNorm  · LMHead               │
│    Block: layer_idx % 4 == 3 ? FullAttn : GatedDeltaNet          │
├──────────────────────────────────────────────────────────────────┤
│  Op layer (kernel registry, dtype/quant dispatch)                │
│    GEMM · GEMV-W4A16 · FlashAttn-2 · DeltaNetScan · MoEDispatch  │
│    RMSNorm · RoPE-partial · SwiGLU · Sampling                    │
├──────────────────────────────────────────────────────────────────┤
│  Tensor & memory:                                                │
│    Tensor (shape/stride/dtype/quant) · Allocator (USM device)    │
│    KV cache (10 layers) · DeltaNet state cache (30 layers)       │
├──────────────────────────────────────────────────────────────────┤
│  Loaders: GGUF (mmap), safetensors (AWQ/GPTQ/AutoRound)          │
├──────────────────────────────────────────────────────────────────┤
│  Backend abstraction (single impl for v1: Xe2/SYCL)              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Directory layout

```
inference-engine/
├── CMakeLists.txt
├── PLAN.md                          (this file)
├── README.md
├── research/                        (committed; re-read per phase)
│   ├── 01_hardware.md
│   ├── 02_programming_stack.md
│   ├── 03_quant_formats.md
│   └── 04_qwen_arch.md
├── third_party/                     (vendored; pinned commits)
│   ├── llama-vocab/                 (BPE port, MIT)
│   ├── cpp-httplib/                 (server, MIT)
│   └── googletest/
├── include/ie/                      (public headers)
│   ├── engine.hpp                   (load/prefill/decode)
│   ├── tensor.hpp
│   ├── dtype.hpp                    (kFP16/kBF16/kQ4_K/kQ5_K/kQ6_K/kQ8_0/kAWQ4/kGPTQ4/kAR4/kAR8)
│   ├── tokenizer.hpp
│   └── sampler.hpp
├── src/
│   ├── core/
│   │   ├── tensor.cpp
│   │   ├── allocator.cpp            (Level Zero / SYCL USM)
│   │   ├── kv_cache.cpp
│   │   └── deltanet_state.cpp
│   ├── loaders/
│   │   ├── gguf_reader.cpp          (mmap-based)
│   │   ├── safetensors_reader.cpp
│   │   ├── awq_loader.cpp
│   │   ├── gptq_loader.cpp
│   │   └── autoround_loader.cpp
│   ├── tokenizer/
│   │   ├── bpe.cpp
│   │   ├── unicode.cpp
│   │   └── chat_template.cpp        (ChatML + <think>)
│   ├── ops/
│   │   ├── gemm_fp16.cpp            (joint_matrix + 2D block load)
│   │   ├── gemv_w4a16.cpp           (Q4_K, AWQ, GPTQ, AutoRound)
│   │   ├── gemv_w5a16.cpp           (Q5_K)
│   │   ├── gemv_w6a16.cpp           (Q6_K)
│   │   ├── gemv_w8a16.cpp           (Q8_0, AutoRound INT8)
│   │   ├── flash_attn.cpp           (FA-2, partial-rotary, QK-Norm, output gate)
│   │   ├── deltanet_scan.cpp        (chunkwise prefill + recurrent decode)
│   │   ├── moe_dispatch.cpp         (top-8 router + sparse expert exec)
│   │   ├── rmsnorm.cpp
│   │   ├── rope.cpp                 (partial 0.25, 10M base, M-RoPE)
│   │   ├── swiglu.cpp
│   │   └── sampling.cpp
│   ├── kernels/sycl/                (.cpp.sycl, AOT for bmg_g31)
│   │   ├── gemm_xmx.cpp
│   │   ├── dequant_q4_k.cpp
│   │   ├── dequant_q5_k.cpp
│   │   ├── dequant_q6_k.cpp
│   │   ├── dequant_q8_0.cpp
│   │   ├── dequant_awq_int4.cpp     (lane permutation [0,2,4,6,1,3,5,7])
│   │   ├── dequant_gptq_int4.cpp    (V1 +1 zero-offset; V2 plain)
│   │   ├── flash_attn_xmx.cpp
│   │   └── deltanet_scan_xmx.cpp
│   ├── kernels/esimd/               (escape hatch, hot loops)
│   │   ├── gemv_w4a16_esimd.cpp
│   │   └── flash_attn_esimd.cpp
│   ├── model/
│   │   ├── qwen35_moe.cpp           (graph builder)
│   │   ├── full_attn_block.cpp
│   │   ├── deltanet_block.cpp
│   │   └── moe_block.cpp
│   ├── engine/
│   │   ├── engine.cpp               (public API)
│   │   ├── prefill.cpp
│   │   └── decode.cpp
│   ├── server/                      (phase 10)
│   │   └── http_server.cpp
│   └── cli/
│       └── main.cpp                 (`ie chat -m model.gguf`)
├── tests/
│   ├── unit/
│   │   ├── gguf_reader_test.cpp
│   │   ├── dequant_test.cpp         (golden vectors vs ggml-quants ref)
│   │   ├── gemm_test.cpp            (vs oneDNN reference)
│   │   ├── flash_attn_test.cpp      (vs naive softmax)
│   │   ├── deltanet_test.cpp        (vs transformers state hook)
│   │   ├── tokenizer_test.cpp       (Qwen2 corpus, CJK, emoji)
│   │   └── sampler_test.cpp
│   ├── integration/
│   │   ├── first_token_test.cpp     (input [9707,11,1879] vs HF logits)
│   │   ├── perplexity_test.cpp      (wikitext-2)
│   │   └── chat_smoke_test.cpp      ("What is 2+2?" → contains "4")
│   └── perf/
│       ├── bench_gemm.cpp
│       ├── bench_dequant.cpp
│       ├── bench_decode.cpp         (tok/s gating; CI fails if <50)
│       └── bench_prefill.cpp
├── tools/
│   ├── convert_hf_to_gguf.py        (wrap llama.cpp's converter)
│   ├── dump_hf_activations.py       (capture transformers fixtures)
│   └── inspect_gguf.py
└── scripts/
    ├── setup_toolchain.sh           (intel/llvm nightly + driver pinning)
    └── run_perf_gauntlet.sh
```

---

## 5. Phase plan (sprint script)

Each phase is sized for ~3–5 working days. Each ends with a green test gate and a perf snapshot. **Don't move on with a red gate.** Re-read the relevant `research/*.md` before opening phase X.

### Phase 0 — Bring-up (3 days) — ✅ done 2026-04-25
- ✅ oneAPI 2026.0 DPC++ + compute-runtime 26.05.37020.3 (at floor) verified; cmake/ninja installed locally.
- ✅ CMake project skeleton with AOT target `intel_gpu_bmg_g31`.
- ✅ `hello_gpu` smoke test: B70 enumerated, vector add runs.
- ✅ `bench_gemm_fp16` via `joint_matrix` (M=8, N=16, K=16, M_REPEAT=8): numerically correct (max-abs 0.0002 on 4096³); ~26% of 183 TFLOPS peak with no SLM tiling.
- (deferred) `unitrace` / VTune profiles — pull when there's a real perf question to answer.
- **Original gate:** sample GEMM ≥50% of 183 TFLOPS at 4096³.
- **Achieved:** 26% of peak. 50% gate **deferred to Phase 3** where the SLM-tiled, 2D-block-load, double-buffered GEMM lives. Phase 0 amended-gate: *"toolchain end-to-end works; `joint_matrix` produces correct numerics on B70"* — both passed. See `bench/baseline/phase0_results.md`.

### Phase 1 — Tensor + GGUF loader (4 days) — ✅ done 2026-04-25
- ✅ `DType` + type-info table (all 40 ggml_type ids, block sizes, bytes/block).
- ✅ `Tensor` type (shape, dtype, host/device flag, view-by-default).
- ✅ `DeviceAllocator` (USM-device wrapper; Level Zero backend selected automatically).
- ✅ `GgufReader` mmap parser (header, KV table, tensor info, alignment).
- ✅ `ie-inspect` CLI: header, dtype histogram, tensor groups with expected-count gating.
- **Gate:** parse `Qwen3.6-35B-A3B-Q4_K_M.gguf` — **PASS** (all 733 tensors accounted for, every group matches expected count, every architecture KV matches research/04). See `bench/baseline/phase1_results.md`.
- New surprise (added to Phase 4 quirks list): `attn_q.weight` shape `[2048, 8192]`, not `[2048, 4096]` — output-gate is folded into Q-proj. Halve and split.
- New mapping (Phase 5): GGUF uses `ssm_*` prefix for DeltaNet tensors (not `linear_attn.*`). `attn_qkv.weight` is fused QKV on DeltaNet layers.

### Phase 2 — Dequant kernels (4 days) — ✅ done 2026-04-25
- ✅ Q8_0, Q4_K, Q5_K, Q6_K SYCL kernels (one per format, fp16 output).
- ✅ Host CPU reference (`include/ie/dequant_ref.hpp`) transcribed from `ggml-quants.c`.
- ✅ Unit tests: synthetic 1024 random blocks per format + real Qwen3.6 tensors (1B elements total). **All bit-exact.**
- ✅ Bench `bench_dequant`: 256 MiB packed per format, output GB/s reported.
- **Gate:** correctness within tolerances + Q4_K ≥ 350 GB/s output write — **PASS**: Q4_K 401 GB/s (115% of gate). Q6_K 339, Q5_K 255, Q8_0 192. See `bench/baseline/phase2_results.md`.
- Optimization note: naive 1-elem-per-item layout hit only 148 GB/s on Q4_K. 4-elem-per-item reorganization (64-item WG) brought it to 401, no special intrinsics needed.
- Phase 3 will subsume these via fused dequant→XMX. The standalone kernels stay as the trusted golden + small-N fallback.

### Phase 3 — Linear primitives & GEMV-W4A16 fused dequant (5 days) — ✅ correctness done 2026-04-25 / ⚠ perf deferred
- ✅ Element-wise kernels: `rms_norm`, `silu`, `swiglu`, `residual_add` (all bit/ULP-exact vs scalar reference).
- ✅ **Partial-rotary RoPE** (factor 0.25 = first 64 of 256 dims; theta=10M; m-rope-as-1d for text-only). Verified pos=0 identity, passthrough beyond n_rotary, norm-preserving rotation.
- ✅ **GEMV-Q4_K W4A16** kernel — bit-exact (max_rel 5e-4 = 1 fp16 ULP) on 4 real Qwen3.6 weight tensors. **220 GB/s effective** at steady state (gate ≥ 365 → MISS, but ~6 ms/token total GEMV cost ⇒ 165 tok/s ceiling — comfortably > 50 tok/s).
- ✅ **FP16 GEMM SLM-tiled** kernel — numerically correct (rel 8.7e-7). **33.5 TFLOPS** (18% peak), regression vs Phase 0 naive (47 TFLOPS, 26%) — the single-buffered SLM is paying barrier-per-K-iter cost.
- **Gate spec missed (perf):** GEMV-Q4_K 220 GB/s vs 365; GEMM-FP16 33 TFLOPS vs 128.
- **Why we ship anyway:** the engine's correctness pipeline is unblocked; the model-level 50 tok/s target has 3× headroom on the GEMV side, and prefill is not on the critical path until Phase 8. Both kernels need a perf revisit (XMX-fused GEMV; 2D block load + double-buffer GEMM) — added to Phase 9 backlog. See `bench/baseline/phase3_results.md`.

### Phase 4 — Full-attention block (5 days) — ✅ kernel-level correctness done 2026-04-25 / ⚠ FA-2 perf deferred to Phase 9
- ✅ `KvCache` (`include/ie/kv_cache.hpp` + `src/core/kv_cache.cpp`): layout `[L_full, kv_h, max_ctx, head_dim]` fp16, append helper.
- ✅ `sigmoid_gate` op for the `attn_output_gate=true` parameterization (Qwen3.6 quirk #5).
- ✅ `full_attention` kernel: online-softmax SDPA, GQA, causal-masked, handles both decode (T=1) and prefill (T>1) on arbitrary ctx_len.
- ✅ QK-Norm composed via existing `rms_norm` (caller does it on Q, K before attention). Partial RoPE composed via existing `rope_partial`.
- ✅ Tests: KvCache round-trip; constant-V → constant-output; causality (future garbage ignored); GQA group invariance + group split; vs scalar fp32 reference at T={1, 16, 256} all within max_abs=2e-4 (50× under the 1e-2 gate).
- ✅ Prefill GFLOPS captured (T=1024 → 1131 GFLOPS; T=256 → 478 GFLOPS).
- **Gate:** "matches HF transformers fixture" — kernel matches scalar fp32 reference within tolerance; full-stack diff vs HF transformers is Phase 8 work (needs out_proj + the rest of the pipeline wired). Phase 4 ships the correctness primitives.
- **Carried perf debt:** decode @ ctx=32k is 29 ms/layer due to 8× redundant K/V reads (naive WG-per-q_head layout). FA-2 rewrite (WG-per-kv-head, SLM-staged K/V chunks, modeled on sycl-tla v0.8 example #6) added to Phase 9 priority list; expected ~8× speedup. See `bench/baseline/phase4_results.md`.

### Phase 5 — Gated DeltaNet block (6 days, highest novelty risk) — ✅ done 2026-04-25
- ✅ Math extracted into `research/05_deltanet_math.md` (618 lines, verbatim line refs to `modeling_qwen3_5_moe.py`).
- ✅ `DeltaNetState` allocator: state `[L_lin=30, n_v=32, k_d=128, v_d=128]` fp32 + conv state `[L_lin, 8192, 3]` fp16 (~60 MiB total).
- ✅ `depthwise_conv1d_causal` kernel (kernel=4, channels-last, conv-state in/out).
- ✅ `compute_g_beta`, `l2_norm_scale`, `gated_rms_norm` primitives.
- ✅ `deltanet_recurrence` — fused 5-step rank-1 scan, one WG per (b, h_v), 128 lanes × 128 fp32 private state column.
- ✅ Chunkwise prefill: not needed — recurrent loop is mathematically T-agnostic (research/05 §8). Skipped intentionally; deferred to Phase 9 only if perf demands.
- **Gate result:** `out_max_abs = 6.7e-8` at T=4096 vs CPU reference (research/05's 12-step algorithm transcribed verbatim). The 1e-2 spec is met **5 orders of magnitude under target**. State diff = 1.7e-8 vs the 5e-3 spec. `prefill T=8 + decode T=1 == prefill T=9` is bit-exact. Conv1d, gated_rms_norm, l2_norm_scale, compute_g_beta all pass closed-form analytical tests.
- **Major correction caught:** `Qwen3_5MoeRMSNormGated` is plain `weight·x_normed`, NOT `(1+weight)` — research/04 had this backwards. Memory + research/04 fixed in this phase. The `(1+weight)` form is for the BLOCK-level RMSNorms (`input_layernorm`, `post_attention_norm`, `output_norm`) — Gemma3-style with weight init=zeros. The DeltaNet ssm_norm is the opposite — plain `weight·x_normed` with weight init=ones. Swapping these silently degrades quality, so research/05 §9 is now the authority.
- See `bench/baseline/phase5_results.md`.

### Phase 6 — MoE router + sparse expert exec (4 days) — ✅ correctness done 2026-04-25 / ⚠ perf deferred (launch overhead)
- ✅ Math extracted to `research/06_moe_math.md` (605 lines, verbatim line refs to `Qwen3_5MoeSparseMoeBlock`).
- ✅ `gemv_q6_K` W6A16 kernel (Q6_K signed 6-bit; needed for `ffn_down_exps` and `ffn_down_shexp`). Bit-exact (max_rel=5e-4) on real Qwen Q6_K tensors.
- ✅ `moe_router` — 1 WG per token, 256-lane fp32 logits + WG-wide softmax + serial top-8 + renorm + ascending-id sort.
- ✅ `shared_expert_gate` — per-token sigmoid scalar from `ffn_gate_inp_shexp`.
- ✅ `scaled_add` — `y += scale·x`.
- ✅ End-to-end test on layer 0 of the real Qwen3.6 GGUF: **router top-8 indices bit-exact**, weights max_abs=5e-5, full block max_abs=1.7e-5 (600× under 1e-2 spec).
- **Perf gate missed:** 2.1 ms/layer vs 1.5 ms spec. Diagnosis: 47 SYCL kernel launches/layer × ~25 µs each = 1.2 ms launch overhead alone; actual compute is ~0.6 ms.
- **Carried perf debt (highest Phase 9 priority):** fused-MoE-block kernel via SYCL graph or batched expert GEMVs — eliminates 46 of 47 launches, expected drop to ~0.6 ms/layer. With 40 MoE layers, this single change buys ~60 ms/token and is the load-bearing optimization for the 50 tok/s gate.
- See `bench/baseline/phase6_results.md`.

### Phase 7 — Tokenizer + sampler (3 days) — ✅ done 2026-04-25
- ✅ Byte-level BPE (`include/ie/tokenizer.hpp` + `src/tokenizer/tokenizer.cpp`). GPT-2 byte map, vocab + merges loaded from GGUF, special-token literal recognition during encode, decode round-trips losslessly.
- ✅ Simplified pretokenizer that matches HF for ASCII / common-prose; CJK + emoji round-trip cleanly. Full `\p{L}\p{N}\p{M}` Unicode regex deferred to Phase 9 backlog (no observed correctness loss; a few edge cases may differ from HF in token *count*, never in *decoded text*).
- ✅ `build_chatml_prompt` for the Qwen3.6 ChatML template (system/user/assistant + `<|im_start|>assistant\n<think>\n` opening).
- ✅ Samplers: `sample_argmax`, `repetition_penalty`, `sample_softmax_topk_topp` (temperature + top-k + top-p + min-p + multinomial inversion sampling, seeded reproducibly).
- **Gate:** round-trips 5 input categories (ASCII prose + CJK + emoji + whitespace edge cases + ChatML), special tokens recognized by ID. **All 13 tests PASS.**
- **Caught and corrected:** research/04 §6.1's claim that `Hello, world → [9707, 11, 1879]` was wrong. The actual IDs in Qwen3.6's GGUF are `[9419, 11, 1814]`. ID-level fixtures must be re-derived from the actual GGUF — Qwen3.6 reassigns positions in the < 151k range, not just appending above. research/04 + memory updated.
- See `bench/baseline/phase7_results.md`.

### Phase 8 — End-to-end correctness (4 days) — ✅ done 2026-04-26 (resolved in Phase 9 correctness sub-phase)
- ✅ Full 40-layer pipeline implemented. `embedding → 40×{rms_norm(1+w) → (FullAttn|DeltaNet) → res → rms_norm(1+w) → MoE → res} → rms_norm(1+w) → lm_head` runs end-to-end.
- ✅ Per-tensor dtype dispatch (Q4_K_M's heuristic mixes Q4_K and Q6_K per layer); `gemv_q(...)` switches based on `QuantPtr.dt`.
- ✅ New kernels: `rms_norm_one_plus_w`, `rms_norm_f32w`, `cast_fp16↔fp32`, `repeat_interleave_heads`, `split_q_gate_per_head`, `embedding_lookup_q4k/q6k`.
- ✅ Workspace allocator sizes to max_T and reuses; KV + DeltaNet caches plumbed; positions array materialized.
- ✅ `ie-forward` CLI: load + tokenize + prefill + argmax + greedy decode loop.
- ✅ Loads ~22 GB Q4_K_M to device in 2.4 s; prefill ~31 tok/s (3-token prompt).
- ✅ **No NaN/Inf** through 40 layers; per-layer residual L2 grows smoothly (4 → 75).
- ⚠ **Output not yet semantically correct.** `Hello, world` argmax = ` réc` instead of a sensible continuation. Different prompts give different (but uniformly-gibberish) outputs, confirming input flows through but math is still off somewhere.
- ✅ **9 bugs caught and fixed total** (6 in Phase 8, 3 more in Phase 9 from llama.cpp source review):
  1. token_embd Q4_K (not Q6_K) detection
  2. Per-layer mixed Q4_K/Q6_K dispatch
  3. ssm_conv1d / ssm_norm FP32→FP16 cast at load
  4. Q-proj per-head Q+gate interleaved layout (not first/second half) — `split_q_gate_per_head`
  5. Workspace pointer-swap aliasing across layers — dedicated pre-GQA buffers
  6. **conv1d weight `[4, 8192]` is kernel-leading-contiguous**: `w[k + c*kernel]`, not `w[k*channels + c]`. Took layer 0 residual L2 from 23 → 4.
  7. **Block-level RMSNorm `(1+w)` is baked into the GGUF** at conversion time; correct kernel is plain `weight*x_normed`. Took L=0 L2 from 4 → 2.4.
  8. **`ssm_a` stored as pre-computed `-exp(A_log)`**, not raw A_log. Comment in `qwen35moe.cpp:238` confirms.
  9. **V heads stored in TILED order**, not grouped. `repeat_interleave_heads` flipped to tiled. L=0 L2 dropped 4 → 1.2 (sane).
- L=0 residual is now reasonable, all 40 layers run without explosion, but argmax is still wrong.
- **Phase 9 priority #0 (NEW):** build `llama-cli` and diff forward-pass activations layer-by-layer to localize the remaining bug. Most likely candidates: full-attn pipeline detail, mRoPE collapse, or conv1d T<kernel edge case.
- See `bench/baseline/phase8_results.md` and `bench/baseline/phase8_trace.txt`.

### Phase 9 — Decode loop + perf push (5 days) — ⚠ correctness ✅ done 2026-04-26 / ❌ perf items not started

#### 9a. Correctness — ✅ done 2026-04-26
- ✅ Built `tools/ie-llama-dump` linked against llama.cpp's Vulkan build (`/home/weezy/llama.cpp/build-vk/`); registers `cb_eval` to capture per-layer residuals as fp32 binaries (`<prefix>_L00..L41.bin`) plus per-block attn output (`<prefix>_A00..A39.bin`). Vulkan because `ggml-sycl` has known regressions on B70.
- ✅ Built `tools/ie-debug` long-running REPL: loads weights once (~22 GB), accepts `P <prompt>`/`R`/`Q` on stdin, opt-in fp32 dumps via `--dump`. **Solves the previous "fresh-process-per-prompt = 22 GB load each = parallel-load OOM crash" failure mode.** Discipline rule: never run `ie-llama-dump` and `ie-debug` simultaneously.
- ✅ `tools/diff_layers.sh` — bash + Python max-abs / max-rel diff per layer.
- ✅ Added `QwenModel::set_dump_prefix()` + per-residual fp32 dumps in forward().
- ✅ **3 final correctness bugs caught and fixed** (12 total Phase 8+9):
  10. **Conv1d kernel direction reversed.** PyTorch `nn.Conv1d` (and llama.cpp `ggml_ssm_conv` in ggml-cpu/ops.cpp:9297) treats `W[0]` as the OLDEST tap and `W[K-1]` as the CURRENT tap. Our impl had `W[0]` as current, indexed against `x[t-k]`. Fix in `src/ops/conv1d.cpp`: read `W[K-1-k]` instead of `W[k]`. Single-token argmax went from `.` to `,` (logit 14.47 — exact match with llama.cpp reference 14.46).
  11. **`gemv_q` is single-token only** — both Q4_K and Q6_K kernels SLM-cooperatively load the full A vector and produce one row of y. For multi-token prefill (T>1), only token 0 was projected; tokens 1..T-1 read uninitialized memory. Added `gemv_q_T` looping helper in `src/model/qwen36.cpp`; replaced all attn/MoE projection sites.
  12. **DeltaNet fused QKV split is per-token, not per-component.** `attn_qkv` produces `[T, SI2=8192]` where each row is `[Q(2048), K(2048), V(4096)]`. The naive 3 contiguous `cast_fp16_to_fp32` calls spliced token-0's K/V into token-1's Q buffer, etc. Fix: per-token strided cast loop. With this fix, multi-token prefill works.
- ✅ **Verification**: layer-by-layer fp32 binary diff vs llama.cpp Vulkan: max_abs < 0.005 through L11, max_abs < 0.05 (FP16 accumulation noise) through L41. Sample completions all match reference behaviour:
  - `Hello, world` → `!` (logit 17.56, **bit-exact match** with llama.cpp)
  - `The capital of France is` → ` Paris`
  - `2+2=` → `4`
  - `Once upon a time` → `,`
- See `bench/baseline/phase9_results.md` and `tools/{ie_debug,llama_dump,diff_layers.sh}`.

#### 9b. Decode loop — ⚠ minimal scaffold in `tools/ie-debug`; pinned-graph optimization deferred to 9c.
- ✅ KV append, DeltaNet state update, position-ID advance — all working in `ie-debug`'s `P <prompt>` command (each call increments `cur_pos`, state buffers persist across commands).
- ✅ Sampler: `sample_argmax` in use; full top-k/top-p sampler exists from Phase 7 but isn't wired into `ie-debug` yet.
- ⏳ **Steady-state decode currently 19 tok/s** (52.5 ms/token, σ ≈ 0.5 ms over 20 sequential T=1 commands). 38% of the ≥50 tok/s gate, 19% of the ≥100 tok/s stretch.

#### 9c. Perf push — ⚠ STILL OPEN (premature v1 ship 2026-05-02 walked back; see end of section)

| Step | Decode tok/s @ ctx=1 | Δ |
|---|---|---|
| Correctness baseline (post-9a)                  | 19.0 | — |
| + Fused MoE (item 1, T=1 only)                  | 27.9 | +8.9 (+47%) |
| + In-order queue + drop kernel `.wait()` (item 4) | 33.8 | +5.9 (+21%) |
| + Shared-expert dev-scalar (kill host roundtrip) | 34.3 | +0.5 (+1.5%) |
| + FA-2 split-K decode (item 2)                  | 34.3 @ ctx=1, **31.6** @ ctx=4k, **20.2** @ ctx=32k | mostly long-ctx win |
| + gemm_q4_K + fused MoE for prefill (items 3+5) | 34.3 @ ctx=1 (no decode change), **PP 55→82 tok/s (+50%)** | prefill win |

**ie-bench (2026-04-26, end of session — items 1+2+3+4+5 done):**

| T_pp | PP tok/s | TG tok/s | TG @ session start | Δ TG |
|---|---|---|---|---|
| 1     | 33   | 34.4 | 34.4 | — |
| 256   | **89**   | 33.3 | 30.8 | +8% |
| 1024  | **87**   | 33.2 | 23.4 | +42% |
| 4096  | **82**   | 31.5 | 11.9 | **+165%** |
| 16384 | †    | 25.3 | (was unmeasurable) | new |
| 32000 | †    | **20.0** | (was unmeasurable) | new |

† Real prefill at 16k+ takes 10+ min via gemm_q4_K with M_TILE=8 (PP at long ctx is improved but bench step still dominated by the prefill itself). Decode at long ctx measured via `--fastforward`.

**Session-cumulative wins:**
- Decode small-ctx: **19 → 34.3 tok/s (+80%)**
- Decode @ ctx=4k:  **11.9 → 31.5 tok/s (+165%)** (FA-2 dominates)
- Decode @ ctx=32k: **(unmeasurable) → 20.0 tok/s**
- Prefill PP:       **57 → 89 tok/s (+56%)** (fused MoE prefill dominates)

**Phase 9 gate is ≥50 tok/s @ 32k = ≤20 ms/token. Currently 49.6 ms/token. Need to halve.**

Gap analysis at 32k:
- Small-ctx baseline (everything except attn): 29.2 ms — needs to drop to ~12 ms (≈2.5×).
- Attn overhead at 32k (post-FA-2): ~20 ms — needs to drop to ~8 ms (≈2.5×).
- Both sides need a ~2.5× compression for 50 tok/s @ 32k.

The 29.2 ms baseline is dominated by the per-token Q4_K GEMV streaming of ~1.84 GB at ~220 GB/s = 8.4 ms ideal, so 21 ms is overhead. Most of that overhead is per-kernel runtime that XMX/2D-block-load (item 3+5) directly addresses.

The 20 ms 32k attn overhead is FA-2 already with optimal KV reads — at gqa=8 we now read each KV element once. To go faster needs better K/V SLM tiling (Bc=128 maybe), or fp8 KV cache, or paged attention to exploit sparser attention patterns (latter two are out of scope for v1).

**Per-token budget at 34.3 tok/s = 29.15 ms.** Pure-bandwidth floor for active weights only (1.84 GB / 220 GB/s effective Q4_K GEMV) ≈ 8.4 ms; remaining ~21 ms is split between (a) per-kernel runtime overhead on ~1000 launches/token, (b) non-bandwidth-bound kernels (DeltaNet recurrence with serial scan, full attention naive layout), and (c) launch overhead the in-order queue can't fully hide. To reach 50 tok/s = 20 ms/tok, need to save 9 ms; to reach 100 tok/s = 10 ms, need to save 19 ms (close to the bandwidth floor — would need sub-200 GB/s active weights via better quants or KV-only-streaming).

**Per-layer cost (measured 2026-04-26):** ~580 µs/layer × 40 layers + ~5.5 ms fixed (embedding + final norm + lm_head). Each layer averages ~25 kernel launches × ~23 µs/kernel — the cost is dominated by per-kernel runtime (compute + barriers + register setup), not CPU launch latency.

**GEMV-Q4_K bench (2026-04-26 post-cleanup):** 185 GB/s on attn_q (2k×8k), 222 GB/s on qkv (2k×8k), 251 GB/s on 4k×8k. Gate is 365 GB/s. Tried `N_PER_WG=32` (was 16) — no measurable change (29.15 ms → 29.18 ms). Larger tile didn't help, so weight-read bandwidth is the actual bound, not SLM cooperative-load amortization. Hitting the gate needs either XMX-fused (item 3) or 2D block load (item 5).

**Launch-fusion is tapped out (2026-04-26):** also tried fusing the two `cast_fp16_to_fp32(α)` / `cast_fp16_to_fp32(β)` launches into a single `compute_g_beta_h16` per DeltaNet layer (60 launches/token saved). No measurable change — the in-order queue already pipelines tiny back-to-back kernels well enough that saving small launches is below measurement noise. The remaining gap to 50 tok/s is **per-kernel runtime**, not launch count. Next moves are item 3 (XMX-fused GEMV) or item 5 (2D block-load GEMM patterns applied to GEMV).

**Carried perf debt from Phases 3 + 4 + 6**, in execution priority. Each item lists expected savings.

  1. **Fused MoE block** — ✅ done 2026-04-26. Two new kernels in `src/ops/moe_fused.cpp`: `moe_decode_gate_up_silu_q4k` (Q4_K gate + Q4_K up + swiglu, K_top experts in one launch, 1 WG per (k, n-chunk)) and `moe_decode_down_q{4,6}k` (Q4_K- or Q6_K-down + per-expert scale + accumulate, 1 WG per H output chunk, loops K_top experts internally). Reads `topk_idx`/`topk_w` from device — no host roundtrip. Eliminates **~38 launches per MoE layer** (40 → 2 routed + 1 zero-init); the 5 shared-expert launches and router stay for now.
     - Per-token: 35.8 ms (σ ≈ 0.4 ms over 22 T=1 commands) = **27.9 tok/s**.
     - Saved: ~16.7 ms/token vs 9a baseline. Expected was ~36 ms, undershot — likely because the per-launch overhead on B70 is closer to ~15 µs than ~30 µs estimated, and/or the kernel is less efficient than the original. Profile next.
     - Decode-path-only optimization. Prefill (T>1) still uses the scalar per-expert loop and is unaffected.
  2. **FA-2 split-K decode** — ✅ done 2026-04-26. New `full_attention_fa2_decode` (T=1) splits ctx into Bc=64 chunks along K, dispatches WG per (kv_head, chunk). Each WG cooperatively SLM-stages its (Bc × head_dim) K and V tiles ONCE, runs the gqa=8 SGs that share this kv_head over the tile, writes per-chunk online-softmax partials. Combine pass merges per q_head. Eliminates the 8× redundant KV reads of the naive WG-per-q_head layout. Workspace: `ws_attn_partials_` FP32 [n_chunks_max, n_q_heads, head_dim+2] = 8.25 MB at max_ctx=32k+64. Effect: TG @ ctx=4k went 11.9 → **31.6 tok/s (2.65×)**; TG @ ctx=32k = **20.2 tok/s** (was unmeasurable before — naive estimate was ~5 tok/s). TG flat across short ctx (34→33 from 1→1k). Tried Bc=128 — slower (SLM occupancy / parallelism trade-off), reverted to Bc=64.

  3. **gemm_q4_K (multi-row) + fused MoE prefill** — ✅ done 2026-04-26 (combined with item 5). New `gemm_q4_K` in src/ops/gemv_q4k.cpp processes M ≤ M_TILE=8 rows in one launch with A SLM-staged across rows. `gemv_q_T` now uses gemm_q4_K M_TILE=8 chunks for Q4_K weights. Also lifted the `T==1` gate on can_fuse_moe so prefill goes through fused stages (per-token loop, 2 launches/token instead of 40). **PP went 55→89 tok/s @ short ctx, 55→82 @ Tpp=4k (~+50%)**. Tried M_TILE=16 — slower (occupancy + register pressure). True XMX-fused via `joint_matrix mat_mad` or 2D block-load is still open as a follow-up perf item if PP needs to go further (cutlass-sycl `02_bmg_gemm_mixed_dtype` is the closest reference).

  4. **In-order queue + drop kernel `.wait()`** — ✅ done 2026-04-26. Switched `DeviceAllocator::init` to construct the queue with `sycl::property::queue::in_order{}`, then bulk-removed 49 of 60 `.wait()` calls in `QwenModel::forward()` (kept the device→host memcpy waits + the host→device positions copy). With in-order semantics, every kernel implicitly depends on the prior submission; the caller's single `.wait()` on the returned lm_head event syncs the whole graph. Saved: 29.6 ms/token vs 35.8 ms = **+5.9 tok/s (+21%)**. CPU-side launch overhead was the dominant remaining cost, not Battlemage launch cost. **Single SYCL graph for the whole step is a separate, smaller follow-up** if needed (record-and-replay would save the remaining ~600 µs of CPU dispatch per token, ~0.5 tok/s).

  5. **(merged into item 3 — done together as gemm_q4_K + fused MoE prefill)**

  6. **GEMM-FP16 2D block-load** via `cl_intel_subgroup_2d_block_io` (research/02 §3.2 / §6.1) — for FP16 GEMM. ⛔ **BLOCKED at hardware level** — same `lsc_load_block2d` path as item 13. See item 13 (and `memory/project_v2_phase1_freeze_diagnosis.md`) for the full driver/firmware matrix tested 2026-04-26. Re-evaluate whenever item 13 unblocks.

  7. **ESIMD `xmx::dpas` escape hatch** (research/02 §2c) for whichever inner loop SYCL JIT compiles poorly. ⛔ **BLOCKED** — register-resident ESIMD on Battlemage requires `cl_intel_subgroup_2d_block_io` for the weight loads (the cute/cutlass ESIMD reference impls use it).  Same HW block as items 6 + 13.

  8. **(NEW) INT8 KV cache** — ✅ **shipped + benchmarked** 2026-04-27 / optimized 2026-05-01.

      **Implementation:** `KvCache` carries optional shadow buffers `k_int8_/v_int8_ + k_scales_/v_scales_` (per-row symmetric INT8, one fp16 scale per (layer, kv_head, position) row of `head_dim` INT8). Allocator wired in `src/core/kv_cache.cpp` behind the `KvCacheConfig::use_int8` flag. Forward path in `qwen36.cpp` selects `full_attention_fa2_decode_int8` (defined in `src/ops/attention.cpp`) at decode (T=1) when `kv.is_int8()`. Prefill (T>1) and naive paths still use fp16. Bench flag: `ie-bench --int8-kv`.

      **Bug found + fixed (2026-04-27):** `full_attention` (T>1 prefill) wrote fp16 K/V only — the INT8 shadow was never populated during prefill. `full_attention_fa2_decode_int8` (T=1 decode) then read uninitialised INT8 rows for all prior positions, producing silent garbage logits in every real chat session starting with T>1 prefill. **Fix:** added `KvCache::quantize_to_int8()` (post-quantizes fp16 prefill rows into the INT8 shadow) and `int8_lengths_[]` watermark (gates INT8 decode path: only enabled if `start_pos == int8_length(layer)`, i.e. every prior position is INT8-populated). Also updated `ie-perplexity` with `--prefill-chunk N` mode and `validate_int8_kv.sh` with a second prefill→decode test. The streaming-T=1 perplexity test never exercised this bug.

      **Quantize-on-write:** WG-per-kv_head, head_dim work-items, cooperative reduce_over_group max-abs → fp16 scale → symmetric INT8 in [-127,127]. Fp16 shadow optional (passed `nullptr` from the model since prefill T>1 paths don't reach this kernel).

      **Dequant-on-read (v2, 2026-05-01):** `K_slm`/`V_slm` changed from `fp16` (32KB each) to `int8_t` (16KB each) → total SLM 32KB+256B → 3 WGs/subslice on Xe2's 128KB SLM (was 1 WG/subslice at 64KB+256B). Two barriers/chunk collapsed to one: scales + raw INT8 tiles loaded together, dequant (`float(tile[i]) * scale`) inlined in the compute loop. Eliminates the occupancy cliff that caused +110% slowdown at 8k context.

      **Standalone ops (2026-05-01):** added `quantize_kv_to_int8` and `dequantize_kv_from_int8` to the op API for contiguous `[n_rows, head_dim]` KV rows. They use the same per-row symmetric INT8 format as the cache (`scale=max_abs/127`, zero rows use scale=1, values clamped to [-127,127]). The FA-2 decode path keeps dequant fused into attention tile loads for bandwidth; these ops are for validation and general cache tooling.

      **Launch-overhead fix (2026-04-27):** ported the `CHUNKS_PER_WG=8` super-chunk pattern from `full_attention_fa2_decode` to the INT8 variant. Each WG now processes 8 inner Bc-tiles with running m/l/out accumulators and writes ONE super-partial — eliminates ~7/8 of the launch overhead at long ctx (n_chunks_max=512 at 32k → n_super_chunks_max=64). Compile-clean against IGC 2.32.7. Combine pass updated to iterate `n_super_chunks`. `ws_attn_partials_` scratch unchanged (still sized for `n_chunks_max`, which over-provisions both paths — safe).

      **Memory cost:** at max_ctx=32k for L_full=10 / kv_h=2 / d=256:
      | | bytes/tok | total @ 32k |
      |---|---|---|
      | fp16 cache (always allocated)            | 10 KiB | 320 MiB |
      | int8 + scales shadow (when `--int8-kv`)  | ~5.04 KiB | ~161 MiB |

      **Measured impact (2026-05-01, Arc Pro B70, Qwen3.6-35B-A3B-Q4_K_M):**

      Sub-kernel profiler (partial kernel only, ie-attn-profile):
      | ctx    | fp16 partial | INT8 partial | delta       |
      |--------|-------------|-------------|-------------|
      | 1,024  | 0.548 ms    | 0.532 ms    | **-3.0%**   |
      | 4,096  | 0.540 ms    | 0.538 ms    | **-0.4%**   |
      | 8,192  | 0.576 ms    | 0.539 ms    | **-6.4%**   |
      | 16,384 | 0.778 ms    | 0.552 ms    | **-29.0%**  |

      End-to-end decode (ie-bench TG tok/s):
      | ctx   | fp16  | INT8  | delta     |
      |-------|-------|-------|-----------|
      | 1     | 33.70 | 33.62 | -0.2%     |
      | 256   | 29.96 | 30.54 | **+1.9%** |
      | 1,024 | 26.90 | 28.05 | **+4.3%** |
      | 4,096 | 27.11 | 27.94 | **+3.1%** |
      | 16,384| 26.55 | 27.82 | **+4.8%** |

      INT8 KV is faster than fp16 at every meaningful context. The 8k occupancy cliff (+110% slowdown) is eliminated.

      **Validation status (2026-05-01):**
      - Compile-clean ✅
      - Prefill→decode correctness: ✅ (`attention_test` ALL PASS, cos-sim > 0.99)
      - `validate_int8_kv.sh`: ✅ streaming drift 0.498%, prefill→decode drift -0.10% (both < 0.5% gate)
      - `ie-bench --int8-kv`: ✅ benchmarked above

  9. **Scatter-gather MoE for prefill** — ❌ tried 2026-04-26 with naive per-expert dispatch (counting-sort + gather + per-expert fused gemm + scatter), and it was **slower at all measured T** (66 vs 89 tok/s @ T=256, etc.). The 256 expert dispatches × 6 launches each (gather, 2× gemm, swiglu, gemm, scatter) is launch-bound when M=avg-tokens-per-expert is small (8-128). To win, the rewrite needs a **multi-expert single-launch kernel** that processes many (or all 256) active experts in one dispatch — joint_matrix-batched. Infrastructure kept (`moe_gather_rows`, `moe_scatter_add`, `ws_moe_x_packed_/token_idx_/token_w_`) for that future work.

  10. **Multi-expert single-launch MoE kernel** — ⚠ shipped iteratively 2026-04-26. Atomics-free version (per-packed-row stage-2 output + separate reduce kernel) replaces the original atomic_ref<float> variant. Final design:
      - Sort tokens by expert (host counting sort, also builds tk_to_packed inverse map)
      - Stage 1 (`moe_prefill_gate_up_silu_q4k`): all 256 experts in one launch, M_TILE=8, N_PER_WG=32
      - Stage 2 (`moe_prefill_down_packed_q{4,6}k`): writes per-packed-row outputs (no atomics, no race), N_PER_WG=32
      - Reduce (`moe_prefill_reduce`): per (token, h) sums weighted K_top contributions via tk_to_packed map
      - 4 launches per MoE layer total
      - T-adaptive: 64 ≤ T < 2048 → multi-expert. T<64 → fused-per-token (setup overhead > work). T≥2048 → fused-per-token (data locality wins; multi-expert touches all 256 experts' weights spread across DRAM).

      **Final PP profile (this hour):**
      | T_pp | PP tok/s | vs hour-2 |
      |---|---|---|
      | 64    | 94.2 | new |
      | 128   | 95.5 | new |
      | 256   | **95.5** | +6% |
      | 512   | 93.2 | new |
      | 1024  | 90.6 | +4% |
      | 2048  | 84.4 | parity (fallback) |
      | 4096  | 81.2 | parity (fallback) |

      Tried but reverted: M_TILE=16 (-7-9%, register pressure); single-stream INT8-native dot (skipped due to scope).

      **Path to actually close the 9× PP gap to llama.cpp Vulkan (865 tok/s @ T=512):** needs `cl_intel_subgroup_2d_block_io` for the Q4_K weight loads (turns DRAM-bandwidth into 2D-block-tile bandwidth, ~3-4× saving) AND `joint_matrix` `mat_mad` for the inner dot product (XMX). Both are multi-day kernel projects. Item 10 closes the in-tree-with-current-kernels lever; further PP gains need new kernel infrastructure.

  12. **Per-section profiling + small fusion wins** — ✅ done 2026-04-26. Added `QwenModel::set_profile/dump_profile` with section-level `q.wait()` boundaries and a `--profile` flag to `ie-bench`. Findings drove three small wins: (a) fused QKV-split-cast for DeltaNet (3*T launches → 1), (b) inline SiLU into `depthwise_conv1d_causal` (saves 30 launches/token), (c) tried Q6_K XMX for `lm_head` and **reverted** — at M=1 the SLM dequant round-trip is overhead vs scalar's inline dequant, ~5% slower.

      **Profile breakdown (decode T=1, ms across 8 steps, 244 ms total):**
      | Section | ms | % |
      |---|---|---|
      | moe_routed_decode | 88.5 | 36.2 |
      | attn_dn_block | 51.2 | 21.0 |
      | lm_head | 43.7 | 17.9 |
      | moe_shared | 19.5 | 8.0 |
      | attn_norm | 12.8 | 5.2 |
      | post_attn_norm | 12.8 | 5.2 |
      | attn_full_block | 11.0 | 4.5 |
      | residuals + final_norm | 4.6 | 1.9 |

      **Profile breakdown (PP T=128, 1278 ms total):**
      | Section | ms | % |
      |---|---|---|
      | attn_dn_block | 597.5 | 46.8 |
      | moe_routed_multiexpert | 420.3 | 32.9 |
      | moe_shared | 154.0 | 12.0 |
      | attn_full_block | 97.2 | 7.6 |
      | lm_head + norms + residuals | <10 | <1 |

      **Strategic finding (the negative ones from this work):** XMX kernels with SLM-staged Q4_K/Q6_K dequant lose to scalar inline-dequant for **bandwidth-bound shapes** (M=1 GEMVs like lm_head and the multi-expert MoE per-(expert) work). They WIN for **compute-bound shapes** (M ≥ 8 multi-row GEMM in projection kernels, where weight reads amortize across rows). To beat XMX-with-SLM on bandwidth-bound shapes, we need 2D block I/O (item 13).

  13. **`cl_intel_subgroup_2d_block_io` + ESIMD `xmx::dpas`** — ⛔ **BLOCKED at hardware level on BMG-G31 stepping C0 (IP 20.2.0)** as of 2026-04-26. **Pivot trigger met** — see "v2.0 path forward" below.

      **Empirical findings (2026-04-26 session):** Day 1-2 plumbing smoke test was wired up (`src/ops/gemm_q4k_esimd.cpp`, `tests/unit/esimd_block2d_smoke_test.cpp`, currently in `git stash@{0}`). Tested both the legacy `__builtin_IB_subgroup_block_read_flat_u8_m8k32v1` form AND the modern `__spirv_Subgroup2DBlockLoadINTEL` op with a 256-byte tile (8 rows × 32 cols u8). With `-Xs "-device bmg_g31"` + `+SPV_INTEL_2d_block_io,...` + `IGC_allowDecompose2DBlockFuncs=0`, IGC emits a well-formed `lsc_load_block2d.ugm V*:d8.32x8nn flat[...]` in the vISA, and final ASM is a single `load_block2d.ugm.d8.a64`. **But: every smoke run hangs the GPU.** journalctl shows `xe 0000:04:00.0 [drm] Xe device coredump` + 4-6 `exec queue reset detected` events per run; smoke test returns sum=0 (kernel times out, host gets the un-touched memset(0) buffer back).

      **Driver stack tested at:**
      - Kernel: 6.17.0-22-generic, xe driver
      - GuC firmware: **70.60.0** (upgraded from 70.44.1 via kernel.org linux-firmware drop-in — the kernel was logging "GuC firmware (70.49.4) is recommended, but only (70.44.1) was found" before the upgrade)
      - compute-runtime / NEO: **26.14.37833.4** (from Intel GitHub release; kobuk-team PPA was at 26.05.37020.3)
      - IGC: **2.32.7** (from Intel GitHub `intel-igc-core-2` + `intel-igc-opencl-2`; PPA was at libigc2 2.28.4)
      - libze: 1.15.37833+4

      All upgrades verified post-reboot via `sycl-ls`. The hang persists across all combinations — driver-stack staleness is **not** the root cause. The HW path for `lsc_load_block2d` itself is broken on this stepping. Cutlass-sycl only tests `bmg_g21` (B580, IP 20.1.0); they do not exercise G31.

      **Soft-freezes during diagnosis:** the cumulative xe-driver GPU resets killed Chrome/VSCode GL contexts and wedged the desktop ~5 times during the session. See memory `project_v2_phase1_blocked_2d_block_io.md` for safe-iteration checklist.

      **v2.0 path forward (revised 2026-04-26):**
      - **(a) Wait** 1-3 months for driver/firmware fixes to land in the kobuk PPA, then `git stash pop` and re-test. Cheapest if not time-pressed.
      - **(b) Pivot to oneDNN `f16:u4`** ⚠ — Intel-tested codepath on B70 (their internal validation includes G31). **Critical constraint: Q4_K_M is NOT compatible with oneDNN `f16:u4`, which is single-level groupwise INT4, because Q4_K_M uses a nested format (super-block fp16 d/dmin + 6-bit sub-scale/min metadata per 256-element block).** Integration requires: (1) write a Q4_K_M → groupwise INT4 repack kernel at model load time, (2) run perplexity gate (fp16 vs repacked INT4 PPL drift ≤ 1%) before claiming any performance win. If PPL is acceptable, closes ~60–70% of the 9× gap to llama.cpp Vulkan (~3–4× PP improvement). Realistic timeline: ~3 weeks (repack + PPL validation + integration). **Do not cite "3–4× PP improvement" without the repack + PPL gate completed.**
      - **(c) joint_matrix `mat_mad` improvements only** — stay in SYCL idiomatic path, push the existing `gemm_q4_K_xmx` (`gemm_q4k_xmx.cpp`) further. Lower ceiling but no new HW-feature dependency.

      **--- Original plan kept below for historical context (projections superseded by HW block) ---**

      **Honest scope:** ESIMD is a different programming model from SYCL data-parallel (vector types, explicit memory access, lane-private storage). The cutlass-sycl reference implementations of mixed-dtype GEMM via ESIMD are 1k+ lines and tuned per shape. A correct first version is 1-2 weeks of focused kernel programming + correctness debugging.

      **Updated impact estimate (after sub-profiling, 2026-04-26):**
      - dn_qkv_proj (the largest single PP cost at 20% of T=128 PP) currently runs gemm_q4_K_xmx at ~47 GB/s effective on its (M=8, K=2048, N=8192) shape — 21% of peak DRAM. 2D block I/O could lift this to 60-70% of peak → ~3× on this kernel alone. Replicated across alpha/beta/gate/ssm_out/MoE projections, projected v2.0 PP target is **~250-350 tok/s** (3-4× current 101 tok/s), closing most of the 9× gap to llama.cpp Vulkan's 865 tok/s.
      - lm_head: from 5.34 ms/token → ~2 ms (3× win), TG @ ctx=1: 34 → ~38 tok/s.

      **CORRECTED priority**: this is the dominant v2.0 lever.  The previously-suggested DeltaNet chunkwise prefill (consultant Action #2) was based on assuming the recurrence is the bottleneck — sub-profiling shows the recurrence is only 0.6% of PP at T=128.  **DeltaNet chunkwise is NOT a v2.0 priority.**  All structural perf wins now route through items 13 (ESIMD GEMM) and 11 (Phase 11 server + perplexity validation). New `src/ops/gemm_q4k_xmx.cpp` with WG tile (M ≤ 8) × BN=64 × BK=256 (one Q4_K block per K iter), 4 SGs each computing one TM=8 × TN=16 sub-tile via `joint_matrix_mad`. Cooperative Q4_K block dequant once per K-iter then reused across 16 inner K-tile mat_mads. fp32 acc → fp16 via local fp32 scratch. `gemv_q_T` dispatches the XMX variant when shape constraints (N % 64 == 0, K % 256 == 0) are met.

      **PP profile (with XMX in projection layers):**
      | T_pp | Before XMX | After XMX | Δ |
      |---|---|---|---|
      | 64    | 94.5  | 98.7   | +4.4% |
      | 128   | 96.0  | **100.5** | +4.7% **← crossed 100 tok/s** |
      | 256   | 95.9  | 100.0  | +4.3% |
      | 512   | 93.3  | 97.0   | +4.0% |
      | 1024  | 89.6  | 92.8   | +3.6% |
      | 2048  | 84.5  | 86.9   | +2.8% |
      | 4096  | 81.6  | 83.2   | +2.0% |

      **Tried but reverted:** `moe_prefill_gate_up_silu_q4k_xmx` — XMX variant of the multi-expert MoE stage 1. My 2-SG-per-WG layout was too narrow and ran -11-13%. The scalar version uses N_PER_WG=32 → 32 SGs per WG for high per-WG parallelism; the XMX rewrite needs a wider design (8+ SGs across N×M tiles). Code kept in `gemm_q4k_xmx.cpp` for future work but not dispatched.

      **Item 11 effectively closes the projection-layer XMX work.**  The remaining PP gap to llama.cpp Vulkan now lives in: (a) the multi-expert MoE stage 1+2 (still scalar), (b) Q4_K weight reads via `cl_intel_subgroup_2d_block_io` (2D block load).  Both are kernel rewrites; both are next-session-sized.

  14. **gemm_q6_K (multi-row Q6_K GEMM)** — ✅ done 2026-05-01. Pre-this, `gemv_q_T` fell back to a per-row `gemv_q6_K` loop for T>1 on Q6_K weights (30 ssm_out + 14 attn_qkv DN + 6 attn_v + ffn_down_shexp on 30 layers) — every prefill row paid the full Q6_K weight read with no amortization. New `gemm_q6_K` in `src/ops/gemv_q6k.cpp` mirrors `gemm_q4_K`'s tile structure (16 SGs × 16 lanes, M_TILE=8, SLM A tile = M×K halfs) with Q6_K dequant lane-mapping identical to `gemv_q6_K`. `gemv_q_T` routes Q6_K T>1 calls to `gemm_q6_K_xmx` when shape allows (N%64==0, K%256==0), else scalar `gemm_q6_K`. `IE_NO_XMX=1` forces scalar.

      **Bench (3 runs, lmhead-q4k variant, post-thermal-noise-aware methodology):**
      | prompt              | prefill before | prefill after (XMX) | Δ      |
      |---------------------|----------------|---------------------|--------|
      | short-chat          | 116.8          | 136.9               | +17.2% |
      | long-instruction    | 114.7          | 124.0               | +8.1%  |
      | codegen             | 113.6          | 123.6               | +8.8%  |
      | math-reasoning      | 114.0          | 124.2               | +8.9%  |
      | long-context        | 119.5          | 128.0               | +7.1%  |

      Decode neutral within thermal noise (50.2-50.4 tok/s, matches post-cross-layer-revert steady state). PPL 6.61 → 6.64 (within drift). Expected was prefill ~115 → ~200-250 (mirroring Q4_K tile-amortization precedent); actual ~115 → ~127 avg. The gap suggests prefill is now bound by something other than Q6_K weight bandwidth — likely Q4_K weights (gemv_q4k_huge for attn_qkv full-attn, ffn_gate/ffn_up_shexp), or attention/conv kernels that don't multi-row at all (DeltaNet conv1d, gdn_chain). Bench file: `docs/bench_results_phase_gemm_q6k.txt`.

  15. **moe_prefill_down_packed_q{4,6}k_v2 (M_TILE=8 SLM amortization for stage-2 MoE down)** — ✅ done 2026-05-02. Audit on item 14's "next bottleneck" pointed at the multi-expert MoE block (32% of pp512). IGC asm dump (Xe2 simd16 / 128 GRF) on the v1 stage-2 kernels (`moe_prefill_down_packed_q4k`, `moe_prefill_down_packed_q6k`) showed **21 load.ugm + 0 load.slm + 0 barriers** per launch — pure DRAM-bandwidth path with no amortization. The `for tk = 0; tk < n_tok; ++tk` outer loop re-streamed each Q-block from DRAM ~22× per launch (avg n_tok/expert ≈ T·K_top/E = 692·8/256). Stage 1 (gate_up_silu) by contrast had 49 load.ugm + 128 load.slm with M_TILE=8 amortization, hence its 4× faster ms/call despite 4× more instructions. New `moe_prefill_down_packed_q{4,6}k_v2` in `src/ops/moe_fused.cpp` mirror `moe_prefill_gate_up_silu_q4k`'s exact SLM pattern (SG_SIZE=16, N_PER_WG=32, M_TILE=8, SLM A_slm[M_TILE × E_ffn] = 8 KiB/WG, cooperative SLM gather of M packed-rows of `h_packed`, inner loop walks `blocks_per_col` ONCE per tile and multiplies each Q-block against M rows from SLM). Compile-time gate `IE_ENABLE_MOE_DOWN_TILE` (default ON); v1 kernels kept callable as fallback.

      **Bench (3 runs, lmhead-q4k variant, head-to-head v1 vs v2 same thermal session):**
      | prompt              | v1 prefill (med) | v2 prefill (med) | Δ      |
      |---------------------|------------------|------------------|--------|
      | short-chat (T=51)   | 130.0            | 130.5            |  +0.4% |
      | long-instruction    | 126.8            | 144.4            | +13.9% |
      | codegen             | 124.9            | 143.7            | +15.1% |
      | math-reasoning      | 124.0            | 146.6            | +18.2% |
      | long-context (T=219)| 131.4            | 156.7            | +19.2% |

      short-chat at T=51 is below the multi-expert threshold (T≥64) and uses the per-token fused decode-style MoE path — v2 kernel doesn't execute, +0.4% is run-to-run thermal noise. Decode neutral 50.6-51.3 tok/s on both paths. PPL on 1791-token corpus: 14.946 (v1) → 14.903 (v2), Δ = −0.04 (PASS, gate ≤ +0.05). Note: the default 255-token built-in PPL sample showed Δ=+0.09 — that's noise on a too-small corpus; PPL stabilizes only beyond ~1k tokens. Bench file: `docs/bench_results_phase_moe_down_tile.txt`.

      **New steady-state prefill (post item-15):** ~144-157 tok/s on the four v2-exercising prompts (long-instruction / codegen / math-reasoning / long-context), up from ~124-131 pre-item-15.  The win scales with avg n_tok/expert: long-instruction T=80 = +13.9% vs long-context T=219 = +19.2%, confirming the amortization lever was real.

      **Architectural takeaway:** stage 2 was the largest single prefill lever (~32% of pp512) and is now ~closer to stage 1's per-call cost.  Next prefill levers (in priority order): (a) moe_routed stage 1 XMX rewrite — already SLM-amortized but still scalar; PLAN.md item 11's 2-SG attempt failed at -11-13%, needs wider 4-SG-per-WG design; (b) moe_shared (Q4_K gate/up + Q6_K down for shared expert) — could potentially gain from the same M_TILE pattern but smaller absolute target (40 calls/forward); (c) attn_full_block — small (~8% of prefill), needs profiling to see if gemv_q4k_huge XMX dispatch hits or falls back.

  16. **FA-2 super-chunk regression fix (`IE_FA2_TARGET_SUPER` 12 → 64)** — ✅ done 2026-05-02 (commit `c7a581b`). Audit/diagnosis caught a long-ctx decode regression introduced by `5be3385`'s `TARGET_SUPER=12` adaptive constant.  At ctx≥16k, `TARGET_SUPER=12` produced exactly 24 super-WGs total (1 per Xe-core on B70's 24 cores) in a single deep wave with no pipeline hiding for per-chunk launch/barrier overhead.  The pre-`5be3385` fixed `CHUNKS_PER_WG=8` gave 5.3 waves at 32k.  The original commit's bench only covered short prompts (51-219 tok), so the long-ctx regression went undiagnosed.

      Bisected by rebuilding at three commits, identical hardware, INT8 KV:
      | commit | 16k tok/s | 32k tok/s |
      |---|---:|---:|
      | `9088ffd` (pre-regression) | 29.7 | 24.3 |
      | `5be3385` (regression cause) | 23.9 | 16.4 |
      | `0e0d4c2` (HEAD before fix) | 25.4 | 17.4 |
      | HEAD + fix (`c7a581b`) | **35.2** | **29.3** |

      Fix: single-constant change in `src/ops/attention.cpp`, gated as `IE_FA2_TARGET_SUPER` (default 64).  At ctx≤1k behavior is unchanged (`CHUNKS_PER_WG=1` either way); at ctx=32k restores `CHUNKS_PER_WG=8` matching the pre-regression baseline.  Section profile @ 32k: `attn_full_block` 40.6 → 17.0 ms/step (−58%).  PPL on 1791-tok corpus: 14.90 → 14.95 (within run-to-run noise; math identical, only WG layout differs).  Bench file: `docs/bench_results_phase_fa2_target_super_fix.txt`.

      **Net long-ctx decode after fix:** **+68% at 32k, +37% at 16k**, vs the un-diagnosed pre-fix HEAD.  Beats even the pre-regression `9088ffd` baseline by +20% at 32k (because the other May-1 commits — lm_head Q4_K, ssm/shared-expert fusions, residual+RMS fusion, gemv_q6k SG-coop, moe_dec_down_q6k SG-coop — are real wins that compound with the FA-2 fix).

  17. **Q4_K SG-coop on `moe_decode_down_q4k`** — ❌ tried 2026-05-02, **REVERTED** (commit `1565be6`). Hypothesis: since the prior `gemv_q4_K` SG-coop revert was due to its existing `vec<uint8_t,16>` per-lane loads (already coalesced), but `moe_decode_down_q4k` uses byte-by-byte `qs[qs_off + i]` in a `#pragma unroll` loop, the latter might still benefit (same pattern that won 50% on Q6_K). Wrong: 0.4-1.7% regression at every ctx (1k/4k/16k/32k).  IGC's loop unroller appears to coalesce 16-iter contiguous byte reads into the same block transaction the explicit `vec<16>` form produces; the `select_from_group` shuffle cost dominates. **Lesson refined:** SG-coop helps only when per-lane access is genuinely STRIDED (e.g. Q6_K's ql+qh+scales mixing); contiguous byte-by-byte unrolled loops are already coalesced by the compiler.  Bench file: `docs/bench_results_phase_moe_dec_down_q4k_sg_coop_REVERTED.txt`.

Where SYCL JIT codegen leaves perf on the floor, port the kernel to ESIMD (escape hatch) — likely candidates: GEMV-Q4_K, FA-2 inner loop, DeltaNet scan.

- **Gate (the big one): ≥ 50 tok/s decode @ 32k ctx, Q4_K_M, batch=1. Stretch ≥ 100 tok/s.** CI bench fails the build if regression > 5%.

#### 9d. Gate status + confidence ratings (updated 2026-05-02)

| Target | TG measurement (2026-05-02 post-fix) | PP measurement | Confidence |
|---|---|---|---|
| **≥50 tok/s decode @ 32k** (gate)         | **29.3** tok/s @ 32k INT8 KV (↑ from 17.4 pre-FA2-fix; would-be 22.0 of pre-`5be3385` already exceeded) | —               | **gap is now 1.7×, not 2.9×.** 50 tok/s @ 32k still requires algorithmic wins (paged attention, fp8 KV, MTP) — research-track for v1.5. |
| **≥100 tok/s decode @ 32k** (stretch)     | 29.3 (3.4× to go)    | —               | **very low** — would need MTP head (1.5-2×) plus structural attn rewrite. Multi-week effort. |
| **≥150 tok/s decode @ 32k** (upper)       | 29.3 (5.1× to go)    | —               | **very low** — bandwidth-roofline at 80% peak is 225 tok/s ceiling. 150 = 67% of ceiling; needs INT4 KV + MTP combined. |
| **≥50 tok/s decode @ small/mid ctx**      | **51.0 @ 1k, 47.7 @ 4k** | —             | **PASSES at ctx≤1k**, 5% short at ctx=4k.  Original "bare-minimum" gate met at the ctx where real chat sessions live. |
| **Match llama.cpp Vulkan TG @ small ctx** | **51 tok/s** @ ctx=1, was 33.8 @ 9d | —      | **EXCEEDED** — now ~140% of llama.cpp Vulkan's 36.3 at ctx=1. |
| **PP @ T=4k**                             | —                    | **130-157 tok/s** (real prompts) | post-item-15 (M_TILE=8 SLM amortization on stage-2 down).  llama.cpp Vulkan PP=512 is 865 tok/s (5-7× ahead — gap is structural via `cl_intel_subgroup_2d_block_io`). |

#### 9e. apples-to-apples vs llama.cpp Vulkan on B70 (updated 2026-05-02; **superseded by `docs/perf_baseline_2026-06-09.md`**)

**2026-06-09 verified head-to-head (fp16 KV, llama.cpp Vulkan b8902 pinned to B70):** engine tg128 = **46.8 tok/s** vs llama.cpp **39.63** → **118%, decode won outright**. PP@512 = **202.9** vs **885.9** → **23%, sole remaining gap**. PPL 6.54 (baseline held). Older numbers below kept for history.

Same B70, same `Qwen3.6-35B-A3B-Q4_K_M.gguf`. Llama.cpp Vulkan via `llama-bench`. TG values for ours from `ie-bench --fastforward --int8-kv` at the corresponding ctx.

| ctx   | llama.cpp Vk TG | ours TG (pre-fix) | ours TG (post-fix) | ratio (post-fix) | notes |
|-------|----------------|-------------------|--------------------|------------------|-------|
| 1     | 36.3           | 33.8              | **53.0**           | **146%**         | exceeded |
| 1024  | ~34.7 (derived)| 33.3              | **51.0**           | **147%**         | exceeded |
| 4096  | ~30.7 (derived)| 30.2              | **47.7**           | **155%**         | exceeded |
| 16k   | (not measured) | 26.4              | **35.2**           | —                | strong |
| 32k   | (not measured) | 22.0              | **29.3**           | —                | strong |

| metric            | llama.cpp Vk | ours (pre-item-14) | ours (post-item-15) | ratio (current) |
|-------------------|--------------|---------------------|----------------------|-----------------|
| PP @ T=512        | **865 t/s**  | ~89 t/s             | ~150 t/s             | 17%             |
| PP @ T=1024       | ~865 (sustained) | ~87 t/s         | ~140 t/s             | 16%             |

**Reframe (post 2026-05-02 fixes):** TG now **exceeds** llama.cpp Vulkan at every measured ctx (was matching, now ~140-155% across the board).  PP is still 5-7× behind because llama.cpp's GEMM uses `cl_intel_subgroup_2d_block_io` and we don't (HW-blocked on BMG-G31 stepping C0).  The original 50 tok/s @ 32k gate is now 1.7× away (was 2.9×) — closeable only via algorithmic structural changes (paged attention, fp8 KV, MTP), not in-engine kernel tuning.

**v1 release criteria — REVISED 2026-05-02 after audit pushback**:

The earlier "v1 shipped" claim was premature.  Decode is genuinely strong (146-154% of llama.cpp Vulkan at chat-relevant ctx) but **prefill at 158 tok/s pp512 vs llama.cpp's 885 = 18%** makes the engine unusable for any prompt longer than ~50 tokens.  At current pp512, a 500-token user prompt costs 3.2s of prefill versus 0.58s on llama.cpp on the same hardware — that kills interactive feel and means the engine can't be the daily driver this project was started to build.

**Hard gates (re-run 2026-06-09 release checklist — see RELEASE.md):**
- TG @ small/mid ctx ≥ 95% of llama.cpp Vulkan ✅ **MET** (117.7%: 46.8 vs 39.75, same-hour)
- ≥50 tok/s decode @ small ctx ✅ **MET at full quality** (52.4 suite / 53.3 INT8 @ ctx=1, v1.1 staged Q6_K GEMV `35a85ea`, PPL 6.54 unchanged; turbo Q4_K-lmhead GGUF: 54.3/56.6 at PPL 6.64)
- Correctness: argmax bit-exact through 40 layers ✅ **MET** (Phase 9; PPL 6.54 reproduced 6× on ship day)
- PPL stable on multi-token corpus ✅ **MET** (6.54 bit-stable through all ship-day optimizations)
- INT8 KV cache validated ✅ **MET** (0.35% drift, validate_int8_kv.sh, 2026-06-09)
- Build clean ✅ **MET** (incl. fresh-clone build; unit tests 7/7 after stale-test fixes)
- **PP @ T=512 ≥ 50% of llama.cpp Vulkan (≥ 443) ✅ EXCEEDED** — 899.7 = 101.6% of llama.cpp itself (2026-06-09)

**Deferred to v1.5/v2.0 (acceptable to ship without):**
- ≥50 tok/s decode @ ctx=32k (algorithmic — paged attention, fp8 KV, or MTP)
- Long-ctx server features (multi-turn KV reuse + defrag — Phase 11)
- Stretch decode targets ≥100 tok/s @ 32k

#### v1 SHIPPED 2026-06-09 — see RELEASE.md

The prefill gate cleared at 101.6% of llama.cpp Vulkan itself (899.7 vs
885.0 pp512, same-hour head-to-head) after the E1–E5 optimization day
(`docs/prefill_attack_plan_2026-06-09.md`).  The 2026-05-02 premature draft
(RELEASE_DRAFT.md) is superseded by the new RELEASE.md, written from the
re-run checklist with one honest flag: absolute decode @ ctx=1 measures 48.5
INT8-KV against the round-50 target (relative gate exceeded at 122%).

**Path to closing the prefill gap (2-3 weeks of focused work):**
1. **Disambiguate ESIMD HW state on BMG-G31 C0** (1 hour smoke test).  Project memory file says cooperative `__spirv_SubgroupBlockReadINTEL` works; the existing `gemm_q4k_esimd.cpp` source (May 1) says all subgroup block-read forms tested so far hang.  Need a new controlled test of cooperative-subgroup 1D reads (NOT the per-lane stride-1 form IGC coalesces into the broken transposed form).
2. **If 1D works:** ESIMD mixed-dtype GEMM via `xmx::dpas` (~1 week).  PDF P0 lever, projected +50-80% prefill closure.
3. **Independent of ESIMD:** Tiled flash-attention for T>1 prefill (~3-5 days).  Replaces the current per-token loop of split-K decode FA-2 with a single tiled-attention kernel mirroring llama.cpp's `flash_attn_ext`.  PDF P1 lever, projected +50-100% prefill on attention-heavy prompts.
4. **If ESIMD path fails on G31:** oneDNN `f16:u4` pivot (~3 weeks).  Requires Q4_K_M → groupwise-INT4 repack at load + PPL re-validation.

### Phase 9 — TurboQuant decision

**TurboQuant is NOT in the plan.** v1 quants (Phase 10) are: Q4_K_M (current daily-driver), Q5_K, Q6_K, Q8_0, AWQ INT4, GPTQ INT4, AutoRound INT4/INT8. TurboQuant (a recent rotation-based PTQ technique) lives outside the bytestream-compatible llama.cpp ecosystem — adopting it would require its own dequant kernel, calibration pipeline, and integration with the model loader. **Recommendation: defer to Phase 10+ unless quality issues emerge with Q4_K_M;** the v1 path doesn't depend on it and the integration cost is comparable to adding GPTQ.

### Phase 9 — Bandwidth target clarification

The B70 Pro spec is 608 GB/s peak HBM/GDDR6. Phase 0 set the **effective sustained** assumption at 70-80% of peak = 425-486 GB/s — this is what cutlass-sycl / sycl-tla actually achieve on Xe2 with 2D block load. The plan's 365 GB/s gate (used in Phase 3) is conservative, sized at 60% of peak so that even a moderately-tuned kernel passes. **Both are correct numbers — gate is realistic floor, ceiling is 80% peak.** The roofline table in §1 of this plan uses the 70-80% effective BW for tok/s ceilings.

### Phase 10 — Other quants (5 days)
- AWQ INT4 loader (safetensors): `qweight` packed along N, `[0,2,4,6,1,3,5,7]` lane permutation, group=128, no g_idx, no +1 zero offset.
- GPTQ INT4 loader: `qweight` packed along K, branch on `quantize_config.json` for V1 (+1 zero) vs V2 (plain). Honor `desc_act`.
- AutoRound INT4 loader: read `packing_format` from quant_config; route to GPTQ or AWQ kernel; AutoRound INT8 stored unpacked.
- Each format gets a dedicated dequant kernel under `kernels/sycl/`.
- **Gate:** each format passes the Phase 8 correctness gauntlet on a converted Qwen3.6-35B-A3B; tok/s within 10% of Q4_K_M.

### Phase 11 — Server + polish (5 days)
- OpenAI-compatible HTTP: `/v1/models`, `/v1/chat/completions` (streaming SSE).
- Long-ctx YaRN scaling for >256k.
- Multiple concurrent chat sessions (single-user, multiple contexts) — separate KV/DeltaNet caches per session.
- README, model-conversion docs, `scripts/run_perf_gauntlet.sh`.
- **Gate:** real chat session with the Qwen Chat reference UI proxied to our server. Eval harness runs perplexity on wikitext-2 within 1% of llama.cpp's number on the same GGUF.

  **Validation tooling status (2026-04-27):**
  - ✅ `tools/ie_perplexity.cpp` — streaming-decode PPL tool. Reset KV/DN, then for each input token i: `forward(ids[i:i+1], T=1, start_pos=i)` → host-copy logits → `nll = -log_softmax(logits)[ids[i+1]]`. PPL = exp(mean NLL).  Built-in 200-token classic-prose sample so the tool produces a number with no corpus arg.
  - Usage:
    - Sanity: `ie-perplexity` — built-in sample, fp16 KV.
    - INT8 sanity: `ie-perplexity --int8-kv` — same sample, INT8 KV cache. Compare PPL delta to fp16; target < 0.5% relative drift.
    - Production: `ie-perplexity --text wikitext-2.txt --max-tokens 1024 [--int8-kv]`.
    - Cross-engine: run `llama-perplexity` on the same GGUF + same first 1024 wikitext-2 tokens; ours within 1% is the Phase 11 ship gate.
  - Output is one TSV trailer line `kv_mode\ttokens\tavg_nll\tperplexity` for scripted comparison.

### Phase 2 backlog (post-50 tok/s)
- MTP head (1.5–2× decode speedup via speculative parallel-token prediction).
- INT8 KV cache (halves the @128k bandwidth bill).
- Vision encoder (`vision_tower`) for multimodal.
- FP8 weights (B70 has no FP8 dpas — would have to emulate; skip until Xe3).
- Multi-stream / batched serving.
- Multi-GPU via PCIe P2P.

---

## 6. Kernel inventory (with target perf)

| Kernel | Where | Phase | Target |
|---|---|---|---|
| `gemm_fp16_xmx` | prefill projections | 3 | ≥70% of 183 TFLOPS |
| `gemv_w4a16_q4k_xmx` | decode projections, MoE down/up | 3 | ≥60% of 608 GB/s |
| `gemv_w5a16_q5k_xmx` | decode projections | 10 | ≥55% of 608 GB/s |
| `gemv_w6a16_q6k_xmx` | decode projections | 10 | ≥55% of 608 GB/s |
| `gemv_w8a16_q8_0_xmx` | decode projections | 10 | ≥70% of 608 GB/s |
| `gemv_w4a16_awq_xmx` | AWQ models | 10 | ≥55% of 608 GB/s |
| `gemv_w4a16_gptq_xmx` | GPTQ models | 10 | ≥55% of 608 GB/s |
| `dequant_q4k → fp16` | as fallback | 2 | ≥350 GB/s output |
| `flash_attn_2_xmx` | full-attn layers | 4 | ≥40% of peak (FA-2 @ Xe2 hits ~78% in sycl-tla; we aim lower while learning) |
| `deltanet_recurrent_decode` | DeltaNet decode | 5 | KV-equivalent: ≤2 ms total for 30 layers |
| `deltanet_chunkwise_prefill` | DeltaNet prefill | 5 | ≥30% of 183 TFLOPS |
| `moe_topk_router` | MoE | 6 | <100 µs/layer |
| `moe_sparse_decode` | 9 expert GEMVs | 6 | ≥60% of 608 GB/s on the active-expert footprint |
| `rmsnorm`, `silu`, `swiglu`, `rope_partial`, `residual_add` | everywhere | 3 | bandwidth-bound, ≥80% of peak |
| `sampling_top_p_top_k` | post-LMHead | 7 | <500 µs at vocab=248k |

---

## 7. Validation strategy

**Unit (per kernel):**
- Random tensor + reference impl in C++ (transcribed from `transformers` / `ggml-quants.c`).
- Tolerances: 0 ULP for byte-exact ops; 1e-3 abs for fp16 reductions; 1e-2 abs across full layers (bf16 vs fp16).

**Per-block correctness:**
- Hook HF transformers via `tools/dump_hf_activations.py`, capture: embed-out, layer-{0,3,4,7,39}-attn-out, layer-{0,3}-deltanet-state, final-hidden, lm_head-logits-top32.
- Diff our engine's outputs against fixtures at each layer boundary.

**Whole-model:**
- Fixed input `[9707, 11, 1879]` → argmax-token-id matches; top-5 ordering matches.
- ChatML "What is 2+2?" → response contains "4" and emits `<|im_end|>`.
- Perplexity on wikitext-2 within 1% of llama.cpp on same GGUF.

**Performance gates (CI-failing):**
- Phase 0: GEMM ≥ 50% peak.
- Phase 9: decode ≥ 50 tok/s @ 32k Q4_K_M.
- Phase 9 ongoing: ≤ 5% regression on the perf-gauntlet baseline.

---

## 8. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **DeltaNet scan numerical correctness drifts from reference** | High | Blocks phase 5–8 | Aggressive per-step state hooks; keep state in fp32; single-step diff harness from day 1 |
| **`joint_matrix` codegen on BMG-G31 has bugs** (e.g. ggml-sycl issues #21893, #21517 are open) | Med | Wastes a week on phantom perf bug | Cross-check every kernel against ESIMD `xmx::dpas` and oneDNN once before optimizing; keep `GGML_SYCL_DISABLE_OPT`-style fallback |
| **2D block load ext not exposed in SYCL on B70** | Med | Forces SLM staging; ~30% perf hit | OpenCL-C kernel via SPIR-V ingestion as fallback; sycl-tla shows the canonical pattern |
| **Tokenizer regex edge cases on multimodal vocab additions (audio/TTS tokens)** | Low | Wrong tokenization for unusual inputs | Lift the exact `pretokenize_regex` from Qwen's tokenizer_config.json; fuzz-test against HF tokenizer |
| **MoE router top-k tie-breaking diverges from PyTorch's `torch.topk`** | Med | Off-by-one expert selection → degraded quality | Match PyTorch's stable-sort semantics; unit-test against transformers fixture |
| **Q4_K block alignment (144 B, not pow2)** breaks vectorized loads | Med | Slow dequant | Pad allocations; use scalar loads for the first/last partial blocks; sycl-tla has the precedent |
| **Partial rotary (factor 0.25) — easy to apply rotation to all 256 dims by mistake** | High (latent bug) | Silent quality drop, hard to detect | Layer-0 attn fixture diff catches this immediately; keep a unit test that asserts `dim ≥ 64` is identity |
| **`attn_output_gate=true` missed during port** | Med | ~5–10% quality drop | Tracked as Quirks Checklist item #5 in research/04 |
| **Chat template `<think>` handling** | Low | Model emits no closing `</think>` | Stop on `<|im_end|>` not `</think>`; verify with smoke test |
| **B70 driver/IGC regressions during the project** | Med | Random breakage on `apt upgrade` | Pin driver versions in `scripts/setup_toolchain.sh`; use intel-graphics PPA frozen channel |
| **MTP weights present but unused — wasted VRAM** | Low | ~few GB wasted | Loader skips `model.mtp.*` tensors by default |
| **Single-user MoE expert dispatch is unusual** (most engines optimize for batched) | Med | Subpar GEMV utilization for tiny experts (FFN=512) | Pre-fuse 9 active experts into one batched GEMV-3D; or accept the cost since FFN is small relative to attention |

---

## 9. What we explicitly do NOT solve in v1

- Multi-GPU.
- Multi-user batching, paged attention, prefix caching.
- Vision encoder, audio encoder, TTS.
- FP8 weights (no dpas support on Xe2).
- Speculative decoding (MTP head ignored at load time).
- Training, fine-tuning, LoRA.
- Windows/Mac. Linux only.

---

## 10. First five concrete commits, in order

1. `chore: project skeleton, CMake + SYCL nightly toolchain detection`
2. `feat(core): tensor type + USM device allocator`
3. `feat(loaders): mmap GGUF reader + tensor info enumeration`
4. `feat(kernels): joint_matrix FP16 GEMM with 2D block load`
5. `feat(kernels): Q4_K dequant kernel + golden vector test`

If you sit down tomorrow and execute those five in order, you'll be ~Phase 2 complete by Friday and the rest of the plan is a derivative of Phase 0.

---

## Appendix A — Pinned versions (lock these in `scripts/setup_toolchain.sh`)

- intel/llvm: nightly-2026-03-22 (or later that passes a smoke `joint_matrix` BMG test)
- compute-runtime: 26.05.37020.3 minimum, 26.14.37833.4 recommended
- IGC: 2.28.4 minimum, 2.32.7 recommended
- Linux kernel: 6.14+ (intel-graphics PPA channel)
- oneDNN: 3.8 (for SDPA + INT4 weight decompression reference)
- sycl-tla: v0.8 (BMG-G31 examples for reference)

## Appendix B — Bookmarks (cited heavily during implementation)

- `research/01_hardware.md` — spec, ISA, tile shapes, pitfalls
- `research/02_programming_stack.md` — kernel idioms, AOT flags, references
- `research/03_quant_formats.md` — exact byte layouts (the daily reference)
- `research/04_qwen_arch.md` — model spec, quirks checklist, KV math
- [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [HF transformers `modeling_qwen3_5_moe.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)
- [llama.cpp `ggml-quants.c`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)
- [intel/sycl-tla v0.8 BMG examples](https://github.com/intel/sycl-tla)
- [oneDNN matmul guide](https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html)
- [ChipsAndCheese Battlemage architecture](https://chipsandcheese.com/p/intels-battlemage-architecture)
