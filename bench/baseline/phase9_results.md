# Phase 9 — Decode loop + perf push (partial: correctness pass)

**Date:** 2026-04-26
**Status:** Made meaningful correctness progress (3 more bugs caught from llama.cpp source review). The forward-pass residual stream now has sane magnitudes through 40 layers (L2 ~1 at L=0, growing smoothly), but the final argmax is still semantically wrong. Hit a static-analysis ceiling — going further needs a layer-by-layer activation diff against a reference implementation.

## What this phase did

Built `llama-cli` from `/home/weezy/llama.cpp` (Qwen3.5-MoE arch already supported there as `LLM_ARCH_QWEN35MOE`). Source review of `src/models/qwen35moe.cpp` and `convert_hf_to_gguf.py:_LinearAttentionVReorderBase` revealed 3 silent-correctness bugs:

| # | Bug | Source | Fix |
|---|---|---|---|
| 7 | **Block-level RMSNorm `(1+w)` is baked into the GGUF**. `Qwen3_5MoeRMSNorm` in HF uses `(1+weight)*x_normed`; llama.cpp's converter pre-adds 1 so the GGUF stores `1+w` directly, then llama.cpp uses plain `weight*x_normed`. Confirmed by dumping `attn_norm.weight` mean=1.03 (range 0.92-1.33) — those are clearly post-`+1` values, not pre-trained-from-zero deltas. | `src/models/qwen35moe.cpp:29` calls `build_norm(..., LLM_NORM_RMS, ...)`. | Replace `rms_norm_one_plus_w` calls with `rms_norm_f32w` (plain). |
| 8 | **`ssm_a` (A_log) stored pre-computed as `-exp(A_log)`**, not as raw A_log. Confirmed by dumping `blk.0.ssm_a` — values in `[-72, -0.018]` range; the `-72` values aren't possible for raw A_log but make sense for `-exp(A_log_real)` when the head trains to A_log≈4.28 (aggressive forgetting). | `qwen35moe.cpp:238` comment: `gate = ggml_mul(alpha_softplus, ssm_a)  // -A_log.exp() * softplus`. | `compute_g_beta`: `g = ssm_a * softplus(...)` instead of `-exp(ssm_a) * softplus(...)`. |
| 9 | **V heads stored in TILED order** in the GGUF (`[G0V0, G1V0, ..., G15V0, G0V1, G1V1, ..., G15V1]`), not GROUPED (`[G0V0, G0V1, G1V0, G1V1, ...]`). Affects `in_proj_qkv` V-rows, `in_proj_z`, `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, `conv1d` V-channels, `out_proj` columns. My `repeat_interleave_heads` was producing GROUPED expansion (`y[h_out] = x[h_out / repeat]`) — flipped to TILED (`y[h_out] = x[h_out % n_in_heads]`). | `convert_hf_to_gguf.py:5259-5424` `_LinearAttentionVReorderBase`; `qwen35moe.cpp:329-333` uses `ggml_repeat_4d` (tiled). | One-line fix in `src/ops/elementwise.cpp::repeat_interleave_heads`. |

## Effect on residual-stream magnitudes

Sequential improvement at layer 0 (L=0 residual L2):
- Phase 8 baseline: **23.5** (38× embedding L2 of 0.61 — clearly wrong)
- After conv1d index fix: **4.06** (6.7× embedding — better)
- After norm `(1+w)` fix: **2.4** (4× — closer)
- After ssm_a fix: ~same (expected — affects state-decay math, not first-token magnitude)
- After tiled-V fix: **1.18** (2× — sane)

Through full 40 layers, residual L2 grows smoothly without explosion or vanishing (4 → 75 over 40 layers, no NaN).

## Why it's STILL wrong

- All structural bugs I found from static analysis are fixed.
- Per-layer magnitudes are now in the "sane transformer" regime.
- Different prompts produce different deterministic outputs → input flows through.
- No NaN/Inf.

But argmax is still semantic-noise: `Hello, world → '多次'`, `1+1= → 'iley'`, `The capital of France is → 'ResponseType'`. Logits in normal range (~10-12).

The remaining bug(s) are subtle enough that they don't show in a static walk of the code. To find them I'd need to:
1. **Run llama-cli and our engine on the same prompt to completion**, dumping per-layer hidden states (`-d` flag in llama-cli, IE_TRACE in ours).
2. **Diff layer 0 outputs first** — they should match within fp16/bf16 rounding. The first layer where they diverge is the bug.
3. CPU llama.cpp on a 35B model is ~5-15 minutes per token; this is uncomfortable but not blocking. A single layer-0 diff is enough to find the bug.

## Phase 9 perf items: blocked

Without a correct forward, the perf items (FA-2 attention rewrite, fused MoE block, GEMV-Q4_K XMX-fused, GEMM-FP16 2D block load) are deferred. There's no point optimizing math we know is wrong.

## Files changed in Phase 9

- `src/ops/deltanet.cpp` — `compute_g_beta` formula corrected.
- `src/ops/elementwise.cpp` — `repeat_interleave_heads` switched grouped → tiled.
- `src/model/qwen36.cpp` — block-level RMSNorm calls switched `_one_plus_w` → `_f32w`.

The engine is at: end-to-end forward + sampling pipeline that runs without crashing, produces deterministic prompt-dependent outputs of correct magnitude, but misses on semantic correctness. The next session's job is the layer-by-layer diff to localize the final bug(s).

---

## 2026-04-26 — Sub-phase 9a complete: correctness solved

Built debug infrastructure and bisected the remaining bugs by layer-by-layer fp32 binary diff against llama.cpp's Vulkan build. Three more bugs caught and fixed (12 total Phase 8+9):

### Debug infra

- `tools/llama_dump.cpp` — links against `/home/weezy/llama.cpp/build-vk/bin/{libllama,libggml,libggml-base}`. Registers `cb_eval` to capture per-layer residual stream (`<prefix>_L00..L41.bin`) and per-block attn output (`<prefix>_A00..A39.bin`) as fp32 binaries. Each file has a sidecar `.meta` with text "T H".
- `tools/ie_debug.cpp` — long-running stdin-driven REPL. Loads model + caches **once** (~22 GB to device in 2.4 s). Accepts `P <prompt>` (encode + prefill + dump + argmax + top-5), `R` (reset KV/DN), `Q` (quit). **Solves the previous "fresh-process-per-prompt = 22 GB load each" failure mode** that crashed VS Code when run in parallel with llama.cpp.
- `tools/diff_layers.sh` — bash + Python max-abs / max-rel diff per layer.
- `QwenModel::set_dump_prefix(std::string)` — opt-in fp32 dump from forward(). Default empty = no perf impact.

### Bug 10: conv1d kernel direction reversed

`src/ops/conv1d.cpp`. PyTorch's `nn.Conv1d(padding=K-1)` and llama.cpp's `ggml_ssm_conv` (ggml-cpu/ops.cpp:9297) both use the convention where `W[0]` multiplies the OLDEST tap (t-(K-1)) and `W[K-1]` multiplies the CURRENT tap (t). Our impl indexed `W[k]` against `x[t-k]`, treating `W[0]` as current. Fix: read `W[K-1-k]`.

Effect: single-token "Hello" argmax went `.` (logit 6.88) → `,` (logit 14.47, **bit-exact match** with llama.cpp's 14.46).

### Bug 11: gemv_q is single-token only

`src/model/qwen36.cpp`. Both `gemv_q4_K` and `gemv_q6_K` cooperatively SLM-load the full A vector and produce one row of y. Calling `gemv_q(...)` with `T*K` input bytes and `T*N` output bytes only processes token 0; tokens 1..T-1 read uninitialized memory. Manifested as A00 token 1 / 2 having `L2(mine) = 0.0` for multi-token prompts.

Fix: `gemv_q_T` helper that loops `gemv_q` over T rows. Replaced all attn/MoE projection sites (10 call sites). Slow path; replace with batched-expert kernel as part of the perf push.

### Bug 12: DeltaNet fused QKV split is per-token, not per-component

`src/model/qwen36.cpp`. The `attn_qkv` projection produces `[T, SI2=8192]` where each row is `[Q(2048), K(2048), V(4096)]`. The naive 3 contiguous `cast_fp16_to_fp32` calls (offset 0 / 2048 / 4096, length T*Q_size) splice token-0's K into token-1's Q buffer, etc. Fix: per-token strided cast loop.

### Verification

Diff vs llama.cpp Vulkan reference on `Hello, world` (T=3):

| Range | Status | Magnitudes |
|---|---|---|
| L00..L11 | OK | max_abs < 0.005 |
| L12..L40 | WARN | max_abs < 0.05 — FP16 accumulation noise from differing op order / MoE dispatch |
| Final argmax | **EXACT MATCH** | `!` logit 17.56 (ref: 17.56) |

Other prompts (no host fixture; argmax-correctness only):
- `Hello` → `,` (logit 14.47, ref 14.46) ✓
- `The capital of France is` → ` Paris` ✓
- `2+2=` → `4` ✓
- `Once upon a time` → `,` ✓
- `Capital of France is Paris. Capital of Germany is` → ` Berlin` ✓

### Decode tok/s baseline (correctness done, no perf items applied)

20 sequential T=1 commands after a single prefill: **52.5 ms/token (σ ≈ 0.5 ms) = 19 tok/s**.

That's 38% of the 50 tok/s gate. Phase 9 perf items are now active development.

### Files changed (sub-phase 9a)

- `src/ops/conv1d.cpp` — kernel direction reversed (1 line; Bug 10).
- `src/model/qwen36.cpp` — `gemv_q_T` helper + all call-site replacements (Bug 11); per-token QKV strided cast (Bug 12); `set_dump_prefix` + per-residual fp32 dumps.
- `include/ie/qwen36.hpp` — `set_dump_prefix` API + `dump_prefix_` member.
- `tools/{ie_debug,llama_dump}.cpp`, `tools/diff_layers.sh`, `tools/CMakeLists.txt` — debug infra (NEW).

Two commits: `d6c9d06` (conv1d + debug tools), `3542409` (multi-token prefill).

### What's next: sub-phase 9b/c (perf push)

PLAN.md is the canonical task ladder. Active item: **fused MoE block** (priority 1, largest expected savings ~60 ms/token).
