# Phase 8 — End-to-end forward pass

**Date:** 2026-04-26
**Hardware:** Intel Arc Pro B70 (BMG-G31)
**Status:** ⚠ **integration runs end-to-end without crashing, but model output is not yet semantically correct.** Pipeline is built and operational; weight load works; ~22 GB Q4_K_M loads to device in 2.4 s; prefill ~31 tok/s; per-layer residual norms grow smoothly through all 40 layers without NaN. Specific math bugs caught and fixed during this phase, but at least one more remains. Phase 9 priority: bisect the remaining bug(s) (likely in DeltaNet recurrence wiring, full-attn pipeline, or mRoPE) — the infrastructure is solid.

## What works

- **Model loader** loads 733 tensors from the GGUF to device, dispatches per-tensor dtype (Q4_K vs Q6_K mixed across layers in this Q4_K_M quant).
- **Workspace allocator** sizes to max_T tokens and reuses across calls.
- **40-layer forward pipeline** runs:
  - Embedding lookup (Q4_K dequant per token, **bit-exact verified** against Phase 2 reference)
  - 30 DeltaNet layers + 10 full-attention layers (alternating per `i % 4 == 3`)
  - Each layer: `attn_norm(1+w)` → attn block → residual → `post_attn_norm(1+w)` → MoE block (router + 8 routed experts + 1 shared expert) → residual
  - Final norm + lm_head GEMV
- **No NaN / Inf** in 40 layers.
- **Per-layer residual L2** grows smoothly (4 → 75 over 40 layers); within-layer activation magnitudes are reasonable.
- **Different prompts produce different argmax tokens**, confirming the input flows through the model.
- All ops invoked are individually validated by Phase 1-7 tests (max_abs ≤ 1e-2 vs CPU references).

## What's wrong

- **Output is semantic-quality gibberish**. `Hello, world` argmax = ` réc`. `The capital of France` argmax = `_cur`. These are deterministic and prompt-dependent, but not the kind of continuations a sane Qwen3.6 produces. Logit values (~14-15) are within reasonable range — no overflow.

## Bugs found and fixed during Phase 8

| Bug | Symptom | Fix |
|---|---|---|
| `token_embd.weight` and `output.weight` were Q4_K and Q6_K respectively (Q4_K_M quant mix), but loader hardcoded Q6_K. | Load failed at first tensor. | Detect dtype, dispatch `embedding_lookup_q4k` vs `embedding_lookup_q6k`; same for lm_head GEMV. Wrote `embedding_lookup_q4k` kernel; verified bit-exact (max_abs=1e-5) vs CPU dequant reference for token id 1814. |
| `ffn_down_exps` is per-layer Q4_K-or-Q6_K (Q4_K_M quant heuristic — not all layers Q6_K). | Load failed at layer 5. | Refactored to `QuantPtr {ptr, dtype}`; added `gemv_q(...)` dispatch wrapper; per-expert byte stride computed from layer's actual dtype. |
| `ssm_conv1d.weight` and `ssm_norm.weight` are FP32, but kernels expect FP16. | Type mismatch. | Cast to FP16 at load time; store both in `LayerWeights`. |
| **Per-layer Q+gate split was wrong.** GGUF `attn_q.weight [2048, 8192]` has the gate fold-in interleaved per-head: `[h0_q(256), h0_gate(256), h1_q(256), h1_gate(256), ...]`, NOT first-half-then-second-half. | Web-fetched HF transformers source confirmed `qk.view(*input_shape, n_heads, head_dim*2).chunk(2, dim=-1)`. | Wrote `split_q_gate_per_head` kernel. |
| **`std::swap(ws_q_fp32_, ws_recurrence_out_)` permanently aliased workspace pointers across layers.** | The DeltaNet path swapped pointers between buffers of different sizes; subsequent layers read garbage. | Added dedicated `ws_q_fp32_pre_` and `ws_k_fp32_pre_` for the 16-head pre-GQA buffers; expand into existing 32-head buffers; no swap. |
| **`ssm_conv1d.weight` shape `[4, 8192]` has kernel-position as the LEADING (contiguous) dim.** My conv1d kernel was reading element `(k, c)` at offset `k*channels + c` — wrong. Correct offset is `k + c*kernel`. | Layer 0 residual L2 was 23 (38× embedding L2); after fix dropped to ~4 (sane). | One-line index fix in `src/ops/conv1d.cpp`. |

## What's likely still wrong

- **Full-attention output** — even after Q/gate fix, the residual stream after layer 3 (first full-attn) shows only a small jump, while DeltaNet contributions keep growing. This is *plausibly* correct (full-attn is well-conditioned), but worth a per-layer correctness diff.
- **mRoPE** — research/04 says M-RoPE collapses to 1D RoPE for text-only inputs **if you feed identical position IDs across the 3 sections**. My implementation uses standard 1D RoPE; this should be equivalent for text but worth verifying against HF's `apply_multimodal_rotary_pos_emb` directly.
- **Per-layer norm weight magnitudes**: dumped `attn_norm.weight` mean=1.03 (range 0.92–1.33). With `(1+w)` formula → multiplier ~2.03. This is the trained value; the model has been trained around this. Plausible but worth cross-checking against another reference forward.
- **Conv1d state initialization on first prefill**: per research/05 §11 the conv state should be zero-padded on the LEFT to length `kernel_size`, so for prefill of T tokens with T ≥ kernel, only the last `kernel-1` tokens of x become the new state. My kernel handles T ≥ kernel correctly. For T < kernel (shouldn't happen for `Hello, world` T=3 < 4 — OOPS that's a case!) the state-update logic is fragile. **Worth specifically checking T=3 case.**

## Code shipped this phase

- `include/ie/qwen36.hpp` + `src/model/qwen36.cpp` — `QwenModel` class with `load`, `ensure_workspace`, `forward` (~600 lines).
- New kernels: `rms_norm_one_plus_w`, `rms_norm_f32w`, `cast_fp16_to_fp32`, `cast_fp32_to_fp16`, `repeat_interleave_heads`, `split_q_gate_per_head`, `embedding_lookup_q4k`, `embedding_lookup_q6k`.
- `tools/forward_test.cpp` (`ie-forward` CLI) — load + prefill + sample argmax + greedy decode loop with prompt configurability.
- `tools/dump_embedding.cpp` (`ie-dump-embedding`) — embedding-lookup correctness vs CPU reference.

## Phase 9 backlog (additions from Phase 8)

In rough priority for getting to 50 tok/s with **correct** output:
1. **Cross-check forward pass against llama.cpp** on the same GGUF. Build llama.cpp's `llama-cli`, run with same prompt, dump per-layer activations. Diff against ours layer-by-layer to localize the remaining bug. ~30 min to build llama.cpp; ~30 min for the diff harness; high information per minute.
2. **mRoPE proper implementation** if needed. Currently treating as 1D RoPE; for text-only this should be correct but verify.
3. **Conv1d T<kernel edge case** — verify by encoding a T=3 prefill twice (once normal, once split as T=2 then T=1) and check the final state matches.
4. **`(1+w)` weight magnitude double-check** — compare a fresh-loaded weight tensor's stats against what HF transformers gets when loading the same model.

The infrastructure is built. Phase 8 doesn't ship a "correct" forward, but it ships a forward that runs and is debuggable.
