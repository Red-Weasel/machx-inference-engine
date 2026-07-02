# Handoff — 2026-06-21 — Gemma-4 MTP self-speculative decode (port in progress)

## ✅ STATUS: COMPLETE (2026-06-21, `a78e292`) — STRICTLY LOSSLESS ON BOTH SIZES, BEATS LLAMA
The full port landed and both gemma4 sizes do **strictly token-lossless** MTP
self-speculative decode that beats llama on B70:
- **31B dense: LOSSLESS, 1.47×** (≈24–25 vs 17 tok/s; llama 20.6).
- **26B-A4B MoE: LOSSLESS, 1.46×** (≈67 vs 46 tok/s; llama ~55).

Tool: `ie-gemma4-spec --gguf <target> --head <mtp-head> [--prompt … --ntok N --K K]`
(K=4 optimal). Heads on disk: 31B `mtp-gemma-4-31B-it-Q8_0.gguf`, 26B
`mtp-gemma-4-26B-A4B-it-Q8_0.gguf` (both in their model dirs). crown 6.4527 bit-exact.

**Commits:** `6f4682a` head forward + loop + read_attention_gemma + batched-verify
proj() + all_logits (0.18×→1.4×); `cb8bfeb` SoA-Q8 draft head (12.4→8.0 ms/round);
`a78e292` per-token verify router → 26B MoE strict lossless.

**Key learnings (banked):**
- Head fwd was CORRECT on the first build — acceptance (~0.76 > llama's 0.61) is the
  signal (a buggy head stays lossless since only verified tokens commit). No op-by-op
  llama diff was needed.
- The perf unlock = wiring `gemv_q4_0_soa_q8_batched` (the de-risk kernel) into the
  gemma4 `proj()` for small T. Without it, verify(T=K) took the slow oneDNN prefill
  path → 0.18×. With it → 1.4×. Also batched the verify lm_head (read token_embd once).
- Verify(T=4) ≈ 1.7× a decode in the FULL forward (the de-risk's "1.2×" was the
  isolated kernel; attention/norms/quantize/lm_head all scale with T). Net ceiling ≈
  accepted/(draft+verify) ≈ 2.7/1.8 ≈ 1.5× — we're at it. Robust across context (BW-bound).
- 26B MoE losslessness: the GPU router's batched `gemv_q_T(T)` differs from the T==1
  decode path → near-tie expert flip → ~1 tok/32 divergence. Fix = per-token (T==1)
  router in the verify range. (Diagnosed: HOST_ROUTER→lossless, NO_FUSED_MOE→still NO.)

**✅ PRODUCTIZED (`892934a`, `f40cc1e`):** `ie run --spec` now works for gemma4.
`Gemma4Model::load_mtp_head` (SoA-Q8 repack of the separate `mtp-*.gguf`) + `mtp_run`
(T=1 head fwd) + `spec_generate` (draft/verify/accept loop, emit() streaming+stops,
rollback implicit) + engine kGemma4 spec-dispatch + CLI `--spec-head` (else auto-finds
the first `mtp-*.gguf` next to the target). Validated on BOTH 31B + 26B (auto-find,
coherent via the product path); default `ie run` byte-unchanged; crown 6.4527, ctest
30/30. 26B needs `IE_GEMMA4_FUSED_MOE=1 IE_GEMMA4_MOE_SOA=1`.

**NEXT (real remaining work):**
1. **Clean-box re-bench** — see `docs/BENCHMARK_METHODOLOGY.md`. The headline pp/tg
   numbers (ours + every comp) must be re-run on a freshly-rebooted idle box vs the
   newest llama build, same session (this box was degraded: gemma pp read 234/585 vs
   the true 367/708). Net spec ratios are box-robust; absolute tok/s is not.
2. **Tune 27B-dense decode** — the one must-have model that loses to llama even with
   `--spec` (decode kernel ALU-bound ~57% BW, headroom to ~80%). Tune AFTER the
   benchmark gives a clean baseline.
3. Settle Coder-30B (`qwen3moe`) standing on the clean re-bench (records conflict).

---
## (original handoff — the plan that was executed) TL;DR
Gemma 4 ships an official 930M MTP draft head (`gemma4-assistant`). It's the lever
to beat llama on gemma-31B decode. The **strategic de-risk PASSED** and **3 of 4
foundational pieces are committed + validated**. Remaining = the EAGLE head forward
+ the draft/verify/accept loop (fully specified below; needs op-by-op GPU validation
against llama). Expected payoff: **~1.4–2× decode → 24–34 tok/s vs llama's 20.6**, lossless.

## Why this is the right bet (de-risk, `a962423`)
- llama.cpp runs the head on B70 at only **20.6 tok/s / 61% accept** (+8% — its Vulkan
  GEMM can't amortize the verify; the "3×" is datacenter marketing).
- BUT our `gemv_q4_0_soa_q8_batched` (BW-optimal int-dot verify) **amortizes on B70**:
  on 31B shapes, verifying **T=4 costs 0.83–1.25× a single decode** (per-token cost
  drops 3.7–5.4× from T=1→T=8). Bit-identical to the decode kernel (lossless gate).
- ⇒ with 61% acceptance, realistic ~1.4–2× → **24–34 tok/s, beating llama.** Worth it.

## Committed pieces
1. **`a962423`** — `gemv_q4_0_soa_q8_batched` verify kernel + `ie-gemv-q4-0-batched-bench`.
2. **`c9f38e7`** — `Gemma4Model::forward(..., hidden_pre_norm)`: optional [T,H] output of
   the pre-output_norm residual = the head's `inp_h`. Default null → crown bit-exact.
3. **`9e34f17`** — `Gemma4AssistantHead` loader (`include/ie/gemma4_assistant.hpp`,
   `src/model/gemma4_assistant.cpp`) + `ie-gemma4-assistant-load-test`. Loads the real
   `mtp-gemma-4-31B-it-Q8_0.gguf` (491 MB, in the 31B model dir): config exact, LOAD OK.

## Exact head spec (validated by the load test)
arch `gemma4-assistant`, weights **Q8_0** + F32 norms. 4 layers, hidden **1024**,
backbone **5376** (=target hidden), ffn 8192, vocab 262144, eps 1e-6, n_q_heads 32.
- L0-2 **SWA**: head_dim 256, 16 KV, n_rot 256, θ 1e4.
- L3 **global**: head_dim 512, 4 KV, n_rot 512, θ 1e6, uses top-level `rope_freqs` [256].
- Per layer: attn_norm, attn_q (wq), attn_q_norm [head_dim], attn_output (wo), no wk/wv,
  post_attention_norm, ffn_norm, ffn_gate/up/down (GeGLU), post_ffw_norm, layer_output_scale [1].
- Top-level: token_embd (Q8_0 [1024,vocab] = head's logits proj, tied), output_norm,
  nextn.pre_projection (Q8_0 [10752,1024]), nextn.post_projection (Q8_0 [1024,5376]).

## Forward (Piece 3) — from `~/llama.cpp/src/models/gemma4-assistant.cpp`
Per draft step, input = (token `t`, target hidden `h` [5376]):
1. `x = target.tok_embd[t] * sqrt(5376)`  (uses the **TARGET's** tok_embd, not the head's).
2. `xh = concat(x, h)` [10752]; `cur = pre_proj · xh` → [1024].
3. for il in 0..3: attn_norm → wq → reshape[head_dim,32] → q_norm(per-head RMS) → RoPE
   (θ/n_rot per layer; global uses rope_freqs) → **attention** (Q only; attends the TARGET
   KV — see below; scale 1.0) → wo → post_attention_norm → +resid → ffn_norm → GeGLU →
   post_ffw_norm → +resid → ×out_scale.
4. output_norm → `logits = head.tok_embd · cur` [vocab] (argmax = draft token);
   `h_next = post_proj · cur` [5376] (the next step's input hidden).

### The attention (the one subtle part)
- **KV sharing** (`llama-model.cpp:2154`): head SWA layers (L0-2) attend the **target's
  L58** KV (SWA, 16×256 — dims match); head global layer (L3) attends the **target's L59**
  KV (global, 4×512 — match). The head writes NO K/V.
- All K draft steps in a burst attend the **same fixed** target `KV[0..pos]` (the last
  verified position) — draft tokens differ only via the `h_next` recurrence, not attention.
- ⇒ need a **read-only attention** (Q·K_cache[0..pos] → softmax → ·V_cache[0..pos]) — adapt
  `full_attention_gemma` to skip the append, or add a small variant. Target KV is already in
  `kcache_[58/59]`/`vcache_[58/59]` from the target forward.

### Kernels Piece 3 needs
- **Q8_0 dense GEMV**: `gemv_q8_0_soa_q8` exists but wants SoA-repacked Q8_0 → repack the
  head's AoS Q8_0 weights at load (cheap, small head), or add a plain AoS Q8_0 gemv.
- The read-only attention variant above.
- Gemma4Model accessors: `kcache(L)`, `vcache(L)`, `token_embd()`/dtype, `n_layers()`,
  `kv_ctx()`, per-layer head_dim/n_kv for L58/L59.

## Loop (Piece 4) — template `tools/ie_qwen35_spec.cpp`
1. Prefill 31B (with `hidden_pre_norm`) → h_last, argmax t0.
2. Draft K tokens: head.draft(h, t) → (logit, h_next); t=argmax(logit); h=h_next. (AR on head.)
3. 31B **batched verify** at the K draft tokens (T=K) → per-position logits + hidden
   (`gemv_q4_0_soa_q8_batched` is the verify GEMV; the gemma4 forward needs a T>1 path that
   routes projections through it AND returns hidden_pre_norm for all K positions).
4. Accept longest greedy-matching prefix; **rollback = KV truncate** to accepted length
   (gemma has NO DeltaNet recurrent state → far simpler than the 27B). Repeat.
5. **Lossless gate**: token-for-token == plain greedy (K=4 & 8). Then bench vs llama 20.6.

## Validation strategy
Op-by-op value-diff the head forward against llama (run llama-server with the head, dump
intermediate tensors) — the same method that found the qwen3next bugs. Budget a focused
session; expect 2-4 subtle bugs (rope/q-norm order, KV-share layer index, embed scale).

## Gotchas (learned today)
- `pkill -f llama-cli` self-kills the calling shell (the pattern matches your own argv) —
  use `pkill -x`. llama-cli `-no-cnv` + piped stdin runs away printing ">" → use llama-server
  + curl for numbers. Vulkan first-load is ~3-4 min (not a hang). Box was clean this session.
