# gpt-oss (OpenAI MoE) — implementation spec for the engine (2026-06-26)

## ✅ STATUS: gpt-oss-20b SHIPS + BEATS llama (2026-06-27) — checklist 1-5 DONE
- **1 register** (`bf70cdd`), **2 loader** (`d931b13`), **3 attention** (`a61443a` `full_attention_gptoss` =
  sinks + SWA window), **4 MoE** (`a61443a` `moe_ffn` = top-4 MXFP4 oneDNN + biases + `swiglu_oai` + SOFTMAX_WEIGHT),
  **5 Harmony chat** (`4009238` `build_harmony_prompt` + chat dispatch + stop + final-channel extraction) — all DONE.
- **WIN vs llama-SYCL** (1×B70): prefill 668 vs 607 (1.10×), decode 31.9 vs 22.35 (1.43×). Decode profile:
  dequant_mxfp4 44% / gemv 28% / naive attn 23-55%. Chat "capital of France?" → res.text="Paris", finish=stop.
- **YaRN** (`b8240c4`): `rope_yarn` ggml-exact (mscale 1.34657 verified) but plain rope BEST ≤4096 (PPL 17.9 vs
  30.5) → OPT-IN `IE_GPTOSS_YARN` for >4096.
- ✅ **MXFP4 decode GEMV DONE (2026-06-27)** — the 44% lever: load-time SoA repack (aligned qs/e planes,
  memory-flat) + W4A8 int-dot `gemv_mxfp4_soa_q8` (`dp4a_ss`, branchless FP4 decode); prefill via
  `dequant_mxfp4_soa_to_Bt`. **Decode 30.5→43.0 tok/s = 1.41×**, PPL-neutral (22.52 vs 22.53), crown
  6.4527 bit-exact, ctest 31/31. `src/ops/gemv_mxfp4.cpp`. Opt-out `IE_GPTOSS_NO_MXFP4_GEMV`, faithful
  `IE_GPTOSS_MXFP4_F16`. (gemv_mxfp4 ~154 GB/s — geometry/ALU headroom remains.)
- ✅ **hd64 DECODE FA-2 DONE (2026-06-27)** — split-K `full_attention_fa2_decode_gptoss` (COPY + per-head
  sink folded in combine) for full-attn layers; SWA layers bounded-naive. **Decode @2K 25.3→53.9 = 2.13×**
  (1.31× @512, flat across ctx), PPL-neutral, crown bit-exact, ctest 31/31. Opt-out `IE_GPTOSS_NO_FA2_DECODE`.
- ✅ **PREFILL wide-tile DONE (2026-06-27)** — `full_attention_gptoss_prefill_tile` = `fa2_tile_wide_impl
  <HD=64, SINK=true>` (proven 27B/gemma tile + per-head sink via a `bool SINK` template param → crown
  byte-identical). **Prefill @4K 327→716 = 2.19×** (1.37× @2K, flat across ctx), crown 6.4527 bit-exact,
  PPL +0.16%, ctest 31/31. Opt-out `IE_GPTOSS_NO_FA2_PREFILL`. gpt-oss now fast on BOTH axes at all ctx.
- **120b — ✅ RUNS on OUR engine (2026-06-28) via MoE tensor-parallel `GptOssTpModel` (`--gpus 2`)** — NOT
  the `GptOssSplitModel` layer-split originally sketched; TP chosen so BOTH cards compute. Multi-shard loader
  merges the 2-file GGUF; host-RAM spill caps the display card (no crash); **decode numerically correct
  (streaming PPL 12.91) + fast (14.6 tok/s > LM Studio 12.42).** All 20b kernels carry over. **2 open
  generation-path bugs block a usable 120b chat** (batched T>1 prefill @E=128; garbage after the Harmony
  prompt — both generation, not the PPL-correct forward). Detail → MASTER_DEV_PLAN §7 +
  `project_gptoss_tp_120b_2026-06-28`. REMAINING perf (deferred): gemv_mxfp4 geometry, SWA decode layers,
  the decode host-sync lever (~252 syncs/token). Research thread still open: a new own MoE-tiering / rotor-KV method.
- Resolved during impl: NO QK-norm; pre-MoE norm = `post_attention_norm`; attn scale 1/√64; SWA `il%2==0`
  (set_swa_pattern(2)); swiglu_oai gate=src0 (clamp above), up=src1 (clamp ±limit), out=glu·(up+1); expert biases
  weighted by routing (weight_before_ffn=false). DECODE-BUG lesson: `host_routes_` max_T-sized + `build_moe_packing`
  uses `.size()` → stale T=1 packing (fix: resize(T)).


Oracle = `~/llama.cpp` (`LLM_ARCH_OPENAI_MOE`, `src/models/openai-moe.cpp`, `llama-graph.cpp`,
`llama-hparams.cpp`). Target = add `kGptOss` to the engine on the **qwen3moe** base (top-k MoE +
dense GQA), plus the 3 new pieces below. Prototype on **gpt-oss-20b** (same arch as 120b).

## Identity
- GGUF `general.architecture` = **`gpt-oss`** (`llama-arch.cpp:116`).
- 120b: 117B total / 5.1B active, 36 layers, n_embd 2880, 128 experts, top-4, 64 q-heads (GQA→8).
- 20b: 20.9B / 3.6B active, 24 layers, n_embd 2880, 32 experts, top-4, 64 q-heads. **Same arch.**

## Tensors (per layer `blk.%d.`)
Standard GGUF names + **biases on nearly everything** + a **sinks** tensor:
- Attn: `attn_norm`, `attn_q`(+`.bias`), `attn_k`(+`.bias`), `attn_v`(+`.bias`), `attn_output`(+`.bias`),
  **`attn_sinks`** = `{n_head}` **F32** (per-head scalar, NEVER quantize).
- Post-attn norm: `post_attention_norm` (RMSNorm, applied after attn residual, before MoE).
- MoE: `ffn_gate_inp`(+`.bias`) router `{n_embd,n_expert}`; experts `ffn_gate_exps`(+`_b`),
  `ffn_up_exps`(+`_b`), `ffn_down_exps`(+`_b`). (Some GGUFs fuse gate+up as `ffn_gate_up_exps`.)
- Global: `token_embd`, `output_norm`, `output` (lm_head).

## Config keys (`gpt-oss.*`, standard GGUF KV)
`block_count`, `embedding_length`, `attention.head_count`, `attention.head_count_kv`,
`attention.key_length` (head_dim), `expert_count`, `expert_used_count` (=4),
`expert_feed_forward_length`, `attention.sliding_window`, `rope.freq_base`,
`rope.freq_base_swa` (optional, SWA layers), `attention.layer_norm_rms_epsilon`. Verify
`expert_used_count==4` (if 0, MoE disabled). `read_dense_config_auto` already reads the dense subset
by arch prefix → `read_gptoss_config` = that + the 4 MoE/window keys (mirror `read_qwen3moe_config`).

## The 3 new pieces (vs qwen3moe, which gives router + scatter/gather + per-expert GEMM + dense attn)

### 1. Attention sinks (NEW — unique to gpt-oss)
A per-head learned scalar that enters the softmax **denominator** as a virtual always-attended token:
```
m   = max(logits, sink_h)              # include sink in the max for stability
den = Σ_j exp(logit_j - m) + exp(sink_h - m)
out = Σ_j (exp(logit_j - m)/den) · V_j   # the sink contributes to den but has NO value → mass leaks out
```
(llama `ggml_soft_max_add_sinks`, `ggml-cuda/softmax.cu:85,123`.) **Our FA kernels do NOT support
this** (`src/ops/attention.cpp:1859` excludes sinks). HOOK: pass `const float* sink` (per-head) into
the FA softmax; fold `exp(sink_h - m)` into the running denominator of the online-softmax (both the
tiled prefill `fa2_tile_wide_impl` and the decode `fa2_partial`/gemv paths). Effort M (the math is a
one-line denominator add per head, but it touches every FA variant).

### 2. Alternating sliding-window / dense (reuse Gemma SWA)
`swa_period=2, dense_first=false` → **even layers windowed, odd layers full-dense** (`il%2==0` is SWA),
window = `attention.sliding_window`. Reuse the Gemma windowing in `fa2_tile_wide_impl`
(`attention.cpp:2243-2347`: per-WG wstart skip + per-key mask). Store a per-layer `is_swa[]` (and
per-layer RoPE base) at load like Gemma's per-layer head geom; dispatch attention per layer.

### 3. RoPE per-layer + MoE gating + biases + clamp
- **RoPE**: SWA layers use `rope.freq_base_swa` + scale_swa; dense use `rope.freq_base`. Per-layer.
- **MoE gating = SOFTMAX_WEIGHT**: select top-4 **then softmax over the 4 selected logits** (NOT
  per-expert sigmoid). qwen3moe's router host-softmax+topk needs the renorm to be softmax-over-topk.
- **Biases**: add bias to attn q/k/v/o projections, the router, and all 3 expert GEMMs. Our GEMV/GEMM
  paths need a `+bias` (cheap; the dense path may already accept attn bias — Qwen2 uses attn bias).
- **Clamped gated-SwiGLU (OAI variant)**: gpt-oss clamps the gate/up before SwiGLU (a `limit`); confirm
  the clamp constant from the HF config (`swiglu_limit`/`clamp`) — apply in the activation kernel.

## Quant / GGUF — RESOLVED + UNBLOCKED ✅
Verified (agent + bartowski/unsloth): **ALL practical gpt-oss GGUFs keep the MoE experts in MXFP4**
(Q4_K_M 11.6GB vs Q8_0 12.1GB = 0.5GB delta → experts untouched; re-quantizing MXFP4 hurts quality, so
nobody does). The engine had no FP4 path → **route (B) taken: MXFP4 dequant SHIPPED** (`1ee4a2e`):
`block_mxfp4` + `dequant_mxfp4` + `dequant_mxfp4_to_Bt` (the latter feeds the existing per-expert oneDNN
MoE GEMM), **bit-exact vs llama** (`mxfp4_dequant_test`, ctest 31/31). So gpt-oss experts now load with
existing infra. Grab `ggml-org`/`unsloth`/`bartowski` gpt-oss-20b **mxfp4** GGUF (≈12GB, fits 46GB free);
**120b ≈ 63GB needs disk freed first** (see roadmap — the EXL3-80B dir is 43GB of reclaim).

## Build checklist (ordered; effort)
1. **[S] Register** `kGptOss`: enum (`model_config.hpp:12`), `detect_arch` (`model_config.cpp:36`,
   `if (a=="gpt-oss") return kGptOss`), `GptOssConfig` + `read_gptoss_config`.
2. **[M] Loader** `GptOssModel` (model class on the qwen3moe template): load attn (+biases), sinks
   (F32), experts (+biases), norms; per-layer `is_swa[]` + RoPE base; 1- and 2-GPU (split) like 80b.
3. **[M] Attention**: sinks in the FA softmax + alternating SWA (reuse Gemma window) + per-layer RoPE.
4. **[M] MoE**: reuse qwen3moe router/scatter/GEMM; top-4; SOFTMAX_WEIGHT renorm; expert+router biases;
   clamped gated-SwiGLU. (The oneDNN MoE-prefill lever + 2-card per-device map from the 80B work apply.)
5. **[S] Tokenizer/template**: gpt-oss uses the **Harmony** chat format — add a template family (or
   ChatML fallback for raw bench first). Tokenizer auto-detects from GGUF.
6. **[S] Validate**: "Paris" coherence on 20b → PPL → bench vs llama-SYCL. Then scale to 120b (load-only).

Total ≈ **L** (the sinks FA kernel + per-layer SWA/RoPE + biased MoE are the real work; everything else
reuses qwen3moe + the Gemma SWA + the 80B 2-card/oneDNN infra). gpt-oss-20b is the validation vehicle.
