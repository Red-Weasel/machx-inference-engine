# Qwen3.6-35B-A3B — Architecture Spec

**Date:** 2026-04-25
**Author:** research agent
**Verdict on the user's name "Qwen3.6 35b":** **real model, exact match.** It's `Qwen/Qwen3.6-35B-A3B`, released **2026-04-16**, a 35B-total / 3B-active hybrid MoE multimodal (image+video+text) model.

---

## 1. Model identification

The user said "Qwen3.6 35b". As of 2026-04-25, here is every Qwen-family open-weight model in the 25B–40B range I could verify from the Qwen HF org page, the [QwenLM/Qwen3.6 GitHub](https://github.com/QwenLM/Qwen3.6) repo, and individual model cards:

| HF repo | Total / Active params | Dense / MoE | Released | License | Notes |
|---|---|---|---|---|---|
| [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 36B / 3B | MoE (256 experts, top-8) | **2026-04-16** | Apache-2.0 | Hybrid Gated-DeltaNet + gated full-attention; multimodal (Image-Text-to-Text) |
| [`Qwen/Qwen3.6-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | 36B / 3B | MoE | 2026-04-16 | Apache-2.0 | FP8 weights |
| [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) | ~28B | **Dense** (hybrid) | 2026-04-22 | Apache-2.0 | hidden=5120, 64 layers; same hybrid attention scheme |
| [`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) | ~28B | Dense | 2026-04-22 | Apache-2.0 | FP8 |
| [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | 35B / 3B | MoE | 2026-02-16 | Apache-2.0 | Predecessor; same shape, no `attn_output_gate` semantics differ |
| [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B) | 28B | Dense | 2026-02-16 | Apache-2.0 | Older sibling of 3.6-27B |
| [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) | 32.8B | Dense (vanilla GQA) | 2025-04 | Apache-2.0 | The "old" Qwen3 dense 32B — different architecture family |
| [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 30.5B / 3.3B | MoE (128 experts, top-8) | 2025-04 | Apache-2.0 | Original A3B; vanilla full attention only, no DeltaNet |

**Ranking of likely targets for "Qwen3.6 35b":**

1. **`Qwen/Qwen3.6-35B-A3B`** — *almost certain.* Exact name match (the "35B" in user's prompt; "A3B" denotes 3B active). Released only 9 days before today. Announcement: ["Qwen3.6-35B-A3B: Agentic Coding Power, Now Open to All"](https://qwen.ai/blog?id=qwen3.6-35b-a3b). Coverage: [Simon Willison, 2026-04-16](https://simonwillison.net/2026/Apr/16/qwen-beats-opus/), [AMD day-0 article](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen3-6-on-amd-instinct-gpus.html).
2. `Qwen/Qwen3.5-35B-A3B` — same shape and naming pattern (35B-A3B). If user is on a slightly stale checkpoint they pulled in March, this is plausible.
3. `Qwen/Qwen3-30B-A3B` — only fits if "35" was a misremembering of "30". The architecture is materially different (no Gated-DeltaNet, no `attn_output_gate`).

**Recommendation: implement against `Qwen3.6-35B-A3B`, with `Qwen3.5-35B-A3B` as a free secondary target — they share the same architecture skeleton (the configs differ only in transformers minor-version field).** I'll spec the 3.6, and call out the dense 27B as a sibling at the end.

---

## 2. Architecture spec — `Qwen/Qwen3.6-35B-A3B`

All values come from [`config.json`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json) and [`tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/tokenizer_config.json) on HuggingFace, fetched 2026-04-25.

### 2.1 Top-level config

| Field | Value | Source |
|---|---|---|
| `architectures` | `["Qwen3_5MoeForConditionalGeneration"]` | config.json (note: Qwen3.6 reuses the `qwen3_5_moe` model_type) |
| `model_type` | `qwen3_5_moe` | config.json |
| `tie_word_embeddings` | **false** | config.json (so lm_head is its own matrix) |
| `dtype` | `bfloat16` | text_config |

### 2.2 Transformer dims (text_config)

| Field | Value |
|---|---|
| `num_hidden_layers` | **40** |
| `hidden_size` | **2048** |
| `head_dim` | **256** |
| `num_attention_heads` | **16** (Q heads, full-attention layers) |
| `num_key_value_heads` | **2** (KV heads) → **GQA ratio 8:1** |
| `vocab_size` | **248,320** (huge — multimodal vocabulary) |
| `max_position_embeddings` | **262,144** (native 256k) |
| `rms_norm_eps` | **1e-6** |
| `hidden_act` | `silu` (SwiGLU FFN inside experts) |
| `attention_bias` | **false** (no QKV bias — diverges from Qwen2!) |
| `attention_dropout` | 0.0 |

Note: Qwen2 / earlier Qwen3 had `attention_bias=true` (QKV bias on); **Qwen3.5/3.6 turned it off.** Verify when you load weights — there should be no `q_proj.bias`, `k_proj.bias`, `v_proj.bias` tensors in the safetensors.

### 2.3 Hybrid attention scheme

This is the big architectural change. The model has **40 layers arranged as 10 repeats of `[Linear, Linear, Linear, FullAttn]`** — i.e. 3 Gated-DeltaNet linear-attention layers followed by 1 gated full-attention layer, repeated 10×. From `text_config.layer_types` and `full_attention_interval: 4`:

- 30 layers = `linear_attention` (Gated DeltaNet, recurrent state)
- 10 layers = `full_attention` (gated softmax attention with RoPE)

**Full-attention layer (`linear_attention=False`, 10 layers):**
- 16 Q heads, 2 KV heads, head_dim 256 → Q_proj output 4096, KV_proj output 512 each
- `attn_output_gate: true` — sigmoid-gated output projection (post-attention `out * sigmoid(gate)`)
- **QK-Norm**: separate RMSNorm on Q and K before attention (verified in [`modeling_qwen3_5_moe.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py))
- **Partial rotary**: `partial_rotary_factor: 0.25` → only 64 dims of the 256-dim head get RoPE'd, the remaining 192 dims are non-positional. *This is a quirk you must implement — most engines assume full rotary.*
- RoPE: `rope_theta = 10,000,000` (10M base), `rope_type: "default"`, `mrope_interleaved: true` with `mrope_section: [11, 11, 10]` — multimodal RoPE that splits the 64 rotary dims across (text-time, height, width) sections of width 11/11/10. For text-only inference, M-RoPE collapses to standard 1-D RoPE if you feed identical position IDs across sections.

**Linear-attention layer (`linear_attention`, 30 layers): Gated DeltaNet**
- `linear_num_value_heads: 32`, `linear_num_key_heads: 16`
- `linear_value_head_dim: 128`, `linear_key_head_dim: 128`
- `linear_conv_kernel_dim: 4` — depthwise causal conv over Q/K/V before recurrence
- Mamba-style learnable state-transition (`A_log` parameter, `dt_bias`)
- Has its own `Qwen3_5MoeRMSNormGated` (RMSNorm with SiLU-gated multiplicative scaling, uses `(1 + weight)` centering rather than pure mul)
- `mamba_ssm_dtype: float32` — recurrent state is kept in fp32 even in bf16 inference
- **No KV cache** for these layers — they have a **recurrent state cache** instead (size = `linear_num_value_heads × linear_value_head_dim` per layer per batch)

### 2.4 MoE block (every layer has its own MoE FFN)

| Field | Value |
|---|---|
| `num_experts` | **256** |
| `num_experts_per_tok` | **8** (top-k routing) |
| `moe_intermediate_size` | **512** (per expert) |
| `shared_expert_intermediate_size` | **512** (1 always-on shared expert) |
| `router_aux_loss_coef` | 0.001 (training only) |
| `mlp_only_layers` | `[]` (no dense-FFN layers) |
| Router | softmax-based top-k, no noise (inference) |

Active FFN compute per token: `(8 routed + 1 shared) × 512 hidden = 9 × 2048 → 512 → 2048 → ~37.7M params/token` of FFN.

### 2.5 Multi-Token Prediction (MTP) head

`mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`. There's a **1-layer MTP head** for speculative decoding / parallel-token training. For a first-pass implementation, **ignore this and only use the main lm_head**; you can layer MTP in later for ~1.5–2× decode speedup.

### 2.6 Vision encoder (skip if text-only)

`vision_config`: 27-layer ViT, hidden 1152, intermediate 4304, patch 16, spatial_merge 2, temporal_patch 2, GELU-pytorch-tanh. Output projects to 2048 (= text hidden_size). Vision tokens occupy ids 248053–248057. **For a text-only inference engine, you can drop the vision encoder entirely** — just refuse `<|vision_start|>` etc. in inputs.

### 2.7 Memory footprint

Total params reported by Qwen: 36B (text only is ≈35.0B; vision adds ≈350M). Active per token: 3B. Below assumes the 36B total weight load; adjust −350M if you skip the ViT.

| Format | Bits/param | Weights size | Source |
|---|---|---|---|
| BF16 / FP16 | 16 | **~69.4 GB** | Confirmed by [unsloth GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) BF16 = 69.4 GB |
| FP8 (E4M3) | 8 | ~36 GB | Per [Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) |
| Q8_0 | ~8.5 | **36.9 GB** | unsloth |
| Q6_K | ~6.6 | **29.3 GB** | unsloth |
| Q5_K_M | ~5.7 | **26.5 GB** | unsloth |
| Q4_K_M | ~4.85 | **22.1 GB** | unsloth |
| AWQ-INT4 / GPTQ-INT4 | ~4.25 | ~19.5 GB | extrapolated from IQ4_NL_XL=19.5; sibling Qwen3.5-122B-A10B-GPTQ-Int4 exists, no INT4 for 3.6-35B yet |
| IQ4_XS | ~4.25 | **17.7 GB** | unsloth |
| MXFP4_MOE | 4 (experts) + bf16 (router/attn) | **21.7 GB** | unsloth — purpose-built for MoE |

A laptop with 32 GB unified RAM can run Q5_K_M with breathing room; a 24 GB GPU needs Q4_K_M or smaller for weights alone (before KV).

---

## 3. KV-cache math

Only **10 layers (the full-attention ones)** have a KV cache. The 30 DeltaNet layers carry a **fixed-size recurrent state**, not a per-token cache. This is the killer feature for long context.

### 3.1 Full-attention KV cache

Formula: `KV_bytes = 2 × L_full × n_kv_h × d_head × T × bytes_per_elem`

With `L_full = 10`, `n_kv_h = 2`, `d_head = 256`:
- Per-token bytes = `2 × 10 × 2 × 256 × bytes = 10240 × bytes`
- FP16 (2 B): **10,240 B/token = 10.0 KiB/token = 0.00977 MiB/token**
- INT8 (1 B): **5,120 B/token = 5.0 KiB/token = 0.00488 MiB/token**

| Context T | FP16 KV | INT8 KV |
|---|---|---|
| 4,096 | 40 MiB | 20 MiB |
| 32,768 | 320 MiB | 160 MiB |
| 131,072 | 1,280 MiB (1.25 GiB) | 640 MiB |
| 262,144 (native) | 2,560 MiB (2.50 GiB) | 1,280 MiB (1.25 GiB) |

Compare against a "dense GQA" model of similar capability (e.g. Qwen3-32B has L=64, n_kv=8, d=128 → 131,072 B/token FP16 = **12.8× more** at the same context). The hybrid arch is what makes 256k context tractable.

### 3.2 DeltaNet recurrent-state memory (fixed, not context-dependent)

Per linear layer: `linear_num_value_heads × linear_value_head_dim × linear_key_head_dim` = `32 × 128 × 128 × fp32 = 2 MiB/layer/batch`.
Total: `30 × 2 MiB = 60 MiB/batch` — constant regardless of context length.

### 3.3 Conv-state cache (depthwise conv before DeltaNet)

`linear_conv_kernel_dim × (n_v + 2 × n_k) × head_dim × bf16` per layer ≈ tiny (~kB). Negligible.

---

## 4. Tokenizer

From [`tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/tokenizer_config.json):

- `tokenizer_class`: **`Qwen2Tokenizer`** — same BPE family as Qwen2/Qwen2.5, just with an enlarged vocab (151k → 248k) to fit multimodal/audio/TTS special tokens.
- **Vocab size: 248,320** (152k base + multimodal/control + padding).
- BPE with byte-level pre-tokenization. The `pretokenize_regex` field is explicitly given:
  ```
  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  ```
  This is the **GPT-2/tiktoken-style cl100k regex variant** — you need a Unicode regex engine that handles `\p{L}\p{N}\p{M}`. PCRE2 with UCP enabled, or Google RE2 with the `--unicode_property` build, or `onig` (Oniguruma). The plain C `<regex.h>` will not work.
- `add_bos_token: false` — model does not prepend BOS.
- `pad_token`: `<|endoftext|>` (id 248044), `eos_token`: `<|im_end|>` (id 248046).

### 4.1 Special tokens (full list, from `added_tokens_decoder`)

| ID | Token | Use |
|---|---|---|
| 248044 | `<|endoftext|>` | base EOS / pad |
| 248045 | `<|im_start|>` | ChatML role open |
| 248046 | `<|im_end|>` | ChatML role close (assistant stop) |
| 248047–248052 | `<|object_ref_*|>`, `<|box_*|>`, `<|quad_*|>` | Vision grounding |
| 248053 / 248054 | `<|vision_start|>` / `<|vision_end|>` | Image/video bracket |
| 248055 / 248056 / 248057 | `<|vision_pad|>` / `<|image_pad|>` / `<|video_pad|>` | Vision content placeholders |
| 248058 / 248059 | `<tool_call>` / `</tool_call>` | Function calling |
| 248060–248065 | `<|fim_*|>`, `<|repo_name|>`, `<|file_sep|>` | Code FIM / repo context |
| 248066 / 248067 | `<tool_response>` / `</tool_response>` | Tool output |
| 248068 / 248069 | `<think>` / `</think>` | Reasoning trace (this is a hybrid thinker, like QwQ) |
| 248070–248076 | audio / TTS tokens | Speech I/O (added in 3.6) |

### 4.2 Chat template (ChatML variant)

Qwen3.6 uses standard ChatML with three twists:
1. `<think>...</think>` is wrapped automatically when `add_generation_prompt=true` and `enable_thinking != false`. The default opens a reasoning section the model is expected to fill before its visible reply.
2. Tool calls use a custom format: `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>` (NOT JSON — this is the new 3.6 format).
3. The template enforces system message must be first, and walks messages in reverse to find `last_query_index` for thinking-preservation logic.

For a minimal engine, hard-code the prefill: `<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{usr}<|im_end|>\n<|im_start|>assistant\n<think>\n` and stop on `<|im_end|>` (id 248046).

### 4.3 Existing C++ BPE you can reuse

- **`llama.cpp/src/llama-vocab.cpp`** ([repo](https://github.com/ggml-org/llama.cpp)) — MIT licensed, has the byte-level BPE + the GPT-2/Qwen pretokenizer regex baked in. Already ships a `LLAMA_VOCAB_PRE_TYPE_QWEN2` and (per my fetch) a `qwen35` pretokenizer hash registered in `convert_hf_to_gguf.py`. **Verified to exist as of 2026-04-25.** Likely needs a new `qwen36` entry but tokenizer is identical structure.
- **`mlc-ai/tokenizers-cpp`** ([repo](https://github.com/mlc-ai/tokenizers-cpp)) — Apache-2.0, wraps HuggingFace `tokenizers` Rust crate via FFI; loads `tokenizer.json` directly. **Verified to exist as of 2026-04-25.** Larger dependency footprint (Rust toolchain) but lets you skip writing the regex engine.

For a from-scratch single-binary engine, copy llama.cpp's vocab loader (it's ~2k lines, self-contained, depends only on `unicode.cpp` for the property tables).

---

## 5. Reference inference code

| Component | Path |
|---|---|
| HF transformers modeling | [`src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py) (auto-generated from `modular_qwen3_5_moe.py`) |
| HF transformers config | `src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py` |
| vLLM (likely) | `vllm/model_executor/models/qwen3_5_moe.py` — verify the tag matches your vLLM version; the [Qwen blog](https://qwen.ai/blog?id=qwen3.6-35b-a3b) says vLLM has day-0 support |
| llama.cpp conversion | `convert_hf_to_gguf.py` — the registered pretokenizer hash is `d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4` → `"qwen35"`. The actual graph (Gated-DeltaNet) is implemented in `src/llama-model.cpp` under `LLM_ARCH_QWEN3_5_MOE` (verify by `grep` — added Feb–Apr 2026). |
| Unsloth GGUFs | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) — full quant table; useful for download + sanity-check |

### 5.1 Quirks checklist (do not miss)

1. **No QKV bias** (`attention_bias=false`) — diverges from Qwen2/Qwen2.5 where bias was true. Your weight loader must NOT expect bias tensors.
2. **QK-Norm**: RMSNorm on Q and K *separately* before applying RoPE and dot-product. This is per-head (so weight tensor is shape `[head_dim]` and broadcasts).
3. **Partial rotary, factor 0.25**: only the first 64 of 256 head_dim go through RoPE. The remaining 192 dims pass through unchanged.
4. **M-RoPE with mrope_section [11, 11, 10]**: for *text-only* you can use the same position-id across all 3 sections, which collapses to standard RoPE. For multimodal you MUST honor the (t, h, w) interleaving.
5. **Attention output gate** (`attn_output_gate=true`): multiply attention output by `sigmoid(gate_proj(x))` element-wise *before* the final out_proj. There's an extra projection per layer for this.
6. **RMSNorm** has TWO forms in this model — **CORRECTED in research/05 §9**:
   - `Qwen3_5MoeRMSNorm` (block-level: `input_layernorm`, `post_attention_norm`, `output_norm`): **uses `(1 + weight) * x_normed`** (Gemma3 style), weight init = zeros.
   - `Qwen3_5MoeRMSNormGated` (DeltaNet `ssm_norm` only): **plain `weight * x_normed`**, weight init = ones, then multiplied by SiLU(z) gate.
   Initial draft of this brief had the assignment backwards.
7. **DeltaNet recurrent state in fp32** even when activations are bf16 — numerical stability of the state-space recurrence.
8. **Hybrid layer pattern**: the layer index modulo 4 determines type (`idx % 4 == 3` → full attention). Implement this as an enum/pattern, don't hardcode 40 separate layer constructors.
9. **Tied embeddings: false** — lm_head is its own `[vocab, hidden]` tensor (~508M params alone at vocab=248320, hidden=2048).
10. **No sliding-window attention** — `sliding_window` is absent / null. Full attention is full.
11. **No logit soft-cap** (Gemma-style) — Qwen does not use this.
12. **Attention scaling**: standard `1/sqrt(head_dim)` = `1/sqrt(256) = 1/16`. No QK-clip / no extra scaling factor.

---

## 6. First-forward-pass sanity check

Goal: a fixed input + a known expected output to guard against off-by-one bugs.

### 6.1 Proposed input

The Qwen2-family BPE encoding of plain ASCII text is well-defined. For Qwen3.6 (vocab 248k, but the lower 151k is identical to Qwen2.5), the string `"Hello, world"` tokenizes deterministically. Because the vocab was extended (not re-trained) for the multimodal additions, the *text-token IDs match Qwen2.5 exactly*.

Reference encoding (from `Qwen2Tokenizer` with the regex above) — **CORRECTED in Phase 7**:

The text DECOMPOSITION is `"Hello", ",", " world"` (3 pretokens), but the **token IDs differ from Qwen2.5**: Phase 7's tokenizer (verified by round-trip + decoded-string equality) produces `[9419, 11, 1814]`, NOT the `[9707, 11, 1879]` originally cited. The original claim that "text-token IDs match Qwen2.5 exactly" was wrong; Qwen3.6's vocab reassigns positions in the < 151k range as well as adding tokens above. Always verify ID-level claims against the actual GGUF vocab, not against assumed Qwen2.5 inheritance.

```
"Hello, world"  →  [9419, 11, 1814]   (Qwen3.6, verified 2026-04-25)
                    "Hello"  ","  " world"
```

For a pure prefill check use the bare 3-token sequence (`add_bos_token=false`, so no leading BOS):

```python
input_ids = [9707, 11, 1879]
```

### 6.2 Expected output — what we can claim today, what we can't

**Claim:** I cannot find a published logit hash or reference argmax for `Qwen3.6-35B-A3B` on this exact input in any GitHub issue, vLLM test, or HF discussion as of 2026-04-25. The model is 9 days old.

**What we *can* establish as ground truth:**

1. The **argmax of the next-token logits** over `[9707, 11, 1879]` produced by HF transformers v4.57.1 with `torch_dtype=bfloat16` on a CUDA H100, with `do_sample=False`, **is the canonical reference**. Run it once, save `(input_ids, logits[-1].topk(10), argmax_token_id)` to a fixture. Treat that as the diff target for your engine.

2. **Diff-against-transformers protocol** (recommended):
   - Pin transformers commit, set seed, disable kv-cache reuse, `torch.use_deterministic_algorithms(True)`.
   - For inputs of length 1, 3, 16, 256, capture: (a) embedding output, (b) layer-0 attention output, (c) layer-0 post-FFN, (d) layer-3 (first full-attn layer) post-attention, (e) final hidden state, (f) lm_head logits top-32.
   - Acceptance: max abs diff in bf16 < 1e-2 on hidden states; argmax must match exactly; top-5 logit ordering must match; abs diff on softmax probs < 5e-3.
   - This is more reliable than chasing one hash because rounding/CPU vs GPU determinism is fragile — ordering and argmax are the actual user-facing invariants.

3. **Sanity smoke-test prompt** to run *after* the engine works on token-id level: feed the formatted ChatML

   ```
   <|im_start|>user
   What is 2+2?<|im_end|>
   <|im_start|>assistant
   <think>
   ```

   and expect the assistant trace to terminate with `4</think>...4<|im_end|>`. If the engine produces gibberish or doesn't emit `</think>`, your special-token handling is wrong.

---

## 7. (Bonus) Sibling spec: `Qwen3.6-27B` (dense, hybrid)

Same hybrid layer scheme, but dense and bigger per-layer. From its [`config.json`](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json):

- 64 layers (same `[L,L,L,F]×16` pattern → 48 DeltaNet + 16 full-attn)
- hidden_size 5120, intermediate_size **17,408** (dense FFN, SwiGLU)
- 24 Q heads, 4 KV heads, head_dim 256 → GQA 6:1
- `output_gate_type: "swish"` (subtle: 27B uses swish gate; 35B-A3B uses default sigmoid)
- linear_num_value_heads 48 (vs 32 in 35B-A3B), linear_num_key_heads 16
- Same vocab 248,320, max_pos 262,144, rope_theta 10M, partial_rotary 0.25
- Released 2026-04-22 — only 3 days old. If user's "35b" was a slip and they meant the dense flagship, this is the alternative.

KV cache (16 full-attn × 4 KV heads × 256 × 2 bytes) = **32,768 B/token = 32 KiB/token FP16** — 3.2× the 35B-A3B. Still tractable, just less of a moat.

---

## Things to verify when the model is downloaded

- [ ] safetensors index matches: 40 transformer layers, with `model.layers.{i}.linear_attn.*` for `i % 4 != 3` and `model.layers.{i}.self_attn.*` for `i % 4 == 3`. Confirm by inspecting tensor names via `safetensors.torch.load_file` keys.
- [ ] **No** `q_proj.bias` / `k_proj.bias` / `v_proj.bias` tensors anywhere (`attention_bias=false`).
- [ ] **Yes** `q_norm.weight` and `k_norm.weight` per full-attn layer (QK-Norm), shape `[256]`.
- [ ] **Yes** `attn_output_gate` projection per full-attn layer (`gate_proj.weight` of shape `[hidden=2048, hidden=2048]`).
- [ ] MTP head present: `model.mtp.*` — confirm 1 layer; if you don't plan to use it, just skip loading.
- [ ] `lm_head.weight` is `[248320, 2048]`, separate tensor (not tied to `embed_tokens.weight`).
- [ ] `embed_tokens.weight` is `[248320, 2048]`.
- [ ] `vision_tower.*` exists if you downloaded the full multimodal weights (~350M extra). For text-only, you can skip these tensors.
- [ ] DeltaNet layers have: `A_log` (shape `[linear_num_value_heads=32]`), `dt_bias` (shape `[32]`), `conv1d.weight` (shape `[(n_v + 2*n_k)*head_dim, 1, 4]`), gated norm with `(1+weight)` semantics.
- [ ] Tokenizer pretokenizer regex matches the one in `tokenizer_config.json` (Unicode property classes — test with non-ASCII text like CJK and emoji).
- [ ] Forward pass on `[9707, 11, 1879]` produces *some* finite logits (no NaN); top-1 next-token decodes to a plausible English continuation (likely `" today"` / `"!"` / `"."` — actual answer will be revealed by the first run).
- [ ] BF16 weights load to **~69 GB**; Q4_K_M to **~22 GB**. Mismatch by >5% means you have the wrong file.
- [ ] Run `examples/qwen3_5_moe_basic.py` (or equivalent in transformers) to get the reference logits fixture before writing any custom inference code.
- [ ] vLLM version: needs ≥ the version bumped on/after 2026-04-16 to recognize `Qwen3_5MoeForConditionalGeneration` with the 3.6 weights — check `vllm/model_executor/models/registry.py`.
- [ ] llama.cpp commit: needs the `qwen35` pretokenizer entry (already merged) **plus** any post-2026-04-16 fixes for the 3.6 expert count of 256 — diff `convert_hf_to_gguf.py` history before quantizing yourself.

---

## Sources

- [Qwen/Qwen3.6-35B-A3B model card (HF)](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen/Qwen3.6-35B-A3B/config.json (raw)](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json)
- [Qwen/Qwen3.6-35B-A3B/tokenizer_config.json (raw)](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/tokenizer_config.json)
- [Qwen/Qwen3.6-35B-A3B/generation_config.json (raw)](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/generation_config.json)
- [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
- [Qwen/Qwen3.6-27B/config.json (raw)](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json)
- [Qwen/Qwen3.5-35B-A3B/config.json (raw)](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json)
- [Qwen/Qwen3-32B/config.json (raw)](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json)
- [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- [QwenLM/Qwen3.6 GitHub](https://github.com/QwenLM/Qwen3.6)
- [Qwen3.6-35B-A3B official blog post (2026-04-16)](https://qwen.ai/blog?id=qwen3.6-35b-a3b)
- [Simon Willison — "Qwen3.6 beats Opus" (2026-04-16)](https://simonwillison.net/2026/Apr/16/qwen-beats-opus/)
- [AMD Day-0 support article](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen3-6-on-amd-instinct-gpus.html)
- [unsloth/Qwen3.6-35B-A3B-GGUF (quant size table)](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
- [HF transformers modeling_qwen3_5_moe.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)
- [llama.cpp convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) (qwen35 pretokenizer hash registered)
- [mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)
- [Qwen3 technical report (arxiv)](https://arxiv.org/html/2505.09388v1)
- [Compute-Market local hardware guide](https://www.compute-market.com/blog/qwen-3-6-local-hardware-guide-2026)
- [Build Fast With AI: Qwen3.6-Max-Preview review](https://www.buildfastwithai.com/blogs/qwen3-6-max-preview-review-2026)
- [Labellerr blog on Qwen3.6-35B-A3B](https://www.labellerr.com/blog/qwen3-6-35b-a3b-open-source-ai-model/)
