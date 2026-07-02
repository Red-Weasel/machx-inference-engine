# Qwen3.6-27B (`qwen35`) GGUF tensor layout — verified

**Source:** `bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_M.gguf`
(17.98 GB, 866 tensors, GGUF v3), read via llama.cpp `gguf-py` 2026-06-11.
Config: `read_qwen35_config` validated — 65 block_count (64 transformer + 1
NextN), hidden 5120, ffn 17408, 24 q / 4 kv heads, head_dim 256, ssm inner 6144 /
48 v-heads / 16 k-heads / state 128, conv 4, full_attn_interval 4, vocab 248320.

Tensor names use the GGUF `blk.N.*` convention. Shapes below are GGUF order
`[in, out]` (column-major vs torch).

## Linear (gated-DeltaNet) layer — `recurrent_layer(il)==true` (48 layers)
Example `blk.0`:

| tensor | dtype | shape `[in,out]` | role / dim derivation |
|---|---|---|---|
| `attn_norm.weight` | F32 | [5120] | pre-attn RMS |
| `attn_qkv.weight` | **Q6_K** | [5120, **10240**] | fused QKV→conv input. 10240 = d_inner 6144 + 2·group 16·state 128 = 6144+4096 (llama.cpp `wqkv`/`build_qkvz`) |
| `attn_gate.weight` | **Q4_K** | [5120, 6144] | the `z` gate (`wqkv_gate`), 6144 = d_inner |
| `ssm_conv1d.weight` | F32 | [4, 10240] | causal conv, kernel 4 over the 10240 conv channels |
| `ssm_alpha.weight` | F32 | [5120, 48] | α proj (per v-head); → softplus(+`ssm_dt`)·−exp(`ssm_a`) |
| `ssm_beta.weight` | F32 | [5120, 48] | β proj (per v-head); → sigmoid |
| `ssm_dt.bias` | F32 | [48] | α bias |
| `ssm_a` | F32 | [48] | −A_log per v-head |
| `ssm_norm.weight` | F32 | [128] | gated RMS over head_v_dim 128 (with z) |
| `ssm_out.weight` | **Q8_0** | [6144, 5120] | output proj d_inner→hidden |
| `post_attention_norm.weight` | F32 | [5120] | pre-FFN RMS |
| `ffn_gate.weight` | Q4_K | [5120, 17408] | dense SwiGLU |
| `ffn_up.weight` | Q4_K | [5120, 17408] | |
| `ffn_down.weight` | Q6_K | [17408, 5120] | |

## Full-attention layer — `recurrent_layer(il)==false` (il=3,7,…,63; 16 layers)
Example `blk.3`:

| tensor | dtype | shape | role |
|---|---|---|---|
| `attn_norm.weight` | F32 | [5120] | pre-attn RMS |
| `attn_q.weight` | **Q6_K** | [5120, **12288**] | **joint Q\|gate**: 12288 = 24 heads·256·2 (query + sigmoid gate, the crown `ws_q_full_`) |
| `attn_k.weight` | **Q5_K** | [5120, 1024] | 4 kv·256 |
| `attn_v.weight` | **Q6_K** | [5120, 1024] | |
| `attn_q_norm.weight` | F32 | [256] | per-head RMS on Q |
| `attn_k_norm.weight` | F32 | [256] | per-head RMS on K |
| `attn_output.weight` | **Q5_K** | [6144, 5120] | 6144 = 24·256 → hidden |
| `post_attention_norm.weight` | F32 | [5120] | pre-FFN RMS |
| `ffn_gate/up/down` | Q4_K/Q4_K/Q6_K | … | dense SwiGLU (same as linear) |

## NextN / MTP layer — `blk.64` — **SKIP for text-only decode**
Carries a full attn block (all Q8_0) **plus** `nextn.eh_proj` [10240,5120],
`nextn.enorm`/`nextn.hnorm`/`nextn.shared_head_norm` [5120]. Used only for
speculative/multi-token prediction. The standard forward runs layers 0..63 and
ignores `blk.64`. (`nextn_predict_layers=1` in config.)

## Shared / non-layer
| `token_embd.weight` | Q4_K | [5120, 248320] | input embeddings |
| `output.weight` | Q6_K | [5120, 248320] | lm_head — **separate tensor, NOT tied** to token_embd |
| `output_norm.weight` | F32 | [5120] | final RMS |

## ⚠ Findings that affect the plan
1. **Mixed quant types include Q5_K and Q8_0** (`attn_k`, `attn_output` = Q5_K;
   `ssm_out` = Q8_0), beyond the dense path's current GEMV set (Q4_K/Q6_K/F16).
   The engine **has SYCL dequant kernels for Q5_K/Q8_0** (`src/ops/dequant_kernels.cpp`)
   but no int-dot GEMV for them. **First-cut fix: dequant Q5_K/Q8_0 weights →
   fp16 at load, route through `gemv_fp16`** (correct, modest extra memory). Native
   int-dot Q5_K/Q8_0 GEMVs are a later perf task. → add to P3d plan.
2. **Linear layers fuse QKV** into `attn_qkv` [5120,10240] (+ separate `attn_gate`).
   The forward must split conv output into q/k/v (k/v dims = group 16·state 128 =
   2048 each side, v = d_inner 6144), exactly llama.cpp `build_layer_attn_linear`.
3. **Full-attn `attn_q` is the joint Q|gate** (12288 = ·2). The crown already does
   this (`ws_q_full_`); reuse the split.
4. **Embeddings not tied** — load `output.weight` (Q6_K) as the lm_head; it fits the
   fast `gemv_q6_K_q8` (K=5120 ≤ SLM budget), like the crown/dense lm_head.
5. **Per-head norm dims:** attn_q_norm/k_norm are [256] (head_dim), ssm_norm [128]
   (head_v_dim). Wire accordingly.
