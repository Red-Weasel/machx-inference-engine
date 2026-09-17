# Qwen3-Next-80B-A3B (`qwen3next`) — verified dataflow spec

Foundation for the forward (the 27B's `qwen35_27b_oracle_dataflow.md` analogue).
Config + every per-layer tensor name/dtype/shape and the dataflow dims below are
**verified against the real GGUF** (`ie-qwen3next-config-test` → "ALL 48 layers
OK"; bartowski `Qwen_Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf`, 46 GB). Write the
forward from THIS, op-by-op-validated vs the llama.cpp `qwen3next` oracle.

## Thesis
Next-80B = **the crown family at 80B** (llama.cpp groups `QWEN3NEXT` with
`QWEN35MOE`): gated DeltaNet + gated full-attn + MoE. It composes from existing
additive engine blocks — **no new math**:
- DeltaNet + gated full-attn → `qwen35_dense.cpp` (the 27B), adapted dims.
- MoE (top-k, int-dot) → `qwen3moe.cpp` (GPU router + the new `moe_qwen3.cpp`
  int-dot down; E_ffn=512 is the int-dot **base case**).
- Shared expert → the crown's pattern (NEW small piece).
- 46 GB > 32 GB B70 → **must** run layer-split across 2×B70 (`DenseModelSplit`
  pattern extended to hybrid+MoE+state). This is the one genuinely new capability.

## Config (verified)
48 layers · hidden 2048 · vocab 151936 · **NEOX rope** n_rot 64 θ1e7 · eps 1e-6 ·
`full_attention_interval=4` → `recurrent_layer(il) = ((il+1)%4)!=0` → 36 DeltaNet
+ 12 full-attn, pattern `DN DN DN ATTN …`. No NextN/MTP.
- **Attention** (full-attn layers): 16 q / 2 kv heads, head_dim 256.
- **DeltaNet**: inner 4096, state 128, n_k_heads 16, n_v_heads 32, k/v head_dim
  128, conv_kernel 4. `conv_channels = inner + 2·n_k·state = 4096 + 2·16·128 = 8192`.
- **MoE**: 512 experts / 10 used, E_ffn 512, **shared expert** E_ffn 512.

## Per-layer tensors (verified, both kinds)
Shared (both): `attn_norm`(F32), `post_attention_norm`(F32 — this is the pre-FFN
norm; there is NO separate `ffn_norm`).
MoE (every layer): `ffn_gate_inp`(F32 [2048,512] router) · `ffn_{gate,up}_exps`
(Q4_K [2048,512,512]) · `ffn_down_exps`(Q6_K [512,2048,512]) · shared expert:
`ffn_gate_inp_shexp`(F32 [2048] sigmoid gate) · `ffn_{gate,up,down}_shexp`(Q8_0).
- **DeltaNet layers**: `attn_qkv`(Q5_K [2048,**8192**]=fused q|k|v conv input) ·
  `attn_gate`(Q4_K [2048,**4096**]=inner, the z-gate) · `ssm_a`(F32 [32]=A_log per
  v-head) · `ssm_ba`(Q8_0 [2048,**64**]=fused [beta(32)|alpha(32)]) ·
  `ssm_conv1d`(F32 [4,8192]) · `ssm_dt.bias`(F32 [32]) · `ssm_norm`(F32 [128]) ·
  `ssm_out`(Q8_0 [4096,2048]).
- **Full-attn layers**: `attn_q`(Q4_K [2048,**8192**]=joint Q|gate=2·16·256) ·
  `attn_k`,`attn_v`(Q8_0 [2048,512]) · `attn_q_norm`,`attn_k_norm`(F32 [256]) ·
  `attn_output`(Q6_K [4096,2048]).
Top-level: `token_embd`(Q?), `output_norm`(F32), `output` (lm_head; tied if absent).
Mixed dtypes per layer (Q4_K/Q5_K/Q6_K/Q8_0) — Q5_K/Q8_0 dequant-to-fp16 at load
(reuse `upload_weight_auto` from qwen3moe/qwen35_dense).

## Forward (per layer il)
```
x → rms(attn_norm) → xn
  recurrent_layer(il)?  DeltaNet(xn)  :  GatedAttn(xn)   → +x (residual)
  → rms(post_attention_norm) → xn2
  → MoE(xn2) + SharedExpert(xn2)      → +x (residual)
→ rms(output_norm) → lm_head
```
### DeltaNet (adapt qwen35_dense; deltas vs the 27B noted ⚠)
1. `attn_qkv`: xn@[2048,8192] → qkv_conv[8192].
2. `depthwise_conv1d_causal`(ssm_conv1d, kernel 4, 8192 ch).
3. split conv: q_conv@[0,2048) k_conv@[2048,4096) v_conv@[4096,8192) (q/k=16·128, v=inner 4096).
4. l2norm q (×1/√128) · l2norm k (×1) · `repeat_interleave_heads` k/q **16→32**
   (⚠ 27B was 16→48; here n_v=32).
5. `ssm_ba`: xn@[2048,64] → split **[beta(32)|alpha(32)]** (⚠ 27B had SEPARATE
   `ssm_alpha`/`ssm_beta` projections; Next FUSES them — split after the proj).
6. `compute_g_beta_h`(ssm_a[32] A_log, ssm_dt.bias[32]).
7. `deltanet_recurrence`(n_v=32, k_hd=128, v_hd=128, state).
8. `gated_rms_norm`(ssm_norm[128]) with z = sigmoid-gate from `attn_gate`(xn@[2048,4096]).
9. `ssm_out`: [4096]@[4096,2048] → [2048].
### GatedAttn (adapt qwen35_dense gated attn, but ⚠ NEOX rope not M-RoPE)
1. `attn_q`: xn@[2048,8192] → `split_q_gate_per_head` → Q[16·256] | gate[16·256].
2. `attn_k`/`attn_v`: xn@[2048,512] → K/V[2·256].
3. per-head Q-norm(attn_q_norm[256]) · K-norm(attn_k_norm[256]).
4. **`rope_partial` NEOX** n_rot 64 θ1e7 (⚠ use qwen3moe's NEOX rope, NOT the 27B's
   interleaved M-RoPE — `general` rope type is NEOX for qwen3next).
5. GQA full attention (16q/2kv, KV cache idx = il/interval).
6. `sigmoid_gate`(gate) · `attn_output`: [4096]@[4096,2048] → [2048].
### MoE + shared expert (every layer)
- Router: xn2@`ffn_gate_inp`[2048,512] → logits[512] (GPU gemm — Win-A pattern) →
  top-10 softmax + renorm.
- Experts: qwen3moe int-dot fused — gather → `moe_prefill_gate_up_silu_q4k_q8` →
  `moe_prefill_down_q6k_q8_gen` (**E_ffn=512 = the int-dot base case**) → reduce.
- **Shared expert (always on)**: g_sh = sigmoid(xn2·`ffn_gate_inp_shexp`);
  y_sh = (silu(xn2@gate_shexp) ⊙ (xn2@up_shexp)) @ down_shexp; out = moe_out + g_sh·y_sh.

## Build order (next sessions)
1. **Loader** (`Qwen3NextModel`, fleet-aware): tensor map above (VERIFIED) +
   per-device placement (`LayerPlan`/`DeviceFleet`) + hybrid caches (KV for 12
   full-attn, DeltaNet state for 36 linear). Gate: "LOAD OK across 2×B70".
2. **Forward** op-by-op vs the llama.cpp `qwen3next` oracle (eval-callback via
   mmap — 46 GB > 30 GB RAM but llama mmaps). Per-op cosine ≥0.999, then PPL/greedy.
3. **Layer-split forward** across 2×B70 (the only way it runs). Crown 6.45 bit-exact
   + ctest throughout. Wire into ie-perplexity + Engine + ie-bench.

Oracle: `~/llama.cpp` has `LLM_ARCH_QWEN3NEXT` (+ `SSM_BETA_ALPHA`/`SSM_A_NOSCAN`).
