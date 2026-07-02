# Phase 1 — GGUF reader baseline

**Date:** 2026-04-25
**Subject:** `/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf` (lmstudio-community quant, 19.71 GiB on disk)
**Tool:** `build/tools/ie-inspect`

## Gate result: PASS

```
Total  733 tensors found  vs 733 expected — GATE PASS
```

All seven tensor groups matched expected count exactly:

| Group | Found | Expected |
|---|---:|---:|
| `token_embd.weight` | 1 | 1 |
| `output.weight` (lm_head) | 1 | 1 |
| `output_norm.weight` | 1 | 1 |
| `blk.{i}.{attn_norm,post_attention_norm}.weight` | 80 | 80 (40 × 2) |
| `blk.{i}.attn_*` (full-attn layers, i%4==3) | 60 | 60 (10 × 6) |
| `blk.{i}.{attn_qkv,attn_gate,ssm_*}` (DeltaNet, i%4!=3) | 270 | 270 (30 × 9) |
| `blk.{i}.ffn_*` (router + 256 experts + 1 shared) | 320 | 320 (40 × 8) |

## Architecture cross-check (KV vs research/04)

| Field | KV value | research/04 §2 | match? |
|---|---|---|---|
| `block_count` | 40 | 40 | ✓ |
| `context_length` | 262,144 | 262,144 native | ✓ |
| `embedding_length` | 2048 | 2048 | ✓ |
| `attention.head_count` | 16 | 16 | ✓ |
| `attention.head_count_kv` | 2 | 2 (GQA 8:1) | ✓ |
| `attention.key_length` / `value_length` | 256 / 256 | head_dim 256 | ✓ |
| `expert_count` | 256 | 256 | ✓ |
| `expert_used_count` | 8 | top-8 | ✓ |
| `expert_feed_forward_length` | 512 | 512 | ✓ |
| `expert_shared_feed_forward_length` | 512 | 512 (1 shared expert) | ✓ |
| `rope.freq_base` | 1e+07 | 10,000,000 | ✓ |
| `rope.dimension_count` | 64 | head_dim 256 × partial 0.25 = 64 | ✓ |
| `full_attention_interval` | 4 | `[L,L,L,F]×10` | ✓ |
| `ssm.conv_kernel` | 4 | linear_conv_kernel_dim=4 | ✓ |
| `ssm.state_size` | 128 | linear_value_head_dim=128 | ✓ |
| `ssm.time_step_rank` | 32 | linear_num_value_heads=32 | ✓ |
| `attention.layer_norm_rms_epsilon` | 1e-06 | 1e-06 | ✓ |

The research brief was correct in every detail. **No surprises in the architecture.**

## Surprises / notes for the engine

1. **Q proj on full-attn layers is `[2048, 8192]`, not `[2048, 4096]`.** With 16 heads × 256 head_dim = 4096, the doubled output is the `attn_output_gate=true` parameterization: Q proj fuses both Q values and the output-gate input into one matmul. Halve the output to recover {Q, gate}. *Update PLAN risk register: this fold-in is NOT documented in research/04 §5.1; add it to the quirks checklist for Phase 4.*
2. **MoE FFN tensors are stored batched along an "experts" axis.** `ffn_gate_exps.weight [2048, 512, 256]` is `(in, out, experts)`, `ffn_down_exps.weight [512, 2048, 256]` is `(in, out, experts)`. Phase 6 sparse-expert dispatch needs to gather along the trailing dim, not load 256 separate tensors.
3. **No MTP and no vision tensors in this quant.** lmstudio-community stripped them. Research/04 §7 verification checklist item for `model.mtp.*` won't apply here — no harm, the engine doesn't use them in v1.
4. **DeltaNet tensors use `ssm_*` prefix in GGUF**, not `linear_attn.*`. Naming map for Phase 5:
   - HF `linear_attn.A_log` → GGUF `ssm_a`
   - HF `linear_attn.dt_bias` → GGUF `ssm_dt.bias`
   - HF `linear_attn.conv1d.weight` → GGUF `ssm_conv1d.weight`
   - HF `linear_attn.norm.weight` → GGUF `ssm_norm.weight`
   - HF `linear_attn.{q_proj,k_proj,v_proj}.weight` → GGUF `attn_qkv.weight` (FUSED)
   - HF `linear_attn.out_proj.weight` → GGUF `ssm_out.weight`
   - The `ssm_alpha`/`ssm_beta` projections are the per-token state-decay/state-update inputs (the "α(x)" and "β(x)" in DeltaNet papers).
5. **Dtype mix**: F32 (301 tensors, ~83 MiB; norms + small 1D bias tensors), Q4_K (371 tensors, 14.93 GiB; bulk weights), Q6_K (61 tensors, 4.69 GiB; "_K_M" upgrades sensitive layers — likely V proj, down_exps, embed/output). All dtype IDs decoded correctly.
6. **Vocab/tokenizer**: gpt2 BPE family with `tokenizer.ggml.pre = "qwen35"`. 248,320 token strings + 247,587 merges. `add_bos_token=false`. EOS=248046 (`<|im_end|>`). Pad=248044 (`<|endoftext|>`). Matches research/04 §4.

Full inspect output saved at `bench/baseline/phase1_inspect.txt`.
