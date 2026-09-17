# V4.1 architecture, read from the shipped reference (2026-09-12)

`00_PORT_SCOPE` described the architecture from the **tech report**. This document describes it
from `inference/model.py`, which ships with the weights. Where they disagree, the code wins.

## Correction 1: there is no causal encoder–decoder split

`00_PORT_SCOPE` §3 says:

> **Causal Encoder–Decoder (CED).** 40 layers = a 20-layer causal encoder + a 20-layer decoder.
> Decoder layers do **not** compute KV from their own hidden states; they project it from the
> encoder's final hidden state `H_{L/2}` with per-layer weights. Halves prefill. Structurally
> new and touches the whole forward pass.

**That is not what the reference implements.** `Transformer.forward` (model.py:1242) is a flat
loop over 40 identical `Block`s. There is no half-split, no `H_{L/2}`, and no cross-layer KV
projection. `Attention.__init__` (model.py:622) is the same code for every layer.

What actually varies per layer is a pair of schedules:

- `compress_ratios[L]` — `0` = sliding-window attention only; `r > 0` = window **plus**
  `index_topk` compressed positions reaching further back, the latent rotated at
  `compress_rope_theta` because one latent stands for `r` tokens.
- `kv_source_layer_ids` / `index_source_layer_ids` — a layer in these lists **owns** a
  `Compressor` / `Indexer`; every other layer *reads the same cache*. The class docstring is
  explicit: *"compress_ratio > 0 does not mean the layer compresses its own KV: only
  kv_source_layers do, the rest read that same cache."*

Shipped schedule (40 backbone + 3 MTP):

| layers | compress_ratio | note |
|---|---|---|
| 0–1 | 0 | sliding window only (matches "first two layers are SWA-only") |
| 2–19 | 2 | 18 layers |
| 20–39 | 1 | 20 layers |
| 40–42 | 0 | MTP |

`kv_source_layer_ids = [2, 8, 14, 20]` (4 compressors for 38 consumers) and
`index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]` (8 indexers). This **is** the report's
"CSA2 Full / Reindex / Reuse" — Full = a source layer, Reuse = a consumer. It is a per-layer
ownership flag, not a new forward pass.

**Impact: the single largest structural risk in the port scope does not exist.** Budget it as a
per-layer `kind` struct, which is how the engine already handles V4's own two schedules
(`DeepSeek4LayerKind`).

## Correction 2: the resident budget, measured

`00_PORT_SCOPE` §4 says the backbone "exceeds available RAM by roughly 55 GiB". Measured from
the shard headers (`scratchpad/budget.py`):

| group | GiB | tensors |
|---|---:|---:|
| routed experts (MXFP4) | 268.95 | 92,160 |
| engram tables (layers 1, 14) | 189.13 | 12 |
| DSpark / MTP drafter | 7.39 | 2,401 |
| attention (FP8) | 4.72 | 520 |
| shared experts | 1.32 | 240 |
| embed | 1.23 | 1 |
| lm_head | 1.23 | 1 |
| vision tower + aligner | 0.90 | 266 |
| norms / gates / mHC | 0.29 | 441 |
| sparse indexer | 0.04 | 32 |
| CSA2 compressor | 0.03 | 11 |
| **TOTAL** | **475.24** | **96,085** |

- backbone incl. vision + MTP, excl. engram: **286.11 GiB**
- **text-only core (no vision, no MTP, no engram): 277.82 GiB**

Against 215 GiB available RAM + 59.6 GiB VRAM = 274.6 GiB, the text core is short by ~3 GiB
before any KV cache or workspace — not 55.

But the shape of the number matters more than the number: **routed experts are 268.95 of the
277.82.** Everything else in the text path is **8.87 GiB**, which fits in one card's VRAM with
50 GiB left over. So the residency problem is entirely an expert-tiering problem, which is the
problem this engine already solves for V4-Flash — plus one new tier (NVMe).

Per-expert cost is **17.93 MiB** (w1+w2+w3+scales, FP4). At `num_experts_per_tok = 6` over 40
layers, a decode token touches **4.2 GiB of expert weights** if every one misses.

## What the port actually is: a fork of `deepseek4`

V4.1's tensor vocabulary is nearly V4's, role for role:

| V4.1 | engine's `DeepSeek4Layer` |
|---|---|
| `attn.wq_a/.q_norm/.wq_b` | `attn_q_a` / `attn_q_a_norm` / `attn_q_b` |
| `attn.wkv` / `.kv_norm` | `attn_kv` / `attn_kv_a_norm` |
| `attn.attn_sink` | `attn_sinks` |
| `attn.wo_a` / `.wo_b` | `attn_output_a` / `attn_output_b` (o_groups LoRA — already there) |
| `attn.compressor.wkv/.wgate/.norm` | `compressor_kv` / `compressor_gate` / `compressor_norm` |
| `attn.indexer.wq_b/.weights_proj` | `indexer_q_b` / `indexer_proj` |
| `ffn.gate.weight/.bias` | `ffn_gate_inp` / `exp_probs_b.bias` |
| `ffn.experts.E.w1/w3/w2` | `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` |
| `ffn.shared_experts.*` | `ffn_*_shexp` |
| `hc_attn_fn/base/scale`, `hc_ffn_*` | same names, and `ds4_sinkhorn4` already implements hc_mult=4 |

Genuinely new work, in order of cost:

1. **Safetensors ingestion instead of GGUF** — done, Phase 2.
2. **FP8 E4M3 + 32×32 block scales for dense weights.** V4's dense is Q8_0/BF16. New dequant.
3. **Per-expert tensors instead of one stacked `[hidden, ffn, n_experts]`.** 92,160 separate
   tensors rather than 120. Binding and the expert-bank upload both change shape.
4. **The `kv_source` / `index_source` sharing.** New, but a flag, not a new pass.
5. **`attn.indexer.wk` + `.k_norm`** on the 4 KV-source layers — a shared indexer K, which V4
   did not have.
6. **Gate `bias_vl`** — a second routing bias selected by an image mask.
7. **Engram** at layers 1 and 14: 189 GiB of table, kilobytes read per token.
8. **NVMe expert tier.**

Deferrable without changing text outputs: vision (0.90 GiB), DSpark/MTP (7.39 GiB) — the
reference itself notes nothing calls `forward_spec`.
NOT deferrable: engram. It adds into the residual stream at two layers, so omitting it changes
every token after layer 1.

## Shapes that differ from V4 and will bite

- `head_dim = 512`, `rope_head_dim = 64` → `nope_head_dim = 448`.
- `num_key_value_heads = 1`: `wkv` is `Linear(5120 → 512)`. **One latent KV head for all 64
  query heads**, which is why the KV cache is ~890 B/token.
- `window_size = 128` — every layer keeps a 128-position raw window cache.
- `o_groups = 8`, `o_lora_rank = 1024`: `wo_a` is `[n_heads*head_dim/8 → 8*1024]` in **bf16**
  (the reference pins that one to bf16 explicitly), `wo_b` is `[8192 → 5120]`.
- `score_func = "sqrtsoftplus"`, `topk_method = noaux_tc`, `route_scale = 1.5`,
  `norm_topk_prob = true`, `swiglu_limit = 10.0` — V4's gate differs.
- `norm_eps = 1e-20` (not 1e-6).
- YaRN: `factor 16`, `original_max_position_embeddings 65536`, `rope_theta 10000`,
  `compress_rope_theta 160000`.

## Addendum 2026-09-12: hyper-connections need no new kernel

V4.1's `hc_split_sinkhorn` is a tilelang kernel, so it cannot run here — but it can be read, and
when read it is **the same algorithm the engine already implements for V4**:

| step | V4.1 `hc_split_sinkhorn_kernel` (kernel.py:407) | engine `ds4_hc_mix` + `ds4_sinkhorn4` (deepseek4_ops.cpp:93, 255) |
|---|---|---|
| pre | `sigmoid(m[j]*scale[0] + base[j]) + eps` | `sigmoid_ref(smix[s]*pre_s + base[s]) + hc_eps` |
| post | `2*sigmoid(m[j+hc]*scale[1] + base[j+hc])` | `2.f*sigmoid_ref(smix[hc+s]*post_s + base[hc+s])` |
| comb | `m[j*hc+k+2hc]*scale[2] + base[...]`, softmax over k, `+eps` | offset `8 + r*4 + c` (= 2·hc for hc=4), softmax over c, `+ hc_eps` |
| Sinkhorn | column-normalise once, then (iters−1) × (row, column), each `+eps` | identical, in registers |

Three independent implementations — DeepSeek's tilelang kernel, this engine's V4 kernel written
months earlier from V4's own reference, and the CPU transcription in
`tools/ds41_reference/kernel.py` — agree step for step. `hc_mult` is 4 in V4.1, which is exactly
the case `ds4_sinkhorn4` special-cases into registers.

So the mHC block is **reuse, not new work**. What differs is only the tensor shapes:
`hc_attn_fn` / `hc_ffn_fn` are `[24, 20480]` F32 here — `mix_hc = (2+4)*4 = 24`,
`hc_dim = 4*5120 = 20480` — with `hc_attn_base` `[24]` and `hc_attn_scale` `[3]`.
