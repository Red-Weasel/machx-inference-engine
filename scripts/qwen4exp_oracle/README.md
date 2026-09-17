# qwen4exp layer-parity oracle

Golden-tensor harness for the Qwen3.8-Flash-Next (qwen4exp) port.  Runs single
decoder blocks of the real model through the HF reference implementation
(`~/references/qwen38-flash-next/modeling_qwen4_exp.py`) on CPU in fp32
and dumps `.npz` goldens the C++ engine is compared against.  No GPU is touched;
the full 125B model is never loaded — one block at a time, with the 26.8 GiB PLE
n-gram table gathered row-sparse.

## Environment

Run everything under **`~/venv/bin/python`** (the DS4 parity venv:
torch 2.12.0+cpu, transformers 5.12.0, numpy 2.4.4).  The `gguf` package (2.2.6,
pure python) is borrowed by appending
`~/vllm-xpu/lib/python3.12/site-packages` to `sys.path` (appended last,
so the venv's own packages always win) — handled automatically by
`qwen4exp_ref.import_gguf()`.

Cosmetic note: executing the modeling file prints eight `[ERROR] ... grid_thw ...
not documented` lines from transformers' auto_docstring lint of the vision/LM
classes.  They are harmless; filter with `| grep -v grid_thw`.

## Files

| file | role |
|---|---|
| `qwen4exp_ref.py` | imports the reference modeling file into transformers 5.12.0 (synthetic `transformers.models.qwen4_exp` package + documented shims), config builder, rotary/mask helpers |
| `gguf_weights.py` | GGUF → HF fp32 weight loader for one block, inverting every converter transform; sparse PLE row reader; `--summary` CLI prints per-tensor checksums |
| `run_block_oracle.py` | block-level oracle CLI (real weights) |
| `run_op_oracle.py` | op-level oracle CLI (synthetic weights, tiny dims, no GGUF I/O) |

## CLIs

```bash
VENV=~/venv/bin/python
cd "~/00 - Inference Engine/scripts/qwen4exp_oracle"

# per-tensor weight checksums for one block (verifies GGUF load + inversions)
$VENV gguf_weights.py --layer 0 --summary

# block goldens: DeltaNet block, QSA full-attention block, PLE block
$VENV run_block_oracle.py --layer 0 --tokens 16 --seed 0 --out blk0.npz
$VENV run_block_oracle.py --layer 3 --tokens 16 --seed 0 --out blk3.npz
$VENV run_block_oracle.py --layer 1 --tokens 16 --seed 0 --with-ple --out blk1ple.npz

# op goldens (hc, deltanet rule, ple, moe) with synthetic weights
$VENV run_op_oracle.py --out-dir ops/ --seed 1234
```

Layer schedule: `il % 4 == 3` → full attention (QSA), else DeltaNet.  `--with-ple`
is valid only for `--layer 1` (config `ple_layer_ids=[2]`, 1-based).  Layer 1
without `--with-ple` sets `layer.ple = None` and dumps the pure DeltaNet+MoE path
(`ple_enabled=0` in the npz).

Everything is deterministic: same `--seed` twice ⇒ bitwise-identical npz
(verified for all dumps below).  Input draw order from the seeded generator is
fixed: (1) wide state, (2) token ids.

## What the block oracle feeds the layer

Mirrors `Qwen4ExpTextModel.forward` for a fresh, unpadded text batch:

* `hidden_states` — fixed-seed `randn` WIDE residual `[1, T, 4*2560]` fp32.
* `position_embeddings = (cos, sin)` each `[1, T, 64]` fp32 from
  `Qwen4ExpTextRotaryEmbedding(cfg)` with 2D `position_ids = arange(T)[None]`
  (the rotary expands to 3 identical T/H/W streams itself — text-only IMROPE,
  which degenerates exactly to partial NEOX rope, see docs/qwen4/14_mrope.md).
* `attention_mask` — eager 4D float causal mask `[1,1,T,T]` (0 visible,
  `finfo(f32).min` above the diagonal); the QSA indexer derives visibility from
  `mask == 0`, exactly as `create_causal_mask` would produce for eager/no padding.
* `conv_mask = None` — no padding; `apply_mask_to_padding_states` becomes the
  identity, matching the full model on an all-ones mask.
* `past_key_values = None` — fresh prefill; the DeltaNet path therefore runs the
  chunked gated delta rule (`torch_chunk_gated_delta_rule`, chunk 64), the exact
  reference prefill path.
* `ple_input_ids` (`--with-ple` only) — fixed-seed `randint(0, 248320)` `[1, T]`
  with `ids[T//2] = 248044` (EOS) to exercise the n-gram reset.
* Attention implementation: `eager` (`config._attn_implementation`), experts
  implementation: `eager` — no flash/hub kernels anywhere.

## Block npz keys

`[T,...]` arrays have the batch dim squeezed; floats are fp32.

| key | shape | meaning |
|---|---|---|
| `layer, tokens, seed, layer_type, ple_enabled` | scalars | run metadata |
| `x_wide` | [T, 10240] | input wide residual (4 streams × 2560) |
| `position_ids` | [T] | `arange(T)` |
| `cos`, `sin` | [T, 64] | rotary tables actually passed in |
| `y_wide` | [T, 10240] | layer output wide residual |
| `hc_attn_mix` | [T, 2560] | post-hc_attn-mix input handed to the token mixer |
| `hc_attn_inject_w` | [T, 4] | attn combine weights `2σ(W·xn/4)` |
| `mixer_out` | [T, 2560] | token-mixer output (DeltaNet `out_proj` / attention `o_proj`), pre-combine |
| `hc_ffn_mix` | [T, 2560] | post-hc_ffn-mix input handed to the MoE |
| `hc_ffn_inject_w` | [T, 4] | ffn combine weights |
| `moe_out` | [T, 2560] | SparseMoeBlock output pre-combine (routed + σ-gated shared) |
| `router_logits` | [T, 512] | router logits (fp32) |
| `router_top_w` | [T, 10] | top-10 routing weights, renormalized to sum 1 |
| `router_top_idx` | [T, 10] | top-10 expert indices |
| `shared_expert_out` | [T, 2560] | shared-expert SwiGLU output BEFORE the σ scalar gate |
| `ple_input_ids` | [T] | (`--with-ple`) token ids used |
| `ple_embed_E` | [T, 2560] | (`--with-ple`) concatenated n-gram embedding E |
| `ple_out` | [T, 10240] | (`--with-ple`) PLE layer output added to all 4 streams |
| `ple_layer_multipliers/…head_offsets/…head_vocab_sizes/…eos_token_id` | | hash constants, echoed from GGUF metadata after asserting they equal the module's buffers |

Combine law for reconstruction:
`y_after_attn = x_wide_after_ple + (mixer_out[:,None,:] * hc_attn_inject_w[:,:,None]).flatten(-2)`,
same for the ffn half; `moe_out = routed + σ(x·w_shexp_gate)·shared_expert_out`.

## Op npz keys (synthetic weights — the weights ARE in the npz)

Norm-weight convention in these dumps: `Qwen4ExpTextRMSNorm` weights are the HF
zero-centered `w` (applied as `1+w`; a GGUF-convention gamma equals `1+w`);
`RMSNormGated` (`deltanet ssm_norm`) weight is raw.

* `hc.npz` — H=32, hc=4, lowrank=8, T=5.  `w_norm, w_down [8,128], w_up [128,8],
  w_inject [4,128]`, `x_wide [5,128]`, synthetic `block_out [5,32]`; outputs
  `mixed [5,32]`, `inject_w [5,4]`, `combined [5,128]` (decoder-layer combine law).
  The head mixer (`use_combine=False`) is asserted identical to `mixed`.
* `deltanet_step.npz` — 2 heads, dk=dv=8, T=8, nonzero `initial_state [2,8,8]`.
  Inputs `q,k,v,g,beta` (+`initial_state`); goldens from BOTH reference rules on
  the same data: `out_recurrent/state_recurrent`
  (`torch_recurrent_gated_delta_rule`, the per-step decode recurrence, unrolled
  internally over T) and `out_chunked/state_chunked`
  (`torch_chunk_gated_delta_rule`, chunk_size=4 ⇒ two chunks).  Both use
  `use_qk_l2norm_in_kernel=True` as the model does.  Cross-agreement measured at
  2.4e-7 max abs.
* `ple.npz` — tiny PLE: H=16, hc=4, embed 32, 4 heads (2 per n-gram), vocab 1000,
  prime base 101 (⇒ head vocabs 101/103/107/109), eos 7, seed 1234.  Contains the
  full (small) embedding table, all projections/norm weights/conv kernel, hash
  buffers (`layer_multipliers`, `head_vocab_sizes`, `head_offsets`), the ids
  (EOS at T//2), input wide state, `E` (n-gram embedding) and `out` [T, 64].
* `moe.npz` — H=32, E=8, top-3, ffn 16, T=6.  Router/expert/shared weights,
  input `x`, `router_logits/top_w/top_idx`, `routed_out`, `shared_out` (pre-gate),
  `out`.  Verified: `out == routed_out + σ(x·w_shexp_gate)·shared_out` exactly and
  top-3 weights sum to 1.

## Converter inversions applied when loading the GGUF (gguf_weights.py)

Sources: `llamacpp-pr27742.diff` (conversion/qwen4exp.py) plus the inherited
converter classes read from `~/llama.cpp/conversion/qwen.py`
(`Qwen3NextModel.modify_tensors`, `_LinearAttentionVReorderBase`).

* **I1 zero-centered norms**: converter stores `gamma = 1 + w`; loader subtracts 1
  for `attn_q_norm`, `attn_k_norm`, `indexer.q_norm`, `indexer.k_norm`,
  `hc_attn_norm`, `hc_ffn_norm`, `ple_norm_key/query/conv` (the global
  `output_hc_norm` follows the same rule but belongs to the head mixer, not to any
  block, so the block loader does not touch it).
  `ssm_norm` is stored RAW (explicitly excluded by the converter rule;
  RMSNormGated's weight is ones-init, not zero-centered) — loaded unchanged.
* **I2 `ssm_a`** `= -exp(A_log)` → `A_log = log(-ssm_a)` (all 48 values verified
  negative in the GGUF).
* **I3 `ssm_dt.bias`** → renamed back to `dt_bias`.
* **I4 DeltaNet V-head reorder** (`_LinearAttentionVReorderBase`, VERIFIED from
  the class body, not inferred): HF groups V heads by K head
  (`[K0:v0..v2, K1:v0..v2, …]`), the converter re-orders to tiled
  (`gguf_head[r*16+k] = hf_head[k*3+r]`, r∈[0,3), k∈[0,16)) so ggml's tile
  broadcast replaces interleaved repeat.  The loader applies the exact inverse
  (same reshape/transpose with (3,16)) to: `attn_qkv` V **rows** (4096:10240),
  `attn_gate` rows, `ssm_alpha`/`ssm_beta` rows, `ssm_a`, `ssm_dt.bias`,
  `ssm_conv1d` V **channels**, and `ssm_out` **columns**.  A round-trip self-test
  runs on every load.
* **I5 conv kernels** stored squeezed `[C, K]` → `unsqueeze(1)` back to Conv1d
  `[C, 1, K]` (`ssm_conv1d`, `ple_conv1d`; `ple_conv1d` gets NO V-reorder).
* **I6 indexer** `index_qk_proj [640,2560]` was split into `indexer.q_proj`
  (rows 0:512) and `indexer.k_proj` (rows 512:640) → concatenated back.
* **I7 experts** `gate_up_proj [512,1280,2560]` was split into `ffn_gate_exps`
  (fused-dim 0:640) / `ffn_up_exps` (640:1280) → concatenated back on dim 1.
  `ffn_down_exps` needs no transform.
* **I8 shared gate** `ffn_gate_inp_shexp [2560]` → reshaped to `[1, 2560]`.
* **Orientation** is NOT a transform: `gguf.quants.dequantize` already returns
  `(out_features, in_features)` row-major = HF orientation (verified for 2D and
  3D expert tensors).

PLE hash constants: `config.seed = 1234` — VERIFIED by brute force:
`_build_layer_multipliers(248320, 3, 0, 1234)` reproduces the GGUF's
`qwen4exp.ple.layer_multipliers = [23703573157769, 20109073645365, 8052911324071]`
exactly.  On every layer-1 load the module's computed buffers (multipliers, 16
prime vocab sizes ≥ 20000003, offsets) are asserted equal to the GGUF metadata,
and `eos_token_id` (248044) is asserted to match `qwen4exp.ple.eos_token_id`.

## Shims applied to run the modeling file under transformers 5.12.0

Listed in `qwen4exp_ref.SHIMS_APPLIED` at runtime.  None replaces math that a
decoder-layer forward executes:

1. Synthetic package `transformers.models.qwen4_exp` + plain-namespace
   `Qwen4ExpConfig/TextConfig/VisionConfig` classes (the configuration file is not
   on disk; the layer classes only read attributes).
2. `integrations.use_kernel_func_from_hub_with_fallback` → identity decorator
   (its 5.14 behavior without a kernels hub IS the decorated torch fallback).
3. `integrations.accelerate.force_accelerate_hooks` → identity decorator
   (device_map machinery only).
4. `masking_utils.create_recurrent_attention_mask` → raising stub (only called by
   the full-model forward, which the oracle never runs; the oracle builds its own
   masks).
5. `utils.generic.get_max_seqlen` → raising stub (vision flash-attn path only).
6. `vision_utils.get_vision_attention_seqlens` /
   `get_vision_interpolation_indices_and_weights` → raising stubs (vision tower is
   never instantiated).
7. `transformers.initialization` helpers exist in 5.12.0; torch.nn.init fallbacks
   are added only if absent (used solely by `_init_weights`, which is never called
   — weights come from the GGUF or seeded randn).
8. Config additions absent from config.json: `seed=1234` (verified, see above),
   `norm_topk_prob=True` (**assumption**, see Uncertainties),
   `intermediate_size=640` (a default the model never reads),
   `_attn_implementation='eager'`, `_experts_implementation='eager'`.
9. The 320,001,536-row `nn.Embedding` is replaced at construction (layer 1 only)
   by a row-sparse stand-in that dequantizes exactly the gathered IQ4_NL rows from
   the GGUF memmap (90 B → 160 fp32 per row); an embedding lookup is an exact row
   gather, so this changes no math.

## Uncertainties (Pineapple rule)

* `norm_topk_prob=True` is an assumption: the key is absent from config.json and
  configuration_qwen4_exp.py is not on disk.  llama.cpp PR27742 hardcodes
  renormalization, and the dumped `router_top_w` sum to 1.  If HF's default were
  False, `router_top_w` and everything downstream of the MoE would change.
* The inherited converter rules were read from `~/llama.cpp/conversion/qwen.py`
  (same era as the PR), not from the PR's own base commit; the +1/-exp/reorder
  rules match everything the diff and GGUF show (ssm_a all negative, folded gammas
  centered near 1), but the PR's exact base copy was not diffed line-by-line.
* Golden values are quantization-faithful, not bf16-HF-faithful: weights are the
  dequantized UD-Q4_K_XL values in fp32, run through fp32 reference math.  That is
  exactly what the engine computes against, but do not compare these dumps to a
  bf16 HF checkpoint run.
