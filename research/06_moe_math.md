# 06 — MoE FFN Sub-Block: Exact Math for `Qwen3_5MoeForConditionalGeneration`

**Target architecture:** Qwen3.5-MoE (`Qwen3_5MoeForConditionalGeneration`), used by Qwen3.5-35B-A3B.
**Goal:** A bit-for-bit CPU reference of the MoE FFN sub-block, mirroring HuggingFace's
`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` torch path, so the C++/SYCL
kernels have a golden to diff against.

All line numbers below refer to the `main` branch of `huggingface/transformers` at the
time of writing (April 2026); they may drift by a few lines between revisions but the
class structure is stable and the math is unchanged from Qwen3-MoE / Qwen2-MoE which
share the same router/shared-expert design.

Files of record:

- `src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`
- `src/transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py`
- `src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py`
- `src/transformers/models/qwen3_moe/modeling_qwen3_moe.py` (parent design — almost identical except no shared expert)
- `gguf-py/gguf/tensor_mapping.py` in `ggml-org/llama.cpp` (GGUF name map)

---

## 1. Top-level block: `Qwen3_5MoeSparseMoeBlock`

`modeling_qwen3_5_moe.py`, lines ~1560–1594.

```python
class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.experts = Qwen3_5MoeExperts(config)
        self.shared_expert = Qwen3_5MoeMLP(
            config,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = torch.nn.Linear(
            config.hidden_size, 1, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        # 1) shared expert FFN — runs unconditionally on every token
        shared_expert_output = self.shared_expert(hidden_states_reshaped)

        # 2) router → top-k indices and (already-renormalised) weights
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)

        # 3) sparse mixture: only the chosen experts contribute
        expert_output = self.experts(
            hidden_states_reshaped, selected_experts, routing_weights
        )

        # 4) per-token scalar gate on the shared branch (sigmoid of a 1-D linear)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
            * shared_expert_output
        )

        # 5) sum sparse + gated-shared
        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output
```

Key observations for the kernel:

- **No `output_router_logits` branch in inference.** The annotated return type
  is `tuple[Tensor, Tensor]`, but the implementation returns a single tensor.
  Router logits, when training/profiling needs them, are captured externally via
  the `OutputRecorder` hook on `Qwen3_5MoePreTrainedModel._can_record_outputs`.
  At inference we **never** evaluate `load_balancing_loss_func` (see §9).
- **The shared expert runs on every token** — there is no conditional skipping.
- **Merge order is fixed:** `expert_output += sigmoid_gate * shared_output`. The
  per-token sigmoid gate is applied **only to the shared branch**, not to the
  routed branch.

---

## 2. Routing math: `Qwen3_5MoeTopKRouter`

`modeling_qwen3_5_moe.py`, lines ~1364–1382.

```python
class Qwen3_5MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok        # 8
        self.num_experts = config.num_experts          # 256
        self.hidden_dim = config.hidden_size           # 2048
        self.weight = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_dim)
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)             # [N, E]
        router_probs  = torch.nn.functional.softmax(
            router_logits, dtype=torch.float, dim=-1                     # *** fp32 softmax ***
        )
        router_top_value, router_indices = torch.topk(
            router_probs, self.top_k, dim=-1                             # top-k AFTER softmax
        )
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)   # *** unconditional renorm ***
        router_top_value = router_top_value.to(router_logits.dtype)      # cast back to bf16/fp16
        router_scores = router_top_value
        return router_logits, router_scores, router_indices
```

Direct answers to the routing questions:

| Question                                           | Answer                                                                                                                                                                                                                                                                                                                              |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Order: softmax→topk or topk→renorm?                | **softmax → topk → renorm**. The full 256-way softmax happens first, then top-8 are picked, then those 8 weights are L1-renormalised so they sum to 1.                                                                                                                                                                              |
| Are top-k weights renormalised?                    | **Yes, unconditionally** in Qwen3.5-MoE. (Note: in the older Qwen3-MoE base, this is gated by `if self.norm_topk_prob:`. Qwen3.5-MoE removed that flag — see §7.)                                                                                                                                                                   |
| Are router logits in fp32?                         | **The softmax is fp32** (`dtype=torch.float` argument). The matmul `F.linear` runs in the input dtype (bf16 in practice). The renormalised top-k weights are cast back to `router_logits.dtype` (bf16) before being returned.                                                                                                       |
| `top_k`?                                           | `config.num_experts_per_tok = 8` (set in `Qwen3_5MoeTextConfig`, line 73 of `modular_qwen3_5_moe.py`).                                                                                                                                                                                                                              |
| `num_experts`?                                     | `config.num_experts = 256` (line 74).                                                                                                                                                                                                                                                                                               |
| Linear bias?                                       | `gate.weight` is a bare `nn.Parameter` of shape `[E, H] = [256, 2048]`; `F.linear` with no bias argument means `logits = x @ W.T`. Same as `nn.Linear(H, E, bias=False)`.                                                                                                                                                            |

---

## 3. Shared expert and the mysterious `ffn_gate_inp_shexp [2048]`

The GGUF tensor `ffn_gate_inp_shexp.weight` of shape `[2048]` (F32) corresponds **exactly**
to the hypothesis: it is `Qwen3_5MoeSparseMoeBlock.shared_expert_gate.weight`, where

```python
self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)
# .weight has torch shape [1, 2048]; GGUF stores it flat as [2048]
```

The forward usage (already shown in §1) is

```python
shared_expert_output = F.sigmoid(self.shared_expert_gate(x)) * shared_expert_output
```

i.e. for each token `x_t ∈ ℝ^{2048}`,

```
α_t = sigmoid( ⟨w_shexp_gate, x_t⟩ )            ∈ ℝ, scalar
shared_t = α_t · SharedExpertSwiGLU(x_t)        ∈ ℝ^{2048}
```

The actual SwiGLU MLP (`Qwen3_5MoeMLP`, lines ~1400–1415) is:

```python
class Qwen3_5MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size: int):
        super().__init__()
        self.hidden_size       = config.hidden_size            # 2048
        self.intermediate_size = intermediate_size             # 512 for shared
        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size, bias=False)
        self.act_fn    = ACT2FN[config.hidden_act]             # "silu"

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

The shared expert uses `config.shared_expert_intermediate_size = 512` (the GGUF
`ffn_*_shexp` tensors `[2048×512]` confirm this). It is a standard SwiGLU FFN.

---

## 4. Per-expert SwiGLU: `Qwen3_5MoeExperts`

`modeling_qwen3_5_moe.py`, lines ~1418–1450 (init) and ~1087–1120 (forward; numbering
within the class file after recent refactors).

```python
@use_experts_implementation
class Qwen3_5MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts      = config.num_experts                  # 256
        self.hidden_dim       = config.hidden_size                  # 2048
        self.intermediate_dim = config.moe_intermediate_size        # 512

        # Fused gate+up: [E, 2*I, H] = [256, 1024, 2048]
        self.gate_up_proj = nn.Parameter(torch.empty(
            self.num_experts, 2 * self.intermediate_dim, self.hidden_dim
        ))
        # Down: [E, H, I] = [256, 2048, 512]
        self.down_proj = nn.Parameter(torch.empty(
            self.num_experts, self.hidden_dim, self.intermediate_dim
        ))
        self.act_fn = ACT2FN[config.hidden_act]                     # silu
```

Activation: `config.hidden_act = "silu"` (default in `Qwen3_5MoeTextConfig`). So the
activation is **SiLU** (a.k.a. Swish-1): `silu(x) = x · sigmoid(x)`. Not GELU.

The math each expert computes per token `x ∈ ℝ^{2048}`:

```
Wgu  = gate_up_proj[i]   ∈ ℝ^{1024×2048}      # fused [gate; up] stacked along dim 0
Wd   = down_proj[i]      ∈ ℝ^{2048×512}

z    = Wgu @ x           ∈ ℝ^{1024}
gate, up = z[:512], z[512:]                    # chunk(2, dim=-1)

h    = silu(gate) * up   ∈ ℝ^{512}             # element-wise
y_i  = Wd @ h            ∈ ℝ^{2048}
```

Note the GGUF layout differs: GGUF stores `ffn_gate_exps` `[2048, 512, 256]` and
`ffn_up_exps` `[2048, 512, 256]` as **separate** tensors, not as a fused
`[256, 1024, 2048]` block. The GGUF loader (or the kernel) is responsible for
either fusing them at load time or matmul'ing them separately and concatenating.

---

## 5. Combine — exact merge order

From the `forward()` in §1, the final per-token output is

```
y_t  =  Σ_{j=1..8}  w_t,j · Expert_{idx_t,j}(x_t)        # routed branch
        +  sigmoid(⟨w_shg, x_t⟩) · SharedExpert(x_t)       # gated shared branch
```

where the routed weights `w_t,1..8` are L1-normalised (sum to 1) and computed in
fp32 then cast back to the input dtype (bf16). The shared-gate scalar
`sigmoid(...)` is computed in the input dtype — there is no fp32 cast around it.

There is **no** outer sigmoid, softmax, or normalisation across the
{routed, shared} pair: they are simply added. The shared expert's *own* gate is
applied before the addition, but it does not affect the routed sum.

---

## 6. Reshape / scatter: how tokens get to experts

`Qwen3_5MoeExperts.forward()`:

```python
def forward(
    self,
    hidden_states: torch.Tensor,        # [N, H]   N = batch*seq
    top_k_index: torch.Tensor,          # [N, K]   long
    top_k_weights: torch.Tensor,        # [N, K]   float
) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        # one-hot then permute to [E, K, N]
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts
        )                                # [N, K, E]
        expert_mask = expert_mask.permute(2, 1, 0)        # [E, K, N]

        # which experts received >=1 token?
        expert_hit = torch.greater(
            expert_mask.sum(dim=(-1, -2)), 0
        ).nonzero()                                       # [E_active, 1]

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue

        # all (top_k_pos, token_idx) pairs assigned to this expert
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])   # both [n_i]

        current_state = hidden_states[token_idx]                       # [n_i, H]

        # fused gate+up matmul, then split
        gate, up = nn.functional.linear(
            current_state, self.gate_up_proj[expert_idx]               # [2I, H]
        ).chunk(2, dim=-1)                                             # each [n_i, I]

        current_hidden_states = self.act_fn(gate) * up                  # [n_i, I]

        current_hidden_states = nn.functional.linear(
            current_hidden_states, self.down_proj[expert_idx]           # [H, I]
        )                                                              # [n_i, H]

        # scale by this token's routing weight for this slot, then scatter-add
        current_hidden_states = (
            current_hidden_states
            * top_k_weights[token_idx, top_k_pos, None]                 # [n_i, 1]
        )

        final_hidden_states.index_add_(
            0, token_idx,
            current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states
```

This is the **"loop over experts, gather their tokens, scatter-add"** pattern (sometimes
called the *expert-major* or *grouped-GEMM* layout). It is **not** the
"for each token, loop over its 8 experts" layout. For the SYCL implementation:

- The per-expert token list comes from `argwhere(top_k_index == e)` — equivalent to
  a stable bucket sort of `(token_idx, slot_idx)` pairs by `top_k_index`.
- A token can hit the same expert in multiple slots only if `top_k_index` has
  duplicates, which `torch.topk` does not produce; so for a given expert all
  `token_idx` entries are unique within that expert's batch.
- The same `token_idx` will appear in *different* experts' batches — that's the whole
  point of `index_add_`. Order of scatter-adds across experts depends on iteration
  order of `expert_hit`. Since float addition isn't associative, **reproducing HF
  bit-exactly requires iterating experts in ascending index order** (which is what
  `nonzero()` produces).

---

## 7. `mlp_only_layers` and `decoder_sparse_step` — does every layer have MoE?

In `modular_qwen3_5_moe.py` lines 70–81:

```python
class Qwen3_5MoeTextConfig(Qwen3NextConfig):
    ...
    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_experts_per_tok: int = 8
    num_experts: int = 256
    intermediate_size  = AttributeError()       # not used at all
    decoder_sparse_step = AttributeError()
    norm_topk_prob      = AttributeError()
    mlp_only_layers     = AttributeError()

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        del self.mlp_only_layers
```

`AttributeError()` is a sentinel that makes the parent-class default unreachable;
`__post_init__` deletes `mlp_only_layers` from the instance entirely. And the
decoder layer, lines ~1291–1308:

```python
class Qwen3_5MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type  = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)        # *** unconditional ***
        self.input_layernorm          = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

**Confirmed:** every layer's `mlp` is a `Qwen3_5MoeSparseMoeBlock`. There is no
`if layer_idx in mlp_only_layers: dense MLP else: MoE` branch — Qwen3.5-MoE
removed that knob entirely. Your GGUF inspect (`mlp_only_layers: []`) is
consistent with this. The hybrid character of Qwen3.5 lives on the **attention**
side (`layer_types[i] ∈ {"linear_attention", "full_attention"}`, GatedDeltaNet
vs. softmax MHA), not on the FFN side.

For the older Qwen3-MoE base class (`Qwen3MoeTopKRouter` in `qwen3_moe`), there
*is* a `norm_topk_prob` flag and a `mlp_only_layers` list. Qwen3.5-MoE drops
both: renormalisation is on, every layer is MoE.

---

## 8. Tensor name map: PyTorch ↔ GGUF

From `gguf-py/gguf/tensor_mapping.py` plus the PyTorch class shapes we just
walked through:

| GGUF tensor (per layer)        | Shape (per GGUF metadata)         | PyTorch parameter                                                       | PyTorch shape           |
| ------------------------------ | --------------------------------- | ----------------------------------------------------------------------- | ----------------------- |
| `ffn_gate_inp.weight`          | F32 `[2048, 256]` (= `[H, E]`)    | `model.layers.{L}.mlp.gate.weight`                                      | `[256, 2048]` = `[E, H]` |
| `ffn_gate_inp_shexp.weight`    | F32 `[2048]`                      | `model.layers.{L}.mlp.shared_expert_gate.weight`                        | `[1, 2048]`             |
| `ffn_gate_exps.weight`         | Q4_K `[2048, 512, 256]`           | `model.layers.{L}.mlp.experts.gate_up_proj` *(top half of dim 1)*       | `[256, 1024, 2048]` fused |
| `ffn_up_exps.weight`           | Q4_K `[2048, 512, 256]`           | `model.layers.{L}.mlp.experts.gate_up_proj` *(bottom half of dim 1)*    | `[256, 1024, 2048]` fused |
| `ffn_down_exps.weight`         | Q4_K `[512, 2048, 256]`           | `model.layers.{L}.mlp.experts.down_proj`                                | `[256, 2048, 512]`      |
| `ffn_gate_shexp.weight`        | Q4_K `[2048, 512]`                | `model.layers.{L}.mlp.shared_expert.gate_proj.weight`                   | `[512, 2048]`           |
| `ffn_up_shexp.weight`          | Q4_K `[2048, 512]`                | `model.layers.{L}.mlp.shared_expert.up_proj.weight`                     | `[512, 2048]`           |
| `ffn_down_shexp.weight`        | Q4_K `[512, 2048]`                | `model.layers.{L}.mlp.shared_expert.down_proj.weight`                   | `[2048, 512]`           |

Notes on the fused split:

- HF stores `gate_up_proj[e] ∈ ℝ^{1024 × 2048}`, where rows `0..511` are the
  gate projection and rows `512..1023` are the up projection. The line
  `gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)` produces
  `gate = (Wgu @ x)[:512]`, `up = (Wgu @ x)[512:]`.
- `convert_hf_to_gguf.py` un-fuses this when writing GGUF, producing two separate
  tensors `ffn_gate_exps` and `ffn_up_exps`. The mapping `MODEL_TENSOR.FFN_GATE_EXP`
  and `MODEL_TENSOR.FFN_UP_EXP` in `tensor_mapping.py` accept both the legacy
  `experts.{i}.gate_proj.weight` form and the modern fused
  `experts.gate_up_proj` form (handled by the converter splitting along dim 1).
- Shape conventions differ: GGUF tensor shapes are listed in **column-major /
  GGML order** (innermost first), so GGUF `[2048, 512, 256]` corresponds to
  PyTorch `[256, 512, 2048]` in `(out, in)` semantics. The converter
  transposes/permutes so that `Wgu @ x` semantics are preserved.
- `ffn_gate_inp_shexp` is exactly the `[1, 2048]` weight of
  `shared_expert_gate = nn.Linear(2048, 1, bias=False)`, flattened to `[2048]`
  in GGUF.

`tensor_mapping.py` excerpts:

```python
# FFN_GATE_INP_SHEXP
"model.layers.{bid}.mlp.shared_expert_gate",          # qwen2moe / qwen3moe / qwen3_5_moe

# FFN_GATE_SHEXP
"model.layers.{bid}.mlp.shared_expert.gate_proj",     # qwen2moe / qwen3_5_moe
"model.layers.{bid}.mlp.shared_experts.gate_proj",    # deepseek

# FFN_UP_SHEXP
"model.layers.{bid}.mlp.shared_expert.up_proj",
# FFN_DOWN_SHEXP
"model.layers.{bid}.mlp.shared_expert.down_proj",
```

---

## 9. Aux loss

`load_balancing_loss_func` lives at lines ~2051–2126 of `modeling_qwen3_5_moe.py`
and is the same Switch-Transformer auxiliary loss used in Mixtral/Qwen-MoE
(mean over layers and experts of `f_i · P_i`, scaled by
`config.router_aux_loss_coef = 0.001`).

It is invoked **only** from the model-level `forward()` when both `labels` are
provided **and** `config.output_router_logits` is truthy (`output_router_logits`
defaults to `False` in `Qwen3_5MoeTextConfig`). In pure inference (no labels,
default config) the function is never called, the router logits aren't even
collected by the `OutputRecorder`, and there is no add to the final loss.

**For the inference engine: no-op. Skip it entirely.**

---

## 10. Dtype rules

| Step                              | dtype                                                                                                  |
| --------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `router_logits = x @ W_gate.T`    | input dtype (bf16)                                                                                     |
| `softmax(router_logits)`          | **fp32** (`dtype=torch.float` argument; computed in fp32, returned in fp32)                            |
| `topk(router_probs, 8)`           | fp32 (operating on the fp32 softmax output)                                                            |
| `topk_probs /= topk_probs.sum()`  | **fp32** (division on fp32 tensor)                                                                     |
| `topk_probs.to(router_logits.dtype)` | cast back to input dtype (bf16) — this is what the experts and the scatter-add see              |
| Per-expert `gate_proj`, `up_proj`, `down_proj` matmuls | input dtype (bf16) for activations; weight dtype is whatever the storage is (Q4_K dequant→bf16)        |
| `silu(gate) * up`                 | input dtype (bf16)                                                                                     |
| `current_hidden_states *= w_t,j`  | input dtype (bf16); `w_t,j` was cast back to bf16 above                                                |
| `final.index_add_(...)`           | input dtype (bf16); explicit `.to(final.dtype)` cast in the source                                     |
| `sigmoid(shared_expert_gate(x))`  | input dtype (bf16)                                                                                     |
| `shared_output * sigmoid_gate`    | input dtype (bf16)                                                                                     |
| Final `expert_output + shared_output` | input dtype (bf16)                                                                                 |

**The only fp32 island is the router softmax + sum-to-1 normalisation.** Everything
else runs at the input activation dtype. To bit-match HF on CPU you must:

1. Promote the router softmax to fp32 (compute `exp`/sum/divide in fp32).
2. Renormalise top-8 in fp32.
3. Cast back to bf16 (or whatever `hidden_states.dtype` is) before any multiply
   into the expert outputs.
4. Iterate experts in ascending index order so the `index_add_` order matches.

---

## 11. Config defaults summary (Qwen3.5-35B-A3B)

From `modular_qwen3_5_moe.py` `Qwen3_5MoeTextConfig` and parent `Qwen3NextConfig`:

| Field                              | Default                         |
| ---------------------------------- | ------------------------------- |
| `hidden_size`                      | 2048                            |
| `num_hidden_layers`                | 40                              |
| `num_experts`                      | 256                             |
| `num_experts_per_tok`              | 8                               |
| `moe_intermediate_size`            | 512                             |
| `shared_expert_intermediate_size`  | 512                             |
| `hidden_act`                       | `"silu"`                        |
| `output_router_logits`             | `False`                         |
| `router_aux_loss_coef`             | 0.001 (training only)           |
| `mlp_only_layers`                  | **deleted** in `__post_init__`  |
| `decoder_sparse_step`              | unused (`AttributeError()`)     |
| `norm_topk_prob`                   | unused (`AttributeError()`); router renormalises unconditionally |
| `intermediate_size`                | unused (`AttributeError()`); only the per-expert `moe_intermediate_size` matters |

---

## 12. 8-step CPU reference algorithm

A direct C++ host-golden transcription. Inputs per call:

- `X ∈ bf16^{N × H}` with `H = 2048`, `N = batch * seq`.
- Per-layer weights: `W_gate ∈ bf16^{E×H}` (router), `W_shg ∈ bf16^{1×H}`
  (shared expert gate), expert tensors `Wgu[e] ∈ bf16^{2I × H}` and
  `Wd[e] ∈ bf16^{H × I}` for `e ∈ 0..E-1`, and the shared expert
  `Wg_sh, Wu_sh ∈ bf16^{Is × H}`, `Wd_sh ∈ bf16^{H × Is}`.
- Constants: `E = 256`, `K = 8`, `I = 512`, `Is = 512`, `H = 2048`.

Pseudocode (one pass over `N` tokens):

```
INPUT : X[N, H]   (bf16)
OUTPUT: Y[N, H]   (bf16)

# Step 1 — Router logits (bf16 matmul)
logits[N, E] = X @ W_gate^T                                                # bf16

# Step 2 — Softmax in fp32, full E=256 axis
for t in 0..N-1:
    m = max(logits[t, :])                            # bf16 → fp32
    s = 0.0_f32
    for e in 0..E-1:
        p32[t, e] = expf(float(logits[t, e]) - float(m))                   # fp32
        s += p32[t, e]
    inv = 1.0 / s
    for e in 0..E-1: p32[t, e] *= inv                                      # fp32 probs

# Step 3 — Top-K=8 (fp32) per token
for t in 0..N-1:
    (idx[t, 0..K-1], val32[t, 0..K-1]) = topk_desc(p32[t, :], K)          # K largest
    s = sum(val32[t, :])                                                   # fp32
    for k in 0..K-1: val32[t, k] /= s                                      # renorm
    for k in 0..K-1: w_bf16[t, k] = bf16(val32[t, k])                      # cast back

# Step 4 — Build expert→tokens buckets (stable, ascending expert order)
bucket = [ [] for _ in 0..E-1 ]                                            # list of (t, k)
for t in 0..N-1:
    for k in 0..K-1: bucket[idx[t, k]].append( (t, k) )

# Step 5 — Routed branch: loop experts in 0..E-1 order, scatter-add
Y[N, H] = 0
for e in 0..E-1:
    if bucket[e].empty(): continue
    n_e = len(bucket[e])
    Xe[n_e, H]   = gather rows X[t]   for (t,k) in bucket[e]
    Z[n_e, 2I]   = Xe @ Wgu[e]^T                                            # bf16
    G = Z[:, 0:I]; U = Z[:, I:2I]
    Hh[n_e, I]   = silu(G) * U                                              # element-wise
    R[n_e, H]    = Hh @ Wd[e]^T                                             # bf16
    for j in 0..n_e-1:
        (t, k) = bucket[e][j]
        scale = w_bf16[t, k]
        Y[t, :] += scale * R[j, :]                                          # bf16 add

# Step 6 — Shared expert SwiGLU on every token
Gs[N, Is]    = X @ Wg_sh^T
Us[N, Is]    = X @ Wu_sh^T
Hs[N, Is]    = silu(Gs) * Us
Sh[N, H]     = Hs @ Wd_sh^T

# Step 7 — Per-token shared gate, sigmoid scalar
for t in 0..N-1:
    a_t = sigmoid( <W_shg[0, :], X[t, :]> )                                 # bf16 scalar
    Sh[t, :] *= a_t

# Step 8 — Sum the two branches
Y += Sh
return Y
```

Bit-exactness checklist for the CPU golden:

- [ ] Use IEEE-754 `float` (fp32) for steps 2–3 (softmax, sum, renorm).
- [ ] Use bfloat16 (or whatever the model dtype is) for everything else,
      with the **same rounding mode** (round-to-nearest-even) HF/torch uses.
- [ ] Iterate experts in ascending index order in step 5; do **not**
      re-order by bucket size or by token-major.
- [ ] Within an expert, accumulate into `Y[t, :]` in the order tokens
      appear in `bucket[e]` (which is the order produced by
      `torch.where(expert_mask[e])` — equivalently, scanning `(t, k)`
      lexicographically in `(t, k)`).
- [ ] `silu(x) = x * sigmoid(x)`, with `sigmoid(x) = 1/(1+exp(-x))`.
- [ ] No bias term anywhere (router, shared-gate, gate/up/down projections
      are all `bias=False`).
- [ ] Skip aux loss / output_router_logits paths.
- [ ] Verify dtype on every multiply: routing weight `w_bf16[t,k]` is bf16,
      not fp32, before it multiplies the expert output.

This algorithm mirrors §12 of `research/05_deltanet_math.md`'s structure and is
ready to transcribe into the C++/SYCL host reference.
