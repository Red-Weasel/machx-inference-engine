# Qwen3.5-MoE Gated DeltaNet — Exact Math Reference

> Target: `Qwen3_5MoeForConditionalGeneration` (HF transformers `main`, snapshot Apr 2026).
> Audience: a C++/SYCL implementer who needs a CPU golden reference that matches HF
> token-for-token. Every equation here is anchored to a line in the HF source.

## 0. Source files (priority order)

| # | File (transformers repo, branch `main`) | Role |
|---|---|---|
| 1 | `src/transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py` (329 L) | source-of-truth for Qwen3.5-MoE; subclasses Qwen3.5 + Qwen3-Next |
| 2 | `src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` (2400 L) | auto-generated, has the full inlined `Qwen3_5MoeGatedDeltaNet` |
| 3 | `src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py` | `Qwen3_5MoeTextConfig` defaults |
| 4 | `src/transformers/models/qwen3_5/modular_qwen3_5.py` | `Qwen3_5GatedDeltaNet` — the parent class that introduces the un-fused QKV/Z/B/A projections |
| 5 | `src/transformers/models/qwen3_next/modular_qwen3_next.py` | `Qwen3NextGatedDeltaNet` — defines reference torch fallbacks (`torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`, `torch_causal_conv1d_update`, `Qwen3NextRMSNormGated`) |
| 6 | `src/transformers/cache_utils.py::LinearAttentionLayer` | conv-state and recurrent-state cache logic |

The class hierarchy collapses to:

```
Qwen3_5MoeGatedDeltaNet  (qwen3_5_moe/modeling_qwen3_5_moe.py:357)
  └── inherits Qwen3_5GatedDeltaNet  (qwen3_5/modular_qwen3_5.py:190)
        └── inherits Qwen3NextGatedDeltaNet  (qwen3_next/modular_qwen3_next.py:335)
```

`Qwen3_5MoeGatedDeltaNet` is a `pass`-only subclass (modular line 158), so the auto-generated body in `modeling_qwen3_5_moe.py:357-534` IS the implementation. All line numbers below refer to that auto-generated file unless stated.

## 1. Config defaults (Qwen3.5-35B-A3B, model_type `qwen3_5_moe_text`)

From `configuration_qwen3_5_moe.py:79-119`:

```python
hidden_size            = 2048
num_hidden_layers      = 40
hidden_act             = "silu"
rms_norm_eps           = 1e-6
attention_bias         = False
linear_conv_kernel_dim = 4         # conv kernel size
linear_key_head_dim    = 128       # head_k_dim
linear_value_head_dim  = 128       # head_v_dim
linear_num_key_heads   = 16        # num_k_heads
linear_num_value_heads = 32        # num_v_heads (so v/k group ratio = 2)
partial_rotary_factor  = 0.25      # for full-attn layers only
# layer_types schedule (auto):
#   layer i is "full_attention" iff (i+1) % 4 == 0, else "linear_attention"
#   ⇒ layers 3, 7, 11, ..., 39 are softmax-attn; the other 30 are DeltaNet.
```

Derived dims for the linear-attention block:

```
key_dim   = num_k_heads * head_k_dim = 16 * 128 = 2048
value_dim = num_v_heads * head_v_dim = 32 * 128 = 4096
conv_dim  = key_dim*2 + value_dim    = 2048 + 2048 + 4096 = 8192
```

These match the GGUF tensor shapes the user listed (Q=2048, K=2048, V=4096, conv channels=8192).

## 2. GGUF ↔ PyTorch tensor name map

| GGUF tensor name (per layer) | Shape | PyTorch attr | Init / role |
|---|---|---|---|
| `attn_qkv.weight` | `[8192, 2048]` (out, in) | `linear_attn.in_proj_qkv.weight` | fused Q‖K‖V projection (`nn.Linear`, no bias). Split sizes `[2048, 2048, 4096]`. |
| `attn_gate.weight` | `[4096, 2048]` | `linear_attn.in_proj_z.weight` | output gate `z` (also called the "second pre-norm gate"). |
| `ssm_alpha.weight` | `[32, 2048]` | `linear_attn.in_proj_a.weight` | per-head decay logits `a` (shape `num_v_heads = 32`). |
| `ssm_beta.weight` | `[32, 2048]` | `linear_attn.in_proj_b.weight` | per-head delta-rule write strength `b`. |
| `ssm_conv1d.weight` | `[8192, 4]` (channels-major, kernel 4) | `linear_attn.conv1d.weight` (`Conv1d` shape `[8192, 1, 4]`) | depthwise causal conv (`groups=conv_dim`, `bias=False`). |
| `ssm_a` | `[32]` | `linear_attn.A_log` | per-head log of A. Init: `A ~ U(0, 16)`, stored as `log(A)`. |
| `ssm_dt.bias` | `[32]` | `linear_attn.dt_bias` | per-head softplus bias. Init: ones. |
| `ssm_norm.weight` | `[128]` | `linear_attn.norm.weight` | gated RMSNorm scale, applied per `head_v_dim`. Init: ones. (Note: this is a **plain** RMSNorm, NOT the `(1+w)` Gemma-style.) |
| `ssm_out.weight` | `[2048, 4096]` | `linear_attn.out_proj.weight` | output projection back to hidden_size. No bias. |

The Qwen3-Next checkpoint uses fused `ssm_beta_alpha [64, 2048]` (PR #16095, llama.cpp). Qwen3.5-MoE **un-fuses** it into `ssm_alpha` and `ssm_beta`.

`conv1d.bias` — note the modular passes `self.conv1d.bias` everywhere (line 459, 470 in `modeling_qwen3_5_moe.py`), but the constructor sets `bias=False` (line 379). So `self.conv1d.bias is None` and the underlying causal-conv calls are simply called with `bias=None`.

## 3. Forward pass — full ordering

This is the **complete** body of `Qwen3_5MoeGatedDeltaNet.forward()` from `modeling_qwen3_5_moe.py:423-534`:

```python
def forward(self, hidden_states, cache_params=None, attention_mask=None):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)  # zeroes pads
    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = (
        cache_params is not None and cache_params.has_previous_state(self.layer_idx) and seq_len == 1
    )

    if use_precomputed_states:
        conv_state = cache_params.layers[self.layer_idx].conv_states          # [B, 8192, 4]
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states # [B, 32, 128, 128] fp32

    # 1) Linear projections from hidden_states  (no activation yet, no norm yet)
    mixed_qkv = self.in_proj_qkv(hidden_states)             # [B, T, 8192]   (= attn_qkv)
    mixed_qkv = mixed_qkv.transpose(1, 2)                   # [B, 8192, T]   (channels-first for Conv1d)

    z = self.in_proj_z(hidden_states)                       # [B, T, 4096]   (= attn_gate)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim) # [B, T, 32, 128]

    b = self.in_proj_b(hidden_states)                       # [B, T, 32]     (= ssm_beta)
    a = self.in_proj_a(hidden_states)                       # [B, T, 32]     (= ssm_alpha)

    # 2) Depthwise causal Conv1d on the fused QKV stream + SiLU
    if use_precomputed_states:                              # decode (T=1)
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv, conv_state,
            self.conv1d.weight.squeeze(1),                  # [8192, 4]
            self.conv1d.bias,                               # None
            self.activation,                                # "silu"
        )                                                   # → [B, 8192, 1], conv_state updated in-place
    else:                                                   # prefill / training
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,                 # SiLU baked into the kernel
                seq_idx=None,
            )
        else:
            # Pure-torch fallback path:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    # 3) Split fused stream into Q, K, V
    mixed_qkv = mixed_qkv.transpose(1, 2)                   # [B, T, 8192]
    query, key, value = torch.split(
        mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
    )                                                       # [B,T,2048] [B,T,2048] [B,T,4096]
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)   # [B, T, 16, 128]
    key   = key.reshape(  batch_size, seq_len, -1, self.head_k_dim)   # [B, T, 16, 128]
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)   # [B, T, 32, 128]

    # 4) Per-head data-dependent scalars
    beta = b.sigmoid()                                                # [B, T, 32]
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # [B, T, 32]
    # Note: g is the LOG decay (will be exp'd inside the kernel). Always negative.

    # 5) Replicate K, Q for v/k head ratio = 2 (broadcast key heads up to value-head count)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)  # → 32 heads
        key   = key.repeat_interleave(  self.num_v_heads // self.num_k_heads, dim=2)  # → 32 heads

    # 6) Recurrence
    if not use_precomputed_states:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=None,                      # always None — chunked path doesn't take a prior state
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,            # ← L2-normalize Q and K (not softmax)
        )
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    # 7) Gated RMSNorm fused with SiLU(z) gate
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)        # [B*T*32, 128]
    z             = z.reshape(           -1, self.head_v_dim)         # [B*T*32, 128]
    core_attn_out = self.norm(core_attn_out, z)                       # RMSNorm(core) * SiLU(z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)    # [B, T, 4096]

    # 8) Output projection
    output = self.out_proj(core_attn_out)                             # [B, T, 2048]
    return output
```

## 4. Depthwise Conv1d — exact spec

Constructor (`modeling_qwen3_5_moe.py:376-383`):

```python
self.conv_dim = self.key_dim * 2 + self.value_dim                # 8192
self.conv1d = nn.Conv1d(
    in_channels  = self.conv_dim,                                # 8192
    out_channels = self.conv_dim,                                # 8192
    bias         = False,
    kernel_size  = self.conv_kernel_size,                        # 4
    groups       = self.conv_dim,                                # 8192 ⇒ depthwise
    padding      = self.conv_kernel_size - 1,                    # 3 (causal padding when truncated to seq_len)
)
```

Key facts:

- **All 8192 channels** go through the conv (Q ∥ K ∥ V together, never split before the conv).
- **Layout**: PyTorch `Conv1d` is channels-first. The implementation `transpose(1,2)` to `[B, conv_dim, T]` before, then back after.
- **Causal padding**: `padding=kernel-1=3` is applied symmetrically by `nn.Conv1d`, then the slice `[:, :, :seq_len]` (line 475) **drops the trailing `kernel-1` outputs**, leaving a left-padded causal conv of length `T`.
- **SiLU**: applied **after** the conv, **before** the QKV split.
- **Per-channel weights**: `conv1d.weight` has shape `[8192, 1, 4]` in PyTorch; GGUF stores it as `[4, 8192]` (kernel × channels). After `weight.squeeze(1)` you get `[8192, 4]`, which is what `causal_conv1d_fn` expects.

The Q-only / K-only / V-only flags are **not** used here — the same kernel is applied uniformly to all three streams.

### 4a. Conv1d torch reference (the slow path you need to match)

From `torch_causal_conv1d_update` (`modeling_qwen3_5_moe.py:211-226`):

```python
def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    # hidden_states: [B, conv_dim, 1]    (decode)
    # conv_state   : [B, conv_dim, kernel_size]    (running window)
    # weight       : [conv_dim, kernel_size]
    _, hidden_size, seq_len = hidden_states.shape          # seq_len=1 in decode
    state_len = conv_state.shape[-1]                       # = 4

    # 1) shift in the new sample(s)
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1)
    conv_state.copy_(hidden_states_new[:, :, -state_len:]) # update window in-place
    # 2) depthwise conv (groups = channels)
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    # 3) SiLU on last `seq_len` outputs
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)
```

For prefill (no precomputed state), the fallback is just (line 475):

```python
mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
```

That's a single `nn.Conv1d` call with `padding=3`, then truncate to `seq_len`, then SiLU.

## 5. Q / K activation and normalization

There is **no separate q_norm / k_norm RMSNorm** inside the DeltaNet block. (Qwen3.5 full-attention layers use `q_norm/k_norm` with head-dim eps; DeltaNet does **not**.) Instead:

1. The **only nonlinearity** applied to Q, K, V before the recurrence is `SiLU` baked into the conv (step 2 above).
2. The kernel is told `use_qk_l2norm_in_kernel=True` (lines 508, 520). This applies an **L2 normalization** to Q and K **inside** the recurrence kernel:

```python
def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm
```

Applied per `head_k_dim=128` slot (the last dim). So the actual queries/keys fed to the recurrence are `q̂ = q / ‖q‖₂`, `k̂ = k / ‖k‖₂`.

**Value V is NOT normalized.** Only Q and K. (Lines 247-249 of `torch_chunk_gated_delta_rule`, lines 319-321 of `torch_recurrent_gated_delta_rule`.)

There is also a `scale = 1 / sqrt(head_k_dim)` applied to Q **inside** both kernels (lines 263 and 328). After L2-norm, Q ends up with magnitude `1/sqrt(128) ≈ 0.0884` per head.

## 6. Per-head decay g and write-strength β

```python
beta = b.sigmoid()                                                 # ∈ (0, 1), shape [B, T, 32]
g    = -A_log.exp() * softplus(a.float() + dt_bias)                # ≤ 0,    shape [B, T, 32]
```

Notes:

- `A_log` ∈ ℝ³², parameter, init = `log(U(0,16))`. So `A = A_log.exp()` ∈ (0, 16), positive.
- `dt_bias` ∈ ℝ³², init = `1`s.
- `softplus(x) = log(1 + exp(x))` ≥ 0.
- Therefore **`g ≤ 0` always**. `g` is a **log-decay**: the kernel uses `exp(g)` ∈ (0, 1] as the per-step state-decay multiplier. With `dt_bias=1` and `a≈0`, softplus≈1.31, so the floor decay per step is `exp(-A · 1.31)`.
- **`g` is a per-head scalar**, not per-state-element. There are 32 values per token (one per V-head), broadcast across `head_k_dim=128` and `head_v_dim=128` of the state.

The `.float()` casts are essential — without them, fp16 loads of `A_log` underflow to `-inf` (the source comment on line 493 says: *"If the model is loaded in fp16, without the .float() here, A might be -inf"*).

## 7. The gated delta rule — recurrent form (decode path)

The single-token recurrent torch reference (`torch_recurrent_gated_delta_rule`, `modeling_qwen3_5_moe.py:315-354`) is the unambiguous mathematical specification. After the per-tensor `transpose(1,2).contiguous().to(torch.float32)`, shapes are:

```
query, key:    [B, H=32, T, K=128]   (after l2norm if use_qk_l2norm_in_kernel)
value:         [B, H=32, T, V=128]
g:             [B, H=32, T]           (log-decay)
beta:          [B, H=32, T]           (sigmoid write strength)
last_recurrent_state: [B, H=32, K=128, V=128]   (state matrix per head)
```

The loop (line 338-349):

```python
scale = 1 / (K**0.5)                       # = 1/sqrt(128) ≈ 0.0884
query = query * scale                      # applied ONCE before the loop

for i in range(seq_len):
    q_t = query[:, :, i]                   # [B, H, K]
    k_t = key  [:, :, i]                   # [B, H, K]
    v_t = value[:, :, i]                   # [B, H, V]
    g_t   = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)   # [B, H, 1, 1]
    beta_t = beta[:, :, i].unsqueeze(-1)                   # [B, H, 1]

    # Step 1: state decay
    last_recurrent_state = last_recurrent_state * g_t       # S ← S · exp(g_t)

    # Step 2: read-out of what the state currently predicts for this key
    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)   # [B, H, V]
    # equivalent to: kv_mem[b,h,v] = sum_k S[b,h,k,v] * k_t[b,h,k]   (i.e. k_tᵀ · S)

    # Step 3: delta = error · write-strength
    delta = (v_t - kv_mem) * beta_t                          # [B, H, V]

    # Step 4: rank-1 update
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    # equivalent to: S ← S + k_t ⊗ delta   (outer product on dims K, V)

    # Step 5: query the state
    core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    # equivalent to: out[b,h,v] = sum_k S[b,h,k,v] * q_t[b,h,k]
```

### As equations

Let the per-head state at time t be `S_t ∈ ℝ^{K×V}`. With `q_t, k_t ∈ ℝ^K` (after L2-norm), `v_t ∈ ℝ^V`, `α_t = exp(g_t) ∈ (0,1]`, `β_t ∈ (0,1)`:

```
   S_t = α_t · S_{t-1}  +  k_t ⊗ ( β_t · (v_t − k_tᵀ · (α_t · S_{t-1})) )
   y_t = (q_t / √K)ᵀ · S_t                               ∈ ℝ^V
```

Equivalently, in the standard delta-rule form:

```
   v̂_t  = k_tᵀ S_{t-1}'                  with S_{t-1}' = α_t · S_{t-1}      # current prediction
   ε_t  = v_t − v̂_t                                                          # error
   S_t  = S_{t-1}'  +  β_t · k_t ⊗ ε_t                                       # corrective rank-1 update
   y_t  = (q_t / √K)ᵀ S_t
```

Sign convention answers:

- **α (the user's `α` in the brief) ↔ `exp(g_t)`**. **Per-head scalar**, broadcast across both K and V dims of the state. Always in `(0, 1]`.
- **β** is **per-head scalar** ∈ `(0,1)`.
- **Sign of the delta-rule update**: the write is `+ β · k ⊗ (v − v̂)`. There is no negative term in the recurrent form. The user's pseudocode `state · α  −  state @ kᵀ k β  +  β v kᵀ` is mathematically identical — expanding the equation above:
  `S_t = α S_{t-1} + β k ⊗ v − β k ⊗ (kᵀ α S_{t-1}) = α S_{t-1} − β k kᵀ (α S_{t-1}) + β k vᵀ`.
- **Q is scaled by 1/√K** (i.e. 1/√128), **once**, before the loop. K is **not** scaled here (the scale lives only on Q).
- **L2-norm** is applied to Q and K outside the loop (lines 319-321), before the `scale = 1/√K` and before the loop.
- **State dtype**: the kernel casts query/key/value/beta/g to `torch.float32` (line 322-324), so the loop, the state, and the readout all run in **fp32**. Output is cast back to `initial_dtype` at line 353.

### State storage

```
last_recurrent_state: torch.Tensor of shape (B, H_v=32, K=128, V=128)  in fp32
```

That's `32 · 128 · 128 · 4 bytes = 2 MiB per layer per batch element` for the recurrent state. With 30 linear-attention layers, ≈ 60 MiB/batch/sequence. Confirmed by `torch_recurrent_gated_delta_rule` line 309-314.

## 8. The chunked path (prefill)

The prefill path uses `torch_chunk_gated_delta_rule` (`modeling_qwen3_5_moe.py:235-312`). It is **mathematically equivalent** to the recurrent form but processed in fixed `chunk_size = 64` blocks, leaning on a `chunk_size × chunk_size` triangular solve to amortize the rank-1 updates.

It is initial-state-agnostic in this codebase: line 506 hard-codes `initial_state=None` for the chunk path. (Qwen3.5 always re-prefills from scratch; subsequent decode tokens go through the recurrent path.)

Outline (skip this if you only need decode parity):

1. Pad `T` up to a multiple of 64; pad Q/K/V/β/g.
2. Apply `scale = 1/√K` to Q.
3. Compute `v_β = v · β`, `k_β = k · β`. Reshape into chunks `[B, H, n_chunks, 64, *]`.
4. Cumulate `g` *within each chunk*; `g_cum[c, i, j] = sum_{l≤i} g[c, l]` then `decay_mask[c, i, j] = exp(g_cum[c,i] − g_cum[c,j])` for `i ≥ j`.
5. Build a `64×64` lower-triangular `attn` matrix:  
   `attn = -((k_β @ kᵀ) ⊙ decay_mask)` masked above the diagonal.  
   Then iteratively solve `attn[i,:i] += attn[i,:i] @ attn[:i,:i]` for `i=1..63` (forward substitution), then add `I_64`. This is the closed-form solution for the within-chunk recurrence.
6. `value_chunk = attn @ v_β`; `k_cumdecay = attn @ (k_β · exp(g_cum))`.
7. For each chunk: pull from previous state (`v_prime = k_cumdecay @ S_prev`), subtract (`v_new = v - v_prime`), compute intra-chunk attention `(q @ kᵀ) · decay_mask`, blend `attn_inter = (q · exp(g_cum)) @ S_prev`, then `out = attn_inter + attn @ v_new`. Update `S ← S · exp(g_cum[-1]) + (k · exp(g_cum[-1] - g_cum))ᵀ @ v_new`.

For your CPU golden, **just use the recurrent loop for any prefill length**. It produces bit-equivalent output to the chunked kernel because both paths implement the same scalar recurrence (only their reduction order differs, and the kernels run in fp32). The recurrent loop is `O(T·H·K·V)` which is fine for golden generation up to a few thousand tokens.

## 9. Gated RMSNorm — `Qwen3_5MoeRMSNormGated`

Defined at `modeling_qwen3_5_moe.py:176-191`:

```python
class Qwen3_5MoeRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))     # init: ones
        self.variance_epsilon = eps                              # 1e-6

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)               # over last dim = 128
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)          # cast back, then scale
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))       # SiLU(z) gate, fp32 mul
        return hidden_states.to(input_dtype)
```

Crucial differences from `Qwen3_5MoeRMSNorm` (the *non-gated* variant used elsewhere):

| Aspect | `Qwen3_5MoeRMSNorm` (e.g. `input_layernorm`) | `Qwen3_5MoeRMSNormGated` (DeltaNet `self.norm`) |
|---|---|---|
| Param init | zeros | **ones** |
| Scale form | `(1.0 + weight) * x_normed` (Gemma3 style) | `weight * x_normed` (plain) |
| Gate | none | yes — multiplies by `SiLU(gate)` post-scale |
| Eps | `rms_norm_eps = 1e-6` | `1e-6` |
| Dim | `hidden_size = 2048` | `head_v_dim = 128` |

So your earlier note (research/04) about `(1 + weight)` applies to the **block-level** RMSNorms, **not** to `ssm_norm`. The `ssm_norm.weight [128]` GGUF tensor is used as-is (`weight * x_normed`) with no `+1` offset. Init of ones means the freshly-initialized norm is identity (before SiLU gate).

Equation (per row):

```
   x_normed = x / sqrt(mean(x²) + 1e-6)        # norm dim = 128 (head_v_dim)
   y        = (weight ⊙ x_normed) * SiLU(z)
            = (weight ⊙ x_normed) * (z · sigmoid(z))
```

Cast detail: variance and rsqrt run in fp32. The scale `weight *` is applied **after** casting back to the input dtype. The SiLU gate multiply is in fp32, and the final cast to input dtype happens at the end. (Line 188-191.)

The reshape choreography (`modeling_qwen3_5_moe.py:527-531`):

```python
core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)   # flatten to [B*T*32, 128]
z             = z.reshape(-1, self.head_v_dim)               # [B*T*32, 128]
core_attn_out = self.norm(core_attn_out, z)
core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)  # [B, T, 32*128=4096]
```

So the norm operates **per (batch, time, value-head)**, treating each of the 32 heads independently across its 128-dim output. The same 128-dim `weight` is shared across heads.

## 10. Output projection

`self.out_proj` is `nn.Linear(value_dim=4096 → hidden_size=2048, bias=False)`. Applied **after** the gated RMSNorm:

```python
output = self.out_proj(core_attn_out)         # [B, T, 4096] → [B, T, 2048]
```

This corresponds to GGUF `ssm_out.weight [2048, 4096]`. The **gating by `z` happens inside the RMSNorm** (multiplied as `SiLU(z)`), **before** `out_proj`. There is no second gate after `out_proj`.

This is different from the **full-attention** layer pattern (`Qwen3_5MoeAttention.forward`, which does `attn_output * sigmoid(gate); o_proj(...)` — a sigmoid gate, not SiLU, applied to the attention output stream **before** the o_proj). Don't conflate the two.

## 11. State caches — exact shapes and update rules

### Conv state (`conv_states`)

Shape: `[B, conv_dim=8192, conv_kernel_size=4]`. Dtype: matches `hidden_states` (typically bf16).

Initialization on first prefill (line 464-465 of modeling, `LinearAttentionLayer.update_conv_state` in `cache_utils.py:757`):

```python
# Pad mixed_qkv (after in_proj_qkv, before conv) on the LEFT to length kernel_size:
conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
# i.e. for prefill_len T ≤ 4: zeros on the left to make total length 4.
# For prefill_len T > 4 the inner update keeps only the last 4 (cache_utils.py:782).
```

After each decode step, `causal_conv1d_update` shifts the window:

```
new_state[..., :3] = old_state[..., 1:]      # drop oldest
new_state[..., 3]  = mixed_qkv[..., 0]        # the new (un-conv'd) projected QKV sample
```

(Implemented as `torch.cat([conv_state, hidden_states], dim=-1)` then `[:, :, -state_len:]` — see `torch_causal_conv1d_update` line 221-222.)

Note: the cached samples are the **input** to the conv (post-`in_proj_qkv`, pre-conv), **not** the conv output. SiLU is applied only to the conv output, not stored.

### Recurrent state (`recurrent_states` / `ssm_states`)

Shape: `[B, num_v_heads=32, head_k_dim=128, head_v_dim=128]`. Dtype: **fp32** (the recurrent kernel runs in fp32 and `update_recurrent_state` clones with the same dtype, see `cache_utils.py:751`).

Per the linear-attention cache mixin (`cache_utils.py:672-755`):
- Lazy-initialized to zeros on first call.
- `update_recurrent_state(new_state, layer_idx)` does an in-place `copy_` to preserve the static address (used by CUDA Graphs).

For Qwen3.5-MoE the cache class is `LinearAttentionAndFullAttentionLayer` (`cache_utils.py:807`) — a hybrid that holds linear-attn states for DeltaNet layers and KV cache for the 10 full-attn layers, dispatched by `layer_types[layer_idx]`.

## 12. Twelve-step CPU reference algorithm

This is the algorithm to transcribe verbatim into your C++/SYCL host golden. Inputs are `x ∈ ℝ^{B×T×2048}` (hidden states going into one DeltaNet layer), all weights named per the GGUF map, and (for decode) the prior conv/recurrent state.

```
INPUTS:
  x                     [B, T, 2048]   bf16 or fp32   (post input_layernorm hidden states)
  W_qkv  = attn_qkv     [8192, 2048]   linear, no bias
  W_z    = attn_gate    [4096, 2048]
  W_b    = ssm_beta     [32, 2048]
  W_a    = ssm_alpha    [32, 2048]
  W_conv = ssm_conv1d   [8192, 4]      (depthwise per-channel)
  A_log  = ssm_a        [32]           fp32
  dt_b   = ssm_dt.bias  [32]           fp32
  norm_w = ssm_norm     [128]
  W_out  = ssm_out      [2048, 4096]
  S      = recurrent_state [B, 32, 128, 128] fp32   (zeros if first call)
  C      = conv_state      [B, 8192, 4]            (zeros if first call)
HYPERPARAMS:
  K_dim = 128   V_dim = 128
  H_v   = 32    H_k   = 16    G = H_v/H_k = 2
  conv_dim = 8192   kernel = 4
  eps_norm = 1e-6   eps_l2 = 1e-6

OUTPUT:
  y                     [B, T, 2048]   same dtype as x

# Step 1: linear projections (matmuls, channels-last)
  qkv = x @ W_qkvᵀ                      # [B, T, 8192]
  z   = x @ W_zᵀ                        # [B, T, 4096]
  b   = x @ W_bᵀ                        # [B, T, 32]   fp32
  a   = x @ W_aᵀ                        # [B, T, 32]   fp32

# Step 2: depthwise causal conv1d + SiLU
#   For each (b, t, c) with c ∈ [0, conv_dim):
#       window = qkv[b, t-3..t, c]   (zero-padded on the left from C[b, c, :] for the first 3 steps)
#       conv_out[b, t, c] = sum_{k=0..3} W_conv[c, k] · window[k]
#       qkv'[b, t, c]     = silu(conv_out[b, t, c]) = conv_out · sigmoid(conv_out)
#   Update C[b, c, :] = qkv[b, T-3..T-1, c] (or last 4 samples).
  qkv ← causal_depthwise_conv1d_silu(qkv, W_conv, C)   # [B, T, 8192]

# Step 3: split QKV
  q   = qkv[..., 0:2048]                     # [B, T, 2048]
  k   = qkv[..., 2048:4096]
  v   = qkv[..., 4096:8192]                  # [B, T, 4096]
  q   = reshape(q, [B, T, 16, 128])
  k   = reshape(k, [B, T, 16, 128])
  v   = reshape(v, [B, T, 32, 128])
  z   = reshape(z, [B, T, 32, 128])

# Step 4: scalars β and log-decay g  (fp32 throughout)
  β   = sigmoid(b)                            # [B, T, 32]
  g   = -exp(A_log) * softplus(a + dt_b)      # [B, T, 32], ≤ 0   (broadcast dt_b across (B,T))

# Step 5: GQA replicate — broadcast 16 K-heads up to 32 V-heads
  q   = repeat_interleave(q, 2, axis=2)       # [B, T, 32, 128]
  k   = repeat_interleave(k, 2, axis=2)       # [B, T, 32, 128]
  # v already has 32 heads.

# Step 6: cast to fp32 for the recurrence
  q, k, v, β, g  ←  cast to fp32

# Step 7: per-head L2-norm of Q and K  (along the last dim, K_dim=128)
  q  ←  q  *  rsqrt(sum(q*q, dim=-1, keepdim) + eps_l2)
  k  ←  k  *  rsqrt(sum(k*k, dim=-1, keepdim) + eps_l2)

# Step 8: scale Q by 1/√K_dim
  q  ←  q  *  (1.0f / sqrt(128))              # ≈ 0.0883883476

# Step 9: gated delta recurrence (per [b, h, t])
#   S has shape [B, 32, 128, 128] indexed S[b, h, kk, vv].
  for b in 0..B-1:
    for h in 0..31:
      for t in 0..T-1:
        α = exp(g[b, t, h])                   # scalar fp32
        β_t = β[b, t, h]                      # scalar fp32
        q_t = q[b, t, h, :]                   # [128]
        k_t = k[b, t, h, :]                   # [128]
        v_t = v[b, t, h, :]                   # [128]

        # 9a) decay
        for kk in 0..127: for vv in 0..127:
            S[b, h, kk, vv] *= α

        # 9b) read-out: kv_mem[vv] = Σ_kk S[kk, vv] · k_t[kk]
        for vv in 0..127:
            kv_mem[vv] = Σ_kk S[b, h, kk, vv] · k_t[kk]

        # 9c) error-corrective delta
        for vv in 0..127:
            δ[vv] = (v_t[vv] − kv_mem[vv]) · β_t

        # 9d) rank-1 update: S += k_t ⊗ δ
        for kk in 0..127: for vv in 0..127:
            S[b, h, kk, vv] += k_t[kk] · δ[vv]

        # 9e) query: out[vv] = Σ_kk S[kk, vv] · q_t[kk]
        for vv in 0..127:
            out[b, t, h, vv] = Σ_kk S[b, h, kk, vv] · q_t[kk]
  # `out` shape: [B, T, 32, 128] in fp32

# Step 10: gated RMSNorm with SiLU(z) gate, per (b, t, h), reducing over last dim (V_dim=128)
  for b, t, h:
      x   = out[b, t, h, :]                   # [128] fp32
      μ2  = mean(x²)                           # scalar fp32
      x   = x * rsqrt(μ2 + eps_norm)
      x   = (norm_w * x).cast(input_dtype)    # weight ⊙ x_normed, cast first to input dtype
      x   = x * silu(z[b, t, h, :].cast(fp32))# SiLU(z) gate in fp32
      out[b, t, h, :] = x.cast(input_dtype)

# Step 11: flatten heads
  out = reshape(out, [B, T, 4096])

# Step 12: output projection
  y = out @ W_outᵀ                            # [B, T, 2048]

# (decode only) write back the updated S into recurrent_state cache.
```

### Numerical-precision checklist (do not skip)

1. **W_a, W_b, A_log, dt_b** evaluated in **fp32**. Loading them in bf16 will silently underflow `A_log.exp()` (the source explicitly hoists `.float()` on these — line 494).
2. **L2-norm** uses `rsqrt(Σx² + 1e-6)` with eps inside the sqrt (FLA convention, comment in `l2norm` at line 230).
3. **RMSNorm-gated** uses `rsqrt(mean(x²) + 1e-6)` — note **`mean`**, not `sum`. (Different from L2-norm.)
4. **Q-scale `1/√128`** is applied **once** (not per token) — line 264 / 329.
5. **The whole recurrence runs in fp32**, including the state. Cast to input dtype only at the very end (line 353).
6. **State is materialized fp32 and cached fp32** — `mamba_ssm_dtype: float32` per the user's note, confirmed by line 322-324 (`.to(torch.float32)` of every input).
7. **Conv1d input is the post-projection, pre-SiLU stream** (the cache stores `mixed_qkv`, not `silu(conv(mixed_qkv))`).
8. **`apply_mask_to_padding_states` zeroes padded tokens BEFORE the projections** — important if you batch sequences of different lengths.
9. **No RoPE inside DeltaNet** — RoPE is applied only in the full-attention layers (every 4th layer). DeltaNet has no positional encoding; the conv1d provides local positional sensitivity.
10. **No bias on any linear in DeltaNet** (`bias=False` on every `nn.Linear` and on `nn.Conv1d`).

### Quick sanity test

Run a single decode step with `B=1`, `T=1`, `S=0`, `C=0`:

- After step 5, `q`, `k`, `v` are deterministic functions of `x`, `W_qkv`, `W_conv`.
- After step 9 with `S_prev=0`: `kv_mem=0`, `δ = β·v`, `S = β·k⊗v`, `out = β·k⊗v · q = β · (qᵀk) · v`. With L2-normed q,k and `q ← q/√K`, `qᵀk = (q̂ᵀk̂)/√K`. So the very first decode output (per head) is exactly `β · v · cos(q̂, k̂)/√K`, scaled by `SiLU(z)` and normalized.

That single-step check catches: wrong sign on β, missing `1/√K` scale, missing L2-norm, wrong axis for the rank-1 outer product, and SiLU vs sigmoid mix-ups in the gated RMSNorm.

---

### Cross-references

- **Algorithm origin**: Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, Dec 2024 (arxiv 2412.06464). Equations 4-7 of the paper match the recurrent block above — `α_t` is the scalar decay, `β_t` the data-dependent write strength, `S` the matrix-valued state, with `(v − k S)` being the prediction error.
- **Reference implementation**: `fla` (flash-linear-attention) `fla.ops.gated_delta_rule.{chunk_gated_delta_rule, fused_recurrent_gated_delta_rule}`. The torch fallbacks above mirror this kernel's contract bit-for-bit.
- **GGUF mapping in llama.cpp**: PR #16095 (Qwen3-Next), with Qwen3.5-MoE inheriting the same scheme but un-fusing `ssm_beta_alpha → ssm_alpha + ssm_beta` and adding a separate `attn_gate` for `z`.
