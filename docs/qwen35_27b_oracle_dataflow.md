# Qwen3.6-27B (`qwen35`) — bit-exact oracle & verified dataflow (P3d Task 2.5)

**Status:** oracle RESOLVED 2026-06-11. This doc is the structural reference for
re-deriving the unfused DeltaNet/gated-attn forward (Task 3b). Numeric bit-exact
diffing happens at Task 4 (GPU) against this same oracle.

## The oracle (reproducible)
Old local llama.cpp HEAD `dcad77cc3` could not load the 27B (predated the MTP/NextN
support series). Current master **`fdc3db9b6` (build `b9598`)** loads it cleanly and
treats `blk.64` as a dense NextN/MTP layer (read via `LLM_KV_NEXTN_PREDICT_LAYERS`).

```bash
# CPU-only oracle build (sufficient to load + dump reference tensors)
cd ~/llama.cpp && git checkout origin/master   # fdc3db9b6 or newer
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
      -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_SYCL=OFF
cmake --build build-cpu --target llama-perplexity llama-eval-callback -j

# Per-op tensor dump (the structural oracle). SERIALIZE: 30 GB box, pause downloads,
# cap threads, watch `free` (an 18 GB load + the 72B download OOM-crashed VS Code once).
GGUF=~/models/bartowski/Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_M.gguf
./build-cpu/bin/llama-eval-callback -m "$GGUF" -p "France" -n 1 -t 10 -c 256 > dump.log 2>&1
```
- `llama-cli` is now **interactive-only** (`-no-cnv` removed → "use llama-completion").
  For scripted oracle runs use `llama-eval-callback` (per-op dump) / `llama-perplexity`.
- A clean op-graph extract (names+shapes, numeric arrays stripped) is preserved at
  `docs/oracle/qwen35_27b_opgraph_L0_L3.txt`.

## Verified geometry (read off the actual GGUF tensors, not the model card)
| quantity | value | evidence (op shape) |
|---|---|---|
| hidden | 5120 | `token_embd.weight{5120, 248320}` |
| vocab | 248320 | same |
| ffn | 17408 | `ffn_gate/up.weight{5120, 17408}` |
| q / kv heads | 24 / 4, head_dim 256 | `attn_q_norm{256,24}`, `attn_k_norm{256,4}` |
| DeltaNet conv_channels | **10240** | `attn_qkv.weight{5120, 10240}`, `ssm_conv1d.weight{4, 10240}` |
| DeltaNet k-heads | **16** × 128 | `q_conv-0{128, 16}` |
| DeltaNet v-heads | **48** × 128 | `ssm_norm` over `norm-0{128, 48}` |
| DeltaNet d_inner | 6144 | `attn_gate.weight{5120, 6144}`, `ssm_out.weight{6144, 5120}` |
| full-attn `attn_q` | **{5120, 12288}** = joint Q\|gate (24·256·2) | `Qcur_full-3{12288}` |
| conv kernel | 4 | `ssm_conv1d.weight{4, 10240}` |
| layers | 64 transformer + 1 NextN (`blk.64`, skip) | block_count 65 |
| full-attn interval | every 4th (`(i+1)%4==0` → i=3,7,…,63) | `blk.3` is full-attn |

## Linear (DeltaNet) layer — op chain (L0, verified)
```
attn_norm        = rms_norm(x, blk.L.attn_norm.weight{5120})
qkv{10240}       = attn_qkv.weight{5120,10240} · attn_norm          # R1: 10240, NOT SI2=SI*2
conv_input{4,10240} -> ssm_conv1d.weight{4,10240} -> conv_output_raw{10240} -> SiLU
   split conv_channels 10240 -> q_conv{128,16}  k_conv{128,16}  v_conv{128·48=6144}
                                (q_off=0, k_off=2048, v_off=4096, v_width=d_inner 6144)
q_conv_predelta  = L2_NORM(q_conv{128,16})       # l2-norm per head, 16 k-heads
k_conv_predelta  = L2_NORM(k_conv{128,16})
v_conv_predelta  = v_conv{128,48}                 # v repeats k 16->48 inside the fused op (repeat=3)
alpha{48}        = ssm_alpha.weight{5120,48}·attn_norm ; +ssm_dt.bias{48} ; softplus ; ·ssm_a{48}  -> gate
beta{48}         = ssm_beta.weight{5120,48}·attn_norm ; sigmoid -> beta
state_predelta   = GATED_DELTA_NET(q_pre,k_pre,v_pre, gate, beta)   # llama fuses; WE re-derive unfused
norm{128,48}     = gated_rms_norm(state, ssm_norm.weight{128}, z)   # z = attn_gate.weight{5120,6144}·attn_norm
linear_attn_out  = ssm_out.weight{6144,5120} · final_output{6144}   # Q8_0 weight
l_out            = x + linear_attn_out
# --- then shared FFN block (see below) keyed on attn_post_norm(l_out) ---
```
**WE must re-derive `GATED_DELTA_NET` from leaf ops** (`split` → `repeat_interleave_heads(3)`
→ `l2_norm_scale`×2 → `deltanet_recurrence(n_v=48)` → `gated_rms_norm`). Diff each named
tensor above against the oracle dump on CPU before any GPU code (landmines R1/R2/R4).

## Full-attention layer — op chain (L3, verified)
```
attn_norm        = rms_norm(x, blk.L.attn_norm.weight{5120})
Qcur_full{12288} = attn_q.weight{5120,12288} · attn_norm            # JOINT Q|gate (24·256·2)
   -> split: Q{256,24} + gate{256,24}                               # ws_q_full_ interleaved path
Qcur_normed      = rms_norm(Q{256,24}, attn_q_norm.weight{256})     # per-head Q-norm
Kcur{1024}       = attn_k.weight{5120,1024} · attn_norm  (Q5_K)     # 4 kv-heads·256
Kcur_normed      = rms_norm(K{256,4}, attn_k_norm.weight{256})
Vcur{1024}       = attn_v.weight{5120,1024} · attn_norm  (4 kv·256)
   -> partial RoPE (n_rot 64, 0.25) on Q,K  ;  FA-2  ;  sigmoid-gate by `gate`
attn_output{5120}= attn_output.weight{6144,5120} · attn_gated{6144} (Q5_K)
l_out            = x + attn_output
```

## Shared FFN + residual order (BOTH layer kinds) — landmine
```
attn_post_norm   = rms_norm(l_out, blk.L.post_attention_norm.weight{5120})  # post-norm AFTER residual
ffn              = swiglu(ffn_gate{5120,17408}·h, ffn_up{5120,17408}·h) ; ffn_down{17408,5120}
out              = l_out + ffn        # FFN adds to the PRE-post-norm tensor (l_out), not attn_post_norm
```
i.e. `norm-(L+1)` takes `l_out-L`. Easy silent-divergence bug if FFN is added to the
post-norm output instead. Confirmed: oracle shows `norm-1 <- (l_out-0{5120})`.

## Op inventory — CONFIRMED against exact shapes (retires the plan's #1 risk)
Verified every leaf op in `include/ie/ops.hpp` is runtime-parameterized for the 27B's
*exact* geometry (48 v-heads / 16 k-heads / conv_channels 10240) — **Task 3 needs ZERO
new GPU kernels; it is pure orchestration.** The plan's "biggest risk = DeltaNet
head-count generalization" is retired:
- `deltanet_recurrence(…, n_v_heads, k_head_dim, v_head_dim, …)` — `n_v_heads` runtime
  (pass 48); only locks `k_head_dim==v_head_dim==128`, which the 27B satisfies. ✓
- `depthwise_conv1d_causal(…, channels, kernel, …)` — `channels` runtime: **pass 10240**
  (the doc-comment's `8192` is the CROWN's Q+K+V concat — landmine R1; do NOT reuse it),
  kernel 4. ✓
- `repeat_interleave_heads(…, repeat, …)` — pass **repeat=3** (16 k-heads → 48). ✓
- `l2_norm_scale`, `gated_rms_norm(_q8)`, `compute_g_beta(_h16)`, `split_q_gate_per_head`,
  `rope_partial`, `full_attention(_fa2_decode)`, `swiglu` — all present, runtime-shaped. ✓
- Dense MLP gate/up/down GEMVs = the P2 `dense_dispatch.hpp` path (K=5120, ffn 17408). ✓
- Q5_K/Q8_0 transposed dequant (`dequant_q5_K_to_Bt`/`dequant_q8_0_to_Bt`) already landed.

So Task 3 = a new `Qwen35DenseModel` that (a) loads qwen35 tensors + dequants Q5_K/Q8_0
at load, (b) allocates the DeltaNet state cache (48 linear layers) + KV cache (16 full
layers), (c) calls the above ops per `recurrent_layer(il)` in the verified order — then
diffs each named tensor vs the oracle on CPU before trusting the GPU path.

## Weight dtypes (per Task 1) needing the new dequant kernels
`attn_k`, `attn_output` = **Q5_K**; `ssm_out` = **Q8_0** → use `dequant_q5_K_to_Bt` /
`dequant_q8_0_to_Bt` + transpose-to-[K,N] for the fp16 GEMV (kernels already landed).
