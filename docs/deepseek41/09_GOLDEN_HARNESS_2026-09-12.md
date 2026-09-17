# V4.1 golden harness — running DeepSeek's own code on this box (2026-09-12)

## The constraint, verified

`00_PORT_SCOPE` called the shipped reference "the single biggest asset here: every format and
every module can be read rather than inferred." True for reading. **It cannot be run here.**

- `inference/kernel.py` is written in **tilelang** and targets CUDA.
- `import tilelang` → `ModuleNotFoundError`; `import triton` → `ModuleNotFoundError`.
- `torch 2.12.0+cpu`, `torch.cuda.is_available() == False`.

So the reference cannot produce numerical goldens as shipped, and "bit-exact against the
reference" is not an achievable target for this port. That had to be established before any
forward-pass work, because it changes what "correct" can mean.

## What the harness does instead

`tools/ds41_reference/kernel.py` replaces **six leaf primitives** and nothing else. Every other
line of `model.py` — Attention, Block, MoE, Gate, Expert, Compressor, Indexer, Transformer, the
routing, the clamps, the masking, the RoPE — executes as DeepSeek wrote it.

| primitive | stand-in | what it costs |
|---|---|---|
| `act_quant`, `fp4_act_quant` | identity | the golden does **not** quantise activations; it is the mathematical intent, not one quantisation schedule |
| `fp8_gemm`, `fp4_gemm` | exact dequant + full-precision matmul | only GEMM *precision* differs; weight **values** are the Phase-2-verified decode |
| `hc_split_sinkhorn` | transcribed line by line from the tilelang kernel | mine, but short and readable |
| `sparse_attn` | **raises** | deliberately not faked — Phase 5c |

Engine output will differ from this golden by activation-quantisation error, which is a design
choice the engine makes for itself (as it already does for V4). The golden fixes the *weights*
and the *composition*, which is what a port can actually get wrong.

## Phase 5a result: the MoE block

`tools/ds41_reference/golden_moe.py` builds the reference's `MoE(0, args)` with layer 0's real
weights (lazily mmapped — only selected experts are ever touched) and runs 8 deterministic
tokens:

```
building reference MoE(0) — 384 experts, dim 5120, inter 2304, top-6, score 'sqrtsoftplus'
routed: indices (8, 6), unique experts hit 31
y: (1, 8, 5120)  absmax 0.015938  std 0.000787
self-check: independent recompute max|diff| 0.000e+00 (rel 0.00e+00)
```

### The self-check matters as much as the golden

The reference's code is authoritative on *composition*, but the **wiring into it is mine** —
which safetensors tensor lands on which `Linear`. A `w1`/`w3` swap, a transposed weight or a
missed clamp would produce a perfectly plausible golden and silently poison everything
downstream, exactly the way the expert nibble order did in Phase 4.

So the driver also recomputes the whole block independently: routing from raw scores, SwiGLU
with both clamps, per-expert dequant from the raw bytes, shared expert, accumulation — and
requires agreement. It agrees at **max |diff| = 0.0**, and the routing indices and weights match
by `torch.equal`. Two independent paths, identical output.

## Files

- `tools/ds41_reference/kernel.py` — the six stand-ins, each documenting what it substitutes.
- `tools/ds41_reference/golden_moe.py` — MoE(0) golden + self-check; writes `moe_in.f32`,
  `moe_out.f32`, `gate_weights.f32`, `gate_indices.i32`, `moe_meta.json`.

Next: `golden_hc.py` (hyper-connections, the stand-in already exists) and then `sparse_attn`,
which is the one primitive that still has no CPU implementation.

## Phase 5c: the whole Block runs

With `sparse_attn` transcribed, `golden_block.py` runs the reference's **`Block(0)`** end to end
on layer 0's real weights — attention, mHC, and the 384-expert MoE:

```
attn out  absmax 2.656633  std 0.462398
block out (1, 8, 4, 5120)  absmax 1.602665  std 0.095868
comb row sums [1.0113, 1.0189, 1.0195, 0.9502]  col sums [1.0000, 1.0000, 1.0000, 1.0000]
```

The comb matrix is doubly stochastic to ~1% on rows and exactly on columns, which is what a
Sinkhorn ending on a column normalisation should give — an independent sign the transcription
behaves.

Layer 0 is the right first target because `compress_ratios[0] == 0`: sliding-window attention
only, no compressor, no indexer, no engram (those are layers 1 and 14). Everything else about it
is the general case.

Goldens dumped at every sub-block boundary: `blk_in`, `blk_pre_mix`, `blk_attn_pre/post/comb`,
`blk_attn_in`, `blk_attn_out`, `blk_out`, `blk_ffn_pre`.

### Three things in the attention that a port gets wrong silently

1. **`wo_a` is block-diagonal over `o_groups`, not a dense matmul.** The reference does
   `einsum("bsgd,grd->bsgr", o.view(b,s,8,-1), wo_a.view(8, 1024, 4096))` — each group projects
   only its own heads. Treating `[8192, 4096]` as one dense matrix has the right shape and the
   wrong answer.
2. **The attention output gets an INVERSE RoPE** before `wo_a`:
   `apply_rotary_emb(o[..., -rd:], freqs_cis, True)`. Omitting it costs nothing structurally and
   everything numerically.
3. **`attn_sink` is a logit with no value vector.** It enters only the softmax denominator, so a
   head can attend to "nothing" and shrink its output instead of being forced to spread weight.

A fourth, for the loader: the reference indexes `self.wo_a.weight` **directly** rather than
through `linear()`, so it needs a real float dtype — `convert.py` dequantises it to bf16. The
shipped checkpoint stores it as F8_E4M3 with a 32x32 scale grid, so the engine must dequantise
it at load rather than keeping it packed like the other dense weights.
