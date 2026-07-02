# 1-bit / BitNet readiness + scoping (owner roadmap #4) — 2026-06-13

**Ask:** "investigate 1-bit models (BitNet-style); if worth it, this likely pulls in an
iOS/Android mobile-deploy track." This is a codebase + oracle-grounded scoping with a
go/no-go. **No engine code was written** — #4 is explicitly investigate-first.

## What BitNet is

BitNet b1.58 = a ternary LLM: weights restricted to **{-1, 0, +1}** (≈1.58 bits) with a
per-tensor fp16 scale. Matmuls become add/sub (no multiplies) → tiny memory + cheap
compute. Current flagship: **Microsoft BitNet-b1.58-2B-4T** (2B params, 4T tokens,
competitive with FP 2B models). Stored ternary-packed in GGUF.

## The trap (verified): the official GGUF is the WRONG format for us

- Downloaded `microsoft/bitnet-b1.58-2B-4T-gguf` → `ggml-model-i2_s.gguf` (1.2 GB).
- **It is `I2_S` (ggml tensor dtype id 36) — a Microsoft *bitnet.cpp fork* type.** Mainline
  ggml does NOT have id 36 (it's a removed/commented slot — `ggml.h` shows 34=TQ1_0,
  35=TQ2_0, 36/37/38 commented, 39=MXFP4). `ie-inspect` correctly errors:
  `unsupported tensor dtype id 36`.
- Consequence: this file runs on **neither our engine NOR the mainline llama.cpp oracle**
  — only on Microsoft's `bitnet.cpp` fork. Anyone who "just downloads the BitNet GGUF and
  tries to run it" hits a wall. (i2_s download deleted to reclaim disk.)
- **The right artifact** is a **TQ2_0** (or TQ1_0) GGUF, which our engine targets (see
  below) and mainline llama.cpp runs. Produce it with the mainline converter, which
  supports it: `convert_hf_to_gguf.py --outtype tq2_0` over the HF safetensors
  (`microsoft/bitnet-b1.58-2B-4T`, ~5 GB BF16). That GGUF is the engine target + the
  oracle source.

## Engine state — partially scaffolded, kernels + arch missing

- ✅ **dtypes already registered**: `dtype.hpp`/`dtype.cpp` have `kTQ1_0=34` (54 B / 256-block)
  and `kTQ2_0=35` (66 B / 256-block) — **block sizes match the oracle's `block_tq1_0`/
  `block_tq2_0` exactly** (TQ1_0: 2+4+48; TQ2_0: 2+64). Someone stubbed the enum and stopped.
- ❌ **No TQ kernels** (no dequant / GEMV / GEMM for TQ1_0/TQ2_0 anywhere in `src/ops`).
- ❌ **No `bitnet` arch** (loader/forward).

## The arch (from the oracle `~/llama.cpp/src/models/bitnet.cpp`)

LLaMA-like dense transformer + three BitNet-specific things:
1. **Two extra sub-norms per layer**: `attn_sub_norm` (RMSNorm on attn output BEFORE the
   `wo` proj) and `ffn_sub_norm` (RMSNorm on the FFN activation BEFORE `ffn_down`). These
   are the signature of BitNet — they renormalize activations feeding each ternary proj.
2. **Per-tensor weight `scale`** on every quantized matmul (`wq_s`, `wk_s`, …, `ffn_gate_s`,
   `ffn_down_s`): the ternary weight is multiplied by its scalar after the matmul.
3. Otherwise standard: RMSNorm (attn_norm/ffn_norm), GQA attention, NEOX `rope_ext`,
   **SwiGLU FFN** (`LLM_FFN_SILU, LLM_FFN_PAR` — ffn_up ∥ ffn_gate, not squared-ReLU),
   final norm + tied `tok_embd` output.

→ It rides ~90% of the existing **DenseModel / `kLlama3`** path (RMSNorm + GQA + NEOX rope +
SwiGLU). The NEW work is: (a) the TQ2_0 dequant/GEMV kernel, (b) the 2 sub-norms per layer,
(c) applying the per-tensor weight scale in the matmul, (d) a `bitnet` arch branch.

## Effort estimate

Comparable to a typical arch add (≈ the Gemma-4 Q4_0 + dense-arch effort), multi-session:
1. **Model prep** (~0.5 day): HF safetensors → TQ2_0 GGUF via mainline converter; confirm
   mainline llama.cpp runs it (oracle baseline PPL).
2. **TQ2_0 dequant + GEMV kernel** (~1-2 days): unpack 2-bit ternary (4/byte) → {-1,0,+1}·scale;
   simplest first version dequants to fp16 and reuses the existing dense GEMV (the GPU does
   fp16 matmul fine — on a B70 the BitNet win is *memory footprint*, not compute). A true
   add/sub ternary kernel is a later perf item. Optionally TQ1_0 too (5/byte packing).
3. **`bitnet` arch loader + forward** (~1-2 days): DenseModel variant + the 2 sub-norms +
   per-tensor scale; validate per-layer cosine + PPL vs the oracle (the usual gate battery).

## Recommendation — go/no-go

**GPU BitNet arch: GO as a breadth/credibility item, but LOW priority vs the launch headline.**
- It's a clean additive arch with a real oracle and correct dtype scaffolding already in place.
  "Mach X runs 1-bit BitNet models" is a legitimate launch breadth bullet.
- BUT be honest about the value on *this* hardware: BitNet's headline win (add/sub, energy,
  tiny RAM) is a **CPU/edge** story. On a 32 GB B70 the only win is memory footprint (2B in
  ~1 GB); throughput won't beat a normal fp16/Q4 2B. So it's "we support it," not a perf crown.
- Sequence it **after** roadmap #5 (perf tuning + the benchmark writeup) — the writeup is the
  actual launch asset; BitNet is breadth garnish.

**Mobile-deploy track: SEPARATE, MAJOR strategic decision — do NOT let #4 balloon into it
mid-launch.** The "1-bit → iOS/Android/Play" implication is a whole new **backend** (the engine
is Linux + SYCL + Intel GPU today; mobile = ARM CPU/NPU, a different compile + kernel target —
months, not days). BitNet is what makes mobile *attractive* (fits a phone), but shipping mobile
is its own initiative. Recommend: greenlight the GPU bitnet arch when breadth bandwidth exists;
treat mobile as a deliberate, separately-scoped program, not a side effect of #4.

## If greenlit, step 1 is unambiguous
Convert `microsoft/bitnet-b1.58-2B-4T` (HF safetensors) → TQ2_0 GGUF with the mainline
converter, confirm the oracle runs it, then build the TQ2_0 kernel → `bitnet` arch → gate
battery. The dtype scaffolding (kTQ1_0/kTQ2_0) is already correct, so the kernel slots in.
</content>
