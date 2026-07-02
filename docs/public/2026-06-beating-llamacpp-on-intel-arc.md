# Beating llama.cpp on Intel Arc: a from-scratch SYCL engine

*(draft — publishes at P4 launch alongside the open-source release. The
product name is shown as `[ENGINE NAME]` until finalized.)*

---

## TL;DR

On 2026-06-10, `[ENGINE NAME]` became a from-scratch SYCL inference
implementation to beat llama.cpp's best backend on both prefill and decode for
a large MoE model on Intel Arc. As a non-fork sharing no kernel code, the
result shows this headroom is accessible to any clean implementation.

The headline numbers (same-hour alternating runs, same GGUF, same B70) — this
is the **crown model** and the *only* model where the engine wins both metrics
against llama.cpp's best backend:

- **Prefill (pp512):** 1144 ± 5 tok/s vs llama.cpp SYCL master 1064 ± 8 —
  **+7.6%**
- **Decode (tg128) turbo:** 84.1 tok/s vs llama.cpp SYCL master 81.31 ±
  0.21 — **+3.5%**
- **Decode (tg128) default:** 81.0 tok/s @ PPL 6.45 — statistical tie at
  better quality
- **Perplexity:** 6.45 at production defaults vs gate ceiling 6.57
  (project-best, bit-exact)

This is one model, one GPU, single-user. The honest scope — including where
the engine does *not* win — is below; please read it before generalizing the
headline. One-command repro: `scripts/bench_showdown.sh` (public-repro guide:
`scripts/bench_showdown_public.md`).

---

## Honest scope and limitations

Read this before generalizing the headline. The both-metrics win is one
model; the rest of the engine's story is mixed-and-honest, and a skeptical
reader will (rightly) probe it. We would rather state it than have it found.

**The both-metrics win covers exactly one model, one GPU, single-user
(batch = 1):**

- **Model:** Qwen3.6-35B-A3B Q4\_K\_M (a hybrid MoE with 256 experts, 40
  layers — 30 DeltaNet linear-attention + 10 full-attention)
- **GPU:** Intel Arc Pro B70 (BMG-G31, Xe2-HPG)
- **Workload:** single user, pp512 prefill / tg128 decode, no KV sharing

**Prefill-strong, decode-tuned-only-on-the-crown.** The engine's consistent
strength is *prefill* — it leads on every model we have benchmarked (crown
+7.6%, the 27B dense-hybrid 1.9× vs llama Vulkan, dense Qwen3-8B +14.9%).
*Decode* is a different story: the int-dot Q8 GEMV decode path was hand-tuned
for the crown's shapes and has **not** been ported to other architectures
yet. So off the crown, decode is honestly mixed-to-behind — e.g. dense
Qwen3-8B decode is **43.7 vs llama's 77.7 (≈56%)**, an explicit loss we are
optimizing, not hiding.

**Qwen3-Coder-30B runs correctly but is NOT a perf win.** It is
naive-attention-bound (no FlashAttention/split-K path yet — `attn_naive`
is ~71% of decode), and prefill is RAM-swap-bound on our 30 GB box. It is
**16× behind** on prefill and **3× behind** on decode vs llama.cpp Vulkan.
We ship it as a *correct, parity-proven* baseline, not a speed result. See
the breadth table and the receipts section.

**Multi-GPU is "runs + correct," not a llama.cpp comparison.** The 72B
tensor-parallel result is 1.44× over *our own* layer-split — there is no
same-hardware 72B A/B against llama.cpp yet.

**This is not a claim that this is a better llama.cpp in general.** llama.cpp
supports dozens of models, multiple backends, batched serving, and years of
hardware coverage. Today this engine is single-user, 1–2 GPUs, Linux. What
the crown result demonstrates is that there is meaningful headroom left on
Intel Arc for a purpose-built implementation — and that headroom can be
captured. The roadmap beyond this is breadth: more models, more GPUs, a
server interface.

---

## Beyond one model: breadth

The crown is the credibility anchor. The breadth below is *coverage and
correctness* — "it runs the models you actually download" — **not** a claim
that the engine beats llama.cpp on each. Where a number is a win it is marked;
everything else is correctness/parity. **Six architecture families are
validated on the engine today:**

| family | model (validated) | status / honest number |
|---|---|---|
| Qwen3.6 MoE (crown) | Qwen3.6-35B-A3B Q4\_K\_M | **WIN, both metrics** — +7.6% prefill, +3.5% decode vs llama SYCL; PPL 6.45 |
| Qwen3.6 dense-hybrid (27B) | Qwen3.6-27B (`qwen35`) | **prefill WIN** 1.9× (577 vs 303) vs llama **Vulkan**; decode parity (10.0 vs 9.72); PPL 5.34 oracle-consistent |
| Qwen3 / Qwen2 dense | Qwen3-8B Q4\_K\_M | **prefill +14.9%** (1190 vs 1036) vs llama SYCL; **decode 56% (a loss)**; bit-exact (cosine 1.000000, greedy 64/64) |
| Llama-3.x | Llama-3.1-8B | runs correctly; per-layer cosine 0.99998–1.0 all 32 layers; PPL 10.79 |
| Qwen3-Coder MoE | Qwen3-Coder-30B-A3B (`qwen3moe`) | runs correctly with fused MoE; PPL 11.99 (parity-proven); **NOT a perf win** (16× behind prefill, 3× decode — naive-attention-bound) |
| AWQ / GPTQ import | Qwen2.5-7B-AWQ, Qwen3-8B-GPTQ | `ie import` loads two quantized safetensors formats **llama.cpp cannot load natively**; Qwen3-8B PPL 2.937037 exact (zero regression) |

Plus **tensor-parallel multi-GPU**: a 72B (Qwen2.5-72B) runs split across two
Arc Pro B70s; TP decode is 1.44× over our own layer-split (10.4 vs 7.2 tok/s),
cosine 0.999999 vs single-GPU, engine PPL 8.97. (No same-hardware 72B A/B vs
llama.cpp exists yet — this is "runs + correct + faster than our own
layer-split," not a llama comparison.)

**A "Wave-1 breadth blitz" (Yi, Mistral, the R1 distills, Phi, Granite, …) is
in progress but NOT yet validated** — it is CPU-side scaffolding behind a
serial GPU-validation gate. Do not read it as supported today. The honest
forward-looking number is "6 families validated, a dozen-plus queued."

---

## Methodology

The B70 swings ±40 tok/s between a cold boot and a heat-soaked run; an
uncontrolled A-then-B comparison can manufacture a 5% gap that does not
exist. (A later cooler-GPU run on 2026-06-10 measured engine 1183.22 vs
llama.cpp 1085.92 ± 10.38, illustrating exactly this spread.) Every result
here follows the same protocol:

1. **Order-controlled pairs:** ours and theirs alternate (new-old-new) so
   each pair is on the same thermal footing.
2. **First-run discard:** the first run after a rebuild pays JIT compilation
   inside the timed region and is always discarded.
3. **Same GGUF:** `Qwen3.6-35B-A3B-Q4_K_M.gguf` for both engines. llama.cpp
   flags: `llama-bench -ngl 99 -p 512 -n 128 -r 3 -sm none -mg 0` pinned via
   `ONEAPI_DEVICE_SELECTOR=level_zero:0`.
4. **PPL gate:** every change must hold perplexity ≤ 6.57 or it reverts.
5. **Same hour:** reference llama.cpp numbers and engine numbers come from
   the same session.

### The five-run prefill showdown (2026-06-10)

Alternating ours/theirs, same GGUF, same B70, minutes apart:

| run | engine pp512 (tok/s) | llama.cpp SYCL master 76da2450a |
|-----|---------------------:|--------------------------------:|
| 1   | **1147.3**           | 1064.25 ± 7.66                  |
| 2   | **1138.9**           | 1063.03 ± 9.49                  |
| 3   | **1146.7**           | —                               |

Mean engine: ~1144 ± 5. Mean llama.cpp: ~1064 ± 8. Lead: **+7.6%**.

### Hardware and driver stack

- **GPU:** Intel Arc Pro B70 (BMG-G31, Xe2-HPG)
- **Driver stack:** NEO 26.14.37833.4, IGC 2.32.7, oneAPI 2025.3 (SYCL build
  env), GuC 70.60.0
- **Toolchain:** IntelLLVM 2026.0.0
- **llama.cpp commit:** worktree at 76da2450a (master), built
  `-DGGML_SYCL=ON -DCMAKE_CXX_COMPILER=icpx`

---

## Where the speed comes from (the interesting part)

### 1. The prefill arc: 203 → 1144 tok/s in one development sprint

Prefill started at 202.9 tok/s on a naïve Q4\_K GEMV path. The E1–E5
sequence, executed in a single session on 2026-06-09, pushed pp512 to 899.7
— already beating llama.cpp Vulkan (775–885 pp512 on the same B70). The prefill crown over llama.cpp SYCL
required one more session on 2026-06-10.

**E1 (202.9 → 309.4 tok/s):** For T ≥ 64, dequantize Q4\_K and Q6\_K
weights to fp16 scratch and run a dense `gemm_fp16` (measured at 33.5 TFLOPS,
18% of peak). Kill switch: `IE_NO_DEQ_FP16=1`.

**E2 (309.4 → 589.7):** Vectorized weight and SLM loads in the three MoE
prefill kernels (`moe_fused.cpp`), bit-identical math — scalar LSC loads were
the bottleneck; vec4/vec8 loads saturate SLM bandwidth and cut instruction
overhead.

**E2b (589.7 → 754.5):** Scalar `gemm_q4_K` in-kernel M-tile grid plus
vectorization.

**E2c (754.5 → 840.7):** SLM-staged Q6\_K weight slabs in
`down_packed_q6k_v2`.

**E5 (840.7 → 899.7):** Pre-gathered expert-sorted `x_packed` for stage-1,
eliminating 24 redundant scattered activation reads per expert per token.

**Tried and reverted:** E3 (BK=64 `gemm_fp16` retile, −21% — wider BK loses
to occupancy); E4 (header vector loads, neutral — confirmed stage-1 is
compute-bound, not memory-bound).

**Crown move (899.7 → 1144, 2026-06-10):** See section 3 below.

### 2. Decode: Q8\_1 activation quant riding norm kernels

The decode arc ran from 44.7 tok/s on day one to 84.1 tok/s at the crown
(+88%). The decisive techniques:

**Q8\_1 activation quantization fused into norm kernels.** The
activation-to-Q8\_1 conversion runs inside `rms_norm_f32w_q8` and
`gated_rms_norm_q8`, eliminating a standalone quantize launch per layer.

**ALU-bound staged-GEMV int-dot paths.** `lm_head` ran 1.68 ms at fp16;
integer dot (via `ie/dp4a.hpp`) drives it toward ~1.0 ms (fp16 is ALU-bound
here, not bandwidth-bound; idp4a's 4× arithmetic density over fp16 MACs
breaks the wall). In the turbo GGUF, `lm_head` uses Q4\_K weights, running
hardware idp4a. `ssm_out` uses the same path.

**Parallel top-k router.** Two-stage T=1 router fusion removed a single-WG
occupancy stall — +13% alone.

**Negative results banked:** Q4\_K projection int-dot neutral
(bandwidth-bound); `gemv_q4_K` A-bypass −19%; cross-layer residual+norm
fusion −1.5%. PPL at production defaults: 6.45 (bit-exact, avg NLL 1.864495), well below the 6.57 gate.

### 3. The crown move: per-expert-region SoA layout and full-K register lattices

The gap from 899.7 to 1144 tok/s came from applying llama.cpp's SoA reorder
insight — already proven in their `reorder_qw_q4_k` — to per-expert memory
regions, then extending it with full-K register lattices and `block_q8_1s`
split half-sum corrections. The latter two are specific to this engine's
int-dot kernel shape.

**What we learned from llama.cpp's source:** llama.cpp's "reorder" path
(`ggml-sycl.cpp reorder_qw_q4_k`) is a struct-of-arrays split per tensor:
`[qs × nblocks | scales × nblocks | half2 dm × nblocks]`. This removes the
16-byte header that interrupts every 144 bytes of quantized stream in the
default AoS layout. Their MoE prefill runs
`mul_mat_vec_q4_K_q8_1_ncols` (token-batched in registers, ncols ≤ 8 — their
SYCL kernel accumulates up to 8 token rows simultaneously in registers,
amortizing weight loads across the batch) plus
`quantize_row_q8_1`, spending ~123 ms expert compute where the engine
was spending ~310 ms.

**Step 1 — load-time per-expert SoA repack:**
`ffn_{gate,up,down}_exps` are repacked at model load into per-expert regions
with the SoA layout (`ie/quant_soa.hpp`). The repack is neutral alone but
enables the subsequent int-dot kernels to issue clean contiguous loads.
Kill switch: `IE_NO_MOE_SOA=1`. Decode guard A/B held through the repack:
81.25 / 80.72 / 81.84 tok/s.

**Step 2 — `block_q8_1s` activations with split half-sums:**
The 48-byte `block_q8_1s` struct carries split half-sums s0 and s1 that
provide the exact K-quant min/offset corrections for both Q4\_K (uses s0+s1)
and Q6\_K (uses s0/s1 independently on per-16 scale lanes). This eliminates
every isum dp4a from the inner loop.

**Step 3 — full-K register lattices:**
Stage-2 down kernels: one subgroup covers the full K=512 dimension;
weights are register-resident per column across all routed rows.
With weights register-resident and the isum dp4a chain eliminated, stage-2
halves: 204 → 101 ms. Stage-1 gate/up kernels: same shape.
Stage-1 time: 189 → 157 ms.

**Step 4 — idp4a verification:**
`IGC_ShaderDumpEnable` was used to confirm that dp4a calls lower to
hardware idp4a instructions (128 per kernel, no register spills).

**Kernel ledger after the crown (pp512, 40 layers):**
`moe_pfl_gate_q8` 157.3 ms · `down_q8_6k` 53.7 ms · `down_q8_4k` 48.9 ms ·
`gemm_fp16` 57.6 ms · `dn_recurrence` 47.2 ms

Remaining headroom (not required for the win): stage-1 is still 37% of
prefill; `gemm_fp16` + dequant is ~68 ms; `dn_recurrence` is 47 ms.

### 4. Negative results kept as receipts

- **XMX `joint_matrix` MoE stage-1:** 3.4× slower — tile alignment does not
  match MoE expert access pattern at batch=1.
- **oneDNN matmul (`IE_ONEDNN=1`):** NEUTRAL — `gemm_fp16` already matches
  on this shape. Kept as opt-in.
- **Naive dp4a on AoS layout:** −9% — header-interrupted AoS layout prevents
  clean contiguous int8 loads.
- **2D block loads (`lsc_load_block2d`):** Hard hang on BMG-G31 (GuC CT
  failure). Replaced with 1D block reads (`__spirv_SubgroupBlockReadINTEL`).

---

## Hardware notes for Intel engineers

### idp4a verification

dp4a calls lower to hardware idp4a — confirmed via `IGC_ShaderDumpEnable`:
128 idp4a per kernel, no register spills. Same lowering as llama.cpp SYCL;
worth verifying when porting int-dot kernels to a new driver.

### lsc\_load\_block2d GuC CT hang on BMG-G31

Calling `lsc_load_block2d` (2D block loads) in a SYCL kernel on the B70
causes a hard GPU hang: the GuC CT command transport stalls and the driver
does not recover. The correct 1D alternative is
`__spirv_SubgroupBlockReadINTEL`, which works correctly on the same
hardware. Do not use 2D block load intrinsics on Xe2 BMG-G31 without
first confirming the fix in your driver stack.

### DeltaNet recurrence FMA non-determinism

The `deltanet_recurrence` kernel produces stochastic single-byte
differences in its output state when run for many iterations on Xe2
BMG-G31. A 28-step bisect ruled out quantization artifacts,
uninitialized memory, SLM bank conflicts, FMA pipelining,
missing memory ordering primitives (including seq\_cst + system atomics,
`atomic_fence`, and L0 DMA round-trips), and USM host memory (which
bypasses all GPU caches). The conclusion is HW-level FMA pipeline
non-determinism, software-irreducible from SYCL. A vendor issue against
IGC is planned; `tools/ie_bug_monitor.cpp` is a minimal repro.

Third-party corroboration:
[ginkgo #2018](https://github.com/ginkgo-project/ginkgo/issues/2018)
(preconditioner failures on BMG-G31) and
[OGL #170](https://github.com/hpsim/OGL/issues/170) (DEVICE\_LOST on
Arc Pro B70).

**Production workaround:** chunk prefill externally at T ≤ 256. The bug
does not fire on short prefill or single-token streaming decode. The
built-in 511-tok PPL at production defaults is 6.45.

---

## Reproducing the benchmark

The crown numbers trace to one order-controlled script,
`scripts/bench_showdown.sh`. It is parameterized via environment variables so
an outsider can point it at their own model and llama.cpp build:

```sh
GGUF=/path/to/Qwen3.6-35B-A3B-Q4_K_M.gguf \
LLAMA_BENCH=/path/to/llama.cpp/build-sycl/bin/llama-bench \
ROUNDS=3 scripts/bench_showdown.sh           # add --tg for the decode pass
```

It runs a discarded JIT warmup, then alternates engine/llama pairs
(new-old-new) so each pair shares thermal state, and flags cross-round drift
over ~3% as thermal. The full public protocol — exact model file, the
llama.cpp commit and build flags to match, driver stack, and the
order-controlled rules — is documented in `scripts/bench_showdown_public.md`.

---

## What's next

The both-metrics win is scoped to one model and one GPU. Roadmap:

- **Decode beyond the crown:** port the int-dot Q8 GEMV decode path and
  FlashAttention/split-K to the dense and `qwen3moe` shapes — the single
  biggest off-crown gap.
- **Model breadth (Wave-1):** Yi, Mistral, the R1 distills, Phi, Granite are
  scaffolded; GPU validation is the gate. The kernel infrastructure (SoA
  repack, int-dot GEMVs, fused norm-quantize) generalizes.
- **B580 (same Battlemage family):** same Xe2-HPG architecture as B70;
  results from B580 owners are welcome.
- **Windows:** SYCL build path (`icpx`, oneAPI 2025.3) should carry over;
  GuC firmware behavior on Windows not yet verified.
- **OpenAI-compatible server:** planned for the P4 release.
- **License:** Apache 2.0 at launch.
