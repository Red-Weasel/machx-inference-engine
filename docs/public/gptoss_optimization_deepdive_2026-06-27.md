# Making gpt-oss-20b fast on Intel Arc — seven kernel-level wins

*How `ie` went from "gpt-oss runs" to "gpt-oss beats llama.cpp on both axes" on a
single Arc Pro B70 — and the one bug that was worth a 5.5× prefill speedup.*

**Hardware:** 1× Intel Arc Pro B70 (BMG-G31 / Xe2, 32 GB, 608 GB/s). **Model:**
OpenAI `gpt-oss-20b` (20.9 B / 3.6 B active, MXFP4 experts). **Result:** prefill
**1.9–4.5×** and decode **1.13–1.16×** vs llama.cpp's SYCL backend at its best config.

---

## The architecture

gpt-oss is a top-4 MoE with three pieces most transformers don't have, each of which
needs its own kernel work:

- **Attention sinks** — a per-head learned scalar that enters the softmax *denominator*
  as a virtual always-attended key with no value (attention mass "leaks out").
- **Alternating sliding-window / dense** layers — even layers attend a 128-token
  window, odd layers attend the full context.
- **MXFP4 experts** — 4.25-bit weights: a shared E8M0 exponent + 32 FP4 nibbles per
  block, decoded through a small signed lookup table.

Getting it numerically correct was step one. Making it *fast* was seven levers.

---

## 1. The MXFP4 decode GEMV, and the 17-byte trap

At decode, the MoE was re-materializing each routed expert's weights to fp16 just to
multiply by a single activation vector — ~10.8 GB of memory traffic per token against a
1.27 GB floor (the weights are only 4.25 bpw). The fix is a fused GEMV that reads the
packed weights once.

The trap: an MXFP4 block is **17 bytes** (1 exponent + 16 nibble bytes). A 17-byte
stride means *no* cross-lane access pattern is coalesced — a first cut over the native
layout ran at **64 GB/s, ~10% of peak**. The fix is a load-time **Structure-of-Arrays
repack**: split each expert into an aligned nibble plane and an aligned exponent plane.
Now 16 lanes striding 32-element blocks read consecutive 16-byte chunks. *Decode
30.5 → 43 tok/s.*

A second subtlety: the canonical CUDA kernel maps nibbles → int8 with a byte-permute
(`prmt`), but on Intel Xe that permute is **software-emulated** (~30 ALU ops). We decode
the FP4 (E2M1) value with branchless bit math in registers instead — no memory LUT, no
emulated permute.

## 2. Flash-decode with sinks — killing the long-context collapse

The naive decode attention does a serial reduction over the whole KV cache: it's
latency-bound and *collapses* as context grows (25 tok/s at 2K). We route the
full-attention layers through a split-K flash-decode kernel and fold the per-head sink
*once*, after the per-chunk partials are merged:

```
m' = max(m, sink);   ℓ = ℓ·exp(m−m') + exp(sink−m');   out ·= exp(m−m')
```

The sink contributes to the denominator but not the value accumulator — exactly the
"mass leaks out" semantics, now in a parallel kernel. Decode goes **flat** across
context. *Decode @2K 25 → 54 tok/s.*

## 3. Wide-tile prefill with sinks

The same sink fold, this time at the end of a query-row tile, dropped into the proven
wide-tile prefill kernel (which reads K/V once per tile instead of per query). The sink
was threaded as a *compile-time* template parameter so every other model that shares the
kernel gets byte-identical code. *Prefill @4K 327 → 716 tok/s.*

## 4. A W8A8 lm_head

The 201k-vocab lm_head is the single biggest decode GEMV. Keeping it in Q8_0 and running
an int-dot GEMV (instead of expanding to fp16) halves the weight read and saves 0.58 GB
of VRAM. *Decode 57 → 60 tok/s.* (This same Q8-non-expert pattern is the lever that will
help fit the 120B sibling.)

## 5. The bug worth 5.5× — per-token GEMVs at prefill

This is the one that mattered most, and it was invisible until we profiled per-kernel.

A prefill of 2048 tokens was issuing **196,608** GEMV launches in a single forward pass —
88% of prefill time. The cause: the engine's batched-GEMM path is gated on dimensions
divisible by 256 (the K dimension) and 64 (the N dimension) — an XMX matrix-tile
constraint. gpt-oss's **hidden size is 2880** (2880 % 256 = 64), so the attention
q/k/v/o projections silently fell back to the *decode* path — a per-row GEMV — looped
once per token. Then, after fixing that, the **32-expert router** (N = 32, fails % 64)
surfaced as the next 54%.

The fix is almost anticlimactic: route those prefill GEMMs through oneDNN directly, which
handles arbitrary dimensions (the MoE expert GEMM already called it with K = 2880). One
helper, two call sites.

*Prefill 759 → 4147 tok/s at 2K — a **5.5×** self-speedup, and the difference between
losing prefill 0.75× and winning it 4.5×.*

**Lesson:** always per-kernel-profile a new architecture for a per-token GEMV explosion.
A dimension gate written for one matrix-multiply backend can silently route a whole class
of weights onto the slow path.

## 6. An honest negative result

We hypothesized that the MoE GEMV's signed×signed dot (`dp4a_ss`) might be
software-emulated on BMG-G31, and tried the unsigned `dp4a_us` form with an offset
correction. It was **slower** — `dp4a_ss` *is* hardware-lowered here, and the offset's
extra reductions just added work. The MoE GEMV is genuinely ALU-co-limited at its
ceiling, not held back by the dot. We keep the result documented so nobody re-runs it.

## 7. Correctness as a constraint, not an afterthought

Every lever above held the engine's reference perplexity gate **bit-exact** (a separate
crown model, NLL 1.864495) and the full 31-test suite green. The int-dot decode paths
(W4A8 / W8A8) move gpt-oss's own perplexity by < 0.1%. The wide-tile prefill moves it
0.16% (floating-point reduction order, the same band as our other production tile
kernels). Greedy factual completions stay correct through the OpenAI Harmony chat format.

---

## Results

| context | `ie` prefill | llama prefill | **prefill** | `ie` decode | llama decode | **decode** |
|--------:|-------------:|--------------:|:-----------:|------------:|-------------:|:----------:|
| 512     | 1795 t/s     | 927 t/s       | **1.94×**   | 58.3 t/s    | 50.3 t/s     | **1.16×**  |
| 2048    | 4147 t/s     | 927 t/s       | **4.47×**   | 57.4 t/s    | 49.9 t/s     | **1.15×**  |
| 4096    | 3428 t/s     | 896 t/s       | **3.83×**   | 55.6 t/s    | 49.4 t/s     | **1.13×**  |

llama.cpp SYCL, flash-attention on (its faster config), single B70, same GGUF.

**A note on fairness:** an earlier internal baseline compared against a stale llama build
(607 prefill / 22 decode). Everything above is a fresh re-bench against a *current*
llama-SYCL — which is much faster than that old number — and `ie` still wins both axes.
Publishing a win against a stale opponent isn't a win.

---

## Why Arc

For a brand-new architecture on Intel Arc, the field is thin: llama.cpp's Vulkan backend
lacks the shaders, Intel's OpenVINO path is preview-only for this class, and the SYCL
backend is the one real single-stream opponent. Being day-one on the architecture *and*
faster than that opponent on both prefill and decode is the whole game: **first, and
fastest, on new architectures on Arc.**
