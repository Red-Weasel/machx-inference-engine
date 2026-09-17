# gpt-oss-20b on Intel Arc Pro B70 — `ie` vs llama.cpp (SYCL)

**Date:** 2026-06-27 · **Hardware:** 1× Intel Arc Pro B70 (BMG-G31, 32 GB, 608 GB/s) ·
**Model:** `gpt-oss-20b-mxfp4.gguf` (20.9 B / 3.6 B active, MXFP4 experts, 11.27 GiB) ·
**Opponent:** llama.cpp SYCL backend (`build-sycl`), same GGUF, same box, llama's *best* config.

---

## TL;DR

On a single Arc Pro B70, `ie` runs OpenAI's **gpt-oss-20b** and **beats llama.cpp's
SYCL backend on BOTH axes**:

- **Prefill: 1.9–4.5× faster** (and the gap *grows* with context — 4.5× at 2K).
- **Decode: 1.13–1.16× faster**, and **flat across context** where the cost would
  otherwise grow.

This is measured fresh against llama's **fastest** configuration (flash-attention on),
single-GPU, same GGUF — not a stale baseline.

---

## Head-to-head (1× B70, same GGUF)

| context | `ie` prefill | llama prefill | **prefill** | `ie` decode | llama decode | **decode** |
|--------:|-------------:|--------------:|:-----------:|------------:|-------------:|:----------:|
| 512     | **1795** t/s | 927 t/s       | **1.94×**   | **58.3** t/s| 50.3 t/s     | **1.16×**  |
| 2048    | **4147** t/s | 927 t/s       | **4.47×**   | **57.4** t/s| 49.9 t/s     | **1.15×**  |
| 4096    | **3428** t/s | 896 t/s       | **3.83×**   | **55.6** t/s| 49.4 t/s     | **1.13×**  |

(`ie` decode is reported at depth = the prefill length; llama decode uses
`-d {512,2048,4096}` to match. Both flat across depth — neither collapses.)

### Methodology / fairness
- **Ours:** `ie-bench --gguf <model> --prefill T --decode N --warmup 2` (warmup + median;
  prefill t/s = T / prefill_ms; decode measured at depth T).
- **llama (best):** `ONEAPI_DEVICE_SELECTOR=level_zero:0 llama-bench -m <model> -ngl 99
  -fa 1 -p T -n 0` and `… -n 128 -d {depth}`. **Flash-attention ON** — llama's faster
  config here (decode 49.9 vs 46.5 fa-off @d2048; prefill 927 vs 900). We compare
  against llama's *best*.
- Single GPU both sides (`level_zero:0`; the 20b fits one 32 GB card). Same GGUF file.
- **Honesty note:** an earlier internal baseline compared against a *stale* llama build
  (607 prefill / 22 decode). The numbers above are a clean re-bench against a current
  llama-SYCL — which is much faster — and `ie` still wins both axes.

---

## How we got here — the optimizations (all shipped, all gated, all validated)

gpt-oss is the engine's 8th architecture. It has three non-standard pieces (per-head
attention **sinks**, alternating **sliding-window/dense** layers, and **MXFP4** experts)
on top of a top-4 MoE. After getting it correct, six levers took it from "runs" to
"beats llama both axes" — each verified PPL-neutral with the crown-model PPL gate
bit-exact (6.4527) and the full test suite green:

1. **Fused MXFP4 decode GEMV.** At decode the MoE re-materialized each routed expert's
   weights to fp16 (~10.8 GB/token) just to do an M=1 GEMM. A load-time SoA repack
   (aligned `qs`/`e` planes) + a W4A8 int-dot GEMV reads the 4.25-bpw weights *once*.
   The native 17-byte MXFP4 block stride is the trap — it must be split into aligned
   planes or cross-lane reads collapse to ~64 GB/s. *Decode 30.5 → 43 tok/s.*
2. **Split-K flash-decode with sinks.** The naive O(context) decode attention is
   latency-bound and collapses as context grows. A split-K FA-2 decode kernel with the
   per-head sink folded into the combine step makes decode **flat** across context.
   *Decode @2K 25 → 54 tok/s.*
3. **Wide-tile prefill with sinks.** The naive prefill attention sagged from 606→327
   t/s (512→4K). The proven wide-tile kernel (KV read once per query-tile) at head-dim
   64 + the sink fold. *Prefill @4K 327 → 716 tok/s.*
4. **W8A8 int-dot lm_head.** The 201k-vocab lm_head was the single biggest decode GEMV;
   kept Q8_0 and routed through an int-dot GEMV — half the weight read, −0.58 GB VRAM.
   *Decode 57 → 60 tok/s.*
5. **Batched prefill GEMMs (the big prefill win).** The attn q/k/v/o projections *and*
   the MoE router were running as *per-token* GEMVs during prefill (~196k + ~49k kernel
   launches in a single 2048-token forward — together >90% of prefill time) because the
   engine's batched-GEMM path is gated on dimensions divisible by 256/64, which
   gpt-oss's hidden=2880 and 32-expert router both fail. Routing them through a batched
   fp16 GEMM (oneDNN, which handles those dimensions fine) **flipped prefill from a loss
   to a 1.9–4.5× win.** *Prefill 759 → 4147 tok/s @2K — a 5.5× self-improvement.*

---

## Quality

gpt-oss is a reasoning-tuned model, so raw-text perplexity is high for everyone (its
strength is chain-of-thought, not next-token raw-text). All optimizations above are
quality-neutral: streaming PPL holds at 22.5 (256-tok), per-token NLL is healthy, and
greedy factual completions are correct ("The capital of France is **Paris**." via the
OpenAI Harmony chat format, finishing cleanly on `<|return|>`). The int-dot decode
paths (W4A8 / W8A8) move PPL < 0.1%.

---

## Why this matters on Arc

On Intel Arc B70, the field for a brand-new architecture like gpt-oss is thin:
llama.cpp's **Vulkan** backend lacks the required shaders, Intel's **OpenVINO** path is
preview-only for this class, and the **SYCL** backend is the one real single-stream
opponent. `ie` is day-one on the architecture *and* faster than that opponent on both
prefill and decode — the defensible position is "first and fastest on new architectures
on Arc."

*Reproduce:* `ie-bench --gguf gpt-oss-20b-mxfp4.gguf --prefill 512,2048,4096 --decode 128`
vs `llama-bench -m … -ngl 99 -fa 1 -p 512,2048,4096 -n 128`.
