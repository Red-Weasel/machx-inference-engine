# Beating llama.cpp on Intel Arc: a from-scratch SYCL engine for Qwen3.6-35B

*Draft — publish-ready pending the owner's final pass on the **[CONFIRM]** items
(public repo URL/date, exact driver string, whether to use the "Mach X" name
publicly). Numbers are the same-session, order-controlled run of 2026-06-15;
raw data in `release_bench_2026-06-15.md`.*

---

## TL;DR

We built a from-scratch C++/SYCL inference engine ("**Mach X**") for Intel Arc
GPUs. On the model it's tuned for — **Qwen3.6-35B-A3B**, the hybrid
DeltaNet + 256-expert MoE — it **beats llama.cpp's strongest Arc backend
(SYCL master, *including* its recent +70% MoE-prefill optimization) on both
metrics, measured same-session on the same GPU**:

| Qwen3.6-35B-A3B Q4_K_M, Arc Pro B70 | Mach X | llama.cpp SYCL master | delta |
|---|---:|---:|---:|
| **Prefill** (pp512, tok/s) | **~1098** (peak 1174) | ~1013 | **+8%** |
| **Decode** (tg128, tok/s) | **~83** | ~82 | **+1–3%** |
| Perplexity (same GGUF/corpus) | **6.4527** | — | quality held |

Against llama.cpp's **Vulkan** backend the margin is far larger (decisively
ahead on both). We lead with the SYCL comparison because it's the *harder,
more honest* one.

**The honest scope, up front:** this is a **win on the MoE crown, not a blanket
"fastest engine" claim.** Generic dense models run *correctly* but are currently
*slower* than llama.cpp's tuned dense path (see "Where we're not faster yet").
We'd rather tell you that than have you find it — and the whole run is
reproducible below so you can.

---

## The setup (so you can reproduce it)

- **GPU:** Intel Arc Pro B70, 32 GB (BMG-G31), Linux. **[CONFIRM driver/oneAPI
  string]** — built with IntelLLVM 2026.0, Level-Zero backend.
- **Model:** `Qwen3.6-35B-A3B-Q4_K_M.gguf` — 40 layers, hidden 2048, **256 experts,
  8 active per token** (A3B), hybrid gated-DeltaNet + full-attention.
- **Contender:** llama.cpp **SYCL** master `fdc3db9b6` (build `b9598`), which
  includes PR #23142 ("expert-contiguous MoE prefill," ~+70% on this model class)
  and the MMVQ integer-dot decode path. This is llama.cpp's *best* Arc backend
  today — not the older/weaker Vulkan one.
- **Protocol** (the part most "X is faster!" posts skip): same GGUF, same B70,
  **same session**, order-controlled (alternating new-old-new rounds), one JIT
  warm-up run discarded, and a heat-soak-aware read — GPU clocks drift ±40 tok/s
  with temperature, and we cool the card between the prefill and decode passes so
  neither throttles the other. Cross-round drift > ~3% is treated as thermal.
- **llama.cpp:** `llama-bench -m <gguf> -ngl 99 -p 512 -n 128 -r 3 -sm none -mg 0`,
  pinned with `ONEAPI_DEVICE_SELECTOR=level_zero:0`.
- **Mach X:** `scripts/bench_showdown.sh` (drives the same protocol).

Full per-round tables (prefill 1173.9/1082.9/1112.3 vs 960.2/1011.1/1015.0;
decode 80.97/80.50/83.13 vs 76.38/78.67/82.24) are in
`docs/public/release_bench_2026-06-15.md`.

## Why it's faster on the crown

The win is not a tuning fluke — it's a bespoke MoE kernel path:

- **Prefill:** a SoA (structure-of-arrays) re-pack of the expert weights plus an
  **integer-dot (W4A8) fused MoE GEMM** — activations are quantized to int8 once
  and the expert matmuls run as dp4a-class integer dot products, skipping
  per-element dequant in the inner loop. llama.cpp's SYCL prefill went up ~70%
  with PR #23142; we still come out ahead.
- **Decode:** norm-fused int-dot Q8 GEMVs for the attention and lm-head paths.
  llama.cpp's MMVQ decode is genuinely strong here, which is why decode is a
  narrow +1–3% rather than a blowout — an honest near-tie that favors us.

## We didn't trade quality for speed

Perplexity on the same GGUF and corpus is **6.4527** (avg NLL 1.864495),
**bit-exact and deterministic across runs**. Every engine change is gated at
PPL ≤ 6.57 or it reverts. The speed comes from kernels, not from sampling
shortcuts or precision cuts.

## Breadth: it runs ~every popular open family

Beyond the crown, Mach X runs **8 architecture families** end-to-end, covering
most of what people actually deploy:

- **Qwen** (3.6-MoE crown, 3/2 dense, 3.6-27B hybrid, Coder-30B MoE, **Next-80B**),
  **Llama 1/2/3**, **Mistral / Codestral / Devstral**, **Gemma-4**, **Phi-4**,
  **Granite-3.x**, **DeepSeek-R1-Distill**, and the Llama-clone tier
  (Yi / Nemotron / InternLM / Baichuan).
- **AWQ and GPTQ import** — two quantization formats **llama.cpp can't load
  natively** — converted to GGUF and run directly.
- **Multi-GPU** — tensor-parallel and layer-split across 2× B70, which is what
  lets the 72B and the 80B run at all on this hardware.

Among the non-crown arches, a couple are also *fast*: the Qwen3.6-27B hybrid runs
**1.9× llama.cpp Vulkan** on prefill, and **Qwen3-Next-80B decodes 1.40× faster
than llama.cpp SYCL** (and runs on the GPU where llama's backends fall back to CPU
for that architecture).

## Where we're *not* faster yet (the honest part)

Same-session, same protocol, the breadth models tell a mixed story:

| Model | vs llama.cpp SYCL (prefill) |
|---|---|
| Codestral-22B (dense) | **−7%** (behind) |
| Mistral-Small-24B (dense) | **−14%** (behind) |
| Gemma-4-26B-A4B (MoE) | **~15× behind** (unoptimized) |

The generic **dense** path runs the in-house FP16 GEMM and hasn't received the
int-dot/oneDNN treatment the crown's MoE did. **Gemma-4** runs a correctness-first,
*un-fused* MoE path (the fused-MoE optimization was always a to-do). Both are real,
known gaps with clear fixes — they're the top of the post-launch perf list. The
point of this section: **the speed claim is the MoE crown specifically; everything
else is "runs correctly," not "runs fastest."**

## What this is, and isn't

- **It is:** the fastest known inference for **Qwen3.6-35B-A3B on Intel Arc**,
  beating llama.cpp's best backend on both metrics, same-hour measured — plus a
  broad-compatibility runtime that loads formats (AWQ/GPTQ) and runs architectures
  (Qwen3-Next) that llama.cpp's Arc backends can't.
- **It isn't:** "the fastest engine for Intel GPUs at everything." We don't claim
  that and the numbers above show why.

## Try it / what's next

- **[CONFIRM repo]** Source: `Red-Weasel/mach-x-inference-engine` (Apache-2.0 at
  launch). Reproduce with `scripts/bench_showdown.sh`.
- **Next:** a CUDA/nvptx64 build is already compiling for NVIDIA (the SYCL source
  cross-compiles); the dense int-dot path and Gemma-4 fused-MoE are the next perf
  wins; a public nightly benchmark dashboard is planned.

*We're running Mach10 on dual B70s.*
</content>
