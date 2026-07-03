# Mach X — LLM Inference Engine for Intel Arc

**A from-scratch C++/SYCL inference engine for Intel Arc GPUs. Built and tuned on 2× Arc Pro B70 — and it beats llama.cpp on Arc, on both prefill *and* decode.**

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Language](https://img.shields.io/badge/C%2B%2B20-SYCL%20%2F%20DPC%2B%2B-orange)
![Platform](https://img.shields.io/badge/Intel%20Arc-Battlemage%20B70-0071C5?logo=intel&logoColor=white)
![Architectures](https://img.shields.io/badge/architectures-8%2B-brightgreen)
![Multi-GPU](https://img.shields.io/badge/multi--GPU-tensor--parallel-success)

Intel Arc is a genuinely capable AI GPU that inference tooling has mostly ignored. **Mach X is built for it from the metal up** — no fork of llama.cpp, no PyTorch, no vendor runtime. Hand-written SYCL kernels (XMX matrix engines, int-dot quantized GEMV, tiled FlashAttention), an OpenAI-compatible server, tensor-parallel multi-GPU, and day-one support for the newest model architectures — often running them fast on Arc *before* anyone else does.

---

## ⚡ Headline: gpt-oss-20b, head-to-head vs llama.cpp

Same GGUF, same GPU (1× Arc Pro B70), llama.cpp on its *fastest* config (FlashAttention on):

| context | **Mach X** prefill | llama.cpp | speedup | **Mach X** decode | llama.cpp | speedup |
|---|---|---|---|---|---|---|
| 512  | **1795** t/s | 927 | **1.94×** | **58.3** t/s | 50.3 | **1.16×** |
| 2K   | **4147** t/s | 927 | **4.47×** | **57.4** t/s | 49.9 | **1.15×** |
| 4K   | **3428** t/s | 896 | **3.83×** | **55.6** t/s | 49.4 | **1.13×** |

**Wins both axes at every context length, and stays flat as context grows.** Clean-box, reproducible (`ie-bench` vs `llama-bench`).

---

## Highlights

- 🏛 **8+ architectures** — Qwen3.6 (hybrid gated-DeltaNet MoE), Qwen3 / Coder / Tongyi MoE, dense Qwen / Llama / Mistral / Granite / Phi, Gemma-4, Qwen3-Next-80B, and OpenAI **gpt-oss 20b + 120b**.
- 🥇 **Beats llama.cpp on Arc** — on prefill *and* decode across the models below.
- 🧠 **Runs the big ones** — gpt-oss-**120b** (117B) and Qwen3-Next-**80B** on 2× B70 via tensor-parallel; **~2.5× faster than LM Studio** on 120b.
- 🔀 **Multi-GPU built in** — `ie serve --gpus 2` (tensor-parallel + layer-split), no P2P required.
- 🔌 **OpenAI-compatible server** + tool-calling (Harmony + Qwen) — point any OpenAI client (or [Hermes](https://github.com/NousResearch)) at `:11435`.
- 📦 **One-command Docker** — `docker pull` (or build) → `ie-docker serve` → running on your Arc GPU in minutes.
- ✅ **Correctness-first** — PPL-validated, per-layer cosine ≈ 1.0 vs a llama.cpp oracle, bit-exact where claimed.

---

## Benchmarks

📊 **[Interactive charts →](https://red-weasel.github.io/machx-inference-engine/benchmarks.html)** · all measured on **Arc Pro B70** hardware; gpt-oss rows are clean-box head-to-head with identical GGUFs.

**gpt-oss-120b** (117B, MXFP4) — 2× B70, tensor-parallel:
| metric | Mach X | LM Studio (same 2 cards) |
|---|---|---|
| decode | **~31 tok/s** (peak 32) | ~12.4 tok/s |
| fit | full MXFP4, display-safe | — |

Coherent Harmony chat (math / poem / factual + multi-turn) and function-calling tool use. Batched-prefill PPL 15.20, bit-identical to T=1.

**Qwen3.6-35B-A3B "crown"** (all-Q8_0, ~36 GB) — 2× B70 vs llama.cpp SYCL layer-split:
| axis | Mach X | llama.cpp | speedup |
|---|---|---|---|
| prefill | **963** t/s | 763 | **1.26×** |
| decode | **63** t/s | 42 | **1.49×** |

PPL 6.36. Hybrid gated-DeltaNet + 128-expert MoE — one of the hardest architectures to run correctly, let alone fast.

**Tongyi-DeepResearch-30B** (qwen3moe) — 2× B70 tensor-parallel, long context (~17K):
| axis | layer-split | tensor-parallel | speedup |
|---|---|---|---|
| prefill | 124 t/s | **291** t/s | **2.35×** |
| decode | 21 t/s | **27.4** t/s | **1.30×** |

**Gemma-4** prefill (sliding-window attention) vs llama.cpp: **2.03× @4K**, **1.91× @8K**, **1.58× @16K**.

**Qwen3.6-27B** dense vs llama.cpp SYCL: prefill **1.21×** (349 vs 288 t/s).

**Speculative decode** (self-drafting MTP head, lossless-greedy): Gemma-4 **1.46×**, Qwen3.6-27B **1.47×**.

> Methodology: `ie-bench --prefill P --decode N` mirrors `llama-bench -pP -nN`; runs are order-controlled and heat-soaked. A few non-gpt-oss figures predate the latest clean-box sweep and are being re-verified — the gpt-oss head-to-heads are ledger-verified.

---

## Supported architectures

| family | examples | notes |
|---|---|---|
| **Qwen3.6** (hybrid gated-DeltaNet) | 35B-A3B MoE, 27B dense | the flagship; DeltaNet recurrence + full-attn + MoE |
| **Qwen3 MoE** | Coder-30B-A3B, Tongyi-30B | dense QK-norm attention + top-k MoE (128 experts / 8 active) |
| **Qwen3-Next** | 80B-A3B | DeltaNet + full-attn + 512-expert MoE |
| **gpt-oss** | 20b, 120b (MXFP4) | OpenAI MoE + attention sinks + Harmony chat |
| **Gemma-4** | 31B dense, 26B-A4B MoE | per-layer head dims, sandwich norms, softcap, SWA |
| **dense** | Llama-3.x, Qwen2/3, Mistral, Granite, Phi-3 | bit-exact vs llama.cpp |
| **import** | AWQ / GPTQ safetensors | `ie import` → native GGUF (formats llama.cpp can't load) |

---

## Built & tested on

**2× Intel Arc Pro B70** — Battlemage (BMG-G31), 32 GB GDDR6 each (**64 GB total**), **608 GB/s** bandwidth, ~183 FP16 TFLOPS via XMX. oneAPI 2026.x / SYCL. All single- and multi-GPU benchmarks above are on this hardware.

---

## Quick start

**Docker (recommended)** — pull the prebuilt image (or build it yourself), then serve any GGUF on your Arc GPU:
```bash
docker pull ghcr.io/red-weasel/ie-engine:latest && docker tag ghcr.io/red-weasel/ie-engine:latest ie-engine
# ── or build from source (~15 min):   docker build -t ie-engine .
./scripts/ie-docker pull llama8b                     # or any Hugging Face GGUF
./scripts/ie-docker serve /models/…/model.gguf --gpus 1
# → OpenAI-compatible server on :11435 (point any OpenAI client at it)
```
Full 5-minute path in **[QUICKSTART.md](QUICKSTART.md)**.

**From source** (needs oneAPI 2026.x + an Intel Arc GPU):
```bash
source scripts/env.sh
cmake -S . -B build -G Ninja && cmake --build build -j
./build/src/ie pull llama8b
./build/src/ie serve <model.gguf> --gpus 1
```

Multi-GPU: add `--gpus 2` (VRAM-aware; tensor-parallel + layer-split). Runs models bigger than one card — e.g. Qwen2.5-72B or gpt-oss-120b across 2× B70.

---

## Under the hood

- **Quantized GEMV** — W4A8/W6A8/W8A8 int-dot kernels (dp4a) over SoA-repacked weights: read each weight once, decode in-register. Q4_K, Q6_K, Q8_0, Q5_K, MXFP4.
- **FlashAttention** — register-tiled SIMD inner loop (no XMX for attention, following the fastest llama-SYCL path), plus split-K decode, sliding-window, and attention-sink variants.
- **MoE** — expert-batched weight-stationary prefill + fused gate/up/down; oneDNN XMX GEMM for the large-M regime.
- **Multi-GPU** — head-sharded attention + expert-sharded MoE (tensor-parallel) with host-bounced all-reduce; layer-split for pure capacity (bit-identical to single-GPU).
- **Speculative decode** — self-drafting NextN/MTP head with batched int-dot verify, lossless vs greedy.

See **[MASTER_DEV_PLAN.md](MASTER_DEV_PLAN.md)** for the authoritative state and roadmap.

---

## License

**Apache License 2.0** — see [LICENSE](LICENSE). Copyright © 2026 Red-Weasel.

Free to use, modify, and ship (including commercially). Apache-2.0's patent grant + retaliation clause protects you and downstream users.

## Support

If Mach X saved you time — or you just want to see more fast inference land on Intel Arc — you can support the work:

[![Ko-fi](https://img.shields.io/badge/Support%20on-Ko--fi-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/redweasel)

☕ **[ko-fi.com/redweasel](https://ko-fi.com/redweasel)**

**All donations go straight back into the project.** Requests and suggestions are welcome — [open an issue](https://github.com/Red-Weasel/machx-inference-engine/issues).
