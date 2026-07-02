# Inference Engine

Native LLM inference engine targeting the Intel Arc Pro B70, written from scratch in C++/SYCL. Daily-driver model: `Qwen/Qwen3.6-35B-A3B`.

> **Status:** v1 shipped; **8+ architectures supported** (Qwen3.6 DeltaNet, Qwen3/Coder MoE, dense Qwen/Llama, Gemma-4, Phi-3, Mistral/Granite, and **gpt-oss-20b/120b** — the 8th arch). **Multi-GPU tensor-parallel is shipped in-product** (`ie run/serve --gpus N`). See [MASTER_DEV_PLAN.md](MASTER_DEV_PLAN.md) for the authoritative state and roadmap.

## Supported models

| architecture (GGUF `general.architecture`) | example | status |
|---|---|---|
| `qwen35moe` (hybrid DeltaNet + MoE) | Qwen3.6-35B-A3B Q4_K_M | crown: pp512 prefill faster than llama.cpp SYCL master; tg128 decode ≈parity, same-hour paired runs † |
| `qwen35` (hybrid DeltaNet + dense MLP) | Qwen3.6-27B Q4_K_M | validated: per-layer cosine ≥0.9995 vs llama oracle, PPL 5.34; prefill faster than llama.cpp **Vulkan**; **decode parity vs Vulkan only** — vs llama.cpp **SYCL** (the faster single-stream opponent) 27B decode is a **LOSS**, never claimed. Re-verified prefill wins vs SYCL (grow with ctx) → see `docs/COMPETITIVE_SCORECARD_2026-06-25.md` §4 † |
| `qwen3moe` (standard MoE) | Qwen3-Coder-30B-A3B Q4_K_M | validated: coherent generation, PPL 11.99 oracle-consistent (llama 20.19 windowed, same directional ratio as the dense/27B paths); dense QK-norm attention + top-k MoE FFN (128 experts/8 used), decode 35 tok/s (unfused v1) |
| `gpt-oss` 20b (OpenAI MoE, MXFP4) | gpt-oss-20b-mxfp4 (20.9B / 3.6B active, 11.27 GiB), **1× B70** | **WIN both axes vs llama.cpp SYCL** (its FA-on best config, same GGUF; measured 2026-06-27, clean-box re-bench pending): prefill **1795 / 4147 / 3428** vs **927 / 927 / 896** t/s = **1.94× / 4.47× / 3.83×** (pp512 / 2K / 4K); decode **58.3 / 57.4 / 55.6** vs **50.3 / 49.9 / 49.4** t/s = **1.16× / 1.15× / 1.13×** (tg@512 / 2K / 4K) — both flat across depth. Streaming PPL 22.5; Harmony chat ("Paris", finish=stop) + function-calling tool use. Repro: `ie-bench --warmup 2` vs `llama-bench -ngl 99 -fa 1`. |
| `gpt-oss` 120b (OpenAI MoE, MXFP4) | gpt-oss-120b across **2× B70 tensor-parallel** (no working P2P) | runs TP via `ie run/serve --gpus 2`: decode **~31 t/s** (peak 32.05; ±15% box noise), prefill **538–679 t/s** @ ctx ≤16k (measured 2026-06-29 on a degraded box — re-bench pending). Batched-prefill **PPL 15.1985, bit-identical to T=1**; load card0 29.3 GB VRAM + 3.4 GB host-spill (display-safe), card1 31.1 + 0.3. Coherent Harmony chat (math/poem/factual + multi-turn) + tool calling. vs LM Studio layer-split **~12.42 t/s** (owner-reported, same 2 cards). |
| `qwen3` / `qwen2` (dense) | Qwen3-8B Q4_K_M | validated bit-exact vs llama.cpp (layer/greedy/PPL parity — `scripts/p2_parity_qwen3.sh`); prefill +15% vs llama.cpp SYCL †, decode unoptimized |
| `llama` (dense, Llama-3.x) | Meta-Llama-3.1-8B-Instruct + Llama-3.2-3B (tied embeds) Q4_K_M | validated: per-layer cosine 0.99998–1.0 vs llama oracle, PPL 10.79 (`scripts/p3_parity_llama3.sh`); load-time Q/K un-permute + rope_freqs scaling |
| AWQ / GPTQ import | Qwen2.5/Qwen3 4-bit safetensors | `ie import <hf_dir> <out.gguf> <tok_ref.gguf>` → native GGUF (formats llama.cpp can't load); `IE_IMPORT_STREAM=1` for RAM-safe import of huge models (72B) |
| **multi-GPU (dense)** | models > one card's VRAM, e.g. Qwen2.5-72B across 2× B70 | **`ie run --gpus 2 <dense.gguf>`** / **`ie serve --gpus 2 …`** (tensor-parallel, in-product) — also `ie-multi-gpu-run [--tp]`. Layer-split is **bit-identical** to single-GPU (`ie-multi-gpu-equality-test`); tensor-parallel is cosine-validated (0.999999, `ie-multi-gpu-tp-test`), **1.44× faster decode** than layer-split. 72B PPL 8.97. |

† Non-gpt-oss competitive figures predate 2026-06-27 and are pending a clean-box re-bench; see `docs/COMPETITIVE_SCORECARD_2026-06-25.md`. (The gpt-oss rows are ledger-verified; see `docs/public/marketing/VERIFIED_CLAIMS_2026-06-29.md`.)

## Quick start

**Docker (recommended)** — one build, then serve any GGUF on your Intel Arc GPU:
```bash
docker build -t ie-engine .                          # ~15 min, one time
./scripts/ie-docker pull llama8b                     # or any HF GGUF
./scripts/ie-docker serve /models/…/model.gguf --gpus 1
# → OpenAI-compatible server on :11435 (point any OpenAI client at it)
```
Full 5-minute path in [QUICKSTART.md](QUICKSTART.md).

**From source** (needs oneAPI 2026.x + an Intel Arc GPU):
```bash
source scripts/env.sh                # oneAPI + toolchain on PATH
cmake -S . -B build -G Ninja && cmake --build build -j
./build/src/ie pull llama8b
./build/src/ie serve <model.gguf> --gpus 1
```

## Layout
- `PLAN.md` — phase-by-phase implementation plan and gates.
- `research/` — load-bearing technical references (hardware, programming model, quant formats, model spec).
- `examples/` — small smoke tests.
- `bench/` — kernel benchmarks; `bench/baseline/` holds Phase 0 numbers.
- `src/`, `include/`, `tests/` — engine code (added per phase).
- `scripts/` — env setup, toolchain probe.

## Hardware target
Intel Arc Pro B70 (BMG-G31, 32 GB GDDR6, 608 GB/s, ~183 FP16 TFLOPS via XMX).

## Toolchain
- Compiler: oneAPI 2026.x DPC++ (`icpx`)
- Driver baseline: compute-runtime ≥ 26.05.37020.3 / IGC ≥ 2.28.4
- AOT target: `intel_gpu_bmg_g31` (the Docker image builds SPIR-V/JIT for portability)

## License

Apache License 2.0 — see [LICENSE](LICENSE). Copyright © 2026 Red-Weasel.

Free to use, modify, and ship (including commercially). Apache-2.0's patent grant +
retaliation clause protects you and downstream users.

## Support

If this saved you time — or you just want to see more fast inference land on Intel
Arc — you can support the work:

[![Ko-fi](https://img.shields.io/badge/Support%20on-Ko--fi-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/YOUR_HANDLE)

> Replace `YOUR_HANDLE` above (and fill in `.github/FUNDING.yml`) with your Ko-fi /
> Buy Me a Coffee / GitHub Sponsors account to activate the button.
