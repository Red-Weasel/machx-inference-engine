# Quickstart

Run the engine in ~5 minutes on a machine with an **Intel Arc B-series GPU** (e.g. B70).

## Requirements
- Intel Arc B-series GPU + the `i915`/`xe` kernel driver, with `/dev/dri/renderD*` present
- Docker

Check the GPU is visible to the host:
```bash
ls /dev/dri/            # expect renderD128 (and renderD129… for multi-GPU)
```

## 1. Get the image
**Pull the prebuilt image** (fastest):
```bash
docker pull ghcr.io/red-weasel/ie-engine:latest && docker tag ghcr.io/red-weasel/ie-engine:latest ie-engine
```
**Or build it yourself** (one time, ~15–20 min):
```bash
docker build -t ie-engine .
```
The build installs oneAPI 2026.x from Intel's apt repo and compiles the engine to
SPIR-V. The runtime image bundles the Intel Level-Zero compute runtime (which
specializes the SPIR-V for your Arc device on first load), so the host only needs
the kernel driver.

## 2. Pull a model
```bash
./scripts/ie-docker pull llama8b            # curated (see: ie-docker pull --list)
# or any GGUF from Hugging Face:
./scripts/ie-docker pull bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```
Models land in `./models` on the host.

> **Which quant to grab.** Best-tested: **`Q4_K_M`, `Q6_K`, `Q8_0`** (and **`MXFP4`** for gpt-oss). If a model fails to load, try one of those first — some quant × architecture combinations (e.g. certain `Q5_K` builds) aren't supported yet. The curated `ie pull` names already use known-good quants.

## 3. Serve (OpenAI-compatible, port 11435)
```bash
./scripts/ie-docker serve /models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --gpus 1
```
Multi-GPU: `--gpus 2` (layer-split; add `IE_QWEN3MOE_TP=1` for tensor-parallel on
qwen3moe). Long context: `--ctx 32768`.

## 4. Use it
```bash
curl http://localhost:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello!"}]}'
```
Point **Hermes** or any OpenAI client at `http://localhost:11435/v1`.

---

### Without Docker (host build)
```bash
source scripts/env.sh          # sets up oneAPI 2026.0 + the B70 AOT target
cmake -S . -B build -G Ninja && cmake --build build -j
./build/src/ie pull llama8b
./build/src/ie serve <model.gguf> --gpus 1
```

### Notes
- **First request is slow, then fast.** Because the image ships SPIR-V, the Level-Zero
  runtime JIT-compiles each kernel on first use — the very first request pays a few
  seconds of compile time; every request after runs at full speed (e.g. Qwen3-4B on
  one B70: ~140 tok/s prefill, ~75 tok/s decode).
- The runtime image is slim (~1.3 GB): it copies only the ~11 oneAPI libs the engine
  and the Level-Zero adapter actually need, not the full 1.2 GB oneAPI runtime.
- `--gpus` picks single vs multi automatically when omitted (VRAM-aware).
- `ie-docker pull` uses `curl` — works for public GGUFs. For private/gated repos,
  pre-download to `./models` on the host (e.g. with `hf download` there).
