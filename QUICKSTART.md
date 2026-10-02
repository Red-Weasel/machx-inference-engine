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

> **Which quant to grab.** Best-tested: **`Q4_K_M`, `Q6_K`, `Q8_0`** (and **`MXFP4`** for gpt-oss). If a model fails to load, try one of those first — some quant × architecture combinations (e.g. certain `Q5_K` builds) aren't supported yet. The curated `ie pull` names already use known-good quants. In a v0.2.6 source build, the two-card Qwen3.8-27B split runs `Q6_K` and `Q5_K_M` natively (22.0 and 24.0 tok/s against 17.2 for `Q8_0`, one request, September 29, 2026).

## 3. Serve (OpenAI-compatible, port 11435)
```bash
./scripts/ie-docker serve /models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --gpus 1
```
Multi-GPU: `--gpus 2` (layer-split; add `IE_QWEN3MOE_TP=1` for tensor-parallel on
qwen3moe). Long context: `--ctx 32768`.

Several requests at once (source builds from v0.2.6; the prebuilt image predates it): without
`--parallel`, the load picks how many requests it serves at once and logs why (`[lanes] auto ...`).
That is 1 on one card and on most architectures, up to 16 on the two-card Qwen3.6-35B-A3B-class and
Qwen3.8-27B splits, and 4 on MiMo-V2.6-Flash, DeepSeek-V4.1-Flash and Qwen3.8-Flash. `--parallel N`
sets the count and `--parallel 1` is the one-request server (needed for images on MiMo-V2.6-Flash and
Qwen3.8-Flash). `curl localhost:11435/props` shows `total_slots` and `slot_ctx`, the positions of each
lane after the first.

## 4. Use it
```bash
curl http://localhost:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello!"}]}'
```
Point **Hermes** or any OpenAI client at `http://localhost:11435/v1`.

Stop it with `curl -X POST http://localhost:11435/admin/shutdown` (or SIGTERM / Ctrl-C) and let it exit by
itself: from v0.2.6 a stop was measured at 0.43–3.08 s on the request lanes of the Qwen3.6-35B-A3B class,
Qwen3.8-27B and Qwen3.8-Flash, and the log ends with `[ie] stopped in X.X s`. Do not kill a server that still
has work on a card.

---

### Without Docker (host build)
```bash
source scripts/env.sh          # sets up oneAPI 2026.x (JIT build; AOT is opt-in, see Notes)
cmake -S . -B build -G Ninja && cmake --build build -j
./build/src/ie pull llama8b
./build/src/ie serve <model.gguf> --gpus 1
```

### Notes
- **First request is slow, then fast.** Because the image ships SPIR-V, the Level-Zero
  runtime JIT-compiles each kernel on first use — the very first request pays a few
  seconds of compile time; every request after runs at full speed (e.g. Qwen3-4B on
  one B70: ~140 tok/s prefill, ~75 tok/s decode).
  The compiled kernels are cached on disk twice: by the driver
  (`~/.cache/neo_compiler_cache`, on by default) and by the SYCL runtime
  (`~/.cache/libsycl_cache`; the engine sets `SYCL_CACHE_PERSISTENT=1` itself unless
  you set it). Mount or persist `~/.cache` in a container to keep repeat loads free.
  An AOT build (`IE_SYCL_TARGET=intel_gpu_bmg_g31` at configure) removes the
  first-use compile entirely for the B70.
  After an upgrade, the first request compiles the kernels that changed, once: with v0.2.6's new kernels
  that was 4.87 s to the first token on a Qwen3.6-35B-A3B-class model and 6.95 s on Qwen3.8-Flash, then
  normal on every later start.
- The runtime image is slim (~1.3 GB): it copies only the ~11 oneAPI libs the engine
  and the Level-Zero adapter actually need, not the full 1.2 GB oneAPI runtime.
- `--gpus` picks single vs multi automatically when omitted (VRAM-aware).
- `ie-docker pull` uses `curl` — works for public GGUFs. For private/gated repos,
  pre-download to `./models` on the host (e.g. with `hf download` there).
