# Mach X — LLM Inference Engine for Intel Arc

**A C++/SYCL local LLM inference engine for Intel Arc GPUs, built and tuned on two Arc Pro B70 cards. XMX kernels, quantized models, and multi-GPU execution.**

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Language](https://img.shields.io/badge/C%2B%2B20-SYCL%20%2F%20DPC%2B%2B-orange)
![Platform](https://img.shields.io/badge/Intel%20Arc-Battlemage%20B70-0071C5?logo=intel&logoColor=white)
[![Model coverage](https://img.shields.io/badge/models-dense%20%2B%20MoE%20%2B%20hybrid-brightgreen)](#supported-architectures)
![Vision](https://img.shields.io/badge/vision-VLM%20ready-purple)
![Multi-GPU](https://img.shields.io/badge/multi--GPU-tensor--parallel-success)

Intel Arc is a genuinely capable AI GPU that inference tooling has mostly ignored. **Mach X is built for it from the metal up** — no fork of llama.cpp, no PyTorch, no vendor runtime. Hand-written SYCL kernels (XMX matrix engines, int-dot quantized GEMV, tiled FlashAttention), an OpenAI-compatible server, tensor-parallel multi-GPU, and day-one support for the newest model architectures — often running them fast on Arc *before* anyone else does.

---

## ⚡ What people run on it

The six models this engine is tuned for, each on **two Arc Pro B70 cards** (64 GB VRAM) with host RAM holding the
experts that do not fit. Dates, workloads and methods are in [Benchmarks](#benchmarks).

| model | weights | prefill | decode | also |
|---|---|---:|---:|---|
| **DeepSeek-V4.1-Flash** | 475 GB safetensors (FP8 dense, MXFP4 experts), 256 GB RAM | **304** tok/s at 2K, **319** at 32K–223K | **12.8** tok/s chat, **14.0** in agent loops | native vision, tool calls, 223K context, 1–2 s follow-up turns from the prompt cache |
| **MiMo-V2.6-Flash** | 178 GB safetensors (FP8 dense, MXFP4 experts), 256 GB RAM | **429–472** tok/s at 4–6K, **434–438** at 32K, **322** at 120K | **23.6** tok/s chat, **24.3** on agent-style copy edits | released and running the same day: its bundled DFlash drafter (speculative decoding), tool calls, thinking on/off, 120K context verified |
| **DeepSeek-V4-Flash** | 155 GB GGUF (MXFP4 experts, Q8_0 dense) | **571** tok/s at 4K | **26.1** tok/s at 4K, **32.7** short | tool calls, prompt cache |
| **GLM-5.3-Flash** | UD-Q4_K_XL GGUF, host-resident experts | **156** tok/s at 16K | **14.5** tok/s at 16K | MTP draft, two-GPU pipelined prefill |
| **Qwen3.8-Flash** (Flash-Next) | 104 GB UD-Q4_K_XL GGUF | **467** tok/s pipelined | **35.0** tok/s chat, **41.8** code (lossless speculative) | native vision |
| **Qwen3.8-27B** | Q8_0 GGUF | **945** tok/s at 2K | **24.5** tok/s (tensor-parallel + speculative) | prompt cache (layer-split) |

Everything runs behind one OpenAI-compatible server (`ie serve`) with tool calls, and the
[Dream Agent Harness](https://github.com/Red-Weasel/Dream-Agent-Harness) drives it as a local agent.

![DeepSeek-V4.1-Flash running locally in the Dream Agent Harness, served by Mach X on two Arc Pro B70 cards](docs/images/dream-deepseek-v41.png)
<sub>DeepSeek-V4.1-Flash on two Arc Pro B70 cards, served by `ie serve` and driven from Dream — reasoning shown, 11.7 tok/s.</sub>

---

## Highlights

- 🐋 **DeepSeek-V4.1-Flash on two B70s** — the 475 GB safetensors checkpoint with host-resident experts: 223K-token context verified, native tool calls and **image input** through `ie serve`, a prompt cache that answers follow-up turns in ~1–2 s, and prompt-lookup speculation that decodes agent tool loops **1.47× faster**. See [the V4.1 numbers](#deepseek-v41-flash).
- 🆕 **MiMo-V2.6-Flash on release day** — Xiaomi's 309B / 15B-active hybrid sliding-window model, released September 22 and running the same day from its safetensors: new XMX attention kernels for its 192/128 head sizes, FP8 dense weights kept FP8 on the card, its bundled DFlash drafter decoding **1.46×** faster, `ie serve` with tool calls, and Dream completing agent tasks on it. See [the MiMo numbers](#mimo-v26-flash).
- 🏛 **Dense, MoE and hybrid models** — GLM-5.3-Flash, DeepSeek-V4.1-Flash, MiMo-V2.6-Flash, DeepSeek-V4-Flash, Qwen3.8-Flash-Next, Qwen3.6, Qwen3 / Coder / Tongyi, Qwen3-Next, Gemma-4, gpt-oss, and Llama-compatible dense models. GLM-5.2 and Tencent Hy4-preview have experimental standalone runners. See [architecture coverage](#supported-architectures) for entry points and status.
- 👁 **Native vision** — DeepSeek-V4.1-Flash (the checkpoint's own vision tower, no extra files), Qwen3.8-Flash-Next and experimental DeepSeek-V4-Flash-Vision-Exp, including image inputs through the OpenAI-compatible API. DeepSeek-V4 vision requires its native vision sidecar weights.
- 🥇 **Beats llama.cpp on Arc** — on prefill *and* decode across the models below.
- 🧠 **Runs the big ones** — gpt-oss-**120b** (117B) and Qwen3-Next-**80B** on 2× B70 via tensor-parallel; **~2.5× faster than LM Studio** on 120b.
- 🔀 **Multi-GPU built in** — `ie serve --gpus 2` (tensor-parallel + layer-split), no P2P required.
- 🔌 **OpenAI-compatible server** + tool-calling (Harmony + Qwen) + **image inputs** (`image_url` data URIs) — point any OpenAI client (or [Hermes](https://github.com/NousResearch)) at `:11435`.
- 📦 **One-command Docker** — `docker pull` (or build) → `ie-docker serve` → running on your Arc GPU in minutes.
- ✅ **Correctness-first** — PPL-validated, per-layer cosine ≈ 1.0 vs a llama.cpp oracle, bit-exact where claimed.

---

## Supported architectures

Coverage below describes this source branch. A standalone runner does not imply
integration with `ie serve`; experimental ports have narrower qualification than
the benchmarked models. Prebuilt container images may lag these source updates.

| Family / GGUF architecture | Models | Entry point and coverage |
|---|---|---|
| **GLM-5.3-Flash** · `glm5next` | UD-Q4_K_XL GGUF | `ie serve` and `ie-glm5next-run`; sparse MLA + KDA, host-resident MoE, two-GPU pipelined prefill and MTP draft; kernel and full-model validation in [PERFORMANCE.md](PERFORMANCE.md) |
| **DeepSeek-V4.1** · safetensors directory (`deepseek_v41`) | Flash (FP8 dense, MXFP4 experts) | `ie serve <model dir>` and `ie-ds41-run`; two-card pipeline, CSA/engram/hyper-connections, three expert tiers (VRAM, pinned host RAM, NVMe), chunked long-context prefill, native DSML tool calls, native vision, prefix cache (memory + disk), prompt-lookup speculation; needs ~256 GB of system RAM |
| **MiMo-V2.6** · safetensors directory (`mimo_v2`) | Flash-RL (FP8 dense, MXFP4 experts) | `ie serve <model dir>` and `ie-mimo26-run`; two-card pipeline, 9 full-attention + 39 sliding-window layers with sinks (K 192 / V 128) on XMX prefill and split-K decode kernels, FP8-resident dense weights, three expert tiers (VRAM, pinned host RAM, NVMe) with a CPU expert path, the checkpoint's DFlash drafter for speculative decoding (on by default; prompt-lookup speculation when a checkpoint has none), XML tool calls, thinking on/off, live-conversation prefix reuse; text only; measured with 256 GB of system RAM. Pro not yet qualified |
| **DeepSeek-V4** · `deepseek4` | Flash (ggml-org MXFP4, Q8_0 dense), Flash-Vision-Exp | `ie serve`; streaming expert caches, long-context sparse attention, prompt caching and structured tool calls; experimental native vision requires sidecar weights |
| **Qwen3.8-Flash-Next** · `qwen4exp` | Qwen4 preview | `ie serve`; DeltaNet + sparse QSA, hyper-connections, PLE embeddings, streamed MoE and native vision |
| **Qwen3.5 / Qwen3.6 / Qwen3.8 hybrid** · `qwen35`, `qwen35moe` | 27B dense (incl. Qwen3.8-27B), 35B-A3B MoE | `ie serve`; gated-DeltaNet + full attention, dense or MoE feed-forward paths |
| **Qwen3 MoE** · `qwen3moe` | Coder-30B-A3B, Tongyi-30B | `ie serve`; QK-normalized attention and routed MoE |
| **Qwen3-Next** · `qwen3next` | 80B-A3B | `ie serve`; DeltaNet + full attention and 512-expert MoE |
| **gpt-oss** · `gpt-oss` | 20b, 120b (MXFP4) | `ie serve`; attention sinks, sliding-window attention, Harmony chat and tool calls |
| **Gemma-4** · `gemma4` | 31B dense, 26B-A4B MoE | `ie serve`; per-layer head geometry, sandwich norms, softcap and sliding-window attention |
| **Qwen dense** · `qwen2`, `qwen3` | Qwen2/2.5/3, compatible Qwen distills | `ie serve`; shared dense transformer path with architecture-specific attention handling |
| **Llama-compatible dense** · `llama`, `phi3`, `granite` | Llama-3.x, compatible Mistral, Phi and Granite GGUFs | `ie serve`; shared dense path; compatibility depends on GGUF architecture and tensor layout |
| **GLM-5.2** · `glm-dsa` | GLM-5.2 | Experimental `ie-glm52-run`; standalone MLA + MoE forward path |
| **Tencent Hy4-preview** · `hyv4` | Hy4-preview | Experimental `ie-hyv4-run`; standalone generation/PPL, two-GPU stage split and STQ1/IQ1 quantization support; no general server qualification claimed |

**Recognized without an inference runtime:** Inkling-Small (`inkling`) and
Laguna S 2.1 (`laguna`). Loader recognition is not runnable model support.

**Weight import:** `ie import` converts supported AWQ, GPTQ and EXL3 safetensors
to native GGUF. Import format support does not add an unsupported architecture.

---

## Latest Intel Arc kernel updates

**September 22, 2026 — MiMo-V2.6-Flash on release day.**

- **Attention kernels for MiMo's head sizes.** Its 9 full-attention layers use 192-wide keys and 128-wide values with
  per-head sinks, outside the engine's existing fast attention kernels. An XMX (joint_matrix) prefill kernel and a split-K decode kernel
  written for those sizes took a 32K prompt from 34.9 to 241 tok/s and 32K decode from 571–680 to 101–105 ms/token;
  with the router, sliding-window and chunking work after them, 32K prefill runs at 434–438 tok/s.
- **Serving defaults, each measured A-B-A.** A residency ranking from chat traffic (−16 % per decode token), a static
  expert tier sized from free VRAM (−11 %), FP8 dense weights kept FP8 on the card (−8 %, perplexity unchanged) and
  prompt-lookup speculation (−13 % on agent turns): 16.1 tok/s decode on held-out Dream prompts, 16.8 tok/s in a Dream
  agent loop. Wikitext-2 perplexity 3.4749 against llama.cpp's 3.4794 on the same checkpoint.
- **DFlash speculative decoding, from the checkpoint's own drafter.** MiMo ships a 5-layer block drafter that reads the
  model's hidden states and proposes 7 tokens at once; the model verifies them in one multi-row step. Drafts the
  drafter rates below 0.7 probability are not sent (an extra verify row costs ~27 ms). Held-out Dream prompts decode
  at **23.6 tok/s** (42.1–42.7 ms/token) against 16.2 plain and 18.1 with prompt lookup, the previous default; verbatim
  copies run as fast as lookup's. The drafter's drafts match a CPU fp32 reference on 1,736 of 1,736 tokens.
  [Measurements and limits](docs/mimo26/MIMO_V26_FLASH_2026-09-22.md).

**September 18, 2026 — DeepSeek-V4.1-Flash sees images, and agent loops decode 1.47× faster.**

- **Native vision from the checkpoint's own tower** (32-block ViT + aligner, 926 MB of weights already in the
  shards): images go in as OpenAI `image_url` parts, including a tool's screenshot. The tower's rows match DeepSeek's
  reference implementation at rel-L2 0.1–0.6 % (the model's own bf16 run is at 5 %), bit-identical run to run. The
  encoder borrows 1.5 GB of VRAM only while an image encodes (0.2 s for 640×480, 2.9 s at the 1,024-token budget),
  and a follow-up turn about the same image is served from the prompt cache without re-encoding it.
  [Tower](docs/deepseek41/94_PHASE55_VISION_TOWER_2026-09-18.md), [in the forward](docs/deepseek41/95_PHASE56_IMAGE_POSITIONS_IN_THE_FORWARD_2026-09-18.md),
  [serving and a 42-step agent soak](docs/deepseek41/96_PHASE57_IMAGES_IN_SERVE_AND_THE_AGENT_SOAK_2026-09-18.md).
- **Prompt-lookup speculation.** Agents spend much of their output copying text they have already seen (a todo list
  rewritten with one change, a file written back, a quoted string). When at least 12 context tokens match, the engine
  drafts the next 7 from that earlier occurrence and verifies all of them in one forward; prose is never drafted. The
  first A/B gave only 1.04×: an 8-row verify sent 17 GB of experts over PCIe, because the CPU expert path served
  one-row steps only. With that path extended to multi-row steps the verify drops 780 → 488 ms, and 42 recorded
  agent requests (18–34K context, replayed byte-identically, A-B-A) decode at **106 → 71.6 ms/token (9.4 → 14.0
  tok/s, 1.47×)**, outputs bit-identical to plain decoding with the CPU split off. On by default in `ie serve`
  (`IE_DS41_LOOKUP=0` turns it off). [Lookup speculation](docs/deepseek41/97_PHASE58_PROMPT_LOOKUP_SPECULATION_2026-09-18.md),
  [multi-row CPU expert path](docs/deepseek41/98_PHASE59_MULTI_ROW_CPU_EXPERT_LEG_2026-09-18.md).

**September 16–17, 2026 — DeepSeek-V4.1-Flash: faster long prompts, tool calls in the server, a prompt cache.**

- **Chunked prefill with the two cards as pipeline stages.** Card 1 works on chunk *k* while card 0 works on chunk
  *k+1*, so both PCIe links stream expert weights at once (a 2,048-token chunk moves ~180 GB of expert bytes).
  Bit-identical to the serial loop. With a gathered continuation attention (each row reads only its ~640 live keys)
  and candidate-block indexer scoring: **32K prompt 165 → 319 tok/s, 24K real text 183 → 344 tok/s, 223K real text
  104 → 319 tok/s** (36 → 12 minutes, needle at 50 % depth still retrieved). A split top-k defect that aborted every
  decode step past 131K keys was found and fixed on the way. [Pipeline](docs/deepseek41/83_PHASE43_PIPELINED_CHUNK_PREFILL_CRITERIA_2026-09-16.md),
  [gathered attention](docs/deepseek41/84_PHASE44_GATHERED_CONT_PREFILL_ATTENTION_CRITERIA_2026-09-16.md),
  [candidate skip + 223K](docs/deepseek41/85_PHASE45_INDEXER_SCORE_CANDIDATE_SKIP_CRITERIA_2026-09-16.md).
- **V4.1 behind the OpenAI server with native tool calls** (the DSML format, checked against the checkpoint's own
  encoder: 654 checks, shipped examples byte for byte), and prompts longer than one 2,048-token forward admitted.
- **Prefix cache.** Small ring checkpoints at every chunk boundary (the latent caches are append-only), other
  conversations kept in host memory, and a client's system prompt + tools written to disk once. A Dream-style chat
  with 90 tool schemas (17.7K tokens): time to first token **68.5 s → 1.1–2.3 s** on later turns, **2.4 s** on the
  first turn of a new server process. Every restore is bit-identical to recomputing (21/21 checks).
  [In memory](docs/deepseek41/86_PHASE46_PREFIX_CACHE_2026-09-16.md), [on disk](docs/deepseek41/87_PHASE47_DISK_PREFIX_CACHE_2026-09-16.md).
- **A served-process thread leak fixed.** Every NVMe expert fill spawned a thread whose OpenMP team was never
  reclaimed: 94K threads and ~0.5 GB per reply until decode starved. Persistent readers keep a server flat
  (~150 threads). [Details](docs/deepseek41/88_PHASE48_READER_THREAD_LEAK_AND_VRAM_2026-09-16.md).
- **Chat decode.** An expert-residency ranking profiled on chat decode (held-out prompts **7.26 → 10.81 tok/s**,
  A-B-A confirmed), engram rows faulted in parallel (host prep 9.4 → 2.7 ms/token), and an NVMe expert file rebuilt
  for that ranking (decode −6 % ms/token, prefill +6 %). A decode token is now expert bytes on PCIe, NVMe and
  E-core DRAM plus ~17 ms of attention. [Ranking](docs/deepseek41/89_PHASE49_CHAT_DECODE_RANKING_2026-09-16.md),
  [decode terms](docs/deepseek41/90_PHASE50_51_CHAT_DECODE_TERMS_2026-09-16.md), [expert file](docs/deepseek41/91_PHASE52_CHAT_EXPERT_FILE_2026-09-17.md).
- **New V4.1 defaults (September 17).** 10 GiB more pinned expert memory (still leaving at least 30 GiB free at load),
  the XMX W4A16 prefill expert route, and an FP8 LM head quantized at load into the checkpoint's own FP8 layout
  (perplexity unchanged: −0.0007 nats paired, 99.46 % top-1 agreement; +4 % decode, 0.66 GB of VRAM returned to
  the expert cache). 2K-token prompts **291 → 304 tok/s**. Each has an off switch (`IE_DS41_PIN_EXTRA_GIB=0`,
  `IE_DS41_PREFILL_XMX=0`, `IE_DS41_HEAD_FP8=0`). [Measurements and gates](docs/deepseek41/93_PHASE54_PIN_XMX_FP8_HEAD_DEFAULTS_2026-09-17.md).

**September 11, 2026 — two-card memory root cause, and the standard DeepSeek-V4-Flash GGUF.**

- **Every two-card run was mirroring its VRAM into host RAM.** On this stack a device allocation made in a
  SYCL context that contains both B70s (including the platform default context that `sycl::queue(device)`
  binds to whenever both cards are visible) is made resident on the peer card, and the kernel driver keeps a
  system-memory copy: 20 GiB of `malloc_device` cost 20 GiB of `MemAvailable`. The allocator, the DeepSeek
  runtime and the fleet's pipeline-split paths now use single-device contexts. GLM-5.3-Flash at 16K context
  pins every stage-1 expert layer for the first time: decode **9.9 → 14.5 tok/s**, prefill **66–106 → 156 tok/s**,
  identical text; the Qwen3.8-27B split takes **1 GiB** of host RAM at load instead of 29 at the same speed;
  DeepSeek two-card runs leave **0 GiB unattributed**. Tensor-parallel fleets keep their shared context on
  purpose (per-device contexts drop 27B tensor-parallel prefill 503 → 280 tok/s).
  [Root cause, mechanism and the runs](docs/glm53/DECODE_HOST_WAITS_2026-09-10.md).
- **ggml-org's DeepSeek-V4-Flash-0731 MXFP4 GGUF (the non-abliterated model) loads and runs at full speed.**
  Its 661 dense tensors are Q8_0; five F32-consumed roles now bind from Q8_0, and Q8_0 dense projections take
  the tuned Q8-SoA / oneDNN route instead of the packed per-element kernels: prefill **66 → 571 tok/s** at
  4,096 tokens, decode **18.9 → 26.1 tok/s** at 4K context, batch perplexity 3.948 on the first 512 wikitext-2
  test tokens. [Dtype map, change and measurements](docs/deepseek4/80_GGML_ORG_Q8_DENSE_BIND_2026-09-11.md).
- **Measured and closed:** GLM's expert CPU/GPU split ratio is flat within noise once every bank is pinned
  (pipelined MTP draft = 1.15× over serial); removing four DeepSeek per-layer host syncs was correct on every
  NLL dump but did not move decode, which is bound by expert-miss DMA at 4K (patch archived);
  DeepSeek expert prefetch is a dead end on the two-card runtime; Flash-Next's opt-in P2P context costs 60 GiB
  of host RAM for ≤ 2% decode. [Host-sync inventory and profile](docs/deepseek4/79_DECODE_HOST_SYNC_PLAN_2026-09-10.md),
  [prefetch verdict and the parked CPU/PCIe miss-split plan](docs/deepseek4/81_DECODE_PREFETCH_VERDICT_AND_CPU_SPLIT_PLAN_2026-09-11.md).

GLM-5.3-Flash now uses faster expert top-k selection, B70 KDA recurrence and
shape-specialized attention indexing. In-model kernel buckets improved by
**1.61×, 1.33× and 1.24×**, respectively, in the measured 24K profiling run.
Shared Q8 projections, FP16 XMX tiles, MLA tile reuse and convolution state
handling also received performance or correctness improvements.

These are kernel gains. Repeated full-model comparisons did **not** establish
a combined throughput improvement; slower runs also showed more expert staging
and host I/O. See [performance and validation](PERFORMANCE.md) for the measured
results, rejected experiments, test coverage and reproduction commands.

---

## Benchmarks

📊 **[Interactive charts →](https://red-weasel.github.io/machx-inference-engine/benchmarks.html)** · all measured on **Arc Pro B70** hardware; gpt-oss rows are clean-box head-to-head with identical GGUFs.

**gpt-oss-20b, head-to-head vs llama.cpp** — same GGUF, same GPU (1× Arc Pro B70), llama.cpp on its *fastest* config (FlashAttention on):

| context | **Mach X** prefill | llama.cpp | speedup | **Mach X** decode | llama.cpp | speedup |
|---|---|---|---|---|---|---|
| 512  | **1795** t/s | 927 | **1.94×** | **58.3** t/s | 50.3 | **1.16×** |
| 2K   | **4147** t/s | 927 | **4.47×** | **57.4** t/s | 49.9 | **1.15×** |
| 4K   | **3428** t/s | 896 | **3.83×** | **55.6** t/s | 49.4 | **1.13×** |

Wins both axes at every context length, and stays flat as context grows. Clean-box, reproducible (`ie-bench` vs `llama-bench`).

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

**DeepSeek V4 Flash Vision-Exp Abliterated** (MXFP4 experts, F16/F32 remaining weights, 164.70 GB GGUF), two B70 cards with expert tensor parallelism. Measured September 10, 2026 after seven further kernel passes:

| prompt/context tokens | prefill tok/s | decode tok/s |
|---|---:|---:|
| 128 | **143.3** | **33.3** |
| 512 | **288.5** | — |
| 4,096 | **574.0** | **27.9** |
| 16,384 | **546.6** | — |

Three fresh unprofiled runs per variant in A/B/B/A/A/B order; single stream, synthetic prompts, 128 generated tokens, 2,048-token prefill chunks and eight decode warmup tokens. Loading is excluded. No speculative decoding or clean reboot. These are text-only measurements for the Vision-Exp derivative.

The 4K and 16K prefill medians changed by **+1.59%** and **+1.32%** against the previous seven-pass candidate. All **511 per-token likelihoods matched exactly**, with perplexity **5.4721**.

Decode medians changed by -0.41% at 128 context and -0.17% at 4K, with overlapping run ranges.

Five passes were retained: attention Q preparation fusion, MoE gather/scatter index reuse, RMS input reuse and in-place RoPE pair dispatch. RoPE also rejects dimensions that previously caused an out-of-bounds write. Q8 quantization grouping and Q8-to-half vector loads were slower and were rejected; both gained regression tests. All 20 affected CTest entries passed on each card. Coverage is still limited to two B70 cards and one compiler/driver combination.

[New run ranges, profiles, hashes and reproduction](bench/deepseek4/2026-09-09/SEVEN_MORE_PASSES.md). [Previous seven-pass results](bench/deepseek4/2026-09-09/SEVEN_PASSES.md).

<a id="deepseek-v41-flash"></a>**DeepSeek-V4.1-Flash** (safetensors: FP8 dense, MXFP4 experts, 475 GB), two B70 cards with 256 GB of
system RAM (≈197 GB pinned expert arena, at least 30 GiB left free), measured September 16–17, 2026, single runs:

| workload | prefill tok/s | decode tok/s |
|---|---:|---:|
| 2,048-token prompt, one forward | **304** | — |
| 32,768 tokens, synthetic repeated text, 2,048-token chunks | **319** | **14.9** (64 tokens) |
| 24,193 tokens, held-out real document | **344** | 10.4 (120 tokens) |
| 223,237 tokens, held-out real document, needle at 50 % retrieved | **319** (700 s) | 4.7 (300 tokens) |
| Dream-style chat: 17.7K-token system prompt + 90 tool schemas, `ie serve`, 6 held-out prompts | first turn **281**; follow-ups served from the prefix cache (1.1–2.3 s to first token) | **12.8** (range 10.5–15.5) |
| agent loop: 42 recorded tool-calling requests with screenshots, 18–34K context, `ie serve` (September 18) | follow-ups from the prefix cache | **14.0** with prompt-lookup speculation (9.4 without) |
| image input: 640×480 / 1920×1080 screenshot (206 / 968 image tokens) | vision encode **0.2 s / 2.9 s** | — |

Perplexity 1.887 on 16,384 wikitext-2 test tokens (exact path). Decode at long context is bound by expert
residency (VRAM and host RAM against the checkpoint), not kernels. Design and measurement notes:
[docs/deepseek41](docs/deepseek41).

<a id="mimo-v26-flash"></a>**MiMo-V2.6-Flash** (safetensors: FP8 dense, MXFP4 experts, 178 GB), two B70 cards with 256 GB of system
RAM (115 GB pinned expert tier; 75 / 70 experts per layer in VRAM), measured September 22, 2026, single runs:

| workload | prefill tok/s | decode tok/s |
|---|---:|---:|
| 10 held-out Dream prompts (7 of 4.1–6.0K tokens), 128 greedy tokens each, with the bundled DFlash drafter (the default) | **429–472** | **23.6** (42.1–42.7 ms/token) |
| the same prompts: plain decoding / prompt lookup (the previous default) | — | 16.2 / 18.1 |
| 3 copy-heavy prompts (repeat a passage, re-emit a file or JSON with a rename), 512 tokens, DFlash | — | **24.3** |
| 32K-token haystack, needles at 10 / 50 / 90 % retrieved 3/3 | **434–438** | — |
| ~120K-token haystack, needle retrieved (ctx 131,072) | **322–324** | 12.3–13.6 |
| Dream agent loop through `ie serve` (write code and tests, run, fix; follow-ups from the live caches), before the drafter | — | **16.8** (14.7–18.9) |

Wikitext-2 perplexity **3.4749** (8 × 2,048 tokens) against 3.4794 ± 0.077 for upstream llama.cpp on a BF16 conversion
of the same checkpoint, and 3.7557 against 3.7526 at 8,192 tokens. Text only; one request at a time.
[Measurements, method and limits](docs/mimo26/MIMO_V26_FLASH_2026-09-22.md).

**DeepSeek-V4-Flash-0731** (ggml-org MXFP4 GGUF: MXFP4 experts, Q8_0 dense, 155 GB, the non-abliterated model), two B70 cards with expert tensor parallelism, measured September 11, 2026 on the same lines as above:

| prompt/context tokens | prefill tok/s | decode tok/s |
|---|---:|---:|
| 128 | **141.6** | **32.7** |
| 512 | **319.5** | — |
| 4,096 | **571.1** | **26.1** |
| 14,612 (serve, 64 greedy tokens) | **463** | **25.3** (30.5 warm cache) |

Before the September 11 loader and dense-route changes the same file measured 73.9 / 92.9 / 66.4 tok/s prefill and 21.8 / 18.9 tok/s decode. Batch perplexity **3.9170** on the packed route and **3.9476** on the shipped route over the first 512 wikitext-2 test tokens (stream-mode 3.9514), run-to-run byte-identical; the 64-token greedy continuation is identical between routes. No external reference for this exact file exists on the test box, so these are consistency figures, not an oracle comparison. [Details](docs/deepseek4/80_GGML_ORG_Q8_DENSE_BIND_2026-09-11.md).

Earlier measurements remain available: [September 3 binary benchmark](bench/deepseek4/2026-09-09/REPORT.md),
[eight-token expert optimization](bench/deepseek4/2026-09-09/OPTIMIZATION.md), and
[indexer optimization](bench/deepseek4/2026-09-09/INDEXER_OPTIMIZATION.md).

**Qwen/GLM kernel update, September 10:** in-place Qwen RoPE measures **2.25–3.04×** at the tested larger prefill shapes; GLM KDA gate **1.09–1.25×** at selected small chunks; half-rounded convolution-to-float fusion **1.04–1.38×** at T128–1024/C8192. Empty convolution calls now preserve dependencies. All eight affected regression tests pass on both B70s.

Paired Qwen 27B continuations, Flash-Next's 75MiB layer-output capture and all 490 GLM likelihood values match exactly. These are shape-specific kernel gains; whole-model timings remain mixed. [Changes, rejected RMS variants, model checks, existing GLM repeat-request variation and reproduction](bench/qwen-glm/2026-09-10/REPORT.md).

**Seven additional kernel passes, September 10:** retained bounded improvements to DeltaNet L2 normalization, fused QKV preparation, float/half gate preparation, QKV conversion, Q/gate splitting, head repetition and GLM KDA normalization. Final production-linked measurements range from **1.08× to 6.93×** across the retained shapes. All **11 affected regression tests pass on both B70s**, including exhaustive half-value checks on the vector conversion path. Paired Qwen/GLM outputs remain exact. Warm model gains are modest; cold GLM throughput declined, and a Qwen first-request delay did not recur on repeat. [Dispatch limits, timing distributions, rejected variants and model validation](bench/qwen-glm/2026-09-10/SEVEN_MORE_PASSES.md).

**Qwen3.8-27B** (Q8_0) — 2× B70, recorded August 15–26, 2026:
| workload | figure |
|---|---|
| short-context decode, layer-split + speculative | **22.0 tok/s** |
| short-context decode, tensor-parallel + speculative | **24.5 tok/s** |
| decode at 18.7K prompt depth, layer-split | **12.9–13.0 tok/s** (plain / speculative) |
| pipelined prefill, 2K / 4K / 9K prompts | **945 / 851 / 731 tok/s** |

The decode rows use the August 26 serving matrix; the prefill rows come from
the August 15 campaign with different prompts. Layer-split retains prompt caching.
See the [dated measurements and source excerpts](bench/deepseek4/2026-09-09/REPORT.md#archived-qwen-measurements).

**Qwen3.8-Flash-Next** (UD-Q4_K_XL, 104 GB, host-resident experts) — 2× B70, recorded August 28, 2026:
| axis | figure |
|---|---|
| speculative decode, K=3 | **35.0 tok/s chat; 41.8 tok/s code** (lossless-greedy) |
| warm pipelined prefill, 4×1024 tokens | **467–468 tok/s** |
| warm serving prefill | **~290 tok/s** |
| decode during sampled agent runs | **~17–22 tok/s** |

These are archived campaign results, not a fresh run of the current build.
Standalone benchmarks and sampled agent workloads use different settings;
the agent runs did not complete their task. See the
[benchmark close-out and agent-run observations](bench/deepseek4/2026-09-09/REPORT.md#archived-qwen-measurements).

Vision demo: a 1236×1343 terminal screenshot is read at 990 vision tokens with OCR-level detail ("VS Code terminal… session capture… segmentation fault…"), ~15 tok/s decode with the image in context.

> Methodology varies by row: the dated DeepSeek and Qwen campaigns describe their workloads and limits above. Historical `ie-bench --prefill P --decode N` rows mirror `llama-bench -pP -nN`. Some non-gpt-oss figures predate the latest clean-box sweep; the gpt-oss head-to-heads are ledger-verified.

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

**DeepSeek-V4.1-Flash server** (2× B70, ~256 GB system RAM, the checkpoint directory):
```bash
./build/src/ie serve /path/to/DeepSeek-V4.1-Flash --ctx 75000
```
The directory needs `engram_tables.json` + `engram_token_map.i32` beside the weights (generate once with
`tools/ds41_reference/engram_tables.py`, which calls the checkpoint's own `inference/engram.py`). Optional beside
it: an expert ranking (`ie_ranking_decode_chat.txt` — the chat ranking is in
[results/ds41-chat-ranking-2026-09-16](results/ds41-chat-ranking-2026-09-16)) and an NVMe expert file written by
`ie-ds41-expert-file`. Prefix entries are cached under `~/.cache/machx-ie` (`IE_DS41_PROMPT_CACHE_DIR=0` turns
that off); `IE_DS41_PROFILE_OUT=<file>` profiles your own traffic into a ranking.

**Use it from an agent desktop:** [Dream Agent Harness](https://github.com/Red-Weasel/Dream-Agent-Harness) launches
and manages `ie serve` (model picker, GPUs, context, tuning) and talks to it over the OpenAI API, including tool calls.
Dream finds a checkout at `~/machx-inference-engine` on its own; set `DREAM_MACHX_DIR=/path/to/machx-inference-engine` for
any other location.

**GLM-5.3-Flash server and Dream controls**:
```bash
./build/src/ie capabilities <model-00001-of-00006.gguf>
./scripts/ie-run-guarded --mem 220G ./build/src/ie serve \
  <model-00001-of-00006.gguf> --gpus 2 --ctx 200000 --threads 8
```
The server streams expert weights from host RAM and budgets resident weights,
context workspaces and expert caches per physical GPU. Dream now offers
**Model → GPUs → Context → Model tuning**, with validated sampling, stop,
threading and overflow controls. See [server controls](SERVER_CONTROLS.md) for
supported options, the observed GLM qualification and remaining limits.

**GLM-5.3-Flash standalone runner** (two B70 cards, host-resident expert banks):
```bash
./scripts/ie-run-guarded --mem 220G ./build/tools/ie-glm5next-run \
  <model-00001-of-00006.gguf> --gpus 2 --ctx 32768 \
  --prompt "Your prompt" --ngen 128
```
Prefill uses chunks of 1024 tokens and overlaps the two stages automatically,
including long `--prompt` input. Generation of 64 or more tokens automatically
uses MTP pipedraft when the model has an MTP head. Short generation, PPL scoring,
prefill benchmarks, plain-decode profiling/logit dumps and expert-parallel experiments do
not automatically load the draft head. Use `--no-pipeline --no-pipedraft` for
serial comparisons, or `--pipedraft` to request drafting explicitly. Drafting
uses extra model memory; its benefit depends on draft acceptance. These options
apply to the standalone runner, not the server.
Exact continuation comparisons use `IE_G5_CPU_MISS=0`; normal q* CPU/GPU expert
routing can produce different continuations between the two schedules.
Warm prefill batches that exceed the expert cache use waves of at most 16
experts, leaving room to retain weights needed later in the chunk. Cold
batches, batches that fit, and decode keep their existing wave width. This
adds no GPU workspace; `IE_G5_PP_WAVE=0` restores the previous half-cache
wave schedule for comparisons.
For prefill kernel timings, set `IE_QUEUE_PROFILING=1` and use
`--ppl <corpus> --ppbench <chunks>`; the runner reports instrumented kernel
time by bucket and device. Device windows cover the whole profile, including
gaps between that device's chunks.

Shared Q8 projections use shape-aware decode workgroups and combine aligned
prefill batches of at least 128 tokens into one grid. Small FP16 XMX GEMMs
use bounded 64-row tiles; larger or unsupported shapes retain the original
128-row path. These dispatches also apply to other models using the same
operators. GLM MLA projections load ahead without changing their accumulation
order, and sparse attention reuses latent tiles across heads. See
[performance and validation](PERFORMANCE.md) for the measured results and limits.

---

## Under the hood

- **Quantized GEMV** — W4A8/W6A8/W8A8 int-dot kernels (dp4a) over SoA-repacked weights: read each weight once, decode in-register. Q4_K, Q6_K, Q8_0, Q5_K, MXFP4.
- **FlashAttention** — register-tiled SIMD inner loop (no XMX for attention, following the fastest llama-SYCL path), plus split-K decode, sliding-window, and attention-sink variants.
- **MoE** — expert-batched weight-stationary prefill + fused gate/up/down; oneDNN XMX GEMM for the large-M regime.
- **Multi-GPU** — head-sharded attention + expert-sharded MoE (tensor-parallel) with host-bounced all-reduce; layer-split for pure capacity (bit-identical to single-GPU).
- **Speculative decode** — self-drafting NextN/MTP head with batched int-dot verify, lossless vs greedy.
- **Vision** — the model's own 449M SigLIP-style ViT ported natively: XMX GEMMs + custom LN / h-w rope / bidirectional packed-attention kernels, numpy-oracle-gated to 2e-6; embeddings splice into the LLM with true 3-stream interleaved M-RoPE (bit-identical to text rope when no image is present).
- **P2P pipeline** — 2-GPU layer-split with device-to-device wide-state push and double-banked chunk pipelining; every transport certified bit-identical.

Per-model design and measurement notes are in **[docs/](docs)**.

---

## License

**Apache License 2.0** — see [LICENSE](LICENSE). Copyright © 2026 Red-Weasel.

Free to use, modify, and ship (including commercially). Apache-2.0's patent grant + retaliation clause protects you and downstream users.

## Support

☕ **Buy me a coffee.** -- Unemployed and extremely grateful for any support -- If Mach X saves you time — or you want to see more fast local inference on Intel Arc —
donations are welcome, one-time or monthly. All donations support the project.

[![Buy me a coffee on Ko-fi](https://img.shields.io/badge/Buy%20me%20a%20coffee-Ko--fi-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/redweasel)

**[ko-fi.com/redweasel](https://ko-fi.com/redweasel)**

Requests and suggestions are welcome — [open an issue](https://github.com/Red-Weasel/machx-inference-engine/issues).
