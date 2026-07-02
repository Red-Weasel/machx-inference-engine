# Benchmark Methodology — "ie" vs the field on Intel Arc Pro B70

The protocol for the publication's headline numbers. The whole point is **apples-to-apples
single-stream latency**: same model artifact, same machine state, same prompts, every engine
measured the identical way, the first (JIT-cold) run discarded. If a number can't be produced
this way, it doesn't go in the headline.

## 0. Clean-box prerequisite (NON-NEGOTIABLE — the bench-clean-box rule)
Stale measurements are the #1 risk (a heat-soaked / swap-thrashed box gave a FAKE 6× gemma
"regression"; degraded gemma prefill read 234/585 vs the true 367/708 — same code, just box
state). Before ANY headline run:
1. **Reboot.** Fresh kernel, no leaked VRAM, no fragmented RAM.
2. **Verify idle:** `free -g` swap ≈ 0 and ≥ ~20 GB free; GPU idle-clocked (~500 MHz / ~55 W)
   and no other process on the card. One GPU workload at a time across all repos/agents.
3. **One model in VRAM at a time.** Serialize loads (30 GB RAM ceiling). Never load a model
   > 0.7× target memory on CPU/1-GPU.
4. Re-check idle between engines (Vulkan/SYCL first-load can heat-soak; let it settle).
If two identical runs vary > ~1.5×, the box is dirty — stop and re-settle.

## 1. The prompt suite (fixed, 5 prompts, spans the workload space)
`benchmark_prompts/` — identical bytes for every engine/model:
| # | file | workload axis |
|---|---|---|
| 1 | `01_short_chat.txt`       | short conversational / explanation |
| 2 | `02_long_instruction.txt` | long structured instruction-following ("thinking") |
| 3 | `03_codegen.txt`          | code generation |
| 4 | `04_math_reasoning.txt`   | math / numeric reasoning |
| 5 | `05_long_context.txt`     | long-context read-then-plan |
Greedy (temperature 0) everywhere → deterministic token ids → run-to-run variance is pure
timing, not sampling. Fixed decode budget (default 128 new tokens) and fixed prompt bytes.

## 2. Per-(model × backend × config) procedure
- **3 runs, discard run 1** (JIT / first-load warm-up — applies to OUR JIT and to every other
  engine's first-kernel compile). Report **median** of runs 2–3 (use 3 runs → median if you
  want a tiebreak). Record min/max too (variance = box-health signal).
- Record, per prompt: **prefill tok/s (pp)** and **decode tok/s (tg)**. Aggregate = median
  across the 5 prompts (report per-prompt table + the median).
- **PPL** is measured SEPARATELY (not part of the speed suite) and ONLY where meaningful — see §5.
- **batch = 1, single stream.** This is the axis we compete on. Concurrency/throughput is a
  DIFFERENT benchmark (Intel llm-scaler's home turf) — if shown, label it separately and never
  mix single-stream tg with concurrent-throughput numbers.
- **Same quant / same GGUF** across engines for a given model row. If quants differ (e.g. our
  Q6 vs a public Q4), say so explicitly — a quant mismatch silently moves tg 20–50%.
- **Verify on-GPU execution.** The single biggest silent error: an engine CPU-falls-back for an
  unsupported op (llama SYCL can drop the gated-DeltaNet kernel → ~0.48 tg, which would INFLATE
  our ratio). Confirm GPU utilization / clocks during the run; reject any suspiciously-slow comp.

## 3. Engines & exact invocations
For each, log: build commit/version, backend, device, quant, and the raw per-prompt numbers.
- **ie (ours):** `ie-bench-suite --gguf <model> --decode 128 --runs 3 --warmup 1`
  (greedy, prefill+decode per prompt, median + kernel breakdown). Spec-decode models use the
  PRODUCT path `ie run --spec` (qwen35-27B and BOTH gemma4 sizes are wired; gemma auto-finds the
  `mtp-*.gguf` head next to the target, or pass `--spec-head`). Note the env fast-config flags per
  arch in the log (gemma-26B: `IE_GEMMA4_FUSED_MOE=1 IE_GEMMA4_MOE_SOA=1`).
- **llama.cpp SYCL:** `build-sycl` (icpx, `-DGGML_SYCL=ON`), `llama-bench -m <gguf> -p <pp> -n
  <tg> -ngl 99` OR `llama-server` + scripted prompts for the 5-suite. Pin the build commit.
- **llama.cpp Vulkan:** `build-vk` (`-DGGML_VULKAN=ON`). NOTE: Vulkan CANNOT run gated-DeltaNet
  arches (no shader → CPU fallback) — those rows are "N/A (no GPU path)", which is itself a
  result. Vulkan is the honest bar for dense gemma4 (SYCL is crippled there ~18% BW).
- **OpenVINO GenAI:** where it has a B70 path. DeltaNet-80B on B70 is preview-only at audit date
  (don't fabricate a number); OpenVINO is strong on PREFILL (XMX) — report it there honestly.

## 4. Models (must-haves) + the config each is benchmarked in
Show the meticulous 5-suite for AT LEAST these; others optional once the method is proven.
| model | our config (the intended one) | comp bar |
|---|---|---|
| **Qwen3.6-35B-A3B (crown, `qwen35moe`)** | 1×B70, Q4_K_M | llama SYCL |
| **Qwen3.6-27B dense (`qwen35`)** | 1×B70; **WITH `--spec`** (native blk.64 MTP head, lossless) | llama SYCL |
| **Qwen3-Coder-Next-80B (`qwen3next`)** | 2×B70, Q4 | llama SYCL (Vulkan = N/A) |
| **Qwen3-Coder-30B-A3B (`qwen3moe`)** | 1×B70, Q4 | llama SYCL |
| **gemma-4-31B dense (`gemma4`)** | 1×B70, QAT-Q4_0; **WITH the MTP assistant head (spec)** | llama Vulkan (+ llama's own MTP spec) |
| **gemma-4-26B-A4B MoE (`gemma4`)** | 1×B70, QAT-Q4_0; **WITH the MTP assistant head (spec)** | llama Vulkan (+ llama's own MTP spec) |

**Gemma is benchmarked WITH the assistant head, by design.** Google ships gemma-4 dense as a
"smart-but-clunky" model meant to be driven by its official `gemma4-assistant` MTP draft head —
nobody should run it without the head. So our gemma rows = `ie-gemma4-spec` (lossless spec
decode), and the fair comp is **llama.cpp running the SAME official head** (its `--spec-type
draft-mtp`), not llama plain decode. This is the cleanest apples-to-apples row in the whole
table (identical artifact). 26B needs `IE_GEMMA4_FUSED_MOE=1 IE_GEMMA4_MOE_SOA=1` + a chat
template; both gemma rows are strictly token-lossless.

## 5. PPL (perplexity) — when it's meaningful, when it's not
- **Spec decode does NOT change PPL** — it's token-for-token lossless, so PPL == the base
  model's. No separate spec PPL row is needed or honest.
- **QAT-Q4_0 gemma GGUFs have meaningless PPL** (~1000+ on wikitext; llama's own reference is
  ~1178 — QAT preserves argmax but miscalibrates logit magnitude). DO NOT publish a gemma PPL
  claim. Quality gate for gemma = greedy coherence (validated on prose + code) + the crown
  6.4527 invariant being bit-exact.
- **Where PPL IS meaningful** (non-QAT models, e.g. crown Q4_K_M, 27B Q4_K_M): measure with
  `llama-perplexity` (the reference, GPU) on wikitext-2 `-c 512` and match OUR streaming
  `--max-tokens N` to the SAME chunk[1] — NEVER compare our full-context streaming PPL to
  llama's windowed multi-chunk average (that protocol mismatch fabricated a fake 2.4× gap).
- The engine-internal quality gate that must hold for every change = `ie-perplexity` crown
  6.4527 bit-exact (a healthy non-QAT model).

## 6. Reporting / honesty rules (what goes in the write-up)
- Every cell = (engine, build, backend, device, quant, GPU-count, batch=1). No mixing.
- **Self-speedup ≠ competitive ratio.** "1.47×/1.62× spec" is vs OUR OWN base decode — present
  as "lossless self-speedup." The COMPETITIVE number for gemma = ours vs llama's SAME-head spec.
- Multi-GPU is labeled (qwen3next/Coder-Next = 2×B70; the rest = 1×B70). "Single-card viable"
  is a feature, not a speed win.
- Report ranges, not cherry-picks (e.g. gemma-26B spec = 1.34× code … 1.46× prose).
- State the gemma PPL retraction explicitly in any write-up.
- Drop/soften per the readiness audit: any dense-27B *decode* win (we lose base; spec narrows
  but verify on the day), all gemma PPL claims, gemma-12B entirely (broken), qwen3moe-30B if the
  clean re-bench confirms it loses both axes.

See `docs/PUBLISHING_READINESS_2026-06-21.md` for the ranked claims + risk audit, and
`docs/benchmark_matrix_2026-06-09.md` for the historical per-model ledger.

## HARD RULE — kernel map on every benchmark (added 2026-06-22)
A benchmark run is INCOMPLETE without the kernel-level map. For every model
benchmarked, ALSO collect the per-kernel breakdown (the tuning tracker):
  - DECODE map:  `~/r1 <model> kmap`        (ie-bench --kprofile-decode)
  - VERIFY map:  `~/r1 <model> kmap-verify` (spec models; --kprofile-verify 4)
This is how we know WHICH kernel owns the time (the tuning lever) per arch.
CAVEAT: IE_QUEUE_PROFILING inflates per-submit cost, so kmap is for RELATIVE
kernel ranking, NOT absolute wall time — confirm any candidate win with a real
ie-prompt-bench/spec run. Run kmap one-model-at-a-time like the speed rows
(it loads + frees one model). 80B 2-card kmap is not wired (single-card only).
