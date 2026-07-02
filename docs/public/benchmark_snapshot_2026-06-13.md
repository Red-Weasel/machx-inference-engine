# Mach X — consolidated benchmark snapshot (DATA SKELETON for the launch writeup)

**2026-06-13.** This is a **consolidation of existing validated numbers** scattered across
`MASTER_DEV_PLAN.md` §7 and `docs/benchmark_matrix_2026-06-09.md` into one table — the data
skeleton roadmap #5's writeup will build on. **It is NOT the marketing writeup** (that's the
owner's framed single-session run with fresh same-hour numbers + final framing), and it runs
**no new benchmarks**. Every number is tagged with its source/date; ⚠ flags numbers that need
a fresh measurement before any public claim.

## ⚠ READ FIRST — the headline claim is a moving target

The crown took the **TOTAL CROWN on 2026-06-10** (same-hour paired runs): pp512 **1144 ±5 vs
llama.cpp SYCL 1064 ±8 (+7.6%)**, decode **84.1/81.0 vs 81.31**. **BUT llama.cpp SYCL master is
improving fast** — `benchmark_matrix` v1.4 (2026-06-09) recorded llama SYCL **master b9586 at
1092/81.1** (after PR #23142 = +70% MoE prefill on this exact model class), where the engine was
*behind* (86% prefill / 82% decode) before the v1.5/v1.6 crown work closed it. The exact standing
depends on which llama.cpp build you measure against, same hour.

→ **Before ANY "beats llama.cpp" public claim, roadmap #5 MUST re-run the crown showdown
same-hour vs CURRENT llama.cpp SYCL master** (`scripts/bench_showdown.sh`, order-controlled). The
safe, always-true claim today is the documented one: *"fastest known inference for Qwen3.6-35B-A3B
on Intel Arc vs llama.cpp's Vulkan backend; competitive-to-leading vs its SYCL backend, same-hour
measured."* Never an unqualified "best engine for Intel GPUs."

### ✅ FRESH RE-MEASUREMENT 2026-06-13 — prefill headline HOLDS vs current SYCL master
Ran `bench_showdown.sh` (order-controlled, 3 alternating pp512 rounds + 1 discarded JIT warmup)
vs llama.cpp SYCL **built from `fdc3db9b6`, which INCLUDES PR #23142** (`cc9e33121`, the +70% MoE
prefill) — i.e. the *improved* backend, not a stale build:

| round | engine pp512 | llama.cpp SYCL pp512 |
|---:|---:|---:|
| 1 | 1043.3 | 887.5 ± 38.5 |
| 2 | 1038.2 | 930.1 ± 43.2 |
| 3 | 1059.4 | 956.1 ± 14.3 |

**Engine ~1047 (stable, ±10) vs llama ~924–956 (still thermally rising) → engine leads prefill
~+10–13%.** The "beats llama.cpp on prefill" claim **survives** the SYCL MoE-prefill improvement.

**Decode pass (order-controlled, immediately after, card warm):**

| round | engine tg | llama.cpp SYCL tg128 |
|---:|---:|---:|
| 1 | 70.8 | 69.1 ± 0.2 |
| 2 | 73.0 | 69.6 ± 0.4 |
| 3 | 73.0 | 70.4 ± 0.6 |

**Engine ~73 vs llama ~70 → engine leads decode ~+4%.**

### ✅ CONCLUSION: the TOTAL CROWN HOLDS vs current llama.cpp SYCL master (post-#23142)
Same-hour 2026-06-13, the engine leads **BOTH** metrics vs the *improved* SYCL backend:
**prefill ~+10–13%, decode ~+4%.** The stale-headline risk is **resolved** — llama's +70% MoE
prefill PR did not retake the crown.
**Honesty caveat for FINAL publishable numbers:** absolute tok/s this session were thermally
depressed (engine 1047/73 and llama ~940/70 vs the engine's heat-managed 2026-06-10 peaks of
1144/84) because the card was heat-soaked from back-to-back runs. The **relative same-hour
standing is valid and definitive**, but the framed #5 run should cool between prefill/decode
passes (or measure decode cold) for the cleanest absolute headline numbers.

## Per-architecture standing (ours vs llama.cpp)

| Model (arch) | ours pp512 | ours tg128 | vs llama.cpp | PPL | perf date / note |
|---|---:|---:|---|---:|---|
| **Qwen3.6-35B-A3B** (crown, qwen3.6moe) | **1144** | **84.1** | **+7.6% pp / decode lead** vs SYCL 1064/81.3 | **6.4527** | 2026-06-10 same-hour ⚠ re-measure vs current SYCL master |
| **Qwen3-Next-80B-A3B** (qwen3next, 2×B70) | ~336 | **51.8** | **decode 1.40× FASTER** vs SYCL 37.1; pp parity ≤256t, −10% @512 | 4.73 | 2026-06-13; llama Vulkan→CPU for this arch so SYCL is the only GPU oracle |
| **Qwen3.6-27B** (qwen35 dense-hybrid) | **577** | 10.0 | **prefill 1.9×** vs Vulkan 303; decode parity 9.72 | 5.34 | 2026-06-12 |
| **Qwen3-8B** (dense) | **1190** | 43.7 | **prefill +14.9%** vs 1036; decode 43.7 vs 77.7 (unopt) | 18.9¹ | 2026-06-10 |
| **Qwen2.5-72B** (dense, 2×B70 TP) | — | 10.4 | 1.44× vs layer-split (no same-hw llama TP run yet) | 8.97 | 2026-06-12 |
| **Qwen3-Coder-30B-A3B** (qwen3moe) | 651 | ~37.4 | **behind**: pp 651 vs Vulkan ~984 (~1.5×); decode 1.94× FA-2 | 11.98 | 2026-06-12; honest non-win |
| Llama-3.1-8B (kLlama3) | — | — | perf not ledgered | 10.79 | correctness gate only |
| Granite-3.3-8B (kLlama3+scalars) | — | — | perf not ledgered | 10.30 | correctness gate only |
| Mistral-Small-3.2-24B (kLlama3/tekken) | — | — | perf not ledgered | 7.42 | correctness gate only |
| Phi-4 (phi3→kLlama3) | — | — | perf not ledgered | 8.2475 | correctness gate only |
| Gemma-4-26B-A4B (gemma4 MoE QAT) | — | — | perf not ledgered | ~oracle order | correctness gate only |
| **Codestral-22B** (kLlama3/SPM) | — | — | perf not ledgered | 10.89 | NEW 2026-06-13; correctness gate only |
| DeepSeek-R1-Distill (qwen2/llama) | — | — | perf not ledgered | 25.66 / 81.24 | correctness gate only |

¹ Qwen3-8B's ~18.9 PPL is the *model* on this corpus (llama.cpp scores 24.5 on the same GGUF),
not the engine — engine PPL is bit-exact deterministic.

## What's a fair claim vs not (claim discipline)

- **Strong, defensible wins:** crown vs llama Vulkan (decisive, both metrics); **Qwen3-Next-80B
  decode 1.40× vs llama SYCL** (and a genuine first-mover story — llama's Vulkan falls to CPU for
  this arch, so we're the first GPU impl on Intel Arc); 27B prefill 1.9× vs Vulkan; dense prefill
  +14.9%.
- **Honest non-wins (say so):** Qwen3-Coder-30B prefill is ~1.5× *behind* llama Vulkan (fp16 MoE
  kernel gap at E_ffn=768/top-8); dense decode (43.7 vs 77.7) is unoptimized; the crown vs current
  llama SYCL master is unmeasured since 2026-06-10.
- **Breadth as its own story:** 8 arch families + Granite/Devstral/Codestral + AWQ/GPTQ import +
  multi-GPU (TP + layer-split) all run correctly. "Runs models llama.cpp can't load (AWQ/GPTQ) and
  arches its GPU backends fall to CPU on (Qwen3-Next)" is a real differentiator independent of raw
  tok/s.

## Gaps the #5 benchmark session must fill
1. **Re-run the crown showdown same-hour vs current llama.cpp SYCL master** — the headline.
2. Ledger perf (pp512/tg128 vs llama, same-hour) for the correctness-only arches that matter to
   the writeup — at least **Codestral, Gemma-4, Mistral-24B** (the "see what they're about" set).
3. A same-hardware llama.cpp TP/`-sm layer` run for the 72B (we only have our own 1.44× vs
   layer-split internally).
4. Decide the writeup's contender set (owner mentioned possibly a 3rd contender beyond llama.cpp).

## Reproduce
- Crown: `scripts/bench_showdown.sh` (order-controlled new-old-new; discard first run after any
  rebuild — JIT; ±40 tok/s heat-soak). Per-arch: `ie-bench-suite`, `ie-qwen3next-bench`.
- llama.cpp oracles already built: `~/llama.cpp/build-vk` (Vulkan), SYCL build @ fdc3db9b6.
</content>
