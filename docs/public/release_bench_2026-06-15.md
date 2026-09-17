# Mach X — release benchmark run (2026-06-15)

The deliberate, same-hour, order-controlled head-to-head measurement pass for the
launch writeup. Protocol per `CLAUDE.md`: `scripts/bench_showdown.sh` (one JIT
warmup discarded, 3 alternating rounds), ±40 tok/s heat-soak band, cross-round
drift >3% = thermal. Same B70 / same session / same GGUFs. llama.cpp contender =
SYCL master `fdc3db9b6` (`build-sycl`, **includes PR #23142** = the +70% MoE
prefill), the toughest current backend.

## Item 6 — Publishable findings (the honest write-up)

**The one headline, and it holds:** *"Fastest known inference for Qwen3.6-35B-A3B
on Intel Arc — beats llama.cpp's strongest backend (SYCL master, including its +70%
MoE-prefill PR #23142) on BOTH metrics, same-session: prefill +~8% (1098 vs 1013
tok/s, peak 1174), decode +~1–3% (83.1 vs 82.2 tok/s)."* This is the claim to lead
with. It survived llama's big MoE-prefill improvement, measured under the same
order-controlled, heat-soak-aware protocol on the same B70.

**Breadth (the second story): runs ~every popular open family** — 8 arch families
(Qwen3.6-MoE crown, Qwen3/Qwen2 dense, Qwen3.6-27B hybrid, Llama-3, Qwen3-Coder-30B
MoE, Qwen3-Next-80B, Gemma-4) covering Qwen / Llama-1·2·3 / Mistral·Codestral·Devstral
/ Gemma / Phi / Granite / DeepSeek-R1 / Yi·Nemotron·InternLM·Baichuan, **plus AWQ &
GPTQ import (formats llama.cpp can't load natively)** and **multi-GPU** (tensor-parallel
+ layer-split) for the 72B/80B.

**The honesty that makes the claim credible (do NOT overclaim):** the *speed* crown
is the **MoE crown only**. This run proves it both ways:
- **Generic dense models are BEHIND** llama SYCL on prefill — Codestral −7%, Mistral −14%
  — because the dense path isn't bespoke-optimized (no int-dot/oneDNN by default).
- **Gemma-4 is ~15× behind** (unoptimized MoE; correctness-first, fused-MoE perf never done).
- A couple non-crown arches DO win, but vs *different* llama backends: **27B 1.9× vs
  Vulkan**, **Qwen3-Next-80B decode 1.40× vs SYCL** (prior same-hour runs).

→ **Public framing:** *"the fastest engine for the Qwen3.6-35B-A3B crown on Intel Arc,
and a broad-compatibility runtime for everything else"* — never "fastest at everything."
Tagline holds: *"running Mach10 on dual B70s."* Next perf work (post-launch): the dense
int-dot/oneDNN path + Gemma-4 fused-MoE — both would move breadth from compatible to
competitive.

---

## Item 1 — Perf state: FROZEN (no last-minute tuning)
Deliberately not landing speculative kernel changes before freezing numbers — that
would risk the crown PPL gate and destabilize the very numbers being measured. The
extensive per-model perf campaign already shipped the wins, and they are the
**default** paths benchmarked here (crown SoA-repack + int-dot MoE; qwen3next
int-dot down default; qwen3moe GPU-router + generalized int-dot down default). No
known safe pending lever. New optimization is a post-launch effort.

## Item 2 — Crown headline: Qwen3.6-35B-A3B (the perf crown)

### Prefill pp512 (clean, cool start)
| round | engine | llama SYCL master |
|---:|---:|---:|
| 1 (coolest) | **1173.9** | 960.2 ± 18.8 |
| 2 | 1082.9 | 1011.1 ± 12.2 |
| 3 | 1112.3 | 1015.0 ± 17.8 |

Engine round-1 **1174** confirms the ~1144 headline capability. Both-heat-soaked
(rounds 2–3): engine ~1098 vs llama ~1013 → **+8.4%**; engine-peak vs llama-plateau
→ +15%. Engine cross-round drift ~8% (mild thermal as it heat-soaks); llama rises
960→1015 (cold-start warm-up to plateau). **Verdict: engine leads prefill ~+8% at
warm-plateau, vs the IMPROVED SYCL backend (post-#23142).**

### Decode tg128 (separate run, card cooled first)
| round | engine | llama SYCL master |
|---:|---:|---:|
| 1 | 80.97 | 76.38 ± 1.6 |
| 2 | 80.50 | 78.67 ± 2.1 |
| 3 (warmest) | **83.13** | 82.24 ± 0.4 |

Card cooled before this pass → numbers back in the documented 80–84 range (vs the
73 that back-to-back gave last session — cooldown confirmed as the fix). At full
warm (round 3): engine 83.1 vs llama 82.2 → **+1.1%**; avg → ~+3%. The improved SYCL
backend's MMVQ int-dot decode is strong, so decode is **parity-to-slight-lead**, not
a blowout — honest. **Crown verdict vs current llama SYCL master (incl. #23142):
prefill +~8%, decode +~1–3% — leads/ties BOTH metrics, same-session 2026-06-15.**

## Item 3 — Breadth models, same-hour vs llama SYCL master (pp512)

| model (arch) | engine pp512 | llama SYCL pp512 | verdict |
|---|---:|---:|---|
| **Codestral-22B** (llama dense) | ~324 | ~349 | engine **−7%** (behind) |
| **Mistral-Small-24B** (llama dense) | ~316 | ~368 | engine **−14%** (behind) |
| **Gemma-4-26B-A4B** (gemma4 MoE) | ~59 (T=478) | 922.6 (pp512) | engine **~15× behind** |

Both dense runs rock-stable (<1% cross-round drift, no thermal). **Honest, important
finding:** on **generic dense Q4_K_M models the engine is BEHIND llama SYCL on
prefill** (~7–14%). The dense `kLlama3` path runs the in-house `gemm_fp16` — it is
NOT bespoke-optimized like the crown's SoA-repack + int-dot MoE, and oneDNN is
crown/qwen35-only by default. This **confirms the claim discipline**: the speed
crown is the **MoE crown ONLY**; dense breadth is *runs-correctly compatibility*,
not a speed win. (A dense-path oneDNN/int-dot optimization is the obvious post-launch
lever — deliberately NOT done now per Item 1's freeze.)

**Gemma-4 (~59 vs 922, ~15× behind):** Gemma-4's MoE forward is correctness-first —
its fused-MoE perf was always a documented TODO that was never done — so it runs the
naive unfused MoE path and is far slower than llama's. It RUNS correctly on GPU
(first-mover — Gemma-4 is brand new), but it is NOT a speed-competitive arch yet.
(Caveat: ours measured at T=478 prefill, llama at pp512 — the ~15× gap dwarfs any
length effect.) The optimized non-crown arches tell the opposite story and are the
ones to cite for breadth speed: **27B (qwen35) 1.9× vs llama Vulkan, Qwen3-Next-80B
decode 1.40× vs llama SYCL** (both prior same-hour runs).

## Item 4 — Qwen2.5-72B same-hardware llama TP: BLOCKED
The imported 72B GGUF is **no longer on disk** (reclaimed) and **disk is 100% full
(1.9 G free)**, so it can't be re-created. Re-import is also the RAM-risky operation
that previously OOM-crashed VS Code and must NOT be retried autonomously. So no
fresh same-hw llama `-sm layer` number this pass. **Use the existing internal
result:** engine 72B tensor-parallel decode **10.4 tok/s = 1.44× vs our own
layer-split** (`ie-multi-gpu-run --tp`, 2×B70), PPL 8.97 — a runs-correctly /
runs-at-all breadth point, not a head-to-head. (P2P all-reduce HW-blocked on this
box → host-bounce all-reduce.) To get a fair llama-TP number later needs disk freed
+ a careful re-import.

## Item 5 — Contender set: DECISION
**Two contenders, both llama.cpp; no viable 3rd on this box.**
- **Primary (headline): llama.cpp SYCL master `fdc3db9b6`** — the strongest llama
  backend on Arc, and it INCLUDES PR #23142 (+70% MoE prefill). Beating this is the
  real claim. This is what every number here is measured against, same-session.
- **Secondary: llama.cpp Vulkan** — the engine's original anchor; it wins decisively
  there (121–188% per `benchmark_matrix_2026-06-09.md`), but SYCL is the tougher,
  more honest comparison so it leads.
- **3rd contender — investigated, NOT viable here:** (a) **Intel llm-scaler-vllm**
  (the production B70 vLLM stack) does **not** support the crown's hybrid-DeltaNet
  arch (its validated list has Qwen3-30B-A3B, not Qwen3.6-35B-A3B); (b) **OpenVINO
  GenAI** — DeltaNet support unconfirmed, not installed; both would need a large
  install on a **100%-full disk**. Recommendation: ship with the two llama.cpp
  backends (the credible, reproducible head-to-head). If a 3rd is wanted later, the
  cleanest is **llm-scaler-vllm on Qwen3-30B-A3B** as a separate "Intel production
  stack" data point on a *model it actually supports* — deferred (needs disk + install).

## Constraints this pass
- **Disk 100% full (1.9 G free):** benchmarks are read-only mmap so runs are safe,
  but NO downloads / imports / large writes (blocks item 4 + a 3rd contender).
- **Thermal:** measured with cooldowns between sustained passes; cross-round drift
  noted per table. Relative same-hour standings are robust to thermal; absolutes
  carry the heat-soak band.
</content>
