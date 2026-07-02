> # ⚠️ SUPERSEDED (2026-06-29) — RETAINED FOR HISTORY ONLY · DO NOT CITE NUMBERS FROM THIS FILE
>
> This 2026-06-21 publishing audit is **stale**. Its **#1 headline claim — "qwen3next-80B decode
> 1.40× vs llama SYCL" — is DEAD.** Re-verified 2026-06-25, the 80B decode is **0.93× (a LOSS)**:
> we are **~7% behind** current llama-SYCL (60.8 t/s on-GPU); the old "1.40×" was measured against a
> slower **old** llama (~37 t/s). Crown decode is now **parity**, not a win. Other claims here have
> also moved since (e.g. the Gemma-4 26B MTP result is **NOT lossless**; the 80B prefill *loss* has
> since **FLIPPED to a win** via the oneDNN MoE lever).
>
> **Current sources of truth (use these instead):**
> - Competitive numbers: [`docs/COMPETITIVE_SCORECARD_2026-06-25.md`](COMPETITIVE_SCORECARD_2026-06-25.md)
> - Every public number + publish status: [`docs/public/marketing/VERIFIED_CLAIMS_2026-06-29.md`](public/marketing/VERIFIED_CLAIMS_2026-06-29.md)
> - Authoritative state + roadmap: [`MASTER_DEV_PLAN.md`](../MASTER_DEV_PLAN.md)

# Publishing-Readiness Report — "ie" on Intel Arc Pro B70 (2026-06-21)

Read-only audit (no GPU runs). Stance: skeptical — prefer "needs re-run" over assuming an old number holds.
Authority for perf numbers = `docs/benchmark_matrix_2026-06-09.md` + memory; this doc is the publishing lens.

## TL;DR — the defensible headline
**"First & fastest single-stream decode on Intel Arc B70 for hybrid gated-DeltaNet arches
(Qwen3-Next-80B, Qwen3.6-35B) — where Vulkan can't run them and OpenVINO is preview-only — beating
llama.cpp SYCL."** PENDING P0 re-runs against the new llama build with **on-GPU execution verified**.
Drop/soften: any dense-27B decode win, ALL gemma PPL claims, gemma-12B entirely, any self-improvement
multiplier presented as a competitive ratio.

## Strongest → weakest publishable claims
1. ~~**qwen3next-80B decode 1.40× vs llama SYCL** (2×B70) — strongest; arch competitors barely run.~~
   **❌ DEAD (re-verified 2026-06-25): 80B decode is 0.93× — a LOSS (60.8 t/s current llama-SYCL; the
   1.40× was vs an old ~37 t/s llama).** See the SUPERSEDED banner + `COMPETITIVE_SCORECARD_2026-06-25.md` §4.
   (Original caveat, now moot: the DeltaNet SYCL kernel can silently CPU-fallback to ~0.48 tok/s.)
2. **crown-35B both axes vs llama SYCL master** (1×B70) — the sanctioned "total crown"; prefill margin
   (+7.6%) is the thinnest part. PMZFX public lists 35B SYCL 615 pp / 54.7 tg.
3. **gemma-4 31B dense both axes vs llama VULKAN** 1.81× pp / 1.33× tg (1×B70, Q4_0). SYCL is CRIPPLED on
   dense gemma4 (~18% BW) → Vulkan is the honest bar; do NOT quote the 6.2×/3.4× SYCL ratios.
4. **dense-27B prefill 1.21×** — ⚠ HIGH RISK: PMZFX public llama 27B-Q4 prefill = **718** vs the 287.7 our
   LOCAL stale SYCL posted → our prefill win may evaporate vs a newer/better llama. Decode we LOSE (15.2 vs
   23.2) — never claim a 27B decode win.

## Claim-risk audit (key points)
- **Stale comp build:** most comps used local llama `fdc3db9b6` (~2026-06-11). Intel llm-scaler shipped
  b8.3/b8.3.1 (May/Jun) improving Qwen3.5/3.6/Coder-Next; llama SYCL had active DeltaNet work. Re-run
  headline comps vs newest llama.
- **Our local SYCL may have under-performed** (27B prefill 287.7 local vs 718 public) → deflates our
  prefill wins, INFLATES our DeltaNet decode wins. Single biggest apples-to-apples risk.
- **Gemma PPL "beats llama" RETRACTED** — QAT GGUFs have meaningless PPL (llama's own ref = 1178). No PPL
  claim for gemma; quality = greedy coherence + crown 6.4527 invariant.
- **gemma-4 12B BROKEN** — exclude entirely; say "26B + 31B" only.
- **qwen3moe-30B loses both axes** (0.66×/0.64×) — not a beats-llama entry.
- **Self-improvement multipliers** (gemma "6.2× pp", qwen3moe "8.1×", MTP "1.20×") are vs our OWN baseline —
  never present as competitive.
- **GPU-count framing:** qwen3next/Coder-Next = 2×B70; crown/gemma = 1×B70. "Single-card viable" is a
  feature, not a perf win.

## Competitors on B70
- **Vulkan: CANNOT run any DeltaNet arch** (missing GATED_DELTA_NET shader → CPU fallback/garbage). Public
  GitHub issues confirm. ← our strongest "they can't run it".
- **OpenVINO: lists Qwen3NextForCausalLM but DeltaNet on B70 = PREVIEW-only** ("full production in a future
  release"); early-look ran on Core-Ultra iGPU (23.4 tg), NOT B70. No published B70 80B-DeltaNet number.
  OpenVINO crushes PREFILL (XMX) — don't fight it there.
- **Intel llm-scaler (vLLM, Battlematrix): FP16, concurrency-oriented.** Published Qwen3-8B 34.7 tg
  single-user → 255 @ 8-concurrent. NO published single-stream 80B-DeltaNet number. ⚠ FRAMING TRAP:
  their value is throughput-at-concurrency, ours is single-stream latency — fair head-to-head needs batch=1.
  "We beat Intel's own stack on 80B-DeltaNet" is currently INFERRED, not measured — soften.
- **llama.cpp SYCL** = the realistic single-stream opponent; runs DeltaNet on-GPU but fused kernel
  "not implemented/fallback unreliable" (issues #20423, #21888). SYCL ≈ 2.2× Vulkan.
- Public independent B70 baseline (PMZFX repo): 35B-A3B 615/54.7 (SYCL Q4) · 27B 718/20.4 (SYCL Q4) ·
  27B 785/15.1 (Q6) · Coder-Next-80B 305/43.4 (SYCL Q4, 2-card).

## Pre-publish checklist (re-measure when GPU free)
**P0:** (1) qwen3next-80B decode vs NEWEST llama SYCL, on-GPU verified (no 0.48 fallback) — the headline.
(2) dense-27B Q4_K_M pp+tg vs newest llama SYCL; reconcile local 287.7 vs public 718. (3) crown-35B both
axes vs newest SYCL master, same-hour.
**P1:** (4) gemma-31B vs newest llama Vulkan, confirm SYCL still crippled. (5) gemma-26B decode (tie is
fragile). (6) llm-scaler batch=1 on one shared model for a MEASURED single-stream data point.
**P2:** (7) own-evidence of Vulkan failing on a DeltaNet arch on this box. (8) confirm OpenVINO has no
prod B70 DeltaNet path at publish date.
**P3:** (9) state gemma PPL retraction in any writeup. (10) wire MTP spec-decode into Engine::generate
before claiming the 1.2× lever as a feature. (11) gemma now wired into ie::Engine ✅ (this session).
(12) re-confirm DeltaNet T≤256→512 lift on current driver; keep long-context out of the headline (T≥1024
unvalidated).

Sources: PMZFX intel-arc-pro-b70-benchmarks · intel/llm-scaler · Phoronix llm-scaler b8.2 · OpenVINO GenAI
supported-models + Qwen3-Next early-look · llama.cpp issues #20423 / #21888.

---

## ★ NEW (2026-06-21 PM) — Gemma-4 MTP self-speculative decode: the cleanest apples-to-apples claim

**This is now arguably the strongest publishable result** — and the rare one with NO comparability
caveats, because it pits OUR engine against llama.cpp on the **identical official artifact**: Google's
`gemma4-assistant` MTP draft head, the SAME head GGUF, the SAME target GGUF, the SAME 1×B70.

**Defensible headline:** *"Lossless MTP self-speculative decode on Intel Arc B70 — we run Google's own
gemma-4 MTP draft head FASTER than llama.cpp does, because our BW-optimal int-dot batched-verify amortizes
on B70 where llama's Vulkan GEMM cannot. Strictly token-lossless on BOTH the 31B dense and the 26B-A4B MoE."*

### Measured (1×B70, gemma-4 QAT-Q4_0 target + Q8_0 MTP head, this session)
| | base decode (ours) | **spec decode (ours, lossless)** | llama spec (same head/B70) | notes |
|---|---|---|---|---|
| **31B dense** | 16.95 tg | **24–27 tg** (1.47× prose / **1.62× code**) | **20.6 tg** (0.61 accept) | we beat llama's own MTP spec |
| **26B-A4B MoE** | 48.7 tg | **58–67 tg** (1.46× prose / 1.34× code) | not separately measured | strictly lossless via per-token router |

- **Lossless = strictly token-for-token == plain greedy.** 31B trivially; 26B MoE needed the per-token
  verify-router fix (`a78e292`) — the ONE non-bit-exact op vs decode was the GPU router's batched gemv.
- **pp (prefill) and PPL are UNCHANGED by spec decode** (it only touches generation). So no new PPL claim is
  created or needed here — the gemma PPL retraction still stands (QAT GGUF, meaningless PPL). Quality gate =
  greedy coherence (validated on prose + code) + crown 6.4527 bit-exact.
- **Why we win:** verify(T=K) on B70 is the bottleneck for spec decode; our `gemv_q4_0_soa_q8_batched`
  reads each weight column once and dots vs all K rows (decode-class BW), whereas llama's Vulkan GEMM can't
  amortize → llama only gets +8% from its head (the "3×" is datacenter-GPU marketing).

### Framing discipline (keep it honest)
- **Competitive claim = "ours 24–27 vs llama's 20.6 on the same head"** (a MEASURED head-to-head, ~1.2–1.3×).
  The "1.47×/1.62×" is vs OUR OWN base decode — present it as "lossless self-speedup," NOT a competitive ratio.
- **llama's 20.6 was measured this/prior session** (llama-server Vulkan, B70, 0.61 accept). RE-VERIFY against
  the newest llama build before publishing (P0) — same as every other comp.
- **Box was degraded at last measurement** (pp 234/585 vs clean 367/708). NET SPEC RATIOS are box-robust
  (both sides same session); ABSOLUTE tg needs a clean-box re-bench (P0).
- **26B MoE spec on code = 1.34×** (lower than prose) — report the range, don't cherry-pick the 1.62×.

### Checklist deltas vs the audit above
- **P3 item (10) DONE**: gemma MTP spec-decode is now wired into `ie run --spec` (`892934a`) — the product
  path, validated on both sizes (auto-finds the head). qwen35-27B was already wired. So the spec claim is a
  shipped feature, not just a tool.
- **NEW P0:** clean-box re-bench of {31B, 26B} base tg + spec tg, same session as a fresh llama-spec run.
- Commits: `6f4682a` (forward+loop+batched-verify), `cb8bfeb` (SoA-Q8 draft), `a78e292` (26B router→lossless).
  Handoff: `docs/HANDOFF_2026-06-21_gemma-mtp-spec-decode.md`.
