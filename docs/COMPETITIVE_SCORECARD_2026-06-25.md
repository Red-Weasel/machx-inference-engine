# Competitive Scorecard — `ie` on Intel Arc Pro B70

**Date:** 2026-06-25 · **Hardware:** Intel Arc Pro B70 (BMG-G31, 32 GB, 608 GB/s) ·
**Opponent:** llama.cpp (SYCL + Vulkan backends), same GGUF, same box.

This document separates **VERIFIED (measured this session, fair single-GPU, fresh
head-to-head)** from **PRIOR (measured earlier, pending re-verification)**. The
distinction is deliberate — see §5 for what is NOT yet a defensible claim.

---

## 1. TL;DR — what we win on

**Verified this session (single-B70, fair, fresh llama head-to-head):**

- **gpt-oss-20b (OpenAI MoE, MXFP4) — WIN BOTH AXES at every length** (2026-06-27, vs
  current llama-SYCL's *best* FA-on config, same 11.27 GiB GGUF, 1×B70): **prefill pp512
  1795 vs 927 = 1.94×, pp2048 4147 vs 927 = 4.47×, pp4096 3428 vs 896 = 3.83×; decode
  tg@512 58.3 vs 50.3 = 1.16×, @2048 57.4 vs 49.9 = 1.15×, @4096 55.6 vs 49.4 = 1.13×**
  (both flat across depth). ⚠ The prior internal "31.9 vs 22.35 / 668 vs 607" was vs a
  STALE llama — re-verified here. Seven levers (MXFP4 fused decode GEMV, split-K FA-2 decode
  + sink, wide-tile prefill + sink, W8A8 lm_head, **batched prefill projections + router** =
  the big prefill flip from per-token-gemv). Write-up → `docs/public/gptoss_benchmark_2026-06-27.md`.
- **Gemma-4 26B-A4B prefill — we beat llama at EVERY length**, 1.58–2.03× vs
  llama's *fastest* backend (SYCL), and 5.4× vs Vulkan.
- **Qwen3-Coder-30B-A3B prefill — long-ctx win** (RE-VERIFIED 2026-06-27 vs current
  llama-SYCL fa1: pp4096 1248 vs 1005 = **1.24×** holds; but pp512 703 vs 1141 = **0.62×
  LOSS** and decode tg@512 50.5 vs 77.3 = **0.65× LOSS**). The interactive-prefill +
  decode picture WEAKENED — current llama is much faster on this mature MoE.
- **Qwen3.6-27B dense prefill — long-ctx win** (RE-VERIFIED 2026-06-27 vs current
  llama-SYCL fa1: pp512 332 vs 282 = **1.18×**, pp4096 448 vs 274 = **1.63×**, pp8192
  708 vs 258 = **2.74×**). Down from the prior 1.96–3.08× — **llama-SYCL improved ~10-20%
  on mature arches since 2026-06-25**, so the old multipliers are optimistic, but 27B
  still WINS prefill at every length and the win GROWS with ctx. (27B *decode* loses
  0.72×, confirmed — never claimed.)
- ⚠ **RE-VERIFICATION NOTE (publishing):** llama-SYCL has gotten faster, especially on
  newer arches (gpt-oss was the extreme case — see top entry). Re-bench EVERY claim
  vs a current llama before publishing. **gpt-oss-20b is the freshest + strongest claim
  (both axes, re-verified today); the dense/MoE prefill wins hold but with smaller
  multipliers; decode is generally llama's.**

**DeltaNet (RE-VERIFIED 2026-06-25 — see §4 for the honest corrections):**

- **Crown-35B prefill: WIN at BOTH short AND long ctx** — short-ctx 1.14× @~1K, and **long-ctx now 1.48×
  self / 1.16× vs llama** via the oneDNN large-M MoE lever (DEFAULT-ON 2026-06-26; flipped the prior long-ctx
  loss; decode untouched). The DeltaNet prefill win, now at every length.
- Crown decode: **parity** (78.1 vs 76.9); 80B: **behind** on both axes (decode 7%,
  prefill 2× from our chunk cap). The prior "1.40× 80B decode win" was vs an OLD
  llama and is now **dead** — current llama-SYCL caught up.
- **Structural moat (still real):** on gated-DeltaNet arches the competition can't
  run on B70 (Vulkan lacks shaders, OpenVINO preview-only) — llama-SYCL is the only
  single-stream opponent. Day-one-on-new-arches + prefill leadership is the headline.
- **Path to re-winning DeltaNet decode:** MTP self-spec decode (proven lossless
  1.2–1.46× on 27B/Gemma) ported to Crown/80B — a scoped project (§4).

---

## 2. Methodology (fairness)

- **Ours:** `ie-gemma4-gen <gguf> "<ids:…>" 1 bench` / `ie-bench --gguf … --prefill T`
  — warmup + 3-run median, prefill tok/s = T / median(prefill_ms).
- **llama-SYCL:** `ONEAPI_DEVICE_SELECTOR=level_zero:0 llama-bench -m <gguf> -p T -n 0 -ngl 99`.
- **llama-Vulkan:** `GGML_VK_VISIBLE_DEVICES=0 llama-bench … -sm none -mg 0` (single B70).
- **Single-GPU pinning is mandatory and verified** — the model (13.4 GB) fits one
  B70; an un-pinned llama run silently splits across both cards + the iGPU and
  invalidates the comparison. Both backends confirmed running on ONE B70.
- **Correctness:** greedy argmax bit-identical across our own kernel variants;
  coherence on real prompts; crown PPL 6.4527 invariant; adversarial kernel review.
- **Box state:** within-run order-controlled A/B (new-old-new) controls for thermal
  variance; absolute tok/s reported plainly.

---

## 3. VERIFIED — prefill wins (this session, single B70)

### Gemma-4 26B-A4B (Q4_0 QAT), prefill tok/s

| T | **ie (ours)** | llama-SYCL | llama-Vulkan | vs SYCL | vs Vulkan |
|---|---|---|---|---|---|
| 4K  | **1642** | 809 | 302 | **2.03×** | **5.4×** |
| 8K  | **1453** | 761 | (collapses) | **1.91×** | ≫ |
| 16K | **1016** | 641 | (collapses) | **1.58×** | ≫ |

- **SYCL is llama's *faster* backend here** (809 vs Vulkan 302 @4K), so this is the
  *harder* bar. Vulkan is fast at pp512 (797) but collapses past ~2K — the O(T²)
  signature of not windowing the SWA layers.
- **The decisive lever = sliding-window attention** for the SWA layers (window
  1024, ~5/6 of layers). It is both a perf lever *and* a correctness fix: full
  causal on SWA was an O(T²) approximation that diverged from the model's design.
  Coherence confirms it's sharper (windowed predicts the correct in-window token
  top-1 where full-causal echoes). Plus: oneDNN MoE GEMMs, a templated flash-attn
  "wide" tile kernel (hd256/512), register-staged KQ dot, and KV-tile occupancy
  tuning (SLM 82→49 KB → 1→2-3 workgroups/Xe-core).
- Session arc @16K: ~95 (naive) → **1016** (10.7× over the session's starting point).

### Qwen3-Coder-30B-A3B (Q4_K_M), prefill tok/s

| T | **ie (ours)** | llama-SYCL | result |
|---|---|---|---|
| 4K  | **1303** | 1016 | **WIN 1.28×** |
| 8K  | **919**  | 849  | **WIN 1.08×** (flipped from parity) |
| 16K | 547      | 624  | 0.88× (behind) |

- Levers: oneDNN MoE GEMMs (the big one), the faithful llama flash-attn tile inner
  loop, then the same occupancy + register-staged-dot levers spread from Gemma.
- 16K is still attention-bound and behind; the remaining gap is the next target.

### Quality / correctness gates (held throughout)
- Crown (Qwen3.6-35B) PPL **6.4527 bit-exact** across all changes.
- `ctest` **30/30**.
- Gemma windowed-attention output **coherent + correct** (adversarial mask review:
  no bugs; coherence: correct top-1 token recall within window).

---

## 4. DeltaNet moat — RE-VERIFIED 2026-06-25 (honest corrections)

The prior "both-axes DeltaNet decode win" was **partly STALE** — current llama-SYCL
has caught up on decode. Fresh same-box head-to-heads, llama **on-GPU verified**
(tg sane, not the ~0.5 CPU-fallback):

| Model | Axis | ours | llama-SYCL | result (re-verified) |
|---|---|---|---|---|
| **Crown 35B-A3B** (1×B70) | **prefill** | 1101 | 962 | **WIN 1.14×** ✅ |
| **Crown 35B-A3B** (1×B70) | decode | 78.1 | 76.9 | **PARITY** (was claimed 1.10×) |
| **Qwen3-Next-80B** (2×B70) | prefill | 311 | 625 | **LOSE 0.50×** (our DeltaNet T≤256 chunk cap) |
| **Qwen3-Next-80B** (2×B70) | decode | 56.6 | 60.8 | **LOSE 0.93×** (the "1.40×" was vs an OLD llama @37) |

**Honest takeaways:**
- The "1.40× 80B decode win" is **dead** — it was measured against a slower old
  llama (~37 tg); current llama-SYCL does 60.8 on-GPU. We're 7% behind.
- Crown decode is now **parity**, not a win — llama-SYCL improved.
- **Crown PREFILL now WINS at BOTH short AND long ctx.** Short-ctx 1.14× @ ~1K (unchanged). **Long-ctx
  FLIPPED to a WIN (2026-06-26): the oneDNN large-M MoE lever takes crown long-ctx prefill 727→1079 tok/s
  (1.48× self) and BEATS llama** (ours-real 1058 > llama-synthetic-best `llama-bench` pp8192 910 = 1.16×;
  the real gap is larger — dummy routing hides the MoE cost, and `llama-cli -f` CPU-stalls on crown's
  DeltaNet real-prefill). The hd256 tile lever (bf73093) fixed only the attention; the **MoE was the
  residual**, fixed by big-chunk + per-expert oneDNN (new `dequant_q4_moe_soa_to_Bt` keeps SoA → decode
  untouched). DEFAULT-ON for max_ctx≥8192 (opt-out `IE_QWEN36_NO_MOE_ONEDNN`); 6.4527 bit-exact; long-ctx
  PPL Δ+0.15%. Crown-only (oneDNN single-card; 80B 2-card-blocked). The earlier "MoE-GEMM-bound dead end"
  was a `T<2048` fused-dispatch cliff, not the kernel math.
- Decode is **bandwidth-bound** (profile: GEMVs ~70%, DeltaNet recurrence only
  3.4%) → both engines hit the same memory wall; we're at parity-to-7%-behind.
- **The lever to decisively win DeltaNet decode = MTP self-spec decode** (amortizes
  weight reads past the BW wall). ⚠ **RE-VERIFIED 2026-06-25: currently 0.81× on the
  27B** (spec 13.19 vs plain 16.33; LOSSLESS confirmed, mean-accept 2.63 — the loss
  is verify-`forward(T=K)` overhead). The prior "1.20× win" (14.34→17.14) does NOT
  reproduce — so MTP is **not even a net win in its current state on the one model
  with a head.** **THREE things must be fixed (not two):**
  0. **Fix the verify path** so spec actually beats plain (the batched-T int-dot
     verify the 1.20× relied on appears not engaged / regressed).
  1. **No MTP head in the GGUFs.** Verified via gguf reader: Crown (lmstudio
     Q4_K_M) and 80B (Momix-44 Q4_K_M) have **0 `nextn` tensors**. Only the 27B
     GGUF (blk.64) and Gemma's separate `mtp-*.gguf` carry a head. → must obtain/
     import a Qwen3-Next-80B or Crown GGUF that *includes* the NextN head.
  2. **Spec hooks are 27B-only.** `all_logits` / `hidden_pre_norm` / DeltaNet
     `ckpt` checkpoint exist only in `Qwen35DenseModel` (27B dense). Crown uses
     `QwenModel`, 80B uses `Qwen3NextModel` — both need the hooks added to their
     forward (the DeltaNet `ckpt` is the hard, correctness-critical part).
  Until (1) is cleared, the decode-MTP win is blocked. (2) is a scoped port.

- **DeltaNet 80B PREFILL gap (was 0.50×) — ✅ FLIPPED via oneDNN MoE, 2026-06-26.**
  The 80B runs on 2×B70 (46 GB). oneDNN (the MoE-GEMM lever that won Coder/Gemma/crown
  prefill) used to be single-card-only: a static ctx bound to card 0 → DEVICE_LOST on the
  multi-card path. **Step 1 (`bb82d44`):** `gemm_onednn.cpp:ctx_for` now keeps one engine
  **per SYCL device** (`ie-onednn-multidev-test`: both B70s run `gemm_fp16_onednn`
  interleaved, no DEVICE_LOST, exact). **Step 2 (`2219c58`/`e0c0d13`/`61a8e1b`/`d48798a`):**
  crown's recipe ported to `qwen3next.cpp` — DeltaNet recurrence sub-chunk ≤512 (`--sweep`
  CLEAN, §1-safe) + per-expert dequant Q4_K/Q6_K → fp16 Bt + `gemm_fp16_onednn`
  (multi-card-safe via the per-device map) + engine pf_chunk raise. The old "small-M dead
  end" was a chunk-cap artifact, not the kernel: a BIG engine chunk decouples the MoE
  (M = T·K/E = T·10/512) from the §1-capped recurrence (sub-chunked internally).
  **✅ CLEAN-BOX A/B (8K prompt, M≈160, new-old-new, 0.6% baseline variance):** today's
  default (chunk 512, int-dot) **480** → big chunk 8192 int-dot **601** → big chunk 8192
  **oneDNN 919/915** = **1.53× over int-dot at the same chunk, 1.91× over today's default.**
  M≈160 was NOT borderline. **DEFAULT-ON for max_ctx≥8192** (opt-out
  `IE_QWEN3NEXT_NO_MOE_ONEDNN` → 596; short-ctx unchanged; default path byte-identical).
  crown 6.4527 bit-exact, ctest 30/30. **Net: 80B long-ctx prefill ~915 tok/s @8K (was ~480).**
  ✅ **llama-SYCL H2H DONE** (`llama-bench` build-sycl fdc3db9b6, SAME GGUF, 2×B70): **pp8192 ours 915
  vs llama 627.3 = 1.46× WIN; pp16384 ours 722.7 vs llama 590.3 = 1.22× WIN.** The 80B's structural 0.50×
  prefill loss is now a prefill WIN over llama at both 8K (1.46×) and 16K (1.22×) — the DeltaNet moat holds
  the 80B on prefill too. (Win narrows at 16K: O(T²) full-attn grows and the MoE lever helps GEMM, not attn.)

**Why the moat still matters (structural):** on hybrid gated-DeltaNet arches the
competition is weak on B70 — Vulkan can't run them (missing shaders), OpenVINO is
preview-only, Intel llm-scaler is throughput-not-latency. llama.cpp SYCL is the only
single-stream opponent and it now ties/leads us on DeltaNet *decode* — so the honest
B70 headline is **"prefill leadership across arches + day-one on new arches,"** not
"fastest DeltaNet decode" (until MTP lands).

### Still pending re-verify
| (none — both items below resolved 2026-06-26) | | | |

### Gemma MTP RE-VERIFIED 2026-06-26 (clean box, own baseline, K=4)
| Gemma-4 31B dense MTP | decode | plain 17.95 → spec 26.55 = **1.48× LOSSLESS ✓** | ✅ SECURED (token-for-token) |
| Gemma-4 26B-A4B MTP | decode | plain 50.04 → spec 72.83 = 1.46× | ❌ **NOT lossless** — near-tie MoE routing; speed only, do NOT claim |

> 26B MTP losslessness regression confirmed (per-token router default-ON insufficient; disabling
> batched-verify doesn't help + is slower 0.77×). 31B MTP lossless win is the gemma decode headline.

### NEWLY VERIFIED 2026-06-26 (apples-to-apples, single B70 pinned, current llama-SYCL `fdc3db9b6`)
| Qwen3.6-27B dense (Q4_K_M) | prefill pp1024 | **572 vs 292.3 = WIN 1.96×** | ✅ same GGUF/box/build |
| Qwen3.6-27B dense (Q4_K_M) | prefill pp4096 | **577 vs 278.9 = WIN 2.07×** | ✅ |
| Qwen3.6-27B dense (Q4_K_M) | prefill pp8192 | **722 vs 234.8 = WIN 3.08×** | ✅ **long-ctx tile lever** |
| Qwen3.6-27B dense (Q4_K_M) | prefill pp16384 | **499 vs 233.9 = WIN 2.13×** | ✅ **long-ctx tile lever** (was 0.52× LOSS) |
| Qwen3.6-27B dense (Q4_K_M) | decode tg | 16 vs 21.6 = LOSE 0.74× | known commodity loss — never claim |

> **27B now WINS prefill at EVERY length (1.96–3.08×).** Two 2026-06-26 results: (1) the "1.21× — HIGH
> RISK, public llama ≈718 may erase it" caveat is **RESOLVED → REFUTED** (current llama-SYCL does 235–293
> flat on this box); (2) the **long-ctx prefill collapse is FIXED** — head_dim-256 full-attn layers routed
> through the Gemma wide-tile kernel (gated ctx≥6144) → 16K 115→499 tok/s, flipping a 0.52× LOSS to a
> 2.13× WIN. Coherence-verified. The same tile lever was ported to crown (bf73093, +57% @16K) and
> 80B/qwen3next (442af21, +12.5% @8.4K) — but attention was only HALF the story: crown/80B are
> **MoE-GEMM-bound** at long ctx. **CROWN long-ctx prefill was then FLIPPED TO A WIN (2026-06-26):** the
> MoE residual was attacked with a big engine prefill chunk (M≈256) + per-expert **oneDNN** GEMM (new
> `dequant_q4_moe_soa_to_Bt`, decode-safe) → 727→1079 tok/s (1.48× self, 1.16× vs llama), DEFAULT-ON. (80B
> stays a long-ctx loss — its MoE oneDNN is 2-card-blocked by the multi-card ctx landmine.)

---

## 5. Honest caveats — what is NOT (yet) a claim

- **No token-exact cross-engine output match yet.** Our Gemma output is validated by
  coherence + adversarial mask review + it being the model's designed (windowed)
  behavior — strong, but not a byte-match vs llama (llama-cli wouldn't complete the
  long-prompt run). Close this before publishing.
- **No Gemma PPL claim.** QAT GGUFs have meaningless absolute PPL (llama's own ref ≈
  1178). Quality = greedy coherence + crown invariant only.
- **Coder 16K, dense-27B decode** — we LOSE these; never claim them.
- **Self-improvement multipliers** (e.g. "16K 95→1016 = 10.7×") are vs our OWN
  baseline — present as engineering progress, never as a competitive ratio.
- **§4 DeltaNet numbers are now RE-VERIFIED (2026-06-25)** — Crown prefill 1.14× WIN
  holds; the decode "moat" did NOT (parity Crown / behind 80B). Don't quote the old
  1.40×/1.10× decode wins.

---

## 6. How we did it (pointer)

Every experiment — including the dead-ends — is in `tools/perf/PERF_LEDGER.md`
(per-lever tok/s, PPL, reasoning, and *why the negatives failed*). The decisive
methodology this session: **profile first** (Gemma 16K prefill was 96.3% attention →
pinned the target instantly), **re-test rejected levers on new shapes** (the
register-staged dot regressed at head_dim 128 but won at 256/512), and **occupancy
is a first-class lever** (workgroups/Xe-core vs the 128 KB SLM cap). See
`docs/BENCHMARK_METHODOLOGY.md` and `MASTER_DEV_PLAN.md §7`.

---

## 7. Pre-publish checklist

1. [x] Re-run the §4 moat numbers vs the current llama SYCL build, on-GPU verified.
       — Crown/80B done 2026-06-25; **27B dense prefill done 2026-06-26 (~2× WIN, refuted the ≈718 risk)**;
       **Gemma MTP done 2026-06-26 (31B lossless 1.48× SECURED; 26B confirmed NOT lossless — speed only)**.
2. [ ] One clean token-exact cross-engine output match on Gemma (long prompt).
3. [ ] Ledger this session's 9 prefill commits into `PERF_LEDGER.md`.
4. [ ] Confirm Coder/Gemma prefill wins reproduce on a cold box (sanity, not gate).
5. [ ] Decide framing: "fastest single-stream on B70 for new arches" + "wins gemma
       /coder prefill vs llama's best backend" — keep the moat and the prefill wins
       as two separate, individually-defensible claims.
