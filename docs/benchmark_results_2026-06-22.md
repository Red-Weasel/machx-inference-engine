# Benchmark results — 2026-06-22 (clean box, post oneAPI 2026.0 update)

Box: 2×B70 (single card unless noted), oneAPI 2026.0 / NEO 26.18, kernel 6.17.0-35.
Protocol: GPU verified idle (≤600 MHz, swap baseline) before every load; one model at
a time; discard cold/JIT first run (ours has `--warmup 4`, llama re-run to discard cold).
Correctness: crown PPL 5.4163 + 27B PPL 5.6022 **bit-identical to pre-update** → oneAPI
update changed no numerics.

## Apples-to-apples — synthetic pp512 / tg128 (THE head-to-head)

`ours` = `ie-bench --prefill 512 --decode 128`  ·  `llama` = `llama-bench -p 512 -n 128`
Same GGUF, same single card, same synthetic workload. llama uses its **fastest** backend
on B70 per model (SYCL for all; Vulkan is SLOWER for gemma4 — see note).

| Model | Arch | Ours pp512 | llama pp512 | pp verdict | Ours tg128 | llama tg128 | tg verdict | PPL (ours) |
|-------|------|-----------:|------------:|:----------:|-----------:|------------:|:----------:|-----------:|
| Crown 35B-A3B   | qwen35moe        | **1116.6** | 1023.3 | **win 1.09×** | **79.6** | 72.5 | **win 1.10×** | 5.4163 |
| 27B dense       | qwen35 (hybrid)  | **352.0**  | 287.2  | **win 1.23×** | 15.1 | **22.0** | lose 0.69× | 5.6022 |
| Coder 30B-A3B   | qwen3moe         | ~646       | ~1179  | lose 0.55× | ~49.7 | **~75.5** | lose 0.66× | 7.8113 |
| Gemma 31B dense | gemma4           | (sweep TBD)| 192    | — | 18.57 base | **23.7** | lose base 0.78× | — |
| Gemma 26B-A4B   | gemma4 MoE       | (sweep TBD)| 847    | — | 51.53 base | 54.1 | ~par 0.95× | — |

## Spec-decode (MTP self-speculative) — our differentiator

| Model | baseline tg | spec tg | net | lossless? | acceptance | vs llama best |
|-------|------------:|--------:|----:|:---------:|-----------:|:-------------:|
| Gemma 31B dense | 18.57 | **26.66** | 1.44× | **YES** (token-identical) | 2.745 | **win 1.12×** (vs 23.7) |
| Gemma 26B-A4B   | 51.53 | 75.46 | 1.46× | **NO** ⚠ | 2.953 | 1.39× but NOT lossless |

⚠ **gemma26 spec is NOT bit-lossless on this build** — 26B MoE near-tie expert routing
diverges between the batched-verify path and T==1 decode. Deterministic, not flaky.
Flagged to re-validate the per-token verify router (commit a78e292) post-oneAPI-update
BEFORE claiming a lossless win. 27B/gemma31 spec ARE lossless.

## Decode kernel maps (relative ranking; profiler inflates per-submit)

**Crown** (GPU tot 11.6 ms): gemv_q4k_q8 13.9% · gemv_q6k_huge_q8 11.3% (lm_head, 1 call)
· moe_dec_gate_q4k 10.5% · gemv_q6k_med 9.2% · fa2_partial 7.9%

**27B dense** (67.6 ms): **gemv_q4k 51.6%** · **gemv_q6_soa 27.2%** · gemv_q8_0_soa 5.6%
· fa2_partial 5.0%  → 79% in two GEMV kernels = the dense-decode lever (Q4 decode ~42%
BW vs llama 64%).

**Coder 30B** (15.7 ms): moe_pfl_gate_q8 17.6% · fa2_partial 15.5% · gemv_fp16 14.1%
· moe_pfl_down_q8_6k_gen 13.1% · moe_pfl_down_q8_4k_gen 12.0% · gemv_q4k 7.1%
→ MoE kernels ~44% + attention 18.5% + fp16 dense-attn proj 14%.

**Gemma31** (46.6 ms): **gemv_q4_0_soa_multi 50.5%** · **gemv_q4_0_soa 27.2%** ·
rms_norm 8.3% → 77.7% in two Q4_0 SoA GEMVs = BW-bound (at ceiling).

**Gemma26** (13.8 ms): moe_pfl_proj_q4_0_soa 36.9% · gemv_q4_0_soa 14.1% ·
gemv_q4_0_soa_multi 13.7% · rms_norm 12.9% · gemv_q6_soa 10.0% (lm_head).

## ⭐ Context-length sweep — the long-context picture (pp/tg vs T)

The single pp512 point HIDES this. Full curve, `ie-bench --prefill 512,1024,4096,16384`
vs `llama-bench -p 512,1024,4096,16384`:

**Crown 35B-A3B (qwen35moe — DeltaNet family):**

| T_pp | ours pp | llama pp | ours tg@depth | llama tg64 |
|-----:|--------:|---------:|--------------:|-----------:|
| 512   | **1123** | 1017 | 81.3 | 73.8 (depth0) |
| 1024  | **1160** | 1011 | 77.8 | — |
| 4096  | 293     | **962**  | 67.9 | — |
| 16384 | 169     | **841**  | 45.9 | — |

**KEY FINDING — long-context prefill collapses on DeltaNet arches.** We WIN short-context
prefill (≤1024) but LOSE 3–5× at long context: 16K prefill takes **97 s** (ours) vs ~19 s
(llama). Cause = the **T≤512 DeltaNet sequential-chunk recurrence** (BMG HW-bug cap,
`docs/known_bugs.md`): at 16K we run 32 sequential chunks; llama's attention prefill
parallelizes. This is the #1 long-context weakness, NOT a strength.

Decode (tg) at depth degrades gracefully (81→46 over 512→16K) — normal KV growth.

**27B dense-hybrid (qwen35 — fewer DeltaNet layers, holds up FAR better):**

| T_pp | ours pp | llama pp | ours/llama |
|-----:|--------:|---------:|:----------:|
| 512   | **349** | 288 | 1.21× |
| 1024  | **549** | 284 | **1.93×** |
| 4096  | **477** | 271 | **1.76×** |
| 16384 | 109     | **231** | 0.47× (collapse) |

27B WINS prefill across the whole practical range (512–4096) — big (up to 1.93×) — and
only collapses at extreme 16K. Crossover is much higher than crown. So the long-context
prefill win is REAL up to ~4–8K (where most real prompts live); only extreme depth hits
the chunk cap. 27B tg@16384 = 10.93 (vs llama tg64 21.93 at depth0).

→ **Optimization lever:** CLAUDE.md says the HW bug is NOT reproducible on NEO 26.18 +
kernel 6.17.0-35 (validated 2026-06-20); `IE_QWEN35_PREFILL_CHUNK` raises the cap. Raising
it would directly fix long-context prefill. ⚠ RISKY (this is the recurrence that can hang
BMG — the thing that froze the box). **User decision before testing** — do NOT enable
unprompted.

## Coder full sweep + per-prompt real-world

**Coder pp sweep (pure MoE, T≤256 cap → worst long-context):**

| T_pp | ours | llama | ratio |
|-----:|-----:|------:|:-----:|
| 512   | 667  | **1196** | 0.56× |
| 1024  | 626  | **1167** | 0.54× |
| 4096  | 240.5| **991**  | 0.24× |
| 16384 | 50.6 | **601**  | 0.08× (12× slower) |

→ Collapse is NOT DeltaNet-only: the T≤256 non-DeltaNet chunk cap serializes coder's
long prefill even worse. **Lifting the prefill chunk caps is the single biggest
long-context lever across ALL arches** (gated `IE_QWEN35_PREFILL_CHUNK` /
`IE_QWEN3NEXT_PREFILL_CHUNK`; ⚠ HW-hang risk, user decision).

**Per-prompt real-world (`ie-prompt-bench`, OUR engine only — llama-cli hangs on these
arches so no head-to-head here). All 5 prompts are short (40–170 tok), so pp scales with
prompt size and tg is ~constant:**

| prompt (≈tok) | Crown pp / tg | 27B pp / tg | Coder pp / tg |
|---------------|--------------:|------------:|--------------:|
| 01 short_chat (40)     | 179.6 / 77.2 | 27.6 / 15.5 | 283.1 / 52.6 |
| 02 long_instruction (70)| 568.9 / 77.1 | 64.8 / 15.4 | 497.3 / 53.4 |
| 03 codegen (55)        | 534.7 / 77.4 | 61.1 / 15.4 | 511.3 / 53.7 |
| 04 math_reasoning (60) | 595.7 / 76.4 | 69.1 / 15.4 | 496.2 / 50.2 |
| 05 long_context (170)  | **911.2** / 76.7 | **172.5** / 15.6 | **709.2** / 51.6 |

This is what looked like "strong long-context": for prompts UP TO the chunk cap (≤512
tok), bigger prompt = higher pp (crown 911 on the 170-tok prompt). Real win in the
common 0–512 tok range; the collapse only appears past the chunk cap (see sweep above).
tg is workload-independent (decode-bound): crown ~77, 27B ~15.4, coder ~52.

## 80B flagship (Qwen3-Next-80B-A3B, 2×B70 layer-split)

Ours = `ie-qwen3next-bench` (5-prompt, 2-card). llama = `llama-bench -sm layer` SYCL 2-card,
same 48 GB GGUF (Momix-44 abliterated Q4_K_M). Decode IS comparable (single-stream ~128 tok);
prefill less so (5-prompt median vs synthetic pp512).

| 80B | ours | llama | verdict |
|-----|-----:|------:|:-------:|
| prefill | 325.7 median (489 @208tok) | **627.9** (pp512) | lose |
| decode  | 56.55 | **60.6** | **lose 0.93×** |

⚠ **CORRECTION TO PRIOR CLAIM.** Project notes claimed "80B decode 51.8 vs llama 37.1 =
1.40× FASTER." Fresh clean measurement: **llama SYCL 60.6 decode — beats us.** The 37.1
was STALE; llama.cpp's SYCL qwen3next path improved and now leads on the 80B. Our decode
did improve (51.8→56.55 post-oneAPI) but llama improved more. The B70-DeltaNet moat has
eroded on this model.

## ⭐ HONEST SCORECARD (clean box, fresh both engines, same GGUF/card)

| Model | prefill (short ≤1K) | prefill (long ≥4K) | decode | net |
|-------|:------:|:------:|:------:|-----|
| **Crown 35B-A3B** | **WIN 1.09×** | lose (DeltaNet cap) | **WIN 1.10×** | ✅ **clean win** (short/mid ctx) |
| 27B dense | **WIN 1.2–1.9×** | lose @16K only | lose 0.69× | mixed: prefill win, decode loss |
| Coder 30B | lose 0.56× | lose 0.08× | lose 0.66× | ❌ lose both |
| Gemma 31B | — | — | base lose 0.78× / **spec WIN 1.12×** | spec-only win (lossless ✓) |
| Gemma 26B | — | — | base par / spec 1.39× | spec NOT lossless ⚠ |
| 80B Next | lose | lose | lose 0.93× | ❌ lose (prior "win" was stale) |

**Bottom line:** the one unambiguous, clean head-to-head win is **Crown 35B-A3B** (both
axes, short/mid context). 27B wins prefill in the practical range but loses decode. The
rest lose or need (lossless) spec-decode to reach/beat parity. Several memory-claimed
"wins" did NOT survive clean apples-to-apples re-measurement — the user was right to insist.

**Top optimization levers (from kernel maps):**
1. Dense Q4 decode GEMV (`gemv_q4k` 51.6% on 27B, ~42% BW vs llama 64%) — the 27B/coder decode loss.
2. Long-context prefill chunk caps (T≤256/512) — collapses ALL arches at ≥4–16K ctx.
3. Coder MoE prefill (`moe_pfl_*`) — loses even at short context.
4. gemma26 spec lossless regression (near-tie MoE routing).

## ⭐ ROOT CAUSE of the dense-decode loss (llama source analysis, 2026-06-22)

The 42%→64% Q4_K decode BW gap is **the weight memory LAYOUT, not the dot product.**
llama.cpp repacks Q4_K weights at load into **Structure-of-Arrays** (all quants ‖ all
scales ‖ all d/min — `~/llama.cpp/ggml/src/ggml-sycl/quants.hpp:86-105`), so each 16-lane
sub-group reads quants as **aligned contiguous int32 loads** and touches each scale/min
once from a separate stream. The interleaved GGUF Q4_K block (144 B, qs not 4-aligned)
forces byte-assembled loads + smears scale bytes through the quant stream → kills
coalescing → 42% BW.

**Fix, ranked by BW impact (copy from llama):**
1. SoA weight repack at load (`quants.hpp:86-105`) — the big one
2. 16-lane work map, 1 row/sub-group (`mmvq.cpp:8-50`, WARP_SIZE=16)
3. Aligned int32 quant reads (`vecdotq.hpp:45`)
4. Fused scale/min via DP4A + Q8_1 activation-sum, no fp16 expansion (`vecdotq.hpp:227-254`)
5. SoA-reorder activation + `reduce_over_group` (`quantize.hpp:63-91`)

**KEY: we ALREADY have this path — `gemv_q6_soa` does exactly this for Q6_K (the 2.5×
27B-Q6 decode win). The work is porting the same SoA int-dot kernel to Q4_K decode.**
It's a bit-identical repack → re-verify PPL gate ≤6.57. ⚠ Caveat: a prior Q4_K-SoA
attempt regressed in the MoE *down-proj* context (memory: qwen3moe DTYPE-CONDITIONAL) —
but llama PROVES Q4_K-SoA wins for *dense* decode, so the regression was impl-specific,
worth revisiting. Full detail: `.claude/agent-memory/ml-inference-optimizer/reference_llama_sycl_q4k_decode.md`.

## Optimization attempt #1: Q4_K SoA int-dot decode — NEGATIVE (honest)

Built `gemv_q4_soa` (SoA repack + int8 dp4a, mirroring the 2.5×-Q6 win), host-verified
bit-identical, wired into qwen35_dense gated opt-in (`IE_QWEN35_Q4_SOA=1`, default OFF).
27B decode A/B (clean, same card):

| variant | decode tg | vs baseline 15.55 |
|---------|----------:|:-----------------:|
| baseline (AoS fp16 W4A16) | 15.55 | — |
| SoA int-dot (SLM activation) | 14.68 | **0.94× (worse)** |
| SoA int-dot (GMEM activation) | 15.85 | 1.02× (noise) |

**Result: parity at best — NOT a win.** VTune's roofline called it: 27B decode is
**occupancy/latency-bound** (27% occupancy, 85% XVE-stalled), **not bandwidth-saturated**
(L3-BW-bound 0%). The SoA fix improves coalescing/bandwidth-efficiency, which doesn't help
when the limiter is too few subgroups in flight. The Q6-SoA 2.5× win came from a terrible
baseline (scalar gemm_q6_K 140 GB/s); the Q4 baseline (gemv_q4_K) is already ~60% BW, so
coalescing had no headroom. Prefill also regressed (forced SoA dequant) — wiring would need
SoA-at-T==1-only. **Kept gated default-OFF** (default path byte-identical; crown gate intact).
LESSON: roofline-classify (VTune) BEFORE building — source analysis found a real difference
that was NOT the dominant bottleneck. The real decode lever = OCCUPANCY (split-K cross-WG
reduction / more rows-per-subgroup), a deeper rewrite.

## Deferred (next session)
- VTune `gpu-hotspots` deep-dive on 27B + Coder decode (real EU occupancy / BW% — the
  kernel maps already name the levers; VTune quantifies headroom). Validated VTune is
  installed (2026.2); not yet run.

## TODO this session
- [ ] Prefill context-length SWEEP (512→16K) both engines — the long-context story
- [ ] Per-prompt 5-prompt real-world breakdown (short/instruction/codegen/math/longctx)
- [ ] next80 (80B, 2-card)
- [ ] VTune deep-dive on 27B + Coder decode (real BW/occupancy)
