# IE Engine — v1 RELEASE

> **Current release: v0.2.21 (October 2026).** This file is the record of the first release, of June 9, 2026, kept as
> written; its figures are for one Arc Pro B70 and a Q4_K_M model and are not comparable with the two-card figures
> below. Later releases are described in the README ("New in v0.2.21", "New in v0.2.20", "New in v0.2.18", "New in v0.2.13", "New in v0.2.8", "New in v0.2.6", "In v0.2.0") and in the GitHub
> release notes:
> [v0.2.21](https://github.com/Red-Weasel/machx-inference-engine/releases/tag/v0.2.21),
> [v0.2.20](https://github.com/Red-Weasel/machx-inference-engine/releases/tag/v0.2.20),
> [v0.2.18](https://github.com/Red-Weasel/machx-inference-engine/releases/tag/v0.2.18),
> [v0.2.13](https://github.com/Red-Weasel/machx-inference-engine/releases/tag/v0.2.13),
> [v0.2.8](https://github.com/Red-Weasel/machx-inference-engine/releases/tag/v0.2.8),
> [v0.2.6](https://github.com/Red-Weasel/machx-inference-engine/releases/tag/v0.2.6),
> [v0.2.0](https://github.com/Red-Weasel/machx-inference-engine/releases/tag/v0.2.0),
> [v0.1.0](https://github.com/Red-Weasel/machx-inference-engine/releases/tag/v0.1.0).
>
> **v0.2.21 in short** (two Arc Pro B70 cards, measured October 5, 2026):
> - Flash-Next read image prompts of more than 2,048 tokens without the image (v0.2.10 through v0.2.20); fixed, both test
>   pictures are right at 281 to 16,747 prompt tokens.
> - Restart points on Flash-Next (one lane): a 15.7K-token prompt with one line changed two thirds in 28.5 -> 18.4 s; image
>   conversations keep theirs (second turn of a 16.7K-token image conversation: 27B 10.2 -> 4.5 s, 35B-A3B class 7.2 -> 1.0 s,
>   Flash-Next 24.8 -> 3.0 s).
> - `--spec` is offered by `ie capabilities` for the 35B-A3B class and Flash-Next, so a client's load screen can show it.
> - Flash-Next's MTP draft head, opt-in (`--spec --spec-head <file>`): greedy decode 28.7 / 28.3 -> 30.8 / 33.7 tok/s, nothing at
>   temperature 0.7, prefill 502 -> 412; off unless a path is given.
>
> **v0.2.20 in short** (Arc Pro B70 cards, measured October 5, 2026):
> - One card runs the Qwen models on the two-card code: Qwen3.8-27B Q6_K prefill 190 / 173 -> 823 / 830 tok/s and decode
>   13.1 / 12.5 -> 22.0 / 20.4; the 35B-A3B class Q5_K_M, which did not load on one card, 1,337 / 1,304 and 87.2 / 82.9
>   tok/s; request lanes, `--spec`, the prompt cache and the restart points on one card; replies byte-equal to two cards.
> - A load is tried as asked: the 27B's load-time memory check is gone, the prompt cache gets the VRAM the cards report
>   free, and the remaining memory estimates (lanes, system-RAM pins) warn instead of refusing.
> - Restart points on the 35B-A3B class's lead request lane.
>
> **v0.2.18 in short** (two Arc Pro B70 cards, measured October 4-5, 2026; versions 0.2.14 to 0.2.18):
> - With `--spec`, a reply that copies text already in the conversation is drafted from that text and checked in one
>   forward: a 1,800-token file edit 108.5 -> 28.4 s on Qwen3.8-27B, 24.3 -> 7.1 s on the 35B-A3B class (and across its
>   request lanes: 12 edits at once 74.3 -> 53.6 s), 68.7 -> 30.7 s on Flash-Next. A long draft is checked through the
>   prefill kernels, so a token at a near-tie can differ from plain decoding.
> - `--spec` on the 27B's Q6_K / Q5_K files: three draft-head faults fixed (Q6_K 21.5 -> 26.9 tok/s).
> - In-place restart points on the 27B and 35B-A3B splits: after one line changed two thirds into a 29.3K-token prompt
>   the next request took 17.5 s instead of 22.9 (27B) and 9.0 instead of 15.2 (35B-A3B class); replies byte-equal.
> - The 27B's load no longer refuses at long context with the prompt cache on.
>
> **v0.2.13 in short** (two Arc Pro B70 cards, measured October 4, 2026):
> - Prefill on the Qwen models: Qwen3.8-27B Q8_0 1,037 / 820 -> 2,962 / 3,001 tok/s (2.4K / 10.7K-token prompts), the
>   35B-A3B class 1,816 / 1,256 -> 2,896 / 2,808, Flash-Next 241 -> 470 tok/s on text never seen before; the 15-agent
>   replay 116.1 -> 77.1-77.2 s.
> - `--spec` on the Qwen3.8-27B split works with sampling, the prompt cache and any prompt length; tokens equal plain
>   decoding's (17.1 -> 24.3 tok/s at temperature 0.7 on a short prompt).
> - Every device allocation of 64 MiB or more is tested at load for two pages sharing one physical page; that fault
>   ended Flash-Next replies after a few tokens on one test card.
>
> **v0.2.8 in short** (two Arc Pro B70 cards, measured October 2, 2026; the Qwen3.6-35B-A3B class, Q8_0):
> - The 15-agent replay takes 116.3–116.5 s where v0.2.6 took 131.3–132.4 s in the same day's runs (−12 %); an
>   agent's decode goes from a median 8.65–8.80 to 9.83–9.86 tok/s and its first token from 3.79–4.79 to 2.58–2.83 s.
> - The engine's serial turns (restore, shared-prefix mark, conversation snapshot, prompt end) run beside the lane
>   pipe instead of pausing it: 84–86 pauses per replay → 0.
> - Decoding lanes regroup without the pauses' bursts: 3.30 → 4.56 rows per decode card step.
> - Tiled prefill attention from 512 positions instead of 6,144: 512-row prefill pieces at 2–6K depth 13–16 % faster,
>   a 2,038-token cold prompt in 1.28 s instead of 1.41 s (that one measured with the threshold set by environment on
>   the v0.2.6 build).
> - Still slow, and changed: the first of 15 conversations that arrive together with one shared prompt gets its
>   first token at 11.3 s (8.6 or 10.2 s on v0.2.6); prompts with prefill pieces ending between 512 and 6,143
>   positions compute other bits than on v0.2.6 (`IE_Q35MOE_FA2_TILE_MINCTX=6144` gives v0.2.6's kernel choice).
>
> **v0.2.6 in short** (two Arc Pro B70 cards, measured September 29 – October 1, 2026):
> - Agent swarms on the Qwen3.6-35B-A3B class (Q8_0): a replay of 15 agents that read and write at once, which did
>   not finish in 336 s before, takes 139.7 s; a shared system prompt is read once, conversations resume in their
>   lane, both cards work during a deep prefill, short requests go first, and workers keep writing while a long
>   prompt prefills.
> - `ie serve` without `--parallel` picks the number of request lanes (`--parallel auto`); `/props` reports
>   `slot_ctx`; an orderly stop takes 0.43–1.08 s where it took 14.58–49.70 s.
> - XML `<function=` tool calls are returned as structured `tool_calls`, and thinking as `reasoning_content`, on the
>   Qwen thinking templates.
> - Native integer-dot Q8_0 kernels, bit-identical: one request's prefill on the 35B-A3B class takes 34–36 % less
>   time at 2–8K tokens, and Qwen3.8-27B at 16 lanes goes from 113.8 to 153.5 tok/s.
> - Native Q6_K / Q5_K on the 27B and 35B-A3B splits: Qwen3.8-27B Q6_K 5.2 → 22.0 tok/s, and Q5_K_M loads (24.0).
> - Still slow: with 15 agents reading at once an agent decodes at a median 8.1 tok/s (25.9 when 16 only write), and
>   a deep prefill beside workers that write long replies is about 22 % slower (`IE_Q35MOE_DECODE_QUOTA=0` reverts).

**Date:** 2026-06-09
**Engine state:** commit `5088da1` + release-checklist fixes
**Hardware:** Intel Arc Pro B70 (BMG-G31, 32 GB GDDR6, Xe2-HPG, 608 GB/s)
**Model:** `Qwen3.6-35B-A3B-Q4_K_M.gguf` (19.7 GiB, hybrid DeltaNet+Attention MoE, 36B total / 3B active)
**Backend:** SYCL / Level Zero, IntelLLVM 2026.0.0, NEO 26.14.37833.4, IGC 2.32.7
**Comparison anchor:** llama.cpp Vulkan build b8902, same GGUF, same GPU,
`llama-bench -ngl 99 -sm none -mg 0`, measured the same hour as the engine numbers.

---

## Headline

**The engine beats llama.cpp Vulkan on both prefill and decode.**

> **2026-06-09 P1 amendment (docs/benchmark_matrix_2026-06-09.md):** the
> llama.cpp **SYCL** backend at master b9586 (post-#23142 + MMVQ) now
> measures 1092 pp512 / 81.1 tg128 on this hardware — ahead of this engine
> (86% / 82%). The Vulkan comparison below remains accurate (engine leads
> any Vulkan build decisively), but "fastest overall on B70" currently
> belongs to llama.cpp SYCL master. Gap-closing work (integer-dot decode
> GEMVs, oneDNN prefill GEMM) is specified in the matrix doc.
>
> **2026-06-10 RESOLUTION — TOTAL CROWN:** both gaps closed the next day.
> Decode crown v1.5 (84.1 turbo / 81.0 default vs SYCL master 81.31), then
> prefill crown v1.6: **pp512 1144 ± 5 vs SYCL master 1064 ± 8 same-hour
> (+7.6%)** via per-expert SoA weight repack + int-dot MoE prefill with
> full-K register lattices and split-half-sum corrections (default ON,
> PPL 6.52).  **The engine now leads llama.cpp's best backend on BOTH
> metrics on B70.**  See docs/benchmark_matrix_2026-06-09.md §v1.6.

| metric | engine | llama.cpp Vulkan b8902 | ratio |
|---|---:|---:|---:|
| **prefill pp512** | **938.6 tok/s** (v1.2) | 885.0 ± 5.5 | **106.1%** |
| **decode tg128** | **66.2 tok/s** (fp16 KV, 5-prompt suite, v1.2) | 39.75 ± 0.06 | **166.5%** |

**v1.2 (same day):** the MoE router was a hidden decode bottleneck — a serial
8×256 top-k scan on one lane (~4 ms/token) plus a scalar-load logit dot.
Vectorized logits + parallel top-8 (packed-key max-reduces): decode
52.4 → 66.2, pp512 +2.8%. Down-kernel M_TILE 8→16 added +1.4% pp512.
Tried and reverted with findings: fp16-packed inner products (−2.7%, IGC
doesn't emit packed-rate FMAs from vec<half,2>), gemm_fp16 double-buffer
(−9.3% — at 4 KB tiles, WG multithreading already hides latency).

Prefill went **202.9 → 899.7 tok/s (+343%) in the final optimization day**
(2026-06-09, E1–E5 in `docs/prefill_attack_plan_2026-06-09.md`), with PPL
held bit-for-bit at 6.54 through every step.

## v1 hard gates

| gate | target | measured 2026-06-09 | status |
|---|---|---|---|
| PP @ T=512 ≥ 50% of llama.cpp Vulkan | ≥ 443 tok/s | **899.7** (101.6% of llama.cpp itself) | ✅ **203% of gate** |
| TG ≥ 95% of llama.cpp Vulkan | ≥ 37.8 tok/s | **46.8** (117.7%) | ✅ |
| PPL stable on multi-token corpus | drift ≤ noise | 6.54 = historic baseline ± 0.03, reproduced 6× today | ✅ |
| INT8 KV cache validated | drift < 0.5% | 0.35% (`validate_int8_kv.sh`) | ✅ |
| Unit tests | all pass | 7/7 (`ctest`) — two stale Phase-5-era sub-tests updated to current kernel contracts | ✅ |
| Build clean | — | clean build; fresh-clone build verified (also fixed `.gitignore` hiding 4 src/core files) | ✅ |
| ≥50 tok/s decode @ small ctx (absolute) | 50 | **52.4 suite / 53.3 INT8 @ ctx=1** (v1.1 kernel, `35a85ea`) | ✅ |

**v1.1 decode update (same day):** the absolute-50 gate initially measured
48.5 (the historical 52.6 predated the bisect cleanup). Closed at full
quality by `gemv_q6_K_slm` — SLM-slab-staged Q6_K GEMV (no sub-group
shuffles, algebraic scale fold) on the lm_head and ssm_out shapes:
decode 46.8 → **52.4** suite, 48.5 → **53.3** INT8 @ ctx=1, PPL unchanged.

## Decode menu (v1.2)

| config | suite tok/s | INT8 @ ctx=1 | pp512 | PPL | vs llama.cpp tg128 |
|---|---:|---:|---:|---:|---:|
| **default** (Q6_K lm_head) | **66.2** | **67.7** | **938.6** | **6.55** | **167%** |
| turbo (`-lmhead-q4k.gguf`) | 69.5 | 73.4 | 947.1 | 6.64 | 175% |

fp16-KV 5-prompt suite (51–219-tok real prompts, 3-run median), ≤0.3% spread.
The turbo GGUF requires no engine changes — `gemv_q()` dispatches by tensor
dtype; pass `--gguf .../Qwen3.6-35B-A3B-Q4_K_M-lmhead-q4k.gguf`.

## Quality

- Built-in 511-tok PPL: **6.54** (fp16 KV), int8-KV prefill-chunked drift 0.35%.
- Every 2026-06-09 optimization was gated on PPL before being kept; E2/E2b
  preserve per-accumulator FP add order (bit-identical), E1/E2c/E5 reproduce
  6.54 exactly.
- Known constraint: DeltaNet recurrence non-determinism on BMG-G31 under long
  single-call prefill — production paths chunk at T≤256 where it never fires
  (`docs/known_bugs.md`; 28-step bisect concluded HW-level, vendor escalation
  pending).

## What made prefill fast (one day, 2026-06-09)

1. **E1** — T≥64 projections dequant Q4_K/Q6_K once to an fp16 scratch and run
   the dense `gemm_fp16` instead of fused quant-XMX (202.9 → 309.4).
2. **E2** — vectorized weight + SLM loads in the three MoE prefill kernels
   (309.4 → 589.7).
3. **E2b** — in-kernel M-tile grid for scalar `gemm_q4_K`; the N=32 alpha/beta
   path was 960 two-workgroup launches (589.7 → 754.5).
4. **E2c** — SLM-staged contiguous Q6_K weight slabs (754.5 → 840.7).
5. **E5** — expert-sorted activation pre-gather; killed 24× redundant scattered
   reads (840.7 → 899.7).

Tried and reverted on gates: E3 (BK=64 gemm_fp16 retile, −21% occupancy loss),
E4 (header vector loads, neutral).

## Deferred to v1.1+

- Recover absolute ≥50 tok/s decode @ small ctx (Q4_K lm_head variant or
  gemv_q6k_huge kernel work).
- ≥50 tok/s @ 32k ctx (algorithmic: paged attention / fp8 KV / MTP).
- gate_up stage-1 is now compute-bound: fp16-product math or XMX retry.
- gemm_fp16 double-buffer at BK=16.
- Server features: multi-turn KV reuse via prefix cache (primitives landed,
  PR #3), defrag, OpenAI-compatible endpoint (Phase 11).
- IGC vendor escalation for the DN recurrence issue (repro tools ready:
  `ie-bug-monitor`, `ie-bug-live`).
