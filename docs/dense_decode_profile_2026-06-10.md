# Dense decode per-kernel profile — Qwen3-8B Q4_K_M (P3b Task 0)

**Date:** 2026-06-10 · **Device:** Intel Graphics [0xe223] (B70) · **GGUF:**
`~/.seal/models/Qwen3-8B-Q4_K_M.gguf` · **Tool:** `ie-bench --kprofile-decode`
(new tool-only flag; arch-agnostic `KernelProfiler` + `ie::ps` wrappers).

**Method:** `ie-bench --ctx 512 --prefill 128 --decode 32 --warmup 8
--kprofile-decode`. The profile captures **one extra, fully-warm T=1 decode
step** taken *after* prefill + 8 warmup + 32 timed decode steps — so JIT is
discarded and caches are hot. Profiled decode at pos=168. Three runs;
numbers are tight (gemm_q6k 5.30/5.35/5.31 ms; gemv_q4k_q8 ~10.31 ms across
all three). tg128-equivalent measured tg = **44.3–44.5 tok/s** (matches the
43.7 ledger baseline). GPU exclusivity held (single workload, pre-checked).

## Measured per-kernel decode table (run 3, representative)

| kernel | ms/token | % of decode | calls/token | avg ms | what it is |
|---|---:|---:|---:|---:|---|
| **gemv_q4k_q8** | **10.323** | **46.6%** | 198 | 0.052 | all Q4_K projections (q/k/o/gate/up + 18 Q4_K ffn_down), int-dot dp4a |
| **gemm_q6k** | **5.311** | **24.0%** | 18 | 0.295 | **18 Q6_K ffn_down — scalar M=1 GEMM cliff** |
| gemv_q6k_huge_q8 | 2.277 | 10.3% | 1 | 2.277 | lm_head Q6_K (fast q8 path) — once/token |
| fa2_partial_fp16 | 2.195 | 9.9% | 36 | 0.061 | FA-2 partial (KV read); launch/latency-bound, ctx-scaling |
| gemv_fp16 | 0.618 | 2.8% | 36 | 0.017 | attn_v F16 projection |
| res_rms_fused | 0.433 | 2.0% | 36 | 0.012 | fused residual+rms |
| rms_norm_q8 | 0.391 | 1.8% | 37 | 0.011 | rms-norm with fused q8 quant |
| rms_norm_f32w | 0.138 | 0.6% | 72 | 0.002 | q/k head norms |
| quant_q8_1 | 0.114 | 0.5% | 90 | 0.001 | activation quant |
| rope | 0.101 | 0.5% | 72 | 0.001 | RoPE |
| fa2_combine_fp16 | 0.100 | 0.4% | 36 | 0.003 | FA-2 combine |
| swiglu / residual_add / fa2_append / embed | <0.06 each | <0.3% each | | | tail |
| **TOTAL** | **22.169** | 100% | | | (= 44.4 tok/s) |

## Roofline sanity (Task 0 Step 3 — effective GB/s per GEMV)

Weight bytes/token from the desk table, divided by measured kernel ms:

| kernel | weight bytes | ms | **eff GB/s** | vs 450 roofline |
|---|---:|---:|---:|---|
| gemv_q4k_q8 | 3312 MB | 10.31 | **321** | 71% — efficiency gap |
| **gemm_q6k (ffn_down)** | 743 MB | 5.31 | **140** | **31% — THE CLIFF** |
| gemv_q6k_huge (lm_head) | 510 MB | 2.26 | 226 | 50% (fits-SLM path; small, leave) |
| gemv_fp16 (attn_v) | 302 MB | 0.62 | **487** | **>roofline — already fast** |

Aggregate weight traffic 4868 MB / 22.17 ms = **220 GB/s** = 49% of roofline
(matches the ledger "50% efficiency"). Roofline for this traffic @450 GB/s =
10.82 ms = 92.4 tok/s.

## Hypothesis verdicts (CONFIRM / REORDER with numbers)

**#1 — Q6_K ffn_down scalar `gemm_q6_K` cliff → CONFIRMED, #1 by a clear margin.**
18 Q6_K ffn_down layers route through `gemm_q6_K` at M=1 (dense_dispatch.hpp:149
via `gemv_q`, because `q6k_slm_gemv_q8_fits(K=12288)` is FALSE). Measured
**140 GB/s** — exactly the predicted slow scalar path, 31% of roofline, on
743 MB/token. It is **5.31 ms = 24.0%** of the whole step. Fixing it to the
~350 GB/s int-dot speed the Q4_K family already gets recovers **~3.2 ms**
(→ ~52.7 tok/s); to 450 GB/s recovers ~3.7 ms (→ ~54 tok/s). This is the
single highest-payoff fix and exactly Task 1 as written. **Keep Task 1 #1.**

**#2 — attn_v `gemv_fp16` bandwidth → REFUTED. Deprioritize / drop.**
Measured **487 GB/s** — *above* the 450 roofline (small F16 traffic served
well from cache). Only **0.62 ms = 2.8%** of the step. The desk's 209–222
GB/s figure was a large-N / prefill regime that does not apply at T=1 decode.
Converting attn_v to Q8_0 would save at most the ~0.3 ms DRAM-traffic delta
while risking PPL — **not worth a task.** Recommend **dropping Task 2** (or
demoting it far below the new #2 below).

**#3 — missing gate/up fusion / launch count → PARTIALLY SUPPORTED, but small.**
gate+up are the two `gemv_q4k_q8` calls inside the 198-call bucket; fusing
them removes 36 launches/token and shares one q8 activation read. But the
launch-overhead signal that *is* visible is **`fa2_partial_fp16`: 36 calls,
2.19 ms, only 11 GB/s on 25 MB of KV** — i.e. latency/launch-bound, not
bandwidth. So launch overhead is real but concentrated in the FA-2 fan-out,
not the gate/up pair. The dual-FFN fusion is still worth it for the shared
activation read, but its ceiling is small (sub-ms). **Keep as a low-priority
cleanup, not the next dispatch.**

**NEW #2 (not in the original 3) — `gemv_q4k_q8` efficiency gap at 321 GB/s.**
The largest single bucket (10.31 ms, 46.6%) runs at only **71% of roofline**.
Closing it to 450 GB/s would save **~2.95 ms** — *comparable to the Q6_K fix*.
This is the generic per-GEMV efficiency gap the plan's "Honest expectations"
flagged as a possible separate desk cycle. After Task 1, this is where the
remaining gap to 70 tok/s lives; reaching 70 from these kernels alone almost
certainly requires both the Q6_K cliff AND a gemv_q4k_q8 retune.

## Confirmed target ranking (reordered by measured ms recoverable)

1. **gemm_q6k ffn_down cliff** — 140→~350+ GB/s, **~3.2 ms** → Task 1 (as written). **HIGHEST PAYOFF.**
2. **gemv_q4k_q8 efficiency** — 321→~450 GB/s, **~2.95 ms** → NEW, larger than old #2/#3. Needs a fresh kernel-retune desk pass.
3. **fa2_partial launch/latency** — 36 serialized launches, ~2.19 ms, latency-bound → fusion/batching candidate.
4. (low) dual gate/up fusion — shared activation read, sub-ms ceiling.
5. **DROP:** attn_v Q8_0 conversion (#2 in plan) — already at 487 GB/s.

## Recommended Task 1 (next dispatch)

**Proceed with Task 1 exactly as written:** K-tiled Q6_K int-dot GEMV
(`gemv_q6_K_q8_ktiled`) for the 18 Q6_K ffn_down decode layers, routed in
`dense_dispatch.hpp` behind `IE_NO_Q6K_KTILED`, A/B'd against the current
`gemm_q6_K` path. It is the confirmed #1 lever (~3.2 ms, 44.5 → ~53 tok/s),
crown-safe (new symbol, dense-only route, crown K≤4096 never hits it), and
the desk math held to the GB/s. **Do not start Task 2 (attn_v) — refuted;**
the second dispatch after Task 1 should instead be a desk pass on the
gemv_q4k_q8 efficiency gap.

## Tooling note

`--kprofile-decode` is the only code change for Task 0: tool-only, in
`tools/ie_bench.cpp`. It does not touch the engine, any kernel, or the crown
prefill `--kprofile` (which stays crown-only). PPL tripwire run as the gate.
