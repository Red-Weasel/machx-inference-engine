# Dense decode slow-path — root-cause finding (2026-06-20)

## ✅ RESOLVED (primary cause) — `enable_profiling` was ALWAYS ON (commit a36c138)
`src/core/allocator.cpp` created every queue with `sycl::property::queue::enable_profiling{}`
unconditionally. On the in-order Level-Zero queue that adds ~1.76 ms HOST overhead PER kernel
submit → decode was SUBMISSION-bound, not GPU-bound (the kernels were always fast: `gemv_q4k_q8`
= 0.0207 ms/call on the crown; the 0.57 ms/call seen on the 4B was the host-gap, not the kernel).
**Fix: profiling is now opt-in (`IE_QUEUE_PROFILING=1`, default OFF).** Measured (ie-bench, clean):
crown decode **3.6 → 13.7 tok/s (3.8×)**, 4B decode **2.55 → 7.9 tok/s (3.1×)**. Applies to EVERY
model/path (crown, dense, the 27B split, drafters). ie-bench self-enables it only for `--kprofile`.

**⚠ STILL A GAP:** the crown posted "decode 81.0" historically; it's 13.7 now. profiling was the
biggest piece but not all of it — decode is STILL partly host-submission-bound (~0.4 ms/kernel
remains; GPU-busy ~11.5 ms vs wall 72.8 ms). Next: chase the remaining per-submit overhead (event
creation per submit? command-list flush? kernel count) AND re-verify the historical 81 wasn't a
different config (--fastforward / int8-kv / fewer steps). The 27B split should be RE-BENCHED with
this fix (its "11 tok/s" was ie-run WITH profiling on — real number now higher).

---
## Original investigation (how we got here)

**Trigger:** spec-decode drafter recon found Qwen3-4B decoding at ~2.2 tok/s (slower than the 27B
target) — "something fundamentally wrong" (owner). Investigated with the BENCH tool (not `ie run`,
which is JIT-noisy and was never the perf-validated path — owner was right to redirect).

## Method
`ie-bench --gguf Qwen3-4B-Q4_K_M.gguf --decode 64 --warmup 30 [--kprofile-decode]`. Raw `forward()`
timing, no chat loop / sampling / detokenize. `--kprofile-decode` gives per-kernel GPU-event times
for one warm decode step.

## Result — the int-dot decode GEMV is the bottleneck
Warm 4B decode = **2.55 tok/s** (392 ms/token) — should be ~50. Per-kernel decode profile:

```
  gemv_q4k_q8     122.7 ms   60.2%   216 calls   0.57 ms/call   <-- DOMINATES
  fa2_partial     47.0 ms    23.1%    36 calls   1.31 ms/call
  gemv_q8_0_soa   30.9 ms    15.2%    18 calls   1.72 ms/call   <-- the 27B Q8 path uses THIS
  gemv_q6k_small   0.76 ms    0.4%    18 calls   0.042 ms/call  <-- a sibling kernel, 13x FASTER
```

**`gemv_q4_K_q8` (the W4A8 int-dot decode) runs at ~0.57 ms/call = ~5% of memory bandwidth** (a
12 MB ffn_gate read should be ~0.027 ms). The ~0.5 ms floor regardless of matrix size → it's
LATENCY/OCCUPANCY-bound, not doing real work. Confirmed REAL (not profiler artifact): the profiler
uses GPU event timestamps, and `gemv_q6k_small` on the SAME run measures 0.042 ms/call — no uniform
inflation. So `gemv_q4_K_q8` is genuinely ~13× slower per call than a sibling decode kernel.

**This is the same kernel CLASS the 27B Q8 path depends on** (`gemv_q8_0_soa_q8`, 1.72 ms/call here).
So this one root cause throttles BOTH the spec-decode drafter AND the shipped 27B. Owner's hypothesis
("that W4A8 kernel was never optimized") is confirmed.

## Secondary signal
Wall decode (392 ms) > GPU-busy (~203 ms at pos=292) → ~half is host-submission gaps between ~250
kernels/token. Decode is ALSO host-feed-bound (kernel count). Fewer/fused kernels would help too,
but the kernel speed is the primary lever.

## Implication for ALL prior `ie run` perf numbers
The 27B "11 tok/s", the no-SLM "regression", single-GPU "thrash" were all measured via `ie run`
(JIT-noisy). They must be RE-MEASURED with ie-bench. The 27B's true decode is unknown until benched
(blocked: ie-bench appears single-GPU; the 27B Q8 needs 2-GPU — may need an ie-bench --gpus path or
a split-aware bench).

## Open question (owner: "dense path was done before, only Q8 was new")
Was `gemv_q4_K_q8` fast before and REGRESSED, or always ~5% BW? Resolve by git-bisecting
`src/ops/gemv_q8dot.cpp` history and benching a known dense model (Llama-3.1-8B) at each point. If a
regression, revert/find it; if always slow, it needs a real rewrite.

## Fix plan (the focused next effort — fast iteration via ie-bench + 4B, loads ~20 s)
1. **Bisect / confirm** regression vs always-slow (above).
2. **Optimize `gemv_q4_K_q8`** occupancy/latency: it's a 1-subgroup-per-column design (N_PER_WG=32,
   SG_SIZE=16) — likely too few resident subgroups to hide memory latency. Try: split-K (multiple
   subgroups per column + reduction), larger N_PER_WG, or a different work mapping. Measure each via
   `--kprofile-decode` (GPU-event truth, fast).
3. **Port the win to `gemv_q8_0_soa_q8`** (same structure) → speeds the 27B directly.
4. Re-validate: dense PPL unchanged (int-dot numerics identical), crown 6.4527, 4B decode tok/s.
5. THEN resume spec-decode (a fast drafter now exists).
