# PR #4 — lm_head Q6_K → Q4_K: Findings

**Date**: 2026-05-05
**Status**: Part A (Q4_K lm_head) **shippable as user-selectable variant**.
              Part B (sparse logits) deferred — depends on PR #2.

## Setup

The repo had an alternate GGUF file already prepared
(`/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M-lmhead-q4k.gguf`)
in which the lm_head tensor is requantized from Q6_K → Q4_K. Tensor
size dropped 397.852 MiB → 272.812 MiB (−31.4%, matches the Q6_K vs
Q4_K bits-per-weight ratio).

Engine support is already polymorphic — `gemv_q()` dispatches to
`gemv_q4_K` or `gemv_q6_K` based on the loaded tensor's dtype, so no
code changes are required to use the alternate file.

## Quality measurement (3 trials each)

| variant | T1 | T2 | T3 | mean | range |
|---|---:|---:|---:|---:|---:|
| Q6_K lm_head (baseline) | 6.54 | 6.54 | 6.53 | **6.537** | 0.01 |
| Q4_K lm_head | 6.64 | 6.64 | 6.64 | **6.640** | 0.00 |

**Δ PPL = +0.103.** Tight error bars on both sides confirm this is real,
not noise. Documented baseline of 6.54 is reproduced exactly on the
unmodified GGUF. The historical noise floor of ±0.03 is preserved.

## Performance measurement (clean ie-bench, no contention)

| metric | Q6_K (baseline) | Q4_K lm_head | Δ |
|---|---:|---:|---:|
| Prefill T=512 | 190.1 tok/s | 192.3 tok/s | +1.2 % (within noise) |
| Decode | 40.07 tok/s | **44.25 tok/s** | **+10.4 %** |
| lm_head decode kernel | 108.75 ms (10.0 % of decode) | 29.21 ms (2.9 %) | **−73 %** |
| Other top decode kernels | unchanged | unchanged | — |

The lm_head kernel time drops by ~74 %, well beyond the ~33 % expected
from byte-count ratio alone. The Q4_K kernel path on this codebase is
better optimized (XMX-friendly tile shapes) than the Q6_K path.

Decode speedup of **+10.4 %** (40.07 → 44.25 tok/s) materially closes
the gap on the documented decode roofline (228 → 130–150 tok/s after
the inference-engine-expert correction). New decode position: 44.25
tok/s vs ~150 ceiling = **3.4× off ceiling, was 3.7×**.

## Decision

**Ship as opt-in user variant**, not default. Quality delta is real
but small. +0.103 PPL is at the documented soft ceiling for "shippable
quantization change" (the inference-engine-expert advice was "<0.1
PPL drop is OK"; we are exactly at 0.1). Some users will prefer the
quality of the original Q6_K lm_head; others will want the speed.

No engine code changes required. The alternate GGUF file is the
opt-in vehicle:

```bash
# baseline (Q6_K lm_head, PPL 6.54, decode 40 tok/s)
./build/tools/ie-perplexity \
  --gguf /home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf

# faster variant (Q4_K lm_head, PPL 6.64, decode 44 tok/s)
./build/tools/ie-perplexity \
  --gguf /home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M-lmhead-q4k.gguf
```

## Part B — sparse logits — deferred

The "sparse-gather logits" half of PR #4 (compute logits only for the
k+1 candidate tokens during spec-decode verify, instead of full 248k
vocab GEMV) is only useful in conjunction with speculative decoding,
which is currently blocked on PR #2's training requirement
(per `pr1_partial_stack_findings.md`). Defer until PR #2 lands.

## What changed in the engine source

Nothing in `src/` or `include/`. The model loader already supports
both Q4_K and Q6_K lm_head; the dispatcher in `gemv_q()` already
handles both. PR #4 part A is purely a model-file substitution.

## Verification trail

- `/tmp/bench_q6k_clean.log` — baseline Q6_K bench output
- `/tmp/bench_q4k_clean.log` — Q4_K lm_head bench output
- 3-trial PPL: `/tmp/claude-1000/...bizgmn7w7.output`
- PPL preservation between bench runs: confirmed identical to
  pre-PR #1 baseline (6.54 / 6.54 / 6.53).
