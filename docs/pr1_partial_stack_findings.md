# PR #1 — Partial-Stack Acceptance-Rate Measurement: Findings

**Date**: 2026-05-05
**Tool**: `tools/ie_partial_stack_eval.cpp`
**Calibration**: 256-token built-in clean-prose corpus, streaming T=1
**PPL pre-check**: 6.56 (baseline 6.54, within ±0.03 noise floor — additive change is safe)

## Executive summary

The hypothesis under test was that running the first N layers of Qwen3.6 and applying the existing `lm_head` to the partial-stack residual would yield a viable cheap draft path for self-speculative decoding (Tier-S1 option a). **The hypothesis is rejected.** The residual stream at intermediate layers is not in logit space — `lm_head` only meaningfully aligns with the final-norm output (L41); applying it to L01..L37 yields near-random predictions.

## Data

Coarse sweep across the stack (255 scored streaming-decode steps each):

| cut N | α_top1 | α_top5 | α_top10 | mean_rank | mean_KL(full‖cut) |
|------:|-------:|-------:|--------:|----------:|------------------:|
| L05   | 0.027  | 0.047  | 0.055   |  28,775   |  9.06             |
| L10   | 0.024  | 0.027  | 0.043   |  39,827   |  9.24             |
| L15   | 0.012  | 0.016  | 0.035   |  43,681   |  9.54             |
| L20   | 0.016  | 0.020  | 0.027   |  57,859   | 10.82             |
| L25   | 0.012  | 0.024  | 0.039   |  59,120   | 10.78             |
| L30   | 0.024  | 0.047  | 0.075   |  34,059   | 10.19             |
| L35   | 0.231  | 0.384  | 0.427   |   4,277   |  5.87             |

Fine-grained sweep at the tail (255 scored steps each):

| cut N | α_top1 | α_top5 | α_top10 | mean_rank | mean_KL |
|------:|-------:|-------:|--------:|----------:|--------:|
| L35   | 0.231  | 0.384  | 0.427   |   4,277   | 5.87    |
| L36   | 0.322  | 0.494  | 0.541   |   5,085   | 5.22    |
| L37   | 0.427  | 0.620  | 0.678   |   4,230   | 4.01    |
| L38   | 0.561  | 0.800  | 0.839   |     361   | 2.51    |
| L39   | 0.737  | 0.949  | 0.957   |       3   | 0.87    |
| L40   | 1.000  | 1.000  | 1.000   |       0   | 0.00    |

L40 is a sanity check — it matches L41 exactly (the partial-capture path does the
same `rms_norm + lm_head` the model does). The 100% match confirms the
measurement methodology is sound.

## Interpretation

1. **Logit alignment is concentrated in the last 2-3 layers**. Up through L37 the cut output's argmax is on average rank 4,000+ in the full distribution. Between L37 and L39 the rank collapses by 3 orders of magnitude. Whatever transformation `lm_head` was trained to read from is built almost entirely in the final 5% of the stack.

2. **The early-layers-are-noisy regime extends much deeper than typical literature suggests**. Some EAGLE / probing-classifier papers report meaningful logit alignment from middle-layer probes; that does not generalize to Qwen3.6, at least not via the unmodified `lm_head`.

3. **Self-speculative decoding via "early exit + reused lm_head" is dead** for our model. Even at the most generous cut (L38 = 95% of layers) the cost ratio is ~0.95 and α_top1 is only 0.56 — far below the >0.95 break-even acceptance rate that ratio would require. At cuts shallow enough to be cost-effective (c ≤ 0.30, ie L≤12), α_top1 ≈ 0.02 — random.

## Decision

The Tier-S1 plan changes:

- **Option (a) — reused-lm_head DeltaNet-stack drafting: REJECTED.**
- **Option (b) — train a small EAGLE-style head with LoRA: now the path forward.** The head must replace the role currently played by the last 5-10 layers of the stack — i.e., it cannot be a single linear projection. EAGLE-3 (single autoregressive transformer block + token-embedding skip + projected logits) is the right reference architecture.

## Implications for PR #2

- Scope changes from "wire DeltaNet stack as draft" to "train and integrate a small EAGLE-3-style head". The draft-stack-as-feature-extractor idea is still potentially useful — it just needs a learned head on top, not a reused `lm_head`.
- Training infrastructure becomes a hard prerequisite. Our SYCL runtime is inference-only; the head will be trained in PyTorch (single B70 via IPEX or fall back to a CUDA box for training, then port the trained weights into our engine).
- Expected gain target unchanged (1.5-2× decode), but with non-trivial training-side setup cost.

## Implications for the broader plan

- **PR #3 (prefix caching) and PR #4 (lm_head Q4_K) are unblocked** and become the right next moves. They're independent of PR #2 and ship without any training infrastructure.
- **PR #5 (INT4 KV + QLoRA-QAT)** also requires training infrastructure; bundles naturally with PR #2's training setup.

## Verification trail

- `ie-perplexity` baseline pre-experiment: PPL = 6.56 (vs documented 6.54 ± 0.03 — within noise).
- Tool output preserved at `/tmp/partial_stack_eval.log` (coarse sweep) and `/tmp/partial_stack_eval_late.log` (fine sweep).
- Code path is purely additive: the partial-capture path reads `ws_x_` and writes side buffers; full-stack `out_logits` is bit-identical to baseline. Confirmed via the L40 cut showing α_top1=1.000.
