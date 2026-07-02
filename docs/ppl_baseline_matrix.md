# PPL baseline matrix — 2026-05-02

Authoritative reference for PPL gate decisions going forward.  Resolves
the mixed-corpus confusion that made it look like PPL "jumped from 6 to
14" (it didn't — different corpora).

## Model + engine artifacts

| | path |
|---|---|
| Original GGUF | `/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf` |
| lmhead-q4k GGUF | `/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M-lmhead-q4k.gguf` |
| llama.cpp build | `/home/weezy/llama.cpp-vulkan/llama-b8902/llama-perplexity` (Vulkan) |
| IE engine | commit `f0ec8c5` (`main`, both experimental flags default OFF) |

## Hardware

Intel Arc Pro B70 BMG-G31 stepping C0, IP 20.2.0.  NEO 26.14.37833.4,
IGC 2.32.7, libze 1.15.37833+4, GuC 70.60.0.

## Corpus 1: built-in 511-token sample (historic reference)

The `kSampleText` array baked into `tools/ie_perplexity.cpp`.  Clean
English prose smoke sample.  This is what the "historic PPL 6.51 / 6.61"
in the docs/PLAN.md was measured on.

| Run | Engine | GGUF | Flags | NLL | PPL | historic ref |
|---|---|---|---|---:|---:|---|
| B' | IE | original Q4_K_M | both OFF | 1.88 | **6.53** | 6.51 (`docs/bench_results_phase_lm_head_q4k_REVERTED.txt`) |
| C' | IE | lmhead-q4k | both OFF | 1.89 | **6.64** | 6.61 (`docs/bench_results_phase_shexp_q4k_REVERTED.txt`) |

Reproduces historic values within ±0.03 = noise floor of this corpus.

## Corpus 2: `/tmp/ppl_test.txt` (PLAN.md head, 1791 predicted tokens)

Built via `cat PLAN.md | head -2000 > /tmp/ppl_test.txt`.  Code-heavy
technical content; naturally higher PPL than clean prose.

| Run | Engine | GGUF | Flags | Methodology | NLL | PPL |
|---|---|---|---|---|---:|---:|
| **A** | llama.cpp Vk b8902 | original Q4_K_M | default | sliding 4×512 | — | **8.67** ±0.74 |
| A2 | llama.cpp Vk b8902 | original Q4_K_M | default | single 1×2048 | — | 22.79 ±3.36 |
| **B** | IE (`f0ec8c5`) | original Q4_K_M | both OFF | streaming, prefill_chunk=256 | 2.69 | **14.80** |
| **C** | IE (`f0ec8c5`) | lmhead-q4k | both OFF | streaming, prefill_chunk=256 | 2.70 | **14.87** |
| **D** | IE (`f0ec8c5`) | lmhead-q4k | both OFF, re-run | streaming, prefill_chunk=256 | 2.71 | **14.98** |
| **E** | IE (`f0ec8c5`) | lmhead-q4k | **ESIMD GEMM ON + FA2 prefill tiled ON** | streaming, prefill_chunk=256 | 2.71 | **14.99** |

## Run-to-run noise floor (identical code, identical corpus)

| corpus | runs | PPL range | noise floor (max−min) |
|---|---|---|---:|
| built-in 511-tok | B', C' (single each, vs historic) | 6.53 vs 6.51 / 6.64 vs 6.61 | **±0.03** |
| /tmp/ppl_test.txt | C, D (back-to-back) | 14.87 vs 14.98 | **±0.11** |

## Gate-threshold recommendations

Future PPL gates should be sized to **at least 2× corpus noise floor** to avoid
noise-bound ship/revert decisions:

| corpus | noise floor (1σ) | recommended gate threshold |
|---|---:|---:|
| built-in 511-tok | ±0.03 | Δ ≤ +0.10 |
| /tmp/ppl_test.txt 1791-tok | ±0.11 | Δ ≤ +0.30 |
| /tmp/ppl_test.txt with 3-run median | ±0.04 (effective) | Δ ≤ +0.10 |

The earlier "Δ ≤ +0.05" gate I used for several experiments was running
slightly under the single-run noise floor of `/tmp/ppl_test.txt`.  Past
ship/revert PPL decisions were defensible (no experiment exceeded ±0.15
on this corpus) but the methodology was tighter than the corpus warranted.

## Methodology caveats

**llama.cpp vs IE PPL are NOT directly comparable.**  Different methodology:
- llama.cpp `llama-perplexity`: sliding-window over fixed-size chunks; PPL
  computed over second half of each chunk (after warmup positions).
- IE `ie-perplexity`: prefill the first `prefill-chunk` tokens, then stream
  the remainder one token at a time, NLL over each streamed token.

Cross-engine apples-to-apples PPL comparison would require matching
methodology — easiest is to re-run llama.cpp with `--chunks 1 -c <T>`
forcing single-chunk evaluation, but even then warmup position handling
differs.  **Use within-engine comparisons only for PPL gate decisions.**

## Findings

1. **No model regression.**  Historic PPL 6.51/6.61 reproduces on the
   built-in corpus (6.53/6.64).  The "6 → 14" was a corpus difference,
   not an engine bug.
2. **Offline lmhead-q4k conversion costs ~0.1 PPL** on small clean
   corpus (B' vs C' Δ = +0.11), ~0.07 on PLAN.md corpus (B vs C Δ =
   +0.07).  Acceptable tradeoff for +12% decode.
3. **Experimental flags (Step 1 ESIMD GEMM + Step 2 FA2 prefill tiled)
   do NOT measurably affect PPL.**  D vs E Δ = +0.01 (within noise).
   Math is preserved; only WG layout / accumulation order differs.
4. **Both ESIMD GEMM and FA2 prefill tiled scaffolds (committed at
   `95ab5d2` and `f0ec8c5`) preserve model quality** — safe to flip ON
   for combined experiments.

## qwen3-8b dense baseline (P2 Task 7, 2026-06-10)

Model: `~/.seal/models/Qwen3-8B-Q4_K_M.gguf` (Ollama blob, `general.architecture=qwen3`),
engine arch branch `qwen3 (dense)`, builtin 511-token corpus, default
(int-dot Q8) decode, streaming T=1.

| Run | PPL |
|---|---:|
| 1 | 18.88 |
| 2 | 18.74 |
| 3 | 18.93 |

~~Baseline = 18.88 (median). Hard gate: PPL ≤ 19.10~~ **SUPERSEDED
2026-06-10** — the run-to-run spread these numbers were drawn from (and the
wider 18.98–19.42 T8 spread) was a software bug: an in-place read/write race
in `rope_partial`, fixed the same day (full bisect:
`docs/dense_nondeterminism_2026-06-10.md`).

### Re-derived dense gate (post rope-race fix, 2026-06-10)

The dense path is now **deterministic**: 10 process-separate runs of
`ie-perplexity --gguf ~/.seal/models/Qwen3-8B-Q4_K_M.gguf --max-tokens 511`
(builtin corpus, default int-dot decode, streaming T=1) all print
bit-identical results — mean ± sd = 2.940491 ± 0.000000, so a mean+3sd band
degenerates to exact equality:

| mode | avg NLL (gate = exact match) | PPL |
|---|---:|---:|
| **DEFAULT since 2026-06-15** — Q6_K→Q8_0 ffn_down repack decode, `--max-tokens 511` | **2.944273** | **18.9968** |
| `--max-tokens 511` (510 predicted), int-dot Q8 decode (now `IE_DENSE_NO_Q6K_REPACK=1`) | **2.940491** | **18.9251** |
| default invocation (511 predicted), int-dot Q8 decode (now opt-out) | **2.937037** | **18.8599** |
| fp16 decode (`IE_NO_Q8_DECODE=1`), `--max-tokens 511` | 2.943471 | 18.9816 |
| fp16 decode (`IE_NO_Q8_DECODE=1`), default invocation | 2.939935 | 18.9146 |

**Default decode changed 2026-06-15: the Q6_K→Q8_0 ffn_down repack
(`IE_DENSE_Q6K_REPACK`, formerly opt-in) is now ON by default** after a
family-wide PPL pass (Llama-3.1/3.2, Granite-3.3, Phi-4, Mistral-24B,
Codestral-22B all within ±0.22% NLL; decode +19%→~2.9×). New default dense gate
= **2.944273** at `--max-tokens 511`. The prior **2.940491** bit-exact
leak-canary is preserved on the opt-out path (`IE_DENSE_NO_Q6K_REPACK=1`) and
`scripts/p2_parity_qwen3.sh` now runs it that way (most-sensitive pure-fp16
detector). Both constants are deterministic (the repack decode int8-quantizes
the activation but is still bit-stable).

**Gate (opt-out canary): avg NLL == 2.940491 exactly at `--max-tokens 511`
with `IE_DENSE_NO_Q6K_REPACK=1`** (same
bit-exact-equality regime as the crown's 1.864495). Any deviation = a real
numerics change, not noise. **The constant is invocation-bound** (P2 T8
finding, 2026-06-10 late): the original derivation ran with
`--max-tokens 511`, which caps the corpus one token short (510 predictions);
the default invocation predicts 511 tokens and scores 2.937037 — also
bit-exact (verified 4/4 process-separate runs across two md5-identical clean
rebuilds; `--max-tokens 511` re-verified 3/3 at the same build). Keep flag
and constant in sync — `scripts/p2_parity_qwen3.sh` pins both.
Crown invariant re-verified at the fix commit: 1.864495 / 6.4527, 3/3 exact
(the crown constant is bound to the **default** invocation, 511 predicted).

**The absolute level (~19) is the MODEL, not an engine bug** — verified
three independent ways (P2 Task 7 oracle):
1. llama.cpp b8902 Vulkan scores **24.54** on the same GGUF + same corpus
   (`-c 256 --chunks 2`; crown GGUF scores 10.37 under identical settings —
   the ie/llama ratio is 0.77 for dense vs 0.62 for crown, same direction).
   Qwen3-8B is a reasoning-tuned model with no base variant; raw prose is
   simply out-of-distribution for it.
2. Layer parity: all 37 comparable residual slots match llama.cpp with
   rel_fro ≤ 1.4e-3, cosine 1.000000 (`scripts/p2_parity_qwen3.sh`).
3. Greedy parity: 64/64 tokens identical to `llama-completion --temp 0` on
   the fp16 decode path. (Default int-dot decode flips one near-tie at
   token 26, top-2 logit gap 0.09–0.13 — documented, within quant noise.)

Per-arch gate command: `scripts/p2_parity_qwen3.sh` (layer + greedy + PPL).

## Llama-3.1-8B-Instruct baseline (P3a, 2026-06-12)

Model: `~/models/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
(`general.architecture=llama`, gpt2/llama-bpe tokenizer, vocab 128256). Runs on
the DenseModel path with the load-time Q/K un-permute + rope_freqs scaling.

| mode | avg NLL (gate = exact match) | PPL |
|---|---|---|
| `--max-tokens 511` (510 predicted) | **2.378209** | **10.7856** |
| default invocation (511 predicted) | **2.380699** | **10.8125** |

PPL 10.79 is a *sane* absolute level — Llama-3.1-8B is not reasoning-tuned, so
raw-prose PPL sits well below qwen3-8b's 18.9 (reasoning-tuned). Deterministic
(exact-match gate, same regime as crown/qwen3).

Per-layer parity vs the llama oracle (`ie-llama-dump`, 2026-06-12): **cos_sim
0.99998–1.000000 on all 32 layers** (`rel_fro` 0.003–0.006 — gemm_fp16, far
tighter than the 27B's oneDNN 0.02). Greedy: the first token after "The capital
of France is" is a near-tie (`' a'` 15.984 vs `' Paris'` 15.945 engine; oracle
flips them at 16.012/15.903 — same top-2 set, fp16-vs-fp32 rounding, the
documented qwen3 token-26 precedent). Coherent generation confirmed.

Per-arch gate command: `scripts/p3_parity_llama3.sh` (layer cosine + near-tie + PPL).

## Standardized PPL command for future use

```bash
# Fast smoke (built-in corpus, 511 tokens, ±0.03 noise floor)
build/tools/ie-perplexity --gguf <model.gguf>

# Production gate (/tmp/ppl_test.txt, 1791 tokens, take 3-run median)
for i in 1 2 3; do
  build/tools/ie-perplexity \
    --gguf <model.gguf> \
    --text /tmp/ppl_test.txt \
    --prefill-chunk 256 --max-tokens 2048 \
    | grep "perplexity"
done

# llama.cpp anchor (same hardware, original GGUF, sliding 4×512)
/home/weezy/llama.cpp-vulkan/llama-b8902/llama-perplexity \
  -m /home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -f /tmp/ppl_test.txt -c 512 --chunks 4 -ngl 999
```
