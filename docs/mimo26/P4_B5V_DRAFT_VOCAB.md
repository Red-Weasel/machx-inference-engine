# P4 B5v — a restricted draft vocabulary for MiMo-V2.6's DFlash drafter (2026-09-26)

Branch `p4-b5v-draft-vocab`, from `p4-b3-mimo-rows` 08c73ba. **Status: measured and NOT merged** -- the code (a knob, default off) stays on that branch (9fd775c); only this record is in the main line, because the measured gain is below noise on MiMo. Motivation: llm-scaler #728 (`docs/INTEL_STACK_LIVING.md` §2.1,
§4) scored only ~50K likely token ids in the MTP drafter and took Qwen3.6-35B from 147.4 to 189.8 tok/s and the 27B from
59.5 to 74.6, acceptance and output unchanged. FR-Spec: the drafter's head scores a frequent subset of the vocabulary; the
target still verifies every draft with the full vocabulary, so the greedy output is unchanged by construction and only the
drafter's cost and its acceptance can move.

## Answer first

**Not worth enabling on MiMo; the default stays OFF.** Measured (R1, `results/mimo26/p4/b5v/r1_off_cpuoff_kprof.log`: the 8
held-out prompts x 64 tokens, drafter 7 / min-p 0.7, CPU leg off, `--kprof`): the drafter's lm_head GEMM costs **2.107 ms per
draft call** -- 1.25 GB of fp16 at 593 GB/s, already at the card's bandwidth roofline -- and the drafter runs 0.30 times per
generated token, so the head is **0.60 ms of the 80.9 ms a token costs: 0.74 % of decode time** (0.47-1.11 % across the
prompts). A draft vocabulary can only shave part of that (K = 64K keeps 43 % of the head's bytes: at most ~0.34 ms/token,
0.4 %), while on this workload it makes 2.6 % of the generated tokens un-draftable at K = 64K (6.9 % at 32K; measured on
the held-out generated ids, see "Coverage"), each of which ends a pass early and costs a verify row of tens of ms. The
expected net is negative; the run-to-run noise of a served A-B-A (+-3 %) is four times the whole ceiling, so no subset run
was spent on the GPU (the coordinator's rule: stop under ~2 %).

| Gate | Result |
|---|---|
| Default (knob unset) byte-identical to the base 08c73ba | **PASS** -- R1 `generated:` md5 65501302fb19267820d991815210fb2d = B3's r15 (the same command on 08c73ba); VRAM 31.21 / 28.05 GB, static/pinned/stream 78/178/8 62/194/8, drafter 3.14 GB on the last card, passes / rows / accepted per prompt all identical to r15. The default-off path allocates nothing: the compact head and its map exist only when `dv_n_ > 0`. |
| Head share of decode | **0.74 % mean** (per prompt 0.47 / 1.11 / 0.56 / 0.63 / 0.57 / 0.89 / 0.94 / 0.65 %) -- under the 2 % bar, so the subset was not enabled on the GPU. |
| Greedy output identical with the subset on | not run (see above); holds by construction -- the target verifies every draft over the full vocabulary, the drafter's ids only decide what is proposed. |
| Acceptance / speed with the subset on | not run; the coverage table below bounds the acceptance loss (2.6 % of generated tokens at 64K, 6.9 % at 32K un-draftable). |
| E2 `--only87` | not run: no default changed, no engine, server or forward code touched; `ie-mimo26-cache-test` sees the drafter only through `Mimo26DFlash`, whose default path is the base's. |
| Host-only test | **PASS** -- `ctest -R mimo26_draft_vocab_test`, 11/11. |

What R1 also showed, and what is worth doing instead: the drafter's own attention kernel (`mimo26_dflash_attn`) costs
0.51-0.58 ms PER LAYER at a 1024-position ring -- 2.6-2.9 ms per draft call, **~27 % of the call, more than the head's 20 %**
-- for ~4 MiB of ring reads and 67 M MACs. One sub-group per (block row, q head) walks the ring serially (1,032 steps with a
group reduce and two exps each); 512 sub-groups leave the card nearly empty, so it is latency-bound (0.036 ms at prompt 00's
40-position context, 0.58 ms at the window). A split-K walk over the ring (the flash-decoding shape the target's decode
attention already uses) would recover most of it with no acceptance cost. That, not the vocabulary, is the drafter's lever.

## Where the drafter's cost goes (before building)

`Mimo26DFlash::draft` (src/model/mimo26_dflash.cpp) is ONE forward of the 8-row block [anchor, 7 x mask] through the 5 dense
drafter layers, then the final norm and the TARGET's fp16 lm_head (`Mimo26Forward::head_weights()`, [152,576 x 4,096], shared
with the target on the last card) over rows 1..7, then a per-row argmax + softmax kernel. Weight bytes read per call, fp16:

| part | bytes per call |
|---|---|
| 5 layers x (q 64 MiB + k 8 + v 8 + o 64 + gate 128 + up 128 + down 128) | 2.64 GB |
| lm_head 152,576 x 4,096 x 2 | 1.25 GB |
| total | 3.89 GB (the head is 32 %) |

At 8 rows every GEMM is a weight-streaming GEMV, so the estimate before measuring was: bandwidth-bound on 3.9 GB, the head a
third of the call, the drafter 0.1-0.45 calls per token (r15: 2-29 calls per 64 tokens) -- **the head ~0.5-1.5 % of decode**.

**Measured (R1, `--kprof`, means over the 8 prompts; `p4b5v_table.py`):**

| drafter kernel (per draft call) | avg ms | per call | share of the 10.2 ms call | ms per generated token | share of decode (80.9 ms/token) |
|---|---|---|---|---|---|
| `mimo26_dflash_gemm` (35 layer GEMMs, 2.64 GB) | 0.137 x 35 | 4.8 ms (550 GB/s) | 47 % | 1.363 | 1.7 % |
| `mimo26_dflash_attn` (5 layers, ring of <= 1024) | 0.51-0.58 x 5 | 2.6-2.9 ms | ~27 % | 0.763 | 0.9 % |
| **`mimo26_dflash_head`** (1.25 GB fp16) | **2.107** | **2.1 ms (593 GB/s)** | **20 %** | **0.599** | **0.74 %** |
| `mimo26_dflash_argmax` | 0.078 | 0.08 ms | 1 % | 0.022 | 0.03 % |
| norms, RoPE, converts, host round trips | | ~0.4 ms | ~4 % | | |
| drafter wall (`drafter 10.2-10.3 ms/call`, 0.30 calls/token) | | 10.2 ms | 100 % | ~3.1 | **3.9 %** |

The head GEMM runs at the bandwidth roofline (Intel's own dense GEMV figure for the card is ~574 GB/s), so the only way to
make it cheaper is to read fewer rows -- which is exactly what a draft vocabulary does, and why it cannot be worth more than
the 0.74 % it costs. The target's own head GEMM over the verify rows (`mimo26.gemm.head`, 2.10 ms per pass, the same 1.25 GB)
is untouched here. Per prompt, head ms/token and share: 00 0.351 (0.47 %), 03 0.954 (1.11 %), 04 0.395 (0.56 %), 05 0.494
(0.63 %), 06 0.421 (0.57 %), 07 0.790 (0.89 %), 08 0.826 (0.94 %), 09 0.561 (0.65 %).

Why the llm-scaler gain does not transfer: their MTP drafter is one transformer layer over a resident model whose decode step
is 5-7 ms, so the drafter's full-vocabulary head was a large share of every token; MiMo's drafter is 5 dense layers
(2.64 GB of fp16 weights per call, 3.14 GB on the card) behind a host-resident-expert target whose step is 70-88 ms.

## What was built

| Piece | Where | What |
|---|---|---|
| Ranking file | `<model>/ie_draft_vocab_mimo26.txt` | '#' comments, then one token id per line, best first. Built by `ie-mimo26-draft-vocab` (`tools/mimo26_draft_vocab.cpp`). |
| Knob | `IE_MIMO26_DRAFT_VOCAB=<K or file>`; `ie-mimo26-run --draft-vocab K|FILE` | K = the first K ids of the model's file; a path = all of that file's ids; unset / 0 = the full vocabulary. **Default OFF**: nothing changes without the knob. `Mimo26DFlash::load` reads the env, so `ie serve`, `ie-mimo26-lanes-test` and `ie-mimo26-cache-test` take it with no change to the engine or server files. |
| Compact head | `Mimo26DFlash::set_draft_vocab`, `init`, `set_head`, `draft` | `init` allocates the compact head [K, 4096] fp16 and the row -> id map BEFORE the target's forward sizes its auto static tier (the tier sees the VRAM taken, as with the rest of the drafter). `set_head` gathers the lm_head's rows of the subset into it. `draft` runs the head GEMM against the compact head (N = K instead of 152,576), the argmax kernel over K columns, and maps the row back to its token id. A row without a finite maximum reports -1 and is refused as before. |
| Probabilities | `draft`, `last_probs`, the `min_p` cut | The per-row probability is the softmax over the K SCORED ids: p_K(id) = exp(l_id - m) / sum_{i in K} exp(l_i - m) >= the full-vocabulary p(id) of the same id, since the ids left out are not in the denominator. The engine's `dflash_minp` 0.7 therefore cuts a few FEWER drafts (never more): a draft that would have been cut with p just under 0.7 passes when the excluded ids held enough mass. Where the target's argmax is outside the subset the drafter proposes the best in-subset id instead, with a p_K that can be high; the target rejects it (a wasted verify row, no output change). Measured under "Acceptance". |
| Profiler buckets | `draft` | `mimo26_dflash_gemm` (the 35 layer GEMMs) and `mimo26_dflash_head` (the head GEMM) under `--kprof`, next to the existing `mimo26_dflash_attn` / `mimo26_dflash_argmax`. Zero cost without a profiler. |
| Reader + test | `mimo26_read_draft_vocab`, `tests/unit/mimo26_draft_vocab_test.cpp` (ctest `mimo26_draft_vocab_test`, host only) | Comments and blank lines skipped, rank order kept, the first K; refused: fewer ids than K, a repeat, an id past the vocabulary, a non-id line, a missing file. |

Not changed: `src/model/mimo26_forward.cpp`, `include/ie/mimo26_forward.hpp` (B3), `src/engine/mimo26_engine.cpp`,
`src/server/*`, `src/model/mimo26_prefix_cache.cpp` (B4). No serve CLI flag (those files are B4's); the env knob covers serve.

## The ranking: public text, disjoint from the evaluation prompts

The held-out prompts (`results/mimo26/p4/prof/heldout_prompts/`) and anything derived from them were NOT used to build the
ranking (a ranking profiled on one workload can be worse than none on another: the expert-residency lesson). Sources, all
public and on disk:

| source (`--corpus`) | files | tokens (engine tokenizer, no special parsing) | distinct ids |
|---|---|---|---|
| `~/llama.cpp/wikitext-2-raw/wiki.train.raw` (English prose) | 1 | 2,518,423 | 38,801 |
| `~/llama.cpp/{src,common,gguf-py,examples}` (C/C++/Python/Markdown, <= 2 MiB each) | 424 | 1,699,710 | 19,131 |
| `~/llama.cpp/docs` (Markdown) | 46 | 117,299 | 8,211 |
| the tokenizer's merge order as a Zipf pseudo-corpus: id i weighs 1 / (i + 1) | -- | -- | 151,643 |

Order: the tokenizer's 32 added tokens first (`<|im_start|>`, `<|im_end|>`, `<tool_call>`, `<think>`, ... -- the delimiters a
chat / tool turn cannot be drafted without), then the 151,643 base ids by the MEAN of their frequency over the four sources
(each weighs the same whatever its size), ties by id. 51,597 ids occur in a corpus; the other 100,046 are placed by the merge
prior alone. For this byte-level BPE the id order IS the merge order (verified: the first 20,000 merges' result ids are 256 +
rank), and merge order is itself a frequency ranking of the tokenizer's training text -- without the prior, early merges the
three corpora happen to miss (tab indentation `\t\n`, `\t   `, ` href`, `,"`: ids 1,335-2,760) fell behind every seen id.
Padding ids 151,675..152,575 (never produced) are left out. File: `~/models/MiMo-V2.6-Flash-RL/ie_draft_vocab_mimo26.txt`,
151,675 ids, md5 0bae081703c38bd33e057381e502ec7f (`results/mimo26/p4/b5v/ranking_build.log`).

**Coverage** (`ie-mimo26-draft-vocab --eval`, `p4b5v_coverage.py` on r15's token lists; the held-out text is evaluated, never
counted): the share of tokens the drafter could propose at all with the top K ids.

| K | wikitext train (in-sample) | llama.cpp code (in-sample) | held-out prompt text (40,041 tokens) | r15 GENERATED tokens (418) |
|---|---|---|---|---|
| 16,384 | 91.5 % | 97.4 % | 87.3 % | 88.8 % |
| 32,768 | 97.6 % | 99.4 % | 92.9 % | 93.1 % |
| 65,536 | 99.9 % | 100 % | 97.7 % | 97.4 % |
| 98,304 | 100 % | 100 % | 99.94 % | 100 % |

The generated tokens outside the top 65,536 are the owner's path and name pieces (`/we`, `/Desktop`, `=B`, `=file`, `ewe`,
` HAND`): frequent in Dream's transcripts, absent from any public text, and by construction not learnable from a corpus
disjoint from the prompts. So on this workload the drafter loses 2.6 % of the tokens at K = 64K and 6.9 % at K = 32K
before any speed is gained -- each such token ends a pass early (the drafts after it are wasted) and costs a verify row.

## Gate runs

Every GPU job went through the main checkout's `scripts/ie-run-guarded` (`--mem 220G --timeout 540`), foreground, after the
preflight (`p4b5v_check.sh`: no Dream desktop, no engine process, both cards <= 40 MiB, no CAT error / GT reset / timeout in
the last 10 min of the kernel log, the lock free) and followed by the same check after 30 s. Logs in `results/mimo26/p4/b5v/`.

| Run | Command | Result |
|---|---|---|
| R1 `r1_off_cpuoff_kprof` | `ie-mimo26-run` (binary md5 948e43871dac69d61ec95a0287c38ce9, commit 9fd775c) on the 8 held-out prompts, `--n 64 --chunk 2048 --ctx 16384 --static 0 --ranking <model>/ie_ranking_mimo26_chat.txt --dflash 7 --dflash-minp 0.7 --kprof --dflash-log`, `IE_DS41_CPU_MISS=0`, knob unset | exit 0, 06:41-06:44; post-check CLEAR. `generated:` md5 **65501302fb19267820d991815210fb2d** (= r15 on 08c73ba); `VRAM 31.21 GB 28.05 GB`, `static/pinned/stream per layer 78/178/8 62/194/8`, `dflash: ... 3.14 GB on the last card` -- all as r15. |

R1 per prompt (r15's decode ms/token in brackets; R1 ran with profiling-enabled queues, r15 without):

| prompt | decode ms/token | passes | rows | accepted | acc/pass | drafter ms/call | calls | head ms/token | head share |
|---|---|---|---|---|---|---|---|---|---|
| 00 | 75.2 (71.6) | 2 | 13 | 10 | 5.00 | 26.6 (first calls) | 2 | 0.351 | 0.47 % |
| 03 | 85.7 (84.3) | 14 | 52 | 35 | 2.50 | 10.2 | 29 | 0.954 | 1.11 % |
| 04 | 70.5 (70.2) | 11 | 69 | 51 | 4.64 | 10.3 | 12 | 0.395 | 0.56 % |
| 05 | 77.9 (77.3) | 13 | 70 | 49 | 3.77 | 10.3 | 15 | 0.494 | 0.63 % |
| 06 | 74.3 (74.1) | 6 | 40 | 28 | 4.67 | 10.3 | 7 | 0.421 | 0.57 % |
| 07 | 89.2 (88.3) | 14 | 57 | 40 | 2.86 | 10.2 | 24 | 0.790 | 0.89 % |
| 08 | 87.9 (87.0) | 12 | 51 | 31 | 2.58 | 10.2 | 20 | 0.826 | 0.94 % |
| 09 | 86.3 (85.7) | 12 | 65 | 47 | 3.92 | 10.3 | 17 | 0.561 | 0.65 % |
| all | 80.9 mean | 84 | 417 | 291 | 3.46 | 10.5 | 126 | 0.599 | 0.74 % |

Passes, rows, accepted and calls equal r15's line for line (the drafter's decisions are unchanged); decode is +0.3-1.7 % over
r15 on the 64-token prompts, the cost of `--kprof`'s profiling queues (prompt 00 is 12 tokens and includes the drafter's
first two calls at 26.6 ms). Not measured: the knob-unset binary WITHOUT `--kprof` against r15 (the code path differs from the
base by one null-pointer test per draft call and profiler pushes predicated on `g_profiler`, the target's own `prof()`
mechanism) -- a 4.5-minute run the gate may want.

Not run (no default changed, the head share ruled the subset out): the 64K / 32K identity runs, the served A-B-A, and E2
`--only87`. If the gate wants them, the commands are the R1 command with `--draft-vocab 65536` (identity: the same md5),
without `IE_DS41_CPU_MISS=0` for the served A-B-A (knob off / on / off), and
`IE_MIMO26_DRAFT_VOCAB=65536 ie-mimo26-cache-test <model> <scratch>/p4b5v_text.txt --only87`.

## Risks and follow-ups

- **The knob is inert by default.** Nothing in the base path changed except `set_head` becoming an out-of-line function and
  the profiler pushes; the default-off run is byte-identical to the base and allocates nothing extra. The code stays behind
  `IE_MIMO26_DRAFT_VOCAB` / `--draft-vocab` for a checkpoint whose drafter head IS a large share (an MTP-style one-layer
  drafter over a resident model, as in llm-scaler's case).
- **If it is ever turned on:** the min-p cut is over the subset's softmax (>= the full-vocabulary probability), so more
  low-value drafts pass; and every token outside the subset costs a pass. On Dream's workload the misses are the owner's
  path / name pieces, which no public corpus contains; a ranking that includes them would have to be built from Dream text
  that is disjoint from the evaluation prompts (the expert-ranking lesson), never from the prompts themselves.
- **The compact head is extra VRAM** (K x 8 KiB: 512 MiB at 64K) on the last card, taken from the auto static tier's budget.
- **Follow-up worth a phase: the drafter's attention kernel.** 2.6-2.9 ms per draft call (27 %) for a 1024-position ring is a
  latency-bound serial walk; a split-K flash-decoding version would save more than the head ever could, with no acceptance
  cost. Second: the drafter's 2.64 GB of fp16 layer weights (47 % of the call) would halve in FP8 or int8 -- but that changes
  the drafts (acceptance), so it needs its own gate.
- **Compiler note:** one icpx 2026.1 frontend segfault while compiling `src/core/deltanet_state.cpp` in this worktree's first
  build; the retry built clean. Not root-caused; the same compiler built B3.
