# V4.1 — Phase 45: the consumer layers score only their candidate blocks (criteria before the result)

## Why

Layers 20-39 (card 1) compress at ratio 1, so at a 224k context their latent cache holds ~224k entries; layers 2-19
(card 0) at ratio 2 hold ~112k. Index sources 24, 28, 32 and 36 consume layer 20's candidate pool (2,048 blocks of 8
= 16,384 latents per row), yet `ds4_indexer_score` still ran every row against EVERY latent -- O(T x NC x 32 heads x
128) -- and `ds41_candidate_apply` then overwrote every non-candidate score with -inf. At 224k that is 4 layers x
~13.7x more dot products than the pool needs; with Phases 43-44 in, the O(NC) score is the largest term left on
card 1 at long context (docs/82's estimate ~7 s per chunk at 224k over all 8 index layers).

## The change

`ds4_indexer_score_candidates`: the keytile (prefill strips) and headlane (decode, small T) shapes with a template
`Keep` flag -- a key whose block is not kept takes -inf directly and its sub-group returns before the dot products
(the key, or the keytile key pair, is uniform in its sub-group, as the existing past-the-end return already relies
on). The kept keys run the unchanged code. At consumer layers with the pool live the forward calls it and drops the
apply; everywhere else nothing changes. V4's instantiations are `Keep = false`. Kill switch `IE_DS41_CAND_SKIP=0`.

## Criteria

1. **Bit-identical**: on the same build, the 32k long test (pool live on card 1 from ~16k) gives the same prefill
   logits FNV and the same 64 decode ids with `IE_DS41_CAND_SKIP=0` and default; the decode test 28/28.
2. **Faster** where the pool is live: the 32k pipelined prefill not slower; the 223k re-measurement records the
   difference.

## Results (build of 09:10, `ds41_work/p43/p45_*`)

**Criterion 1 -- bit-identical: MET.** 32k long test, pipelined, same build:

| | `IE_DS41_CAND_SKIP=0` | default (skip) |
|---|---|---|
| prefill logits FNV | `136ba7d427287a6b` | `136ba7d427287a6b` |
| 64 decode ids FNV | `c14c73474cec53b2` | `c14c73474cec53b2` |
| decode test | -- | 28/28 |

**Criterion 2 -- at 32k, no gain and no loss beyond noise:** 318.4 vs 307.7 tok/s, but the difference is on CARD 0
(busy 97.1 vs 101.0 s), whose layers the change does not touch; card 1, where it applies, is 92.6 vs 92.2 s. At
32k the pool is live only from NC 16,384 and cuts a consumer layer's keys by at most 2x, on a term worth ~0.5 s of
a chunk. Its size shows at 224k (the re-measurement below).

## Found en route: Phase 39's split top-k threw past 131,072 keys

The first 223k run of the day aborted after its prefill: `ds41_indexer_topk_split: too many chunks for one level-2
sort (n_keys > 131,072 at top_k 512)`. A 223,236-token prompt ends with a 4-token tail fed one token at a time
(docs/65), and card 1's layers compress at ratio 1, so those T = 1 rows carried ~223k keys into the split, whose
eligibility test (`T <= 8 && n_keys > 8192`) never checked its own level-2 capacity. Phase 39 was gated at 24k and
32k; nothing had run past 131k since. `ds41_topk_split_wanted` now also requires ceil(n_keys / 4,096) x top_k <=
16,384; a longer row takes the radix work-group that served 223k before Phase 39 -- the same picks in the same order
(the split's own bit-identity contract), so nothing else changes. **Consequence: every decode step past ~131k
tokens would have aborted since `b92b928`.** A three-level split would restore the Phase 39 speed there (radix is
linear in n_keys: ~2.4 ms per call at 224k by docs/70's slope, x 5 index layers on card 1).

Also: a pipeline stage thread now catches exceptions and returns them as the chunk's error (an uncaught throw on a
stage thread would terminate the process rather than fail the prefill).

## 223k, re-measured with Phases 43-45 and the split fix (09:17-09:30, `p43/r223k_b.log`)

`ie-ds41-run` on the 223,237-token held-out document, capacity 224,000, the 200k real-text ranking, 300 greedy tokens,
`IE_DS41_STAGES=1` -- docs/70's FINAL command:

| | docs/70 (2026-09-15) | now |
|---|---|---|
| prefill | 2,155.5 s = **104 tok/s** | **700.1 s = 319 tok/s (3.08x)** |
| decode, 300 tokens | 66.80 s = **4.49 tok/s** | **63.75 s = 4.71 tok/s** |
| needle at 50 % depth | retrieved | **retrieved** |
| pipeline stages (109 chunks) | -- | card 0 busy 691.8 s (6.35 s/chunk), card 1 666.1 s (waited 32.6 s for input) |
| host MemAvailable floor | not recorded | 35 GiB |

The prefill rate at 224k now equals the 32k rate (308-318): the O(NC) prefill terms no longer bind, and card 0 --
whose per-chunk time is the same at 32k and 224k -- sets it. This run carries all three changes, so the candidate
skip's own share at 224k is not separated (an `IE_DS41_CAND_SKIP=0` 223k run would; ~12 min of GPU). Decode's last
step: layers 215.2 = attn 33.8 + moe 177.0 (join 83.5 + tail 62.7), hit rate 70.8 % -- the residency wall of docs/70.

## Phase 45's own share at 223k (09:38-09:53, `p43/r223k_noskip.log`)

Same command, `IE_DS41_CAND_SKIP=0`, 8 tokens: **prefill 874.2 s = 255 tok/s against 700.1 s = 319 tok/s with the skip
(+25 %)**. Card 1 busy 858.8 s -> 666.1 s (-22 %); with the skip off card 1 was the slower stage (card 0 waited 167.6 s
for it), with it on card 0 is. Criterion 2: MET at the length it was built for.

## V4 kernel tests on this build (09:54)

The V4 model shares both templated kernels. `deepseek4_attn_segs_test` (two-segment axis bit-identical to the
contiguous one on every shape), `deepseek4_attn_xmx_gate_test` (0 failures), `deepseek4_attn_xmx_prep_test`
(0 failures), `deepseek4_indexer_tile_test` (272 cases on 4 devices), `deepseek4_decode_gate_test` (router and indexer
top-k bit-identical, attention inside its bound): all PASS. The V4 model itself was not loaded.

## Unit test (09:58): `tests/unit/ds41_cand_skip_test.cpp`

`ds4_indexer_score_candidates` against `ds4_indexer_score` + `ds41_candidate_apply`, random data, 32 heads x 128, block
8, row 0 all kept / row 1 none / the rest 30 %: T 1 x 20,000 and 16,385 keys, T 7 x 16,391 (headlane), T 64 x 16,389,
256 x 40,001, 130 x 33, 65 x 131,077 (keytile; odd counts leave a tail key pair and a partial block): **0 differing
entries of ~20 M on every shape, PASS.**
