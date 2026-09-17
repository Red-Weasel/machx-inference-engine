# V4.1 — Phase 39: the indexer top-k split across work-groups (criteria before the build)

## The term

`ds4_indexer_topk` at decode is ONE work-group per query row: at 32k it sorts 16,384 packed keys in 128 KB of SLM
with 1,024 lanes (306 us per call, 8 index-source layers = 2.45 ms/token); past the SLM wall (n_keys > 16,384) it
is the radix shape, O(n_keys) per row -- at 224k, ~7x the 32k row, an estimated ~2 ms per call, ~17 ms/token
(estimate, not measured; the 224k profile of docs/70 predates the gathered attention and was not read for this
term). Either way the card runs one work-group for the whole term.

## The change

A two-level EXACT top-k: level 1 splits the key axis into chunks of 4,096 and sorts each chunk's keys in its own
work-group (T x n_chunks work-groups side by side), emitting each chunk's top-k as indices; level 2 gathers those
candidates' keys and sorts them once (n_chunks x k <= 16,384 keys, the shape the single kernel already runs at 32k),
emitting the row exactly as `ds4_indexer_topk` does: descending, pitch top_k, the causal sentinel -1.

Bit-identical by construction: every member of the row's top-k is in its chunk's top-k (the same total order),
the union's top-k is therefore the row's, and ties resolve on the same packed key (score, then smaller index).
Dispatched for n_keys > 8,192 at T <= 8 (decode); prefill rows keep the single kernel (T work-groups already).
`IE_DS41_TOPK_SPLIT=0` restores the single kernel.

## Criteria

1. **Bit-identical:** a unit check of the split against the single kernel on random rows at n_keys in {16,384,
   32,768, 114,688} with random causal cutoffs (all 512 picks equal, sentinels included); the profiled decode test's
   criterion lines identical to Phase 38's; `multi` 38/38.
2. **Speed:** `ds4_indexer_topk` per call at 32k below 200 us (from 306), in the long test's decode profile.
3. **Long context, measured not estimated:** the 224k long test's decode profile before/after for the top-k term,
   if a run fits the night; otherwise the 32k number stands and the 224k claim stays an estimate in this doc.

## The unit check (`bench/ds41_topk_split_check.cpp`, random rows with exact ties and causal cutoffs inside the
## range, one row with fewer reachable keys than top_k; the single kernel via `IE_DS41_TOPK_SPLIT=0`)

| n_keys | T | single kernel | split | speed-up | picks |
|---|---|---|---|---|---|
| 16,384 | 1 | 759 us | **46** | 16.5x | identical |
| 16,384 | 4 | 195 | 46 | 4.2x | identical |
| 32,768 | 1 | 467 | **80** | 5.9x | identical |
| 32,768 | 4 | 436 | 63 | 6.9x | identical |
| 65,536 | 1 | 727 | **104** | 7.0x | identical |
| 65,536 | 4 | 762 | 135 | 5.7x | identical |
| 114,688 (the 224k context) | 1 | 1,226 | **191** | 6.4x | identical |
| 114,688 | 4 | 1,272 | 280 | 4.6x | identical |

(The single kernel at 16,384 x T = 1 is the 1,024-lane bitonic over 128 KB of SLM; in situ it measured 306 us per
call, the bench's 759 includes its first-launch shape.) At 224k the eight index layers' term goes from an estimated
~10 ms/token to ~1.5.

Which gates exercise the split: the decode test's prompts and the multi test's scenarios have fewer than 8,192
keys at decode, so they cannot see it (they pass trivially). The model-level bit-identity gate is the 24k held-out
text (~12k latents), whose 120 greedy tokens must equal the Phase 34 stream exactly; the 32k long test gives the
in-situ kernel time.

## Results: PASS -- the split is the default for decode rows over 8,192 keys

| criterion | result |
|---|---|
| 1. bit-identical | unit check identical at every shape; the 24k held-out text's 120 greedy tokens **identical** to the Phase 34 stream (`p39_text.log` vs `dx_text.log`; ~12k latents, the split engaged); the decode test and multi cannot see it (< 8,192 keys) |
| 2. 32k, in situ (profiled long test) | Phase 34: 8 calls/tok, 2.45 ms/token, avg 306.5 us -> Phase 39: neither the single kernel nor the split's two kernels appear in the decode table any more -- the table lists the top entries only and its last row is ds41_hc_mixes_part 0.50 ms/token, so the term is below that from 2.45; the named-kernel total moved 38.8 -> 33.8 ms/token across Phases 38 + 39, of which Phase 38 is ~3.1 |
| 3. 224k | not run tonight; the unit check's 114,688-key row (1,226 -> 191 us) stands as the measured kernel figure, the token-level number as an estimate |

32k long test, default mode: **62.1 ms/token = 16.11 tok/s** (64.2 after Phase 38, 67.1 after Phase 35). Named
kernels under profiling 38.8 -> 33.8 ms/token across Phases 38 + 39. The 24k held-out run: 10.26 tok/s (9.74 before
the two passes).
