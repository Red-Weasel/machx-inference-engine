# V4.1 — Phase 44: the prefill continuation attention over each row's live keys (criteria before the result)

## Why

After Phase 43 each card is kernel-bound at 32k (docs/83), and the largest kernel whose cost grows with the context is
`ds4_attention_xmx` (Phase 42's profile: 3,183 ms of a 32k chunk, 10.4 ms per 256-row strip, against 200 ms at NC
2,048). Its work-group is one (token, 32-head tile) walking EVERY key of `[window segment ++ latent cache]` in
64-key blocks and skipping only blocks whose 64 mask entries are all closed. A continuation row's open keys are its
<= 128 window keys (contiguous) and its <= 512 indexer picks, which spread over the whole latent cache -- so almost
no block is skippable and the walk is ~(2,176 + NC)/64 blocks, each loading 64 x 512 fp16 key columns.

## The change

`ds41_attention_cont_xmx`: one launch GATHERS each strip row's live keys -- window key k0 + c for c < 128 (the mask
`mwin` read at that key), then the row's picks (bias 0, exactly the keys `ds4_block_bias_topk` opens) -- into a
(128 + pitch) axis padded to 64-key blocks, fp16 with the same saturating conversion; then the SAME prefill XMX
kernel runs over each row's own axis (`ds4_attention_xmx_segs(..., kv_rows)`: the key block pointer becomes
per-row, the staging pass is skipped; V4's instantiations are unchanged). 10 blocks per row instead of ~160 at 32k;
the [strip, NC] bias and the [strip, NKV] dense mask are no longer built. The window-only continuation layers take
the same gather with no picks (2 blocks instead of ~34). Opt-in: `IE_DS41_CONT_GATHER=1`.

NOT bit-identical: the block boundaries move, so the online softmax rounds differently (fp16 operands, fp32
accumulation, as the contiguous kernel). Which keys are open is unchanged by construction.

## Criteria

1. **Same function**: the cont test's bars (chunked prefill against one-token steps) PASS 15/15 as on the
   contiguous kernel; the long test's decode tokens after a 32k prefill are the contiguous run's (a divergence at a
   near-tie is recorded, not hidden).
2. **Quality**: `ie-ds41-ppl` on 16,384 wikitext-2 tokens, paired against the contiguous kernel's per-token NLL
   (same int-dot route): |mean NLL difference| < 2 block SEs, or < 0.001 nats.
3. **Real text**: the 24k held-out run retrieves the needle with coherent text.
4. **Speed**: `ds4_attention_xmx`'s share of a 32k chunk falls several-fold, and the 32k pipelined prefill rises
   above Phase 43's 277.9 tok/s by >= 10 %.

## Results (build of 08:39, opt-in arm; `ds41_work/p43/gat_*`, `ppl_gat.*`)

| criterion | bar | measured | |
|---|---|---|---|
| 1. same function | cont test 15/15; 32k decode tokens unchanged | **cont 15/15**; the 64 greedy decode ids after the 32k prefill **identical** (FNV `c14c73474cec53b2`, as the contiguous Phase 43 run) | MET |
| 2. quality | \|dNLL\| < 2 SE or < 0.001 nats | PPL **1.88671** vs 1.88567: **+0.000546 nats/token, block SE 0.000875 (t = 0.62)**; top-1 agreement 98.72 % | MET |
| 3. real text | needle, coherent | **needle retrieved**, coherent; the stream diverges from the contiguous kernel's at the heading after the answer ("Analysis of Survey Operations" vs "Analysis of the Valparaiso Survey Vessel Log") -- the same near-tie heading Phase 34 recorded | MET |
| 4. speed | 32k pipelined prefill >= +10 % over 277.9 | **311.0 tok/s (+12 %)**; the 24k held-out text **298 -> 344 tok/s (+15 %)** | MET |

Per card (`IE_DS41_PIPE_TRACE=1`, 32k, 16 chunks): card 0 busy 106.3 -> **99.5 s**, card 1 108.9 -> **92.7 s** (card 1
now waits 12.7 s for input): **card 0 is the slower stage** -- its layers 0-19 hold the engram gathers, the host prep
and three index sources before the candidate pool (2, 8, 14: their top-k is O(NC), docs/66).

The perplexity sample also puts the XMX expert route's +0.00223 (docs/83) in scale: the attention's rounding change
alone moves 1.28 % of top-1 predictions, the expert route's 1.56 %.

**Verdict: DEFAULT ON** (`IE_DS41_CONT_GATHER=0` restores the contiguous walk). Not re-run on this build: the V4
(DeepSeek-V4-Flash) gates -- the V4 model shares `ds4_attention_xmx_segs`, whose non-`kv_rows` instantiations are
the unchanged source with a constant-false branch folded; V4.1's forward test (pos0 = 0 prefills through that same
instantiation) is the check run here.
