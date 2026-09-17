# V4.1 port — Phase 26 criteria: the decode attention's masked-column scan (written BEFORE the build)

**The term, measured in Phase 25.** With the residency ranking fixed, decode at 262,144 tokens reads 232.5
ms/token and **attention is now the largest single term at 121.3 ms/token**, up from 35.3 at ctx 32,768. It fits
`26 + 3.4e-4 x NC` ms/token across three measured contexts (8,192: 28.2; 32,768: 35.3; 262,144: 121.3, predicted
115), so ~90 ms of it at 200k+ is the term that scales with the latent cache.

**Where it goes.** At decode `T = 1` the dispatch takes `ds4_attention_split<64>`
(`ds4_attn_split_factor`: `T <= 1 -> sp = 64`). Each of the 64 sub-groups walks a **stride-SP slice** of the KV
axis, `for (i = s; i < n_kv; i += SP)`, and for every column all **16 lanes of the sub-group redundantly read the
same mask float** and branch on it (`deepseek4_attn.cpp:727-731`). The indexer selects `index_topk = 512`
compressed entries precisely so attention need not look at the rest, and the kernel does skip a masked column
before touching its K row — but it still pays **one full sub-group iteration per masked column**. At
`n_kv = WIN + NC = 262,272` that is 262,272 iterations per (query, head), times 64 heads and 38 compressed layers,
to reach the ~640 columns that are actually live.

## The change, and why it is bit-identical BY CONSTRUCTION

Ballot 16 of the sub-group's OWN columns at a time and skip the block when none is live:

```
for (i = s; i < n_kv; i += SP * kSG) {
    ic   = i + lane * SP;                                  // this lane's candidate, still on the SP stride
    live = ic < n_kv && (!mask || mask[t * n_kv + ic] > kMaskedCut);
    if (!any_of_group(sg, live)) continue;                 // 16 of this slice's columns, all masked
    for (j = 0; j < kSG; ++j) { ii = i + j * SP; if (ii >= n_kv) break; <the existing body at i = ii> }
}
```

The column set each sub-group owns is unchanged — `s + (0..15) * SP`, `s + 16*SP + (0..15) * SP`, ... is exactly
`s, s+SP, s+2SP, ...` — and the inner loop visits it in the same ASCENDING order, so every online-softmax state
update happens on the same column in the same sequence with the same operands. The inner body stays
**lane-uniform** (it reads the mask at `ii`, one address for all lanes, exactly as today), which is what
`reduce_over_group` requires. Nothing about the arithmetic moves.

> **DECISION RULE, fixed now.** This is a pure skip-granularity change, so it must be **BIT-IDENTICAL**: the decode
> test's 28 criteria with the forced-routing logits at exactly today's `4.948e-03`, and every reported value
> identical to the pre-change build, plus forward / replay / generate / multi / rollback / dspark / cont / candidate
> at their current counts. A digit that moves means the column set or its order moved, which is a defect, not a
> tolerance question — and it keeps the kill switch honest. It is KEPT only if decode at ctx 32,768 improves by
> **>= 5 %** on >= 2 samples; if it is inside the noise at 32,768 it is re-measured at 262,144 before being judged,
> because the term it targets is only ~9 ms of 83 at 32,768 but ~90 ms of 232 at 262,144. `IE_DS4_ATTN_BALLOT=0`
> restores today's per-column scan exactly.

## Pass criteria for THIS phase

1. **Bit-identity**, per the rule above, verified against the same binary with the kill switch set.
2. **Decode measured at ctx 32,768 AND 262,144**, >= 2 samples each at 32,768, with the stage breakdown so the
   saving is visible in the `attn` line specifically and not just the wall.
3. **Prefill unaffected**: pp2048 within 2 %, and the long test's stage rates unchanged — the prefill path takes
   the XMX kernel, not this one, so a change here should be invisible to it. If prefill moves, something else did.
4. **The arithmetic checked against the measurement**: the model says attention costs `26 + 3.4e-4 x NC`; report
   what the ballot does to the `3.4e-4` coefficient, since that coefficient IS the thing being fixed.
5. **The GPU survives**: zero `xe ... Timedout job` / reset lines.

## Explicitly NOT in this phase

A gathered attention that takes the indexer's 512 indices directly and drops the dense mask altogether — that is
the larger win (it removes the O(NC) mask READ as well as the iteration) but it needs the picks in ascending
order, which `ds4_indexer_topk` does not emit, so it is a bigger change with a sort in it. This phase takes the
16x that needs no new kernel and no new ordering argument, and measures what is left.

## Results, part 1: the BALLOT is FALSIFIED on its own bar, and the reason is structural

At ctx 32,768 with a decode-phase ranking, 512 decode steps: **87.5 ms/token with the ballot against 87.1
without** — inside the noise against the 5 % bar this phase fixed beforehand. It is bit-identical (28/28, every
reported value equal with it on and off), so the change was sound; it simply does not pay.

The attention line DID move, 37.6 -> 35.2 ms/token, but that is 2.4 ms of an 11.1 ms NC-linear term rather than
the 16x the iteration count suggested, and the mechanism is worth recording because it would mislead anyone
repeating the idea: **the sub-group's columns are on the SP STRIDE**, so a lane-parallel test reads kSG addresses
`SP * 4` = 256 B apart — **kSG separate cache lines, where the per-column form had all lanes broadcast from ONE.**
Iterations fall 16x and memory operations rise 16x, so only the loop overhead is recovered. Per the decision rule
it is **not kept**: the code is removed rather than left behind a flag, since a failed opt-in path is a
maintenance cost and this document is the durable record.

## Results, part 2: the GATHERED path, bit-identical, and why its win is at LONG context only

Deferred in this doc's own "NOT in this phase" section, then built once the ballot failed, because the ballot's
post-mortem showed the dense mask READ is the floor: the cure has to stop reading a dense mask at all.

`Ds4KvSegs` gains `b_picks` / `n_picks`; the split kernel walks segment a densely on its stride and then visits
segment b THROUGH THE PICK LIST, `O(n_picks)` instead of `O(n_b)`. Bit-identity has three parts, all structural:
every a column precedes every b column so the slice is still visited in ascending order; `b_picks` is required
ASCENDING (`ds41_sort_picks_asc`, a bitonic sort with no SLM atomics, because `ds4_indexer_topk` emits by
descending score and the online softmax is order-dependent); and the filter `i % SP == s` keeps each sub-group's
column set exactly what it was, so the per-slice partials the combine tree reduces do not move. The per-column
arithmetic was factored into ONE lambda shared by both paths so they cannot drift.

**Verified: 28/28 with every reported value identical to the dense scan.**

**A REAL BUG THIS CAUGHT, worth recording.** The first attempt passed `n_picks = index_topk` (512). But
`ds4_indexer_topk` emits `min(index_topk, n_keys)` picks, and at the decode test's 12-token prompt that is a
handful — so the kernel read stale picks from an earlier layer and the gate went to **23/5 with the worst layer at
9.4e-1**. The count must be `sh_topk_w`, and the SORT needs the row PITCH separately from the valid count for the
same reason. The gate caught it immediately; a tolerance-based bar might not have.

**Measured at ctx 32,768: 84.6 ms/token (attn 36.3) against the dense 87.1 (attn 37.6)** — again small, and the
arithmetic says why. Each sub-group scans the whole 512-entry list to find its own ~16, so the loop goes from
`n_kv / SP` = 1,028 iterations to 512: only **2x** at this length. At 262,144 the same comparison is 8,196 -> 512,
a **16x** reduction, which is why this phase's rule sends an inside-the-noise result at 32,768 to be re-measured
at 262,144 rather than judged. That run is in flight, carrying the 262k-specific ranking as well.

**The next refinement, named with its arithmetic rather than built on a hunch:** bucket the picks by slice so a
sub-group iterates only its own ~`n_picks / SP` = 16 entries instead of scanning all 512. That is a further ~32x
on this term, and it is only worth building if the 262,144 measurement shows the remaining scan still matters.

## Results, part 3: both fixes at 262,144 — 4.30 -> 5.80 tok/s, and the remaining budget, itemised

`ie-ds41-long-test` with the 262k-specific decode ranking AND the gathered attention, 512 decode steps:

| ms/token | 32k ranking + dense scan | 262k ranking + gather | saved |
|---|---|---|---|
| **wall** | **232.3** | **172.3** | **-60.0** |
| attention stage | 121.3 | **92.2** | -29.1 (the gather) |
| MoE call | 102.0 | **71.2** | -30.8 (the ranking) |
| .. join | 33.1 | **4.8** | -28.3 |
| .. tail | 37.3 | 37.0 | — |
| .. groups | 29.4 | 27.8 | -1.6 |
| hit rate | 66.7 % | **70.8 %** | |
| disk MiB/token | 45.6 | **6.2** | 7x less |
| **tok/s** | **4.30** | **5.80** | **1.35x** |

Prefill was untouched at 103.7 tok/s and its stage rates are identical to the pre-gather run (152.1 / 135.9 /
113.7 against 152.0 / 135.9 / 113.6), which is **criterion 3 satisfied**: the gather touches only the split kernel,
which the prefill path does not use. The ranking's contribution also matched the prediction made from the ranking
overlap BEFORE the run (69.8 % at top-22 -> "worth 20-30 ms"; it gave 31).

**Where the remaining time is, from a post-gather profile at 32,768 scaled on each kernel's measured NC slope.**
Projected at the founder's 200,000-token goal, against the 100 ms/token that 10 tok/s requires:

| term | ~ms/token at 200k | scales with NC? |
|---|---|---|
| `ds4_indexer_topk` | ~26 | **yes — measured EXACTLY 4.00x on NC x4** |
| MoE `tail` | ~28 | **yes — 8.6 -> 37.0 from 32k to 262k at an UNCHANGED 46-47 CPU experts/token, so its growth is unexplained** |
| attention kernel residual | ~21 | partly: each slice still scans all `n_picks` to find its own ~`n_picks / SP` |
| attention base (projections, RoPE, mask build) | ~26 | no |
| MoE `groups` | ~22 | no |
| dense GEMVs (`gemv_fp8`, `gemv_f16_rows`, `gemm_mxfp4`) | ~24 | no |
| **total** | **~147 = 6.8 tok/s** | |

So ~47 ms must come out of ~147 to reach the goal at 200k, and the three NC-linear items above are worth roughly
that much between them. **The MoE tail is ranked first for investigation, not because it is the largest but
because its scaling is UNEXPLAINED** — the CPU expert count does not move between those contexts, so something
else in that term does, and twice in this campaign (the ranking, and before it the whole residency chain) an
unexplained scaling has been where the real win was rather than in the term that merely looked biggest.

`ds4_indexer_topk` has a designed fix waiting: at the four CONSUMER layers the candidate stage has already
restricted the live set to `candidate_topk_blocks * candidate_block_size` = 16,384 entries, so their fine top-512
need not scan all NC. Gathering those 16,384 and selecting within them is ~16x less work on ~62 % of this term,
and it can be made BIT-IDENTICAL by teaching the top-k an optional original-index map so `pack_key`'s strict total
order is unchanged — the top 512 are candidates under either form, since non-candidates are at -inf and there are
16,384 candidates against 512 picks.
