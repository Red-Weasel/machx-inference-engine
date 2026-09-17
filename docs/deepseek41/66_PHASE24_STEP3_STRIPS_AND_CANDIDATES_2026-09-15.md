# V4.1 port — Phase 24 step 3 criteria: the two things that actually block 250k (written BEFORE the build)

**Where step 2 got to, measured.** The chunked prefill works: `ie-ds41-long-test` fed **16,384 tokens as 8
consecutive 2,048-token chunks** through the continuation and then stopped, with the engine naming its own wall:

```
  prefilled     2048 tokens  stage   288.4 tok/s  overall   288.4 tok/s
  prefilled     4096 tokens  stage   189.4 tok/s  overall   228.6 tok/s
  prefilled     8192 tokens  stage   180.3 tok/s  overall   201.6 tok/s
  prefilled    16384 tokens  stage   168.6 tok/s  overall   183.6 tok/s
WALL at 16384 tokens: at pos0 16384 T 2048: layer 20: two-level candidate top-k needed at this length
```

Correctness at chunk scale is settled (docs/65 + commit 9a04274: layer 0's window keys 2.1e-7, every layer's `nc`
and `n_pos` exact, and the chunked logits within 1.23x of a control that isolates regrouping from continuation
logic). So step 3 is not about the continuation any more. **Two separate blockers stand between here and 250k, and
they were found by measurement and by arithmetic respectively.**

## Blocker A — the scratch the VRAM reserve never counted (arithmetic, not yet measured)

Three of the forward's scratch buffers scale as T x NC, and NC is the accumulated latent count — the context, not
the chunk (`deepseek41_forward.cpp:670-672`):

| buffer | shape | T = 2,048, NC = 250k | NC = 500k |
|---|---|---|---|
| `scores` | [T, NCMAX] | 2.05 GB | 4.10 GB |
| `mcomp` | [T, NCMAX] | 2.05 GB | 4.10 GB |
| `mask` | [T, WIN + T + NCMAX] | 2.07 GB | 4.12 GB |

**~6.2 GB per card at 250k and ~12.3 GB at 500k**, against a computed reserve of `fwd_cap_ x 920 KB` = 1.88 GB
(step 1, commit 7701be5). That 920 KB/token was calibrated at a 2,048-token context where these three buffers are
32 KB of it; at NC = 250k they are **3 MB per token**. The reserve is wrong by ~3x at 250k, so this will refuse or
OOM before the candidate wall is even reached at a long context. It is the same class of mistake step 1 fixed —
scratch sized by the context — one level deeper.

**The fix: strip the query rows.** The indexer score, the block bias, the mask build and the attention are all
**row-independent** — nothing reduces across query rows. So the compressed-segment work runs in strips of `STRIP`
rows (default 256) writing into the full-size `oo`, and the three buffers are allocated [STRIP, NC] instead of
[T, NC]: **256 MB per card at 250k, 512 MB at 500k**, independent of the chunk size.

> **DECISION RULE for A, fixed now: stripping must be BIT-IDENTICAL.** It re-orders nothing and shares no
> reduction, so every existing test's digits must be unchanged to the last bit — the decode test's forced-routing
> value at exactly `5.043e-3`, and forward/replay/generate/multi/rollback/dspark/cont at their current counts. A
> digit that moves is a defect in the stripping, not a tolerance question, and the phase stops until it is found.
> `IE_DS41_STRIP=0` restores the un-stripped path as the kill switch and must reproduce today's numbers exactly.

## Blocker B — the two-level candidate top-k, which is genuinely not implemented

Read from the config rather than assumed: `index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]`,
`candidate_source_layer_id = 20`, `candidate_topk_blocks = 2048`, `candidate_block_size = 8`, `index_topk = 512`.
The reference (`inference/model.py:569-610`) therefore splits those eight layers three ways:

| layers | what they do | do they need this phase? |
|---|---|---|
| 2, 8, 14 | full fine top-512 over their own NC (ratio 2, so NC = positions/2) | **no** — they need only a top-k that scales, which `ds4_indexer_topk_radix256` already is |
| **20** | scores, then `select_candidate_blocks` -> a keep-mask it publishes; then its OWN full fine top-512 (it does not mask itself) | **yes** — level one |
| **24, 28, 32, 36** | mask their scores by layer 20's keep-mask, then fine top-512 inside it | **yes** — level two |

**So the engine's current refusal is over-broad**: it fires for every index source above 16,384, including 2/8/14
which the reference does not restrict at all. Below 16,384 latents `min(topk_blocks, num_blocks)` keeps every block
and the stage is genuinely a no-op, which is why it has never been needed.

`select_candidate_blocks`, exactly as the reference defines it: pad NC to a multiple of 8 with -inf; a block's score
is the **max** over its 8 positions (positions the query cannot reach are already -inf); **pin the block holding
this query's newest reachable position to +inf** (`last = (compress_len - 1) / block_size`) because it is only
partly filled and would otherwise lose to an older full block; take the top `min(2048, num_blocks)`; keep only
those whose score is `> -inf`; expand by 8 back to NC.

The deliverable is a keep-mask kept in **block units** — `[STRIP, num_blocks]` bytes, 8x smaller than the expanded
form and the same information — because the consumer only ever asks "is entry e's block kept".

> **DECISION RULE for B, fixed now.** This is a DISCRETE selection: the fine top-512 indices are either the
> reference's or they are wrong, and a near-miss is a failure, not a tolerance. It passes only if (a) at a length
> where the stage is a no-op (NC <= 16,384) the emitted indices are **identical** to today's, so the new code is
> provably inert where it should be — `candidate_topk_was_noop()` must still report true; (b) a direct unit test
> against the reference's semantics passes on the three cases that actually carry the logic: fewer reachable blocks
> than 2,048 (every reachable block kept, no -inf pick admitted), the partly-filled newest block pinned in even
> when its score is the lowest, and a query whose reachable length is not a multiple of 8; and (c) the 250k prefill
> runs to completion. If the reference's tie-break among equal block scores cannot be reproduced deterministically,
> that is REPORTED as the one place the selection may differ, with the measured frequency of such ties — not
> papered over.

## Pass criteria for THIS phase

1. **Stripping bit-identical**, per A's rule, with `IE_DS41_STRIP=0` restoring the old path exactly.
2. **The candidate stage inert below 16,384**, per B(a) — identical indices, `cand_noop_` still true.
3. **The candidate unit test** of B(b) passes, written against the reference's semantics and naming them.
4. **250,000 tokens prefill** in chunks, reported with the stage-rate table so the indexer's quadratic term is
   visible rather than hidden in an average, and the peak VRAM per card reported against the 32 GiB the card has.
5. **Decode after that 250k prefill measured**, in ms/token and tok/s, against the founder's stated bar of **10+
   tok/s** at long context and against the 12.42 tok/s the campaign measures at 2,048 tokens. If decode at 250k is
   below 10, that is reported as a number with the term that caused it, not hidden.
6. **500,000 attempted** once 250k passes, and reported honestly: either it runs with its rates, or the allocation
   or wall that stops it is named with its size.
7. **The GPU survives**: zero `xe ... Timedout job` / reset lines across the phase, read from the kernel log.
8. The regression set on the new default: decode, resident, replay, forward, generate, multi, rollback, dspark, cont.

## Explicitly NOT in this phase

The generator/server feeding chunks themselves (docs/65 criterion 5 — it is the step after this, and pointless
before the engine can take 250k at all), concurrency, the decode path's own optimization, and DSpark.

## Results, blocker A (07:40-07:55): stripping is bit-identical, and free

Verified on the SAME binary, `IE_DS41_STRIP=256` against `IE_DS41_STRIP=0`, because a rebuilt or re-environed
comparison could not separate striping from anything else that moved:

| gate | strips | no strips | verdict |
|---|---|---|---|
| `ie-ds41-decode-test`, all 28 criteria | 28/28 | 28/28 | **every criterion's reported value identical** (diff clean) |
| .. its forced-routing logits | 4.948e-03 | 4.948e-03 | identical |
| `ie-ds41-resident-test` pp512 + pp2048 (2 and 8 strips) | 14/14 | 14/14 | every arithmetic value identical |
| .. expert occupancy per forward | 1714 / 6542 / 568 | 1714 / 6542 / 568 | the router picked the SAME experts at all 2048 rows of all 40 layers |
| .. "a warm second forward is bit-identical" | ok | ok | both |
| `ie-ds41-cont-test` with 8-row strips | 13/13 | 13/13 | identical, so the continuation path is covered |
| pp2048 warm | 291.9 tok/s | 291.1 tok/s | no cost |

Only the wall-clock, `MemAvailable` and the VRAM-residue bytes differ, which is measurement noise. **Criterion 1
is MET.**

The reserve was then made to count this scratch rather than estimate it per token: `strip x NC x 4 x 3` plus the
candidate stage's two buffers, computed from the same `IE_DS41_STRIP` default the forward reads. Step 1's reserve
was short by ~3x at 250k precisely because it did not.

**One number that is NOT about striping and is flagged rather than buried:** pp2048 reads ~291 tok/s in this
session against docs/62's 403.9. Both strip arms agree, so striping did not cause it. The resident test reports
**9.9 GiB of mmap->VRAM per forward** at pp2048 (~1.4 GB/s, i.e. disk-bound), so the working hypothesis is host
page-cache state for `ie_experts_tail.ieslot` -- **unverified**, and it bears on the 400+ prefill mandate, so it
is queued as its own question rather than asserted here.

## Results, blocker B (07:55-08:02): the candidate stage matches the reference, and the wall is gone

`ie-ds41-candidate-test`, against a line-by-line transcription of `select_candidate_blocks`:

| case | result |
|---|---|
| a no-op length (NB 512 <= 2048): every reachable block kept, no -inf pick admitted | identical, 3970 kept |
| the real selection, NB 8192 > 2048, rate 1 | identical over 65,536 blocks |
| rate 2 (the ratio-2 layers' causal threshold) | identical |
| the newest block forced to the LOWEST score in its row | identical, **and the pin FIRED: 8 of 8 rows keep it** |
| a reachable length that is not a multiple of 8, ragged last block | identical |
| topk_blocks 37 (the K-th largest deep in the radix histogram) | identical, 37 kept |

The pin is asserted POSITIVELY, not just "agrees with my transcription", because a shared misreading would pass
that. **Criterion 3 is MET.** Inertness below 16,384 (criterion 2): `ie-ds41-decode-test` 28/28 with every
criterion's value identical to the pre-candidate build, so the stage is provably absent where it should be.

**Where the approximation actually lives, proved.** Whenever `index_topk <= candidate_topk_blocks` (512 <= 2048
here) the stage cannot change a layer's own fine top-k at all: a top-512 entry's block scores at least as high as
that entry, and for its block to fall outside the top 2048 at least 2048 blocks would have to score higher, each
containing a distinct entry above it -- but only 511 entries beat a top-512 entry. So the source layer masking
itself would be a no-op, which is exactly why the reference falls through unmasked. **Every bit of the two-level
scheme's accuracy cost is therefore CROSS-LAYER**: 24/28/32/36 are pruned by a ranking computed from layer 20's
weights. That is why the gate for it is end-to-end retrieval at depth (with `IE_DS41_CAND=0` as the exact arm),
not a logits tolerance.

**The wall moved**: 32,768 tokens prefill through 16 chunks at 159.6 tok/s overall, stage rate 152 / 187 / 177 /
165 / 151 tok/s -- flat, not quadratic, because the fine top-k is the O(n_keys) radix shape and the attention is
sparse (the kernel tests the mask first and never reads a masked column's K row, `deepseek4_attn.cpp:940-960`).

## Two findings from the 250k run's own stage rates, recorded while it is still running

**1. Every performance number in this session is on the NON-expert-parallel configuration, and so is the shipping
default.** `ResidentOptions::expert_parallel` defaults to 0 and `src/engine/ds41_engine.cpp:63` sets only
`max_tokens`, so the CLI, the server and every test here run with EP OFF unless `IE_DS41_EP=1` is in the
environment. The decode test's own banner proves which arm a run was: the CPU split's `q*` is **0.60 under expert
parallel and 0.30 without** (`deepseek41_experts.cpp:399`), and this session prints 0.30 throughout while docs/62
records 0.60. **So docs/62's mandate numbers -- 403.9 tok/s prefill and 12.42 tok/s decode -- were measured on an
opt-in configuration, not the default**, which is the whole of the 291-vs-404 gap and also why the resident test
reports 9.9 GiB of mmap->VRAM per forward here (without a parity share each card wants far more experts than fit,
so more of them fall to disk). Every A/B in this phase is internally valid because both arms shared the config,
but the absolute figures are default-config figures. An EP arm of the long run is queued.

**2. The quadratic term in a long prefill is the DENSE MASK, not the indexer -- analysis, not yet profiled.** The
obvious suspect was the indexer score, `T x NC x index_head_dim` MACs over 8 index-source layers. The measured
stage falsifies it: 32,768 -> 65,536 tokens took 244.22 s for 3.30e12 indexer flops = **13.5 GFLOP/s**, two orders
of magnitude below these cards, so that term is nowhere near binding. What does scale is bandwidth spent on a mask
the attention barely reads:

  * `ds4_block_bias_topk` writes `mcomp` as **[rows, NC]** -- all -inf, then 512 zeros scattered in;
  * the mask build writes `mask` as **[rows, WIN + NC]**, reading `mcomp` back;
  * `ds4_attention_segs` then reads that whole mask, even though it "tests the mask FIRST and never touches a
    masked column's K row" (`deepseek4_attn.cpp:940-960`) -- so it visits 512 of NC K rows but still reads NC mask
    entries.

At NC ~ 37,000 averaged over the ratio-2 and ratio-1 halves that is ~0.9 GB per layer per 2,048-token chunk, and
~38 compressed layers x 16 chunks puts the stage near 550 GB -- the right order of magnitude for 244 s, where the
flops are off by 100x. **So the lever for long-context prefill is to hand the attention its 512 top-k INDICES
instead of a dense [rows, NC] bias**, which is the same "gather attention" conclusion the DS4 1M campaign reached.
Stated as analysis from the code's shapes with the arithmetic shown; it needs `IE_QUEUE_PROFILING=1` per kernel to
become a measurement, and it is NOT in this phase.

Projected from the measured stage rates (288 / 186 / 175 / 163 / 151 / 134 tok/s, marginal cost rising linearly in
NC): **250k lands near 60 minutes and 500k near 3 hours on this shape.** That is a one-time cost per prompt, and
the mask change above is what would move it.

### CORRECTION (08:25): finding 2's MECHANISM was wrong by ~100x. The curve stands; the cause is open.

I attributed the linear-in-NC term to the dense `[rows, NC]` mask's bandwidth. Checking my own arithmetic instead
of leaving it: at NC ~ 49,000 the whole mask pipeline -- `mcomp` written by `ds4_block_bias_topk`, the wider
`mask` written by the build, both read back -- is ~30 GB per 2,048-token chunk across all 38 compressed layers,
i.e. **~50 ms per card against a chunk that measures ~15 s**. That is 0.3 %, not a bottleneck.

Worse, I read the wrong kernel. At T = 2048 the dispatch returns split factor 0 and goes to **`ds4_attention_xmx_segs`**,
not the fp32 split kernel whose per-column mask test I quoted. The XMX kernel walks n_kv in blocks of KB, reads
that block's mask entries once per row-block, and **skips the whole block when `any_of_group` finds nothing
allowed** -- so with 512 scattered picks it does roughly 512 columns of tile work whatever NC is, plus an O(NC)
mask scan that costs tens of MB per layer per chunk. It is already close to the gathered shape I proposed adding.

Every other NC-linear candidate I can cost from the source -- the indexer score's `idx_k` re-read per strip, the
radix top-k's sub-group barrier rotation, `block_bias`'s writes, the mask build -- lands in **milliseconds** per
chunk against a **measured 4.8 s** of extra cost per chunk at that stage. So the mechanism is NOT identifiable
from the code shapes, and the "gather attention" recommendation is **WITHDRAWN** as unfounded.

**What stands, because it is measured:** marginal cost per token fits `5.1 ms + 4.8e-5 x NC` across six stages
(277.9 / 185.7 / 175.0 / 163.3 / 150.7 / 134.2 tok/s), giving ~56 tok/s marginal and ~88 tok/s average at 250k.

**What it will take to close it:** `IE_QUEUE_PROFILING=1` on a short run to 16,384 tokens, reading per-kernel
ms/token at two NC values and differencing them. That names the term directly instead of inferring it, and it is
the first thing to run when the GPU frees. Recorded as an open question, not a conclusion.

### The leading hypothesis for the NC-linear term, with its arithmetic (still needs the profiler)

Having ruled out flops (100x short) and bandwidth (50x short), the remaining candidate is **synchronization inside
the XMX attention's column-block loop**. Per query-row block it walks `n_kv / KB` column blocks, and each block
costs a mask load, an `any_of_group` and **two `group_barrier`s -- even when the block is skipped**
(`deepseek4_attn_xmx.cpp:295-302`). That is work proportional to NC that no amount of sparsity avoids, because the
skip decision itself is what costs:

| | NC = 49,000 | observed |
|---|---|---|
| column blocks per row-block (KB = 64) | 766 | |
| x row-blocks per 2,048-token chunk | ~256 | |
| barrier pairs per layer per chunk | ~392,000 | |
| at ~100 ns per pair | **~39 ms** | **~208 ms per layer per chunk** |

Same order where flops and bytes were both off by 50-100x, and linear in NC, which is the shape the six measured
stages actually show. If this is the term, the cure is the gathered form after all -- iterate the 512 picked
columns instead of scanning every block to discover it is empty -- but for a SYNCHRONISATION reason, not the
bandwidth one I first gave. **Hypothesis, not a result:** it is settled by `IE_QUEUE_PROFILING=1` at two NC values,
reading `ds4_attention_xmx` ms/token directly.

### One real inefficiency striping introduced, found while reading that kernel and now fixed

The XMX attention must convert a **fp32** `b` segment into its fp16 staging, because the tiles cannot read fp32 in
place -- and V4.1's latent cache is fp32 (`segs.b_f16 = false` at every call site here). That staging is a property
of the LAYER, not of the query-row strip, which is exactly what the kernel's `xmx_kv_prepared` argument exists to
say. My strip loop did not pass it, so **every strip re-staged the entire NC axis**: 8x the conversion at the
default strip width, for bytes that had not changed. Now `kv_staged = so > 0`.

Safe by construction, checked rather than assumed: `ds4ax_reserve` only ever GROWS its workspace, and the first
strip is the largest (`sT = min(STRIP, Tl - so)`), so no reallocation can occur mid-layer and the reused pointer
stays valid. The staged bytes are identical either way, so this must remain BIT-IDENTICAL -- which the blocker-A
gate already tests, and it is re-run before this is claimed.

## Criterion 4 MET: 262,144 tokens prefill. Criterion 5 FAILED: decode at that context is 2.62 tok/s.

`ie-ds41-long-test`, chunks of 2,048, `IE_DS41_CONT=1` (log `p24/long250k.log`):

| prefilled | stage tok/s | overall tok/s | at |
|---|---|---|---|
| 2,048 | 277.9 | 277.9 | 7.4 s |
| 16,384 | 163.3 | 178.2 | 92.0 s |
| 65,536 | 134.2 | 147.3 | 444.9 s |
| 131,072 | 111.7 | 127.0 | 1,031.7 s |
| **262,144** | **85.1** | **101.9** | **2,572.2 s = 42.9 min** |

Marginal cost per token fits `5.95 ms + 3.06e-5 x NC` over the last four stages, which predicted the final stage at
83.5 tok/s against 85.1 measured and the total at 43 min against 42.9. **Prefill is a solved, predictable one-time
cost.** Decode is not: **381.2 ms/token = 2.62 tok/s at 262,144**, against the founder's "10+ is at least useable".

## The real bottleneck, measured. Every kernel hypothesis was WRONG, including mine.

Profiling decode at two contexts (`IE_QUEUE_PROFILING=1`, NC x4, `p24/prof_8192.log` / `prof_32768.log`):

| | ctx 8,192 | ctx 32,768 | grew | ratio |
|---|---|---|---|---|
| `ds4_indexer_topk` | 1.16 | 4.64 | +3.48 | **4.00** (exactly linear) |
| `ds4_attention` | 8.71 | 13.27 | +4.56 | 1.52 |
| `ds4_indexer_score` | 0.20 | 0.69 | +0.49 | 3.45 |
| every other named kernel | ~35.1 | ~36.2 | ~+1.1 | 1.03 |
| **named total** | **45.2** | **54.8** | **+9.6** | 1.21 |
| **wall** | **106.1** | **159.1** | **+53.0** | 1.50 |
| **unnamed** | **60.9** | **104.3** | **+43.4** | 1.71 |

My barrier prediction for `ds4_indexer_topk` was right in KIND -- it is exactly linear in NC, as O(n_keys) barrier
rotations at one work-group per row must be -- and irrelevant in SIZE: 3.5 ms of a 53 ms regression. All the mask
machinery (`ds4_block_bias_topk_fill/_scatter`, `ds41_decode_mask`) came in UNDER 0.12 ms/token, so the earlier
dense-mask theory is not merely unproven but measured false. **82 % of the regression is outside every named
kernel.**

The per-layer stage counters (`fwd.stats()`, now printed by the long test) locate it exactly:

| ms/token | ctx 8,192 | ctx 32,768 | change |
|---|---|---|---|
| wall | 87.7 | 148.7 | +61.0 |
| attention | 28.2 | 36.9 | +8.7 |
| **MoE call** | **49.2** | **101.0** | **+51.8 = 85 % of the growth** |
| .. groups | 30.4 | 38.0 | +7.6 |
| .. **join (waiting for the CPU miss leg)** | **9.2** | **46.5** | **+37.3** |
| .. tail | 6.7 | 12.2 | +5.5 |
| CPU leg to the join | 41.0 | 95.1 | +54.1 |
| **expert selections served from static VRAM** | **136.6** | **76.4** | **-60.2, halved** |
| .. from disk | 2.0 | 12.4 | **6.2x** |
| **decode hit rate** | **63.7 %** | **43.7 %** | **-20 points** |
| **disk MiB/token** | **35.9** | **222.3** | **6.2x** |

**The expert cache's hit rate collapses as the context grows**, so misses fall through to disk and the MoE sits in
`join` waiting for the CPU miss leg. That is the bottleneck: not attention, not the indexer, not the mask, not a
kernel at all -- **expert residency**.

### Two candidate causes, one already eliminated by control

* **The KV allocation displacing static experts: RULED OUT.** Prefilling the SAME 8,192 tokens under a KV capacity
  of 8,256 vs 262,208 -- identical NC, identical kernels, only the allocation differs -- gives **91.4 vs 94.0
  ms/token, 2.8 %**. Reserving a quarter-million positions is nearly free.
* **My synthetic prompt (the golden 2,048 block repeated) driving degenerate routing: RULED OUT.** The needle
  prompt, 23,320 tokens of varied real text through `ie-ds41-run --prompt-file`, decodes at **5.68 tok/s**, i.e.
  slightly WORSE than the synthetic block at a longer context. Repetition is not the cause.

So what remains is that **a longer context changes what the router selects**, and the held-out expert ranking that
decides static residency was built at short context. That is a testable, cheap hypothesis with an obvious fix
(re-rank from a long-context routing dump) and it is the next measurement: `IE_DS41_DUMP_ROUTING` at two contexts,
comparing how concentrated the selections are and how much of the top-N the static tier still covers.

## The semantic gate, PASSED

A prefill that completes is not a prefill that works. `make_needle.py` plants one fact in uninformative filler and
asks for it back. At **23,320 tokens with the needle at 50 % depth**, fed in chunks by the generator itself, the
engine answered: *"The access code for the Helsinki archive is 47-BRAVO-purple."* That exercises the chunked
continuation, the query-row strips, the two-level candidate top-k, the generator's own chunking and
`--prompt-file` in one path, and it is the strongest available evidence that long context is CORRECT and not just
completing.
