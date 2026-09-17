# V4.1 — Phase 33: a rank-strided pick partition (criteria before the build)

## Why

Phase 32 made the gathered attention's pick partition T-independent by handing the kernel the row PITCH (512) as
`n_picks`. The contiguous partition then gives every slice `512 / SP = 8` list positions, so below NC = 512 only
`n_valid / 8` of the 64 slices carry any segment-b work: the 12-token prompt's `ds4_attention` went 88.6 -> 145.6 ms
over the decode test's 32 steps (+1.8 ms/token). Unchanged at 2k and above, where every slice has 8 picks anyway.

A RANK-STRIDED partition -- slice `s` takes list positions `s, s + SP, s + 2 SP, ...` -- is T-independent for the
same reason (the valid picks sit at `[0, n_valid)` with the sentinels after them, so position == rank), spreads a
short row over `min(n_valid, SP)` slices, and does the same 8 iterations per slice at 512 picks. It regroups the
per-slice partials, so like Phase 30 it is judged on the gates, not on bit-equality with the contiguous form.

`IE_DS4_ATTN_PICKPART`: 0 = stride by column (Phase 26), 1 = contiguous (Phase 30), 2 = rank-strided (this).

## Criteria

1. Gates on the build with mode 2 the default: `multi` 38/38, `rollback` 74/74, `dspark` 153/153, decode 28/28.
2. Short context: the 12-token prompt's `ds4_attention` back within noise of the pre-Phase-32 88.6 ms / 32 steps.
3. Long context: the 2k prompt's `ds4_attention` within noise of 212.8 ms / 32 steps (no regression where it matters).
4. Behaviour: 120 real decode tokens at 24k of held-out text produce identical output text with mode 2 and mode 1.

If 2 fails to recover, or 3 regresses beyond noise, mode 1 stays the default and this doc records the numbers.

## Result of the rank-strided form: criteria 1-3 met, criterion 4 FAILED -- not the default

| | contiguous (mode 1) | rank-strided (mode 2) |
|---|---|---|
| decode test | 28/28 | 28/28 |
| multi / rollback / dspark | 38 / 74 / 153 | 38 / 74 / 153 |
| 12-token prompt `ds4_attention`, 32 steps | 145.6 ms | **90.4** (pre-Phase-32: 88.6) |
| 2k prompt `ds4_attention`, 32 steps | 212.8 ms | 199.2 (one run) |
| 24k held-out text, 120 tokens | 9.73 tok/s | 9.39 (one run each) |
| output text at 24k | — | **diverges after 91 words** ("coring" vs "handling", a greedy near-tie) |

The regrouped partials move a near-tie at 24k, so "identical text" is not met and the criterion is not being
re-argued after the fact. Mode 2 stays as `IE_DS4_ATTN_PICKPART=2` with these numbers; nothing else changes.

## The second form: contiguous, sized by the row's VALID COUNT (criteria before the build)

The sort knows each row's valid count (the sorted row is ascending with the sentinels last, so it is one binary
search per row). Hand it to the kernel and let the contiguous run be `per = ceil(n_valid / SP)`:

- **T-independent**, because `n_valid` is a property of the row (its causal reach), not of the step.
- **Bit-identical at T = 1 to the Phase 30 form** (the partition before the Phase 32 pitch change), because at a
  one-row step k == n_valid (no sentinels: NC == thr at every decode position for ratio 1 and ratio 2 alike), so
  `per = ceil(k / SP)` is what Phase 30 computed. It equals the PITCH-sized form only where n_valid > 448 (`per`
  = 8 either way), i.e. at every context >= 1k -- below that the two differ, which is the point. (The first draft
  of this criterion said "bit-identical to mode 1 everywhere"; that was wrong, and the decode test showed it.)
- At NC < 512 a short row is spread over `n_valid` slices again (`per` = 1), the pre-Phase-32 layout.

Criteria: (1) the four gates as above; (2) the 12-token prompt's `ds4_attention` back within noise of 88.6 ms;
(3) the 2k prompt's `ds4_attention` at 212.8 ms, its decode-test criterion lines bit-for-bit the pitch-sized
run's, and the 12-token prompt's bit-for-bit the Phase 30 run's (`oa_fp8.log`); (4) the 24k text IDENTICAL to
mode 1's `pp33_text_1.log` -- expected by construction, verified by run.

## Result of the count-sized form: every criterion met -- it is the default

| | pitch-sized (Phase 32) | rank-strided (mode 2) | **count-sized (default now)** |
|---|---|---|---|
| decode / multi / rollback / dspark | 28 / 38 / 74 / 153 | 28 / 38 / 74 / 153 | **28 / 38 / 74 / 153** |
| 12-token prompt `ds4_attention`, 32 steps | 145.6 ms | 90.4 | **89.1** (Phase 30: 88.6) |
| 2k prompt `ds4_attention`, 32 steps | 212.8 ms | 199.2 | 214.2 (noise; same partition) |
| decode-test criterion lines, 12-token prompt | — | — | bit-for-bit the Phase 30 run's (`oa_fp8.log`) |
| decode-test criterion lines, 2k prompt | — | — | bit-for-bit the pitch-sized run's (`oa_final.log`) |
| 24k held-out text, 120 tokens | 9.73 tok/s | 9.39, text diverges | 9.49, **text identical** to the pitch-sized arm |

So the short-context cost Phase 32 introduced is gone (+1.8 -> +0.0 ms/token at a 12-token prompt), nothing
above 1k context changed by a bit, and the T-row step still equals its one-row steps. The 2k prompt's wall in
these profiled runs moved 86.1 -> 91.1 ms/token between the pitch-sized and count-sized runs with the named
kernels at 40.0 vs 40.1 -- expert-traffic noise between runs (the 12-token prompt's wall swings 166 -> 246 the
same way), not the kernel, which is bit-identical there.

Mechanics: `ds41_sort_picks_asc` writes each row's valid count (`n_valid[t]`, one binary search by lane 0 over
the sorted row), `Ds4KvSegs::b_pick_n` carries it, and the contiguous partition uses `per = ceil(n_valid / SP)`.
`IE_DS4_ATTN_PICKPART`: 0 stride-by-column, 1 (default) contiguous by count, 2 rank-strided (opt-in, recorded above).
