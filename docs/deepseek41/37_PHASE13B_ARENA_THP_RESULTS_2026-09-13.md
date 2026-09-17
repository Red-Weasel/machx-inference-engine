# V4.1 port — Phase 13b results: the pinned arena on imported huge-page memory — reverted, the costs sink it (the CPU-leg mechanism question stays open)

**Criteria:** `36_PHASE13B_ARENA_THP_CRITERIA_2026-09-13.md`. **Starting line:** 6485afa (the block
time unconditional): decode at the 2,048-token context 158.3 ms/token with the CPU split off, 149.6
with it on at q* 0.30 (CPU 0.82 ms per expert), the resident load 38 s, the replayed 2,048-token
prefill 7.56 s.

## What was built

`Ds4HostArena::init_set_per_layer` with `IE_DS4_ARENA_THP=1`: each 16 GiB segment an anonymous
`mmap` + `MADV_HUGEPAGE`, touched on 8 threads (one write per 2 MiB), imported with
`zexDriverImportExternalPointer` (via `zeDriverGetExtensionFunctionAddress` on `libze_loader`),
released and unmapped in `free_storage`; the init line reports the `AnonHugePages` delta; a missing
extension or a failed import is a refusal. ~70 lines in `include/ie/expert_stream.hpp` /
`src/core/expert_stream.cpp`, built and measured, then **reverted** (below). The standalone bench
option that motivated it (`ie-ds41-cpu-expert-bench`, `IE_DS41_BENCH_IMPORT=1`) stays committed as
the record of the standalone result.

## Measured (`scratchpad/p13/measure13.sh`, one build, idle machine, 2,048-token context, steps 4-35)

| arm | arena | AnonHugePages of the two 79.8 GiB arenas | load | replayed prefill | ms/token | CPU ms per expert |
|---|---|---|---|---|---|---|
| A: split off | pinned USM (today) | — | 38 s | 7.56 s | **158.3** | — |
| B: split on, q* 0.30 | pinned USM | — | 38 s | 8.19 s | **149.6** | 0.82 |
| C: split off | imported THP | +79.8 / **+47.2** | 44 s | **10.81 s** | 163.3 | — |
| D: split on, q* 0.30 | imported THP | +79.8 / **+52.7** | 53 s | 10.69 s | 155.6 | **0.80** |
| E: split on, q* 0.15 | imported THP | +79.8 / +52.5 | 51 s | 10.65 s | 165.6 | 0.67 (0.68 in docs/35 without) |

**Criterion 4 — the CPU leg's mean did not move:** 0.82 → 0.80 ms per expert with the whole of
card 0's arena on 2 MiB pages and only 59-66% of card 1's (gate 13's reading: the treatment was
fully applied on one card and ~60% on the other, and the per-card CPU leg is not printed — so
"huge pages do not help the CPU leg" is NOT what this measured; what it measured is that this
arena did not move the mean, and its other costs decide the phase regardless). The standalone trend (0.39 at 32 GiB → 0.51 at 96 GiB pinned, 0.39 at 96 GiB
imported THP) does not carry into the tier: inside the tier the CPU's rate is set by the DMA it
shares the host memory with (docs/35: 0.65 at zero link traffic → 1.28 at 890 MiB/token), and the
page tables were the smaller term, if a term at all — "Pineapple": the per-card split of that 0.80
is not printed, so a card-0-only gain hidden in the mean cannot be excluded, but the token got
slower either way.

**Criterion 3 — the path off got slower, not the DMA:** the link's rate over the groups' wall is
22.5 GB/s in both A and C, the groups column identical (79-81 ms), and the extra 5 ms/token sits
in the per-layer prep (A 0.3-1.5 ms per token, C 5.1-7.3), with the replayed prefill 7.56 → 10.81 s
(+43%). Card 1's arena never got more than 66% of its pages huge (compaction after the first
80 GiB), so the remainder is a userptr on 4 KiB pages with 4 KiB GPU mappings — the likely cause
of the prefill cost, not verified ("Pineapple"); the prep cost is unexplained. Neither was worth
chasing once criterion 4 failed.

**Criterion 2 — bytes identical, verified:** with the imported arena the tier test PASS (pread pack
== mapping pack byte for byte, the routed output tracks the reference, a second call bit-identical)
and the full decode test PASS twice (the split on and off: the four greedy tokens, the forced
digits at all 40 layers, the state hashes, 0 FAIL lines). The strategy is correct; it is slow.

## Decision (criterion 5)

Not shipped: the arena stays pinned host USM; the code is reverted rather than left as an opt-in
that measures worse on every axis (load, prefill, decode, and no CPU gain). What the phase bought
is the measurement: the CPU miss path's rate inside the tier is a host-memory-sharing problem,
not a page-table problem, so the remaining lever on the split is fewer DMA bytes (expert parallel
over two links halves the per-link stream; lever C's pinned RAM cuts the disk misses), not a faster
CPU kernel. The founder's decode table in docs/35 stands: ~150-158 ms/token today, ~10-12 tok/s
the honest ceiling with the levers left, 20 out of reach single-stream at this context.
