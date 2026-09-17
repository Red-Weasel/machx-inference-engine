# V4.1 — Phase 48: the mmap reader thread leak (fixed), and where the "unused" VRAM goes

## The leak

A served process slowed with every reply. Measured on `ie serve` (ctx 75,000, Dream's 90 tools):

| | threads | RssAnon | MemAvailable | decode on the same prompts |
|---|---|---|---|---|
| before, after 9 profiling replies | **94,178**, +10,367 per 120-token reply | 5.2-5.8 GB, +1.6-2.5 GB per 300-token reply | **36 -> 14-15 GiB, swap exhausted, ~40 % CPU system time** | 7.1 -> 4.8 -> 1.75 -> 2.7 -> 1.7 -> 1.3 tok/s |
| after the fix, 4 replies | 49 -> 100 -> 119 -> **138 (flat)** | **~500 MiB (flat)** | **38 GiB (flat)** | 9.6 / 5.5 / 7.8 / 8.6 tok/s |

Cause: `Ds41ExpertTier::moe()` filled the mmap-tier experts on a FRESH `std::thread` per call (every MoE layer of every
decode step that touches an mmap expert, and every prefill layer), and that thread opened an OpenMP team
(`parallel_fill`). libiomp5 never reclaimed those teams, so every call left kReaders threads and their stacks behind.
Pre-existing since the reader was introduced (Phase 9-11); short test processes never lived long enough to show it,
and a long-lived server shows it within minutes -- it is also what docs/87's "the ranking is slow on chat" finding
actually measured (retracted there).

Fix: two PERSISTENT reader threads per tier (`MmReader mmr_[2]`, `mm_reader_post` / `mm_reader_wait`): [1] spawned
with every core allowed for prefill with the CPU leg on (the old code relaxed the caller's affinity before each spawn),
[0] with the caller's partition otherwise -- the same affinity each call's thread inherited before. Each team is
created once. The job body is unchanged.

Gates on the build: decode 28/28, multi 38/38; 32k long test 319.1 tok/s prefill (308-318 before) and the 64 decode
ids identical (`c14c73474cec53b2`).

## The VRAM question: 27.5 / 32 GB "used" per card

Sampled every second through the 32k long test (`ds41_work/rank48/vram.log`):

| phase | card 0 MiB | card 1 MiB | free of 32,656 |
|---|---|---|---|
| loaded, idle | 27,452-27,567 | 26,419 | ~5-6 GB |
| **prefill, 2,048-token chunks** | **31,175-31,268** | **30,607-30,725** | **1.4 / 1.9 GB** |
| decode after it | 29,299 | 30,725 | 3.4 / 1.9 GB |

The ~6 GiB kept free per card at load is the per-forward reserve (`vram_reserve`, and the computed batch workspace +
scratch it floors), and a 2,048-token prefill chunk uses ~3.7-4.3 GB of it (the chunk's scratch, the MoE batch
workspace, the grow-only mmap bank, attention workspaces). It is idle between prompts, not unused: at the peak 1.4 GB
is left on card 0, and docs/23 measured a 3 GiB reserve pushing a prefill into driver eviction (22 s vs 12.5 s). One
more static slot per layer costs 20 x 17.9 MiB = 358 MiB per card, so what could be taken safely (~1 GB) is ~3 slots
of 384 per layer.
