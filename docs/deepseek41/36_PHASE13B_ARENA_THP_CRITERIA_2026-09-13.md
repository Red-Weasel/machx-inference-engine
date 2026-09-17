# V4.1 port — Phase 13b gate criteria: the pinned arena on imported huge-page memory (written BEFORE the build)

**Why (docs/deepseek41/35 step 3, measured):** inside the tier the CPU miss path computes an expert in
0.65 ms with no DMA in flight (1.28 with 890 MiB/token of it), against 0.36-0.39 standalone. One of
the two measured causes is the arena itself: 160 GiB of host USM is a `/dev/dri` mapping on 4 KiB
pages, and the page tables of a pinned arena that size fit no cache — standalone, the same 96
experts spread over a pinned arena read 0.39 ms at 8 and 32 GiB and **0.51 at 96 GiB**. Standalone
the fix is measured too: the same 96 GiB as anonymous memory with `MADV_HUGEPAGE`, handed to the
driver with `zexDriverImportExternalPointer`, is entirely on 2 MiB pages, reads at **0.39 ms per
expert**, and the H2D DMA from it runs at **26.6 GB/s — the pinned buffer's rate** — with the GPU's
M = 1 expert computed from the imported bytes agreeing at the same 3.1e-4.

**Starting line:** 4c9ac63 plus gate 13's fix (the block time unconditional) — decode at the
2,048-token context 159-160 ms/token with the split off (the clean control), ~151-158 with
`IE_DS41_CPU_MISS=1` at q* 0.30 (CPU 0.90-1.17 ms per expert inside the tier); the resident load
38-39 s; the pinned bytes over the groups' wall 18.4-18.9 GB/s.

**Re-sighted headroom (gate 13 finding 1c):** the split delivers ~9 ms today. Its bound is the
DMA it removes (1378 → 504 MiB ≈ 45 ms at the link's rate) minus the cache erosion it causes
(~6 ms), so the CPU leg costs ~30 ms of a possible ~40; halving the CPU's time per expert is worth
an estimated 10-15 ms/token ("Pineapple" — an estimate, the run decides), not the 42 the
65 → 23 projection implied.

## Scope

`Ds4HostArena::init_set_per_layer` (src/core/expert_stream.cpp, shared with the V4 tier) gains one
opt-in allocation strategy, `IE_DS4_ARENA_THP=1`: each segment (whole layers, the 16 GiB target, as today) is
an anonymous `mmap` with `MADV_HUGEPAGE`, touched so its huge pages exist, then imported with
`zexDriverImportExternalPointer` (fetched through `zeDriverGetExtensionFunctionAddress` on the
queue's Level Zero driver); `free_storage` releases it (`zexDriverReleaseImportedPointer`) and
unmaps. Nothing else changes: `slot()`, `is_pinned()`, the pack path, the DMA path, the
staging/bounce buffers (small, pinned USM as today). A missing extension or a failed import is a
refusal with the segment named — no pageable or 4 KiB fallback is substituted.

## Pass criteria

1. **Observable allocation.** With the env on, the init prints the segments, the bytes, and the
   `AnonHugePages` delta of the process (from `/proc/self/smaps_rollup`) so a partially-huge
   allocation is visible, not assumed. With the env off, the arena is byte-for-byte the code of
   4c9ac63 (pinned USM) — the V4 tier and every other user see no change.
2. **Bytes identical.** `ie-ds41-tier-test` (pread pack == mapping pack byte for byte, the routed
   output tracks the reference, a second call bit-identical) and `ie-ds41-decode-test` (the four
   greedy tokens, the forced digits at all 40 layers, the state hashes) PASS with the env on, both
   with `IE_DS41_CPU_MISS=1` and without it.
3. **The DMA is not slower.** The decode bench's "pinned bytes over the groups' wall" with the CPU
   path off is within 10% of the pinned-USM build's on the same day (18.4-18.9 GB/s), and the
   path-off ms/token is within the run-to-run spread (±3 ms) of 200.4.
4. **Measured with the path on**, the 2,048-token context, steps 4-35: the CPU ms per expert and the
   ms/token at q* 0.30, and a re-sweep at 0.15 / 0.45 (the balance moves if the CPU got faster);
   the load-time delta (the touch of 160 GiB is paid at load). The number is what it is.
5. **The default decision, stated:** on for the V4.1 tier if 2-3 hold and 4 improves the token;
   otherwise it stays opt-in with the measurement recorded. The V4 tier keeps the env off either way
   (its measurement is a follow-up, not this phase).

## Explicitly NOT in this phase

The staging/bounce buffers; the CPU path's share of the DMA contention (the other measured cause);
lever C (more pinned RAM); the attention cost at T = 1; expert parallel (Phase 14).
