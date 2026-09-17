# V4.1 port — Phase 15 gate criteria: the mmap tier's host-memory passes (written BEFORE the build)

**Where prefill stands after Phase 14 (docs/39):** pp2048 308-319 tok/s with the two links streaming
the pinned tier in parallel, and the wall is the HOST MEMORY, not the links: each link streams
~12.5 GB/s during the pass against 26 + 26 alone, because the mmap tier's bytes cross that memory
four times — `pread` O_DIRECT into the bounce (1 write), the permute into the pinned staging slot
(1 read + 1 write), the staging → VRAM DMA (1 read) — beside the pinned tier's one pass (arena →
link). Per pp2048 pass: ~100 GiB pinned × 1 + ~30 GiB mmap × 4 ≈ 220 GiB of host traffic in ~6.6 s
(~33 GB/s average, more at the peaks) — the peer's accounting (its message of 21:00), read off the
code path in `ds41_slot_pack_pread` and consistent with the ~45 GB/s the pass measures ("Pineapple"
on the exact multiplier: the pass count is read from the code, not instrumented; step 0 measures
it). The fill leg (3.6-4.0 s per pass) and the owner's groups (4.1-4.5 s) are now within half a
second of each other, so the mmap tier's host cost is on the binding path.

**Two levers in that one currency, ranked:**
- **C2 — a permute-free on-disk layout for the mmap tier.** The experts the mmap tier serves,
  written once in the engine's slot layout (the 16-byte SoA nibble blocks + E8M0 scales, exactly
  the bytes `ds41_slot_pack` produces) so a fill is one O_DIRECT `pread` straight into the pinned
  staging slot: 4 passes → 2 (the disk's DMA write, the staging → VRAM read). **−60 GiB of host
  traffic per pass** and −1.1 s of permute per reader. Constraints measured before the build: the
  main NVMe (Samsung 9100 PRO, ~12 GB/s O_DIRECT, 97 GB free) must hold it — the Secondary NVMe
  reads 5.0 GB/s alone, 6 with two readers (the two drives do run concurrently: 10.9 + 5.1 GB/s),
  so a file there would turn a 2 s fill into a 4-6 s one. The file therefore holds only the
  ranking's TAIL (the experts beyond the static + pinned cutoff of every layer), and its size is
  set by C1.
- **C1 — the pinned cap from the live rule instead of the 160 GiB constant.**
  `ds4_host_pin_cap_live` (docs/deepseek4/40: min(MemAvailable − max(16 GiB, MemTotal/8),
  MemTotal − max(16 GiB, MemTotal/4)) = ~184 GiB on this 249 GiB box with 215 GiB available)
  pins ~24 GiB more = ~17 more experts per layer per card (or 34 per layer single-tier), moving
  the next-hottest ranks from 4 passes to 1: ~−25 GiB per pass, and fewer disk misses at decode
  (today 15 per token, 25-45 ms).
- P2P for the x / rows staging removes 0.3 s of a pass and ZERO host passes: not in this race.

**The arithmetic, stated before the build.** C1: the mmap tier per pp2048 pass ~30 → ~18-22 GiB
(the tail's mass, from the held-out ranking — measured in step 0). C2 on top: those bytes at 2
passes instead of 4. Host traffic per pass ~220 → ~140-150 GiB; if the links recover toward
20 GB/s each as the memory frees, the owner's groups ~4.1 → ~2.8-3.2 s and the fill leg ~4.0 →
~1.8 s; the pass ~6.6 → ~5.0-5.4 s → **~380-410 tok/s** at pp2048, ~150-165 at pp512. Decode: the
disk misses' bytes and the permute in the mmap group (the "mmapgr" / tail columns) fall → ~150 →
~135-140 ms/token. The number is what it is.

## Step 0 results (measured 21:54, before any C2 code)

- **0(c) — O_DIRECT into pinned host USM is REFUSED:** `pread` returns EFAULT ("Bad address", 0
  bytes) into both `sycl::malloc_host` and `aligned_alloc_host(4 KiB)` buffers; into
  posix_memalign memory it reads 64 MiB at 11-12 GB/s, bytes identical to a buffered read.
  **Anonymous THP memory imported into the driver takes the direct read** (11.0 GB/s, bytes
  identical) **and still DMAs to the GPU afterwards** (13.7 GB/s for a 64 MiB copy, bytes on the
  GPU identical). So C2's shape is: the staging slots become imported anonymous memory (16 ×
  18.8 MB per tier), the fill is one direct `pread` into them, and the DMA is unchanged — the two-
  pass version stands. GO.
- **0(d) — the permute is instruction-bound:** one thread 3.80 ms per slot (9.9 GB/s of traffic)
  against `memcpy` 0.89 ms (42 GB/s); eight threads 8.79 ms per slot (34.2 GB/s) against 4.59
  (65.5 GB/s — the host ceiling). The shuffle runs at half the bandwidth the same cores move
  with memcpy, so C2 returns CPU time as well as the two passes.

## Scope, in order; each step measured before the next

**C0 (added after step 0(d), the peer's read of it): the vectorised permute.** `permute_plane` was
a scalar byte loop, and the shuffle it performs — the even outputs are the low nibbles of bytes
0-7 paired with those of bytes 8-15, the odd outputs the high nibbles, interleaved — is five SSE2
instructions per 16-byte block. It costs one function, no format and no disk, and it helps EVERY
mmap expert, the ranks outside C2's window included; the scalar version stays as the reference
and `ds41_permute_plane_selftest` (run by the tier test on the model's plane sizes) must equal
it byte for byte. Measured first: the micro-bench against memcpy, then the resident test's
permute share per reader per pass (1.1-1.4 s today). If it lands near memcpy, the gap C2 closes
is one host pass rather than a pass plus an instruction-bound shuffle, and C2's worth is
re-judged on that baseline before the 33 GB is written. **Measured (22:02, the SSE2 version,
byte-identical to the reference on 9 random planes):** one thread 3.95 → **1.72 ms per slot**
(memcpy 0.87); eight threads 8.23 → **6.82 ms per slot** at 44 GB/s of traffic against memcpy's
4.71 at 64 (the host ceiling) — instruction-bound no longer, ~1.45x memcpy at width. Its share
of the fill leg is measured in the chain below; an AVX2 widening is the obvious next 20% if
the fill leg still shows the permute.

0. **Measure before building:** (a) the ranking's tail mass — with the held-out profile, how many
   mmap experts and bytes per pp2048 pass at pinned 228 and at the C1 count, from
   `ie-ds41-placement-bench` or the resident test's "occupied experts/forward" line; (b) the host
   pass count — the resident test's fill-leg breakdown (reads / permute per reader) already
   separates the permute; report it as the measured share of the four passes. (c) **Whether an
   O_DIRECT `pread` can land in pinned host USM at all** (the peer's pre-read P1: the staging slots
   are `sycl::malloc_host`, a /dev/dri mapping — the same property that blocked huge pages in
   Phase 13b): `scratchpad/p15/odirect_usm_probe.cpp` reads 64 MiB at a 4 KiB-aligned offset into
   posix_memalign memory, `malloc_host`, `aligned_alloc_host` and anonymous THP memory imported
   with `zexDriverImportExternalPointer`, and reports the return, the bytes against a buffered
   reference, the rate, and the H2D from each buffer afterwards. If USM refuses or demotes, the
   staging slots become imported anonymous memory (16 × 18.8 MB per tier; the docs/37 mechanism,
   whose DMA measured at the pinned rate, and 500x smaller than the arena whose huge pages ran
   out at 66%) so C2 keeps its two passes. **If that fails too, C2 saves NO host passes** (the
   peer's correction of my first draft): a bounce still in the path is pread-write + copy-read +
   copy-write + DMA-read = four, exactly today's, because `permute_plane` is already a single
   streaming read + write — the shuffle costs no extra traffic, only instructions. The
   both-refused branch survives only as a CPU-time change, and only if the permute is
   instruction-bound rather than bandwidth-bound (its 0.60 output bytes per cycle at 4.8 GHz
   against ~5 ops per byte says instruction-bound; 5.8 GB/s per reader × 8 ≈ 46 GB/s says
   bandwidth-bound; the two readings disagree) — settled by (d) below. Step 0(c) is therefore
   go / no-go for C2, not a shape question.
   (d) `permute_plane` against `memcpy` over the same 18.8 MB buffers on one core and on eight
   (`scratchpad/p15/permute_vs_memcpy.cpp`): if memcpy is no faster at the same traffic the
   shuffle is bandwidth-bound and the both-refused branch is "drop C2".
1. **C1.** `ResidentOptions::host_pin_cap == 0` means the live cap (`ds4_host_pin_cap_live`,
   its `why` string printed at init) **with a third term the rule does not have** (the peer's
   pre-read of 21:08): both of the rule's terms reserve against the box at LOAD time and nothing
   reserves against what it will want afterwards — at 184 GiB pinned the run would leave ~28 GiB
   available for its whole life (160 leaves 52 today), and this box has a history at that end (the
   harness's low-memory watchdog, the 2026-08-27 freeze). So the forward also requires **at
   least 40 GiB to stay available after the pin**: cap = min(live rule, MemAvailable − 40 GiB) ≈
   175 GiB on this box, ~+15 GiB over the constant. The entry points default to 0. Measured and
   printed: MemAvailable DURING the run at the new cap (the resident test already checks the
   drop; the number that stays is the one to act on) — expected ~37 GiB, not 40: the floor holds
   at the moment of pinning and the engine's own unpinned footprint (bounce buffers, staging
   slots, host-side stream buffers, the process) comes out of the same pool afterwards, ~3 GiB
   today (160 pinned leaves 52 of 215 − 160 = 55); a reading materially below ~30 would mean
   something scales with the run that this arithmetic misses (the staging and bounce sets with
   the reader count and slot size, the host buffers with max_tokens) and is chased before the
   cap ships — and the case the rule cannot see — with
   the model loaded at the cap, a ~10 GiB consumer (a `-j8` build of one tool) is started and the
   run survives with the box out of systemd-oomd's pressure state. Then pp512 / pp2048 and decode
   with EP on and off; the per-card accounting line.
2. **C2.** A tool `ie-ds41-expert-file` writes `<model>/ie_experts_tail.ieslot` (header: magic,
   the slot layout, the per-layer window, the (layer, expert) → offset table; then the slots,
   4 KiB-aligned) for a WINDOW of the tail — ranking positions [297, 341) of every layer by
   default, ~33 GB: the whole tail (~60 GB) would leave the system volume under the 50 GB floor,
   and the ranking is by frequency so a pass touches mostly the tail's head (the covered share of
   the mmap traffic is measured, not assumed: the tier prints its coverage and the stats count
   the experts filled from the file per call); the
   tier, at init, opens it if present and its hash matches the ranking in use (else the pack path
   as today, said on stderr — not silently), and the mmap fill becomes one `pread` into the
   staging slot (allocated 4 KiB-aligned for O_DIRECT, or imported anonymous memory per 0(c)).
   `ie-ds41-tier-test` gains: whole slots read from the file equal the pack path's bytes, byte
   for byte, over several experts (one at a layer boundary, one at the table's end, one whose
   ranking position is the cutoff itself) — the same comparison the test already makes for the
   pread pack, not a per-plane sample (P2).
   Measured: as in 1, plus the fill leg's reads / permute breakdown (the permute must read 0).
3. **Correctness, the whole set** (resident, decode, generate, replay, tier, forward) with C1 and
   C2 on, EP on and off: every digit equals today's (the bytes are the same bytes; nothing in the
   arithmetic changes) — the dumps at 0.0 as in Phase 14.
4. **The founder-facing statement:** prefill and decode against 400+ / 20+ with this phase's
   measurement in the lever table; what remains (P2P's 0.3 s, attention at T = 1, FP8 dense).

## Pass criteria

Steps 0-2's measurements reported whatever they say; step 3 holds; the file is refused, not
trusted, when its layout does not match, and the slots it serves are the pack path's bytes (the
init-time sample check, and the tier test's whole-slot comparison over several experts — gate
15 reconciled these two paragraphs, finding 6; there is no ranking hash: the file is keyed by
(layer, expert) and a ranking change only changes coverage, finding 9); disk space is checked before the write — the tool prints the projected
size and the free space it would leave, refuses to leave the main NVMe (the founder's system
volume) under **50 GB**, and writes only with an explicit `--confirm` (P3); the default with no
file present is byte-for-byte today's path. Coverage is per expert: an mmap expert the file does
not hold takes the pack path, and the init line prints how many of this tier's mmap experts the
file covers — a partial file is visible, never silent.

## Explicitly NOT in this phase

P2P; the CPU miss split under EP (its cores); the attention path at T = 1; an expert file for
the pinned tier (it is read once at load — the pack path stays).
