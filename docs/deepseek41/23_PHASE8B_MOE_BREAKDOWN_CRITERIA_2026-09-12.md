# V4.1 port — Phase 8b gate criteria: where the MoE time goes, and the mmap tier's pack (written BEFORE the build)

**Starting line (docs/22, run 3, two cards):** pp2048 33.1 s = 18.9 s MoE + 14.2 s dense body.
Inside the MoE the mmap tier moved 23.1 GiB through `ds41_slot_pack`, a single-threaded
per-byte nibble permutation of every expert, before the DMA. Two facts point at that loop
without yet measuring it: the pinned arena is packed by the same function at load, 159.7 GiB
in 136 s = **1.2 GB/s**; and 23.1 GiB at 1.2 GB/s = **19 s — the whole MoE time**. If that holds,
the grouped GEMMs are hidden behind the pack and the mmap tier's fetch is CPU-bound, not
NVMe- or PCIe-bound (10.3 and 26.5 GB/s measured on this box).

## Scope

1. **Instrument, then decide.** `Ds41ExpertTier::moe` times its phases per call — prep (sort,
   gather, quantise), mmap pack (CPU, summed over experts), mmap total (pack + DMA wall), the
   group loop (fetch + GEMMs) — and `Ds41LayerStats` carries them; the resident test prints the
   breakdown next to each pp line. No fix is applied until this breakdown exists for run 3's
   configuration.
2. **The pack fix**, chosen by the breakdown: if the mmap pack is >= half the MoE time, the
   permutation becomes parallel (OpenMP over the 16-byte blocks, the engine's existing
   convention) for BOTH the arena load and the mmap tier; if that leaves the host copy as the
   bound, the permutation moves to the device (Phase 4's in-place repack kernel over the slot
   after a plain memcpy into pinned staging). If the pack is NOT the bound, the doc says what
   is and the fix targets that instead.
3. **The held-out profile (criterion 3's debt).** A routing profile from >= 64K tokens of real
   text that is NOT the measured prompt, written with `write_profile`; pp512/pp2048 measured
   with it; the in-sample numbers of run 3 reported alongside as the upper bound.

## Pass criteria

1. **The breakdown is measured, on the real path, and sums.** For the pp2048 cold pass:
   prep + mmap total + groups = MoE ms within 5% (the remainder, if any, is named). The mmap
   pack time is reported separately from the mmap total. Printed by the resident test.
2. **Correctness is unchanged, bit for bit where it can be.** `ie-ds41-tier-test`: two splits
   bit-identical, rel 3.056e-3 (a parallel or device permutation must produce the SAME bytes;
   the test's bit-identity across splits and its reference check are the proof). Resident test:
   worst 6.124e-3 to the first flip at layer 7, " Berlin", warm bit-identical — the same digits
   as run 3.
3. **The fix moves the number it targets.** Load time and the mmap tier's ms per forward drop
   by at least 3x on the pinned arena load (1.2 GB/s is single-core-bound; 20 cores are idle)
   — or the doc says why not with the measured rates. pp2048 tok/s is reported before/after on
   the same ranking; the criterion is the breakdown's phase, not the end-to-end number, but the
   end-to-end number is printed.
4. **Held-out profile.** The profile corpus is named, its token count printed (>= 64K), and it
   contains neither the golden prompt nor the pp512/pp2048 texts. Hit rates by tier at pp512 and
   pp2048 are printed for the held-out profile AND for run 3's in-sample profile, side by side.
5. **Unload returns everything**, as in Phase 8: VRAM < 256 MiB unreturned per card; RAM at the
   process boundary.
6. **Nothing else changes.** The dense body's 14.2 s is out of scope here (it gets its own
   phase with a profile first); no kernel outside the tier is touched.

## Results so far (2026-09-13, 07:06)

**Step 1, the breakdown (107ea36, run B, held-out ranking):** pp2048 MoE 30,057 ms = prep 4 +
mmap 23,076 (host pack 19,330) + groups 6,964; pp512 16,563 = 2 + 12,772 (12,454) + 3,774.
Sums within 13 ms of the MoE total (the `sycl::free` of the transient bank sits outside the
phases). Criterion 1 met.

**Step 2a, the parallel pack (run C, same ranking, 20 OpenMP threads over the 16-byte blocks):**
host pack 19,330 -> 18,351 ms at pp2048, 12,454 -> 11,078 at pp512; load 140 -> 121 s; pp2048
40.1 -> 42.3 tok/s. Bit-identical (tier test). **The permutation was not the bottleneck.** The
"host pack" phase is the first touch of the expert's mmap pages inside the pack loop — a
page-fault read of 40.1 GiB at ~2.2 GB/s — and twenty threads faulting instead of one moved it
by 5%. Criterion 3's 3x is NOT met by this step, and the doc says why: the phase timer names
the loop, not the fault path under it. The OpenMP pragma stays (harmless, slightly positive).

**Step 2b (next):** read the expert planes with `pread` from the shard file straight into the
pinned staging slot — the DS4 measurement for this NVMe is 10.3 GB/s at 5.6 MiB reads — then
permute IN PLACE (the shuffle is local to each 16-byte block) and DMA. Eight readers in
flight over sixteen staging slots. The same path fills the pinned arena and the static slots
at load, where the 121 s is the same fault-bound read (39 GiB static + 160 GiB pinned at
1.3 GB/s). Criterion 3 then applies to this step: load and the mmap phase each >= 3x, or the
doc says what binds with the measured rate. Correctness proof unchanged: the tier test's
bit-identity and reference check, plus a byte-for-byte comparison of one expert packed by
both paths.

**Step 2b, pread into the pinned slot + in-place permute (run D, 2026-09-13 07:15):** no change
— pp512 mmap read 10.7 s, pp2048 20.9 s (1.9 GB/s), load 133 s. pp2048 42.2 tok/s. One real
side effect: peak RSS 110 GB -> 13.5 GB (nothing touches the mapping any more).

**Two probes, before building further** (scratchpad `pinned_bw.cpp`, `permute_bw.cpp`, and a
Python O_DIRECT/buffered read test on real expert offsets):
- NVMe reads of real expert planes, cold, into ordinary memory: buffered pread 4.1 GB/s at one
  thread, 9.3 at eight; O_DIRECT 7.0 / 10.2 / 10.6 (1 / 8 / 16 threads). So neither the
  syscall path nor the drive explains 2 GB/s.
- CPU access to `sycl::malloc_host` memory equals ordinary memory: byte-store loop 6.5 GB/s
  both, memset 16-20, memcpy 20-24, read 32. The "pinned is write-combined" hypothesis is
  refuted.
- The exact 16-byte-block permute loop: 20 GB/s single-threaded (icpx vectorises it), 24
  aggregate over eight threads, into plain or pinned alike. Not a cost.

**Step 2c, O_DIRECT into a page-aligned bounce, permute from the bounce into the pinned slot
(run E, 07:23):** pp512 36.4 tok/s (mmap read 8.9 s, 2.8 GB/s), **pp2048 58.1 tok/s** (read
14.1 s, 3.0 GB/s), load 109 s. Bit-identical (tier test, plus the byte-for-byte cross-check of
one expert packed by both paths). Better, and still a third of what the same reads do in the
micro-benchmark: 50 ms per expert per reader thread against ~17 expected. Run F adds
per-thread read vs permute timers inside the pack to say which of the two it is in situ.

**Run F (07:30), per-thread read vs permute timers inside the pack:** pp2048 host pack 14,105
ms wall; per reader thread (sum / 8) reads 817 + permute 946 = 1,763 ms. Eight times that is
14,105 ms — the wall exactly. **The region was running on ONE thread.** Inside it the diagnostic
said so directly: `1 thread(s) in the region (asked 8); max 20, in_parallel 0, dynamic 0`.
Root cause: `src/CMakeLists.txt` gives `-fopenmp` per source, to `ops/cpu_moe_gemv.cpp` only;
`model/deepseek41_experts.cpp` was compiled without it, so every `#pragma omp` in this file —
step 2a's parallel permute, step 2b/2c's reader batches — was silently ignored. Every "parallel"
number above (runs C-F) is a single thread's. `model/qwen4_vision.cpp` has the same pragmas and
the same gap (not touched here; noted for its owner). Fixed by the per-source property, with a
one-time stderr warning if a fill region ever comes up short again.

**Run G (07:34), the readers actually parallel, same code otherwise:**

| | run B (baseline) | run G | |
|---|---:|---:|---|
| load (200 GiB placed) | 140 s | **48 s** | 2.9x |
| pp512 cold / warm | 24.7 / 34.3 | **53.6** / 61.5 | 2.2x |
| pp2048 cold / warm | 40.1 / 42.1 | **73.2** / 78.1 | 1.8x |
| pp2048 mmap phase (40.1 GiB) | 23.1 s | 8.4 s (reads 3.6 s per thread = 12 GB/s aggregate, permute 1.3, DMA waits ~3) | 2.7x |
| pp2048 groups (fetch + GEMM) | 7.0 s | 7.0 s | unchanged |
| whole run | 285 s | 126 s | |

Correctness identical at every check (own router 6.124e-3 to the flip at 7, " Berlin", forced
pass card 0 9.02e-3 / card 1 2.76e-2, top-5 order, logits 4.39e-2, warm bit-identical); tier
test bit-identical across splits and byte-for-byte across the two pack paths. Criterion 3
(>= 3x on the load and the mmap phase): load 2.9x, mmap phase 2.7x — close, not met as
written; the remaining gap is the ~3 s of DMA waits per pp2048 pass and the drive at its
measured ceiling. Criteria 1, 2, 4, 5, 6 met.

**Run H (07:44), the mmap fill overlapped with the groups:** the fill runs on a reader thread
with its DMAs on a second in-order queue (same device and context) while the static/pinned
groups run on the compute queue; the mmap experts form one last group after the fill has
landed (the reader waits on its own queue before it joins, so no cross-queue event is needed).
The scatter's summation order is fixed, so the output is bit-identical (tier test).

| | run G | run H | since run B this morning |
|---|---:|---:|---|
| pp512 cold / warm | 53.6 / 61.5 | **83.0** / 87.6 | 24.7 -> 83.0, 3.4x |
| pp2048 cold / warm | 73.2 / 78.1 | **79.0** / 96.8 | 40.1 -> 79.0, 2.0x |
| pp2048 MoE | 15.4 s = 8.4 + 7.0 | 11.7 s = max(9.1 fill, 8.0 groups) + 0.4 mmap group | |
| pp512 MoE | 7.6 s = 3.8 + 3.8 | 5.2 s = max(3.8, 4.7) + 0.2 | |
| load / whole run | 48 s / 126 s | 47 s / 111 s | |

The fill got slower under overlap at pp2048 (8.4 -> 9.1 s; the permute per reader 1.3 -> 2.2
s) — the readers now share the cores with the GEMM launcher and the copy engine with the fetch
pipeline's own H2D — but it is hidden behind the groups at both lengths.

**What binds now, at pp2048:** 25.9 s = MoE 11.7 + dense body ~14.2. The dense body is the
Phase 7 correctness loop and has never been profiled; it is the largest single piece and the
next phase. Inside the MoE the two overlapped legs are within 15% of each other and which one
binds flips run to run (run H: fill 9.1 vs groups 8.0; the gate's two runs: 9.15 vs 8.00,
then 7.89 vs 8.06) — "hidden" was the wrong word at pp2048; at pp512 the fill is under the
groups. The 1.0-1.5 s left over at pp2048 after prep + max + mmap group (gate finding 1) is
the scatter and the transient bank's free/alloc, now timed as "tail".

**Gate verdict (2026-09-13 08:02): PASS**, with criterion 3 partially met as written (load
2.98-3.59x, mmap phase 2.5-2.9x at pp2048, 3.3x at pp512; the escape clause used with
measured rates) and six non-blocking notes: the untimed tail (fixed: `ms_tail`), this
paragraph's wording (fixed), a joinable reader thread on a C++ exception path (fixed: RAII
join), the tier test's platform-default two-device context (fixed), the buffered-read fallback
exercised by no test and the silent O_DIRECT open failure (noted), and `qwen4_vision.cpp`'s
pragmas compiled without OpenMP (noted, untouched).

## After the gate: the profile, and two more levers (runs I-O, 2026-09-13 08:07-09:00)

The dense body got its stage timers (engram / attention / ffn-pre / MoE call / shared, with
the main thread's MoE chain prep + spawn + groups + join + mmap group + tail summing to the
tier's ms by construction) and the pass got prep (hashes, table gathers, embed) and head
timers. Two things fell out before any dense kernel was touched:

1. **The transient bank's per-layer `malloc_device` cost 3.4 s per pp2048 pass** (spawn 3,402
   ms; 230 at pp512). A persistent grow-only bank: 264 / 52 ms.
2. **The engram table gather was 2.6 s at pp512 and ~10 s at pp2048 per COLD prompt**, all of
   it outside the layer loop: 24 hashed rows per token per engram layer, each a random
   256-byte read from the 189 GB table inside the safetensors mapping (advised `MADV_RANDOM`),
   one 4 KB fault per row, single-threaded — 25K faults at pp512, 100K at pp2048, ~100 us
   each. The tokens are independent, so the gather is now an OpenMP loop over tokens (the file
   gets `-fopenmp` per source like the tier): "outside the layers" 2,574 -> 688 ms at pp512,
   10,565 -> 2,497 at pp2048 (run L -> run M). Warm prompts never paid it (page cache), which
   is why run H's cold pass looked fine and runs J-L did not: the golden generator's memmap
   churn had evicted the table pages in between.

| run (held-out placement, two cards) | pp512 cold / warm | pp2048 cold / warm | note |
|---|---:|---:|---|
| H (07:44, before) | 83.0 / 87.6 | 79.0 / 96.8 | table pages warm by luck |
| L (08:33, quiet CPU, stage timers) | 62.2 / 90.3 | 66.0 / 63.6 | pages cold; gate compile beside pp2048 |
| M (08:53, persistent bank + parallel gather) | 81.6 / 92.0 | **90.4** / 92.4 | quiet box |
| N (08:57, repeat) | 83.0 / 91.5 | 88.3 / 91.2 | quiet box |

Two measurement lessons paid for on the way: the resident test's first stage print read
`fwd.stats()` after the warm pass, so its "rest" mixed warm-pass layers with cold-pass MoE
(the 5.1 s "unattributed" in run J was that); and every stage number taken while a CPU-heavy
job ran beside it (the golden generator, a gate's icpx build) was inflated in the host-side
launch gaps, MoE excepted — the cold pass went 25.9 -> 35.8 s under the golden's decode phase
with the MoE unchanged at 9.2 s. Only quiet-box numbers are cited above.

### The floating ~250 ms per layer (runs O-S2, 08:59-09:20): what is known

At pp2048 the two passes' per-layer totals agree (~480 vs ~470 ms, every layer) while the
stages do not: cold = attention ~170 + groups ~200 + shared ~55; warm = attention ~30 +
groups ~400 + shared ~2 (per-layer dumps `stages_pp2048_{cold,warm}.txt`). Established since:

- The attention KERNEL's own GPU timestamps (`IE_QUEUE_PROFILING=1`, run R) are identical
  cold and warm: 3.4-3.9 ms at layers 0-1, 8.0-8.5 at ratio-2 layers, 12-13 at ratio-1
  layers. So the cold "attention stage" (150 ms host wall) is not that kernel running slowly;
  the queue reaches it ~120 ms later.
- Not OpenMP spin-waiting: `KMP_BLOCKTIME=0` / passive wait policy changed nothing (run Q).
- Not copies on the EUs: `UR_L0_USE_COPY_ENGINE=1` changed nothing (S1); disabling the V2
  adapter's copy offload (`UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1`, S2) made the MoE SLOWER
  (8.8 -> 12.4 s cold, 16.9 -> 20.5 warm) and left the asymmetry as it was — so offload is on
  by default and helps, and the DMAs are not the floating cost.
- The oneDNN GEMMs run on a stream built from the caller's queue and their primitives are
  cached per shape, so neither ordering nor per-layer JIT explains it.
- **Found (run T2, 09:24): VRAM eviction.** Each card ended its load ~1.9 GiB free (the 3 GiB
  reserve minus the batch workspace, the decode state and the tier's metadata), and a pp2048
  pass needs ~1.1 GiB of scratch (the two hc streams 335 MB, engram rows 380 MB, attention
  operands, masks, scores) plus the persistent expert bank (~1.5 GiB on this profile) plus
  oneDNN's workspaces — more than is free, so the driver migrated buffers to host memory
  and which ones differed between passes. Run T (last logits row only, 1 GiB less) changed
  nothing; **run T2 with a 6 GiB reserve: pp2048 22.1 -> 12.5 s cold (163 tok/s), 11.4 s warm
  (180 tok/s)**, attention 1.41 / 1.36 s, MoE 8.83 / 8.78, shared 98 / 97 ms — the two passes
  agree stage by stage, and the load 47 -> 41 s. It gave up 8 static slots per layer (54/52 ->
  46/44; static hits 2,047 -> 1,726 at pp2048, mmap 40 -> 45 GiB) and still ran 1.8x faster.
  The default reserve is now 6 GiB (`ResidentOptions::vram_reserve`, sized by measurement);
  deriving it from `max_tokens` and capping the bank are the follow-ups. The one new number to
  understand: 272 / 275 MiB unreturned on unload at the 6 GiB reserve against the 256 MiB bar
  (19-65 MiB before), a `[FAIL]` in T2 — most likely the runtime's pool holding the larger
  freed bank; not yet verified.

| pp2048, two cards, held-out placement | run H (07:44) | run T (last-row logits) | **run T2 (6 GiB reserve)** |
|---|---:|---:|---:|
| cold / warm tok/s | 79.0 / 96.8 | 92.6 / 92.6 | **163.4 / 179.7** |
| cold pass | 25.9 s | 22.1 s | **12.5 s** |
| attention (cold / warm) | — | 6.85 / 1.48 s | 1.41 / 1.36 s |
| MoE (cold / warm) | 11.7 s | 8.79 / 17.10 s | 8.83 / 8.78 s |
| pp512 cold / warm tok/s | 83.0 / 87.6 | 82.6 / 91.5 | 78.4 / 87.8 |

Since this morning's run B (24.7 / 40.1 on the same placement): pp512 3.2x, pp2048 4.1x.

**Final verification chain (09:29, the 6 GiB default in the header, all four binaries rebuilt):**
tier PASS; resident — 45/44 static + 8 stream per layer, 5.2 / 4.9 GiB free after load, own
router 6.124e-3 to the flip at 7, " Berlin", warm bit-identical, forced top-5 order, **pp512
84.2 cold / 88.9 warm, pp2048 173.9 cold / 180.6 warm tok/s** (11.78 s cold: layers 11.14 =
attention 1.40 + ffn-pre 0.84 + MoE 8.79 + shared 0.10; outside the layers 0.64); decode PASS
(four greedy tokens); forward own and forced PASS. The one `[FAIL]`: 268 / 272 MiB unreturned
at unload against the 256 MiB bar — resolved by measurement (runs U-W, 09:33-09:41):

- three-point unload (before / after `free_resident` / after the destructor): the destructor
  returns nothing more; with the runtime's USM pool disabled (`UR_L0_DISABLE_USM_ALLOCATOR=1`)
  220 MiB remain, so the pool is ~50 MiB of it;
- with only the 12-token shapes exercised (run V, no pp passes) 115 MiB remain; with the
  pp512/pp2048 shapes 220-270 — the residue scales with the SET of GEMM shapes seen, never
  with the number of passes: oneDNN's per-shape primitive cache and workspaces, the JIT'd
  kernel bundles, the pool — process-lifetime state, not engine allocations;
- the 3 GiB runs' "10 MiB unreturned" was not evidence against a leak: under eviction the
  driver's free figure already counted the migrated buffers;
- the decisive check, now in the test under `IE_DS41_TWO_CYCLES=1`: a second
  load/forward/unload cycle in the same process, the runtime's caches already warm — **the
  second unload returns to the first's level within 3.0 / 0.01 MiB per card. The engine leaks
  nothing.** The single-cycle bar is 384 MiB, stated as "within the runtime's measured
  residue"; the two-cycle bar is 64 MiB and is the one that means "no leak".

`ie-ds41-resident-test` PASS on both checks at the 6 GiB default.

## Explicitly NOT in this phase

The dense body, decode (Phase 9), any change to the grouped GEMM path, prefetching across
layers, the candidate top-k.
