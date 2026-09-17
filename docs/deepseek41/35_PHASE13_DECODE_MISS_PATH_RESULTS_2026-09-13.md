# V4.1 port — Phase 13 results: the decode miss path

**Criteria:** `34_PHASE13_DECODE_MISS_PATH_CRITERIA_2026-09-13.md` (written before the build, with
the founder's 400+/20+ goal and the arithmetic that says single-stream decode lands near 12-15
tok/s on this hardware). **Starting line:** 4520911 — decode 5.0 tok/s at the 2,048-token
context, 200 ms/token: 65 ms pinned fetch over one PCIe link, 26-46 ms disk, 12 ms expert GEMMs,
~40 ms dense body, ~12 ms the rest.

## Step 1 — the residency policy at decode, measured before anything was built

`ie-ds41-placement-bench` (`tools/ds41_placement_bench.cpp`, a scratch tool at the time, committed
at gate 13's request so the table can be re-derived; the only knob is `ResidentOptions::stream_slots`, the static count derives from the VRAM budget so the slot
total per layer is constant at 53 / 52 per card, and the pinned arena stays 228 experts per layer
under the 160 GiB RAM cap). Greedy, steps 4-35, the held-out ranking; the 4,096-token context is
the pp2048 text twice, so its routing statistics are the 2,048 text's.

| placement (static + LRU per layer) | context | ms/token | tok/s | pinned MiB/token | disk MiB/token | hit rate | mmap experts/token |
|---|---|---|---|---|---|---|---|
| 45 + 8 (today) | 12-token prompt | 306.9 | 3.26 | 1505 | 775 | 47.0% | 43.2 |
| 45 + 8 | 2,048 | **200.4** | **4.99** | 1378 | 275 | 61.6% | 15.3 |
| 45 + 8 | 4,096 | 200.6 | 4.99 | 1429 | 238 | 61.3% | 13.3 |
| 20 + 33 | 12-token prompt | 334.6 | 2.99 | 760 | 1037 | 58.2% | 57.8 |
| 20 + 33 | 2,048 | 223.3 | 4.48 | 891 | 406 | **69.9%** | 22.7 |
| 20 + 33 | 4,096 | 224.4 | 4.46 | 998 | 354 | 68.6% | 19.8 |
| 2 + 51 / 1 + 51 (pure LRU) | 2,048 | 251.8 | 3.97 | 771 | 507 | **70.3%** | 28.2 |
| 2 + 51 / 1 + 51 | 4,096 | 247.5 | 4.04 | 886 | 445 | 69.1% | 24.8 |
| 2 + 51 / 1 + 51 | 12-token prompt | 396.5 | 2.52 | 632 | 1292 | 55.3% | 72.1 |

(The 0 + 52 request is refused by the budget check: card 1 has 52 slots and needs one static;
51 is the pure-LRU shape this budget allows.)

**What it says.** The LRU policy is the better allocator for the PCIe tier at decode — at the
2,048-token context 20 + 33 raises the hit rate from 61.6% to 69.9% and cuts the pinned bytes
by 35% (1378 → 891 MiB per token, ~22 ms of link time) — and it is WORSE overall (200 → 223 ms)
because a static slot is also protection against the disk: the 25 static experts per layer it
gave up were ranks 20-44 of the profile, so the pinned window slid to ranks 20-247 and ranks
248-272 fell to the mmap tier, whose misses cost more per byte than the link (22.7 disk experts
per token instead of 15.3, +131 MiB from disk, +23 ms net). Levers B and C of docs/34 are
coupled: the LRU policy pays only if the mmap boundary is held, which means more pinned RAM
(lever C) — or the CPU miss path, which does not care which tier a miss came from as long as the
bytes are in host RAM. The pure-LRU shape says where the policy saturates: 70.3% at 2,048 and
69.1% at 4,096, the same ~70% as 20 + 33, so the LRU's ceiling at this context is a 44% cut of
the PCIe bytes (1378 → 771 MiB per token, ~28 ms of link time) — worth having only with the
disk boundary held, i.e. 25-45 more pinned experts per layer (18-32 GiB more pinned RAM) or the
misses served from host RAM by the CPU. Placement stays at 45 + 8 for the rest of the phase.

## Step 2 — the CPU MXFP4 expert kernel (`cpu_moe_mxfp4.cpp`, `cpu_moe_mxfp4_test`, `ie-ds41-cpu-expert-bench`)

One expert from the pinned arena's slot exactly as the GPU reads it (the 16-byte SoA nibble
blocks, one E8M0 scale per block): gate and up over x[5120], silu with the clamp, down — fp32
throughout, AVX2 + FMA (this CPU has no AVX-512): the 16 nibbles of a block decoded with one
`pshufb` against the signed magnitude table, four FMAs per block against x, one scale multiply
per block; rows split over an OpenMP team. Exactness: the nibble table and the E8M0 half-scale
equal the reference formulas (16 nibbles, 253 exponents, the two subnormals); the AVX2 GEMV
equals the scalar dequant + double-accumulated dot within **2e-7** on the model's two shapes;
the expert composition equals a scalar recomputation within **2.3e-6** over 5,120 outputs; and
on a real expert (layer 0, expert 0, packed by `ds41_slot_pack`) it agrees with the GPU's own
M = 1 path within **3.1e-4** (the GPU quantises the activation to Q8; the CPU keeps it fp32).

Rate, cold (96 experts of layer 0 = 1.7 GiB rotated, best of three rounds, `OMP_PLACES=cores
OMP_PROC_BIND=close`, `taskset` to the core set):

| core set | ms per expert | experts per ms | slot bytes per second |
|---|---|---|---|
| 8 P-cores (0-7) | 0.43-0.50 | 2.0-2.3 | 38-42 GB/s |
| **12 E-cores (8-19)** | **0.36** | **2.75-2.81** | **51.8-52.7 GB/s** |
| all 20 | 0.52 idle; **7.8-11.9 under a ~400% CPU load** (gate 13) | 1.92; 0.1 | 36.1; unusable |

The E-cores alone are the fastest set — 51.8 GB/s of the ~60 GB/s this rig's host RAM streams
(the GLM q* measurement) — and gate 13 sharpened the all-20 row: a 20-thread pinned team on a
20-core box has no slack and collapses 15-23x under any other load, while the E-core set measures
the same 0.36 under that load. Against the link's 22 GB/s = 1.2 experts per ms, the E-cores are worth 2.3 links, and
they leave the P-cores to the host thread and the mmap readers. That sets q*: of the m pinned
misses in a layer, the PCIe share is 1.2 / (1.2 + 2.75) = **0.30** and the CPU takes the rest;
at the 2,048-token context's 92 misses per token that is a projected 92 / 3.95 = **23 ms** for
what costs 65 ms over the link alone.

## Step 3 — the q* split

The split lives in `Ds41ExpertTier::moe` at T = 1 (`IE_DS41_CPU_MISS=1`, off by default in this
phase): of the m pinned misses of a layer, the first q*·m go over the link as before and the rest
go to a worker thread whose 12-thread OpenMP team is pinned to the E-cores (`IE_DS41_CPU_CORES`,
default `8-19`); the host thread and its mmap reader team are pinned to the complement. The
scatter still walks a CPU expert's packed row, with weight 0 and a zeroed row, so it adds
nothing; the worker's fp32 partial is copied up and added to y after the join. The activation
stays fp32 on the CPU (the GPU quantises it to Q8), so the CPU expert differs from the GPU's by
the 3.1e-4 of step 2.

**Two levers, not one — gate 13's isolation.** The first runs measured **358.3 ms/token** with
the OpenMP default against **158.4** with `OMP_WAIT_POLICY=passive`, and the cause is the reader
team, not the worker: after every read region its threads spin for libiomp5's default 200 ms
block time on the P-cores and displace the host thread that submits the GPU work (the groups'
wall 31 → 41-75 ms, the tail 7 → 69-147). The first build set the policy from the entry points'
`main`, the generate test — which sets nothing — failed its 210 ms bar at **527 ms/token**, and
the fix moved into the tier (`kmp_set_blocktime(0)` in each thread that forks a team: the host
thread at init, the mmap reader thread of every `moe` call, the worker) — gated on the CPU path.
That gating confounded the phase's control: "off" kept the readers spinning, so "off vs on"
measured two changes. Gate 13 separated them with q* = 1.0 (the path on, no expert to the CPU,
traffic byte-identical to off: 1378.3 MiB, 104.8/119.8/15.3, 61.6%), on an idle machine, two
runs each:

| arm | ms/token | link MiB/token | CPU experts/token |
|---|---|---|---|
| off as shipped in 4c9ac63 (readers spinning) | 183.1, 189.3 | 1378.3 | 0 |
| on, q* = 1.0 — the block time alone | **159.2, 160.1** | 1378.3 | 0 |
| on, q* = 0.30 — the block time and the split | **150.9** | 504.3 | 55.2 |

Of the ~35 ms, **~26 ms is the block time and ~9 ms is the CPU split**. Under a ~400% CPU load
(the founder's blender): off 346.7, q* = 1.0 **158.3**, q* = 0.30 168.8 and 194.4 — the block
time makes decode immune to other load, and the split then loses 10-35 ms competing for cores.
So the block time is now unconditional (numerics-free, no cores, on by default: `init` sets it
before the first team, the reader thread and the worker set their own), and the split stays
opt-in with its ~9 ms and its contention caveat. My re-measurement on that build is below.

**The q* sweep**, 2,048-token context, steps 4-35 after the replayed pp text, each q* one
36-step run (the run-to-run spread on one setting is ±3 ms):

| q* (PCIe share of the pinned misses) | ms/token | tok/s | pinned MiB/token over the link | CPU experts/token | CPU compute ms/token | ms per CPU expert |
|---|---|---|---|---|---|---|
| off with the readers spinning (step 1's bench; gate 13's decode test 183-189) | 200.4 | 4.99 | 1378 | 0 | — | — |
| off, block time 0 (the clean control: gate 13's q* = 1.0) | 159-160 | 6.3 | 1378 | 0 | — | — |
| 0.0 | 172.3 | 5.80 | 0 | 112.9 | 73.9 | 0.65 |
| 0.15 | 159.8 | 6.26 | 132 | 90.2 | 61.5 | 0.68 |
| **0.30** (default) | **158.1** | **6.32** | 504 | 55.2 | 49.8 | 0.90 |
| 0.45 | 155.7 | 6.42 | 561 | 50.5 | 44.0 | 0.87 |
| 0.60 † | 165.8 | 6.03 | 890 | 28.1 | 36.1 | 1.28 |

(† measured on the block-time build below, on which 0.30 and 0.45 read 161.8 and 161.7 / 159.3
— the same within the spread.) Against the clean control the split is worth **~9 ms/token**
(159-160 → 151-158); against the shipped 4520911 the two levers together are 200 → 151-160 (5.0 →
6.3-6.6 tok/s). The curve is flat from 0.15 to 0.45 within the spread and worse at both ends, so
the default stays at the arithmetic's 0.30. The projection of step 2 (65 → ~23 ms on the fetch) did not happen: the
CPU's experts per ms inside the tier are 0.8-1.5, not the 2.75 of the bench, for two measured
reasons and none of three suspected ones:

1. **The link's DMA and the CPU share the host memory.** The ms per CPU expert rises
   monotonically with the pinned MiB per token: 0.65 at 0, 0.68 at 132, 0.87-0.90 at 504-561,
   1.28 at 890 — at q* 0.60 the CPU reads 14 GB/s while the link takes 19. The two tiers are
   not independent lanes; that is why q* saturates where it does.
2. **The pinned arena's page tables.** Standalone, the same 96 experts spread over a pinned
   arena with every page touched: 8 GiB 0.39 ms, 32 GiB 0.39, **96 GiB 0.51** — the 4 KiB
   page-table entries of a big pinned arena stop fitting any cache, and the tier's arena is
   160 GiB (not measured at 160; the trend is — "Pineapple"). Huge pages are the fix, and the
   host USM allocation cannot have them (a /dev/dri mapping). **Measured standalone, the way
   out** (`IE_DS41_BENCH_IMPORT=1`): the same 96 GiB as anonymous memory with `MADV_HUGEPAGE`,
   handed to the driver with `zexDriverImportExternalPointer` (fetched by
   `zeDriverGetExtensionFunctionAddress`, present in this driver) — all 96 GiB on 2 MiB pages
   (`AnonHugePages` 100,663,296 kB), the CPU back at **0.39 ms per expert**, the H2D DMA from it
   **26.6 GB/s, identical to a pinned buffer**, and the GPU's M = 1 expert computed from the
   imported bytes agrees at the same 3.1e-4. The arena on imported THP memory is Phase 13b.
3. *Not* the team's wake latency: a 3 ms idle gap before each expert costs 0.38 → 0.55 ms
   standalone, and a 20 ms block time on the worker's team only (`kmp_set_blocktime` on the
   worker thread, the reader team staying passive) plus a polling wait removes it standalone
   (0.37) — and changed nothing in the tier (161.7 vs 159.3 ms/token, 0.99 vs 1.08 ms per
   expert on one build). Reverted: it costs eleven spinning E-cores for nothing measured.
4. *Not* the frequency: the E-cores read 4.80 GHz through the decode window (`scaling_cur_freq`
   sampled every 2 s); the P-cores 5.3-5.6.
5. *Not* the pinned mapping itself: slots in pinned USM host memory measure 0.37 standalone
   (`IE_DS41_BENCH_PINNED=1`).

**The split erodes the VRAM cache** (gate 13 finding 2): a CPU-served expert never enters a
stream slot, so it is not there for the next token — stream-slot hits fall 42.9 → 36.7 (q* 0.30)
→ 6.8 (q* 0.0), the hit rate 61.6% → 58.9% → 46.5%, and the experts needing service (link + CPU)
rise 76.9 → 83.3 → 112.9. A third reason q* saturates, beside the shared host bandwidth and the
page tables; the accounting closes at every setting (static + stream hits + link + CPU + mmap =
240.0, e.g. 104.6 + 36.7 + 28.1 + 55.2 + 15.4). The link sits idle ~80% of the token at q* 0.30,
so filling the slots with the CPU-served experts in that idle time is a candidate, not a plan.

**Where the step goes now** (q* 0.30, 2,048 context, per token): moe 85-90 ms — the link's
504 MiB in the groups (~27 ms), the 15 disk experts' 276 MiB in the mmap group (25-45 ms), the
CPU's 50 ms overlapping both and sometimes the last to finish; attention 30; the shared expert 7;
ffn_pre 3; the head 2.4; setup / misc 4-5; engram 1.2.

## The levers, corrected (criterion 5)

docs/34 credited lever A ("CPU miss compute") with this phase's result; the isolation says:

| lever | measured | cost | under other CPU load |
|---|---|---|---|
| the reader team's block time (unnamed before) | **~26 ms/token** (183-189 → 159-160) | none; on by default | immune (158 at a 400% load) |
| the CPU q* split, q* 0.30 (`IE_DS41_CPU_MISS=1`) | **~9 ms/token** (159-160 → 151) | 12 E-cores while decoding | negative (169-194) |
| Phase 13b, the arena on huge pages | headroom ≤ the DMA the split removes minus the erosion: the CPU leg costs ~30 of a possible ~40 ms today, a 2x faster CPU leg is worth ~10-15 ms — an estimate ("Pineapple"), measured in docs/37 | load-time touch of 160 GiB | — |

The step at 2,048 tokens is now ~151-160 ms (6.3-6.6 tok/s). Its dense body — attention 30,
the shared expert 7, ffn_pre 3, the head 2.4, setup / misc 4-5, engram 1.2 — is ~46 ms with no
expert cost at all. 20 tok/s is a 50 ms step, so it needs that dense body AND every expert byte,
disk read and host cost of a 552B MoE inside the remaining **4 ms** — that is what is out of
reach single-stream on this hardware at this context (gate 13 corrected the earlier wording,
which read as if 46 ms alone precluded it); the honest ceiling with the levers left (expert parallel over both links, lever C's pinned RAM
against the disk, the huge-page arena, attention at T = 1) is ~10-12 tok/s. Prefill 400+ remains
the expert-parallel lever (Phase 14).

**Re-measured on the fixed build** (the block time unconditional; `scratchpad/p13/measure13.sh`,
idle machine): the split off **158.3 ms/token** (6.32 tok/s; 1378.3 MiB over the link, 42.9 stream
hits, 61.6%), the split on at q* 0.30 **149.6 ms/token** (6.69 tok/s; 504.3 MiB, 55.2 CPU experts
at 0.82 ms each) — the split's own value **8.7 ms/token**, the gate's 9. One side effect the split
carries at prefill: with the path on, the host thread and the reader team are pinned to the eight
P-cores for the whole run, and the replayed 2,048-token prefill measures 8.19 s against 7.56 s
off (+8%) — the pinning belongs to the decode step, not to the process; a follow-up.

## Correctness under the split (criterion 3)

`scratchpad/p13/regression_chain.sh`, one GPU job at a time, on the build with the split and the
entry-point env recipe (before the library-level block time): with `IE_DS41_CPU_MISS=1` — the
decode test PASS (0 FAIL lines: the four greedy tokens, the forced digits at all 40 layers, the
state hashes, the 36-step benches), the resident test PASS, the replay test PASS (20/20 prefixes,
KL(exact‖replay) < KL(exact‖truncated)), the forward test PASS, the tier test PASS (the routed
output tracks the reference at 3.1e-3, a second call bit-identical); the generate test's checks
all PASS (golden continuation, streaming, odd prompt, stop string, continuity at 12/130/512/1030
within 5e-2 with the same argmax, the replay test's four tokens) and its **210 ms bar FAIL at 527
ms/token** — the wait policy, fixed at the library level above. Without the env: the decode test
PASS and the generate test PASS at 190 ms/token (the loop's 173-181 with the machine's other
load), so the path off is unchanged. `cpu_moe_mxfp4_test` and `ds41_residency_plan_test` PASS.
