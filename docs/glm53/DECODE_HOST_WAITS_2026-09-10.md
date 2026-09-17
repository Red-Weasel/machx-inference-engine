# GLM-5.3-Flash decode: where the host waits are (2026-09-10)

Measured (this session, 22:08; `ie-glm5next-run --gpus 2 --ctx 32768 --ppl wiki.test.raw --prefill 16000
--chunk 1024 --ngen 64`, IE_G5_CPU_MISS=1 IE_G5_PIN_BANKS=1 IE_G5_PIN_MAX_GIB=95 IE_QUEUE_PROFILING=1):
116 ms/token at 16K depth. GPU busy 39 ms (dev 0: 19.1 ms in a 45.8 ms window; dev 1: 20.2 ms in a
69.7 ms window; stages serial). Host waits/token: stage 0 cpu-join 15.7 ms; stage 1 cpu-join 14.4 ms +
pool 27.9 ms + final 1.8 ms; MoE submit 2.9 + 3.8 ms. GPU kernels: expert GEMVs ~55%, g5_attend_sel
8.2% (3.2 ms/token), KDA 1%.

Read-only map by a second Claude session (00-inference-engine-2e, 22:20; line numbers = tree at 438e480):

## cpu-join (t_cpu_join, glm5next.cpp:3629-3634)
Under q* (IE_G5_QSTAR default 0.40, 3339-3342; per-stage IE_G5_QSTAR_A/_B 3343-3350) the first
round(q*·m) of a layer's m cache misses become GPU fills and the rest are computed on the host (3361-3372).
Host leg: cpu_worker_.submit at 3439 — one persistent thread + its OpenMP team (header 761-778),
sched_setaffinity core_lo..core_hi (4899-4911), team = cores in range or 8 (3436-3437); by default BOTH
stages' teams sit on the same P-cores (g5_auto_core_split 3424-3425, justified at 3413-3423). It
dequant-GEMVs Q4_K gate/up + Q5_K down per CPU expert from the pinned bank or the mmap (3384-3393,
3440-3476) into a host ring slot (kCpuAccRing 4; previous H2D waited at 3382); the main thread joins at
3632 (CpuWorker::wait 4965-4967) and issues a 16 KB async H2D + add (3650-3660). The join = CPU leg
minus that layer's GPU waves.

## pool (t_pool_copy)
At decode the miss fill takes the INLINE bounce path (2888-2910) when T <= 4, fill lanes are off and the
layer's bank is NOT pinned: wait for the pin-ring slot's previous H2D (2892-2895 = t_ring_wait), then
pread_fill_ (4405+, PreadPool::run is blocking) reads the expert's three tensors from the FILE into the
pinned slot ON THE MAIN THREAD (t_pool_copy, 2900-2903), then the H2D on copyq_ (2904). The 27.9 ms/token
on stage 1 is synchronous file-to-pinned staging for the ~40% of misses that go to the GPU on layers
whose banks were not pinned. Bypass already in the tree: IE_G5_PIN_BANKS (2878-2887: "ONE async DMA per
miss, zero host bytes"); selection at 825-920 pins layer by layer from layer_lo_ while
avail_gib >= need*pin_overhead + floor (floor 40 GiB, IE_G5_PIN_FLOOR_GIB — the memguard hook forbids
lower; cap IE_G5_PIN_MAX_GIB per stage; each success logs "layer L pinned", skips counted in n_skipped).
Each stage instance runs that rule against the RAM left after the other stage pinned, so stage 1 is
the one that gets skipped layers — matching pool showing up on stage 1 only. IE_G5_FILL_LANES moves the
pread to lane workers (mmap path only, documented stale-slice hazard) — pinning is the one to use.

## Stage hand-off
Plain mode is serial by construction: fwd lambda (glm5next_run.cpp:221-223) = stage 0 forward_range
with wide_out_host (D2H + q.wait(), 5080-5084) -> stage 1 forward_range + lm_head + q.wait()
(5106-5116) -> host argmax (736-750); token t+1's id exists only after that argmax and stage 0's first
op is its embedding (943): no token-independent stage-0 work exists. The tree already overlaps
speculatively: --pipedraft (602-724; auto when 2 stages and ngen >= 64): MTP drafts t+1 on card 0
(IE_G5_MTP_STAGE=head), stage 0 snapshots (694) and runs the draft in a std::async (689-699) while stage
1 runs token t (702); mismatch -> restore_state + serial redo (713-720). What limits it: ~55%
acceptance (a serial ~46 ms stage-0 redo on ~45% of rounds); both stages' CPU teams, pread pools and
main threads share the same 8 P-cores and DRAM bandwidth (3418-3420); and overlap can only hide stage 0
(~46 ms) under stage 1 (~70 ms), whose window is 71% host waits.

## Smallest changes, in order (measurable without code)
1. Pin stage 1's banks: raise IE_G5_PIN_MAX_GIB (check n_skipped and the per-layer "pinned" lines);
   expected pool 27.9 -> ~0, stage-1 window ~70 -> ~42 ms, token 116 -> ~88 ms serial.
2. Re-tune IE_G5_QSTAR_B: 0.25 was tuned "when host staging drowned the knob" (3338); with zero-host-byte
   fills the optimum moves toward more GPU fills and a shorter CPU leg, which shrinks cpu-join.
3. Only then re-measure --pipedraft: with stage 1 at ~42 ms its ceiling becomes ~46 ms/token.
4. If the join is still the long leg: stage-1 team on the E-cores while stage 0 keeps the P-cores under
   pipedraft (IE_G5_CPU_CORES_B; measured worse on 09-02, but with staging on the P-cores too).

## Follow-up (22:25): why the seven unpinned layers hurt twice, and the honest option ranking
- Stage 1's log: 15 layers pinned (61.2 GiB), 7 left on mmap (layers 38-44, all in shard 05, 28.8 GiB
  nominal). It stopped at the 40 GiB RAM floor (IE_G5_PIN_FLOOR_GIB; the memguard hook forbids lower),
  not the 95 GiB cap. Stage 0 pinned all 20 of its layers.
- Measured with mincore one minute into a run: 0.0% of those seven layers' expert tensors were in the
  page cache (pinning ~144 GiB nominal reclaims the file cache; the previous run's pages did not
  survive). Every first-touch expert is a 14.5 MB NVMe read (2-5 ms): that is the 27.9 ms/token of
  pool (GPU fills, main-thread pread) AND why stage 1's q* CPU experts cost 1.264 ms each vs 0.462 on
  stage 0 (the OpenMP GEMV page-faults the same file). DS4 benchmarks on the same box mmap a 165 GB
  file and evict whatever accumulated between runs.
- Per-layer q* (3 lines: qstar chosen by w.bank_pinned at 3351, the warming clamp at 3363 gated on
  bank_pinned, an IE_G5_QSTAR_MMAP env) is possible but, with qf == 0 on unpinned layers, their VRAM
  caches stop admitting experts (3636-3640) and the cold faults move into the CPU leg — worse from cold.
- Ranking: (1) founder lowers the floor so the last seven layers pin (pool -> 0, no hazard) — whether
  they actually FIT is unverified: pinned USM is invisible to /proc/meminfo, which is why the floor
  exists after the 2026-08-28 near-freeze; (2) IE_G5_FILL_LANES=1 or 2 moves the mmap-layer preads off
  the main thread (2016-2024, 2911+; lanes touch exactly the unpinned layers) at the documented
  stale-slice hazard (~1 per 160k fills, i.e. one per ~11k tokens at ~14 fills/token) — opt-in, founder
  call; (3) per-layer q*_mmap. Measurements queued: IE_G5_QSTAR_B 0.15/0.40/0.60 at 16K depth, then
  IE_G5_FILL_LANES=2.

## Measurements (22:22-22:30) and a correction
- IE_G5_QSTAR_B sweep, 16K depth, 64 tokens, unprofiled (this run pinned 21 + 14 layers, 8 on mmap):
  0.40 -> 10.27 tok/s (97 ms/token; stage-1 CPU experts 1593 at 1.603 ms each, join 1.35 s);
  0.15 -> 10.95 tok/s (91 ms/token, +6.6%; 2425 CPU experts at 1.067 ms, join 1.81 s).
  0.60 was killed mid-load by the Claude Code harness's low-memory watchdog (see below) — not measured.
- Correction (peer, 22:32, measured with the runner live): pinned USM IS subtracted from MemFree/
  MemAvailable, only unattributed. With the current pin set MemAvailable is 19.4 GiB of which 15.6 is
  reclaimable cache; pinning the last 7 layers needs ~39 GiB real. It does not fit; lowering the floor
  would only remove safety. DROP that option. The same number means those 7 layers can never be
  page-cache resident in steady state (19.4 GiB incl. cache < 28.8 GiB needed): every miss on them is a
  2-5 ms NVMe read of a 14.5 MB expert, on whichever leg takes it.
- Corrected ranking: (1) IE_G5_PREFETCH (opt-in; layer-ahead prediction, own pf ring + pool: setup
  2027-2035 / 2211, predictor rows 2595-2640, issue_pf 3796) — the only mechanism that puts an NVMe read
  UNDER the previous layer's compute; A/B at the same 16K line with a high q*_B. (2) Per-layer q* with
  q*_mmap = 1.0 (fills install the expert in VRAM so the next activation hits; the CPU leg installs
  nothing and stalls the team) — three lines (3351, 3363 warming clamp gated on bank_pinned, env parse).
  (3) Weight the VRAM slot budget by miss cost: IE_G5_SLOT_PROFILE's per-byte hit-count heap (2123-2158+)
  x (w.bank_pinned ? 1 : k) at the priority push. (4) IE_G5_FILL_LANES=2 hides only ~1 ms of a 2-5 ms read
  and carries the stale-slice hazard — measure for the record only.
- HAZARD: a GLM run launched as a Claude Code background command can be killed mid-run by the harness's
  low-memory watchdog (MemAvailable ~19 GiB is the run's normal steady state); the repo memory records a
  hard-killed GLM run producing a kernel bad-page wedge. This one exited cleanly (taint 12288 = O+E only,
  no B/D; no D-state or leftover processes). Do not launch GLM/DS4 runs as background commands from
  that harness; run them from a terminal (or foreground within the tool timeout).

## Prefetch estimate for the mmap layers (peer, 22:36; from the predictor code + recorded measurements, no new runs)
- The predictor (glm5next.cpp:2594-2640) runs layer L+1's OWN router on layer L's normalized input (depth 2:
  L+2's router), issue path 3757-3796 with its own two bounce buffers (pf_pin_[2], header 583-589) and pool;
  a prefetched slot counts as consumed when its first demand reader takes the fill event (3525-3530 /
  4748-4752); the runner prints issued/consumed (glm5next_run.cpp:864-869).
- Recorded inputs: depth-1 accuracy 71.4%, consumed ~62% of issued (docs/glm53/PORT_PLAN.md:324); depth-2
  63.2% (code comment 3804-3806); short-context effect "misses 107 -> 52/tok" (PORT_PLAN.md:370); on the
  09-02 G0 numbers prefetch raised the hit rate (85.1%) and still lost to q* because "bytes, not hits, are
  the objective" (CAMPAIGN_2026-09-02.md:57-59); per-layer prefetch gating was FALSIFIED then
  (PORT_PLAN.md:323) — when every layer was PCIe-bound. No accuracy measurement exists at 16K or per layer.
- Why the old verdict need not transfer: on pinned layers a miss is a ~0.6 ms DMA on the link the wasted 38%
  of prefetch bytes also use; on the 8 mmap layers a demand miss is a 2-5 ms NVMe read on the critical path
  (the serial pread chain, CAMPAIGN_2026-09-02.md:60-64) while the NVMe idles for ~88 of the 116 ms. There
  the objective is critical-path latency, not bytes.
- Estimate: hidden fraction ~= consumed (0.62) x in-time (0.5-0.8: one stage-1 layer of lead ~3.2 ms vs
  2-5 ms reads, two bounce buffers serialize the chain). Depth 1 as-is: ~30-50% of the mmap-layer miss
  latency, ~8-14 ms of the 27.9 ms/token pool (7-12% of the token) IF those layers' misses are routed to
  the fill path (q*_mmap = 1.0); with q* 0.40 only the GPU-fill 40% is addressable (3-6 ms). Depth 2
  (~6.4 ms lead): ~55-60% hidden, ~15-17 ms/token, at ~1.6x prefetch bytes on an idle NVMe. A deeper pf
  ring (4-8 buffers: header 585 + the allocation loop 2031-2035) lifts depth 1 toward that without the
  extra bytes.
- Before any code: one profiled run at 16K reading the runner's issued/consumed counts for the mmap layers
  only; and any prefetch on mmap layers must be gated to those layers (w.bank_pinned is available at the
  issue site, 3757: wn = layers_[pred_ln]) or it re-runs the falsified PCIe-bytes experiment on the pinned
  ones. A consumed prefetch warms nothing lasting (19.4 GiB available): latency hiding only.

## G1 measured (23:35-23:50): prefetch is the wrong lever at depth with unpinned layers
Same 16K line, IE_G5_QSTAR_B=0.15, detached launches (this run pinned 21 + 14 layers, 8 on mmap):
- A baseline: 10.03 tok/s (the 22:26 sweep run of the SAME config gave 10.95 — run-to-run spread ~9%, so
  single-run deltas under ~10% are noise; the "+6.6%" for 0.15 vs 0.40 is inside that band).
- B IE_G5_PREFETCH=1: 2.63 tok/s (3.8x SLOWER). 7182 prefetches issued for 64 tokens (112/token),
  60.5% consumed; the CPU leg shrank (stage 1: 423 CPU experts at 3.495 ms each — 2.8x slower than in A,
  NVMe contention) and decode collapsed: every prefetch on the 8 mmap layers is a 2-5 ms NVMe read through
  a two-buffer ring, and the ungated prefetch also fills the pinned layers over PCIe. Prefill also slowed
  (92.5 -> 82.6 tok/s).
- C IE_G5_PREFETCH=2: 1.91 tok/s (5.3x slower; 9991 issued for 64 tokens, 47.2% consumed; prefill 80 tok/s).
Decision: prefetch as implemented is closed at depth (rule 3). Gating it to the mmap layers would still
turn CPU-side work into serialized NVMe reads on the critical path; the objective on those layers is
fewer NVMe reads, i.e. a higher VRAM hit rate on exactly those 8 layers -> miss-cost-weighted slot
budget (IE_G5_SLOT_PROFILE x k for unpinned layers) is the next candidate, measured A/B/B/A.

## G2 measured (23:52-00:25): miss-cost-weighted slot budget — inside the noise, reverted
Code: IE_G5_SLOT_PROFILE priority x IE_G5_SLOT_MMAP_WEIGHT for unpinned layers (patch kept in
results/glm53-host-waits-2026-09-10/, not applied). 16K line, q*_B 0.15, decode tok/s:
P (profile dump) 8.49 | U 9.77 | W1 9.91 | W4 9.47 (9 unpinned that run) | W8 10.52 (9 unpinned) |
W4b 10.17 | Ub 10.12. Baseline median 9.95; weighted medians W4 9.82, W8 10.52 (single). The weighting
did move slots (unpinned layers 61..84 -> 32..138/147) and the stage-1 CPU cost per expert fell
(1.31 -> 0.94-0.96 ms) but the CPU-expert count rose (2425 -> 2850-2889: fewer slots on the pinned
layers) — a wash within the ~+-8% run-to-run band. Rule (+10% median) not met -> reverted.
Run-to-run spread on identical configs tonight: 8.49-10.95 tok/s, so single-run deltas under ~10% are
not evidence at this depth.

## The pin-overhead constant is stale (measured 00:30-00:42, 2026-09-11)
Standalone SYCL programs on this stack (oneAPI 2026.1, compute-runtime 26.22, kernel 7.0.12-p2pwl):
- sycl::malloc_host + host touch: 8 GiB -> MemAvailable drop 8.02 (1.002x); 8 GiB in 64 chunks 0.98x;
  16 GiB 1.002x; 100 GiB in 25 chunks, twice: 1.001x and 1.005x. 2 MiB alignment: 1.007x.
- GPU DMA reads and a kernel reading all of 64 GiB of pinned memory: +0.1 GiB, on both cards.
- sycl::malloc_device 16-24 GiB (device memset, then an H2D): 0.03 GiB of host RAM; the driver keeps
  ~2 GiB of staging after a 2 GiB H2D from pageable memory (persists after free).
So pinned USM is 1.0x, not 1.37x: the kPinOverhead=1.37 default (glm5next.cpp ~875-886, from the
2026-09-03 "160 GiB pinned consumed ~219" note) charges each layer 37% more than it costs, and that is
why stage 1 stops 7-8 layers short of pinning everything while the box has the RAM: 222 available - 173
GiB of banks = 49 >= the 40 GiB floor. The ~57 GiB "unattributed" in the 22:29 in-run snapshot is
therefore not the pinning; it is still unexplained (candidates: load-time transients, the driver's
pageable-H2D staging pool, or the runner's own buffers) and is being watched at 5 s resolution in G3.
G3 = the existing knob IE_G5_PIN_OVERHEAD=1.0 vs the default, floor untouched, A/B/B/A at 16K.

## ROOT CAUSE of the phantom RAM (2026-09-11 00:50): multi-device contexts mirror VRAM into host RAM
Standalone measurement (fdinfo drm-resident-gtt + MemAvailable, this box: oneAPI 2026.1, compute-runtime
26.22.38646.7, xe on 7.0.12-p2pwl):
- ONEAPI_DEVICE_SELECTOR=level_zero:0 (one device visible): malloc_device of 1, 2, 3, 4, 6, 8, 12, 16, 18,
  20, 24 GiB single allocations and 1x20 / 2x10 / 4x5 / 8x2 GiB: host RAM taken 0.01-0.07 GiB, resident-gtt 0.
- level_zero:gpu (both B70s visible): 16 GiB x1 -> 16.00 GiB of host RAM, resident-gtt +16; 20 GiB x1 ->
  20.08; 1 GiB x20 -> 20.13 GiB. Same in an explicit two-device sycl::context and in a plain
  sycl::queue(device), because queue(device) binds to the platform's DEFAULT context = every visible device.
So with two cards visible, every device allocation carried a resident system-memory placement: GLM's
2 x 22.5 GiB expert caches + dense weights + caches (~57 GiB, exactly the unattributed amount in the
22:29 snapshot), and likewise every two-card DeepSeek / split-model run. The "1.37x pin overhead" was
this mirror, never the pinning (pinned host USM measured 1.00x at 8-100 GiB).
Fix (src/core/allocator.cpp DeviceAllocator::init): the queue is created in an explicit single-device
sycl::context(picked). init_with() (the fleet's shared context, the P2P context) is unchanged and still
pays the mirror — the TP/split/P2P paths need a cross-device context for their host-USM handoffs, so
their weights/KV keep a system placement until a VRAM-only allocation property is found (open question).
G4 = GLM 16K line with IE_NO_P2P=1 (per-device allocators) after the fix, with a memory watch.

## G4 measured (00:54-01:03): per-device contexts — the mirror is gone and GLM is +45% at 16K
Same 16K line (q*_B 0.15, IE_G5_PIN_MAX_GIB=120, no IE_G5_EP so both stage allocators use the fixed
DeviceAllocator::init), memory watch every 5 s:
- C1: stage 0 21 layers pinned, stage 1 **22 layers pinned, 0 left on mmap** (was 14 + 8 on mmap).
  Prefill 16000 tok 156.5 tok/s (was 66-106 across the seven G2 baseline runs). Decode **14.45 tok/s** (was
  8.49-10.52, median 9.91, across those seven runs). Stage-1 CPU experts 0.563 ms each (was 1.3-1.5). MemAvailable steady at 42 GiB during
  the run (min 42; the floor is 40) versus 16-18 GiB before — i.e. ~25 GiB more RAM AND 29 GiB more
  pinned, ~57 GiB recovered as measured.
- C1b: appended below.
Mechanism (peer research, 00:58, documented in intel/llvm, unified-runtime and compute-runtime sources):
the Level Zero V2 adapter (default on Xe2+) makes every USM device allocation resident on all peer
devices in its context; NEO satisfies peer residency by exporting the BO as a dma-buf and importing it on
the other card; xe's dma-buf import migrates the BO to system memory when the PCI pair is not P2P-capable
for dma-buf (xe_dma_buf.c: pin -> xe_bo_migrate(XE_PL_TT) unless every importer allows peer2peer). The
owner's first use validates the BO back into VRAM and the system pages go to the TTM page pool
(pages_limit 124.5 GiB here), unattributed and outside MemAvailable until reclaim pressure — which is
why a used cache "lives in VRAM" while the RAM stays gone. Escape hatch for paths that need a shared
context (fleet TP/split, P2P): SYCL_UR_L0_RESTRICT_USM_RESIDENCY_TO_P2P=1 (residency only on peers with
peer access explicitly enabled) and enabling peer access only around the buffers that actually cross
cards, not at context creation (allocator.cpp:102-103, 355). Legacy adapter: UR_L0_USM_RESIDENT=0x001.
[RETRACTED 06:55: the env flag was measured inert on this stack in every context kind — see the close-out.]

## Phase close-out (2026-09-11 01:05-06:50): C1b, DeepSeek two-card check, process default
- C1b (repeat of C1): decode 14.45 tok/s, prefill 156.4 tok/s, 22 stage-1 layers pinned, min MemAvailable
  41 GiB, text md5 identical to C1 and to all seven G2 baseline runs.
- DeepSeek4Runtime had the same bug (deepseek4.cpp: q_store_ queues built as sycl::queue(device) -> default
  two-device context); fixed the same way. Two-card check on the fixed binary (ie-ds4-bench, same line as the
  09-09 campaign): tg 33.8 tok/s at 128 ctx, 25.0 at 4096 (baseline 32.73-32.95 / 24.53-26.57, the 19:49-20:17
  p6/g6 runs), pp 144/330/655 at 128/512/4096; min MemAvailable during the run 82 GiB = 219 GiB before the run
  minus the 2 x 73.585 GB pinned expert arena (137.07 GiB, ds4_ctx_bench.err), i.e. 0 GiB unattributed — a live
  mirror of the two cards' VRAM (~2 x 32 GB) would have left ~22 GiB; batch PPL 4.1393 with the NLL dump
  byte-identical to the pre-fix dump.
- Shared-context paths (fleet TP/split via init_with, the qwen4exp P2P context) cannot use a single-device
  context. SYCL_UR_L0_RESTRICT_USM_RESIDENCY_TO_P2P=1 was tried as a process default and measured INERT,
  so it was removed again (06:55): 27B Q8 split serve (--gpus 2, fleet shared context, no peer access since
  engine.cpp passes enable_p2p_transfers=false) with the env forced 0 vs 1: 28 GiB of host RAM taken at
  load both ways, 16K prefill 674/672 tok/s, decode 14.6/14.6 tok/s, identical text. Standalone
  (scratchpad shared_ctx_vram2, 2 x 4 GiB device allocations): two-device context WITHOUT peer access ->
  7.97 / 8.06 GiB of host RAM with the flag 0 / 1; WITH peer access -> 8.05 / 8.05; per-device contexts ->
  0.02 / 0.02. The "escape hatch" in the previous section was a source-reading inference, never a
  measurement — retracted. What the fleet's shared context buys today: the all_reduce_sum_fp16 pinned-slot
  exchange (allocator.cpp ~626: each card DMA-writes its half into the peer's pinned slot, then H2D into its
  own device scratch, host-observed) instead of the fully host-staged D2H/add/H2D reduce. The existing
  IE_NO_SHARED_CTX=1 fallback (per-device contexts) is the only known cure for the mirror on this path;
  its cost is measured in the next section.

## Fleet context kind on the 27B split (06:54-06:58): per-device contexts cost nothing, save 29 GiB
27B Q8 split serve, --gpus 2 --ctx 20000, 15396-token prompt, 96 greedy tokens, A/B/B/A (scratchpad
q27_sctx.log; memory watch every 5 s):
| leg | context | host RAM taken at load | load | prefill tok/s | decode tok/s | text md5 |
| shared1 | shared (default) | 29 GiB | 19 s | 675 | 14.6 | 5c9aa6aeec |
| perdev2 | IE_NO_SHARED_CTX=1 | 0 GiB | 16 s | 682 | 14.6 | 5c9aa6aeec |
| perdev3 | IE_NO_SHARED_CTX=1 | 1 GiB | 16 s | 680 | 14.6 | 5c9aa6aeec |
| shared4 | shared (default) | 29 GiB | 18 s | 675 | 14.6 | 5c9aa6aeec |
The pipeline-split paths (qwen35/qwen3moe/qwen35moe/qwen3next/dense splits) never call the all-reduce, so
the shared context buys them nothing and costs the whole two-card weight set in host RAM. The tensor-
parallel paths (dense and gptoss by default with --gpus 2; qwen35 and qwen3moe behind IE_QWEN35_TP /
IE_QWEN3MOE_TP) use the shared context for the all-reduce pinned-slot exchange; measured next (q27_tp.log)
before deciding whether the flip is global or split-only.

## Fleet context kind on the 27B tensor-parallel path (06:58-07:00): the shared context earns its RAM there
Same prompt and measurements, IE_QWEN35_TP=1 (scratchpad q27_tp.log):
| leg | context | host RAM taken at load | prefill tok/s | decode tok/s | text md5 |
| tp_shared1 | shared (default) | 38 GiB | 503 | 14.4 | 5c9aa6aeec |
| tp_perdev2 | IE_NO_SHARED_CTX=1 | 1 GiB | 280 | 14.3 | 5c9aa6aeec |
Prefill -44% host-staged (the per-layer all-reduce of T x H fp16 goes D2H/add/H2D instead of the pinned-slot
exchange); decode within noise. Decision (rule pre-set in the gate criteria): the shared context stays the
default of DeviceFleet::init (every TP path and every tool unchanged); the engine's four pipeline-split
call sites (qwen35 split, qwen3moe split, qwen3next, qwen35moe split; engine.cpp) pass shared_ctx=false and
get per-device contexts. IE_NO_SHARED_CTX=1 still forces per-device everywhere. Still mirroring, by design or
unmeasured: TP fleets (38 GiB on the 27B), the qwen4exp (Flash-Next) P2P context (needs a two-device context
for the push handoff; its two-card weight set is mirrored — open item), and IE_G5_EP on GLM.

## G5 measured (07:29-08:01, commit 83554c2): q*_B sweep with every bank pinned — flat; pipedraft now 1.14x
Same 16K line (IE_G5_CPU_MISS=1 IE_G5_PIN_BANKS=1 IE_G5_PIN_MAX_GIB=120, IE_G5_QSTAR_A=0.5 explicit, 2 cards,
detached, memory watch every 5 s; scratchpad glm_levers.log); criteria pre-registered in
scratchpad/gates/glm_levers_criteria.md (rule: a new default only if both runs of a candidate beat both Q015 runs by
>= 10% with identical text). Every run: exit 0, stage 1 "22 layers pinned, 0 left on mmap", min sampled
MemAvailable 42-43 GiB, text md5 49c2667bab (identical across all q*_B values and the serial run — the CPU and GPU
expert legs agree on these 64 tokens).
| run | q*_B | pipedraft | decode tok/s | prefill tok/s | B window (63 rounds) | drafts accepted | A-redo |
| Q015 | 0.15 | auto | 14.40 | 154.4 | 2.84 s | 33 (52.4%) | 1.25 s |
| Q025 | 0.25 | auto | 14.36 | 154.6 | 2.76 s | 32 (50.8%) | 1.29 s |
| Q040 | 0.40 | auto | 14.43 | 154.6 | 2.60 s | 30 (47.6%) | 1.36 s |
| Q060 | 0.60 | auto | 14.11 | 153.6 | 2.67 s | 31 (49.2%) | 1.33 s |
| Q080 | 0.80 | auto | 14.03 | 152.8 | 2.83 s | 31 (49.2%) | 1.35 s |
| S015 | 0.15 | --no-pipedraft | 12.59 | 156.1 | — | — | — |
| Q060b | 0.60 | auto | 14.67 | 155.2 | 2.51 s | 31 (49.2%) | 1.33 s |
| Q040b | 0.40 | auto | 14.71 | 155.3 | 2.49 s | 30 (47.6%) | 1.36 s |
| Q015b | 0.15 | auto | 14.86 | 155.2 | 2.68 s | 33 (52.4%) | 1.25 s |
Reading: 14.03-14.86 across every q*_B, and the last three runs are the three fastest regardless of the knob — the
knob is inside the run-to-run drift (~+-3%) now that no leg touches the NVMe. Rule 3 not met -> no code change;
the code default 0.40 stands and 0.15 is no longer a recipe requirement. The draft acceptance varies 47.6-52.4%
with the leg split while the verified text is identical: the MTP draft (card 0) sees leg-dependent rounding in
stage 1's hidden state that the main argmax does not. Pipedraft's gain is 1.14x (12.59 -> 14.4-14.9) versus 1.24x
pre-fix: stage 1 got faster, so less of stage 0 can hide under it.
Where the token goes now (Q015b, 4.31 s / 64 = 67 ms): stage-1 window 2.68 s = 42 ms/round (dev-1 GPU busy was
~20 ms in the 22:08 profile, so ~half of B is still host: cpu-join, fill issue, MoE submit), serial stage-0 redo
1.25 s = 20 ms/token (a full stage-0 pass on the ~48% of rounds whose draft missed), A-wait 0.31 s = 5 ms.
Next levers, each a design decision: (1) the redo — draft 2 candidates (top-2 of the MTP head) and run stage 0 as a
T=2 batch so a second-choice hit avoids the serial redo (acceptance ~50% -> est. 65-70%, saving ~6-8 ms/token
if the T=2 stage-0 pass costs ~the T=1 pass on a bandwidth-bound MoE); (2) B's host half — a per-layer profile at
16K with all banks pinned (IE_QUEUE_PROFILING=1 disables auto pipedraft, so profile the serial line) to split
cpu-join vs fill issue vs submit; (3) prefill: 155 tok/s at 16K is the other daily-use number.
(Gate finding on this sweep, 08:05: the eight pipedraft runs had stage-1 layer 44 on mmap — the head stage also pins
the MTP kit and the 1.37 pin charge put the last layer 1-3 GiB out of reach at 217-218 GiB launch RAM; C1/C1b had
launched at 220. The sweep is re-run after the 1.37 -> 1.0 default change; see the next GLM section.)

## G6 (08:11-08:43): pin charge 1.37 -> 1.0 default; the sweep re-run with every bank pinned — still flat
Change: glm5next.cpp kPinOverhead default 1.37 -> 1.0 and glm5_memory_policy.hpp pin_overhead 1.37 -> 1.0 (server
mode); the unit test's pin-budget expectation updated to the exact new value (88 GiB for 226 - 10 - 40 over two
stages). Same nine runs, DEFAULT env (no IE_G5_PIN_OVERHEAD), launch guard raised to >= 215 GiB with wait/retry
(never waited). Every pipedraft run: stage 0 "21 layers (86.8 GiB), 0 left on mmap" (MTP kit pinned), stage 1
"22 layers (90.0 GiB), 0 left on mmap"; S015 20 + 22; all exit 0; text md5 49c2667bab x9.
| run | q*_B | pipedraft | decode tok/s | prefill tok/s | B window | drafts accepted | A-redo |
| Q015 | 0.15 | auto | 14.50 | 156.3 | 2.76 s | 33 (52.4%) | 1.26 s |
| Q025 | 0.25 | auto | 14.87 | 156.6 | 2.52 s | 32 (50.8%) | 1.29 s |
| Q040 | 0.40 | auto | 14.47 | 156.2 | 2.50 s | 30 (47.6%) | 1.37 s |
| Q060 | 0.60 | auto | 14.50 | 156.2 | 2.49 s | 31 (49.2%) | 1.33 s |
| Q080 | 0.80 | auto | 14.47 | 156.1 | 2.61 s | 31 (49.2%) | 1.35 s |
| S015 | 0.15 | --no-pipedraft | 12.61 | 156.3 | — | — | — |
| Q060b | 0.60 | auto | 14.50 | 156.1 | 2.49 s | 31 (49.2%) | 1.33 s |
| Q040b | 0.40 | auto | 14.52 | 156.2 | 2.49 s | 30 (47.6%) | 1.37 s |
| Q015b | 0.15 | auto | 14.63 | 156.2 | 2.72 s | 33 (52.4%) | 1.25 s |
Reading: with layer 44 pinned the band is 14.47-14.87 (was 14.03-14.86 with it on mmap): the last mmap layer was
worth ~0.3 tok/s at most, and q*_B stays flat within +-1.5% — rule 3 (>= 10%) not met, code default 0.40 stands.
Pipedraft 1.15x (12.61 -> 14.5-14.9). Memory: the sampled MemAvailable minimum was 39 GiB (integer-truncated, i.e.
39.0-39.99) on every run: the 40 GiB floor is enforced before each pin, and the run's own transients (~1 GiB) land
after the last pin — the pre-fix runs sat at 42-43 only because a 4.4 GiB layer was NOT pinned. Whether the floor
should include a transient reserve is a policy question for the founder (the floor value itself is memguard-protected);
no freeze, no allocation failure, all runs exited 0.
Where the token goes (unchanged): B window 40-44 ms/round, serial stage-0 redo ~20 ms/token on ~50% draft misses,
A-wait 5-8 ms. Next levers are design decisions, listed under G5.

### G6b (08:47-08:51): a 2 GiB transient reserve in the pin check — the floor and the last layer exclude each other
Gate verdict on G6: FAIL on the memory clause as written (25/377 samples at 39.x GiB). Remedy tried (pre-declared):
pin check = need x 1.0 + floor + 2 GiB reserve, one confirmation run (Q025b, same line, 218 GiB at launch): stage 0
21 layers pinned, stage 1 **21 layers, layer 44 left on mmap**, decode 14.09 tok/s, low-water 43 GiB, exit 0, text
identical. Arithmetic: both stages' banks with the MTP kit = 176.8 GiB; 218 - 176.8 - ~1-2 GiB of run transients =
39-40 GiB — with every bank pinned the box sits exactly at the floor, so ANY reserve above the floor unpins layer 44
at today's launch RAM, and the old 1.37 charge (4.38 x 0.37 = 1.6 GiB) had been doing the same thing by accident.
What the last layer is worth: 14.47-14.87 tok/s pinned (G6) vs 14.03-14.86 (G5) and 14.09 (this run) with it on
mmap — at most ~0.3 tok/s, inside the run-to-run band.
DECISION FOR THE FOUNDER (the 40 GiB floor is their memguard-protected rule): (A) floor = true low-water mark
(reserve kept, this tree): safe, layer 44 stays on mmap at ~218 GiB launch RAM, decode ~14.1-14.5; (B) floor =
pin-time check only (reserve removed, kPinOverhead 1.0): every layer pins, low-water 39.x GiB, decode ~14.5-14.9;
(C) status quo 1.37 (behaves like A by accident). The 1.37 -> 1.0 constant is correct under every option; only the
reserve line differs. Left uncommitted in the tree as (A) pending the answer; the G5/G6 measurements above stand.
Two more, from the peer (08:53): (C') reserve + IE_G5_NO_PIN_MTP=1 (existing knob: the ~4 GiB MTP-kit bank stays on
page cache, layer 44 pins) — MEASURED (Q025c, 08:53-08:57): stage 0 20 layers, stage 1 22 layers, low-water 43 GiB,
but the draft step slows (mtp 0.74 s vs 0.28 per 63 rounds, A-wait 0.66 vs 0.3) and decode drops to **13.27** —
worse than A; (D) free ~2-3 GiB of host RAM from agent tooling before a GLM launch (with no engine running the box
shows ~30 GiB used: a codex process 4.0 GiB, three codebase-memory MCP servers 1.3+1.0+0.9, others) — launch RAM
218 -> ~221-224 pins layer 44 with the reserve intact at no model cost; the founder's environment, not an action.
Summary for the decision: A 14.1 (safe), B 14.5-14.9 (39.x GiB low-water), C' 13.3, D = A's safety with B's speed
if the RAM is there.
FOUNDER DECISION (2026-09-11 12:08): A — keep the 40 GiB floor strict. Committed as the 1.0 pin charge + 2 GiB
transient reserve; the recipe for D (free tooling RAM before a GLM launch) stands as the way to get the last layer
pinned under the same floor.

## Flash-Next (qwen4exp) two-card handoff: IE_P2P=1 versus the default bounce (08:02-08:10)
`ie serve <Flash-Next UD-Q4_K_XL> --ctx 20000 --gpus 2`, 15396-token prompt sent twice per leg (second = prompt
cache), 96 greedy tokens, A/B/B/A (scratchpad q4e_ctx.log; criteria pre-registered in
scratchpad/gates/q4e_ctx_criteria.md):
| leg | IE_P2P | host RAM taken at load | prefill tok/s | decode tok/s (1st / cached 2nd) | text md5 |
| dflt1 | unset | 78 GiB | 352 | 27.0 / 32.0 | c622921475 |
| p2p2 | 1 | 138 GiB | 356 | 27.5 / 32.0 | c622921475 |
| p2p3 | 1 | 139 GiB | 356 | 27.6 / 32.0 | c622921475 |
| dflt4 | unset | 80 GiB | 465 | 28.2 / 31.9 | c622921475 |
The default's 78-80 GiB is the pinned host expert bank by design (qwen4exp.cpp host_bank, ~72 GiB) plus workspaces;
IE_P2P=1 adds a 60 GiB mirror (both stages' adaptive VRAM expert caches + dense weights in the two-device P2P
context; the peer's read-only estimate was 52-55). Decode is within 1-2% either way (cached-prompt decode 32.0 on all
four legs); prefill is noisy across legs (352 vs 465 on the two default legs) and the default is not slower. Nothing
the founder launches sets IE_P2P (grep of scripts/, README, SERVER_CONTROLS, QUICKSTART, docs, profiles, units), so
day-to-day Flash-Next serving already takes the per-device path and is mirror-free after 83554c2. Decision (rule 4,
gate PASS): no code change; IE_P2P=1 is not a serving recommendation (60 GiB for <= 1-2% decode); its 5%
pipelined-prefill win in tools/qwen4exp_run2.cpp stays a tool-level opt-in. The decision rests on decode parity at
16K depth plus the measured mirror, NOT on prefill: the two default legs differ by 32% between themselves (352 vs
465; 465 matches the certified pp 467-468 line, and the server logs show identical caches and no errors), so no
default-path prefill figure is quoted from this A/B. Gate hypothesis, unverified: the 26.8 GiB per-layer
token-embedding table (host mmap, read during prefill) was page-cache cold for dflt1 (right after the GLM sweep
reclaimed the cache) and for the p2p legs (~79 GiB left for cache), warm for dflt4.
