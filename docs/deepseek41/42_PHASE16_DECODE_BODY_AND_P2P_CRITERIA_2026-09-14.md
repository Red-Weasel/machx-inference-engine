# V4.1 port — Phase 16 gate criteria: the decode step's body, and P2P for the EP staging (written BEFORE the build)

**Where things stand (docs/41, the re-baselined text):** prefill pp2048 389 warm / 340 cold tok/s
with EP + the live cap + the expert file (goal 400+); decode 111 ms/token = 9.0 tok/s (goal 20,
out of reach single-stream). The two links now stream at ~18.5 GB/s each during a pass (85% of
their 22), so the prefill wall is close to the links themselves; the decode step is ~46 ms of
dense body (attention ~30 at 2,048 ctx) plus the slower card's share of the expert misses.

## Step 0 — measure before building (no GPU code changes)

(a) **The decode kernel profile at T = 1** with the engine's KernelProfiler (`IE_QUEUE_PROFILING=1`
on the decode bench, as Phase 11 did): the top kernels by time per token at the 2,048-token
context — attention (the window, the indexer, the sparse gather), the hc mixes, the router, the
shared expert, the head; and the host gaps between them. The phase's target is whichever term
the profile names, bounded to one kernel change at a time (Phase 11's method: one change, one
build, kill switch, measured, kept or reverted).
(b) **P2P staging feasibility** (`~/ds41_work/p2p_ipc_probe.cpp`): a buffer allocated in
card 1's single-device context, exported with `zeMemGetIpcHandle` and opened in card 0's
single-device context in the same process, pushed into by card 0's queue (the fabric is
push-only; peer reads are broken on this board), the bytes verified on card 1, and the host-RAM
cost of a 1 GiB IPC-opened buffer read from /proc/meminfo (a mirror would show ~1 GiB). If it
works: the EP staging (x out, the rows back — ~0.3-0.4 s of a pp2048 pass through pinned host
buffers today) becomes two pushes per layer with no shared context and no mirror. If it does not:
P2P staging is off the table until the stack changes, and the doc says so.
(c) **The cold pass with the expert file** (docs/41's open item, +0.75 s on the first pp2048 pass
after load): two resident runs, the pass breakdown of each — is it the imported staging's first
DMA mappings (a one-time cost per load, which serving never sees) or something per pass?

## Scope, in order

1. The decode term the profile names, one bounded change (kill switch, measured, kept/reverted).
2. If 0(b) works: the EP staging over IPC-opened buffers (`IE_DS41_EP_P2P=1`), bit-identical
   (the same rows land, the dumps at 0.0), measured on pp512 / pp2048 and decode.
3. The CPU miss split under EP: the two tiers' workers on a core partition — with
   `IE_DS41_EP=1 IE_DS41_CPU_MISS=1` and no `IE_DS41_CPU_CORES`, card 0's tier takes E-cores 8-13
   and card 1's 14-19 (six each; `Ds41ExpertTier::set_cpu_cores` before init; an explicit
   `IE_DS41_CPU_CORES` is shared as given), each tier's host thread and reader team pinned to
   its own complement as today — measured at decode against EP alone (111 ms) and against the
   split alone (Phase 13's ~9 ms), with the E[max] line: the split's experts come off the slower
   card's misses or they buy nothing. The prefill cost of the pinning (docs/35: readers confined
   to the P-cores, +8%) is measured too; if it holds, the pinning moves to the decode step.

## Pass criteria

Step 0 reported whatever it says; every change bit-identical to the path before it (the dumps)
or its numerical difference stated with the bars judging; the whole test set PASS; the founder
table updated with each measurement; defaults unchanged unless a measured, gated win says
otherwise.

## Explicitly NOT in this phase

A bigger tail window (the system volume's 50 GB floor decides, the founder's call); FP8-resident
dense; attention beyond the one bounded change the profile justifies.
