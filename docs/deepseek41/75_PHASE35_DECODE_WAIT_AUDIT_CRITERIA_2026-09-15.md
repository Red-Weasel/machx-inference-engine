# V4.1 — Phase 35: the decode wait audit (criteria before the build)

## The founder's observation, and what the code says

"The GPUs are running at around 50 % utilization." The profile agrees: at the 2k prompt the named kernels are
35.5 ms of an 83 ms token, and the long test's stage split at 32k puts the attention STAGE at 22.3 ms with ~9 ms of
kernels in it, the MoE stage at 42.6 with ~9 ms of expert GEMMs (the rest of that one is expert bytes over PCIe and
the CPU leg's join -- a different problem, docs/70).

The attention stage's gap has a mechanism in the code: the per-layer decode path calls `.wait()` after almost every
launch -- `ds4_rms_norm`, the mask builds, `ds4_rope_apply`, `ds4_indexer_topk`, `ds41_sort_picks_asc`, the ring
snapshot, `o_b`, `ds4_hc_mix` (twice), the second `ds4_rms_norm`, `ds4_router_topk` -- about **12 per layer, ~480
per token**. On an in-order queue a wait after a launch serializes the host with the device: the next kernel cannot
be enqueued until the previous one has finished, so every launch's latency (tens of microseconds on this stack) is
paid in full and the device idles between kernels. Exactly one of those waits feeds a host decision: the routing
readback (`h_idx` / `h_w`) that drives the expert tiers. The rest are ordering habits the in-order queue already
guarantees, or diagnostics.

## The change

Remove the per-layer waits that feed no host read, keeping: the routing readback; any wait before a HOST write into
a buffer a queued kernel may still read (audited one by one); and the waits when `IE_DS41_STAGES=1` is on, because
the stage timer needs them (the stage split becomes a diagnostic mode, not the default). Bit-identical by
construction: the same kernels in the same order on the same in-order queue.

## Criteria

1. **Bit-identity:** the profiled decode test's criterion lines identical to Phase 34's (`p24/dx_decode.log`);
   `multi` 38/38, `rollback` 74/74, `dspark` 153/153.
2. **Speed, like for like:** the 2k prompt's wall (profiled decode test) and the 32k long test's decode rate,
   against Phase 34's 83.2 ms/token and 75.7 ms/token. The named-kernel totals must not move (same kernels);
   only the gaps may.
3. **The stage split still works** under `IE_DS41_STAGES=1` (waits restored in that mode), and the numbers it
   prints are recorded next to the default mode's so the gap it was measuring is visible.
4. If a removed wait turns out to be load-bearing (a test fails), it goes back with a comment saying what host
   access it protects -- not the whole set.

## Results

16 wait sites routed through `hostsync` (`IE_DS41_STAGES` set, or a probe attached); kept as they were: the routing
readback, the logits readback, host-vector uploads, the ratio-2 group bookkeeping, the expert-tier / EP waits.

| criterion | result |
|---|---|
| 1. bit-identity | decode test 28/28, criterion lines **identical** to Phase 34's (`p24/w35_decode.log` vs `dx_decode.log`) |
| 2. 2k prompt wall (profiled decode test) | **80.8 ms/token = 12.37 tok/s** from 83.2 = 12.01; named kernels 35.5 = 35.5 (same kernels), min step 70.3 -> 67.0 |
| 1. row identity | multi **38/38**, rollback **74/74**, dspark **153/153** |
| 2. 32k long test, 512 decode steps | default (waits out) **67.1 ms/token = 14.89 tok/s**; `IE_DS41_STAGES=1` (waits in, the Phase 34 behaviour) 69.2 = 14.44 -> **-2.1 ms/token** |
| 3. stage split under `IE_DS41_STAGES=1` | intact and sane: layers 64.9 = attn 21.6 + ffn_pre 3.1 + moe 39.0 + shared 0.4 (default mode prints an enqueue-time attribution, meaningless by design) |

**A caveat on the headline.** Phase 34's 32k figure (75.7 ms/token) was measured with `IE_QUEUE_PROFILING=1` on --
the A/B script carried it -- and profiling costs ~6 ms/token here. The like-for-like for THIS phase is the stages
arm against the default arm on the same binary: 69.2 -> 67.1. The unprofiled decode rate at 32k is now
**14.89 tok/s**, and the unprofiled Phase 34 equivalent was 14.44, not 13.21.

**Verdict: PASS -- bit-identical, -2.1 to -2.4 ms/token (3 %) at 2k and 32k.** Smaller than the ~12 waits per layer
suggested, because most of those launches were already short and the device had other queued work; the idle the
founder sees is mostly elsewhere, see below.

## Where the idle half actually is (the founder's question)

At 32k with the split intact (stages arm), per token: attention stage 21.6 ms of which ~10 is kernels (the dense
projections, the indexer, the 1.8 ms XMX attention) -- so ~11 ms of launch gaps and the strips' host-side
bookkeeping remain there; ffn_pre 3.1; the MoE stage 39.0 of which ~9 is expert GEMMs and ~30 is waiting for
expert bytes -- 513 MiB/token from the pinned host tier at 17-20 GB/s over PCIe, plus the CPU leg's join; head
2.4. So of a 67 ms token the device computes for roughly 30 ms: **the idle is ~30 ms of expert-byte wait (the
hardware wall of docs/70: routing concentration vs VRAM, prefetch measured impossible at 19 % predictability) and
~10 ms of attention-stage gaps**. The second is the next kernel-side target: fewer, fused launches in the attention
stage (the dense projections are 8-10 launches per layer at 10-40 us each).

**Correction (2026-09-16, measured):** the "~10 ms of attention-stage gaps" above was an estimate and it is wrong.
From the Phase 34 32k log, which carries the stage split and the kernel table in one run: the attention stage's
22.3 ms holds ~20.7 ms of kernels (q_b 4.18, o_b 3.96, o_a 3.10, indexer top-k 2.45, the XMX attention 2.0, q_a
0.68, engram 0.67, indexer score 0.65, hc_mixes ~1.0, norms/rope/masks ~1.5, kv/idx_q ~0.55), so its launch gaps
are **~1.6 ms**, and launch fusion is not worth a phase. The unnamed 37 ms of that token is the MoE stage's wait:
its 42.6 ms holds ~15.5 of kernels, and the log names the critical path -- the CPU leg: 44.7 experts/token at
0.633 ms each (28.3 ms compute, 35.2 ms span), the GPU joining 2.7 ms after its own groups. 0.633 ms per 17.7 MB
expert is 28 GB/s of CPU throughput on a DDR5 part; that leg is the next candidate, not the launches.
