# V4.1 port — Phase 18 results: the decode step's overhead terms (completed after the 06:40 reboot)

**Criteria:** `46_PHASE18_DECODE_OVERHEAD_TERMS_CRITERIA_2026-09-14.md` (written before the build). **Starting
line:** the Phase 17b path (`IE_DS41_EP=1 IE_DS41_EXPERT_FILE IE_DS41_DENSE_FP8=1 IE_DS41_FP8_PACKED=1`, the
live cap, the 2,048-token context): 95.9 ms/token (docs/45), reproduced at 95.7-96.2 on every build of
this phase. Gate 17 (Phases 17 / 17b) was being run by the previous session's evaluator when this
phase started; its verdict had not landed when card 1 went down (below). Scripts and logs:
`~/ds41_work/p18/` (`p18_bench.sh`, `p18_correct.sh`, `freq_sampler.sh`, `p18z_*.log`).

## What was built (the three terms of docs/46), and what each measured on one build

All arms on ONE binary (`ie-ds41-decode-test` md5 `2baa98bb…`; every arm sets all three switches
explicitly, so the defaults do not enter), `IE_DS41_BENCH_ONLY=long`, steps 4-35 after the replayed
2,048-token text, the bytes and the routing identical in every arm (1,262.9 MiB pinned / 22.4 MiB disk
per token, experts 116.3 / 122.5 / 1.2 static / pinned / mmap, 52.0 stream hits, hit rate 70.1%; EP:
the slower card 153.9 of 240, link misses 55.2 / 15.3, the remote tiers' wall 36.1-36.8 ms):

| arm (`IE_DS41_SHARED_EARLY` / `IE_DS41_SETUP_FAST` / `IE_DS41_FP8_PACKED`) | ms/token | tok/s | step-1 columns: moe (prep, groups, tail) / shared / setup |
|---|---|---|---|
| base = the Phase 17b path (0 / 0 / 1) | **96.1** | 10.41 | 62.8 (0.6, 37.1, 20.8) / 5.5 / 3.6 |
| **all three (1 / 1 / 3)**, two samples | **88.1, 88.3** | **11.35, 11.33** | 62.2 (0.6, 38.7, 19.4) / 0.4 / 0.1 |
| shared-early off (0 / 1 / 3) | 93.1 | 10.74 | 61.5 (0.6, 37.1, 19.8) / 5.5 / 0.1 |
| setup-fast off (1 / 0 / 3) | 91.0 | 10.99 | 62.7 (0.6, 38.8, 19.7) / 0.4 / 3.2 |
| table variant 1 (1 / 1 / 1) | 91.3 † | 10.95 | 62.2 (0.5, 38.5, 19.2) / 0.4 / 0.2 |
| table variant 2 (1 / 1 / 2) | 88.3 | 11.32 | 63.0 (0.6, 38.7, 19.8) / 0.4 / 0.1 |

† its step 1 carried a 5.7 ms host-prep outlier (engram rows); the run-to-run spread of one arm is
~±1.5 ms (base 95.7 / 96.1 / 96.2 across three builds; all-on 88.1 / 88.3 / 88.3 across two).

**Decode 96.1 → 88.1-88.3 ms/token = 11.3 tok/s (−8%)**, from two of the three terms:

1. **The shared expert under the first group's fetch: −4.9 ms** (93.1 → 88.1; the shared column 5.5 → 0.4,
   the groups column +1.5 where the fetch was not the longer leg). The first placement — enqueued on the
   owner's queue right BEFORE the tier call, as the criteria drafted it — bought only 1.3 ms: the tier's
   own prep went 0.6 → 5.1 ms/token, because its first enqueues blocked the host thread behind the
   shared kernels on the in-order queue and the first group's fetch issued that much later (exactly the
   risk docs/46 named; the tail shrank 20.6 → 17.3 as the owner finished later and waited less for the
   remote card). Staging the tier's three per-call packing copies through pinned host memory did NOT
   remove the block (measured: prep still 5.1), so the block is not the pageable source; it is kept
   anyway (an async copy is never worse). What removed it is the criteria's named fallback: a
   **pre-groups hook** in the tier (`Ds41ExpertTier::set_pre_groups`), called on the caller's thread
   right after group 0's fetch is issued — the forward hands it the shared expert's five enqueues, they
   run under the DMAs, the add and the hc mix stay after the tier (the sum's order unchanged).
2. **The step's setup: −2.9 ms** (91.0 → 88.1; the setup column 3.2-3.6 → 0.1): the decode scratch
   persistent per card (taken in the same order every step, freed with the state), the five RoPE
   round trips as one pinned copy and two launches of the same kernel with no waits, and the
   layer-14 → card 1 source-cache bounce (5.2 MB D2H + H2D per step at 2,048 positions, growing with
   the context) skipped, since layer 20 is a kv source and an index source and overwrites both before
   any consumer reads them.
3. **The FP8 table decode's micro-steps: nothing at the step level.** Variants 1 (fp16 table), 2 (fp32
   table) and 3 (fp32 table + the block's activations staged as fp32) are bit-identical to the scalar
   decode on all nine dense shapes (`gemv_fp8_test`, 46 checks PASS — the test now covers the two
   shapes the FP8 path runs that it did not, `idx_wq_b` [4096, 1280] and `engram_wkv` [25600, 6144], and
   keeps `o_a`'s [8192, 4096] for Phase 19; gate 17's read-only finding 2) and measure 91.3 / 88.3 /
   88.1-88.3 — indistinguishable within the spread. **The kernel table (after the reboot, the committed
   build): the device time did not move either** — `gemv_fp8_e4m3_f16t` 9,280 calls, 467.4 ms / 50.4 µs
   per call with variant 1 against 471.6 ms / 50.8 µs with variant 3, 14.6 ms/token both; the named
   total 45.8 ms/token in both arms (attention 174.7 µs per call, o_a 117.2 µs unchanged). The kernel
   is not bound by the per-weight conversion; docs/45's floor question needs a different instrument
   (the local-memory table loads, or the weight loads' cache behaviour at 16 columns per work-group). `IE_DS41_FP8_PACKED` now names the
   variant (0 scalar, 1-3; unset = 0 until gate 17's verdict moves the default).

**Step 0(a), the GPU clock during decode** (`freq_sampler.sh`, sysfs `act_freq` / `cur_freq` every ~14 ms
through the base and the all-on arms, ~10,600 samples each): `act_freq` reads 0 in 92-93% of the
samples (the card is power-gated at the sampling instant) and 2,735-2,752 MHz on average when not
(2,800 nominal; 87-110 of ~750-830 busy samples below 2,800); `cur_freq` sits at 2,800 on card 0 and at
2,800 for 73-85% of the run on card 1 (817-1,050 for the ~14% while card 0 loads). The instrument
cannot resolve the sub-millisecond ramp after power-gating, so the attention kernel's 175 µs per call
against its own 80-90 µs bench (deepseek4_attn.cpp, SP = 64) stays unexplained here; the A/B that
would decide it — `min_freq` 2,800 on both cards for one bench — is a root-only sysfs write and is the
founder's call (nothing in this phase changed a system setting). Step 0(b): 48 `malloc_device` per
card per step plus 10 RoPE syncs plus the 5.2 MB bounce, from the code; term 2's −2.9 ms is the sum.

## The finding that stopped the phase: the CPU miss split under EP crashes, and card 1 did not survive it

docs/42's scope 3 (the CPU miss split under EP on a core partition: `split_cores`, the code was in the
tree, unmeasured) ran as a pure env arm on this build (`IE_DS41_CPU_MISS=1` with EP): both tiers printed
their partitions (cores 8-13 / 14-19), the replayed prefill ran, and the FIRST decode step died with
`UR_RESULT_ERROR_DEVICE_LOST`. The cause, by code reading (not by a run): in export mode the remote
tier's `y` is null (the rows are the product), and the CPU leg's final add `y[i] += dp[i]` writes
through it; the guard that refused the combination (docs/39) went when the partition was added, so
the path had never been runnable. The fix in this tree: an exporting tier skips the CPU leg with a
one-time notice (the owner's tier still splits its own misses) — **UNTESTED**, because the fault took
card 1 (PCI 0000:09:00.0) into the state the 2026-09-02 note describes: `drm_sched_job_timedout` →
repeated GT resets at 01:38 (`xe_guc_submit.c:1580` WARNING, "Timedout job … in no process"), every
later load on it `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, and a 1M-element trivial kernel on it hangs
(an engine reset at 01:40, then "Fault response: Unsuccessful -EINVAL" page faults at 01:41) while the
same probe on card 0 passes. **A reboot is needed before any two-card run.** Phase 20 (the split under
EP) needs the CPU results exported as rows, not added into y — a design change, criteria first.

## After the reboot (06:40): the verification that was blocked, on the committed build (`7249d85c…`)

- **Criterion 1, bit-identity: PASS.** The full decode test with `IE_DS41_DUMP_DIR`, switches all off vs all on:
  **0 of 200 dump files differ** (`cmp`, the forced step's M / L files at 40 layers, both cards), the forced digits
  identical (card 0 worst 9.871e-3, card 1 1.417e-2, logits 6.663e-3, the first own-router flip at layer 10 at
  a 2.107e-3 near-tie), `DECODE TEST: PASS` in both arms; `gemv_fp8_test` 46 of 46.
- The kernel table (above): term 3 moves no device time; the decision on the table variants is the unit
  test's bit-identity plus 'no cost', not a win.
- **Criterion 3, prefill unchanged: PASS.** The resident test with the switches on: `RESIDENT TEST: PASS`, the
  forced digits the Phase 17 path's (9.024e-3 at layer 18, 2.755e-2 at layer 38, logits 4.195e-3), 22 / 21 static
  slots, **pp512 162.3 cold / 167.1 warm, pp2048 403.3 cold / 411.0 warm tok/s** (docs/44: 157-161 / 390-395;
  gate 15's re-check 409 / 407) — the 400+ prefill goal holds on this run.
- **Criterion 4 (so far): generate PASS** (the four greedy tokens the golden's, the continuity KL < 0.02 at
  T = 12 / 130 / 512 / 1030 with the same argmax, the exact path's first four tokens, 90 ms/token over the
  greedy run against the 210 bar).
- **Replay PASS** with the switches on (the exact path's greedy tokens 82 3104 5645 66 / 438 223 22 7640 as before).
- **Tier (+ the expert file) PASS, forward (non-resident) PASS** (" Berlin"); the replay's criteria 3-5: next token
  agrees with the exact path's at 22 of 24 prefixes, mean KL 0.0033 (max 0.0168), as docs/41.

**Criteria 1-4 all hold on the committed build; defaults flipped ON** (`IE_DS41_SHARED_EARLY=0` / `IE_DS41_SETUP_FAST=0`
are the kill switches; `IE_DS41_FP8_PACKED` stays 0 until gate 17's verdict). The remaining untested line is the
export-mode guard on the CPU miss split (Phase 20 runs it first, with a criteria doc).

## Not done at the time of the card fault (now run above where marked), in the order the gate needs them

The kernel table for term 3 (`IE_QUEUE_PROFILING=1`, both arms); criterion 1's dumps A/B (`IE_DS41_DUMP_DIR`,
all switches off vs on, `cmp` at 0.0 on the M / L files) and the full decode test's digits — the bit
identity of terms 1 and 2 rests today on construction (the same kernels, the same order of the final
add; the same RoPE kernel on the same inputs; scratch that is fully written before it is read) and on
every arm's routing, bytes and greedy sequence being identical, NOT on the dumps; criterion 3's
resident test (prefill unchanged: terms 1 and 2 are decode-only by construction); criterion 4's test
set (generate, replay, tier, forward) with the switches on. `p18_correct.sh` runs all of it in three
calls after the reboot. Defaults: the two wins are ON since the verification above (`=0` the kill switches); the measured binary
(`2baa98bb…`) differs from the committed one (`ffdd9fc2…` at c771120; gate 18 finding 1) only in those defaults and the export-mode guard.

## Gate 17 + 18 (a fresh evaluator, 07:03-07:36, one build `ffdd9fc2…`, logs `~/ds41_work/gate18/`): PASS / PASS

Re-run on one build: fp16 dense 107.3 → FP8 scalar 99.1 → table 95.3 → shared-early alone 93.1 / setup-fast alone 91.3 →
the defaults 88.6 / 88.6 ms/token; the dumps 0 of 200 differ in every pair (switches off vs the bare defaults vs +PACKED=3,
EP on, EP off, the control arm); resident FP8 pp2048 407.5 / 413.8 against fp16 410.5 / 414.3 (the pre-reboot cold −8% was
a one-run outlier); replay 22 of 24 / KL 0.0033; generate, tier, forward PASS. Findings, taken here: (1) the md5 line above;
(2) docs/45 criterion 2(a) is NOT met in magnitude — 14.7 ms/token = 374 GB/s of the true 5.49 GB against the ≤ ~12 ms bar;
gate 17 passes on 2(b), 2(c), 1 and 3, and "at the bandwidth floor" is not claimed anywhere (docs/45 amended); (3) an empty
tier returned before the pre-groups hook (unreachable today; guarded); (4) the NC = 4 GEMV branch had no test shape (added);
(5) `h_w_pk2_` serves two copies in one call (benign on the in-order queue; docs/49's list). Defaults after the gate:
`IE_DS41_DENSE_FP8` ON (`=0` the fp16 dense), `IE_DS41_FP8_PACKED` 1 (`=0` the scalar decode).

## The founder-facing statement

| | before Phase 13 | Phase 17b (docs/45) | **now (all switches, one build)** | the goal |
|---|---|---|---|---|
| decode at 2,048 ctx | 200 ms (5.0 tok/s) | 95.9 (10.4) | **88.1-88.3 ms (11.3 tok/s)** | 50 ms (20) |
| prefill pp2048 | 280 tok/s | 390-409 | unchanged by construction (not re-measured: card 1) | 400+ |

What remains on decode, by size: the slower card's link time (E[max]: 55 of the 70 misses per token
land on one link; the CPU split under EP once its export path exists, Phase 20), `o_a` in FP8 (4.7 ms
of fp16 bandwidth and 1.3 GB of VRAM, Phase 19), the T = 1 attention's clock question, the head in Q8
(a precision decision). The structural lever is DSpark: the checkpoint carries the three MTP layers
(`mtp.0-2`: attention, a 128-expert MoE, the confidence and Markov heads, `dspark_block_size` 5,
targets 37-39) — a multi-phase build that amortises the dense body and the expert fetch over the
accepted draft tokens; the founder's decision, criteria first.
