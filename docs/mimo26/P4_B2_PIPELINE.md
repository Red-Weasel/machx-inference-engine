# P4 B2 — MiMo-V2.6 card-pipelined lanes (2026-09-26)

Design: `~/ds41_work/p60/p4_mimo_batching_design.md` (B2 section). B1: `docs/mimo26/P4_B1_LANES.md`. Plan:
`~/ds41_work/p60/multimodel_parallel_plan.md`. Branch `p4-b2-mimo-pipeline`, from `p4-b1-mimo-lanes` f4ac925.

Owner's constraint: additive. One lane (the default) must stay byte-identical to today and as fast. Serving is unchanged:
`mimo26_load` still forces `parallel = 1`, and B4 does the serving work.

## Answer first

With 2 lanes, card 0 runs one lane's step while card 1 runs the other's.

| Arm | 2-lane aggregate | Solo, A-B-A | Ratio |
|---|---|---|---|
| Served config: CPU expert leg on, plain decoding (no drafter) | **28.17 tok/s** | 16.79 / 16.49 tok/s | **x1.69** |
| CPU leg off | 18.58 tok/s | 11.05 / 11.05 tok/s | x1.68 |

- The gate asks for x1.3; x1.69 passes it.
- With the CPU leg off, each lane's output is byte-identical to its solo run, including the drafter.
- One lane is unchanged versus HEAD.
- The owner's "60 %" is the per-client busy that Dream reports. During solo decode it is 51-64 % per card. With 2
  pipelined lanes it becomes 87-99 %.

## What was built

| Piece | Where | What |
|---|---|---|
| Pipe | `Mimo26Forward::pipe_start / pipe_submit / pipe_reset_lane / pipe_stop`, `stage_loop` | One host thread per card ("stage"), and one mutex for the stage queues. Each lane has at most one step in flight. `pipe_submit` embeds the lane's rows into that lane's own host buffer and queues the lane on stage 0. Stage s runs `run_card(card s, lane)` and hands the lane to stage s+1. The last stage commits the lane's `n_pos` / `hi_end` and then calls `done(lane, logits, features, rows)` **on the last card's stage thread**. The drafter shares that card's queue, so drafter work belongs there. The callback may submit the lane's next step. `pipe_stop` waits until no step is in flight, joins the threads and returns the first stage error. During the pipe, `forward()` and `select_lane()` refuse. The diagnostics (probe, routing profile, `IE_MIMO26_DUMP`) are refused at `pipe_start`. Image positions are refused by `pipe_submit` (serial `forward()` only). |
| Per-call lane state | `run_card(Card&, ci, const Call&)` | The call carries its lane (the caches `lanes_[l].kv[card]` and the capacity `lanes_[l].cap`), T, pos0, and host pointers (residual, positions, features, logits). The B1 pointer swap into `Dense::k/v` is gone. `select_lane` only swaps positions, and `state_spans` reads the active lane's caches. At one lane, `lanes_[0].kv` holds the pointers `upload_card` allocated, so the launches and pointers are the same as before. |
| Per-stage / per-lane host buffers | `Card::h_rlogits / h_ridx / h_rw` (per card = per stage); `LaneState::h_x / h_pos / h_feat / logits` (per lane, allocated at the first `pipe_start`) | The serial `forward()` keeps its member `h_x_`, `h_pos_` and `h_feat_`. The `static` locals left in `run_card` are all `static const` env reads (thread-safe init). The layer-indexed `stats_` never overlap between cards. |
| CPU leg split (option) | `Mimo26Options::split_cpu_cores` | Card 0's tier gets E-cores 8-13 and card 1's gets 14-19, with the stage threads kept off them. Default **false**, because it was measured slower (below). |
| Per-lane tier counters | `lane_tier_stats(lane, card)` | Per decode step (T ≤ 8): static and pinned experts, stream-slot hits, CPU experts, and moe ms. Filled by both the serial and the pipelined path. |
| Vision provider per lane | `LaneState::vis` | `set_vision_provider` / `clear_vision` act on the active lane (coordinator note 1 on B1), so each lane's request brings its own images. |
| Lanes refusal fix | `upload_card` | With an explicit `n_static`, the refusal now counts `n_static + stream_slots` slots per layer. It used to count `n_static` only (coordinator note 5). |
| Gate tool | `ie-mimo26-lanes-test --pipeline [--aba] [--split]` | Runs pipelined lanes against solo with the per-sequence identity from B1 (logits FNV per step, tokens, drafts + probabilities, drafter bookkeeping), plus mid-run re-admission from the callback. It also reports the steady-window aggregate tokens/s (details below), A-B-A solo arms after a warm-up, per-lane stream-slot hits, and per-card engine-class busy from `/proc/self/fdinfo`. With `--cpu-miss` the identity lines are `[info]`, not gated. |

**Steady window:** it starts when every first-wave lane has done 4 decode steps and ends when the first of them finishes.
It counts every decode completion inside it.

## Gates (all observed)

**Run discipline.** Every run went through the main checkout's `scripts/ie-run-guarded --mem 220G --timeout 540`, in the
foreground. Before each run: no Dream, cards at 26/34 MiB, and `journalctl -k --since "-10 min"` clean. After each run:
30 s, then the same check. Every check was clean. The scripts and all 15 logs are in `results/mimo26/p4/b2/`, which is
untracked.

**Common arguments.** `$M = ~/models/MiMo-V2.6-Flash-RL`. Prompts are `results/mimo26/p4/prof/heldout_prompts/NN.txt`.
Files 00, 01 and 02 are byte-identical (md5 e0034d67…), so only one of them was used. `--ctx 16384 --lane-ctx 12288`.

**HEAD** is the B1 tip f4ac925, built separately. Its `ie-mimo26-run` md5 is d79af602…, the same binary as B1's gate
(a). The branch `ie-mimo26-run` md5 is ca82958e….

| Gate | Command | Result |
|---|---|---|
| (i) pipelined == solo, CPU leg off, drafter ON | `ie-mimo26-lanes-test $M --prompt-file 03 --prompt-file 04 --prompt-file 05 --prompt-file 06 --lanes 2 --n 64 --pipeline` | **PASS.** All 4 sequences are identical to solo: 64 logits rows, 64 tokens, drafts + probabilities, and drafter bookkeeping. 264 forwards and 2 mid-run re-admissions (a finished lane took the next prompt from inside the callback). |
| (i') B1 serial path unchanged | `... --prompt-file 07 --prompt-file 08 --prompt-file 09 --lanes 3 --n 48 --slot-arm` (serial) | **PASS.** All 3 identical to solo. Slot arm: A (5,505 positions) kept from lane 1 (579 MiB, 222 ms) and restored on lane 2 (60 ms), and the continuation is identical to the control. |
| (ii) 1 lane vs HEAD, CPU leg off | `IE_DS41_CPU_MISS=0 ie-mimo26-run $M --prompt-file {00,03,04,05,06,07,08,09} --n 64 --chunk 2048 --ctx 16384 --static 0 --ranking $M/ie_ranking_mimo26_chat.txt --dflash 7 --dflash-minp 0.7` | **PASS.** The `generated:` lines have the same md5, 65501302… in both runs. VRAM (31.21 / 28.05 GB) and static/pinned (78/178, 62/194) are identical. Decode ms/token per prompt is within 0.2 % (HEAD 71.7 84.3 70.1 77.3 74.1 88.3 86.9 85.7; branch 71.6 84.4 70.2 77.4 74.1 88.4 87.0 85.6). |
| (ii) 1 lane vs HEAD, CPU leg on (served), H-B-H-B | same command with the CPU leg on | Tokens are identical in all 4 runs (md5 fc37056c…). Decode ms/token summed over the 8 prompts: HEAD 333.4 and 329.9, branch 340.0 and 327.7. The branch mean is +0.7 %; the per-prompt run-to-run noise is about ±5 %. |
| (iii) A-B-A, plain, CPU leg on as served | `ie-mimo26-lanes-test $M --prompt-file 03 --prompt-file 04 --lanes 2 --n 128 --pipeline --aba --no-dflash --cpu-miss` | Solo A1 16.79, **pipelined 28.17 aggregate**, solo A2 16.49 tok/s: **x1.693**. Per-lane step latency is 70.8 ms, against 59.6 ms solo. |
| (iii) A-B-A, plain, CPU leg off | same, without `--cpu-miss` | Solo A1 11.05, **pipelined 18.58**, solo A2 11.05 tok/s: **x1.681**. Both lanes are identical to solo (128 rows), and A2 == A1. |
| (iv) E2 | `ie-mimo26-cache-test $M <text: cat src/model/*.cpp> --only87` | **PASS**, 15 ok. The prompt end is served (8192 / 8188 / 8191); the snapshot is 68.8 MiB and restored in 10 ms; logits, 16 tokens and drafter state are identical to the control. |

Also verified:
- A build of all targets succeeds.
- These host ctests pass (8/8): `ds41_prefix_key_test`, `idle_spin_test`, `server_admission_test`, `mimo26_host_rules_test`,
  `ie_vitals_test`, `ds41_sampler_test`, `openai_proto_test`, `mimo26_lanes_test`.

Final binaries: `ie-mimo26-lanes-test` 2f7a72c8… (the run (i') binary), `ie-mimo26-run` ca82958e…, and `ie-mimo26-cache-test`
e2456264….
- The A-B-A timing and sweep runs used lanes-test 26d62fad…. It differs from the final binary only in the core-split flag
  (`--no-split` then, `--split` now) and its default.
- Run (i) used 01676dbd…, which lacks the per-client fdinfo lines.

**Timing caveat.** Other agents were building on the box at times. The 1-minute load average before the runs
was 1.5-4.6. The ~±5 % run-to-run noise in the H-B-H-B pair is the best estimate of that effect. The x1.68-1.69 ratios are
same-process A-B-A, and the two solo arms bracket the pipelined arm within 2 %.

## CPU-leg split and q* (2 pipelined lanes, plain, CPU leg on)

| Cores | q* | Pipelined aggregate tok/s | Solo in the same process |
|---|---|---|---|
| both tiers on 8-19 (served, unsplit) | 0.20 | 26.50 | 16.55 |
| unsplit | **0.30 (default)** | **28.17** | 16.79 |
| unsplit | 0.45 | 27.85 | 16.02 |
| split 8-13 / 14-19 | 0.30 | 25.30 | 14.74 |
| split | 0.45 | 26.74 | 15.22 |

**Result:** keep the default — unsplit, q* 0.30.
- Two 12-thread OpenMP teams on the same 12 E-cores beat two private 6-core halves. Likely reason (not verified): the two
  cards' legs rarely run at the same instant, so each team gets most of the 12 cores.
- The split option stays available, default off.
- A finer sweep (0.25 / 0.35) was not run.

## B0 premise: what the owner's "60 %" is

The tool reads the ie process's own DRM clients from `/proc/self/fdinfo`. It uses xe's `drm-cycles-<class>` /
`drm-total-cycles-<class>` for one client per card and accumulates only over the decode windows.

| Window | Card 0000:04 (layers 0-23) | Card 0000:09 (layers 24-47) |
|---|---|---|
| Solo decode, CPU leg on | bcs 61.7 %, ccs 63.9 % | bcs 55.5 %, ccs 56.2 % |
| Solo decode, CPU leg off | bcs 61.8 %, ccs 63.2 % | bcs 50.1 %, ccs 50.7 % |
| 2 lanes pipelined, CPU leg on | bcs 99.6 %, ccs 99.7 % | bcs 98.4 %, ccs 98.5 % |
| 2 lanes pipelined, CPU leg off | bcs 99.2 %, ccs 99.4 % | bcs 87.0 %, ccs 87.6 % |

Reading:
- Dream's figure is, per client, the max over classes. That is the 51-64 % in the solo rows, which matches the owner's
  "60 %".
- During solo decode, each card's context is active only about half the step. The other half is the layer split, where
  the card waits for the other card.
- Pipelining fills that gap (88-99 %).
- **bcs and ccs read almost the same on every card and in every window.** So these counters do NOT separate copy-engine
  time from compute time here. The design's question "is the 60 % copy or compute" stays open: unknown.
  - Most likely, xe's per-class cycles for this client reflect time the context is resident or active, not per-engine
    work. That is a guess.
  - A kernel profile (M2, `--kprof`) or `intel_gpu_top` engine view would resolve it.
- rcs, vcs and vecs are 0.

## Stream-slot hits per lane (the B1 hypothesis)

Decode steps only, per card, mean per step. The tool computes these as `stream_hits / experts_pinned` from the tier's
own counters.

| Run | Card 0 hit rate | Card 1 hit rate | moe ms/step, card 0 / card 1 |
|---|---|---|---|
| Solo, CPU leg off | 17.9 % (14.5 of 80.9 pinned) | 29.0 % | 39.1 / 28.0 |
| 2 lanes pipelined, CPU leg off | 11.2 / 11.5 % | 21.5 / 19.3 % | 42.5, 41.2 / 32.5, 29.1 |
| Solo, CPU leg on | 35.6 % (12.6 of 35.4 GPU-pinned; 45.7 on the CPU) | 49.9 % | 20.5 / 15.9 |
| 2 lanes pipelined, CPU leg on | 31.5 / 29.5 % | 44.9 / 43.2 % | 24.0, 24.6 / 21.7, 19.5 |
| 3 lanes serial (B1 path, prompts 07-09) | 5.9-6.5 % | 6.8-11.4 % | 46-48 / 41-52 (solo arm: 18.8 % / 30.8 %, 41.4 / 36.5 ms) |

Confirmed: lanes interleaving on one card share its 8 stream slots per layer, and each lane's hit rate falls. It drops
about 6-8 points at 2 lanes and to a third at 3 lanes. That costs extra pinned transfers per step (moe +2-5 ms at 2 lanes,
+5-15 ms at 3 lanes).

The pipeline already absorbs this and still gives x1.69. B3/B4 could win it back with more stream slots per layer, or
per-lane slot partitions. Both cost VRAM, which comes out of the static tier.

The B1 doc's "+7-11 % per step interleaved" is corrected there as inconclusive and order-confounded (coordinator note 3).

## Risks

- **Identity scope.** Identity with the CPU leg off was shown on 4 distinct prompts × 64 tokens (drafter on, 2 lanes, mid-run
  re-admission) and 2 × 128 (plain). Not covered:
  - more than 2 lanes pipelined;
  - sampling at temperature > 0;
  - long contexts (lanes near capacity);
  - a stage error mid-flight (the error path is coded but not exercised).
- **CPU leg on.** Pipelined lanes are not bit-reproducible, and neither are two solo runs: A2 differed from A1 at prompt
  04's 2nd token. This matches today's serving history dependence.
- **Features lifetime.** The `features` pointer handed to `done` stays valid only until the callback submits that lane's
  next step, because card 0 then rewrites the lane's buffer. The tool feeds the drafter first.
- **Thread use.**
  - The drafter is touched only on the last card's stage thread while the pipe runs. A caller that uses it elsewhere races.
  - `lane_pos()` / `piping()` are read without the lock from the callback thread. This is safe only because
    `pipe_start` / `pipe_stop` are not concurrent with steps.
- **oneDNN.** Its primitive caches are `thread_local` per queue, so each stage thread builds its own. Identity held, but
  a first-step JIT cost lands on the first pipelined steps. The timing window skips 4 steps per lane.
- **Excluded while pipelining:**
  - vision (image positions are refused, serial only);
  - the diagnostics (refused at `pipe_start`);
  - `IE_MIMO26_CHECK` (allowed; it prints from two threads).
- **Prefill vs decode.** A 2,048-row prefill chunk occupies a card for ~4.5 s and stalls the other lane's decode behind
  it (the design's mixed-prefill concern). It is not measured here: the steady window starts after both prefills.
- **VRAM per lane** comes out of the static tier: card 1 went from 62 to 60 static experts per layer at lane-ctx 12,288
  (the tool's auto tier). The per-token cost of that is not isolated.

## What B3 (row-batched groups) needs

1. **Batch invariance first.** Add a per-op unit test at M = 1 vs M = k for `mimo26_gemv_fp8` (M ≤ 8), the router logits,
   the o-proj (`gemm_nt_f16_onednn`, where oneDNN's kernel choice varies with M), the tier's expert GEMMs, and the scatter.
   This decides whether an "invariant mode" can be byte-identical to solo.
2. **A forward over several lanes' rows.** Concatenate the rows. Norms, GEMVs, the router and `tier.moe` run over all of
   them; attention loops over lanes with per-lane `pos0`, T, `lanes_[l].kv` and `cap`. `run_card`'s `Call` becomes a list
   of (lane, row range, pos0).
   - The per-stage host buffers are already per card.
   - The per-lane h_x / h_pos / h_feat buffers are the natural slices.
   - The 8-row ceiling (`kDecodeRows`, the FP8 GEMV, the CPU leg's `kMaxRows`) caps the group.
3. **Combine with B2.** Run two groups, one per card stage, each holding up to 8 rows. The pipe's per-lane in-flight flag
   becomes per group.
4. **Stream slots.** Row batching widens each step's expert union. Measure hits per group with `lane_tier_stats`, and
   consider more slots per layer.
5. **q\*.** Use `qstar_multi` (0.20, tuned on single-sequence verifies) for T > 1. Re-sweep it for cross-sequence rows,
   whose union grows faster.
6. **Gate against B2.** The B2 pipelined aggregate (28.2 tok/s at 2 lanes) is the number to beat. The design's B3 target
   is ≥ 1.6x solo at B = 4.

## Coordinator notes on B1, handled here

- Note 1: the vision provider is now per lane.
- Note 3: the timing sentence in the B1 doc is corrected.
- Note 4: the B1 doc's prompt counts are corrected (00-02 are one prompt; 8 distinct). The B2 identity gates use distinct
  prompts: 03-06 and 07-09.
- Note 5: the explicit-`n_static` refusal count is fixed.
