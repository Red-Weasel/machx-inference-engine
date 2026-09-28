# P4 B3 — MiMo-V2.6 row-batched groups in the lane pipe (2026-09-26)

Design: `~/ds41_work/p60/p4_mimo_batching_design.md` (B3 section). B1: `docs/mimo26/P4_B1_LANES.md`. B2:
`docs/mimo26/P4_B2_PIPELINE.md` (its "what B3 needs" list is what was built). Branch `p4-b3-mimo-rows`, from `p4-b2-mimo-pipeline`
652844c. Owner's constraint: additive; one lane (the default) stays byte-identical to HEAD and as fast. Serving is unchanged
(`mimo26_load` still forces `parallel = 1`; B4 does the serving work).

## Answer first

Several lanes' decode rows now go through ONE forward: the norms, the FP8 GEMVs, the router, the expert tier and the LM head
run over the group's rows, attention per lane against its own caches. Every one of those shared ops is **bit-identical row by
row** at M = 1..8 -- including oneDNN's o-proj and lm_head, the two the design flagged -- so grouped lanes are byte-identical to
their solo runs with the CPU leg off, and there is no separate "invariant mode" to pay for (`IE_MIMO26_ROWS_INVARIANT=1` exists
as a fixture and was not needed).

| Gate | Result |
|---|---|
| Byte-identical greedy tokens vs solo, CPU leg off | **PASS** -- 4 lanes as 2-row groups (6 prompts x 64, 2 mid-run re-admissions; and 4 x 128 A-B-A), 4 lanes as 3+1-row groups (4 x 64), 2 lanes with the drafter (4 x 64; no group steps formed there -- 2 lanes run B2's shape -- so the grouped-drafter path was first covered by the gate's G04, PASS). Logits after every step, tokens, drafts + probabilities and drafter bookkeeping all equal solo. Per-op: 45 checks (36 gated + 9 reported) bit-identical; the gate extended the test to every M from 2 to 8, all bit-identical. |
| Beat 28.2 tokens/s aggregate at 2 lanes | **NOT MET at 2 lanes**: 28.07 / 28.07 (two runs), equal to B2's 28.17 within noise. With 2 lanes on 2 cards the AUTO policy is B2's shape (one lane per card), because a 2-row group would leave a card idle. The aggregate does pass 28.2 at 4 lanes: **28.85 tokens/s** in R17, the only builder run of the final binary at its defaults, and **29.98** in the gate's G05 on the same binary; R8 (30.06: a pre-knob binary with `IE_DS41_QSTAR_MULTI=0.35`, equivalent for plain decoding) and R12 (30.08: 16 stream slots) ran the same share in other configurations. B2's shape at 4 lanes gave 25.76. |
| >= 1.6x solo at 4 lanes | **PASS** -- served config A-B-A: x1.71 at the tier's q*_multi 0.20; **x1.85** with the group share 0.35 (the new default: R17 x1.851 on the final binary, the gate's G05 x1.898; R8 / R12 x1.851 / x1.849 in the configurations noted above); CPU leg off x1.84. |
| 1 lane == HEAD | **PASS** -- `ie-mimo26-run` on the 8 held-out prompts, CPU leg off, drafter on: the `generated:` lines have the same md5 as the B2 binary's (65501302fb19267820d991815210fb2d), VRAM and static/pinned identical, decode ms/token per prompt within 0.3 %. |
| E2 `ie-mimo26-cache-test --only87` | **PASS**, 15 ok (served 8192 / 8188 / 8191 from the prompt end, snapshot 68 MiB in 36-38 ms, restored in 10 ms). |

Where the B3 gain comes from, and why it is modest: two lanes' rows share almost no experts. At 2 rows per group the tier
touches 189 static + 49 pinned experts per layer-step against 100 + 37 for one row (served config), and the CPU leg gets 120
experts instead of 47. So a cross-sequence row costs ~80 % of a whole step (the expert traffic), and grouping saves only the
dense ~20 % (the GEMVs, the norms, the router, the head, the host round trips). The design's ~22-27 ms per extra row was
measured on consecutive tokens of ONE sequence, which share far more experts.

## What was built

| Piece | Where | What |
|---|---|---|
| Group calls | `Mimo26Forward::Call` = `{T, x, pos, segs}`; `Seg` = `{lane, T, pos0, r0, all_rows, feat, logits}`; `run_card` | One call is a group of segments, each one lane's rows laid one after another in the call's host buffers. The norms, the FP8 GEMVs (M = T <= 8), the router (+ the host top-k), the tier's `moe(T)`, the residual adds and the final norm run over the T rows. **Attention runs per segment**: each segment's Q/K/V/attn rows at `r0`, that lane's `kv[card][layer]`, capacity and `pos0`, through the same kernels (the split-K decode attention for a segment of <= 8 rows, the XMX prefill kernels above). The features are copied per segment into its lane's buffer. The LM head: one segment = the pre-group launches (its last row, or all rows); a group = ONE GEMM over the span from the first wanted row to the last (<= 8 rows, the 1.25 GB head weight read once), each lane's rows copied out. One segment at r0 = 0 issues exactly the launches the B2 forward issued on the same pointers: `forward()` is a one-segment call. |
| Grouping in the pipe | `pipe_start(done, group_lanes = 0)`, `stage_loop`, `LaneState`, `Group` | `pipe_submit` puts the lane in `lq_`, the queue stage 0 groups from: the waiting lanes in submit order, at most `cap` lanes and `kDecodeRows` (8) rows; a step of more rows (a prefill chunk) runs alone. **AUTO** (`group_lanes` 0, the default): cap = ceil(lanes in flight / cards) -- 1 or 2 lanes on 2 cards give one lane per group (B2's shape, since a card would otherwise idle), 4 lanes give two 2-row groups pipelined across the cards. `group_lanes` 1 = B2, N = up to N lanes. A one-lane group reads the lane's own buffers; a multi-lane group gathers the rows into its own (8 x dim floats). The group travels through the stages as a unit; the last stage commits every lane and runs their callbacks one after another **with stage 0 held back (`pgate_`)**, so the lanes a finishing group resubmits meet in one group again (without this the first callback's resubmit would run alone). |
| Group q* | `Ds41ExpertTier::qstar_multi / set_qstar_multi`, `IE_MIMO26_QSTAR_GROUP` (default **0.35**) | A multi-lane step splits its pinned misses at the group share; a one-lane step keeps the tier's own (`qstar_multi` 0.20, tuned on one sequence's verify rows, so the drafter's verifies and V4.1 are untouched). Swept below. |
| Counters | `LaneTierStats::rows`, `group_tier_stats(card)` | A step of several lanes' rows is counted per card in the group counters (the tier's counters are per call, not per row); one-lane steps stay per lane. |
| Gate notes from the B2 gate (6-9) | `free_all`, `pipe_submit`, `stage_loop`, header | (6) teardown waits on EVERY card's queue before freeing anything. (7) a lane's in-flight claim is check-and-set under the lock before the embedding gather (a refused step gives it back), and it is held THROUGH the done callback: only the callback's own thread may submit (once) or reset that lane meanwhile, so no other thread's step rewrites the buffers the callback reads. (8) after a stage error every `pipe_submit` is refused until `pipe_stop` (documented, B4 decides recovery). (9) documented: a callback that keeps resubmitting blocks `pipe_stop`; `pipe_stop` from inside a callback deadlocks. |
| Per-op test | `tests/unit/mimo26_rows_test.cpp` (ctest `mimo26_rows_test`, SKIP 77 without an Arc) | M = 2, 3, 5, 8 rows vs the 1-row call on each row, bytes, at MiMo's shapes. |
| Tool | `ie-mimo26-lanes-test --group-lanes G --stream-slots S --invariant` | The B2 gate tool with the group cap, the tier's stream slots and the invariant fixture; a `GROUPS` tier line per card. |

## Per-op batch invariance (`results/mimo26/p4/b3/rows_test.log`, 45 checks: 36 gated + 9 reported, PASS)

| Op (MiMo-V2.6-Flash shape) | M = 2, 3, 5, 8 vs 1-row calls, row by row |
|---|---|
| `mimo26_rms_norm` (fp16 and fp32 out, H 4096) | bit-identical |
| `mimo26_gemv_fp8`: qkv N 13568 K 4096 (16 cols/wg), mlp gate N 16384, mlp down N 4096 K 16384, N 200 (4 cols/wg), N 300 (8 cols/wg) | bit-identical, all five shapes |
| `mimo26_router_logits` (E 256) | bit-identical |
| The tier's 2..8-row expert path: gather + cast, Q8_1 of the packed rows as one stream, per-row M = 1 gate/up jobs in ONE grouped launch (fp16), SwiGLU, requant, per-row down jobs (fp32 rows, `fp32_out`) | bit-identical at every stage (gate fp16 / SwiGLU fp16 / down fp32) |
| (info) one M = 8 job through the M-tiled kernel (unused by a decode step) | bit-identical |
| `ds4_expert_scatter_accum_f32`: 8 tokens x 8 experts vs 1-token scatters | bit-identical |
| oneDNN `gemm_nt_f16_onednn` o-proj N 4096 K 8192 (reported, not gated) | **bit-identical** at M = 2, 3, 5, 8 |
| oneDNN `gemm_nt_f16_onednn` lm_head N 152576 K 4096 (reported, not gated) | **bit-identical** at M = 2, 3, 5, 8 |

So the "invariant mode" question is answered: with the CPU leg off, a grouped step IS the serial arithmetic. The oneDNN result
is an observation on this build (oneDNN 3.x on the B70 with `IE_ONEDNN_DETERMINISTIC` on) -- unverified: another oneDNN
version could pick an M-dependent kernel, which is why the test stays and why `IE_MIMO26_ROWS_INVARIANT=1` exists.

## Gates (all observed; logs in `results/mimo26/p4/b3/`, untracked, mirrored in the main checkout)

**Run discipline.** Every run went through the main checkout's `scripts/ie-run-guarded --mem 220G --timeout 540`, foreground,
one at a time. Before each run: no Dream, no engine process, cards at 26/34 MiB, the last 10 minutes of `journalctl -k` free of
CAT errors / GT resets / timeouts, the lock free (`p4b3_check.sh`); after each run: 30 s, then the same check. **Every check was
clean; no device error in 18 GPU runs.** `$M = ~/models/MiMo-V2.6-Flash-RL`; prompts `results/mimo26/p4/prof/heldout_prompts/NN.txt`
(03-08, 8 distinct); `--ctx 16384 --lane-ctx 12288`; the tool's auto static tier and the chat ranking. Binaries: lanes-test
05eda7bf… (R1-R10, per the run logs; this doc first said 2e82fa59 for R1-R3), 9b7fd04d… (R11-R13, R17, the group q* knob); `ie-mimo26-run` c2539f6e…;
`ie-mimo26-cache-test` 89bc9f05…; HEAD = the B2 build's `ie-mimo26-run` ca82958e… (the B2 doc's branch binary).

### Identity (CPU leg off; `[ ok ]` = logits after every step, tokens, drafter drafts + probabilities + bookkeeping equal solo)

| Run | Arm | Result |
|---|---|---|
| R1 `r1_ident4_auto` | 4 lanes, AUTO groups, plain, prompts 03-08 (6), n 64 | **PASS**, all 6 sequences. 396 forwards, 2 mid-run re-admissions (a finished lane took the next prompt from inside the callback), 66.7 s. Card 0: 124 group steps of 2.00 rows; card 1: 124. The second wave (2 lanes) ran as one-lane groups (63 steps per lane). |
| R2 `r2_ident2_dflash` | 2 lanes, drafter ON, prompts 03-06, n 64 (B2's gate i on this binary) | **PASS**, all 4 sequences incl. drafts and bookkeeping; 264 forwards, 2 re-admissions. No group steps formed (2 lanes = B2's shape), so this is B2's gate (i) on the B3 binary, not a grouped pass; the gate's G04 covers the grouped drafter path (PASS). |
| R3 `r3_ident4_g4` | 4 lanes, `--group-lanes 4`, plain, prompts 03-06, n 64 | **PASS**, all 4. The groups settled at **3 + 1 rows** (63 steps of 3.00 rows per card, lane 3 alone for its 63 steps): the fourth lane's prefill ended after the first three had started decoding, and a group forms from the lanes WAITING at that moment -- a phase offset that never closes (see risks). |
| R13 `r13_aba4_cpuoff` | 4 lanes, AUTO, plain, prompts 03-06, n 128, A-B-A | **PASS**, all 4 sequences (128 rows) and A2 == A1. |
| R14 / R15 | `ie-mimo26-run` 8 prompts x 64, `--dflash 7 --dflash-minp 0.7`, HEAD (B2) vs B3 | **PASS**: same `generated:` md5 65501302fb19267820d991815210fb2d; VRAM 31.21 / 28.05 GB and static/pinned 78/178, 62/194 identical; decode ms/token HEAD 71.8 84.3 70.1 77.4 74.1 88.3 86.9 85.6, B3 71.6 84.3 70.2 77.3 74.1 88.3 87.0 85.7. |
| R16 `r16_e2_only87` | `ie-mimo26-cache-test $M <cat src/model/*.cpp> --only87` | **PASS**, 15 ok. |

### Throughput (A-B-A: solo A1, pipelined, solo A2 in one process; plain decoding, n 128, 4 prompts 03-06 unless noted)

The steady window starts when every first-wave lane has done 4 decode steps and ends when the first of them finishes.

| Run | Arm | Solo A1 | Pipelined aggregate (ms per lane step) | Solo A2 | x vs mean(A1, A2) |
|---|---|---|---|---|---|
| R5 `r5_aba2_served` | **2 lanes**, AUTO (= B2's shape), served (CPU leg on), prompts 03-04 | 16.85 | **28.07** (71.1) | 16.99 | x1.659 |
| R6 `r6_aba2_group2_served` | 2 lanes, `--group-lanes 2`, served | 16.94 | 28.07 (71.1) | 15.70 | x1.720 -- **no 2-row group formed** (0 group steps: the phase offset, as R3), so this repeats R5 |
| R7 `r7_aba4_group1_served` | **4 lanes** as one-lane groups (B2 with 4 lanes), served | 16.42 | 25.76 (155.0) | 16.02 | x1.588 |
| R4 `r4_aba4_served` | 4 lanes, AUTO 2-row groups, served, tier q*_multi 0.20 | 16.43 | 27.97 (142.8) | 16.27 | x1.711 |
| R10 `r10_aba4_qm030` | same, `IE_DS41_QSTAR_MULTI=0.30` | 16.02 | 28.45 (140.2) | 16.12 | x1.770 |
| R8 `r8_aba4_qm035` | same, `IE_DS41_QSTAR_MULTI=0.35` | 16.46 | **30.06** (132.7) | 16.02 | **x1.851** |
| R11 `r11_aba4_qg040` | same, `IE_MIMO26_QSTAR_GROUP=0.40` (the knob binary) | 16.30 | 29.48 (135.3) | 15.59 | x1.849 |
| R9 `r9_aba4_qm050` | same, `IE_DS41_QSTAR_MULTI=0.50` | 16.04 | 26.41 (151.0) | 16.10 | x1.644 |
| R12 `r12_aba4_slots16` | 4 lanes AUTO, group share 0.35, **16 stream slots** | 16.25 | 30.08 (132.7) | 16.29 | x1.849 |
| R17 `r17_aba4_final` | 4 lanes AUTO, **the final binary at its defaults** (group share 0.35, 8 slots) | 16.10 | **28.85** (138.3) | 15.08 | **x1.851** (x1.792 vs A1) |
| R13 `r13_aba4_cpuoff` | 4 lanes AUTO, **CPU leg off** | 10.37 | 19.09 (209.0) | 10.36 | x1.841 |
| R1 (n 64, 6 prompts, CPU leg off, no A-B-A) | 4 lanes AUTO | 10.66 | 20.07 (171.3) | -- | x1.883 vs A1 |
| R3 (n 64, CPU leg off, no A-B-A) | 4 lanes `--group-lanes 4` (ran as 3 + 1) | 10.91 | 15.59 (254.1) | -- | x1.429 vs A1 |

Reading:
- **4 lanes as two 2-row groups beat 4 lanes as four one-lane groups** (B2's shape): 27.97 vs 25.76 at the same q*, +8.6 %;
  with the group share 0.35, 30.06 vs 25.76, **+17 %**. That is the B3 gain over B2 at 4 lanes.
- **At 2 lanes there is nothing to batch without idling a card**; AUTO keeps B2's shape and matches B2's number (28.07 vs 28.17).
- **Per-lane latency** at 4 lanes: 133-143 ms per step in 2-row groups vs 155 ms as one-lane groups, vs 61 ms solo.
- 3 + 1 rows (R3) is the worst shape: the 3-row group's step is ~2.6x a one-row step while the lone lane's card idles.
- Bigger groups cost more per row than they save (R3 vs R1 at the same CPU-leg-off config: x1.43 vs x1.88), because the
  expert union grows almost linearly with cross-sequence rows.

### q* for groups (the sweep above, served config)

| Group share | 0.20 (tier default) | 0.30 | **0.35** | 0.40 | 0.50 |
|---|---|---|---|---|---|
| Aggregate tokens/s | 27.97 | 28.45 | **30.06** | 29.48 | 26.41 |
| CPU-leg experts per 2-row group step, card 0 / 1 | 120.5 / 98.6 | 107.6 / 89.1 | 100.4 / 81.6 | 93.4 / 76.7 | 72.4 / 58.7 |
| Stream-slot hit rate, card 0 / 1 | 36.0 / 46.4 % | 23.6 / 32.7 % | 23.7 / 33.9 % | 18.1 / 27.1 % | 11.1 / 18.5 % |
| moe ms per group step, card 0 / 1 | 58.0 / 51.2 | 55.8 / 50.2 | 53.2 / 47.6 | 53.9 / 47.7 | 61.5 / 54.9 |

**Default: 0.35** (`IE_MIMO26_QSTAR_GROUP`), applied only to steps of several lanes' rows. The tier's `qstar_multi` 0.20 was tuned
on one sequence's speculative verify rows, where an expert carries several rows and a transfer pays for all of them; two lanes'
rows share almost no experts (an expert carries ~1 row), so the split sits near the one-row q* (0.30). The 0.35 / 0.40 difference
(30.06 vs 29.48) is inside the run-to-run noise (unverified: one run per point, ±3 %); 0.50 is clearly worse.

### Stream slots (R12 vs R8 / R17)

16 slots per layer instead of 8: the solo hit rate rises from 36 / 50 % to 50 / 63 % (card 0 / 1) and the groups' from 24 / 34 %
to 41 / 51 %, but the aggregate is the same (30.08 vs 30.06) and the static tier loses 8 experts per layer (74 -> 66, 68 -> 60).
**Kept at 8.** The hits saved are transfers of experts the CPU leg or a static slot would otherwise have served; at this
q* the step is not bound by the pinned transfers.

### Stream-slot hits per group (the B2 hypothesis, continued)

| Step kind (served config) | Static + pinned experts per layer-step, card 0 / 1 | Hit rate, card 0 / 1 | CPU-leg experts |
|---|---|---|---|
| Solo, one row | 99.6 + 37.2 / 118.6 + 36.7 | 36.4 / 49.9 % | 47.1 / 36.7 |
| One-lane group, 4 lanes interleaving (R7, lane 0; the 4 lanes span 89-104 + 31-38 / 95-131 + 27-40) | 100.3 + 33.7 / 117.2 + 31.8 | 24.1 / 30.8 % (lanes 23.5-25.3 / 29.0-34.5 %) | 50.0 / 43.0 |
| 2-row group, q* 0.20 (R4) | 188.9 + 49.0 / 223.8 + 48.3 | 36.0 / 46.4 % | 120.5 / 98.6 |
| 2-row group, group share 0.35 (R8) | 189.0 + 68.9 / 224.0 + 65.2 | 23.7 / 33.9 % | 100.4 / 81.6 |
| 2-row group, CPU leg off (R13) | 186.1 + 170.8 / 221.5 + 149.1 | 3.3 / 6.0 % | 0 |
| 3-row group, CPU leg off (R3) | 275.6 + 241.0 / 317.2 + 204.1 | 2.0 / 5.1 % | 0 |

A 2-row group's expert union is 1.75-1.9x a row's: the rows come from different sequences. With the CPU leg off, the union of
pinned experts (171 / 149 per layer-step) swamps the 8 stream slots (3-6 % hits) -- there the pinned transfers ARE the step.

### fdinfo busy (per-client engine cycles, the B2 premise)

| Window | Card 0000:04 | Card 0000:09 |
|---|---|---|
| Solo decode, served | bcs 60.6 %, ccs 62.7 % | bcs 56.6 %, ccs 57.4 % |
| 4 lanes as 2-row groups, served (R4) | bcs 98.3 %, ccs 98.5 % | bcs 95.5 %, ccs 95.6 % |
| 4 lanes as 2-row groups, CPU leg off (R13) | bcs 97.6 %, ccs 97.8 % | bcs 87.8 %, ccs 87.9 % |

As in B2: both cards are busy through the step; bcs and ccs read the same, so these counters still do not separate copy from
compute (unverified, unchanged).

**Timing caveat.** Other agents were building and running on the box (1-minute load 1.2-5.5 before the runs). All
ratios are same-process A-B-A; the two solo arms agree within 1-3 % except R6 / R11 / R17, where A2 fell 6-7 % below A1 (the x
vs A1 alone: R6 x1.657, R11 x1.809, R17 x1.792). The 0.35-share runs spread 28.85-30.08 (R8, R12, R17; only R17 is the final binary at its defaults, and the gate's G05 on it gave 29.98); one
run per sweep point, so the 0.30 / 0.35 / 0.40 ordering is inside that noise and only 0.20 and 0.50 are clearly apart.

## Criterion (b): near-ties with the CPU leg on (run by the gate, not by the builder)

The plan's (b) asks that the default mode diverge from solo only at near-ties. With the CPU leg off the batched path IS the
serial arithmetic (above), so (b) holds trivially there. In the served config (CPU leg on) the gate ran a teacher-forced scan:
4 lanes AUTO, prompts 03-06, 97 rows each (388 rows); near-tie = the other pick is in A1's top 3 and the gap is <= 0.75
(MiMo's lookup-gate rule).

| Comparison (vs solo A1) | Argmax flips | Near-ties | Worst non-near-tie gaps | Mean abs delta-logit |
|---|---|---|---|---|
| B3 groups (G20) | 11 | 9 | 3.04 and 10.07 | 0.46 |
| B2 shape, `--group-lanes 1` (G21) | 9 | 8 | 3.03 | 0.458 |
| Solo A2 vs A1 (G20 / G21) | 6 / 5 | 3 / 2 | up to 11.7 | 0.24 / 0.23 |

Every non-near-tie flip sits at a special-token position (151667 / 151658). So the literal (b) is not reachable with the CPU leg
on even solo against solo; row batching shows no fidelity loss beyond B2's accepted pipelining. Unverified: one run per arm and
388 rows, so "same as B2" is order-of-magnitude, not statistical. The plan's (b) is restated here as: with the CPU leg on,
grouped lanes diverge from solo no more than B2's pipelined lanes and solo-vs-solo do.

The gate also ran multi-row segments inside a group end to end for the first time (G19: 47 group steps per card at 4.15 rows
each, all 6 sequences byte-identical, CPU leg off) -- the path B5's verify steps will use.

## Defaults picked

- **Grouping AUTO** (`pipe_start(done, 0)`): cap = ceil(lanes in flight / cards). 1-2 lanes on 2 cards = B2 exactly; 4 lanes =
  two 2-row groups. Rows per group <= `kDecodeRows` (8); a step of more rows runs alone.
- **`IE_MIMO26_QSTAR_GROUP` 0.35** for steps of several lanes' rows; the tier's own `qstar_multi` (0.20) for one-lane steps.
- **Stream slots stay 8**; `IE_MIMO26_ROWS_INVARIANT` off (the batched path is bit-identical anyway).
- The o-proj and the head run batched over the group's rows; the head over the wanted-row span.

## Risks and limits

- **Grouping is opportunistic.** A group forms from the lanes waiting at stage 0 at that moment; lanes whose steps are out of
  phase (a prefill that ends later, a lane alone) never merge -- R3's 3 + 1 and R6's "no 2-row group at `--group-lanes 2`". AUTO
  avoids the worst shape at 4 lanes in plain decoding because 2-row groups re-form every step (the gate's G16: 186 group steps
  per card with staggered entry, x1.80), but it does NOT always engage: with the drafter on and a 599-row chunk the gate saw only
  2 group steps in a whole run (G14, B2's shape) -- cause unverified, likely the heavier last-stage callback. At 3 lanes it alternates 2 + 1, and the tool's
  shapes above are what the design's scheduler (B4) inherits. A "wait for the cap" policy was not built (it would idle a card).
- **The drafter's steps do not group**: a lane's verify step is 1 + 7 = 8 rows, so two lanes' drafter steps exceed 8 rows and
  run one per group (B2's shape, R2). The design's row budget (B5) decides how to share the 8 rows between drafts and lanes.
- **Identity scope**: greedy (temperature 0), <= 4 lanes, <= 128 tokens, contexts far from capacity, 2- and 3-row groups. Not
  covered: sampling, 5-8-row groups (only the per-op test covers those M), lanes near capacity, a stage error mid-flight (the error
  path is coded, not exercised).
- **CPU leg on**: not bit-reproducible (as B2 and as today's serving history); A2 differed from A1 at step 0 on every prompt.
- **The group q* was tuned on one shape**: 4 prompts, plain decoding, 2-row groups, 4 lanes, served config. Other shapes
  (3-row groups, groups mixed with a drafter row budget, other prompts) are unverified. The tier's share is set per `moe`
  call from `run_card` (`set_qstar_multi`); the V4.1 forward never calls it.
- **Error semantics (B2 gate note 8)**: after any stage error, `perr_` refuses every `pipe_submit` until `pipe_stop`; B4 decides
  per-lane recovery.
- **Hazards (note 9)**: a done callback that keeps resubmitting blocks `pipe_stop` forever; `pipe_stop` from inside a callback
  deadlocks (it joins the callback's own stage thread).
- **`pgate_` holds stage 0 back during every callback, one-lane groups included** (B2 did not). The cost is nil with this tool's
  light callbacks; B4's heavier callbacks (sampling, streaming, the drafter) pay it -- unmeasured.
- **Per-lane attribution** of tier counters is lost inside a group (the tier counts per call): `group_tier_stats` is per card.
- **Diagnostics label rows by the first segment's `pos0`** (`IE_MIMO26_CHECK` prints); the diagnostics are refused in the pipe
  anyway (serial forward = one segment).
- **Prefill vs decode**: a 2,048-row chunk still occupies a card for ~4.5 s and stalls the decoding lanes behind it (the
  design's mixed-prefill concern, unmeasured here: the steady window starts after the prefills).
- **VRAM**: 3 extra lanes at ctx 12,288 cost 1.2 / 1.1 GB per card out of the static tier (74/182 and 68/188 static/pinned
  per layer, against 78/178 and 62/194 with one lane at ctx 16,384). The per-token cost of that is inside the solo arms above
  (16.0-16.9 tokens/s vs B2's 16.5-16.8 with 2 lanes).

## What B4 (serving) needs from this

1. A MiMo stepper that calls `pipe_submit` per active lane and lets AUTO form the groups; `group_lanes` stays a knob.
2. The row budget between lanes and drafts (B5), since drafter steps fill a group alone.
3. Prefill chunks capped while lanes decode (the design's 512-row proposal), measured.
4. Per-lane error recovery instead of `perr_` failing the pipe.
