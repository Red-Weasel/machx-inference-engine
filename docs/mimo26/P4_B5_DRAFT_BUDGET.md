# P4 B5 — MiMo-V2.6's drafter with parallel lanes: a draft budget (2026-09-26)

Design: `~/ds41_work/p60/p4_mimo_batching_design.md` §1.3 (the drafter) and §4 "B5". B2 pipe: `docs/mimo26/P4_B2_PIPELINE.md`;
B3 groups: `P4_B3_ROWS.md` (the gate-corrected version on `deepseek4-vision-exp`); B4 serving: `P4_B4_SERVE.md` (4d0e648); the
drafter's cost: `P4_B5V_DRAFT_VOCAB.md` (on `deepseek4-vision-exp`). Branch `p4-b5-mimo-draftbudget`, from `p4-b4-mimo-serve` 4d0e648.

Owner's constraints: additive; `--parallel 1` and one decoding lane stay byte-identical to B4; the default flips only on a
measured win.

## Answer first

B4 ran the DFlash drafter only while exactly one lane decoded, so 2-4 served lanes made about what one drafted request makes
alone. B5 lets several decoding lanes draft within a **budget of rows per group step** (`IE_MIMO26_DRAFT_BUDGET`, default now
**8**). The budget's even share gives each lane `floor(8 / lanes per group) - 1` drafts: 7 per lane at 2 lanes (one lane per group,
pipelined across the cards), 3 per lane at 3-4 lanes (two lanes per group).

| Gate | Result |
|---|---|
| (a) one lane == HEAD (4d0e648) | **PASS.** `--parallel 1` SSE bytes equal the B4 captures (sse 1 / 3 raw-identical, sse 2 identical with its three wall-clock fields masked); `serve_test.py` all ok; `ie-mimo26-run` generated md5 **65501302fb19267820d991815210fb2d** with VRAM and static / pinned identical. At `--parallel 4`, one lane decoding: the solo arms equal HEAD's 4/4 (drafter on, CPU leg off). |
| (b) 2 and 4 lanes, served A-B-A: aggregate >= the drafter-off lanes AND the drafted solo | **PASS at both.** 2 lanes: **28.38** tok/s against **23.41** with the drafter-off lanes (x1.21) and **22.76** for the same prompts drafted solo (x1.25). 4 lanes: **28.46** against **25.18** (x1.13) and **22.46** (x1.27). One run per arm on a busy box (unverified). |
| (c) CPU leg off, greedy: each lane == its drafted solo run, or divergences only at near-ties | **PASS.** Same verify shapes (drafter k = 3 everywhere): batch == solo **4/4**, cancel **3/3**. The policy's shapes (7 alone, 3 in pairs): the teacher-forced scan finds the pipe bit-identical to the same shapes run solo, and every shape-induced flip a near-tie; the served first divergences read with the top-3 gap log are all near-ties (gap 0.20-0.67). |
| (d) no memory growth, no device errors | **PASS.** RssAnon 335 → 819 (after wave 1) → 836 MiB over 20 requests (+17 MiB after wave 1; B4's own runs: +9 and +14); VRAM flat; threads 54 flat. No CAT error, GT reset or timeout in any of the 13 GPU runs. |

Default picked: **`IE_MIMO26_DRAFT_BUDGET` 8, even share, AUTO groups** (`kMimo26DraftBudgetDefault`, commit e0a2027). Weighting
stays off (`IE_MIMO26_DRAFT_WEIGHT=0`: not measured in serving, see "Not done").

## What was built

| Piece | Where | What |
|---|---|---|
| The draft budget | `Mimo26Serve::draft_budget`, `serve_prepare_decode`, `serve_draft_k` (`src/engine/mimo26_engine.cpp`) | `IE_MIMO26_DRAFT_BUDGET` = the rows of one group step (<= `kDecodeRows`, 8) that the lanes' anchors and their DFlash drafts share while two or more lanes decode. **Even share** (the default mode): every lane gets `floor(budget / g)` rows, its anchor plus `floor(budget / g) - 1` drafts (at most the drafter's 7), with `g` = the pipe's lanes per group: its AUTO `ceil(decoding / cards)`, or `IE_MIMO26_GROUP_LANES`. At 2 lanes on 2 cards each lane drafts 7; at 3-4 lanes 3; at 5-8 lanes 1; beyond that 0 (plain rows). The drafter's min-p 0.7 cut still applies, so a lane offers at most its share. **One lane decoding keeps the exact B4 path**: the same `draft(l.next, dflash_k, ...)` call. Budget 0 = the B4 rule (plain one-row steps whenever two or more lanes decode). |
| Acceptance weighting | `IE_MIMO26_DRAFT_WEIGHT=1`, `mimo26_draft_k_weighted`, `mimo26_draft_acc` (`include/ie/mimo26_engine.hpp`) | Inside a done callback: the rows of the group being answered (its lanes' even shares, at most the budget), less what the group's earlier lanes resubmitted, are split between this lane and the group's later lanes. Each lane first gets one row (its anchor). The spare rows then go one at a time to the lane whose next draft is the most likely to be kept: `acc^(k+1)` for a lane holding `k` drafts, where `acc` = the lane's per-draft acceptance (decayed kept / (kept + rejections), prior 0.8). A lane may get 0 drafts. Rows an earlier lane's min-p cut leaves go to the later lanes, and every group stays within the budget. At a turn's release (no callback group) the even share applies. Host-tested only (below). |
| The group row cap | `Mimo26Forward::pipe_start(done, group_lanes, group_rows)` | A multi-lane group holds at most `group_rows` rows (0 = `kDecodeRows`, the B3 pipe); the serving layer passes the budget. A lone lane's step of more rows (a prefill chunk, or a lone lane's 8-row verify) still runs alone. A budget below the lanes per group therefore splits groups (use 0, not 1, for plain rows). |
| The callback's group | `Mimo26Forward::pipe_cb_lanes()` | Callable from inside a done callback only: the lanes of the group whose callbacks are running (set and cleared under the pipe's lock by the last stage; other threads get an empty list). |
| The drafter's contexts | `serve_df_ctx` | With a budget set, every lane's context follows every step: `IE_MIMO26_DF_FEED=0` no longer skips it while several lanes decode, because the lanes draft then. The rings were already per lane (B1/B4). |
| /health | `Engine::serving_status_json` | `draft_budget`; `draft_calls` / `draft_ms` (the draft passes since load, every lane, and their time); `draft_offered` / `draft_accepted`; `draft_shared` (draft calls made while other lanes decoded); `decode_steps` / `grouped_steps` (lane steps, and those that ran in a group of several lanes). |
| Diagnostic | `IE_MIMO26_GAP_LOG=1` | Every sampled row's pick and its top three raw logits, by lane and by the position the row predicts (gate c, at the served divergences). |
| Test tool | `tools/mimo26/parallel_serve_test.py` | `aba --n-list 2,4` (several concurrent arms between the solo arms); `--solo-all` (every prompt's drafted solo, so the solo reference has the concurrent arms' prompt mix); `--prompt-files`; per-arm drafter figures; server rates summed over gap-free spans. `identity --diverge-ok` (a batched lane's divergence from its solo run is reported, not failed) and `--ref-solo`. |
| Not changed | `mimo26_run_ids` (`--parallel 1`), the serial `forward()`, `Mimo26DFlash` (the drafter), every other model | -- |

Host checks: the eight host ctests the B2-B4 builders ran pass 8/8 (`mimo26_host_rules_test`, `mimo26_lanes_test`, `ds41_prefix_key_test`,
`idle_spin_test`, `server_admission_test`, `ie_vitals_test`, `ds41_sampler_test`, `openai_proto_test`). The policy rules have 30 host
checks (`p4b5_policy_test.cpp`, including 128 two-lane groups staying within 8 rows). They are compiled from the scratchpad against
this branch's header and copied to `results/mimo26/p4/b5/scripts/`, not registered as a ctest: `tests/` is outside this builder's files.

## Gates (all observed)

**Run discipline.** Every GPU job ran through the main checkout's `scripts/ie-run-guarded --mem 220G --timeout 540`, in the foreground,
one at a time, in the coordinator's window 12:04-12:59.
- Before each job (`p4b5_check.sh`): no Dream desktop, no engine process, no compiler running, both cards at 26 / 34 MiB, the last
  10 minutes of `journalctl -k` free of CAT errors / GT resets / timeouts, and the GPU lock free.
- After each job: 30 s, then the same check. Every check of these 13 runs was clean.
- One other agent's build started at 12:20:34 in the 30 s gap after run 3. My preflight waited until it exited (12:21:38).
- Server: `ie serve ~/models/MiMo-V2.6-Flash-RL --host 127.0.0.1 --port 1148x --ctx 32768 --parallel 4 --slot-ctx 16384`, stopped with
  `POST /admin/shutdown`; every server exited 0.
- Logs: `results/mimo26/p4/b5/<run>_{server,test,cpu}.log` (untracked); scripts: `results/mimo26/p4/b5/scripts/`.

**Binaries (md5).**

| Binary | Commit | Runs |
|---|---|---|
| HEAD `ie` e9fe4ab8…, `ie-mimo26-run` 8c21b2eb… | 4d0e648 (copied from the B4 builder's tree; its src/include/tools equal 4d0e648 file by file, and no source is newer than the binary) | c_ident_head |
| `ie` bb3b8e00… | 6190494 (budget default 0, gap log with the top 2) | g_p8, g_p0, g_g1, c_ident_b0, c_ident_k3, c_ident_k7 |
| `ie` 8fedd31f… | + the top-3 gap log, the per-request line's wording | c_ident_k7b |
| **`ie` c5b5a8f0…**, `ie-mimo26-run` cac41fc0… | **e0a2027 = the committed tree (default budget 8)** | a1_branch_serve, a3_branch_run, d_memory, c_ident_nodf |
| `p4b5_tf_scan` 23af8f46… | the gate harness, against 6190494's `libie_core.a` | c_tfscan |

The throughput and identity runs on the earlier binaries set `IE_MIMO26_DRAFT_BUDGET` explicitly (8 or 0), which is exactly the
final binary's behaviour with the default 8 or the knob 0. The changes after 6190494 are two diagnostics and the default.

### (b) Throughput: served A-B-A (CPU leg on, greedy, thinking off, no tools)

Command:
`parallel_serve_test.py <port> <pid> aba --n-list 2,4 --max-tokens 256 --prompt-dir results/mimo26/p4/prof/heldout_prompts --prompt-files 03,04,05,07 --warm --solo-all`.
- Each held-out transcript (5,408-6,008 tokens) is first prefilled alone into its own lane.
- A1: every prompt alone, one after another. The lone lane drafts 7, B4's rule.
- B2: prompts 03-04 concurrently. B4: all four concurrently. A2: the solos again.
- Rates are the server's own: /health `tokens` over the 0.5 s samples in which exactly 1 / N lanes decode, summed over gap-free spans.
- "Mix solo" is the drafted-solo rate of the same prompts, from the server's `[gen]` lines of A1 and A2.

| Run (binary bb3b8e00) | Solo A1 / A2 | 2 lanes | x mix solo (22.76 / 23.38 / 22.13) | 4 lanes | x mix solo (22.46 / 23.00 / 21.59) |
|---|---|---|---|---|---|
| `g_p8`: **budget 8** (7 drafts at 2 lanes, 3 at 4) | 22.77 / 21.84 | **28.38** (17.0 s) | **x1.25** | **28.46** (28.6 s) | **x1.27** |
| `g_p0`: budget 0 = the drafter-off lanes (B4) | 23.27 / 22.53 | 23.41 (21.5 s) | x1.00 | 25.18 (39.5 s) | x1.09 |
| `g_g1`: budget 8, one lane per group (7 drafts at 4 lanes too) | 21.93 / 21.30 | 27.10 (17.5 s) | x1.22 | 26.96 (26.0 s) | x1.25 |

- **Budget 8 vs the drafter-off lanes:** x1.21 at 2 lanes (28.38 / 23.41), x1.13 at 4 lanes (28.46 / 25.18). Normalized by each
  server's own solos: x1.25 and x1.16.
- **Budget 8 vs the drafted solo:** x1.25 and x1.27. Per lane: 14.2 tok/s at 2 lanes (drafter-off lanes 11.7) and 7.1 at 4 (6.3).
- **Two lanes per group with 3 drafts vs one lane per group with 7 drafts, at 4 lanes:** x1.27 vs x1.25 of their own solos. Equal
  within the noise, so the AUTO pairing stays.
  - I had estimated, from B4's traces, that one lane per group with 7 drafts would win: a same-sequence draft row costs ~31 ms in
    a lone step and a cross-lane row ~80 % of a step. That estimate was not confirmed.
  - `g_g1` also ran under the heaviest background load: its solo 8-row step took 317 ms against 296 in the other two runs.

| Drafter, from /health (g_p8) | Draft passes | ms per pass | Drafting ms per token | Drafts kept | Tokens per lane step | Lane steps grouped |
|---|---|---|---|---|---|---|
| Solo (A1) | 216 | 10.3 | 2.28 | 764 / 808 (95 %) | 4.54 | 0 |
| 2 lanes | 96 | 10.2 | 2.03 | 387 / 392 (99 %) | 5.03 | 0 (one lane per group, pipelined) |
| 4 lanes | 248 | 10.2 | 3.12 | 567 / 588 (96 %) | 3.29 | **248 / 248** |
| 4 lanes, one lane per group (g_g1) | 135 | 10.2 | 1.96 | 567 / 586 (97 %) | 5.20 | 0 |

- **What the drafter costs with several lanes** (the question in the brief). Each draft pass is ~10.2 ms on the last card's stage
  thread, inside the lane's done callback.
  - It is on the critical path: the lane's next step needs its drafts, and card 1 cannot run the other group meanwhile.
  - At 2 lanes it is 2.0 ms per token, ~6 % of the aggregate's 35 ms per token. At 4 lanes in pairs it is 3.1 ms per token, ~9 %:
    four passes per cycle, one per lane, each for at most 3 drafts.
  - It does not erase the gain. A batched pass for a pair (one forward of the drafter for both lanes' blocks) could win back
    about half of it at 4 lanes. Not built (see "Not done").
- **Pairing engages with the drafter on.** At 4 lanes every lane step ran in a 2-lane group (248 / 248; /health `grouped_steps`).
  - The B3 gate's G14 (AUTO not engaging with the drafter, in the lanes-test tool) did not reproduce in serving. Its cause stays
    unverified.
  - My reading: in serving, a finishing pair's callbacks resubmit both lanes under B3's `pgate_`, so the pair re-forms at every
    step.
- **Why 2 drafted lanes gain less than 2 plain lanes did in B2** (x1.25 over the drafted solo, against B2's x1.69 over the plain
  solo). The pipeline overlaps heavy steps poorly.
  - An 8-row verify step takes 296 ms alone and a mean 428 ms while the other lane's step runs on the other card (step trace,
    submit to callback, 189 and 48 steps).
  - Per lane, a step (of any row count) completes every 354 ms at 2 lanes, for 5.03 tokens, against every 199 ms solo, for 4.54.
  - Both cards' expert work then runs at once: two PCIe links, the one 12-core CPU expert leg, host DRAM.
  - Unverified: which of these saturates is not profiled.

**Background load (every rate is uncertain).** The box was not quiet:
- load average 4-12;
- Dream's test suite (`python`, `chrome-headless`) and the owner's firecrawl stack (`beam.smp`, bursts of 3-4 cores);
- samples with another process over 50 % CPU: 16 / 40 (g_p8), 4 / 42 (g_p0), 32 / 41 (g_g1).

The within-run ratios are the robust figures. The cross-run comparison (budget 8 vs 0) rests on one run each, with margins of
+13 % and +21 %, above the ±5 % run-to-run noise B2-B4 measured. The coordinator's suggested quiet-box repeat was not run (below).

### (c) Identity (CPU leg off, greedy; `IE_DS41_CPU_MISS=0`)

`parallel_serve_test.py identity --n 4 --max-tokens 96 --cancel-idx 1 --cancel-after 6`: 4 Dream-shaped conversations (~1.62k tokens,
8 tools, a different first system token each), warmed into their own lanes. Then a solo arm (one at a time: a lone lane), a batch arm
(all 4 at once), a cancel arm (all 4, request 1 closed after 6 chunks) and one more request. Every arm is served from the lanes' live states.

| Run (binary) | Configuration | Result |
|---|---|---|
| `c_ident_nodf` (c5b5a8f0) | drafter off (`IE_MIMO26_DFLASH=0`): plain rows everywhere, `--ref` B4's own `c_identity_final_ref.json` (e9fe4ab8) `--ref-solo` | **PASS**: batch == B4's batch 4/4, solo == B4's solo 4/4, batch == solo 4/4, cancel 3/3, cancel == batch. |
| `c_ident_head` (HEAD e9fe4ab8) | drafter on (B4: a lone lane drafts 7, lanes together plain) | the reference texts; batch vs solo 2/4, cancel 2/3 (B4's own finding: request 1 at char 208, request 2 at 210) |
| `c_ident_b0` (bb3b8e00) | drafter on, **budget 0** | solo == HEAD's solo **4/4** (the one-lane path, gate a). Batch == HEAD's batch 3/4 -- see the note below. |
| `c_ident_k3` (bb3b8e00) | drafter on, **budget 8 with `IE_MIMO26_DFLASH=3`** (3 drafts for a lone lane and in the groups: the same shapes everywhere) | **PASS, strict**: batch == solo **4/4**, cancel **3/3**, cancel == batch. The 4 batched lanes drafted while the others decoded (21-34 draft calls each). This run did not record how many of their steps ran in pairs (no /health sampling in identity mode); g_p8's 4-lane arm paired 248 / 248. |
| `c_ident_k7` / `c_ident_k7b` (bb3b8e00 / 8fedd31f) | drafter on, **budget 8** (the default: 7 drafts alone, 3 in pairs) with `IE_MIMO26_GAP_LOG=1` | batch vs solo 1/4 and 3/4, cancel 1/3 and 1/3 (the shapes depend on when the lanes join: a lane that decodes alone drafts 7). **Every first divergence is a near-tie** (`p4b5_gaps.py`, the top-3 rule below). |

- **Note on `c_ident_b0`.** With the drafter on, B4's own "batch" arm is not a fixed shape. A lane that happens to decode alone for
  a step (while the others take their serial turns, or at the end) drafts 7 by B4's rule.
  - In HEAD's run, lanes 0 and 3 did that once each; in the budget-0 run, lanes 2 and 3 did (2 and 1 passes, all "0 while other
    lanes decoded").
  - Request 1 (lane 2) diverged at char 250. The budget-0 run never drafted with other lanes decoding (`draft_shared` 0).
  - With the drafter off, where the shapes are fixed, the branch equals B4 byte for byte (`c_ident_nodf`).
- **Served divergences with the gap log** (`c_ident_k7b`; solo row = the reference, near-tie = the other pick in its top 3 within
  0.75 of its top logit):
  - batch lane 1 at position 1708: solo 10135 (25.916) vs 19597 (25.718): gap 0.198.
  - batch and cancel lane 3 at 1714: 320 (28.176) vs 304 (27.621): 0.555.
  - cancel lane 1 at 1657: 4616 (28.704) vs 3047, the solo row's 3rd (28.031): 0.672.
  - The first `c_ident_k7` run logged only the top 2 and so read the 1657 case as "not a near-tie". The top-3 log of the
    deterministic solo arm settles it.
- **Teacher-forced scan** (`c_tfscan`: `p4b5_tf_scan`, adapted from the B3 gate's harness; 3 held-out transcripts x 96 positions;
  every arm feeds the reference's greedy trajectory). The arms differ only in how the rows are stepped:

| Arm vs arm | Rows | Not bit-identical | Argmax flips (near-ties) | Max ref gap at a flip | Mean \|dlogit\| |
|---|---|---|---|---|---|
| p2x8 (2 lanes through the pipe, 8-row verify steps) vs s8 (solo, 8-row steps) | 291 | **0** | 0 | -- | 0 |
| p4x4 (4 lanes, 2-lane groups of 4+4 rows: 22 group steps per card at 8.00 rows) vs s4 | 291 | **0** | 0 | -- | 0 |
| s8 (the drafted solo's shape) vs s1 (one row per step) | 291 | 288 | 3 (3) | 0.109 | 0.324 |
| s4 (the budget's shape at 4 lanes) vs s1 | 291 | 288 | 1 (1) | 0.213 | 0.304 |
| s4 vs s8 | 291 | 288 | 4 (4) | 0.541 | 0.319 |

  So the pipe and the groups add nothing: a lane's logits are exactly those of its own step shape run alone. What differs between a
  lane's batched run and its drafted solo run is the verify step's row count (7 vs 3 drafts). That moves the logits by a mean
  0.3 (max 13.9 on some tail logit) and flips a greedy pick only at near-ties: 8 of 8 flips here, gaps 0.009-0.541. This is the
  P5 behaviour of the drafted verify, the same class as solo drafting against one-row decoding.

### (a) One lane == HEAD

| Check | Run | Result |
|---|---|---|
| `--parallel 1` SSE bytes (`p4b5_sse_capture.py`: 3 cold requests; Dream sampling T 0.7 seed 1234 thinking on; the same with `ie_vitals`; a greedy tool call) | `a1_branch_serve` (c5b5a8f0) | sse 1 **a6268600…** and sse 3 **eb5ae0ab…**: raw-identical to B4's captures from HEAD 1f0382e and from B4's branch binary 83d9244f. sse 2 (`ie_vitals`): identical to both once the summary's three wall-clock fields are masked (16ce03ce… all three; `p4b5_sse_cmp.py`). |
| `serve_test.py` | same run | every line ok (chat, stream, tool_call, tool_turn, cancel, reuse, memory 361-362 MiB flat over 20 requests, props) |
| `ie-mimo26-run`: 8 held-out prompts x 64, `--chunk 2048 --ctx 16384 --static 0 --ranking <chat ranking> --dflash 7 --dflash-minp 0.7`, `IE_DS41_CPU_MISS=0` (B3's R14/R15, B4's a3) | `a3_branch_run` (cac41fc0) | generated md5 **65501302fb19267820d991815210fb2d** (== R15 / B4's a3). VRAM 31.21 / 28.05 GB and static / pinned 78/178, 62/194 identical. Decode ms/token 72.0 84.5 70.2 77.5 74.1 88.5 87.0 85.8 (R15: 71.6 84.3 70.2 77.3 74.1 88.3 87.0 85.7). |
| One lane decoding at `--parallel 4` | `c_ident_b0` (budget 0), `c_ident_k7`'s and `c_ident_k7b`'s solo arms (budget 8) | the solo arms equal HEAD's solo arm 4/4. The one-lane path is `serve_prepare_decode`'s unchanged B4 call. |

Unverified on (a)'s SSE check: the comparison is against B4's captures of 1f0382e and 83d9244f, not a fresh capture of the 4d0e648 binary
itself. B4's doc says the difference from 83d9244f to 4d0e648, the two `pipe_submit` refusals, cannot be reached at `--parallel 1`.
The `ie-mimo26-run` check does compare against the same command on the HEAD binary (B3's R15, 65501302…).

### (d) Memory and device errors

`d_memory` (c5b5a8f0, default budget 8, `IE_MIMO26_PROMPT_CACHE_GIB=0` as in B4): `parallel_serve_test.py memory --requests 20 --n 4 --max-tokens 48`.
- 20 Dream-shaped requests in waves of 4, 20/20 complete. The lanes drafted together: 300 of 316 draft calls came while others decoded.
- **RssAnon** 335 → 819 (after wave 1) → 827, 830, 834, 836 MiB: +17 MiB after wave 1, under the 64 MiB criterion.
  - B4 measured +9 MiB on the same test and its gate +14 MiB over 16 requests ("the RssAnon creep", a soak unverified).
  - So this is the same small creep. Whether it is bounded is unverified: 20 requests do not show it.
- **VRAM** [30326, 31090] → [30345, 31110] → [30345, 31109] MiB. Threads 52 → 54 → 54.
- No device error in any run (the pre- and post-checks above).

## Defaults picked

- **`IE_MIMO26_DRAFT_BUDGET` 8** (was 0 = B4's rule), from the (b) A-B-A. 0 restores B4 exactly.
- **Even share** (`IE_MIMO26_DRAFT_WEIGHT` 0).
- **AUTO groups** (`IE_MIMO26_GROUP_LANES` 0). One lane per group was measured equal within noise at 4 lanes and not better.
- The budget is the pipe's group row cap. A value below the lanes per group splits groups: 1 = one plain lane per group, worse than 0.
- Unchanged:
  - the drafter's k = 7 and min-p 0.7 for a lone lane;
  - the group q* 0.35 for multi-lane groups (B3's, tuned on 1-row lanes);
  - 8 stream slots;
  - `IE_MIMO26_DF_FEED` (ignored while a budget is set).

## Risks

- **Throughput evidence is one A-B-A per configuration on a busy box.** Solo arms agree within 3-5 % inside each run, and the
  budget-vs-0 margins (+13 %, +21 %) exceed B2-B4's ±5 % noise. The quiet-box repeat the coordinator suggested was deferred.
  Prompts: 4 held-out Dream transcripts. Drafts are highly accepted there (92-99 % of the offered drafts kept, after the min-p cut).
  Lower-acceptance text (prose, the synthetic Dream-shaped prompts) was not measured. With low acceptance, 3-7 drafts per lane could
  cost more than they return. Unverified: the min-p cut limits that, but it is unmeasured.
- **Greedy replies now depend on other traffic in one more way.** A lane's verify shape depends on how many lanes decode at that
  step: 7 drafts alone, 3 in pairs. That moves greedy near-ties (gate c).
  - This is the same class as B4's lone-lane drafting and as the CPU leg. Replay and identity checks belong at `--parallel 1`, or with
    the drafter off (design 2.6).
  - B4's gate risk "with the drafter on, a greedy reply can depend on other traffic (near-ties unverified)" is now verified for
    these runs: every first divergence was a near-tie.
- **The drafter's ~10 ms passes sit on the last card's critical path**, one per lane per step: 6-9 % of the aggregate here.
- **The acceptance weighting is host-tested only.**
  - It needs a callback group of several lanes, so it only acts when lanes pair (3+ lanes).
  - A lane that loses every spare row to a better neighbour stops updating its estimate until it drafts again, for example when it
    decodes alone.
- **Lane-count mismatch while a lane prefills** (speed only). The even share uses decoding lanes; the pipe's AUTO cap uses lanes in
  flight. While a chunk runs, a share can be sized for groups the pipe then splits. This is transient.
- **ba287bd was not merged here** (the permission classifier denied the merge; the coordinator merges at integration). Its lane
  choice ranks lanes that leave `min(max_tokens, capacity / 4)` of room first. Every run here fits every lane:
  - (b): 5.4-6.0k-token prompts + 256 tokens, lanes of 16,384 / 32,768;
  - (c): ~1.6k + 96; (d): ~1.6k + 48.

  So, likely but not verified, by code reading: ba287bd would choose the same lanes and the numbers carry over. The merge should conflict
  trivially: both branches edit the adjacent per-request `[mimo26 dflash] lane` / `[mimo26 lanes] lane` fprintf lines. Keep both.

## Not done

- **Measured in serving:**
  - the acceptance weighting (`IE_MIMO26_DRAFT_WEIGHT=1`);
  - budgets other than 0 and 8;
  - 3 lanes;
  - the synthetic Dream-shaped prompts (`s_p8` / `s_g1` / `s_p0` exist in the run script);
  - the coordinator's quiet-box repeat of the 2-lane budget 8 vs 0 arm.

  All were deferred to keep the GPU window at ~55 min.
- **A batched drafter pass per group** (one drafter forward over both lanes' blocks). It would save ~4-5 % at 4 lanes by the /health
  figures above and needs a deferred group submit. Not built: the pairs' gain is modest, and one lane per group already matches pairs.
- **A repo ctest for the policy rules.** The test lives in `results/mimo26/p4/b5/scripts/p4b5_policy_test.cpp`: `tests/` is outside
  B5's files.
- **The group q\*** (0.35) and the stream slots were not re-swept for drafted groups (2 lanes x 4 rows).
- **Rebalancing the layer split** so that card 1, which holds the head, the drafter and the callbacks, does less: follow-up idea, not
  measured.

## Independent gate (2026-09-26): PASS

Gated on the final integration tree 2a9c994 (B5 merged at f188114, plus B6b), built independently (`ie` 568f9856).
- (a) `--parallel 1` SSE bytes equal B4's recorded HEAD captures (sse 2 with its wall-clock fields masked); the
  `ie-mimo26-run` identity md5 is 65501302fb19267820d991815210fb2d; `/health` has no lanes fields at `--parallel 1`.
- (b) Direction confirmed, one run per arm: 2 lanes, budget 8 x1.25 over solo against budget 0 x1.04.
- The B4 reply-budget lane rule still holds with lanes decoding together (4 lanes, 662 grouped steps).
- (c) With k=3 and the CPU leg off, batch == solo 4/4 and the 3 cancel survivors == solo.
- (d) RssAnon +12 MiB after wave 1, VRAM flat, threads 54; `/admin/shutdown` with 4 drafting lanes exits cleanly in 1.5 s.
- Not tested by anyone: the weighted mode, budgets other than 0 and 8, and 3 lanes.
