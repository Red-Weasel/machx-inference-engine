# P4 B7 — GLM-5.3-Flash request lanes and the two-card decode pipe (2026-09-26)

Design: `~/ds41_work/p60/p4_b6_b8_design.md` section 5.2 (B7a lanes + B7b pipe). Template: MiMo B1-B3
(`docs/mimo26/P4_B1_LANES.md`, `P4_B2_PIPELINE.md`, `P4_B3_ROWS.md`). Branch `p4-b7-glm-lanes` from `deepseek4-vision-exp`
3140283. Owner's constraint: additive; one lane (the default) must stay the pre-lane model. Serving is unchanged: `ie serve`
still runs GLM one request at a time. The serve integration and the prompt-end snapshot (the design's B7c/B7d) are not in
this release: the owner ranks GLM low priority, so the lanes stay default-off.

## Answer first

Two conversations now decode at once on GLM-5.3: card 0 runs one lane's half of a step while card 1 runs the other lane's.

| Arm (2 lanes, 2 cards, prompts at ~2K tokens) | 2-lane aggregate | Solo | Ratio |
|---|---|---|---|
| Served config: CPU expert leg on (q* 0.40), banks pinned, A-B-A | **22.26 tok/s** | 13.28 / 13.27 tok/s | **x1.676** |
| CPU leg off (identity config) | 18.16 tok/s | 9.54 tok/s | x1.90 |

- With the CPU leg off, every lane's output is **byte-identical** to its solo run: the fp16 logits after the prompt and after
  every decode step (FNV over the bytes) and the greedy tokens, 4 agent prompts x 48 tokens with 2 mid-run re-admissions
  (and the serial interleave too, in the smoke run).
- Each card goes from 58-60 % busy during solo decode to 99.5 % with 2 pipelined lanes (served config, `/proc/self/fdinfo`).
- One extra lane at the default 32K context costs 244.5 MiB of VRAM on card 0 and 270.3 MiB on card 1, taken out of the
  expert cache (22.49 -> 22.25 and 22.49 -> 22.23 GiB; the minimum slots per layer 78 -> 77 and 71 -> 71); host RAM: 64
  MiB of residual buffer per lane, allocated before the load so the pinning floor counts it. MemAvailable never went under
  60.1 GiB in any lanes run (floor 40).
- Gate (a), 1 lane == HEAD through `ie-glm5next-run`: **PASS**. HEAD 3140283 and the branch give identical text on 2
  held-out prompts with the CPU leg off and on 1 in the served config, with the same VRAM, expert cache and pinned banks and
  decode within noise.
- Gate (b) was re-run on the final binary (850e3eec, 6f79be7's tree) in a later window: **PASS**, the same numbers as on
  the earlier binary. A first re-run attempt at 09:41 stalled and ended in a card 1 engine reset at its timeout kill (see
  "Incident"; cause not proven, leading theory other agents' compile load).
- Owner decision (via the coordinator): GLM is low priority, so there is no serve integration for GLM in this release. The
  lanes stay default-off (one lane), so GLM's behaviour does not change for anyone.

## What was built

| Piece | Where | What |
|---|---|---|
| Request lanes per stage | `Glm5NextModel::set_seq_lanes / select_lane / n_lanes / lane / lane_ctx / lane_pos / lane_bytes` (`include/ie/glm5next.hpp`, `src/model/glm5next.cpp`) | A lane is one sequence's state on a stage: the KDA scan + conv state (`DeltaNetState dn_`), the MLA latent cache, the DSA pooled index keys and the open pool's key/gate rolls, and the depth `kv_len_`. GLM is two model objects (one per card), so a per-object swap is enough: `select_lane` swaps the lane's pointers, `max_ctx_`/`n_pools_` (its capacity and pool stride) and `kv_len_` into the live members, no copy. Lane 0 is the state `init_runtime` always allocated; with one lane (the default) nothing new is allocated and `select_lane` is never used, so the launches are the pre-lane ones. The workspaces stay sized by lane 0's context, which every lane's is at most. Refused with the MTP kit (spec decode, pipedraft) and `IE_G5_EP_DECODE`. The member is `seq_lanes_`; `lanes_` stays the fill-lane I/O workers. |
| Lane VRAM | `init_runtime` | Lanes 1..n-1 are allocated after the workspaces and before the expert cache, and `lane_bytes() x (n-1)` is subtracted from the cache budget (`IE_G5_ECACHE_MB`, or the server's cache budget), so the stage's VRAM total does not move. A lane that does not fit the budget is refused with the numbers. The allocated bytes are checked against `glm5_lane_bytes` (`include/ie/glm5_server.hpp`, next to `glm5_runtime_reserve`, for the serve item). |
| Stage body | `Glm5NextModel::lane_stage` | What the serial lanes and the pipe both run for one stage of one lane's step: select the lane, reset it when pos0 == 0 (a new sequence, as the engine resets at pos 0), `forward_range`, and on the tail stage the fp16 logits into the caller's host buffer. On any error (or exception) it drains its queue before returning, so nothing of a failed step still reads the caller's ids/residual buffers. `pipe_refusal()` names what a pipe cannot run (the MTP kit, EP decode, spec verify, the serial-only diagnostics). |
| The card pipe | `Glm5LanePipe` (`include/ie/glm5_lanes.hpp`, `src/model/glm5_lanes.cpp`; no SYCL) | One host thread per stage, one step in flight per lane, a per-lane host residual crossing the stages. A step goes stage 0 -> 1 in FIFO order; the last stage commits the lane's position and calls `done(lane, T, pos0)` on its own thread, which may submit the lane's next step. MiMo's hazard rules are kept: the claim is check-and-set under the lock and held THROUGH the done callback (only that callback's thread may submit the lane, once); a stage error drops the step, latches, and every submit is refused until `stop()` returns it; `stop()` waits for no step in flight and joins; `stop()` from a callback returns an error instead of deadlocking. New here: a lane whose step failed part-way takes only a new sequence afterwards (its KDA state may have advanced on some layers), and the per-lane buffers are allocated and touched in the constructor. |
| Host test | `tests/unit/glm5_lanes_test.cpp` (ctest `glm5_lanes_test`, no GPU, no SYCL) | Fake stages: every step through every stage in order with its own lane's residual pattern (a cross-lane mix-up fails it), the stages overlapping on different lanes, the claim rules, pos0 validation, the error latch and the failed-lane rule across a restart, a throwing callback, `stop()` from a callback; `glm5_lane_bytes` against the design's 0.24 + 0.26 GiB at 32K. Clean under ThreadSanitizer (g++ `-fsanitize=thread`: 6 runs of the first version, 4 of the final one with the failed-lane rule). |
| Gate tool | `tools/glm5next_lanes_test.cpp` -> `ie-glm-lanes-test` | `ie-glm5next-run`'s two-card load (blocks [0, 23) + [23, 45), `IE_G5_ECACHE_MB` 23040 per card unless set, banks pinned with `IE_G5_PIN_BANKS=1`) with L lanes. Every prompt SOLO on lane 0, then over the lanes: `--serial` (B7a, one lane's step at a time, round robin) and/or `--pipeline` (B7b); a finished lane takes the next queued prompt mid-run. Per sequence: FNV of the fp16 logits after the prompt and after every decode step, tokens, and the lane's final position, all against solo. `--aba` adds a warm-up and a second solo arm and reports the steady-window aggregate (from every first-wave lane's 4th decode step to the first of them finishing). Also: lane VRAM vs the arithmetic, expert-cache hits and CPU experts per phase, per-card engine busy from `/proc/self/fdinfo`, MemAvailable's minimum per phase vs the 40 GiB floor. `--cpu-miss` keeps the q* leg on (identity lines then informational); `--dry` prints token counts and planned lane VRAM without a device. |

## Gates (observed; logs in `results/glm53/p4/b7/`, untracked)

**Run discipline.** Every run: the main checkout's `scripts/ie-run-guarded --mem 225G --timeout 420-470`, foreground. Before
each: no Dream app, both cards at 26/34 MiB, `journalctl -k --since "-10 min"` clean of CAT/reset/timeout lines, MemAvailable
>= 190 GiB, no compile and no other engine process. After each: 30 s, then the same kernel-log check: clean every time. The
wrapper is `p4b7_gpu.sh` (scratchpad; the `.cmd` file next to each log records the command, the `IE_` env and the binary's
md5). Model: `~/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL`. Binaries: `ie-glm-lanes-test` 85e169b9aac3c81110fadc7d0c18407a
(bdb6f66's tree) for s1, b1, c1; the final build from 6f79be7's tree (the tip's code; later commits are docs) is
`ie-glm-lanes-test` 850e3eecefc8a2a74ff167ac96329f73 (w1, b3) and `ie-glm5next-run` 9b4d1680703794ad535bc1832788aa87
(gate (a)); HEAD's `ie-glm5next-run` b1a2797550aa5ef0e6eb5411d4ddc8f5. Since 09:52 the preflight also refuses while any
`icpx`/`clang`/`clang++`/`ninja`/`cc1plus`/`ld` runs, and a read-only monitor samples every 20 s into `<run>.mon`. Prompts: the MiMo held-out agent transcripts (`results/mimo26/p4/prof/heldout_prompts/03-09`)
re-rendered in GLM's chat template (`p4b7_mkprompts.py`: `[gMASK]<sop>`, `<|user|>`, `<|assistant|><think></think>`, tool
calls as `<tool_call>NAME<arg_key>..`, results as `<|observation|><tool_response>..`), cut at a turn boundary, ended with the
assistant header: glm_03 1,360 tokens, glm_04 2,588, glm_06 2,327, glm_07 2,517, glm_08 1,211, glm_09 1,831.

| Gate | Run | Result |
|---|---|---|
| smoke (serial lanes B7a + pipe, CPU leg off) | s1_smoke: `--pipeline --serial --lanes 2 --n 12 --max-prompt 1024`, prompts 08, 03, 09 | **PASS**. Interleaved == solo and pipelined == solo for all 3 sequences (12 logits rows + 12 tokens each), 1 mid-run re-admission in each mode. |
| (b) 2 pipelined lanes == each lane's solo run, CPU leg off, 4 prompts incl. agent ones, mid-run re-admissions | b1_pipe_identity: `--pipeline --lanes 2 --n 48`, prompts 07, 03, 09, 08 | **PASS**. All 4 sequences byte-identical to solo: 48 logits rows (fp16 FNV) and 48 tokens each; 197 pipe steps, **2 mid-run re-admissions**. glm_07's 2,517-token prompt puts its decode past the selection width (2,051), so the DSA indexer's real top-k (more pools than the 512 it keeps) was part of the identity. The generated text is tool calls (`<tool_call>Read<arg_key>file_path</arg_key>...`). |
| (b) re-run on the final binary (850e3eec, 6f79be7) | b2_pipe_identity_final (09:41), same command | **Did not complete**: stalled in the solo phase and ended by the guard's timeout (see "Incident"). |
| warm-up of the final binary | w1_warmup_final (10:59): `--pipeline --lanes 2 --n 4`, prompts 07, 03; `--timeout 540` | PASS (4 rows + 4 tokens each == solo), 223 s. The JIT caches did not change (libsycl_cache 1,497 files / 135,136 KiB, neo_compiler_cache 2,758 / 551,600 before and after). |
| **(b) re-run on the final binary** | b3_pipe_identity_final (11:03): the b1 command, `--timeout 540`, no compiler running | **PASS**: all 4 sequences byte-identical to solo (48 logits rows + 48 tokens each), 197 steps, **2 mid-run re-admissions**; x1.901 (18.15 vs 9.55 tok/s); MemAvailable minimum 62.96 GiB; 307 s wall; kernel log clean. The expert-cache counters equal b1's exactly (solo 54,121 hits / 110,312 misses, pipelined 58,041 / 106,392): with the CPU leg off the cache behaviour is deterministic too. JIT caches unchanged across the run. |
| (c) A-B-A aggregate at 2 lanes, served config | c1_aba_served: `--pipeline --aba --cpu-miss --lanes 2 --n 128 --max-prompt 2000`, prompts 04, 06 | **x1.676**: solo A1 13.28, pipelined **22.26** aggregate (245 decode steps in an 11.0 s steady window), solo A2 13.27 tokens/s. Per-lane step 89.6 ms (submit to logits) vs 75.3 ms solo. CPU experts per decode step 71.6 solo / 73.6 pipelined / 76.7 solo A2. |
| (d) memory and device health | all lanes-test runs | MemAvailable minimum 63.55 GiB (s1), 64.31 (b1), 64.46 (c1), 60.12 (w1), 62.96 (b3), from the end of the load to the exit; floor 40. Both stages pinned every bank (82.7 + 90.0 GiB, 0 left on mmap). Kernel log clean before and 30 s after every run except b2 (the incident). |
| (a) 1 lane == HEAD | `ie-glm5next-run $M --gpus 2 --ctx 32768 --ppl <prompt> --prefill P --ngen 48 --no-pipedraft`, HEAD (3140283, exported and built separately; md5 b1a27975...) vs branch (6f79be7; md5 9b4d1680...), interleaved H-B-H-B-H-B | **PASS**. glm_07 (P 2,300, CPU leg off): text identical (md5 090684de...), decode 8.48 / 8.49 tok/s. glm_09 (P 1,800, off): identical (fd043fe2...), 9.21 / 9.25. glm_04 (P 2,000, **served**, q* on): identical (36a5d33c...), the same q* split (2,011 + 1,851 CPU experts), 12.38 / 11.92 tok/s (-3.7 % on a 48-token decode: noise, unverified -- nothing on the one-lane decode path changed). Every pair: device 5.46 + 4.39 GiB, expert cache 78 / 71 slots per layer and 22.49 + 22.49 GiB, pinned 20 + 22 layers (82.7 + 90.0 GiB). Note `--prefill` encodes the prompt file without special-token parsing: the same fixed token stream for both binaries. |

With the CPU leg on, identity is informational and not expected: pipelined vs solo differ from logits row 1, and so do the
two solo arms (A2 vs A1 at row 2 on glm_04; glm_06 identical) -- the q* leg's CPU/GPU split follows the expert cache's history
(`CAMPAIGN_2026-09-02.md`), as in serving today.

### Timing (steady windows; same process A-B-A)

| | Solo decode | 2 lanes pipelined | Ratio | Per-lane step | Card busy (fdinfo, card 0 / card 1) |
|---|---|---|---|---|---|
| CPU leg off (b1) | 9.54 tok/s (104.8 ms/step) | 18.16 tok/s (85 steps in 4.7 s) | x1.90 | 112.0 ms | solo 54.8 / 56.8 %, pipelined 88.8 / 94.9 % |
| served (c1) | 13.28 / 13.27 tok/s (75.3 ms/step) | 22.26 tok/s (245 steps in 11.0 s) | x1.676 | 89.6 ms | solo 59.9 / 58.4 %, pipelined 99.5 / 99.6 % |

- The served ratio sits inside the design's x1.5-1.7 prediction. The per-lane step grows 19 % under the pipe. Likely causes
  (not profiled): the two q* teams share the P-cores 0-7 (the auto placement puts both stages there;
  `set_cpu_partition(true)` only changes the log line) and the DRAM they stream from, and both PCIe links fill at once.
- As in MiMo B2, xe's bcs and ccs cycles read the same per client, so the fdinfo busy is "context active", not a copy/compute
  split.
- The steady window of the CPU-off run is short (4.7 s: its first-wave prompts differ in length); x1.90 there is
  indicative, x1.676 in the served A-B-A is the number.
- Prefill is not pipelined across a lane's own chunks (one step in flight per lane); two lanes' prefills do overlap.
  Serial prefill measured 72-90 tok/s at 1.2-2.6K tokens. The first 1,024-row chunk of a process ran ~3x slower (32.7 vs
  10.9 s in s1; likely JIT and first touch, not profiled).

### What one extra lane costs (default lane context 32,768)

| | Stage 0 (card 0, blocks [0, 23): 5 MLA + 18 KDA) | Stage 1 (card 1, [23, 45): 6 MLA + 16 KDA) |
|---|---|---|
| KDA scan fp32 + conv f16 | 74.5 MiB | 66.3 MiB |
| MLA latents f16 [ctx, 512] | 160.0 MiB | 192.0 MiB |
| DSA pooled keys + open-pool rolls | 10.0 MiB | 12.0 MiB |
| **One extra lane** | **244.5 MiB** (256,419,840 B, = `glm5_lane_bytes`) | **270.3 MiB** (283,389,952 B) |
| Expert cache at 1 lane (IE_G5_ECACHE_MB 23040; gate (a), both binaries) | 22.49 GiB, 78 slots/layer (min) | 22.49 GiB, 71 slots/layer (min) |
| Expert cache at 2 lanes | 22.25 GiB, 77 slots/layer (min): -0.24 GiB | 22.23 GiB, 71 slots/layer (min): -0.26 GiB |
| Host RAM per lane | 64 MiB residual buffer (1,024 rows x 4 x 4,096 fp32) + 4 KiB ids + 0.3 MiB logits | |

Each extra lane is ~1 % of a card's expert cache (~17 and ~18 slots of ~14.7 MiB, spread over 20 and 22 MoE layers). The per-token cost of
that was not isolated (estimate: well under the run-to-run noise at this size). The context is linear in the latents: at
16K a lane costs 159.5 + 168.3 MiB, at 64K 414.5 + 474.3 MiB.

## Incident (2026-09-26 09:26-09:50): two stalls, card 1 engine resets at a timeout kill

- **09:25:56**, gate (a) HEAD run, served config (`a_head_04_served`, first attempt): loaded normally, then the first 1,024-row
  prefill chunk took **151.4 s (6.8 tok/s)** where every other run took 11.7-22.2 s, the second chunk 20.1 s (normally
  5-6 s), and decode printed one token before `ie-run-guarded`'s 400 s timeout sent SIGTERM (rc 124). The tool's handler
  drains both cards and exits; the kernel log stayed clean and the cards went back to idle. The identical retry at 09:34:56
  ran normally (172 s, rc 0). (The retry overwrote the first attempt's log files; the numbers here are from its output as
  read at 09:33.)
- **09:41:33**, the final gate (b) re-run (`b2_pipe_identity_final`, tip binary 850e3eec..., CPU leg off): loaded normally
  (156.4 s), then produced no output for ~290 s while in its SOLO phase -- plain serial forwards on lane 0, before the pipe
  starts; the same step took ~38 s in b1. The 450 s timeout's SIGTERM came at 09:49:06 (rc 124), and at that second card 1
  (0000:09:00.0) logged 174 `Fault response` lines (172 `-EINVAL`, 2 `-ENOENT`) and `Engine reset: engine_class=ccs ... guc_id=6`
  and `engine_class=bcs ... guc_id=10`. No further xe lines through 09:50:42 (no reset loop). GPU work stopped and the
  coordinator was told.
- **State after (coordinator's check, 09:52):** card 1 logged 172 `-EINVAL` and 2 `-ENOENT` fault responses and one
  ccs + bcs engine reset at 09:49:06 and nothing since; both cards back to 26/34 MiB; no zombie or D-state engine process;
  kernel taint O/E only (no B/D).
- **Leading theory (coordinator, unverified): compile load, not a GPU fault.** Other agents' builds ran during both stalls
  (the B4 gate's two builds around 09:25-09:41, B5's around 09:50); memguard blocks builds only while an `ie-*-run` is
  alive, and `ie-glm-lanes-test` does not match its pattern; GLM's CPU leg and pinned DMA are the most host-bandwidth-
  sensitive of the models. My preflight checked for `ninja`/`cmake`/`icpx` only at the start of each run, so a build
  starting later was not seen. The preflight now also refuses while any `icpx`/`clang`/`clang++`/`ninja`/`cc1plus`/`ld`
  process runs.
- **A second theory (coordinator, unverified): first-use JIT** of GPU kernels in a freshly built binary, slowed by the CPU
  load. The evidence is mixed. The SYCL on-disk cache (`~/.cache/libsycl_cache`, enabled by the engine) got 3 new files at
  09:30, inside the first stall (source not attributable), and NONE between 09:31 and 09:58, i.e. none during the second
  stall (unless the kill came before any write). The driver's cache refreshes file mtimes on hits, so its stamps say nothing
  after the fact. The first 1,024-row chunk per process measured 32.7 s (lanes binary 85e169b9, 1st process), 22.2 / 20.9 s
  (HEAD run tool, 1st / 2nd process) and 11.7 / 11.7 s (branch run tool, 1st / 2nd).
- **Afterwards:** the window re-opened at 10:55 with no compiler running (checked by the preflight). A warm-up of the final
  binary (223 s) and the gate (b) re-run (307 s) both ran normally with `--timeout 540`, the new per-20 s monitor showing
  no compiler at any sample, and neither JIT cache changed across either run.
- **Cause not proven.** At each preflight: no other `ie` process, both cards at 26/34 MiB, kernel log clean,
  no compile running. Both stalls happened in code paths that ran normally minutes before and after (the HEAD binary's own
  serial forwards in the first). Checked afterwards: both physical PCIe links at 32 GT/s x8, no AER/link messages, no xe
  throttle reasons (at idle). One correlate: at 09:42:59, during the second run's load, a Docker container's memory cgroup
  (8 GiB limit; the Firecrawl node workers) went into an OOM storm (10 oom-killer invocations, 57 kernel lines mentioning OOM at 09:42:59, failcnt 914,590); nothing like it
  during the first stall. The resets look like the consequence of the timeout ending a process with GPU work in flight (the
  known "never kill a GLM run" hazard), not a cause (inferred from the timing, not proven).

## Defaults picked

- **Lane context 32,768** for lanes 1..n-1 (`--lane-ctx`; lane 0 keeps the stage context). Every lane may be at most the stage
  context: the indexer's score workspace and the [chunk, ...] workspaces are sized by it.
- **Lanes 1** everywhere by default (the model, the run tool, serve): nothing changes until a caller asks for lanes.
- **q\* placement unchanged**: both stages' CPU teams on the P-cores 0-7 (the measured serial and pipedraft default);
  a split (stage B on the E-cores 8-19) was not measured for 2 pipelined lanes (unverified: the per-lane step's +19 % suggests
  it is worth one A-B-A).
- **No row groups** (the design's call, from MiMo B3's small gain on sparse MoE).
- One step in flight per lane; prefill chunks go through the pipe like decode steps (T <= the chunk, 1,024).

## Risks and limits

- **Identity scope**: greedy, 2 lanes, <= 48 tokens with the CPU leg off, prompts up to 2.6K tokens (decode just past the
  selection width). Not covered: 3+ lanes, sampling, long contexts near a lane's capacity, a stage error mid-flight on the GPU
  (the error paths are covered by the host test with fake stages only).
- **CPU leg on** (served): lanes are not bit-reproducible and neither are two solo runs (above); replay at one lane.
- **Prefill stalls decode**: a lane's 1,024-row chunk holds a card for several seconds while the other lane's decode waits
  behind it (the design's mixed-prefill concern). Not measured here (the steady windows start after the first-wave prefills).
- **Serve** is untouched: `--parallel N` still queues GLM requests FIFO, with no prefix cache; its integration is not in
  this release (owner decision). If it is picked up later: `glm5_lane_bytes` and the pipe's `host_bytes()` are what it
  must add to `glm5_runtime_reserve` and `host_reserve`, and the lanes' logits go through the engine's GPU sampler.
- Cross-lane contamination produces fluent wrong text; only the per-step identity gate catches it.
- **Load time** ~125-156 s with every bank pinned: every gate run fits the 9-minute rule only with 1-2K-token prompts.
- **Two unexplained stalls** in plain serial forwards (one on the HEAD binary), the second ending in card 1 engine resets
  at the timeout kill (see "Incident"). The runs had 1.5-2.7x margin between their expected wall time and the guard's
  timeout (itself capped by the 10-minute tool limit), and the stalls were 3-10x slowdowns. Until the cause is known, a
  run that stops printing should be diagnosed live (`xpu-smi stats`, the process's thread states) before the timeout ends
  it.

## Runbook

```
M=~/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf; P=results/glm53/p4/b7/prompts
IE_G5_PIN_BANKS=1 scripts/ie-run-guarded --mem 225G --timeout 420 build/tools/ie-glm-lanes-test $M --pipeline --serial --lanes 2 --n 12 --max-prompt 1024 --prompt-file $P/glm_08.txt --prompt-file $P/glm_03.txt --prompt-file $P/glm_09.txt
IE_G5_PIN_BANKS=1 scripts/ie-run-guarded --mem 225G --timeout 540 build/tools/ie-glm-lanes-test $M --pipeline --lanes 2 --n 48 --prompt-file $P/glm_07.txt --prompt-file $P/glm_03.txt --prompt-file $P/glm_09.txt --prompt-file $P/glm_08.txt
IE_G5_PIN_BANKS=1 scripts/ie-run-guarded --mem 225G --timeout 470 build/tools/ie-glm-lanes-test $M --pipeline --aba --cpu-miss --lanes 2 --n 128 --max-prompt 2000 --prompt-file $P/glm_04.txt --prompt-file $P/glm_06.txt
```
Observed: `GLM LANES TEST: PASS` each; 269 s, 301 s (b1; b3 on the final binary 307 s) and 407 s wall including the
load. Run only with no compiler running on the box (the preflight's rule since the incident); a warm-up of a freshly built
binary first (`--pipeline --lanes 2 --n 4`, prompts 07 + 03, 223 s) was the coordinator's advice and costs little.

Gate (a), one process per (binary, prompt), interleaved HEAD / branch (`p4b7_gate_a.sh`; HEAD = `git archive 3140283`
built in its own tree; `IE_G5_PIN_BANKS=1`, plus `IE_G5_CPU_MISS=0` for the identity pairs):
```
ie-glm5next-run $M --gpus 2 --ctx 32768 --ppl $P/glm_07.txt --prefill 2300 --ngen 48 --no-pipedraft   # off
ie-glm5next-run $M --gpus 2 --ctx 32768 --ppl $P/glm_09.txt --prefill 1800 --ngen 48 --no-pipedraft   # off
ie-glm5next-run $M --gpus 2 --ctx 32768 --ppl $P/glm_04.txt --prefill 2000 --ngen 48 --no-pipedraft   # served
```
The generated text is compared from stdout (after the last `[pf]` line up to `[host] per token`), stderr kept apart
(`p4b7_cmp_a.py`). Observed: 116-172 s each, text identical in all three pairs.

## Gate (lightweight, 2026-09-26 12:06): PASS

The gate rebuilt branch and HEAD (binaries byte-identical to the builder's), ran the host tests (3/3; ThreadSanitizer 6/6
clean) and one gate (a) identity job: glm_09 at 1 lane, CPU leg off, branch and HEAD give identical text (md5 fd043fe2) with
the same expert cache, device and pinned-bank figures. Non-blocking notes: the test tool's SIGTERM handler does not stop the
pipe's stage threads before draining (best-effort drain in pipe mode, tool only); `pipe_refusal()` does not refuse the
`IE_G5_SLOT_VERIFY` / `IE_G5_SLOT_DUMP` diagnostics (function-static counter); HEAD's first 1,024-row prefill chunk was 5-10 s
slower than the branch's in 4 of 4 pairs (cause unknown; the second chunk and decode are equal).
