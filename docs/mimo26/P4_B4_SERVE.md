# P4 B4 — MiMo-V2.6 `ie serve --parallel N`: several requests decode together on the lanes (2026-09-26)

Design: `~/ds41_work/p60/p4_mimo_batching_design.md` (sections 2.4-2.6, 3, 4 "B4"). B1 lanes: `docs/mimo26/P4_B1_LANES.md`;
B2 pipe: `P4_B2_PIPELINE.md`; B3 groups: `P4_B3_ROWS.md`. Branch `p4-b4-mimo-serve` from `p4-b3-mimo-rows` 1f0382e.

Owner's constraints: additive; `--parallel 1` (the default) is the engine the server always had, byte for byte; every other
model's serving is untouched.

## Answer first

- **All six B4 criteria pass** (table in "Gates"). Final binaries: `ie` e9fe4ab8…, `ie-mimo26-run` 8c21b2eb….
  - (a) `--parallel 1` == HEAD: SSE bytes, `serve_test.py`, `slot_serve_test.py` and the `ie-mimo26-run` token md5 (65501302…).
  - (b) 4 concurrent Dream-shaped requests complete.
  - (c) + (f): with the CPU leg off and the drafter off, batch == solo 4/4 and the cancel survivors == solo 3/3, on two
    binaries, with identical texts across the two servers.
  - (d) an arrival's TTFT while 3 lanes decode is within solo + one chunk.
  - (e) RssAnon +9 MiB after wave 1 over 20 requests; VRAM and the thread count flat.
- **Throughput** (the server's own count, /health `tokens` while all N lanes decode):
  - 2 lanes: 22.4-23.5 tok/s, about what one request makes alone with the drafter (x1.05).
  - 4 lanes on the held-out transcripts: 29.5 tok/s, matching B3's lanes test (28.9-30.1) on the same prompts. The all-4
    window was only 5 s because one reply ended in a tool call. The per-step trace agrees: 135.9 ms per lane step over 137
    steps = 29.4 tok/s, against B3's 138.3 ms.
  - 4 lanes on the synthetic Dream-shaped prompts: 24.5 tok/s (x1.30 their drafted solo).
- **Pipe pause / resume** keeps the stage threads across the serial turns. It is kept: correct, and not slower.
  - The "~300 ms first steps after a restart" that motivated it does not reproduce with a per-step trace. The first decode
    step after a turn costs ~90 ms against ~80 ms later, the same with a pause as with a full stop and restart.
  - With 40 turns inside one 512-token reply, the reply ran at 15.2 tok/s either way.
- **Mixing chunk default changed from 512 to 2,048** (no cap). A MiMo prefill chunk's cost is mostly per chunk: 91 rows
  take 1.5 s, 512 rows 2.4-4.7 s, 1,622 rows 3.9 s. So the 512 cap doubled an arrival's TTFT (9.8 vs 4.4 s), and the
  decoding lanes stalled longer (9.9 s vs 4.5 s for the same few tokens).
- **The group gate (`pgate_`, B3) costs ~0.7 % of stage 0's time at 4 lanes and ~0 at 2 lanes.** Not changed.
- **`pipe_submit` now refuses a step once `pipe_stop` has begun, or while the pipe is paused.** It is the V4.1 fix, plus the
  paused state.
- **Gate follow-up: the lane choice now honours the reply budget.** The gate found a 15,993-token prompt with max_tokens
  2,000 sent to a 16,384 lane (391 tokens of room: "length") while lane 0 sat idle. Lanes that leave
  min(max_tokens, a quarter of the lane) for the reply now rank first. This is host-tested; no GPU run by the builder
  (below).

## What was built

| Piece | Where | What |
|---|---|---|
| Lanes at load | `Engine::mimo26_load` | `--parallel N` gives the forward N lanes (`Mimo26Options::lanes`) and the drafter N context rings (`add_lanes`). Lane 0 keeps `--ctx`; lanes 1..N-1 get `--slot-ctx` positions each (0 = 32,768, capped at `--ctx`). The forward's B1 refusal (their caches come out of the auto static tier) is the load-time refusal, with the numbers. `mimo26_load` no longer forces `parallel = 1`. |
| Serving state | `Mimo26Serve` (`include/ie/mimo26_engine.hpp`) | One record per lane under one mutex: busy, capacity, prompt and live ids, prefill cursor, sampler settings + rng + repetition window, outbox of committed ids with their sampler stats, draft events, finish, want_stop, lost, parked, and the drafter's per-lane state. Plus the protocol flags (`piping`, `pause`, `turn_busy`, `turn_waiters`, `stopping`) and the /health counters. |
| Request path | `mimo26_run_lanes` (`src/engine/mimo26_engine.cpp`) | Used at `--parallel > 1`; `mimo26_run_ids` (the serial path) at `--parallel 1`, unchanged. A request: (1) waits for an idle lane it fits; (2) takes the SERIAL TURN, chooses its lane and runs the prefix step on it (`Mimo26PrefixCache::prepare`), then releases the turn (the first chunk is submitted); (3) waits for the prompt, with a liveness probe between chunks; (4) takes the turn again for the prompt-end snapshot (`prompt_done`, #87) and the first token, then releases it (the first decode step is submitted); (5) only consumes its outbox: UTF-8 assembly, the server's token callback (a declined callback sets `want_stop`), and `ie_vitals` into the request's window. |
| The pipe's callback | `serve_done` | Runs on the last card's stage thread for every completed step. A prefill chunk: the drafter's context, the lane's live ids, then the next chunk (or the lane parks, or the prompt is complete). A decode step of T rows (`rows[0]` the anchor, `rows[1..]` drafts): the serial verify rule row by row (`Ds41Generator::sample_row` on a host copy; row r's sample judges draft r + 1), `serve_commit` (the emit rule: eos = "stop" and not committed, "length" at max_new), `pipe_rewind_lane` for rejected rows, the drafter's context for the kept rows, and the next step (drafted while the lane is alone) submitted from the callback. A non-finite logits row is a target fault (#74): the request fails and the lane is cleared. |
| The serial turn | `serve_take_turn` / `serve_release_turn` | `select_lane`, `state_spans`, `set_state`, `rewind` and the prefix cache are serial-only while the pipe runs, so a request PAUSES the pipe. `pause` is set, and every callback parks its lane instead of resubmitting. `pipe_pause()` waits for the steps in flight. The stage threads stay, and their thread-local oneDNN contexts with them. The holder works alone on the cards. The release prepares the parked lanes' next rows (drafting now, while the drafter is this thread's) and `pipe_resume()`s the pipe with them. The first request starts the pipe; a stage error stops it (`pipe_stop` fails every running lane), and the next release starts a new one. `IE_MIMO26_PIPE_PAUSE=0` stops and restarts the pipe at every turn, the 5b71003 behaviour, kept as a fallback and for the A/B below. One turn at a time. |
| Mixed prefill and decode | `serve_submit` | A prefill chunk is `IE_MIMO26_MIX_CHUNK` rows while another lane is busy or a request waits for the turn, else the forward's 2,048. **Default 2,048, i.e. no cap** (512 was measured worse, below). Alternation is the pipe's FIFO: a chunk (which runs alone as a group) and the decode lanes' group take turns, because each lane resubmits from its callback behind the others. |
| Lane choice | `serve_choose` → `mimo26_choose_lane` (`include/ie/mimo26_host_rules.hpp`, host-tested), `Mimo26PrefixCache::servable` | Among the idle lanes the prompt fits (ids < capacity), the lanes that leave its REPLY ROOM rank first: capacity - prompt >= min(max_tokens, capacity / 4), i.e. the whole budget, or a quarter of the lane when the budget is larger (gate follow-up, below). Only when no idle lane leaves it do the others compete, and the reply is then cut at the lane's room (the log line shows the reply cap). Within either set, in order: (1) the lane whose state serves the most of the prompt (its live ids or its prompt-end snapshot: the prefix cache's own reading, with the forward selected on that lane), if that is at least the cache's `min_tokens` (1,024; below it the prefill saved is not worth cutting or evicting a conversation, so an EMPTY lane comes first, then the shorter matches); (2) the smallest capacity (short prompts leave lane 0 to the long ones); (3) the least recently released. A prompt only lane 0 fits waits for lane 0. `prepare()` on the chosen lane then does what it always did: a host slot that serves clearly more is swapped in, with the lane's own conversation kept in a slot first (the LRU idle lane's eviction, `keep_live`); a branch keeps it too; the snapshot serves a discarded reply. P7 continuation rule: a prompt that re-reads fewer than `min_tokens` (1,024) positions of a lane's last prompt CONTINUES that conversation and cuts it without keeping it. So Dream sub-agents whose prompts share a long system prefix and differ only in the task are served as continuations of one another's lane. That is fast (the shared prefix is cached), but the old conversation is not kept. |
| Drafter rule (B4) | `serve_prepare_decode` | A lane drafts with DFlash (k = 7, min-p 0.7, as at `--parallel 1`) only while it is the ONLY lane decoding; with two or more decoding, every lane takes plain one-row steps. Every lane's drafter context follows every step, so a lane drafts again as soon as it is alone. B5 owns the row-budget policy. Prompt-lookup speculation is `--parallel 1` only (forced off at N > 1, logged). |
| Cancel | `want_stop` | Set by the request thread when the server's callback declines (client gone, shutdown), or by the prefill liveness probe. The callback ends the lane with "abort" at its next completion (a parked lane at the turn's release). The lane's live ids stay consistent with its caches, so an aborted conversation is still reusable. |
| /health | `Engine::serving_status_json`, `openai_server.cpp` | MiMo at `--parallel > 1` adds `lanes_active` (lanes owned by a request), `decoding` (lanes in the decode phase), `tokens` (ids committed, every lane, since load), `step_ms` (an exponential average of the decode steps, submit to callback), `rows_per_step`, `turns` (serial turns taken), `paused_ms` (the time the pipe spent paused for them: the turn's work and the resume, not the drain before it), and `gate_ms_1` / `gate_ms_n` (the group gate's cost, below). Every other arch and `--parallel 1` return "", and /health is unchanged. |
| Diagnostics | `serve_done`, the per-request log line | `IE_MIMO26_STEP_TRACE=1`: one stderr line per step (lane, prefill or decode, rows, ms from submit to callback, ms into the current pipe run, lanes decoding). The per-request line `[mimo26 lanes] lane N: ...` gives the done callback's time for the request, of it the verify loop ("sampling") and the drafter's context feed, and the drafter's draft passes ("drafting", in the callback or at a turn's release). The earlier "sampling" figure ran to the end of the callback and so included the draft pass and the next submit; fixed. |
| Vitals | the outbox | Each committed id carries the stats of the sample that produced it; the request thread adds them to its `VitalsWindow`, and draft passes per lane likewise. It is the same thread that reads the window for the stream, so there is no data race. `decode_tps` in the summary is the request's own tokens over its own decode wall, so it reflects the shared steps. |
| Forward additions (B3's files) | `Mimo26Forward::pipe_rewind_lane`, `pipe_error`, `pipe_pause`, `pipe_resume`, `pipe_paused`, `pipe_gate_ms`; `pipe_submit` | `pipe_rewind_lane(lane, n)`: the lane keeps positions [0, n), under the `pipe_reset_lane` rule (an idle lane, or the lane whose callback is running, from that thread); for a verify's rejected rows. `pipe_error()`: the first stage error while the pipe runs, so a request whose own step failed (its group runs no callback) notices from its wait. `pipe_pause` / `pipe_resume`: the stage threads idle on their condition variable, `piping()` is false, the serial API works on the active lane, and the positions move between the serial members and the lane table as at `pipe_stop` / `pipe_start`; `pipe_stop` works from either state. `pipe_gate_ms()`: stage 0's gated time (diagnostic). `pipe_submit` refuses once `pipe_stop` has set `pstop_` (a step queued then would find a stage thread gone and strand its lane half-stepped; the V4.1 fix) and while paused (`piping()` is read unlocked, so a submit racing `pipe_pause` could otherwise start a step while the serial API owns the cards). The serving code's own protocol (`stopping` and `pause` under the serve mutex) already kept both races out; the refusals make the forward's contract hold by itself. |
| Test | `tools/mimo26/parallel_serve_test.py` | Dream-shaped requests (a ~2.5k-token agent system prompt, 8 tools, a task; streamed), or the held-out transcripts (`--prompt-dir`: 03.txt, 04.txt, …). Modes: `concurrent` (b), `identity` (c, f), `ttft` (d), `memory` (e; RssAnon, VRAM, threads), `aba` (throughput; server-side /health rates, `--warm`), `turns` (the turn's cost to a decoding lane). Standard library only. |
| Not changed | `mimo26_run_ids`, the serial `forward()`, every other model | -- |

### Defaults picked

- `--slot-ctx` 0 = 32,768 positions per extra lane (the design's proposal; capped at `--ctx`). One lane at 32K costs ~1.34 GB
  over both cards, out of the static tier (B1).
- `IE_MIMO26_MIX_CHUNK` **2,048** = no cap (measured against 512: criterion (d) below). The design's v1 proposal was 512.
- `IE_MIMO26_PIPE_PAUSE` 1: the turn pauses the pipe; 0 stops and restarts it (the A/B below found no difference).
- The lane choice's reply reserve: min(max_tokens, capacity / 4) (gate follow-up). A quarter, not the whole budget: the
  server's default budget (16,384) and Dream's window-sized budgets exceed a small lane's capacity, so the whole budget
  would send every request to lane 0 and evict the lead's idle conversation there. With no idle lane leaving the reserve,
  the request takes today's choice rather than waiting (no queueing).
- Lane groups: the pipe's AUTO (B3's default: ceil(lanes in flight / cards) lanes per group). The identity gates run this
  served default with only the CPU expert leg off. `IE_MIMO26_ROWS_INVARIANT` and `IE_MIMO26_QSTAR_GROUP` are B3's knobs
  and stay theirs.
- Drafter: on at N > 1, used only by a lone decoding lane (above).
- `pgate_` unchanged (its cost below).
- Images: refused at `--parallel > 1` ("images are served at --parallel 1 only"): the pipe runs text positions only
  (B2: `pipe_submit` refuses image ids).

## Hazards honoured (B1/B2/B3 notes)

- Only the done callback's thread resubmits or rewinds a lane while the pipe runs. The request thread submits only from the
  turn's release, with every lane idle and the pipe paused or not yet started.
- `pipe_pause` and `pipe_stop` are never called from a callback: only from `serve_take_turn` on a request thread with the
  serve mutex released, and from teardown.
- After a stage error, `perr_` fails every submit until `pipe_stop`. The turn's `pipe_pause` returns the error, `pipe_stop`
  clears it, and every running request fails ("error: mimo_v2 pipe: ..."). Their lanes are cleared (`serve_clear_lane`:
  reset, drafter reset, `live_lost`). A lone failing lane (no callback) is seen through `pipe_error()` from the request
  thread's 1 s wait tick.
- The feature buffer is read (`add_context`) before the lane's next submit, in the same callback.
- The drafter is used only from the last card's stage thread (inside the callback), or from a request thread holding the
  turn with the pipe paused (its stage threads blocked on their condition variable, so no other thread touches the last
  card's queue).
- Teardown: `~Mimo26Bundle` sets `stopping`, then `pipe_stop` drains and joins the stage threads (running or paused), then
  every queue is drained before the frees. Every server in the runs below shut down with exit 0, from both states.

## Gates

**Run discipline.** Every run went through the main checkout's `scripts/ie-run-guarded --mem 220G --timeout 540`, in the
foreground, one at a time, sharing the lock with the B6a (DeepSeek-V4.1) and B7 (GLM) builders. From 08:24 to 09:12 the
coordinator gave B4 the GPU. Before each run: no Dream desktop, no engine process, cards at 26 / 34-36 MiB,
`journalctl -k --since "-10 min"` clean, the lock free (`p4b4_check.sh`). After each run: 30 s, then the same check. Every
check of this builder's runs was clean (a NOT CLEAR after a run was the next builder's job starting), and there was no CAT
error, GT reset or timeout. Server: `ie serve ~/models/MiMo-V2.6-Flash-RL --host 127.0.0.1 --port 1148x` plus the arguments
below (`--ctx 32768 --parallel 4 --slot-ctx 16384` unless noted), stopped with `POST /admin/shutdown`.
Logs: `results/mimo26/p4/b4/<run>_server.log`, `<run>_test.log` and `<run>_cpu.log` (untracked).
Scripts (copied beside the logs, with the HEAD binaries built from 1f0382e): `results/mimo26/p4/b4/scripts/p4b4c_runs.sh <run>`.

**Binaries (md5):**

| Binary | Commit | Runs |
|---|---|---|
| HEAD `ie` 2aa2ce8f…, `ie-mimo26-run` c2539f6e… | 1f0382e | a1_head_serve, a2_head_slot (and B3's R15 = the a3 reference) |
| `ie` 7792f6e9… | 78c1275 + pause/resume finished + diagnostics | t_turns_pause, t_turns_stop |
| `ie` cc2e1a38… | + the verify-loop timer fix (mix chunk default still 512) | t_aba2_*, c_aba4, c_aba4_heldout, c_identity, c_identity_dflash, c_ttft, c_ttft_2048 |
| `ie` 83d9244f… | + mix chunk default 2,048 | c_memory, c_concurrent, a1_branch_serve, a2_branch_slot |
| `ie` e9fe4ab8…, `ie-mimo26-run` 8c21b2eb… | + the `pipe_submit` refusals (4d0e648, the gate's PASS) | c_identity_final, c_aba4_heldout_warm, a3_branch_run |
| **`ie` 10bc6ca6…, `ie-mimo26-run` 72f43324…** | **+ the gate follow-up: the lane choice honours the reply budget** | host tests only (`mimo26_host_rules_test`); `ie-mimo26-run` changed only because `ie_core` now links `mimo26_choose_lane` |

The changes between these binaries do not touch the paths the earlier runs measured:
- The timer fix is a diagnostic.
- The mix chunk default only changes a prefill chunk's size while another lane is busy. The identity arms prefill alone,
  and the 2,048 TTFT arm set it by env.
- The refusals are two checks under a lock, which `--parallel 1` never reaches.
- The identity gate was re-run on the final binary.

| Criterion | Run(s) | Result |
|---|---|---|
| **(a)** `--parallel 1` == HEAD: `serve_test.py` and `slot_serve_test.py` unchanged incl. the SSE bytes (IE_VITALS.md's method); the `ie-mimo26-run` token md5 == HEAD | `a1_head_serve` / `a1_branch_serve` (`--ctx 32768 --parallel 1`): `p4b4_sse_capture.py` (3 cold requests: Dream sampling T 0.7 seed 1234 thinking on; the same with `ie_vitals`; greedy tool call), then `serve_test.py`. `a2_head_slot` / `a2_branch_slot` (`--ctx 65536`): `slot_serve_test.py --main-chars 100000 --sides 4 --side-chars 12000:32000`. `a3_branch_run`: B3's R14/R15 command (8 held-out prompts 00, 03-09 x 64, `--chunk 2048 --ctx 16384 --static 0 --ranking <chat ranking> --dflash 7 --dflash-minp 0.7`, `IE_DS41_CPU_MISS=0`) | **PASS.** SSE 1 and 3: same bytes (a6268600…, eb5ae0ab…, id/created masked). SSE 2 (`ie_vitals`): same bytes once the summary's three wall-clock fields are masked as well (86b5c34c… both). Raw, only `decode_tps` 17.3596 / 17.3177, `prefill_ms` 69.0 / 108.8 and `restore_ms` differ. `serve_test.py`: every line the same (chat, stream, tool_call, tool_turn, cancel, reuse, memory 362 MiB flat over 20 requests, props). `slot_serve_test.py`: every line the same (main 32,056 tokens, sides 4,208-9,663, turn 2 served 32,116 cached, PASS; TTFTs within 0.1 s). `ie-mimo26-run`: generated md5 **65501302fb19267820d991815210fb2d** == B3's R15, which ran exactly the HEAD binary c2539f6e. VRAM 31.21 / 28.05 GB and static/pinned 78/178, 62/194 identical. Decode ms/token 71.9 84.4 70.2 77.5 74.1 88.7 87.0 85.8 vs R15's 71.6 84.3 70.2 77.3 74.1 88.3 87.0 85.7. |
| **(b)** at `--parallel 4`, 4 concurrent Dream-shaped requests complete | `c_concurrent`: 4 streaming requests at once (T 0.7, thinking on, ~1,620-token prompts with 8 tools, 192 tokens each) | **PASS.** 4/4 (length, length, tool_calls, tool_calls). Steady window (all 4 decoding) 22.58 tok/s (644 deltas in 28.5 s); decode window 23.01 tok/s; 16.73 tok/s wall incl. the prefills. TTFTs 12.5 / 16.6 / 16.9 / 17.0 s: four 1.6k prompts arriving at once take turns. The 512-chunk default of the earlier `b_concurrent` run (847b1cdc…) gave 20.6-28.2 s. /health: lanes_active up to 4, step_ms 77-818, rows_per_step 1.0. |
| **(c)** a mid-batch cancel leaves the other lanes intact | `c_identity` (cc2e1a38) and `c_identity_final` (e9fe4ab8): `IE_DS41_CPU_MISS=0 IE_MIMO26_DFLASH=0`, the served default groups. 4 conversations (a different first system token each), 96 tokens, thinking off, greedy. Warm: each once, alone, into its own lane. Then solo one at a time; then all 4 at once; then all 4 with request 1 closed after 6 content chunks. Every arm is served from the lane's live state (cached = prompt - 1 in every arm), so the arms differ in the decode grouping only | **PASS, both runs.** Request 1 closed after 6 chunks; the 3/3 others are identical to solo, and the cancel arm is identical to the batch arm. The server answered request 1 again afterwards (served from its lane, 32 tokens). |
| **(d)** an arrival's TTFT while others decode ≤ solo TTFT + one chunk | `c_ttft` (chunk cap 512) and `c_ttft_2048` (`IE_MIMO26_MIX_CHUNK=2048`, now the default): TTFT of a 1,624-token prompt on a fresh server; then 3 requests decode (400 tokens) and a prompt sharing no prefix arrives | **PASS at both.** 512: solo 4.88 s, busy **9.83 s** (+4.95 s; allowance 6.2 s = the solo time of 2,048 rows). 2,048: solo 4.77 s, busy **4.42 s**. The decoding lanes made 6/6/7 tokens during the 9.9 s prefill at 512, and 4/3/5 during the 4.5 s one at 2,048. Note: the solo TTFT here is a fresh server's FIRST request, so it carries first-use costs. The gate's warm solo gave busy = solo + 0.52 s, still well inside the allowance. |
| **(e)** RssAnon and VRAM flat over 20 requests | `c_memory` (`IE_MIMO26_PROMPT_CACHE_GIB=0`: a kept conversation is P7's design, not a leak): 20 requests in waves of 4, 48 tokens each | **PASS.** 20/20 complete. RssAnon 335 → 813 (wave 1) → 817, 820, 822, 822 MiB (+9 MiB after wave 1). VRAM [30326, 31090] → [30345, 31109] → [30345, 31109] MiB. Threads 52 → 54 → 54. RssAnon still creeps a little per wave (+2-4 MiB here; the gate saw +14 MiB over 16 requests), within the 64 MiB tolerance. Whether it levels off over a long soak is unverified. |
| **(f)** each lane's text == its solo run (CPU leg off, greedy) | `c_identity`, `c_identity_final` (as (c)) | **PASS, both runs.** Batch == solo **4/4** (96 / 96 / 96 / 76 tokens; tool_calls, length, length, tool_calls). The two runs' texts are also identical to each other (solo 4/4, batch 4/4), across binaries cc2e1a38 and e9fe4ab8. |

**The drafter-on reference** (`c_identity_dflash`: the same run with the drafter on, compared with `c_identity`'s texts):
- The plain batch arm is **4/4 identical** to the drafter-off server's batch, and the cancel arm == the batch arm.
- Solo, a lone lane drafts, and 2 of 4 texts differ from the plain solo at a greedy choice: request 1 at character 208
  ("- File exists" / "- [ ] File exists") and request 2 at 210. This is the P5 behaviour of the drafted verify at
  `--parallel 1`: its multi-row steps move greedy near-ties.
- The within-server checks (f) batch == solo 2/4 and (c) cancel 2/3 therefore FAIL with the drafter on by design: that
  server's solo arm drafts and its batch arm does not.
- Unverified: these two positions were not confirmed as near-ties by a teacher-forced scan.
- Replay and identity checks belong at `--parallel 1`, or with the drafter off (design 2.6).

## Throughput

All rates are the server's own unless noted: /health `tokens` (committed ids) over the 0.5 s samples in which exactly 1 / N
lanes decode. After a `<tool_call>` the server buffers a reply, so SSE deltas undercount; the client-side figures are in
the logs. Greedy, thinking off, no tools, `IE_MIMO26_STEP_TRACE=1`. Solo = a lone lane, which drafts.

**Background load (every rate here is uncertain):**
- Most runs shared the CPUs with other work: another session's Dream test suite (pytest + Playwright chromium, 1-3 cores),
  a `beam.smp` process (bursts of 3-7 cores), and one B6a build at ~07:38 (`t_aba2_stop2`, discounted).
- `<run>_cpu.log` samples the load every 5 s.
- The solo drafted rates varied ±25 % run to run; the concurrent arms were stable.

### Pipe pause / resume vs stop / restart at every turn (`IE_MIMO26_PIPE_PAUSE=0`), `--parallel 2`

| Run | Turn mode | Solo A1 | 2 lanes (all decoding) | Solo A2 | Turns |
|---|---|---|---|---|---|
| `t_aba2_pause` (07:26) | pause | 22.21 | **22.39** (22.0 s) | 20.55 | 8 |
| `t_aba2_stop` (07:30) | stop | 15.65 | **22.48** (22.0 s) | 21.31 | 8 |
| `t_aba2_stop2` (07:37, B6a building: discounted) | stop | 23.94 | 22.64 (19.5 s) | 16.67 | 8 |
| `t_aba2_pause2` (07:44) | pause | 23.60 | **23.45** (21.0 s) | 20.51 | 8 |

| Run | Turn mode | Long reply (512 tokens) with 20 short requests = 40 turns inside it | Paused per turn | First decode step after a turn (1 row, 2 lanes) / later steps |
|---|---|---|---|---|
| `t_turns_pause` | pause | 15.17 tok/s | 9.2 ms | median 92.8 / 79.7 ms |
| `t_turns_stop` | stop | 15.06 tok/s | 7.2 ms | median 90.9 / 79.6 ms |

**Read:** no measurable difference.
- The ~300 ms first steps after a restart are not in the per-step trace. Likely the earlier /health readings were decode
  steps queued behind a prefill chunk: the traces show 1-row decode steps of 2.4 s (`t_aba2_pause`) and 2.7 s (`c_aba4`)
  submitted while a chunk ran.
- Each stop run had one slow solo (15.65, 16.67) and neither pause run had one. Tracing each row count's first use on
  fresh stage threads did not show a consistent cost (in `t_turns_stop` the first uses were faster than the repeats), so
  this is read as noise (unverified).
- Pause is kept: the same speed, no thread join and create per turn (two per request), and the stage threads keep their
  per-thread state.
- Threads after 8 turns: 52 in both modes.

### 4 lanes

| Run | Prompts | Solo A1 | 4 lanes (all decoding) | Solo A2 | vs drafted solo |
|---|---|---|---|---|---|
| `c_aba4` | synthetic Dream-shaped (~930 tokens, no tools), 256 tokens | 17.97 | **24.54** (31.5 s) | 19.89 | x1.30 |
| `c_aba4_heldout` | held-out 03-06 (5.4-6.0k tokens) arriving cold, 192 tokens, 512 cap | 18.18 | 28.34 (5.0 s) | 23.14 | x1.37 |
| `c_aba4_heldout_warm` | held-out 03-06, each prefilled into its lane first (`--warm`), 256 tokens | 24.15 | **29.54** (5.0 s); 3 lanes 22.75 (27.0 s) | 24.25 | x1.22 |
| B3 lanes test (R17 final, R8) | held-out 03-06, the lanes-test tool, plain A-B-A solos | ~16 (plain) | 28.85 / 30.06 (138.3 / 132.7 ms per lane step) | ~16 | x1.85 vs plain solo |

- On the held-out prompts the served 4-lane rate matches B3's lanes test.
  - The all-4 window is short because request 3 (06.txt) ends in a tool call after ~40 tokens.
  - The per-step trace is the sturdier reading: a 1-row lane step with 4 lanes decoding has a median of **135.9 ms** (137
    steps) = 29.4 tok/s, against B3's 138.3 / 132.7 ms.
- The synthetic Dream-shaped prompts decode slower both solo (18-20 vs 24 tok/s) and at 4 lanes. Likely the chat expert
  ranking was profiled on real transcripts and fits their routing better. Unverified: the hit rates were not compared.
- In `c_aba4`, the 1-row step per lane had a median of 63 ms at 1 lane (plain), 126 ms at 3 lanes and 157.5 ms at 4 lanes
  (2 groups of 2).

### Costs inside a step

- **The done callback per decode step:**
  - greedy at 4 lanes: 0.72 ms (verify loop 0.13, drafter-context feed 0.57);
  - T 0.7 (`c_concurrent`): 1.7 ms (verify loop 0.9, feed 0.65);
  - a lone drafted lane: 11.3 ms, of it 10.2 ms the DFlash draft pass, which runs on the last card inside the callback,
    as at `--parallel 1`.
  - The callbacks run on the last card's stage thread, so at 2 lanes (2 x ~1 ms per ~87 ms cycle) they hold that card
    ~2-3 %.
- **The group gate (`pgate_`):** stage 0 sat with a lane waiting and a group free for the following totals (/health
  `gate_ms_n`):
  - `c_aba4`: 268 ms over the run, ~200 ms of it during the ~30 s in which all 4 decode (~0.7 %, ~0.5 ms per 2-lane group
    step);
  - `c_concurrent`: 332 ms; `c_memory`: 191 ms; `c_aba4_heldout_warm`: 174 ms.
  - Behind one-lane groups (`gate_ms_1`): 0-1.9 ms per run.
  - The gate keeps the lanes a finished group resubmits together. Dropping it for one-lane groups saves nothing measurable
    and would change the grouping, so it is unchanged.
- **Serial turns:** 7-9 ms of paused pipe per turn (`paused_ms`: the turn's work and the resume) when the prefix step is a
  live-state reuse. Before it, the drain waits for the steps in flight (at most one step each), and the GPU keeps working
  through them. A turn that restores a host slot or takes a snapshot costs its copies (P7).

### What this means for the owner's case (Dream's lead plus sub-agents)

- At `--parallel 2` the two requests together produce about what one produces alone with the drafter (22-23 vs 20-24
  tok/s), each at half the latency (11-12 tok/s per lane).
- At `--parallel 4` on real transcripts, 4 lanes make ~29.5 tok/s against 24 solo (x1.22), ~7.4 tok/s each.
- Speculation and batching spend the same resource (rows through the expert union). B5's row budget (drafts for every lane
  within the 8-row group) is what would lift the aggregate above the solo drafted rate.
- An arriving 1.6k prompt now waits ~4.4 s (its own prefill), and the decoding lanes pause for about that long.

## Findings

- **(Gate follow-up) The lane choice ignored the reply budget.**
  - `serve_choose` took the smallest idle lane whose capacity exceeded the prompt, and never looked at max_tokens.
  - On `--ctx 32768 --parallel 4 --slot-ctx 16384`, the gate's 15,993-token prompt with max_tokens 2,000 went to lane 1
    (16,384 positions: 391 tokens of room, finish "length") while lane 0 sat idle. A thinking reply could be cut before
    its answer.
  - Fix: lanes that leave min(max_tokens, capacity / 4) for the reply rank first, then today's order within them.
  - With no such lane idle, the request takes today's choice rather than waiting. Waiting for lane 0 could stall a
    sub-agent behind the lead's whole turn.
  - The rule is a pure function (`mimo26_choose_lane`) with 16 host checks, including the gate's case alone and
    repeated (its own lane holding the conversation). The old rule fails the first check.
  - Trade-offs:
    - A conversation on a small lane that grows past three quarters of it moves to an idle lane 0 and re-prefills there
      once, rather than cutting its replies.
    - A fresh request whose reserve only lane 0 leaves takes lane 0 even while lane 0 holds another idle conversation.
      P7 keeps that conversation in a host slot when there is room.
  - GPU-checked by the gate (re-check of ba287bd, one guarded run on a fresh `--ctx 32768 --parallel 4 --slot-ctx 16384`
    server, host slots off): the 15,993-token prompt with max_tokens 2000 now takes lane 0 (reply cap 2000) and returns 2000
    tokens (finish `length` at the full budget, after a full 8-chunk re-prefill: TTFT 36.5 s); a fresh 2,177-token prompt
    at the server default budget (16,384) still lands on an empty small lane (reply cap 14,207), not on lane 0. The gate's
    random property checks (2,000,000 lane sets) found 0 mismatches against the reference rule, and every departure from
    the old rule leaves strictly more reply room.
- **The ~300 ms restart cost did not reproduce** (above). The earlier reading came from the /health `step_ms` EMA, which
  also averages the decode steps that wait behind a prefill chunk (likely the source). A per-step trace
  (`IE_MIMO26_STEP_TRACE=1`) separates them.
- **A MiMo prefill chunk costs mostly per chunk, not per row.**
  - Submit to callback, with 0-3 lanes decoding (step trace, `c_ttft*`): 88-91 rows 1.5-1.6 s, 512 rows 2.4-4.7 s, and
    1,622-1,627 rows 3.9 s. These times include any wait behind the decoding groups.
  - Likely each chunk streams nearly every routed expert whatever its rows (inferred, not profiled).
  - So the 512-row mixing cap cost the arrival 2.2x the prefill time and stalled the decoding lanes longer, and three
    concurrent arrivals' own prefills slowed from 9.3-9.8 s to 12.8-17.6 s TTFT.
  - The default is now the full chunk. For a very long arrival (e.g. 30k tokens = 15 chunks) the decoding lanes crawl for
    the whole prefill either way. Estimate: 2,048 should still finish it in about half the time, but that was not
    measured.
- **A prompt's KV bytes depend on how it was prefilled** (the earlier builder's finding, 05:36-05:39).
  - Arms whose prompts were prefilled with different chunking (one 1,620-row chunk alone vs 512-row chunks under the
    mixing cap) diverged at a greedy near-tie in 2 of 4 texts.
  - With `--parallel > 1`, a reply can therefore differ from the `--parallel 1` reply of the same prompt when the prompt
    ran in different chunks. The prefill GEMMs follow M (oneDNN kernel choice, and the XMX prefill attention's tile
    shape).
  - It is the same class of near-tie divergence as lookup and the drafter. The decode grouping itself is byte-identical.
- **The earlier serve script's `@PID@` was the `timeout` wrapper**, not the server. `pgrep -f … | head -1` matches the
  wrapper's command line first, so a RssAnon or thread reading through it read the wrapper. No earlier pass criterion used
  it (criterion (e) had not been run). Fixed in `scripts/p4b4c_serve.sh` (the process named `ie` with that port). The
  `serve_test.py` memory line above (362 MiB flat) is the server's.
- **Lane choice under `min_tokens`:** a re-sent ~930-token prompt goes to an empty lane rather than its own conversation's
  lane (the 9701fc2 rule: a match under 1,024 does not cut a conversation while an empty lane exists). It costs that
  request one prefill (`t_aba2_*`: concurrent request 0, 0 cached) and keeps the old conversation for a continuation.
  Intended.
- **Continuations under the P7 rule:** prompts that share the long system + tools prefix and differ only in the task are
  served as continuations of one another's lane (`c_ttft`: request 3, 1,568 of 1,617 cached). Fast, but the earlier
  conversation is cut, not kept.

## Risks

- The held-out 4-lane server-side rate rests on a 5 s all-4 window (~150 tokens). The per-step trace (137 lane steps)
  agrees, but it is still a short run.
- Throughput numbers carry the background load above.
- **Two long conversations can take turns evicting each other from lane 0** (from the gate's code reading, not run --
  unverified): one conversation past three quarters of a small lane and another too big for a small lane. With host slots
  off or full, each turn re-prefills fully (~36 s per 16k tokens); with host slots on, a keep plus a restore per turn. The
  old rule left the smaller one on its own lane (no thrashing, but a cut reply).
- The no-queue fallback can still cut a reply at a small lane's room when lane 0 is busy (host-tested, not run on the
  GPU); waiting for lane 0 below some room floor would be an owner decision.
- The mixing chunk default (2,048) was measured with ~1.6k-token arrivals and 3 decoding lanes only.
- The `pipe_submit` refusals have no automated test: the pipe needs initialised cards, and the race window is not
  reachable from one thread. The serving protocol already excluded both races, and the identity and held-out runs on
  e9fe4ab8 exercised the path.
- **With the drafter on (the default), a greedy reply can depend on other traffic.**
  - A lone decoding lane drafts, and the verify's multi-row steps can move a greedy near-tie. So the same request can
    produce a different text when it runs alone than when other lanes decode alongside it (`c_identity_dflash`: 2 of 4
    texts).
  - Unverified: these are assumed to be near-ties (the P5 finding at `--parallel 1`) but were not verified by a
    teacher-forced scan.
  - Replay and identity checks belong at `--parallel 1`, or with the drafter off.
- **/health can wait behind a serial turn** (from code reading; not measured).
  - `serving_status_json` takes the serve mutex. The turn holder keeps it while it chooses the lane, runs the prefix step
    and takes the prompt-end snapshot.
  - A host-slot keep or restore copies ~0.25 s / ~0.07 s per 8K tokens (P7 section 7), so a long conversation's keep can
    hold /health for a second or more.
  - The pipe's callbacks also take the mutex, for ~1-11 ms each.
- The reply reserve (a quarter of the lane) is a heuristic (the fraction is unverified). A reply longer than that can still
  be cut on a small lane when lane 0 is busy, or when the conversation grew on its lane before the reply.
- With the pipe paused, a turn's serial work runs while the stage threads exist but are blocked. Anything later added to a
  turn must not wake them (only `pipe_submit` does, and it refuses while paused).

## Not done

- A GPU check of the lane-choice fix (`ie` 10bc6ca6…). The builder ran host tests only; the coordinator's GPU window for it
  was pending. The gate's case is one guarded server run: a 15,993-token prompt with max_tokens 2,000 at `--parallel 4
  --slot-ctx 16384` should land on lane 0 (the `[mimo26 lanes]` line shows `reply cap 2000`) and not finish "length" at
  391.
- A teacher-forced scan of the drafter-on divergences.
- A 1,024-row mixing chunk, and arrivals much longer than 1.6k tokens.
- `ie-mimo26-run` was not re-run on the HEAD binary in this session. B3's R15 on the same binary (c2539f6e) is the
  reference md5.
- `c_concurrent` (b), `c_memory` (e) and the a1 / a2 runs ran on 83d9244f, not the final e9fe4ab8 (the difference is the two
  refusals).
