# P4 B6b — DeepSeek-V4.1-Flash `ie serve --parallel N`: several requests decode together on the lanes (2026-09-26)

Design: `~/ds41_work/p60/p4_b6_b8_design.md` (the B6 part, "B6c, serving"). Template: MiMo B4 (`p4-b4-mimo-serve` 4d0e648,
`docs/mimo26/P4_B4_SERVE.md`). B6a: `docs/deepseek41/P4_B6A_LANES.md` (lanes + the two-card lane pipe). Branch
`p4-b6b-ds41-serve` from `p4-b6a-ds41-lanes` d0e39c7.

Owner's constraints: additive; `--parallel 1` (the default) is the engine the server always had, byte for byte; every other
model's serving is untouched.

## Answer first

- **Every B6b criterion passes on the GPU** (table in "Gates"). Sixteen guarded jobs ran 13:05-14:00: every server exited 0,
  every postcheck was clear, and the kernel log shows 0 CAT / reset / timeout lines. Branch binary `ie` 23902a0f... (e2738f7);
  HEAD `ie` e266c3ee... (d0e39c7, B6a's).
  - (a) `--parallel 1` == HEAD:
    - SSE bytes identical on all 3 requests (T 0.7 thinking; a tool call; greedy thinking).
    - Identity texts identical 3/3 (solo and batch).
    - The Phase 57 vision e2e test passes 4/4 on both, with the same answers.
  - (b) 2 and 3 concurrent Dream-shaped requests (8 tools, thinking on, T 0.7) complete. The tool calls parse, and
    reasoning_content comes per lane.
  - (c) A request cancelled mid-batch leaves the others identical to their solo runs (2/2, lookup on and off).
  - (d) An arrival's TTFT while 2 lanes decode: 7.88 s against 7.08 s solo (+0.80 s; one chunk ~8.3 s).
  - (e) Over 20 requests: RssAnon +10 MiB after wave 1 (flat from wave 6), VRAM +10 MiB per card, threads flat at 119.
    - For comparison, the `--parallel 1` HEAD run grew 37 -> 300 threads over its 16 requests.
    - The serial worker is why the lanes' count stays flat.
  - (f) CPU leg off, greedy, 3 lanes: batch == solo 3/3 with prompt lookup on and off.
    - With lookup on, the verify steps go through the pipe: 7 passes, 47 drafts accepted in the copy task's 96 tokens.
    - The lanes' solo texts also equal `--parallel 1`'s 3/3, and lookup on == off.
- **Throughput.** Measured with the server's own count: /health `tokens` while exactly 1 / 2 / 1 lanes decode, in the served
  configuration, lookup on, greedy.
  - Solo 14.81 and 13.99 tok/s; **2 lanes 24.92 tok/s over a 19 s window = x1.73** the solo mean (per lane 12.46).
  - Median one-row step, submit to callback: 67.8 ms alone, 76.8 ms per lane with 2 decoding.
  - A first A-B-A whose second prompt was a copy task gave x1.72 over a 6.5 s window.
  - 3 lanes (T 0.7): 118.7 ms per lane step = ~25 tok/s aggregate, i.e. no more than 2 lanes. With two cards, a third lane adds
    capacity, not speed (unverified: one run).
- **Also measured:**
  - Images at `--parallel 2`: an image request takes its serial turn while another lane decodes. The answer is correct, and the
    follow-up is served from its lane's live state.
  - The reply-room lane choice: a 3,748-token prompt with max_tokens 700 took lane 0 and produced 700 tokens, where a 4,096-token
    lane would have cut it at 348.
  - A lone 7,572-token prefill runs in its turn, is interrupted at a chunk end by an arrival, and finishes through the lane pipe,
    with the same text as at `--parallel 1`.
  - The disconnect probe: 6/6 streams cut mid-generation and freed in 0.3 s, 0.000 cores idle, threads flat.
- **Fixed before the first GPU run** (review and the B6a gate):
  - Restores are bounded by the destination lane's capacity.
  - The lanes of a failed pipe are reset before reuse.
  - A cold prompt's lane is reset for the pipe (`pipe_submit` would have refused it).
  - The reply-room lane choice (MiMo B4's gate finding, applied here).
  - Details are in "Hazards honoured".

## What was built

| Piece | Where | What |
|---|---|---|
| Lanes at load | `Engine::ds41_load` | `--parallel N` (1-4) gives the forward N lanes (`ResidentOptions::lanes`, B6a). Lane 0 keeps `--ctx`; lanes 1..N-1 hold `--slot-ctx` positions each (0 = 32,768, capped at `--ctx`); their state comes out of the static expert tier (B6a's refusal, with the numbers, is the load-time refusal). `opts_.parallel = 1` stays for N = 1 only. Refused at N > 1, at load, with a message: DSpark (`IE_DS41_SPEC=1`), `IE_DS41_PROFILE_OUT`, `IE_DS41_DUMP_ROUTING`, expert parallel (`IE_DS41_EP`), one card. |
| Serving state | `Ds41Serve` (`include/ie/ds41_engine.hpp`) | One record per lane under one mutex: busy, capacity, phase (prefill / prompt ready / decode / done), the request's ids, the prefill plan and its cursor, positions, sampler settings + rng + repetition window, the lane's prompt-lookup index, the committed ids (outbox), finish, want_stop, lost, parked. Plus the protocol flags (`piping`, `pause`, `turn_busy`, `turn_waiters`, `stopping`), the /health counters, and the serial worker. |
| Request path | `ds41_run_lanes` (`src/engine/ds41_engine.cpp`) | Used at `--parallel > 1`; `ds41_run_ids` (the serial path) at `--parallel 1`, unchanged. A request: (1) waits for an idle lane it fits (probing the client every second); (2) takes the SERIAL TURN, chooses its lane and runs the prefix step (`prefix_prepare` on that lane) and the prefill plan; the prefill runs in the turn when every other lane is idle (below), when image positions are left, or when nothing precedes the prompt's end, and otherwise through the lane pipe; (3) waits for the prompt, probing the client between pieces; (4) takes the turn again for the prompt's end: the checkpoint before the think tag, the tag row, the prompt-end checkpoint, the disk entry of the system prefix, the first token; (5) only consumes its outbox: UTF-8 assembly with the serial emitter's hold-back, the server's callback (a declined callback sets `want_stop`). |
| The pipe's callback | `serve_done` | Runs on the last card's stage thread for every completed step. A prefill piece: the lane's position, then its next piece (a plan chunk, or a one-row step up to the planned end), or the prompt is ready (its last row kept for the turn). A decode step of T rows (`rows[0]` the anchor, `rows[1..]` lookup drafts): the serial loop's verify row by row (`Ds41Generator::sample_row` on a host copy; row r's sample judges draft r + 1), `serve_commit` (the emit rule), `pipe_rollback` to the rows kept (every verify step, as the serial loop), the lookup draft and the next step submitted from the callback. |
| Sampler and emit rules | `serve_commit`, `Ds41Generator::sample_row` / `lookup_policy` / `repeating` | `run()`'s: eos = "stop" and not committed; the repetition stop (counted, not emitted); "length" at the budget; greedy / T / top-k / top-p / min-p / the repetition penalty over the prompt tail + output; seed 0 = the serial path's constant. Prompt lookup per lane: the index over the prompt, a copy of >= 12 tokens, up to 7 drafts, capped at the tokens left; eos in a verify drops its row. |
| The serial turn | `serve_take_turn` / `serve_release_turn` | `select_lane`, the prefix-cache calls, `forward`, `forward_pipelined` and `rollback_to` are serial-only while the pipe runs, so a request PAUSES the pipe: `pause` is set and every callback parks its lane instead of resubmitting; `pipe_pause()` waits for the steps in flight; the stage threads stay. The release resumes the pipe with every parked lane's step -- unless another request waits for the turn: the turn is then HANDED OVER with the pipe still paused (no drain between two turns; `handovers` in /health). The first request starts the pipe; a stage error stops it (every running request fails, its lane is forgotten) and the next release starts a new one. |
| Pipe pause / resume | `Ds41Forward::pipe_pause`, `pipe_resume`, `pipe_paused`, `pipe_error` | Kept from MiMo B4, and for V4.1 required: a stage thread runs the engram gather's OpenMP region, and a thread that ran one and exits leaves its OpenMP team behind (the adca570 leak: 94k threads). Stopping and restarting the pipe at every turn would leak a team per turn. While paused, `piping()` is false and the serial API works on the active lane; `pipe_submit` is refused (under the lock); `pipe_stop` works from either state; the destructor and `free_resident` stop a paused pipe too. |
| The serial worker | `serve_serial`, `Ds41Serve::worker` | The turn's device work (the prefix step, a serial prefill, the prompt's end, clearing a lost lane) runs on ONE persistent thread, not on the HTTP request threads: each request thread that ran a forward would otherwise keep its own OpenMP team (the HTTP pool has `parallel + max_queue + 4` threads). The turn holder posts the work and waits, probing the client every second; a throw in the work comes back as the request's error. |
| Lone prefill | `serve_serial_prefill`, `Ds41Forward::forward_pipelined(..., stop, n_done)` | A prompt that arrives while every other lane is idle prefills in its turn with the cards pipelined over its chunks, as `--parallel 1` does (the lane pipe runs one step per lane at a time, so a lone prompt's chunks would not overlap across the cards: ~155-165 instead of ~310 tok/s, an estimate from docs/83). `forward_pipelined` gained an optional `stop` asked before every chunk after the first: when another request waits for the turn (or the client left), the first stage starts no more chunks, the started ones finish, and the rest of the plan goes through the lane pipe. |
| Prefill plan | `Ds41Generator::plan_prefill` | `run()`'s planned branch moved into a function both paths call (the chunk ends at the first and the last user message, the think-tag end, the disk-entry end), so a lane prefills a prompt with exactly the launches `--parallel 1` uses. `tests/unit/ds41_prefill_plan_test.cpp` checks it against a verbatim copy of the old inline plan on 333k random prompts, and that every plan is admissible step by step in the lane pipe. |
| Mixed prefill and decode | `IE_DS41_MIX_CHUNK` | A prefill chunk's rows while another lane is busy or a request waits for the turn. **Default = the forward's chunk (2,048, no cap)**: MiMo B4 measured a chunk's cost as mostly per chunk (a 512-row cap doubled an arrival's TTFT and stalled the decoding lanes longer). A capped plan writes no disk entry (other chunks compute other bits; `deepseek41_numerics_inputs.txt` updated). |
| Lane choice | `serve_choose`, `ds41_choose_lane` (`include/ie/ds41_serve_rules.hpp`), `Ds41Forward::prefix_servable` | A pure function with host checks (`tests/unit/ds41_serve_rules_test.cpp`). Among the idle lanes the prompt fits (ids < capacity): (0) the lanes that leave the reply room first -- capacity - prompt >= min(max_tokens, capacity / 4) (MiMo B4 gate follow-up ba287bd: a 15,993-token prompt with max_tokens 2,000 went to a 16,384 lane and was cut at 391 tokens while lane 0 sat idle; a quarter when the budget is larger, because the server's default 16,384 and Dream's window - prompt budgets would otherwise pin every request to lane 0; with no such lane idle, the old order and no queueing); then (1) the lane whose OWN state (live position or a complete checkpoint, `prefix_prepare`'s rule, read only) serves the most of the prompt, when that is at least `min_slot_tokens` (1,024; below it an EMPTY lane comes first) -- or, V4.1's addition, when the match reaches the end of the lane's last prompt (a follow-up turn or the same prompt again: taking the lane loses nothing, so a short conversation's follow-up, e.g. an image chat, stays on its lane instead of re-prefilling on an empty one); (2) the smallest capacity (short prompts leave lane 0 to the long ones); (3) the least recently released. Host slots and disk entries are shared: `prefix_prepare` on the chosen lane swaps in a host slot or a disk entry when it serves clearly more, keeping the lane's own conversation in a host slot first (the eviction). The reply budget on the lane is `ds41_reply_budget` (`run()`'s rule in 64-bit: an unlimited max_tokens cannot wrap). |
| Images | `ds41_chat` -> `ds41_run_lanes(..., provider)` | The lane pipe runs text positions only (B6a refuses negative ids), so a prompt with image positions left after the prefix step prefills entirely in its serial turn (the provider set on its lane, then cleared); its decode goes through the pipe like any other. The other lanes wait for that turn. |
| Cancel | `want_stop` | Set by the request thread when the server's callback declines (client gone, shutdown), by the prefill liveness probe, or while waiting for a lane. The lane ends with "abort" at its next completion (a parked lane at the turn's release); its state stays consistent, so the conversation is reusable. |
| /health | `Engine::serving_status_json`, `openai_server.cpp` | V4.1 at `--parallel > 1` adds `lanes_active`, `decoding`, `tokens` (ids committed, every lane, since load), `step_ms` (an exponential average of the decode steps, submit to callback), `rows_per_step`, `turns`, `paused_ms` (the turns' work, the pipe paused), `handovers`. Every other arch and `--parallel 1`: "" and /health unchanged. `/props` `prompt_cache_slots` = lanes + 1 with host slots (N = 1: unchanged). |
| Diagnostics | `IE_DS41_STEP_TRACE=1`, the per-request line | One stderr line per step (lane, prefill / decode, rows, position, ms submit to callback, lanes decoding). The per-request line `[ds41 lanes] lane N: ...` gives the cache source, the prefill's chunks in the turn and through the pipe, the lookup passes, the callbacks' time, the turns and handovers. |
| Test | `tools/ds41_parallel_serve_test.py` | MiMo B4's tool for V4.1: `concurrent` (b; tool calls parse, reasoning per lane), `identity` (c, f; two copy tasks so lookup verifies; `--save` / `--ref` across servers), `evict` (a host slot restored on another lane), `ttft` (d), `memory` (e; RssAnon, threads, the server's own VRAM from procfs fdinfo), `aba` (throughput; `--tasks`), `room` (the reply-room lane choice), `interrupt` (a lone prefill handed to the pipe mid-plan), `vision`, `sse` (a). Standard library only; nothing in it talks to the GPU. Host tests: `tests/unit/ds41_prefill_plan_test.cpp`, `tests/unit/ds41_serve_rules_test.cpp` (29 checks). |
| Not changed | `ds41_run_ids`, the serial forward path, every other model, `engine.cpp` | -- |

### Defaults picked

- `--slot-ctx` 0 = 32,768 positions per extra lane (the design's proposal; capped at `--ctx`).
- `IE_DS41_MIX_CHUNK` = the forward's chunk (no cap), after MiMo B4's measurement (not re-measured for V4.1).
- Lone prefill in the turn with `forward_pipelined`, interruptible at a chunk end (`IE_DS41_PIPE_PREFILL=0` keeps `run()`'s kill
  switch).
- Prompt lookup: on at N > 1 exactly as at N = 1 (`IE_DS41_LOOKUP=0` off), per lane, verify steps through the pipe.
- The turn handover: on (no knob).

## Hazards honoured (B6a and MiMo B2-B4 notes)

- Only the done callback's thread resubmits, rolls back or ends a running lane; a request thread submits only from the turn's
  release, with the pipe paused or not yet started.
- A lane that nothing served (reused 0, or the cache off) is reset in the prefix turn (`reset_state` on it): the serial `forward()`
  resets at a pos0 = 0 step by itself, but the lane pipe admits a step only at the lane's n_pos, a pos0 = 0 prefill included, so a
  cold prompt on a lane that still held an old conversation would be refused (found in review before the first GPU run; B6a's
  lanes test resets with `pipe_reset_lane` for the same reason).
- `pipe_pause` and `pipe_stop` are never called from a callback: only from `serve_take_turn` (with the serve mutex released) and
  from teardown.
- Lock order: the serve mutex, then the pipe's (`pipe_submit`, `pipe_rollback`, `pipe_resume` under the serve mutex); the pipe
  never calls the serve code with its own mutex held. The serial worker's mutex is taken only with the serve mutex released.
- After a stage error, every running request fails and its lane is forgotten (`serve_clear_lane` in a turn). A lone failing lane
  (no callback) is seen through `pipe_error()` from the request threads' 1 s waits.
- **A stage error leaves its lane half-stepped** (the B6a gate): an error on card 1 leaves card 0's latents and the lane's
  look-back advanced while n_pos is not committed, and the pipe does not reset the lane (the serial `forward()` drops the live
  state on an error; the pipe's error path only releases the claim). Every lane that can have had a step in flight is a running
  one (a done lane never resubmits), so `serve_pipe_failed` marks every running lane lost, and each is reset
  (`reset_state` on that lane) by its own request, in a turn, before the lane is released for reuse. A lost lane stays busy until
  then, so no other request can take it half-stepped. The failed pipe's stage threads are joined and the next release starts a
  new pipe (their OpenMP teams are left behind: errors only).
- **Host-slot and disk restores are bounded by the destination lane's capacity** (the B6a gate): lanes differ in capacity (lane 0
  `--ctx`, the others `--slot-ctx`) and a host slot kept from a bigger lane could hold more positions than a smaller lane's latent
  and index-key buffers. `prefix_prepare` now only selects a slot checkpoint or a disk entry the active lane can hold
  (`pc_ckpt_fits`: its position and every layer's latent count within the lane), and `pc_load_slot` refuses one that does not fit
  before any copy (as `pc_load_disk` does). On the serving path the guard cannot trigger: a request only takes a lane that holds
  its prompt (T < capacity), and a restore serves at most T - 1 positions, whose latent counts are at most T - 1. It closes the
  out-of-bounds device write for any other caller of `prefix_prepare` (the check is two comparisons per candidate; not
  host-testable without a forward; unverified: the claim that no other caller exists beyond the tools).
- Teardown: `~Ds41Bundle` sets `stopping`, joins the idle serial worker, `pipe_stop` drains and joins the stage threads (running
  or paused), every queue is drained, then the frees.

## Gates

**Run discipline.**
- Every job ran through the main checkout's `scripts/ie-run-guarded --mem 240G --timeout 540`, in the foreground, one at a time,
  in the coordinator's window (12:59-14:10).
- Before each job (`check.sh`): no Dream desktop, no engine process, no compiler or linker, both cards <= 40 MiB (26 / 34),
  `journalctl -k` clean for 10 min, the GPU lock free. After each job: 30 s, then the same check.
- All 16 postchecks were CLEAR; the kernel log had 0 matches 12:55-14:00.
- Servers: `ie serve ~/models/DeepSeek-V4.1-Flash --host 127.0.0.1 --port 1149x` with the arguments below, stopped with
  `POST /admin/shutdown` (every exit 0).
- Each server's prompt-cache disk store was a fresh scratch directory (`IE_DS41_PROMPT_CACHE_DIR`), never the owner's `~/.cache`.
- A procfs-only sampler logged MemAvailable, RssAnon, threads and the server's own VRAM (fdinfo) every 5 s.

Logs: `results/deepseek41/p4/b6b/<run>_{server,test,samples,cpu}.log` (untracked; a copy is in the main checkout's
`results/deepseek41/p4/b6b/`). Scripts: `results/deepseek41/p4/b6b/scripts/` (`runs.sh <run>`, `serve.sh`, `check.sh`,
`cmp_a.sh`, `summary.sh`). The test client: `tools/ds41_parallel_serve_test.py`.

| Criterion | Run(s) (config) | Result |
|---|---|---|
| **(a)** `--parallel 1` == HEAD | `a1_head` / `a1_branch` (`--ctx 32768 --parallel 1`, `IE_DS41_CPU_MISS=0`): `sse` (3 cold streaming requests: T 0.7 seed 1234 thinking on; greedy thinking off with a get_weather tool; greedy thinking on), then `identity --n 3` (warm / solo / batch / cancel). `a2_vision_head` / `a2_vision_branch`: `~/ds41_work/p57/vision_e2e.py` | **PASS.** SSE md5 (id / created masked) 7dbe219a..., c9797011..., 4c808aae...: identical. Identity texts: solo 3/3 and batch 3/3 identical; the tool's ok / FAIL lines identical. Vision e2e: 4/4 PASS on both, with the answers, reasoning and usage lines identical (MACHX 4721; the cached follow-up names the blue circle; a different image shares no cached image tokens; the FAIL rows 00 03 06 09). Decode timing matches too (e.g. 107.1 vs 107.3 ms per one-row step in the lookup accounting). |
| **(b)** concurrent Dream-shaped requests with tools complete | `b_concurrent2` (`--parallel 2 --slot-ctx 32768`, served config): 2 streaming requests at once, ~1.8k-token prompts, 8 tools, thinking on, T 0.7, 192 tokens. `b_concurrent3` (`--parallel 3`): 3 at once | **PASS.** N = 2: 2/2 (tool_calls, length); 2 tool calls parse; reasoning in the thinking reply. Lane 1 (a copy task) ran 7 lookup verify passes, 46 drafts accepted. The first request's prefill started alone in its serial turn (the cards pipelined) and was handed to the lane pipe at a chunk end when the second arrived (1 chunk in the turn, 1 in the pipe; 1 handover). N = 3: 3/3 (tool_calls, length, length); tool calls parse; reasoning in 2/3; lanes_active max 3. |
| **(c)** a cancel mid-batch leaves the others intact | `f_ident_lookup` / `f_ident_nolookup` (`--parallel 3`, `IE_DS41_CPU_MISS=0`, lookup on / `IE_DS41_LOOKUP=0`): all 3 at once, request 1 closed after 6 content chunks | **PASS, both.** The 2/2 others are identical to their solo texts. The server answered request 1 again afterwards from its lane (32 tokens; with lookup: 3 passes, 18 drafts accepted). |
| **(d)** arrival TTFT while others decode <= solo TTFT + one chunk | `d_ttft` (`--parallel 3`): a 1,744-token prompt on a fresh server; then 2 requests decode (400 tokens) and a 1,747-token prompt sharing no prefix arrives | **PASS.** Solo 7.08 s (a fresh server's first request, prefilled in its turn with the cards pipelined); busy **7.88 s** (+0.80 s; allowance 8.3 s = the solo time of 2,048 rows). The arrival's prefill ran through the lane pipe (2 chunks). One decoding lane made 56 deltas during the 7.9 s prefill; the other had ended in a tool call, whose deltas the server buffers. |
| **(e)** RssAnon / VRAM / threads flat over 20 requests | `e_memory` (`--parallel 2`, `IE_DS41_PROMPT_CACHE_GIB=0`: a conversation kept in a host slot is the cache's design, not a leak): 20 Dream-shaped requests in waves of 2, 48 tokens | **PASS.** 20/20. RssAnon 845 -> 890 (wave 1) -> 892, 894, 896, 899, 900, 900, 900, 900, 900 MiB (+10 after wave 1). The server's own VRAM (fdinfo) [27348, 27299] -> [28692, 27795] -> [28702, 27805] MiB (+10 per card, flat from wave 6). Threads 62 -> 119 -> 119. For comparison, the `--parallel 1` runs (`a1_*`) grew 37 -> 300 threads over 16 requests: each HTTP pool thread that runs a forward keeps an OpenMP team (the engram gather). |
| **(f)** each lane == its solo run (CPU leg off, greedy, lookup on and off) | `f_ident_lookup`, `f_ident_nolookup` (as (c)): 3 conversations with different first system tokens (two copy tasks), warmed into their own lanes one at a time (8 tokens), then solo one at a time, then all at once. Every arm is served from its lane's think-tag checkpoint (cached = prompt - 1) | **PASS, both.** Batch == solo **3/3** (96 tokens each). With lookup on, the copy task ran 7 verify passes (56 rows, 47 drafts accepted) through the pipe in every arm. The lanes' solo texts == `a1_branch`'s (`--parallel 1`) 3/3, and lookup on == lookup off 3/3 (solo and batch). Then `evict`: a new conversation took the least recently used lane (its conversation kept in a host slot). Every conversation asked again was served from cache -- two restored from host slots onto OTHER lanes (lane 2 <- lane 0's, lane 0 <- lane 2's) -- with texts identical to solo 3/3, in both runs. |
| Throughput A-B-A | `t_aba2_plain` (`--parallel 2`, served config, `IE_DS41_STEP_TRACE=1`): tasks 2 and 4 (no copy), greedy, thinking off, no tools, 320 tokens, `--warm`. `t_aba2`: tasks 0 and 1 (1 = a copy task), 256 tokens | `t_aba2_plain`: solo A1 **14.81** tok/s over 17.0 s, **2 lanes 24.92** tok/s over 19.0 s, solo A2 **13.99** tok/s over 19.0 s: **x1.73** (per lane 12.46). No arm drafted (lookup on, nothing to copy). Step trace: one-row step median 67.8 ms with 1 lane decoding (571 steps), 76.8 ms per lane step with 2 (492 steps). `t_aba2`: 14.04 / **24.27** over 6.5 s / 14.23 = x1.72 (lane 1's copy task drafted: 7 passes). |
| Reply room (MiMo B4 gate finding) | `r_room` (`--ctx 32768 --parallel 2 --slot-ctx 4096`): a prompt calibrated to 3,748 tokens, max_tokens 700, then a short request | **PASS** (13:47). Lane 0 (reply cap 700): 700 tokens, finish length; a 4,096 lane would have cut it at 348. The short request took lane 1 (the small lane). A first run (13:32) placed the request the same way (lane 0, reply cap 700), but the test's check failed on the model's behaviour: under Dream's system prompt it answered with a tool call after 88 tokens. The check now fails only on a "length" finish short of max_tokens, and the prompt is plain. |
| Images at N > 1 | `v_vision` (`--parallel 2`): a 600-token text reply decodes; 1 s in, an image request (the Phase 57 fixture) arrives; then its follow-up | **PASS.** "MACHX 4721" and the three shapes; the image request prefilled in its serial turn (1 chunk). The text reply made 31 deltas during the 6.5 s image request and completed (600 tokens). The follow-up served 270 of 288 positions from its lane's live state (the continuation preference kept it on its lane) and named the blue circle. |
| Lone prefill handed to the pipe | `i_interrupt1` (`--parallel 1`, CPU leg off: the reference) / `i_interrupt2` (`--parallel 2`): a 7,572-token prompt A; 4 s later a 961-token request B | **PASS.** At N = 2, A's prefill ran 1 chunk in its turn (the cards pipelined), stopped when B waited for the turn, and ran its other 3 chunks through the lane pipe beside B's 2. A's text == the reference. TTFTs: A 30.3 s, B 26.4 s at N = 2 (A 20.2 s, B 32.8 s at N = 1, where B waited for all of A). |
| Disconnects | `x_disconnect` (`--parallel 2`): `tools/serve_disconnect_probe.py --rounds 6 --idle 70` | **PASS.** 6/6 streams cut mid-generation (FIN and RST alternating); the slot was free 0.3 s later each time. 70 s idle: 0.000 cores, idle_spin 0; threads 62 -> 119 -> 119. |

Costs seen in the runs:
- The done callback runs on card 1's stage thread. At T 0.7 with tools and thinking it took 93-158 ms per request over 114-191
  steps (~0.8 ms per step); greedy, ~0.05 ms per step.
- A cached turn (the prompt served from its lane's think-tag checkpoint): prefill 62-256 ms, including the tag row.
- `paused_ms` counts the turns' whole work, including a lone prefill run in its turn. In `t_aba2_plain`: 6 turns, 10.4 s, of which
  ~10.2 s were the two warm prefills.
- Background load: another session's headless Chrome and Python test processes took 20-170 % CPU in a few 5 s samples
  (`<run>_cpu.log`). Every rate above is uncertain at the few-percent level.

## Risks

- **3 lanes add no decode throughput over 2.** In `b_concurrent3`, a lane step took 118.7 ms with 3 decoding (~25 tok/s), against
  ~25 at 2. `--parallel 3` buys a third concurrent request, not speed. Unverified: one run, T 0.7, short replies.
- **Untested configurations.** The runs used `--ctx 32768` with 32,768-position lanes and ~1-8k-token prompts. Not run:
  - the owner's configuration (`--ctx 75000`) at N > 1;
  - Dream's real 18k-token agent prompts (90 tools);
  - a disk entry written from a lane (system prefixes >= 4,096 tokens). That path is the serial path's `prefix_persist`, called in
    the turn (not exercised here).
- With the CPU leg on (the served default), a lane's text is not bit-reproducible run to run, as at `--parallel 1`.
- **Sampling cost on card 1.** The done callback samples on card 1's stage thread. At T > 0 with long verify steps (up to 8 rows;
  ~1-4 ms per row on `ds41_sampler_test`'s rows) that time delays card 1's next stage. Measured here: ~0.8 ms per step at T 0.7.
- **An interrupted lone prefill finishes later.** Its remaining chunks run through the lane pipe, where one lane's chunks do not
  overlap across the cards (`i_interrupt2`: A 30.3 s against 20.2 s alone).
- A prompt with image positions left prefills entirely in its serial turn, and the other lanes stall for it (6.5 s in `v_vision`).
- Stage-error recovery (every running lane failed and reset, a new pipe) was not exercised on the GPU (no fault injection); it is
  argued from the code.
- The capacity guard on host-slot and disk restores has no host test (the selection lives inside `prefix_prepare`).
- `IE_DS41_MIX_CHUNK` below the full chunk was not measured.

## Not done

- A served A-B-A on recorded Dream agent traffic (`~/ds41_work/p58/requests`, 18k-token prompts): it does not fit a 9-minute job
  with cold prefills.
- `--ctx 75000` at N > 1; the mix-chunk cap; an N = 4 run.
- A GPU fault-injection test of the pipe-error path.

## Merge notes

- `include/ie/engine.hpp` and `src/server/openai_server.cpp`: the same `serving_status_json()` hook as MiMo B4 (`p4-b4-mimo-serve`),
  with V4.1 comments. B4 defines `Engine::serving_status_json()` in `mimo26_engine.cpp`, this branch in `ds41_engine.cpp`: merging
  both needs one definition (`if (ds41_) ...; if (mimo26_) ...`) and one comment.
- `engine.cpp`: no change (V4.1's chat dispatch already goes to `ds41_chat`, which routes to the lanes).

## Independent gate (2026-09-26): PASS

Gated on the final integration tree 2a9c994 (`ie` 568f9856), where B4's and B6b's `serving_status_json()` became two
arch members behind one dispatcher in `engine.cpp`.
- (a) `--parallel 1` SSE bytes equal HEAD's on 3/3; the HEAD captures were confirmed to come from d0e39c7.
- (f)/(c) batch == solo 2/2 with lookup on; the cancel survivor == solo; slots restored onto the other lane == solo.
- Two cold 5.9k-token prompts at once: texts and usage equal `--parallel 1`, and the disk entries written from the lanes
  match in every live byte.
- Throughput A-B-A: 2 lanes 25.92 tok/s against solo 14.86 / 14.90, x1.74.
- (e) Flat. Thread growth at `--parallel 1` (37 -> 300) is bounded and predates B6b: one OpenMP team per HTTP pool thread.
- MiMo on the same build: identity md5 65501302…, `/health` fields correct. No kernel-name collisions (checked on the
  object files).
- Non-blocking notes: a prompt only lane 0 fits holds an admission slot while it waits; a request waiting for a serial turn
  does not check its client; stage-error recovery has not been fault-injected.
