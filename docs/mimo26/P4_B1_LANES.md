# P4 B1 — MiMo-V2.6 lanes with serial interleaving (2026-09-25/26) — all gates PASS

Design: `~/ds41_work/p60/p4_mimo_batching_design.md` (B1 = section 4). Plan: `~/ds41_work/p60/multimodel_parallel_plan.md`.
Branch `p4-b1-mimo-lanes` from `deepseek4-vision-exp` 5c55d35. Owner's constraint: additive; one lane (the default) must be
byte-identical to the pre-lane engine. Serving is NOT changed (`mimo26_load` still forces `parallel = 1`; that is B4).

## What was built

| Piece | Where | What |
|---|---|---|
| Forward lanes | `Mimo26Options::lanes / lane_ctx`, `Mimo26Forward::select_lane / lane / n_lanes / lane_bytes` | Each lane has its own full-layer K/V (linear to its ctx, V padded to 192), SWA rings (`ring_` slots), `n_pos`, `written_end`, capacity. `select_lane` swaps the active lane's pointers into `Dense::k/v` and its positions into `n_pos_/hi_end_/cap_`. Lane 0 = the buffers `upload_card` always allocated. Kernels, `forward`, `state_spans`, `set_state`, `capacity()` read `cap_` (== `max_ctx` on lane 0). |
| VRAM | `upload_card` | Extra lanes are allocated per card after the workspaces and **before** the auto static tier sizes, so their bytes come out of the static tier (the design's ~1-2 %/token per lane, an estimate). Refused before allocating, with the numbers, unless free VRAM leaves `IE_MIMO26_VRAM_RESERVE_GIB` (1.5) + the tier's batch workspace + its smallest tier (n_static, or stream_slots + 1 per MoE layer) + the vision reserve on card 0. The allocated bytes are checked against the arithmetic (`mimo26_lane_kv_bytes`). |
| Drafter lanes | `Mimo26DFlash::add_lanes / select_lane` | One context ring (5 layers x K/V) + bookkeeping (`ctx_lo/end/hi`) per lane; call after `init`, before the forward's `init`. |
| Host slots | `Mimo26PrefixCache` | The live bookkeeping (`live_prompt_`, `live_uses_`, `drop_id_`, the #87 snapshot) is per lane, swapped in by `sync_lane()` at `prepare / prompt_done / live_lost` (no-op with one lane; errors if the drafter is on another lane). Slots are shared across lanes; a slot whose `written_end` exceeds the active lane's capacity is skipped. So: keeping an idle lane's conversation in a slot and restoring it into ANY lane is the existing E1/E2 `state_spans`/`set_state` path. |
| Host bookkeeping | `include/ie/mimo26_lanes.hpp`, `src/model/mimo26_lanes.cpp` (no SYCL) | `Mimo26Lane` (prompt, live ids, prefill cursor, max_new/n_new, next id, rng, 512-id repetition window, `VitalsWindow*`, outbox, finish, tick), `mimo26_lane_commit` (the engine's emit rule), `Mimo26LaneSet` (admit to LRU idle lane, serial round robin `next()`, `release`, `lru_idle`), `mimo26_lane_kv_bytes`, `mimo26_lanes_fit`. |
| CPU test | `tests/unit/mimo26_lanes_test.cpp` (ctest `mimo26_lanes_test`) | Lane bytes vs the design (27,648 B/token; rings 39x8x2176x320x2 B; one lane at 32K = 1,341 MB over both cards vs the design's ~1,360), fit refusal, commit rule, round robin, mid-run finish/re-admit, LRU. |
| GPU gate | `tools/mimo26_lanes_test.cpp` (`ie-mimo26-lanes-test`) | Solo runs on lane 0, then the same prompts interleaved over L lanes (one forward at a time; finished lanes take the next queued prompt on a dirty lane). Per sequence: logits FNV after the prompt and after EVERY step, tokens, drafter drafts+probs before every step, drafter bookkeeping — all must equal solo. `--slot-arm`: A on lane 1, another prompt on lane 1 keeps A in a host slot, A's continuation restored on lane 2 (or 0) == a control that never left lane 0. CPU miss leg off unless `--cpu-miss`. |

## Results

Verified (observed):
* Build of all targets: OK. ctests `ds41_prefix_key_test, idle_spin_test, server_admission_test, mimo26_host_rules_test,
  ie_vitals_test, ds41_sampler_test, openai_proto_test, mimo26_lanes_test`: 8/8 passed.
* **Gate (a) lanes = 1 == HEAD** (2026-09-25 23:51-23:58, cards idle 26/34 MiB, Dream closed, `IE_DS41_CPU_MISS=0`):
  `ie-mimo26-run` on the 10 held-out prompt files (00, 01 and 02 are byte-identical, md5 e0034d67…: 8 distinct prompts) (`results/mimo26/p4/prof/heldout_prompts`), `--n 64 --chunk 2048 --ctx 16384
  --static 0 --ranking <chat> --dflash 7 --dflash-minp 0.7`. HEAD binary (built from 5c55d35, md5 840403713b41db825fffe5caab6631d6)
  vs branch binary (md5 d79af602c4d394d16facd9eaad546a30): the 10 `generated:` lines are identical (md5
  3b199ccf940f0a2b7665cfd2ab790f99 both), prompt ids identical, VRAM 31.21 / 28.05 GB and static/pinned 78/178, 62/194 per
  layer identical, decode ms/token per prompt within 0.5 % (e.g. 84.4 vs 84.2, 88.8 vs 88.3). Kernel log clean after both.
  This covers the default path including the drafter/verify loop; it compares tokens, not logits bytes (unverified: logits
  byte-identity at one lane is argued from the code — same launches on the same pointers — not measured).

**Gates (b)-(d), timings and E2 — resumed after the reboot, 2026-09-26 00:44-00:59, all PASS.** Each run via the main
checkout's `scripts/ie-run-guarded --mem 220G --timeout 540`, foreground, preflight clean before each (no Dream, cards 26/34
MiB, `journalctl -k --since "-10 min" | grep -iE "CAT error|reset|timed out"` empty), 30 s wait + the same kernel-log check
after each: empty every time. Binaries built at 3a57754: `ie-mimo26-lanes-test` md5 d6422191286d8d2817829a6347c079e3,
`ie-mimo26-cache-test` md5 5e6dae48b3da486dbb70605170ca2ba7. CPU miss leg off (the tools set it), drafter ON, held-out prompts.

| Run | Args | Result |
|---|---|---|
| L = 2 | prompts 00, 03, 04, 08; `--lanes 2 --n 64 --ctx 16384 --lane-ctx 12288 --slot-arm` | exit 0, `PASS` (4 m 13 s). All 4 sequences interleaved == solo: 64 logits rows (fp32 FNV after the prompt and every step), 64 tokens, drafts + probs and drafter bookkeeping. 262 forwards, 260 lane switches, 2 mid-run re-admissions (a finished lane took the next prompt on a dirty lane). Slot arm: A (6,002 positions) on lane 1 kept in a host slot (592 MiB, 226 ms) when another prompt arrived there, restored on lane 0 (60 ms), the continuation's logits / 16 tokens / drafter state identical to the control. |
| L = 3 | prompts 01, 05, 06, 07, 09; `--lanes 3 --n 64 --ctx 16384 --slot-arm` | exit 0, `PASS` (4 m 57 s). All 5 sequences identical to solo; 328 forwards, 326 switches, 2 re-admissions; lane order 012012...; slot arm: A (5,738) kept from lane 1 (585 MiB, 227 ms), restored on lane 2 (62 ms), identical. |
| E2 | `ie-mimo26-cache-test --only87` | exit 0, `MIMO26 CACHE TEST: PASS` (3 m 44 s), 15 ok: served 8192 / 8188 / 8191 from prompt end, snapshot 68 MiB in 37-38 ms, restored in 10 ms, logits / 16 tokens / drafter identical to the control (same numbers as the E2 gate on 2026-09-25). |

**(d) Lane VRAM**, logged at init and checked against `mimo26_lane_kv_bytes` (`[ ok ]` both runs): one extra lane at ctx
12,288 = 0.401 + 0.374 GB (cards 0 / 1), at 16,384 = 0.464 + 0.425 GB; drafter ring 0.042 GB per lane (card 1). The
arithmetic at 32,768 gives 1.342 GB for both cards (~0.67 GB/card) — the design's ~0.68 GB/card. Cost seen in the
auto static tier: L = 2 at 12,288 took card 0 from 78 to 76 and card 1 from 62 to 60 static experts per layer; L = 3 at
16,384: 75 and 59.

**Timings (interleaved vs solo, same process, same prompts):**

| | solo wall | interleaved wall | ratio | decode ms/step solo | interleaved |
|---|---|---|---|---|---|
| L = 2 | 69.7 s | 71.0 s | 1.018 | 105.2 | 112.2 (+7 %) |
| L = 3 | 95.6 s | 99.1 s | 1.036 | 110.4 | 123.1 (+11 %) |

As expected, serial interleave gives no gain (aggregate 3.61 vs 3.67 and 3.23 vs 3.35 tokens/s over the whole run,
prefill included). The per-step comparison above (+7-11 % interleaved) is **inconclusive, order-confounded**: solo always ran
first on cold caches, and the B1 gate's own rerun measured interleaved/solo wall 0.951 (interleaved faster). The stream-slot
mechanism itself was later measured per lane in B2 (docs/mimo26/P4_B2_PIPELINE.md): interleaving lanes on a card does lower
its stream-slot hit rate (18-31 % solo -> 6-11 % with 3 serial lanes), but the per-step cost of that was not isolated
from order here. The decode step here includes a draft of 7 every step (~10 ms) and the tool's per-step FNV of the logits,
so the absolute ms/step is not serve speed.

### GPU incident 2026-09-26 00:01:43

Card 0 (0000:04:00.0) logged `Engine memory CAT error [18]: class=bcs, guc_id=0` at 00:01:43 and then a GT reset loop
(`Kernel-submitted job timed out ... in no process`, ~5,000 "Timedout job" lines by 00:05:44, still going). It began
**while other agents' GPU ctests were running** (P1 builder's `ctest`, `agent-aed3f9d470edc59ee/.../deepseek4_forward_test`,
`kernel_movement_test`), three minutes after my last run had ended with a clean kernel log (23:58:32 + 30 s). My
`lanes2` run started at 00:04:47 — my idle check looked at VRAM and processes, NOT the kernel log — and died after 10 s
with `UR_RESULT_ERROR_DEVICE_LOST` (it did not reach the model's init output). Attribution of the CAT error to the other
tests is unverified (timing only). Same signature as fix-list #54. The box was rebooted (~00:34); the gates above then ran on a clean card.

## Runbook (the commands the gates above ran)

Each run foreground via `scripts/ie-run-guarded --mem 220G --timeout 540`, cards idle, Dream closed, **and a clean
`journalctl -k` for the last minutes**; 30 s wait + kernel-log check after each.
```
ie-mimo26-lanes-test $M --prompt-file 00.txt --prompt-file 03.txt --prompt-file 04.txt --prompt-file 08.txt --lanes 2 --n 64 --ctx 16384 --lane-ctx 12288 --slot-arm
ie-mimo26-lanes-test $M --prompt-file 01.txt --prompt-file 05.txt --prompt-file 06.txt --prompt-file 07.txt --prompt-file 09.txt --lanes 3 --n 64 --ctx 16384 --slot-arm
ie-mimo26-cache-test $M <cat src/model/*.cpp> --only87
```
Observed: every line `[ ok ]`, `MIMO26 LANES TEST: PASS` / `MIMO26 CACHE TEST: PASS`; interleaved/solo wall 1.02-1.04 (B1 is
serial: no gain expected); 3 m 44 s to 4 m 57 s each.

## Risks

* Identity was shown on 8 distinct held-out prompts (00 and 01 are the same file content) x 64 greedy tokens with L = 2 / 3 and the CPU leg off; sampling (temperature
  > 0) per lane and longer runs are not exercised (the sampler state lives in `Mimo26Lane`, used from B4).
* Interleaved decode steps: the +7-11 % per step above is order-confounded (inconclusive); lane switching does cost stream-slot hits (measured in B2).
* Cross-lane contamination produces fluent wrong text; only the identity gate catches it.
* `select_lane` clears `features()`: a caller must feed the drafter right after each forward, before switching.
* The prefix cache's per-lane bookkeeping relies on the caller selecting the drafter's lane with the forward's; mismatch is
  an error from `prepare`/`prompt_done`, logged (not returned) from `live_lost`.
* Lane VRAM comes out of the static tier: per-token cost of each lane unmeasured (design estimate +1-2 %).
* The tier's stream-slot LRU is shared across lanes: harmless with the CPU leg off; with it on, lanes are not
  bit-reproducible (as across a server's history today).
* Diagnostics (`stats_`, `profile_`, `n_calls_` dump index) are shared across lanes.

## What B2 (card-pipelined lanes) needs next

1. Stage threads, one per card, with a one-slot mailbox (lane, rows, residual, positions). `run_card` must take the lane's
   KV explicitly instead of the swapped-in `Dense::k/v` (two lanes are active at once, one per card) — i.e. replace the
   B1 pointer swap by per-call lane state (`LaneState` already holds `kv[card][layer]`, `cap`).
2. Per-stage host buffers: `h_x_`, `h_pos_`, `h_rlogits_`, `h_ridx_`, `h_rw_`, `h_feat_`, and `n_pos_/hi_end_` per lane
   (committed when the lane's last stage finishes); the `static` counters in `run_card`/`mimo26_run_ids`.
3. The drafter on card 1's stage thread (shares that queue); per-lane `features()`.
4. CPU leg split per card (V4.1's "8-13"/"14-19") and a q* sweep; gate with the CPU leg off: output == B1.
5. Gate: B1's `ie-mimo26-lanes-test` identity (the reference), then A-B-A aggregate at B = 2 >= 1.3x solo plain.
