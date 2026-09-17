# V4.1 — Phase 49: a decode-phase residency ranking profiled on chat traffic (+47 % decode for Dream-style chats)

## Why

`ie serve` placed experts by `ie_ranking_heldout.txt`, a PREFILL histogram of a 65k-token corpus. docs/67-69 measured
that a ranking is only valid for the (phase, length, workload) it was profiled on and that a matched decode ranking
was a 2.4x lever at long context. Dream's traffic is chat decode at ~18k context (its 90 tool schemas).

## How it was made

`IE_DS41_PROFILE_OUT=<file>` (engine, docs/88): the decode routing of every served request, summed per (layer,
expert) and rewritten as a ranking after each request. Server at ctx 75,000, Dream's 90 tools, 14 Dream-style prompts
(code, networking, email, databases, summaries, fitness, React debugging, naming, cron, MoE explainer, rewriting,
travel, algorithms, a web-search tool call), 300 tokens max each, temperature 0: **3,813 decode tokens, 912,480
selections** (`ds41_work/rank48/rank_chat_decode.txt`, `profile.out`).

## A/B on 6 HELD-OUT prompts (not profiled: bash, HTTPS, SaaS checklist, inflation, haiku, blog outline)

Same build, same server settings, each arm a fresh process, the disk prefix entry loaded (1.2-2.6 s to first token):

| run order | ranking | decode, 6 prompts | per prompt tok/s |
|---|---|---|---|
| 2 | held-out prefill profile (the old default) | **7.26 tok/s** (1,657 tok / 228.3 s) | 7.44 7.89 7.92 6.57 5.62 8.62 |
| 3 | **chat decode profile** | **10.81 tok/s** (1,583 tok / 146.5 s) | 12.15 12.10 12.99 8.07 8.45 11.78 |
| 4 (A-B-A) | held-out prefill profile again | 7.35 tok/s (1,687 tok / 229.5 s) | 7.52 8.04 7.37 7.03 5.66 9.38 |

**+47-49 %, every held-out prompt faster, and the A-B-A rules out run order.** Streams differ in places (the CPU and
GPU expert paths are not bit-identical, so placement moves a near-tie; one held-out reply ended in a tool call instead
of at the length cap).

Prefill is unaffected: the 16k long test 296.9 (held-out) vs 300.5 tok/s (chat). Decode on the long test's SYNTHETIC
repeated block is slower with the chat ranking (8.18 vs 6.16 tok/s) -- the workload rule again, in the other direction.

## Shipped

`<model>/ie_ranking_decode_chat.txt` (added beside the model; `ie serve` prefers it when present, `IE_DS41_RANKING`
overrides, deleting the file restores the held-out profile). The best ranking for a given user is still one profiled
on THEIR traffic: run the server with `IE_DS41_PROFILE_OUT` for a working session and point `IE_DS41_RANKING` at it.
