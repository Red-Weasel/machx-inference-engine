# V4.1 — Phase 57: images in `ie serve`, and the long agentic soak

## Serving

- `Engine::ds41_chat` takes a turn's images (OpenAI `image_url` data URIs, any role -- Dream sends a tool's
  screenshot as a following user message named `dream_visual_evidence`), puts one `<｜deepseek_image｜>` per image
  where the parser's marker sat, probes each image's size from its header (the plan, and so the span's length,
  needs no decode), and expands every placeholder into its span of image positions (docs/95's negative ids).
- The tower (`Ds41Bundle::vis`) is staged in pinned host memory at load (1.1-1.4 s), BEFORE the runtime sizes its
  pinned expert pool. Nothing of it sits on a card: an encode leases 1,487 MiB on card 0 and returns it.
- Encoding is lazy: the forward asks a per-request provider for an image position's row; the provider decodes,
  encodes and lays out that image on its first row. A position the prefix cache already holds is never asked for,
  so a follow-up turn about an image does not re-encode it.
- `IE_DS41_VISION=0` leaves the tower out; a load without it refuses image requests with the reason. Speculation
  (`IE_DS41_SPEC=1`) refuses them (the drafter has no image path). `ie capabilities` reports `vision: true`, which is
  what Dream reads to send images.

## The think-tag checkpoint

A chat prompt ends `<｜Assistant｜><think>` (or `</think>`); the next turn renders the same assistant turn with the
other tag, so two consecutive prompts part at T - 1 and the end-of-prompt checkpoint at T was never reusable -- every
turn fell back to the last user message's start, re-running that whole message (with an image: its encode and its
hundreds of tokens). The planner now ends one token early, checkpoints there, and runs the tag as a one-row step. The
image follow-up went from 38 to 264 of 325 tokens cached with no re-encode (`~/ds41_work/p57/e2e2.log`).

## End to end (`~/ds41_work/p57/vision_e2e.py`, ctx 75,000)

| request | result |
|---|---|
| read text + shapes (640 x 480, 206 tokens) | "MACHX 4721", red square, blue circle, green triangle; encode 230 ms |
| follow-up in the same chat | "blue circle"; 264 of 325 cached; prefill 2.6 s vs 6.0 s |
| same words, a different (noise) image | only the system prompt shared (38 cached); no invented text |
| 1920 x 1080 table screenshot (968 tokens) | exactly rows 00, 03, 06, 09; encode 2.9 s |

## The agentic soaks (Dream's 90 tool schemas, simulated tools, one conversation)

`agent_soak.py`: three screenshots inspected with parallel `see` calls, one `note` each, `update_todos`, a report
through `memory_write`, `done`; a recall question; a re-check against the stored report. **9/9, 8 model calls.**
The notes are exact down to the table: all twelve values (000, 919, 838 ... 109, "descend by 81"), the four FAIL rows,
the header text, and the noise image called unreadable rather than described.

`agent_long.py`: twelve generated screenshots (a random code + four units each, ALARM or normal), processed STRICTLY
one at a time -- `see`, `note`, `update_todos` -- then a report and recall questions about the earliest images.
**42 model calls, 18.4 minutes, context grown to 33,788 tokens: every code and every alarm set in the report is
right (12/12, including a batch with two rows named "unit 23"), the recall answers are right, no malformed tool call,
no server error.** (The first scoring printed two FAILs; the checker searched for "unit 53" where the report wrote
"53" -- the model was correct.)

| per call (after the first) | value |
|---|---|
| cached / prompt | median 0.996 (text steps 0.999+; an image step adds its 600-970 new tokens) |
| decode at 20-34k context | 8-11 tok/s |
| server threads | +19 per request until every HTTP pool thread has run one, then flat at 340 (below) |
| server RSS | flat (~13-17 GB, of which RssAnon ~1.6 GB; the rest is mapped shards) |

**Threads.** Engram's gathers are an OpenMP region on the calling thread; this runtime gives every root thread its
own 19-worker team and never reclaims it. The HTTP pool is fixed (parallel 1 + queue 8 + 4 = 13 threads), so the
growth is bounded at 13 teams -- measured flat at 340 over the last 40 calls. Not the 09-16 leak (a NEW thread per
call, unbounded). Worth tidying (one persistent team for the gathers), not urgent.

**Where an agent's time goes.** 13 `update_todos` calls decoded ~3,900 tokens, 7.5 of the 18.4 minutes, and nearly
all of each was the previous todo list copied out again. Phase 58 (docs/97) targets exactly that.
