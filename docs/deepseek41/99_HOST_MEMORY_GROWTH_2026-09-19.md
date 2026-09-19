# V4.1 — `ie serve` host memory growth under a day of Dream traffic

## What happened

A Dream session (2026-09-18 20:30 → 09-19 12:55, ~250 requests, 184 images, dozens of full re-prefills) grew the
server's anonymous memory from about 15 GB to **31.6 GB resident + 4.2 GB swapped**. MemAvailable fell to 3.2 GB,
swap was full, and the out-of-memory killer would have picked the engine, which holds ~100 GB of pinned host
memory; a hard kill of such a process has wedged the kernel before. Dream was closed by hand; the teardown was
clean (both cards back to 26 / 34 MiB, taint unchanged).

## What it was not

- **Not the vision tower.** `ie-ds41-vision-leak` (new, tools/ds41_vision_leak.cpp) runs the serving path's
  per-image work (decode → `encode_gpu` → `assemble_rows`) on real Dream screenshots: 48 encodes, RssAnon flat at
  82 MiB.
- **Not the prefix cache's host slots** in the live run: the cache logged 49 "not keeping a host slot" and 1 kept
  slot (MemAvailable was under its 24 GiB floor all night). The slots are capped (4 GiB) and budgeted by design.

## What it was: freed memory the heap kept

`IE_MEM_REPORT=1` (new) prints RssAnon and glibc's heap after every request. The replay
(`~/ds41_work/p60/leak_replay.py`) is Dream-shaped: a growing conversation, 2 screenshots per round, a system-prompt
change every 6th round:

| round | RssAnon | heap in use | free in arenas |
|---|---|---|---|
| 0 | 963 MiB | 239 | 132 |
| 9 | 2,223 MiB | 710 | 960 |
| 21 | 4,482 MiB | 1,745 | 1,927 |

A request's transient buffers (prefill chunks, image rows, bounce copies; many below glibc's dynamic mmap
threshold, allocated on 13 server threads) are freed into the per-thread arenas, and glibc keeps them. Over the live
run's ~250 requests that is the bulk of the growth.

## The fix

`release_free_heap()` in `src/server/openai_server.cpp`: `malloc_trim(0)` right after every generation, both the
streaming and the non-streaming path (`IE_MALLOC_TRIM=0` turns it off). Same replay:

| round | RssAnon before | RssAnon with the trim |
|---|---|---|
| 9 | 2,223 MiB | 1,493 MiB |
| 21 | 4,482 MiB | 2,771 MiB (−38 %) |

What remains is in use and bounded: the host slots the prefix cache keeps on purpose (4 GiB cap, never below the
24 GiB MemAvailable floor) and per-context data limited by the context window. The trim costs 0.5-2.5 ms per request.

## Gates

- `vision_e2e.py` against the trimmed server: 4/4 (reads MACHX 4721; cached follow-up; different image shares no
  cached image tokens; the 968-token full-HD table).
- Build of `ie` and `ie-ds41-vision-leak`; the vision leak probe flat over 48 encodes.

## Not done

A longer replay that reaches the live run's scale (hundreds of requests, 40-60K contexts) was not run; the live
number (~30 GB) is attributed to the same mechanism by its shape (slots refused all night, heap-held memory grows
with every request), not measured after the fix.
