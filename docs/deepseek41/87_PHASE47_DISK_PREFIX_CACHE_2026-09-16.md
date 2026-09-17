# V4.1 — Phase 47: prompt prefixes on disk, and checkpoints aligned to message boundaries

## Why

After Phase 46 a chat's later turns reach their first token in ~2 s, but the FIRST turn of every new server process
still re-prefilled the client's system prompt + tools (17.7k tokens for Dream: ~70 s), and a later process knew
nothing of the previous one.

## The change

1. **Boundary-aligned chunks** (`Ds41Generator::run`, cache on): the chunk plan includes where the FIRST and the LAST
   user message begin (the `<｜User｜>` special token -- a token boundary in any rendering), so a checkpoint lands
   exactly on the system prompt + tools, and on the point a regenerated or edited reply resumes from. Even-length
   pos0 = 0 chunks, never 2..8 rows before a boundary; if a boundary cannot be landed on, the plain plan runs.
2. **Disk entries** (`Ds41Forward::prefix_persist` / `pc_load_disk`, `src/model/deepseek41_prefix_cache.cpp`): after a
   prompt, the state at the first-user boundary (>= 4,096 tokens) is written once per distinct prefix -- the rings and
   halves from the checkpoint block, the latents / index keys read from the devices, tmp + rename on a background
   thread. `prefix_prepare` loads an entry whose whole prefix the prompt holds when it beats the live state and host
   slots. File key: model shape + a fingerprint of the embedding bytes + this executable's size and mtime (a rebuild
   may change the arithmetic). Budget 8 GiB (LRU by mtime), never below 32 GiB free on the filesystem.
   `$XDG_CACHE_HOME` or `~/.cache` `/machx-ie/deepseek41-prefix`; `IE_DS41_PROMPT_CACHE_DIR` (`0` = off),
   `IE_DS41_PROMPT_CACHE_DISK_GIB`.

## Gate

`ie-ds41-cache-test` arm 5: S's state at 8,192 persisted (60 MiB, gathered 12 ms + written 14 ms), the cache torn
down (a new process's view), Z prefilled, S resumed from the file in 10 ms: last logits and 16 greedy tokens
**identical** to the no-cache control; a prompt diverging inside the entry's prefix does not load it. **21/21**
(`tools_live/cache_test5.log`). The first run of arm 5 FAILED -- the persist wrote the index keys over the latents
(one offset for both passes) -- and the arm is what caught it.

Decode 28/28, generate 15/15, openai_proto OK on the final build.

## Live: two server processes, Dream's 90 tools

| | prompt | cached | time to first token | |
|---|---|---|---|---|
| process A, turn 1 | 17,733 | 0 | 70.0 s | wrote a disk entry: 17,724 tokens, 118 MiB (gather 64 ms, write 27 ms) |
| process B (restarted), turn 1, a different question | 17,740 | 17,724 | **2.4 s** | loaded the entry in 25 ms |
| process B, the same request again | 17,740 | 17,724 | 1.8 s | identical text to B's first answer |

## Found: decode speed on this workload is the ranking, not the cache -- RETRACTED (docs/88: a thread leak)

Process B decoded 374 tokens at 3.2 tok/s (party ideas) against A's 8.9 tok/s (145 tokens, "what model are you").
A/B with `IE_DS41_PROMPT_CACHE=0` in a fresh process, the same request: 70 s prefill, **3.2 tok/s** decode (380
tokens) -- the cache is not the cause. The same request repeated in-process was 2.7 tok/s, so a cold page cache is
not either. The server's residency ranking (`ie_ranking_heldout.txt`) does not match chat-style generations: the
lever docs/69 measured at 2.4x at long context (a decode-phase ranking profiled on the workload).

**RETRACTED (22:10, docs/deepseek41/88).** The slow decode was not the ranking. Every mmap-tier fill spawned a fresh
reader thread whose OpenMP team libiomp5 never reclaimed: a served process reached 94,178 threads, +10,367 per
120-token reply, ~0.5 GB of anonymous stack per reply, MemAvailable 36 -> 15 GiB with swap exhausted -- and a
cache-off process's own 17.7k-token prefill spawns thousands of them before its first decode step, which is why the
A/B could not tell the arms apart. With persistent readers the same prompts decode at 5.5-8.6 tok/s (from 1.7-3.2).
