# V4.1 — Phase 46: the prefix cache (prompt cache) for `ie serve`

## Why

Dream sends its whole conversation plus ~90 tool schemas on every turn (17.7k tokens on turn 1). V4.1 had no prompt
cache, so every turn re-prefilled everything: **68 s to first token on every turn** of a chat, at 260 tok/s.

## What the state at a position is (and why a checkpoint is small)

| state | per layer | at position P |
|---|---|---|
| latent KV cache `comp_kv`, index keys `idx_k` (kv-source layers 2, 8, 14, 20) | [cap, 512] / [cap, 128] | the first nc(P) rows -- **append-only**: later positions never rewrite them |
| sliding-window ring `win_kv` (every layer) | [128, 512] | slot p % 128 holds position p's key: **overwritten** by later positions |
| ratio-2 held half `part_kv` / `part_gate` (layers 2-19) | [2, 512] | the open group when P is odd |
| `nc`, `part_valid` | host | counts |
| `all_ids_` | host | the engram look-back |

So restoring P = copy back the rings and halves captured at P, set nc / part_valid, truncate `all_ids_`. The latents
need no copy. A checkpoint is ~5.2 MiB per card (pinned host).

## The design (`src/model/deepseek41_prefix_cache.cpp`)

- **Checkpoints** at every prefill chunk boundary (captured per card at the end of its layers, so a pipelined prefill
  captures card 0 while card 1 is a chunk behind) and at the end of every prompt (`prefix_checkpoint`). 24 kept; the
  lowest position is evicted first.
- **`prefix_prepare(ids)`** picks the largest reusable position <= min(common prefix, ids.size() - 1): the live state
  itself when the prompt extends everything it holds, else the best live checkpoint, else a host slot.
- **Host slots**: a whole other conversation (its latents at its head + its checkpoints, pageable, 4 GiB default,
  LRU, never below 24 GiB MemAvailable). The live conversation is kept in one when an unrelated prompt arrives, or when
  a restore would drop >= 1,024 PROMPT tokens (not when it only drops generated reasoning an agent client does not
  echo -- that would copy the conversation to host every turn and evict other slots).
- The generator runs the rest as continuations from the reused position (odd positions included), pipelined when
  there are several chunks, never leaving 2..8 rows for the end.
- A forward that fails or throws with the cache on drops the live state (it may be half-applied).
- On in `ie serve` (`Engine::ds41_load`) unless `--no-prompt-cache` / `IE_NO_PROMPT_CACHE`, `IE_DS41_PROMPT_CACHE=0`, or
  speculation; `IE_DS41_PROMPT_CACHE_GIB` sets the host budget. `usage.prompt_tokens_details.cached_tokens` reports it;
  capabilities advertise `prompt_cache` for V4.1.

## Gate: `ie-ds41-cache-test` -- bit-identical against controls (CPU miss split off)

Real text (wikitext-2), ctx 16,384, chunks of 2,048. Each arm's last logits (fp32 FNV) and 16 greedy tokens against a
control computing the same chunks with no cache involvement:

| arm | what went through the cache | result |
|---|---|---|
| 1 | checkpoint at 8,192 restored after 40 dirty decode steps | reused 8,192, **identical** |
| 2 | live state kept, continuation from odd position 8,213 | reused 8,213 (live), **identical** |
| 2b | ODD checkpoint (8,713, a held half) restored after 16 dirty steps | reused 8,713, **identical** |
| 3 | S swapped to a host slot by an unrelated prompt, prefilled Z, S swapped back | saved 90 MiB in 30 ms, loaded in 16 ms, **identical** |
| 4 | a prompt diverging at 5,000: back to the checkpoint at 4,096 | reused 4,096, **identical** |

**18/18 PASS** (`ds41_work/tools_live/cache_test2.log`; re-run on the final build: `cache_test3.log`).

## Live, `ie serve` at ctx 75,000 with Dream's own 90 tool schemas, streamed (`tools_live/turns.py`, `server4.log`)

| turn | prompt | cached | time to first token | prefill wall |
|---|---|---|---|---|
| 1 "what model are you?" | 17,733 | 0 | 68.5 s | 68.2 s |
| 2 read_notes tool call | 17,906 | 17,878 | **2.1 s** | 1.76 s |
| 3 tool result -> answer | 18,009 | 17,970 | **2.3 s** | 2.17 s |
| 4 follow-up | 18,170 | 18,154 | **2.3 s** | 2.14 s |
| side request (another system prompt) | 17,726 | 0 | 65.5 s | 65.1 s (main conversation kept in a host slot, 125 ms) |
| 5 back to the main chat | 18,187 | 18,170 | **1.7 s** | 1.41 s (host slot swap included) |

The remaining ~1.4-2.2 s per cached turn is the 16-40 new tokens as one continuation chunk: at that size it is
expert-byte bound (~60-130 ms per token), about what decode steps would cost.

## Not covered / known limits

- A request whose SYSTEM text differs shares only the BOS token: the tools section is rendered after the system text
  (the model's format), so a side request with its own system prompt re-prefills its tools.
- Nothing persists across a server restart (host slots are in memory).
- Speculation (`IE_DS41_SPEC=1`) keeps the cache off: the drafter's rings are not checkpointed.
