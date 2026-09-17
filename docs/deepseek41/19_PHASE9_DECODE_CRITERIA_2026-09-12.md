# V4.1 port — Phase 9 gate criteria: decode (written BEFORE the build; Phase 8 = residency)

**Scope:** single-token steps at `start_pos = n` after a prefill, with every cache carried
across steps, checked against the reference's own `generate` loop (`golden_decode.py`): the
first decode step layer by layer, then the chosen tokens for N steps.

## What decode is, from `generate.py:64-78` and the module caches in `model.py`

The reference feeds **only the new token**: `model.forward(tokens[:, prev_pos:cur_pos], prev_pos)`.
Everything else lives in the modules and must be carried by the engine:

| cache | owner | decode behaviour (model.py) |
|---|---|---|
| `window_kv_cache [win=128, 512]` | every layer | ring: slot `pos % 128` overwritten; query sees the whole ring, oldest first; slots > pos are −1 (`get_window_topk_idxs`, :410) |
| `compress_kv_cache [max/ratio, 512]` | kv-source layers | ratio 1: one latent per token. ratio 2: `Compressor` holds `kv_state`/`score_state` for the partial group and emits a latent only when `(pos+1) % ratio == 0` (:458-486) — **decode at odd positions produces no new compressed entry** |
| `k_cache` (index keys) | kv-source layers | same cadence as the latent (:539-548) |
| `shared_attn.topk_idxs` | index-source layers | recomputed per step over `compress_len = (pos+1)//ratio` keys; consumers reuse |
| `NgramHashState.cache` | engram | compressed ids of past tokens; the 4-gram look-back reads it (:159-175) |
| RoPE | all | token position `pos` for q/kv; group position `pos + 1 − ratio` for a new latent (:751) |

Two consequences that the tests must exercise, because prefill never does:

- **The still-filling ring.** For `pos < 127` the window has empty slots → −1 indices. This is
  the `-1e30` running-max floor in `sparse_attn` that the Phase 5 gate flagged as unexercised:
  a row with *some* −1s is exercised by any decode step; a row with *all* −1s cannot occur
  (the current token is always present), so the floor's full-invalid case stays a named
  non-case, and the test must say so rather than claim it.
- **The partial group.** At ratio 2, step at even `pos` adds a latent; at odd `pos` it does not.
  A test that decodes only 1 token sees one of these; **decode ≥ 2 tokens** so both occur.

## Pass criteria

1. **Prefill state is what decode needs.** After the prefill of the golden prompt, the engine's
   window ring, compressed caches and index keys equal the values the reference holds at the
   same point — checked for layers 0 (window), 2 (ratio 2 source) and 20 (ratio 1 source) at
   3e-3, *before* any decode step. A cache that is wrong at the boundary makes every later
   comparison uninterpretable.
2. **The first decode step matches layer by layer**: `d_layer_out_{L}` at the same per-layer
   tolerances as Phase 7, plus `d_engram_out_{1,14}`. This step is at `pos = 12` (even), so it
   adds a latent at ratio 2 and reads a 13-slot ring at layer 0.
3. **The next token matches** the golden's for step 0, and the **top-5 set** matches.
4. **N ≥ 4 steps reproduce the golden's tokens greedily**, which forces at least two odd and two
   even positions through the partial-group state machine. Any divergence is reported with the
   step and both top-5 lists. Divergence at step k > 0 is a **soft** failure only if the golden's
   own top-1/top-2 margin at that step is below the accumulated logit error (report it); a
   divergence with a clear margin fails.
5. **Positions are correct.** A negative control: run step 0 with the RoPE position off by one
   and show it fails criterion 2 — so the position bookkeeping is demonstrated load-bearing.
6. **The engram hash cache carries.** The step-0 hashes for the new token must equal the
   reference's (they reach back into the prompt for the 4-gram), checked as integers.
7. **No prefill regression**: `ie-ds41-forward-test` still passes on the same build.
8. **Memory is bounded per step**: the per-step allocation is the routed experts for one token
   (≤ 6 per layer) plus nothing that grows with the step count except the caches themselves.

## Explicitly NOT in this phase

Speed (Phase 8/10), batching, the candidate top-k at length, the NVMe tier, sampling other than
greedy, stop sequences, the serving loop, tokenisation inside the engine.
