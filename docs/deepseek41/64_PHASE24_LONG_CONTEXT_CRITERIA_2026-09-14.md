# V4.1 port — Phase 24 criteria: load and prefill at 200,000 tokens (written BEFORE the build)

**The founder's requirement (21:59):** "yes i want to load at least 200k". Today the engine refuses at 200,000 with
`ds4_expert_batch_ws_alloc: max_tokens*top_k*max(H,EF) overflows uint32`, and 32,768 is the largest context that
loads (verified 21:55, 53 s, clean shutdown).

## Why it refuses, and why it is NOT the streaming path

The expert WEIGHTS stream exactly as designed — a static tier in VRAM, a pinned tier in host RAM, the tail mmapped
from disk — which is why a 475 GiB checkpoint runs on 64 GiB of VRAM at all. What refuses is **scratch space sized
by the context**, because the engine assumes a prompt is prefilled in ONE forward:

| allocation | how it is sized today | at ctx 200,000 |
|---|---|---|
| the expert batch workspace (`ds4_expert_batch_ws_alloc`, per card) | `max_tokens × top_k` rows × (H or EF) buffers | **101 GiB** — the uint32 element-count guard refuses first |
| the expert-parallel staging (`ep_x`, `ep_ypc`, per card) | `max_tokens × dim` and `max_tokens × top_k × dim` | **~16 GB** |
| the forward's own stream scratch (`hA`, `hB`, per call) | `T × hc × dim` floats, twice | **~33 GB** at T = 200,000 |
| the latent / index caches (`comp_kv`, `idx_k`, `imp_*`) | `max_tokens`, legitimately — this IS the KV | ~1.5-2.6 GB per card |

So a single-call 200,000-token prefill needs over 150 GB of VRAM. The KV itself is fine: **V4.1 keeps only ~12.5 KB
per position per card**, so 200,000 positions is ~2.5 GB. **The context is affordable; the un-chunked prefill is not.**

**And the forward currently forbids the alternative.** `decode = pos0 > 0` and at `pos0 > 0` a call must be one token
(or ≤ 8 rows with `IE_DS41_DECODE_MULTI`, the DSpark path). There is no way to hand it the prompt in pieces, so the
`--prefill-chunk` the CLI advertises is ignored by this architecture — verified, no references in the V4.1 path.

**The arithmetic with chunk-bounded scratch** (chunk C = 2,048, per card): batch workspace 1.06 GiB + EP staging
0.17 GB + stream scratch ~0.4 GB + the real KV at 200k ~2.6 GB = **~4.2 GB**, against ~20 GB of experts and ~6 GB of
dense. **200,000 fits with room to spare** — this is the claim the phase is built on, and step 1 tests it directly.

## The phase, in two steps, each with its own gate

**Step 1 — make it LOAD at 200k** (the founder's literal ask, and the cheap half). Decouple the two context-scaled
scratch buffers from `max_tokens`: add `ResidentOptions::max_forward_tokens` (default 2,048, the prefill chunk), size
the expert batch workspace and the EP staging by it, and keep the latent/index caches on `max_tokens` because they
are the KV. Nothing about the forward's admission changes yet, so a prompt longer than `max_forward_tokens` is
refused with a clear message naming the limit instead of dying in an allocator.

> **DECISION RULE for step 1, fixed now:** it passes only if (a) `ie serve <dir> --ctx 200000` loads on both cards and
> `/props` reports capacity 200000, (b) every existing test's numbers are unchanged at the 2,048-token context —
> decode digits identical, decode/resident/replay/forward/generate/multi/rollback/dspark all passing — and (c) the
> pinned RAM and the static-slot counts move by less than 2 %. If the 200k load still fails after the decoupling, the
> phase reports which allocation refused and its size, and does not proceed to step 2 until that is understood.

**Step 2 — make it PREFILL at 200k** (its own criteria doc and gate). Admit `forward(ids, T, pos0)` with `T > 1` and
`pos0 > 0` as a **prefill continuation**, so the generator can feed the prompt in chunks of `max_forward_tokens`:
each chunk attends to the window ring's filled slots and the accumulated latents (`nc`) plus its own rows causally,
appends its keys and latents, and carries the ratio-2 compressor's half-pair across the chunk boundary. The P2/P3
work already proved every one of those mechanisms at T ≤ 8 rows and bit-identically (docs/55, 56); step 2 lifts them
to chunk scale using the prefill's own kernels and segment attention rather than the 8-row row-identical path.
**Digits will move** (a different grouping of the same sums), so step 2 is judged on the existing logits bar and the
same argmax, not on bit-identity — stated here, before the build.

## Pass criteria for THIS phase (step 1)

1. **`--ctx 200000` loads**, both cards, `/props` capacity 200000, and the load time within 15 s of today's 53 s at
   32,768.
2. **The refusal for an over-long prompt is clear**: a prompt longer than `max_forward_tokens` returns a message
   naming the prompt length and the limit, not an allocator failure — and the server surfaces it as a load/generate
   error rather than aborting.
3. **Nothing regresses at 2,048**: the decode test's forced-routing digits identical to today's (`5.043e-3` with the
   split on), and decode / resident / replay / forward / generate / multi (`IE_DS41_CPU_MISS=0`) / rollback / dspark
   all pass with their current counts.
4. **The memory actually moved where the arithmetic says**: report the batch-workspace and EP-staging bytes per card
   at ctx 2,048 / 32,768 / 200,000, which must now be identical across all three, and the state bytes, which must
   scale with the context.
5. **A kill switch**: `IE_DS41_MAX_FWD=N` overrides `max_forward_tokens`, and setting it to the context restores
   today's sizing exactly (so the old behaviour is one env var away).
6. **The GPU survives**: zero `xe ... Timedout job` / reset lines across the whole phase's runs, verified in the
   kernel log — the teardown drain from 8a6ae6a stays honest under the new sizes.

## Explicitly NOT in this phase

Step 2's continuation (its own doc and gate), any change to the KV's own sizing, the decode path, DSpark, the
bounded-replay design, and concurrency.
