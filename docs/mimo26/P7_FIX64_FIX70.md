# P7 — fix list #64 (engine half) and #70 (MiMo host slots)

2026-09-24. Branch `mimo26-fix-64-70` (from `deepseek4-vision-exp` ba5cf39). **Status: built and GPU-tested 2026-09-24 15:13-16:36 (section 7); the NaN found by the guard is on branch mimo26-fix-74.** The pure **E1 gate: PASS (independent evaluator, 2026-09-24 19:32; follow-ups: cost-aware one-shot eviction, fix list #77; swap-path admission counts the slot being restored, #78).**
host-side rules compile and pass their CPU unit test (61 checks; the 7/7 mutation sweep was run on the 55-check version). Sections 6 and 7 hold the GPU-window results.

| file | what |
|---|---|
| `src/server/openai_server.cpp` | #64a: one `[req] generation error:` line per error reply, streaming and not |
| `src/engine/mimo26_engine.cpp` | #64b: a drafter fault is contained; #70: the prefix step goes through the cache; host-slot options at load |
| `include/ie/mimo26_host_rules.hpp`, `src/model/mimo26_host_rules.cpp` | the pure rules: ring runs, servable prefix, continuation test, admission policy, fp16 range check |
| `include/ie/mimo26_prefix_cache.hpp`, `src/model/mimo26_prefix_cache.cpp` | #70: `Mimo26PrefixCache` (host slots, pinned bounces, save / restore) |
| `include/ie/mimo26_forward.hpp`, `src/model/mimo26_forward.cpp` | the state as device spans, `set_state` |
| `include/ie/mimo26_dflash.hpp`, `src/model/mimo26_dflash.cpp` | the drafter's state as device spans, `set_state`; the fp16 refusal in `add_context` |
| `include/ie/mimo26_engine.hpp` | the bundle owns a `Mimo26PrefixCache` |
| `tests/unit/mimo26_host_rules_test.cpp` | CPU unit test (no SYCL) |
| `tools/mimo26_cache_test.cpp` | `ie-mimo26-cache-test`, the GPU gate |
| `tools/mimo26/slot_serve_test.py` | the live A/B (verifier-then-main) |

## 1. #64 (engine half)

### 1a. Every error reply is logged

`finish_reason = "error: ..."` went to the client with nothing in `machx.log` (the server's stderr). The turn that died
at 12:59 left no trace. Both paths in `openai_server.cpp` (non-streaming, and the streaming provider) now call
`log_gen_error` first thing inside their `starts_with("error:")` branch, so the non-streaming client-image 400 is logged
as well. The line:

    [req] generation error: mimo_v2 decode: sycl: ... (chatcmpl-12; prompt 135012 tok, 134998 cached; completion 512 tok)

It is flushed at once, because `exit_on_device_loss` may end the process 2 s later. This is the common server, so every
model's `ie serve` gets the line. There is exactly one HTTP server (`run_openai_server`, called only from
`src/cli/main.cpp`), and every arch reaches it through `Engine::chat`.

### 1b. A drafter fault costs the drafter, not the request, and not the cache

Before this change every error in `mimo26_run_ids` did `b.live.clear(); b.fwd.reset();` and failed the request. That
included the DFlash drafter's own errors. So the next request re-read the whole conversation (~133k tokens at ~300 tok/s
is 8 minutes) for a fault in a component whose output the target verifies anyway. Now:

* A **drafter fault** is any error or exception from `Mimo26DFlash::draft` or `Mimo26DFlash::add_context`: a
  non-finite draft row ("no finite maximum"), a context feature fp16 cannot hold (1c), an oversize context batch, or a
  SYCL exception on its card. It is handled by `df_fail`, which:
  1. logs one line: `[mimo26 dflash] drafter fault at position P (N tokens generated): <reason> -- drafting off for the
     rest of this request, the drafter reset; the target's caches are intact`;
  2. turns drafting off for the rest of the request (`df_on = false`). Decoding continues one row at a time; prompt
     lookup continues if it is on. No partial drafts from the failed call are used;
  3. resets the drafter only (`Mimo26DFlash::reset()`). The live state, the forward and the host slots are untouched;
  4. does not fail the request. The end-of-request stats line adds `; OFF after a drafter fault at token K`.
* A **target** forward failure (prefill, decode or verify) still clears the live state, as before, because its caches
  may be inconsistent. It now also calls `cache.live_lost()`. The host slots are kept, so a slot restored for this
  request can serve the retry.
* A diagnostic knob, `IE_MIMO26_DFLASH_FAULT=N`, makes the process's N-th draft fail as a real fault would. Its only
  purpose is to let the gate drive this path inside the real engine loop.

**Why the drafter re-syncs correctly after `reset()`.** The drafter's invariant (D): its context is
`[ctx_lo, ctx_end)` with `ctx_end <= target.n_pos`, and for every position p in it, ring slot `p % R` holds the k/v of
the target's features at p. `draft()` attends only to `[max(ctx_lo, p - window), p)` with `p = ctx_end`.

* `reset()` sets `ctx_lo = ctx_end = hi = 0`. The context is empty, so (D) holds trivially. Stale or NaN-poisoned ring
  slots are never read, because no position maps to them inside the context.
* The next request's prefix step calls `rewind(L)`. With `ctx_end = 0` this is a no-op (`L >= ctx_end`), and `L == 0`
  resets again.
* Each prefill chunk calls `add_context(rows, pos0 = off + n - rows)` with `pos0 > ctx_end`. That is the **gap rule**:
  `ctx_lo = pos0`, the rows are written, and `ctx_end = off + n = target.n_pos`. (D) holds.
* Each decode step adds its kept rows at `pos == ctx_end`: contiguous, so `ctx_end` stays equal to `target.n_pos`.
* So the first draft after the prefill anchors at the right position, over a shorter context (only the last chunk's
  exported rows). The effect is lower acceptance until the context refills. It is never a wrong output, because every
  draft is verified by the target.
* `ie-mimo26-cache-test` arm 6 checks this by running: after a reset, the next prompt's prefill leaves
  `ctx_lo == 8192` and `ctx_end == target.n_pos`, and the target's logits are unchanged.

### 1c. "no finite maximum" and the fp16 question

`draft()` reports "no finite maximum" when a draft row's argmax kernel finds no element `> -inf`. That means the whole
row is NaN (or -inf). The row comes from `x16 = rms_norm(x32)`, so something upstream went non-finite. There are two
fp16 stores on the way:

1. `feat16 = half(target residual)`, the input to the fc GEMM. If |x| is 65,520 or more, the value rounds to inf. The
   fc row is then inf, the hidden norm gives inf/inf = NaN, and the k/v at that position are NaN. Every draft that
   attends to the position, which is the next ~1,024, fails.
2. The drafter's own SwiGLU store `h16`. This is the same failure mode that the vision tower hit (block 27) and that was
   fixed there.

The evidence is thin, and the hypothesis is **unverified** (the 12:59 reason was never logged). Against it: P3b's probe
found no residual or FFN value above 20,000 on an agent transcript. For it: MiMo has shown fp16 overflows twice before,
in an expert's down output (150,585) and in the vision SwiGLU (85k-144k), and today's conversation had images and ran
past 130k tokens.

**What this change does:** it refuses the feature, fast, and says why. It does not patch the numerics. `add_context`
scans the rows it is about to use (`mimo26_first_non_f16`) before touching any bookkeeping or ring slot. If a value is
NaN, ±inf, or has magnitude ≥ 65,520 (it would round to fp16 inf under round-to-nearest-even), it returns
`dflash: target layer L feature V at position P (dim D) is outside fp16's range`, and 1b contains it. In-range features
are unchanged, bit for bit. The test uses integer bit patterns, not float compares, because the host TUs build under
icpx's default fast fp model, where NaN / inf compares may fold away. The same caveat may apply to the existing
`IE_MIMO26_CHECK` probe's `std::isfinite`; that is unverified. Cost: one pass over the rows (5 × 4,096 floats per row):
~4-10 ms per 1,024-row prefill chunk and tens of µs per decode step (estimates).

**If the log shows the refusal firing**, the exact fix is a per-row power-of-two prescale:

* Rows whose max |x| is ≥ 65,520 are scaled by 2^-k before the fp16 conversion, and ctx32 is scaled back by 2^k after
  the fc GEMM, before the hidden norm. The pattern is the vision tower's P6.1b fix.
* It is exact up to fp16 subnormal loss. Rows under the threshold keep today's bits.
* Test plan: (a) in `ie-mimo26-dflash-check`, a dumped feature row with one dimension set to 1e5 gives finite drafts,
  and they equal a CPU fp32 reference within its usual tolerance; (b) the unmodified rows give bit-identical drafts
  before and after; (c) held-out acceptance does not change.

A saturating conversion (clamp to ±65,504) would be simpler, but it is inexact exactly where it acts. I did not add
either without evidence that the case occurs.

## 2. #70: MiMo host slots

### 2.1 The problem

MiMo kept one conversation: the live state, which is the full layers' K/V, the SWA rings and the drafter's ring. Any
request that did not extend it replaced it. Ten small Dream verifier requests after the owner's 135k-token turn meant
the owner's next turn re-read everything, image re-encodes included: ~8 minutes before the first token.

### 2.2 The state and its size (from `~/models/MiMo-V2.6-Flash-RL-UNCENSORED/config.json`)

| part | layout on the card | what a slot keeps | bytes |
|---|---|---|---|
| 9 full-attention layers (0, 5, 11, 17, 23, 29, 35, 41, 47), 4 KV heads | K `[4, max_ctx, 192]`, V `[4, max_ctx, 192]` (V padded to head_dim for the XMX FA-2) | rows `[0, n)` per head, V padding included | **27,648 per token** (23,040 if V were stored unpadded) |
| 39 SWA layers, 8 KV heads, window 128 | a ring of R = 2,176 slots (window + 2,048-row chunks), K `[8, R, 192]`, V `[8, R, 128]`, position p at slot p mod R | the slots of the positions the ring holds, `[max(0, written_end - R), n)`: at most two runs per head | 199,680 per ring position, **414.4 MiB when full** (any n ≥ 2,176) |
| DFlash drafter, 5 layers, 8 KV heads | a ring of Rd = 2,048 slots, K/V `[8, Rd, 128]` | the context `[ctx_lo, ctx_end)` | 20,480 per position: 20 MiB after a prefill (1,024), at most 40 MiB |
| bookkeeping | `n_pos`, `written_end`; the drafter's `ctx_lo`, `ctx_end`, `hi`; the prompt length; the use count | kept | — |
| ids | the tokens at positions `[0, n)` (image positions are negative ids) | kept | 4 per token |

Worked sizes:

| conversation | slot size | DMA alone at 20 GB/s | re-prefill it replaces (~300 tok/s) |
|---|---:|---:|---:|
| 135,000 tokens | **3.92 GiB** (4.21 GB) | 210 ms | ~7.5 min |
| 133,000 | 3.87 GiB | 208 ms | ~7.4 min |
| 8,000 (a verifier) | 0.63 GiB | 34 ms | ~27 s |
| 3,000 | 0.50 GiB | 27 ms | ~10 s |

Expected wall time is higher than the DMA figure (estimate, to be measured with `--big`). Slots live in pageable memory
and every transfer is staged through a 64 MiB pinned bounce: DMA, then a host memcpy, one wait per fill, with the first
touch of fresh pages on a save. **Estimate for 135k: 0.6-1.5 s to keep, 0.5-1 s to restore.**

Nothing else is state:

* The forward's other buffers are per-call workspaces.
* The expert tier's stream slots cache weights, not the conversation. Its CPU-leg split is history-dependent and already
  documented as such; `IE_DS41_CPU_MISS=0` makes it deterministic.
* The lookup n-gram index is rebuilt from `live` every request.
* The sampler's repeat window comes from the prompt.

**Images**: an image position is a negative id, derived from the image's bytes and its slot. Its K/V sits in the caches
like any other position's, and prefix matching compares the ids, so nothing else needs saving. A restored prefix
containing an image is not re-encoded. The request's vision provider still encodes any image position past the served
prefix.

### 2.3 Policy: `Mimo26PrefixCache::prepare`, the pure rules in `mimo26_host_rules.*`

For each request, three lengths are computed:

* `L_live` = `mimo26_servable(live, written_end, R, window, ids)`: the common prefix, capped at `ids.size() - 1`, and 0
  when the ring no longer holds the window before it. This is the exact P3b rule (`max(held, written_end) - L + window
  <= R`), now a tested function.
* `L_slot` = the best host slot under the same rule, using that slot's own ids and `written_end`.
* The decision:
  1. **Swap** if `L_slot >= L_live + 256`. The swap margin is 256: a swap moves hundreds of MiB, and 256 tokens is
     about 0.6-0.9 s of prefill. First the live conversation is kept (next item). Then the slot is restored onto the
     cards and into the drafter, and `live` becomes its ids. Source: `slot K`.
  2. Else **live** if `L_live > 0`. If this request *branches* off the live conversation, the live state is kept in a
     slot before it is cut. A branch is a request that would re-read more than 1,024 positions of its last prompt:
     `mimo26_continues(L, prompt_end, 1024)` is false. The typical case is a side request that shares only the system
     prompt and tools. A normal next turn only cuts the regenerated tail (e.g. reasoning a client does not echo), which
     is never worth a copy (V4.1's reasoning). Source: `live`.
  3. Else **none**: the live conversation is kept if worth it, and the prefill starts over.
* **Keeping the live state** happens only when the cache is on, `n >= 1,024`, and the admission policy finds room.
* **Continued vs one-shot.** Each state carries a use count: how many requests *continued* it (served at least its last
  prompt minus 1,024). A slot restored for a continuation is dropped once the prompt has run, because the live state
  supersedes it. A slot served to a branch stays.
* **Admission** (`mimo26_plan_admit`) is LRU within two tiers:
  * one-shot states (use count 0) are evicted first, least recently used first;
  * continued conversations come after, least recently used first, and only a continued newcomer may evict one;
  * the slot being restored is never a victim.
  * If the newcomer cannot fit within the budget and the MemAvailable floor even with everything it may evict gone,
    nothing is evicted. It is refused, and the refusal is logged with its reason: too large, budget held, or floor.
  * Why not plain LRU: in the #70 pattern the owner's conversation is the least recently used state the moment the side
    requests begin. Plain LRU evicts it first once the budget fills. Here, once the owner's conversation has been continued at least once, only another continued conversation can
    displace it. The CPU test and the GPU arm 4 both show this; plain LRU fails arm 4.
* **Budget and floor**:
  * `IE_MIMO26_PROMPT_CACHE_GIB` (default 16; 0 = host slots off, live reuse only) is the byte budget.
  * Outside the budget (#78 doc note, section 8): the pinned bounces, 64 MiB per queue (2 x 64 MiB on two cards; the
    drafter shares the last card's), and the #87 prompt-end snapshot (at most ~89 MiB).
  * `IE_MIMO26_CACHE_KEEP_FREE_GIB` (default 24, as V4.1) is the MemAvailable floor a new slot must leave.
    MemAvailable is read at each save, and an evicted slot's bytes count as returned: slots are large allocations, and
    `ie serve` pins glibc's mmap threshold at 256 KiB (`src/cli/main.cpp`), so they are mmap'd and unmapped on free.
  * `IE_MIMO26_PROMPT_CACHE=0` or `--no-prompt-cache` turns all prefix reuse off, as before.
  * Why 16 GiB: it holds the 135k conversation (3.92 GiB) plus ~19 verifier-size slots. At load the pinned expert tier
    leaves at least 40 GiB of MemAvailable (its own rule), and 40 − 24 = 16. MemAvailable read 108 GiB during this work
    with the owner's server up.
* **Log lines**:
  * `[mimo26 cache] kept the live conversation as slot K: N tokens (continued|one-shot), X GiB in T ms; S slot(s), Y of 16 GiB`
  * `[mimo26 cache] restored slot K: N tokens, X GiB in T ms -- it serves L of the prompt's M tokens (the live state L')`
  * `[mimo26 cache] evicted slot K (...)`, `[mimo26 cache] not keeping ... : <reason>`, `[mimo26 cache] slot K is live again: dropped`
  * In the `[gen]` line: `restore N ms from slot K` or `... from live`. `restore_ms` is the whole prefix step
    (save + restore).

### 2.4 Transfers

Every copy goes through a pinned bounce (64 MiB) allocated in **the queue's own context**, one per queue. The drafter
shares the last card's queue, so it shares that bounce. A pageable or foreign-context pointer in a SYCL memcpy is the
minutes-long stall of 2026-09-01; `deepseek41_prefix_cache.cpp` has the same rule. A bounce fill is consecutive pieces
of the spans, so it maps to one contiguous range of the slot: one host memcpy and one `wait_and_throw` per fill.

### 2.5 Failure behaviour

| event | effect |
|---|---|
| the admission policy refuses (budget, floor, too large) | logged with the reason; nothing evicted; request proceeds (the conversation will be re-read if it returns) |
| the host allocation fails (`new (nothrow)`) | logged; nothing evicted (the allocation comes before the evictions: it is an mmap reservation, so it takes no MemAvailable until the copy touches it); request proceeds without the slot |
| a SYCL error while keeping (device → host) | request fails `error: mimo_v2 prefix cache: sycl: ...` (logged by #64a); live state cleared; the other slots kept, except any this keep had already evicted |
| a SYCL error or a span-size mismatch while restoring | request fails the same way; the cards' state is undefined, so the forward and the drafter are reset and `live_lost()` is called; the slot stays for a retry |
| a target forward fails after a restore | live cleared, `live_lost()`; the restored slot is **not** dropped (it is dropped only when the prompt has run), so the retry restores it again |
| the client aborts mid-prefill after a restore | the slot is kept (no `prompt_done`); the live state holds the prefix it reached |
| a drafter fault (1b) | drafter reset only; slots untouched; a later save records the drafter's empty context, which the restore re-syncs as in 1b |

### 2.6 Why a restore is exact

`Mimo26Forward::state_spans(card, n, hi)` lists exactly the device bytes any later forward can read for positions below
`n`:

* the full layers' rows `[0, n)`: linear caches, rows past `n` are rewritten before they are read;
* each ring's slots of `[max(0, hi - R), n)`. The P3b rule already says the ring holds nothing older, and the attention
  kernels read only positions inside the window. The NaN-poisoned-cache tests of P3a hold them to that for unread and
  masked slots.

The drafter's spans are its context `[ctx_lo, ctx_end)`. A keep copies those bytes. A restore writes the same bytes back
to the same addresses, then `set_state(n, hi)` and the drafter's `set_state(lo, end, hi)` restore the bookkeeping. The
state any later forward or draft can observe is therefore byte-identical to the saved one. With the CPU expert leg off
(the only history-dependent arithmetic), the next logits are bit-identical to an uninterrupted run over the same chunks.
The gate checks this by running (arms 1, 2, 2b, 4, 5). The restored ring keeps its `written_end`, so partial matches
inside a slot obey the same window rule as the live state (arm 2b).

### 2.7 Known limits (not changed here)

* **The 2,048-token ring limit.** A prefix is reusable only when the rings still hold the window before it: at most
  R − window = 2,048 positions below the state's written end, for the live state and a slot alike. This was already
  true of the live state. If a client's next prompt diverges further back, the whole conversation is re-read even with
  host slots. Examples: dropping the reasoning of a reply longer than ~2,048 tokens, or of a whole previous task (Dream
  keeps reasoning for the current task only). The fix would be V4.1-style ring checkpoints: the 128-position window per
  SWA layer, 25.6 MB each, at every prefill chunk boundary, restored at the best one below the divergence. That is
  follow-up work. It may explain some of the owner's other 8-minute waits (unverified): the logs never said which
  path a turn took. The new `[gen]` / `[mimo26 cache]` lines will.
* **No disk entries** (V4.1 Phase 47). A new process re-reads everything.
* **V padding.** The full layers' V is saved padded (+20 % of the per-token bytes). Saving 128 of 192 columns would
  need strided copies.

## 3. GPU-window runbook

Preconditions:

* The owner has stopped the live `ie serve` on 11435 with `POST /admin/shutdown`, the orderly stop.
* The GPU check (HARD RULE 2) shows both cards free.
* Nothing else runs on the GPUs. Every GPU job goes through `scripts/ie-run-guarded`, one at a time, launched from a
  terminal in the foreground.
* Do not compile while a model is loaded. Steps marked **[GPU]** need the cards.
* In one terminal, `WT` is this worktree (the path with spaces, quoted) and `M` is the model.

### 3.1 Build (no GPU)

```bash
export WT="<the worktree path>" M="$HOME/models/MiMo-V2.6-Flash-RL-UNCENSORED"
```

```bash
source /opt/intel/oneapi/setvars.sh
```

```bash
cmake -S "$WT" -B "$WT/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DIE_ENABLE_ONEDNN=ON -DIE_SYCL_DEVICE_HINT=bmg_g31
```

```bash
cmake --build "$WT/build" -j4 --target ie ie_mimo26_cache_test mimo26_host_rules_test ie_mimo26_reuse mimo26_ops_test mimo26_model_test ds41_sampler_test
```

`-j4` because icpx has segfaulted at -j8 / -j16 on this box. A fresh build dir compiles all of ie_core. The alternative
is to cherry-pick the branch onto the main tree and build incrementally there.

### 3.2 Tests

**Step 1, no GPU:** the CPU unit test.

```bash
ctest --test-dir "$WT/build" -R mimo26_host_rules_test --output-on-failure
```

Expected: 61 `[ ok ]` lines and `MIMO26 HOST RULES: PASS`.

**Step 2, no GPU:** the test text, 2.5 MB of the engine's own code. Bit-identity does not depend on how well-conditioned
the text is.

```bash
cat "$WT"/src/model/*.cpp > /tmp/mimo26_cache_text.txt
```

**Step 3 [GPU]:** the gate.

```bash
"$WT/scripts/ie-run-guarded" --mem 220G --timeout 2400 "$WT/build/tools/ie-mimo26-cache-test" "$M" /tmp/mimo26_cache_text.txt
```

Expected: every line `[ ok ]` and `MIMO26 CACHE TEST: PASS`.

* Arms 1, 2, 2b, 4 and 5: logits FNV and 16 greedy tokens identical to their controls. With the drafter, its
  bookkeeping, drafts and probabilities are identical too.
* Arm 4: evictions happen (fewer than 6 slots) and the conversation is still restored.
* Arm 5: nothing is kept, and the result still matches.
* Arm 6: the gap-rule resync, and three refused features.
* Runtime estimate: 10-15 min.

**Step 4 [GPU]:** the drafter-off path.

```bash
"$WT/scripts/ie-run-guarded" --mem 220G --timeout 2400 "$WT/build/tools/ie-mimo26-cache-test" "$M" /tmp/mimo26_cache_text.txt --no-dflash
```

Expected: PASS.

**Step 5 [GPU]:** size and time of a 131k-token slot.

```bash
"$WT/scripts/ie-run-guarded" --mem 220G --timeout 4800 "$WT/build/tools/ie-mimo26-cache-test" "$M" /tmp/mimo26_cache_text.txt --big 131072 --ctx 135000
```

Expected: `big: ... kept in T1 ms, X MiB; restored in T2 ms`.

* X ≈ 3,891 MiB: 27,652 B/token, plus the 414.4 MiB ring, plus a 20 MiB drafter context.
* T1 and T2 around a second each (the estimate this step checks).

**Step 6 [GPU]:** the existing MiMo unit tests, unchanged.

```bash
ctest --test-dir "$WT/build" -R "mimo26_ops_test|mimo26_model_test|ds41_sampler_test" --output-on-failure
```

Expected: PASS / SKIP exactly as before.

**Step 7 [GPU]:** `ie-mimo26-reuse` on the P3b gate's inputs, as that gate ran it. Expected: unchanged. The live path's
rule is now `mimo26_servable`, the same arithmetic.

### 3.3 The server steps [GPU]

Start a **test** server on port 11440. Use the owner's usual serve flags; `--ctx` must hold the main conversation.

```bash
"$WT/scripts/ie-run-guarded" --mem 220G --timeout 7200 "$WT/build/src/ie" serve "$M" --ctx 150000 --port 11440 2>&1 | tee /tmp/mimo26_p7_B.log
```

Stop it (every time, before the next start):

```bash
curl -s -X POST http://127.0.0.1:11440/admin/shutdown
```

**Step 8:** the P3b serve test, against the server above. The pid comes from `pgrep -f "build/src/[i]e serve"`.

```bash
python3 "$WT/tools/mimo26/serve_test.py" 11440 "$(pgrep -f 'build/src/[i]e serve')"
```

Expected: 7/7 as before. The load log has `[mimo26] host slots ON: up to 16.0 GiB ...`.

**Step 9: the #70 A/B.** B is the server above, with the default settings:

```bash
python3 "$WT/tools/mimo26/slot_serve_test.py" 11440 /tmp/mimo26_cache_text.txt --main-chars 400000 --sides 10
```

Expected in B:

* `PASS`: main turn 2 cached ≥ 90 % of turn 1's prompt.
* Main turn 2's TTFT ≈ restore (~1 s) plus the new suffix.
* The log has `[mimo26 cache] kept ... restored slot K ...` and `[gen] ... restore N ms from slot K`.

Then A: shut down, and restart with host slots off (the behaviour before #70):

```bash
IE_MIMO26_PROMPT_CACHE_GIB=0 "$WT/scripts/ie-run-guarded" --mem 220G --timeout 7200 "$WT/build/src/ie" serve "$M" --ctx 150000 --port 11440 2>&1 | tee /tmp/mimo26_p7_A.log
```

```bash
python3 "$WT/tools/mimo26/slot_serve_test.py" 11440 /tmp/mimo26_cache_text.txt --main-chars 400000 --sides 10
```

Expected in A: `FAIL`, with main turn 2 cached ≈ 0 and its TTFT ≈ the full prefill (minutes). Record both TTFTs. That
pair is the before/after number.

**Step 10:** the branch rule. Default server again:

```bash
python3 "$WT/tools/mimo26/slot_serve_test.py" 11440 /tmp/mimo26_cache_text.txt --main-chars 400000 --sides 10 --shared-system 20000
```

Expected: `PASS`. The log shows the main conversation kept when the first side request shares only the system prompt.

**Step 11: #64a.** Restart with vision off:

```bash
IE_MIMO26_VISION=0 "$WT/scripts/ie-run-guarded" --mem 220G --timeout 7200 "$WT/build/src/ie" serve "$M" --ctx 150000 --port 11440 2>&1 | tee /tmp/mimo26_p7_64a.log
```

Send a 1×1 PNG, non-streaming:

```bash
curl -s http://127.0.0.1:11440/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[{"type":"text","text":"What is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}}]}]}'
```

and streaming:

```bash
curl -sN http://127.0.0.1:11440/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"m","max_tokens":8,"stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"What is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}}]}]}'
```

Expected: the client gets the error as before. The log now has two lines of the form `[req] generation error: mimo_v2
image input: vision is disabled (IE_MIMO26_VISION=0) (chatcmpl-N; prompt 0 tok, 0 cached; completion 0 tok)`.

**Step 12: #64b.** Restart with an injected drafter fault:

```bash
IE_MIMO26_DFLASH_FAULT=3 "$WT/scripts/ie-run-guarded" --mem 220G --timeout 7200 "$WT/build/src/ie" serve "$M" --ctx 150000 --port 11440 2>&1 | tee /tmp/mimo26_p7_64b.log
```

```bash
python3 "$WT/tools/mimo26/serve_test.py" 11440 "$(pgrep -f 'build/src/[i]e serve')"
```

Expected:

* 7/7, with no request failing.
* One `[mimo26 dflash] drafter fault ... dflash: injected fault (IE_MIMO26_DFLASH_FAULT)` line, and one stats line
  ending `; OFF after a drafter fault at token K`.
* Every later request drafts again (`passes` > 0 in its stats line).
* The reuse turn's `[gen]` line shows `from live`: the cache survived the fault.

## 4. Risks and open questions

* **Uncompiled code (closed 15:14: it compiled cleanly on the first try and passed the tests in section 7).** Everything outside the pure rules is uncompiled and unrun: the forward / drafter spans, the cache
  class, the engine wiring, the tool and the server line. A compile error or a logic bug may be waiting.
  "reviewed twice" is not "compiled".
* **Bit-identity rests on masked slots never entering the arithmetic.** The P3a NaN-poison tests support it, but I have
  not re-verified it for the ring + prefill path. Arms 1-5 will say.
* **The copy time is estimated** (0.6-1.5 s for 135k). `--big` measures it. The pageable first-touch cost is the unknown.
* **Defaults are proposals for the owner**: 16 GiB budget, 24 GiB floor, 1,024-token minimum and slack, 256-token swap
  margin, and eviction that is not plain LRU (one-shot first).
* **The 12:59 cause is still unknown.** #64a makes the next one visible. The fp16 refusal names an out-of-range feature
  if that was it.
* **The pre-existing 2,048-token ring limit (2.7)** can still force full re-reads. Ring checkpoints are the follow-up.

## 5. `/props` reports `prompt_cache_slots` (coordinator, 2026-09-24 15:26)

Dream decides whether a side request (its verifier, a subagent) is cheap by asking the server how many conversations it
keeps at once. `/props` now carries `"prompt_cache_slots": N`, counting the live conversation:
- `0`: nothing is reused at all (prefix reuse off: `IE_MIMO26_PROMPT_CACHE=0`; V4.1 with the prefix cache off; any arch
  with prompt caching off);
- `1`: only the live conversation's KV is reused, so a side request evicts it (MiMo with `IE_MIMO26_PROMPT_CACHE_GIB=0`;
  arches without host slots);
- MiMo with host slots: `1 + budget / one full-length state` (`Mimo26PrefixCache::guaranteed_slots`: the state bytes for
  `max_ctx` positions plus a 64 MiB margin for the drafter ring). A floor for full-length conversations, not a maximum;
- V4.1 with host slots: `2` ("at least one saved conversation"; the exact count depends on their lengths).
Code: `Engine::prompt_cache_slots()` (engine.cpp) dispatching to `mimo26_prompt_cache_slots` / `ds41_prompt_cache_slots`;
`Ds41Forward::prefix_cache_options()` added for the V4.1 read. Verified in the GPU window by `curl /props` on the test
server (section 7's serve A/B, `curl /props`): 4 at ctx 150000 with the 16 GiB budget, 1 with the budget off.

### 5a. Review of section 5 (P7 author, 2026-09-24)

The code is sound and small. Five notes, none blocking:

1. `guaranteed_slots` sizes a slot as `spans(max_ctx, ...)` plus 64 MiB with the drafter. A real slot also holds its ids
   (4 B/token, 0.54 MB at 135k). The 64 MiB margin covers them with the drafter; without it, the count can be one too
   high when the budget lands just above a multiple. Add `max_ctx * 4`.
2. "Guaranteed" and "a floor" overstate it. A new slot is also refused under the MemAvailable floor (`keep_free`), and a
   one-shot state may not evict continued conversations. So the live count can be lower than this number. "Up to"
   wording would be safer.
3. `mimo26_prompt_cache_slots()` returns 1 when prefix reuse is off. Then not even the live conversation is reused, so
   0 would be accurate, if the `/props` contract allows 0.
4. `Engine::prompt_cache_slots()` returns 1 for "the other arches". Unverified: the Qwen path has
   `core/prefix_cache.cpp`, a multi-entry trie. If `ie serve` uses it there, 1 under-reports.
   Verified by the E1 gate (2026-09-24): the crown Qwen `PrefixCache` holds up to 32 entries and the Qwen3-Next fleet
   cache 12, so 1 under-reports there -- the safe direction for a floor.
5. `ds41_prompt_cache_slots()` returns 2. V4.1's default host budget is 4 GiB, so a long conversation's slot may not
   fit, and the floor applies too. 2 is "usually", not "at least".

## 6. #74: the NaN at position 128,783 (GPU window, 2026-09-24)

### 6.1 What the window saw

`ie-mimo26-cache-test --big 131072 --ctx 135000` (coordinator's run, branch at c299dc6 + /props, CPU expert leg off)
passed its standard arms. Then the big arm's fresh prefill of the concatenated `src/model/*.cpp` text (2,561,676 bytes)
stopped with:

    dflash context at 128000: dflash: target layer 47 feature -nan at position 128783 (dim 0) is outside fp16's range

The #64 guard caught a **NaN in the target's own residual stream** after layer 47. It sat in the 1,024 exported rows
[128000, 129024) of the chunk [126976, 129024). The scan runs layers 0, 11, 23 and 35 before 47, so those were finite
over those rows. This is the shape of the owner's 12:59 failure: a server error inside a ~133k-token turn, with the
live cache cleared afterwards.

### 6.2 The leading hypothesis, from the code (to be measured, not assumed)

The expert tier's prefill route is W4A16 XMX. There, gate and up land in fp32, and then `ds4_swiglu_clamped_to_f16`
stores the SwiGLU product as **fp16** (`h_h`, `src/ops/deepseek4_ops.cpp`: `y[i] = sycl::half((g * sigmoid(g)) * u)`)
before the down GEMM. MiMo passes `limit = +inf` (no clamp). A product of 65,520 or more becomes inf. Through the down
GEMM it becomes inf, then inf − inf = NaN, and the fp32 scatter carries it into the residual whatever the routing
weight.

The port record already shows two related facts:

* A saturating version of this shared store was **reverted**. It broke DS4's IEEE-overflow parity tests, and "MiMo
  never needed it: max product 24,245" was measured on other text.
* The same pattern, an fp16 SwiGLU store, overflowed in the vision tower. It was fixed there with a per-row
  power-of-two scale.

Layer 47's MoE also produced the P2 defect (expert 70, a raw down output of 150,585).

A corollary for serving: with the CPU expert leg on (the default), an expert that misses VRAM may run on the CPU in fp32
and not overflow. Whether a turn hits the NaN then depends on the stream-slot history. Unverified: this could explain
why the failure looked intermittent, but it is unmeasured.

### 6.3 The probe

* `mimo26_scan_f32` / `mimo26_scan_f16` (pure, CPU-tested) are bit scans: the non-finite count, the first bad element,
  and the largest finite magnitude. No float compares, so the fast fp model cannot fold them away.
* `Mimo26Forward::set_probe(layer, row)` is diagnostic and off by default. It scans every op of one layer by bits:
  the input, attention norm, qkv, q/k/v, attention, o_proj, residual, ffn norm, router or the dense FFN's
  gate/up/SwiGLU, MoE or down, and output. It also keeps the watched row's MoE input and routing in
  `expert_recompute.py`'s format.
* `ie-mimo26-nan-probe` (`tools/mimo26_nan_probe.cpp`):
  * It exports **every** layer's residual for every row of every chunk and scans them, plus each chunk's last-row
    logits.
  * At the first bad chunk it rewinds exactly one chunk, which the SWA ring (window + chunk) allows. It checks the
    rerun is bit-reproducible and re-runs the chunk with the op probe on the first bad layer.
  * `--offset K` shifts the content K positions earlier, to tell content from position.
  * `--dflash` runs the serving path instead. At the guard's refusal it emulates the OLD code: the refused positions'
    drafter ring slots are set to NaN, as the unguarded fp16 conversion made them, then one `draft()` runs.

### 6.4 The runs (each command on its own; the GPU check before every model load)

`WT` = this worktree; `M` = `~/models/MiMo-V2.6-Flash-RL-UNCENSORED`; `TEXT` = the 2,561,676-byte concatenation
(md5 `c0c6969664948ff0abc6304805b496b3`); `OUT` = a scratch directory.

```bash
source /opt/intel/oneapi/setvars.sh
```
```bash
cmake -S "$WT" -B "$WT/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DIE_ENABLE_ONEDNN=ON -DIE_SYCL_DEVICE_HINT=bmg_g31
```
```bash
cmake --build "$WT/build" -j4 --target ie_mimo26_nan_probe mimo26_host_rules_test
```
```bash
"$WT/build/tests/mimo26_host_rules_test"
```
```bash
xpu-smi stats -d 0; xpu-smi stats -d 1
```
A. Localize: the first bad layer, op and row, plus the MoE row dump.
```bash
"$WT/scripts/ie-run-guarded" --mem 230G --timeout 2400 "$WT/build/tools/ie-mimo26-nan-probe" "$M" "$TEXT" --ctx 135000 --upto 131072 --dump "$OUT"
```
B. For each expert the dumped row routes to: max |gate|, |up|, |silu(gate)·up| and |down| in fp64 (CPU).
```bash
python3 "$WT/tools/mimo26/expert_recompute.py" "$M" "$OUT/moe_L<L>_pos<P>.bin"
```
C. Content or position: with the same content 4,096 positions earlier (same chunk alignment), does the NaN move to
P − 4,096?
```bash
"$WT/scripts/ie-run-guarded" --mem 230G --timeout 2400 "$WT/build/tools/ie-mimo26-nan-probe" "$M" "$TEXT" --ctx 135000 --upto 131072 --offset 4096
```
D. Other text at the same positions: does anything go non-finite near 128k?
```bash
"$WT/scripts/ie-run-guarded" --mem 230G --timeout 2400 "$WT/build/tools/ie-mimo26-nan-probe" "$M" "$TEXT" --ctx 135000 --upto 131072 --offset 262144
```
E. The serving path, and what the old code did.
```bash
"$WT/scripts/ie-run-guarded" --mem 230G --timeout 2400 "$WT/build/tools/ie-mimo26-nan-probe" "$M" "$TEXT" --ctx 135000 --upto 131072 --dflash
```

Built 16:11-16:14 in the worktree's own build dir (icpx 2026.1, Release, -j4). This was the first build of any of the
P7 SYCL-side code in this worktree: `ie`, `ie-mimo26-cache-test`, `ie-mimo26-nan-probe`. No warnings came from the
changed files. One transient icpx segfault (ScalarEvolution, in the unrelated `deepseek4.cpp`) went away on a plain
rebuild. `mimo26_host_rules_test` passes 61/61 **as built by icpx**, whose default fast fp model confirms the bit-based
checks hold there.

### 6.5 The sampled-logits guard (coordinator's decision: required whatever run A shows)

A row about to be sampled must not hold a non-finite logit:

* On such a row the greedy `std::max_element` returns token 0.
* The sampler's sort gets an inconsistent comparator (undefined behaviour).
* With the #64b containment, a NaN that spreads through the target's full-attention layers would otherwise become a
  silent garbage reply. The old code at least errored, through the drafter.

`mimo26_run_ids` now bit-checks every row it samples (`sampled_ok`):

* the final prefill logits;
* each plain decode step;
* each verify row, just before `sample_row`.

It checks sampled rows only. An intermediate chunk's last row is never sampled, and a NaN confined to it (the last
layer's own FFN at that row) poisons no cache.

On a non-finite logit it logs `[mimo26] <phase>: N of V logits non-finite at position P -- not sampled; ...`, adding
the drafter's earlier refusal when this request had one (which names the target layer and position). It then clears
the live state and the drafter and calls `live_lost()`. The request fails as
`error: mimo_v2 <phase>: non-finite logits at position P`, which the server's #64a line also logs.

Red first:

* 9ef8de0 adds only the diagnostic `IE_MIMO26_LOGITS_FAULT=N` (the process's N-th sampled row overwritten with NaN).
  That binary is kept as the red build, md5 `a19da6e37ea7c82235e4087e3b0305cd`.
* The guard is the next commit.
* The red/green runs use `ie run` with the fault at 1 (the prefill's row). Red expects a reply starting with token 0
  ("!") and no error. Green expects no reply text and the guard's line.

### 6.6 The fix if run A confirms the fp16 SwiGLU overflow (designed, not implemented until measured)

A second pass after `ds4_swiglu_clamped_to_f16`, gated by a MiMo-only `Ds41ExpertSource` flag:

* One work-group per packed row checks the row's fp16 `h_h` for inf/NaN by bits.
* A clean row is **not written at all**: bit-identical by construction.
* An overflowing row is recomputed from the fp32 `g_f` / `u_f` that are still in the workspace, times 2^-k (the smallest
  k that keeps the row's largest product under 65,520), and k is recorded per packed row.
* Before the fp32 scatter, `w_pk[p] *= 2^k[p]`. The down GEMM is linear and power-of-two scaling commutes with fp32
  rounding, so the scatter adds exactly the true row. The only loss is the fp16 subnormal rounding of that row's tiny
  products.
* The same pass covers the mmap group.
* The shift array (TK int32) is zeroed per call, so CPU-leg and int-dot rows keep 0.

Test plan:

* `expert_recompute.py` agreement on the dumped row.
* Runs A and C clean (no non-finite value).
* Normal rows bit-identical: the per-chunk logits FNV of a non-overflowing text before and after.
* The PPL gate unchanged (wikitext 8 × 2,048: 3.4749, where no product overflows).

**The decode int-dot route** (T ≤ 8) keeps gate/up **and** the SwiGLU product in fp16 (`g_h`, `u_h`, `h_h`), then
quantizes to q8_1. It is exposed the same way, and more, since gate/up are fp16 there too. It is reported, not
changed, until measured.

### 6.7 Results

**Run A** (16:37-16:46; `ie-mimo26-nan-probe`, worktree build at dfed851, CPU expert leg off, both cards idle before
the load). The text is 790,673 tokens; positions [0, 131,072) were prefilled in 2,048-row chunks with every layer's
residual scanned.

* Every chunk up to [124928, 126976) is clean: no non-finite value in any layer, and the last-row logits finite.
  The residual's max finite |x| per chunk is 450-2,700 (always at layer 47).
* Chunk [126976, 129024) has **one non-finite row: position 128,783** (row 1,807 of the chunk, token id 17), **in
  layer 47 only**. Layers 0-46 are clean over the whole chunk, and **the chunk's last-row logits are finite**
  (argmax 19,811).
* That row's residual max through the layers: 0.4 (L0-L7) → 5-12 (L13-L37) → 13-29 (L37-L45) → 45.8 (L46) → layer 47
  all 4,096 NaN.
* The layer-47 op probe (the chunk re-run after a one-chunk rewind; **reproduced bit for bit**):

  | op | block non-finite | block max finite | watched row n-f | row max |
  |---|---:|---:|---:|---:|
  | in / attn_norm / qkv | 0 / 0 / 0 | 126.6 / 35.6 / 613.1 | 0 | 45.8 / 5.9 / 594.6 |
  | q / k / v | 0 / 0 / 0 | 690.0 / 14.8 / 20.7 | 0 | 638.5 / 8.0 / 20.7 |
  | attn / o_proj / x+attn | 0 / 0 / 0 | 20.7 / 177.4 / 134.6 | 0 | 20.7 / 19.0 / 49.4 |
  | ffn_norm / router | 0 / 0 | 56.3 / 35.7 | 0 | 9.0 / 35.7 |
  | **moe** | **4,096** (row 1,807, col 0) | 1,447.2 | **4,096** | — |
  | out | 4,096 | 1,456.1 | 4,096 | — |

* The row's routing: experts 208 (0.215), 156 (0.227), 246 (0.216), 202 (0.153), 90 (0.143), **70 (0.0000)**,
  104 (0.027), 216 (0.019).

**Run B** (`expert_recompute.py`, fp64 from the checkpoint, on the dumped row):

| expert | weight | max \|gate\| | max \|up\| | max \|silu(g)·up\| | max \|down\| |
|---|---:|---:|---:|---:|---:|
| 208 | 0.2149 | 1,206.2 | 1,660.9 | 49,172.3 | 4,407.3 |
| 90 | 0.1431 | 237.0 | 401.5 | 26,211.3 | 2,069.3 |
| **70** | **0.0000** | 1,144.1 | 1,775.1 | **67,504.3 — fp16 overflow** | 238,520.6 |
| 156, 246, 202, 104, 216 | 0.019-0.227 | ≤ 239 | ≤ 564 | ≤ 7,505 | ≤ 1,600 |

The true weighted output is finite: max |y| = 1,010.0 at dim 1,933.

**Localized.** In the tier's XMX prefill route, `ds4_swiglu_clamped_to_f16` stores expert 70's
`silu(gate)·up` = 67,504 as fp16, and it rounds to inf. The down GEMM turns that into NaN, and the fp32 scatter's weight
of ~0 cannot cancel a NaN. It is the same layer, the same expert and the same bias selection (a weight below 5e-5) as
the port's P2 defect. `fp32_out` fixed the down store then; this intermediate is the other fp16 store on the same
path. Gate and up (1,144 / 1,775) fit fp16 comfortably, so only the product overflows.

What it means:

* **The target is fine (case A).** The NaN arises after layer 47's attention and layer 47 is the last layer, so no KV
  entry is poisoned and no other row's logits are touched. The chunk's last row was finite.
* Only the drafter reads the row, through its layer-47 feature.
* The old code poisons the drafter's ring there. A draft attending to it fails "no finite maximum", and the request
  errors and loses its cache. This happens if the row is in the drafter's window at draft time (the last chunk's
  export, or a decode step).
* The new code refuses the row (#64 guard). Drafting is off for that request and the output is unaffected.

**Run C** (16:47-16:56; the same text from token 4,096, so the content at text position 128,783 sits at position
124,687): **no non-finite value in any of the 131,072 positions**. Every chunk is clean, including [122880, 124928),
which now holds that content. The last-row logits are finite throughout.

So the overflow is not a property of the token alone. Dropping the first 4,096 tokens of context changes the residual
entering layer 47, and with it expert 70's product. That product was 67,504, only 3.0% over the 65,520 limit (expert
208 on the same row reached 49,172). It is a marginal, context-dependent event: rare, reproducible for a fixed input,
and it moves when the context moves. That fits the owner's single occurrence at ~133k.

**Run D** (16:57-17:06; build v2 = dfed851 plus the probe's `--watch` / `--decode-at` / per-chunk logits FNV, the tier
unchanged; other text, tokens [262144, 393216), at positions [0, 131072)): **no non-finite value in 131,072
positions**. All 64 chunks are clean, with the residual max 312-2,918 and the logits finite. Nothing about positions
near 128k produces a non-finite value by itself; **the event is content-dependent**.

**Run E** (17:06-17:15; build v2; the serving path, with the drafter on the last card, its 5 target layers exported and
`add_context` after every chunk):

* 62 chunks were added.
* Chunk [126976, 129024) was **refused**: `dflash: target layer 47 feature -nan at position 128783 (dim 0) is outside
  fp16's range`. That is the coordinator's message, bit for bit.
* The target's last-row logits were finite throughout. The same NaN appeared at the same position, and the NaN chunk's
  argmax matches run A (19,811), even though the drafter's VRAM changes the static tier. Full bit-identity is checked
  against the fixed run's per-chunk FNVs below.
* **The old code's outcome, emulated.** The same rows were added, and the NaN position's ring slots set to NaN, as the
  unguarded fp16 conversion made them. The next draft then failed with **`dflash: draft row 0 has no finite maximum`**.

**For the record: run E reproduces the 12:59 failure's signature through the old code path.** The chain is:

1. A NaN feature arises at the last layer (layer 47's MoE: expert 70's SwiGLU product of 67,504 in fp16).
2. The drafter's next draft fails with "no finite maximum".
3. The old `mimo26_run_ids` turned that into `error: mimo_v2 dflash: draft row 0 has no finite maximum` and cleared the
   live cache.

The live log could not say this at the time; #64a now logs the reason.
Likely, not verified: this is the same failure class, observed on this text. The 12:59 turn's own input is not available,
so its exact row is not reproduced.

**The sampled-logits guard, red then green** (`ie run`, `IE_MIMO26_LOGITS_FAULT=1` = the prefill's sampled row set to NaN,
`--temp 0`, the prompt "Say hello in exactly five words."):

* **Red** (17:15, the knob without the guard, 9ef8de0, md5 `a19da6e3…`): the reply printed **`!dlrow olleH`**. The NaN
  row was sampled as token 0 ("!"), the reply went on from it, and the exit was normal: **silent garbage**.
* **Green** (17:16, with the guard, dfed851 + probe, md5 `8f3f046f…`): **no reply text**, and on stderr
  `[mimo26] prefill: 152576 of 152576 logits non-finite at position 15 -- not sampled; the request fails and the
  conversation's caches are cleared`.

**PPL baseline** (17:17-17:19, the unfixed build, `ie-mimo26-ppl` wikitext-2 test, 8 × 2,048, `--static 0`, CPU expert
leg off): **3.5609** over 8,184 scored tokens, top-1 68.46 %. The per-position NLL + argmax go to `ppl_base.nll`.
The P2 record's 3.4749 was the base checkpoint; this is the abliterated `-UNCENSORED` copy.

### 6.8 The fix, as built (v3, 17:19)

It is the design of 6.6:

* `ds4_swiglu_f16_rescale_overflow` and `ds4_scale_pow2_rows` in `src/ops/deepseek4_ops.cpp`. They are new kernels;
  `ds4_swiglu_clamped_to_f16` is unchanged.
* `Ds41ExpertSource::f16_rescale`: MiMo sets it, V4.1 does not.
* `Ds41ExpertTier::row_shift_` holds TK ints. It is zeroed per XMX call and set by the rescale pass on the group and
  mmap SwiGLU rows. It is re-zeroed on CPU-leg rows, and applied to `w_pk` just before the fp32 scatter. The int-dot /
  decode route does none of this.

**DS4's own gates on the shared file** (17:20): `deepseek4_swiglu_f16_test` PASS, 408 cases on 4 devices.
`deepseek4_ops_gate_test` GATE PASSED, every negative control breached. These are the tests that caught the reverted
saturating change.

**Run A′** (17:21-17:29; the fixed build, the text and offset of run A, `--watch 128783 --layer 47`):

* **No non-finite value in 131,072 positions.**
* The layer-47 op probe on the watched row: every op is finite. `moe` has max 1,009.6 and the row's `out` max is
  1,008.8. The routing is identical to run A, expert 70 included.
* The chunk's rerun reproduced its last-row logits bit for bit.

**The fp64 agreement** (`tools/mimo26/moe_agree.py` on the fixed engine's MoE output row against the checkpoint
recompute): the engine row is finite (0 of 4,096). **Relative L2 5.8e-4, cosine 1.000000**, max |diff| 0.86 at dim
1,876 (engine 433.21 vs reference 432.35), against max |y| 1,009.98. That is the level of the fp16 activations the XMX
route feeds; the rescaled row adds no visible error.

**Bit-identity on the long text.** Run E (unfixed build, drafter on) and run A′ (fixed build, drafter off) printed each
chunk's last-row logits FNV for the same prefill. **All 63 chunks both ran are identical**, the overflow row's chunk
[126976, 129024) included; its last row is not the overflowing one. So:

* the fix leaves every non-overflowing row bit for bit as it was;
* the drafter's VRAM, which changes the static expert tier, does not change the arithmetic.

**Run C′** (17:30-17:38; the fixed build, offset 4,096): **no non-finite value in 131,072 positions**. The 64 per-chunk
last-row argmaxes are identical to run C (the unfixed build, whose probe printed no FNV).

**The decode route** (17:39-17:48; `--decode-at 128783` on the fixed build: a prefill of [0, 128783), then position
128,783 run two ways):

* **As a one-row step** (int-dot experts; gate, up and SwiGLU stored fp16): it is routed to expert 70 again (weight
  0.0000), **the MoE row is all NaN (4,096 of 4,096)**, and **all 152,576 of the step's logits are non-finite**. In fp64
  the row's expert 70 reaches max |h| **68,774** (gate 1,118 and up 1,724 fit fp16).
  * Old code: that step samples token 0 (garbage).
  * With the 6.5 guard: the request fails `non-finite logits`.
* **As row 0 of a 64-row chunk** (XMX route, fixed): finite. `moe_agree` gives relative L2 5.5e-4, cosine 1.000000.

**The decode int-dot route does reach the overflow.** The same treatment is needed there: a rescale over `ds4_swiglu_clamped_h`'s fp16 rows before `quantize_q8_1`.
Its per-32-block scale makes the q8 codes of a row scaled by 2^-k identical, and fp32_out already routes MiMo's decode
rows through the fp32 scatter, so `ds4_scale_pow2_rows` applies unchanged. This is the next change. The pinning test is
`--decode-at 128783`: the one-row step must come out finite.

**The decode fix (v4, 17:50)**: `ds4_swiglu_h_rescale_overflow` is the int-dot variant (fp16 gate / up in); the tier's
`rescale` is now on for every fp32-scatter call.

* **The pin test** (17:51-18:00, `--decode-at 128783`): the one-row step's MoE is **finite** (max 934.1), and **its
  logits are all finite** (0 of 152,576 non-finite; the unfixed build had all 152,576 non-finite). The 64-row chunk
  row is finite too.
* **fp64 agreement** of the fixed decode row: relative L2 2.1e-2, cosine 0.99978, max |diff| 39.6 of max |y| 944.
  This is the int-dot route's W4A8 precision (x and h quantized to q8_1). The rescaled expert 70 has weight ~0, so its
  row adds nothing either way. Unverified: the route's baseline on a clean row was not measured separately.
* **PPL** (18:02, the fixed build): **3.5609**, top-1 68.46 %. The per-position (NLL, argmax) file is **byte-identical**
  to the unfixed baseline's (65,472 bytes, 8,184 positions).
* **The cache test with the drafter on** (18:02-18:19, `ie-mimo26-cache-test --big 131072 --ctx 135000`): **30 ok,
  MIMO26 CACHE TEST: PASS**. The big arm that met the NaN in the GPU window now passes: 131,072 tokens kept (3,890 MiB)
  in 1,506 ms and restored in 336 ms. The drafter saw finite features all the way.

**#74 status: fixed and verified on both GPU expert routes.** Not changed:

* The CPU expert leg is fp32 and cannot overflow.
* The dense layer 0's own fp16 SwiGLU (`mimo26_swiglu_f32` → `h16`) was not seen overflowing: a 24,245 maximum in the
  port record, and no layer-0 NaN in any run here.
* A gate or up that itself passes fp16 on the int-dot route would still produce inf. It was not observed: the maximum
  here is 1,775. The sampled-logits guard would fail such a request visibly.

## 7. GPU-window results (coordinator, main tree build/ from this branch, 2026-09-24)

- Build: incremental, exit 0 in 81 s, no warnings from the new files. `mimo26_host_rules_test`: 55 ok, PASS.
- `ie-mimo26-cache-test` default: PASS, 29 ok; keep 8k tokens (0.64 GiB) in 230-270 ms, restore in 65-72 ms; arms 1-6
  bit-identical to their controls. `--no-dflash`: PASS, 19 ok.
- `--big 131072 --ctx 135000`: the standard arms passed, then the run stopped on the new feature guard: "target layer 47
  feature -nan at position 128783 (dim 0)". That is a real fp16 overflow in layer 47's expert 70 (|silu(g)*u| = 67,504),
  found and fixed on branch `mimo26-fix-74`; not a cache defect. `--big 120000 --ctx 125000`: PASS, 3.51 GiB restored in
  319 ms.
- Serve A/B (`tools/mimo26/slot_serve_test.py`, ctx 150000, main 123,741 tokens + 10 side requests of 3.6-10.1k):
  A, host slots off: turn 2 served 0 cached, TTFT 415.3 s. B, host slots on: 123,782 cached, TTFT 1.7 s; the state
  (3.63 GiB) was kept in 1,403 ms and restored in 593 ms. `/props`: prompt_cache_slots 4 (B) vs 1 (A).
- Note on the eviction rule: the owner's conversation counts as one-shot until it has been continued once; in serve B it
  was kept as "slot 1: 123782 tokens (one-shot)". With the 16 GiB budget it would be evicted by the 20th one-shot side
  state before that first continuation.
- The P7 label here names this engine fix batch; the port plan's P7 ("Dream vision up-front") is a different item.

## 8. E2: #87 (the prompt-end snapshot), #77 (cost-ranked one-shot eviction), #78 (swap admission) — 2026-09-25

Branch `mimo26-fix-77-78-87` (from `deepseek4-vision-exp` 419bbc6). **Status: built, CPU-tested; GPU verification
PENDING** (Dream was open on the owner's desktop for the whole session, so no model was loaded; section 8.6 is the runbook).

| file | what |
|---|---|
| `include/ie/mimo26_host_rules.hpp`, `src/model/mimo26_host_rules.cpp` | `mimo26_prompt_snapshot` (#87); `Mimo26SlotCost::tokens` and the cost-ranked one-shot tier (#77); `keep_leaves` (#78) |
| `include/ie/mimo26_prefix_cache.hpp`, `src/model/mimo26_prefix_cache.cpp` | the snapshot (take at `prompt_done`, restore in `prepare`); `keep_live(..., keep_leaves)` on the swap path |
| `include/ie/mimo26_forward.hpp`, `src/model/mimo26_forward.cpp` | `state_spans(..., rings_only)` |
| `src/engine/mimo26_engine.cpp` | `prompt_done` can fail (a device-to-host copy): the request fails as a failed keep does |
| `tests/unit/mimo26_host_rules_test.cpp` | 26 new checks (87 in all) |
| `tools/mimo26_cache_test.cpp` | arm 7 / 7b / 7c, `--only87`, `--ttft87 N` |

### 8.1 #87: what happened, and what is position-dependent

The owner's turn: a 74,327-token prompt, a reply cut at 16,384 tokens, then the next message. The live state then held
90,710 positions written to at least that; the next prompt shared the first 74,327. The P3b rule (`mimo26_servable`)
needs the SWA rings to still hold the 128 positions before the divergence, i.e. `written_end - L + 128 <= 2,176`: 16,511
> 2,176, so the live state served nothing. The request found no slot either, so the engine kept the live state as a slot
(90,710 tokens, 2.6 GiB, useless: its tail was the discarded reply) and prefilled all 91,640 tokens: 319 s.

Everything the next forward can read at positions below the prompt's end P, and whether a reply's decode changes it:

| state | after a reply of D tokens | needed to serve L <= P |
|---|---|---|
| 9 full layers, K/V rows `[0, P)` | untouched: decode writes rows >= P only (a verify's rejected rows too) | nothing to do |
| 39 SWA rings, slot `p % 2,176` | the slots of positions < P + D - 2,176 now hold reply positions | the 128 positions before L |
| the drafter's ring (2,048 slots) and `ctx_lo / ctx_end / hi` | overwritten the same way; `ctx_end` moved to P + D | its context `[ctx_lo, P)` for a bit-identical draft |
| `n_pos`, `written_end` | P + D, >= P + D | restored with the rings |
| the expert tier's stream slots / CPU-leg split | weights, not the conversation (the CPU leg's split is history-dependent, as before) | — |
| the lookup n-gram index, the sampler window | rebuilt from the prompt each request | — |

MiMo has no recurrent or indexer state. So a bit-identical restore of the prompt's end needs the rings' slots near P and
the drafter's context, nothing else.

### 8.2 The snapshot

At `prompt_done` (the prompt has run, nothing decoded yet) the cache copies to the host:

* each SWA ring's slots of positions `[s0, P)`, `s0 = max(P - 128 - margin, written_end - 2,176, 0)`, margin 128
  (`Options::snap_margin`); `Mimo26Forward::state_spans(..., rings_only = true)` lists them;
* the drafter's context `[ctx_lo, ctx_end)` and its bookkeeping (`Mimo26DFlash::state_spans`, as the host slots do).

`prepare` then computes a third length, `L_snap = mimo26_servable(live[0, P), hi_syn, ...)` with `hi_syn = s0 + 2,176`:
the written_end under which the restored rings hold exactly `[s0, P)`. It serves every L in `[s0 + 128, P]`, so
`[P - 128, P]` for any prompt of 256 tokens or more: a discarded reply (L = P), the generation prompt
`<|im_start|>assistant\n<think>` dropped with it (L = P - 4 or so), a regenerate (L = P - 1). When `L_snap > L_live`
and no slot beats it by the swap margin, the snapshot is written back, `set_state(P, hi_syn)` and the drafter's
`set_state(lo, end, hi)` restore the bookkeeping, and the caller's normal cut to L follows. The `[gen]` line says
`from prompt end`, and one log line:

    [mimo26 cache] restored the prompt-end snapshot: 68.8 MiB in T ms -- it serves L of the prompt's M tokens (the live state 0: N positions written to W)

Such a request re-reads at most `snap_margin` of the prompt, so it continues the conversation: the reply's state is
**not** kept as a slot (it was 2.6 GiB of nothing useful in the incident).

**Why it is exact.** The full layers' rows `[0, P)` never changed. The rings' slots of `[s0, P)` and the drafter's
`[ctx_lo, P)` are the bytes they held at the prompt's end, written back to the same addresses. `hi_syn` makes every
later rule (`mimo26_servable`, `state_spans` for a later keep) treat only `[s0, P)` as held, never the slots the reply
overwrote. Slots past P are rewritten before any forward reads them (as for every cut). So every forward and draft
after the restore reads the bytes it read at the prompt's end, and with the CPU expert leg off the logits are
bit-identical to the prompt-end state cut to L; arm 7 checks this against exactly that control.

**Validity.** The snapshot belongs to the live conversation while the cards hold its prompt's rows: it is dropped at a
cut below P, a swap to a slot, `live_lost()`, a fresh start, and replaced at the next prompt's end. "Expire when the
conversation continues past the reply" is the last of these: the next prompt's `prompt_done` replaces it.

**When none is taken:** host slots off (`IE_MIMO26_PROMPT_CACHE_GIB=0` or prefix reuse off: no pinned bounces), a
prompt under `min_tokens` (1,024; re-reading it is cheap), linear SWA caches, or `hi_syn` past the capacity -- then no
reply the capacity allows can push the rings that far, and the live state itself serves `[P - 128, P]` (checked
exhaustively in the CPU test).

**Memory: constant, outside the budget.** The snapshot does not grow with the context:

| part | bytes |
|---|---:|
| rings: 256 positions x 199,680 B (39 layers x 8 heads x (192 + 128) x 2 B) | 51,118,080 (48.75 MiB) |
| drafter context after a prefill: 1,024 positions x 20,480 B | 20,971,520 (20 MiB); 40 MiB at most (a 2,048-position context) |
| **total** | **68.75 MiB typical, 88.75 MiB at most, at ctx 8k or 250k alike** |

It is one pageable host buffer that grows to the largest snapshot and is reused. It is **outside**
`IE_MIMO26_PROMPT_CACHE_GIB`, as are the two 64 MiB pinned bounces (one per card's queue; the drafter shares the last
card's): in all, at most ~217 MiB of host RAM beyond the budget. It is not admitted against the MemAvailable floor
either; it is smaller than the floor's margin by two orders of magnitude.

**Cost:** one device-to-host copy of ~69 MiB per request that prompts 1,024 tokens or more, inside TTFT. Estimate
10-40 ms (not measured yet; arm 7 prints it as "the prompt-end snapshot: X MiB in T ms").

### 8.3 #77: one-shot victims cheapest to rebuild first

`Mimo26SlotCost` carries `tokens` (the positions the slot holds). In the one-shot tier the admission order is fewest
tokens first, then least recently used; the continued tier keeps its pure LRU order and still comes only after the
one-shot tier, and only for a continued newcomer. So the owner's 123,782-token first turn, one-shot until its first
continuation, now outlives any number of 8k side states: a new side state evicts the oldest other side state (CPU test
"admit cost: the 20th side state ..."; before this change it evicted the first turn). Residual: the greedy order can
evict several small states when one larger one would have made room (as the LRU order could); a large one-shot state
that is never continued stays until only large states are left to evict.

### 8.4 #78: the swap path's admission count

On a swap for a continuation, the slot being restored is dropped once the prompt has run. `keep_live` now passes
`keep_leaves = cont` to `mimo26_plan_admit`, which leaves that slot out of the **budget** count (it was already never a
victim). It still counts against the MemAvailable floor, whose bytes it holds until `prompt_done`. So near a full budget
the live state is no longer refused with a log line that blamed "conversations a one-shot state may not evict". `cont`
is now computed before the keep (it was after the load; same inputs). Consequence: between the keep and `prompt_done`
the store can hold up to one slot over the budget; if the prompt fails the restored slot stays for the retry and the
next admission counts it again.

### 8.5 Verification so far (CPU)

**Test first.** The 26 new CPU checks were compiled against the header declarations with the old behaviour
(the new field and flag ignored, the snapshot rule a stub): **10 FAIL**, the rest (boundaries that the old code also
meets) pass. With the implementation: **87 `[ ok ]`, MIMO26 HOST RULES: PASS** (g++ -O1 -Wall -Wextra, and the icpx
build's ctest).

The GPU arm 7 was committed on its own first (1bcfcd8), against the unchanged cache API: built there it is the red
control.

### 8.6 GPU runbook (pending; the coordinator's window)

Preconditions before EACH run: `ps -eo args | grep -c '[d]ream desktop'` and `ps -eo args | grep -c '[i]e serve'` both
0; `xpu-smi stats -d 0` and `xpu-smi stats -d 1` memory used ~26 MiB. One run at a time, in the foreground, through the
guarded runner. Each run is sized for under ~9 minutes (an estimate: sized from section 7's rates, not measured with
these arms; `--only87` without `--ttft87` is the smallest).

```bash
export WT="<the worktree path>" M="$HOME/models/MiMo-V2.6-Flash-RL-UNCENSORED"
```
```bash
cat "$WT"/src/model/*.cpp > /tmp/mimo26_cache_text.txt
```
Binaries: `$WT/build/tools/ie-mimo26-cache-test.red` (built at 1bcfcd8, md5 b16220668074d8ce0d86ba86dd80970d) and
`$WT/build/tools/ie-mimo26-cache-test` (rebuilt at the follow-up commit that moves the snapshot allocation inside the try; its md5 is in the E2 report).

Every run uses `--timeout 600`, per the 9-minute rule. Caution: on overrun `ie-run-guarded` KILLS the run (`timeout
--signal=TERM --kill-after=20`), and a killed model run can wedge the kernel (see the killed-run notes). The limit is a
last resort, not a plan: the runs are split so each should finish well inside it. Unverified: their durations are
estimated, not measured; if run 4 (all arms) looks long, run it only in a window where a longer limit is acceptable.

1. Red: arm 7 on the pre-fix cache. Expected: arms 7/7b/7c FAIL (`served 0 ... from none`, a slot kept).
```bash
"$WT/scripts/ie-run-guarded" --mem 220G --timeout 600 "$WT/build/tools/ie-mimo26-cache-test.red" "$M" /tmp/mimo26_cache_text.txt --only87
```
2. Green, arm 7. Expected: every arm 7 line `[ ok ]` (bit-identical logits, tokens and drafter state), `served 8192 ...
   from prompt end`, the snapshot's size (~68 MiB) and copy / restore times printed.
```bash
"$WT/scripts/ie-run-guarded" --mem 220G --timeout 600 "$WT/build/tools/ie-mimo26-cache-test" "$M" /tmp/mimo26_cache_text.txt --only87
```
3. Green, arm 7 plus the TTFT at a 16,384-token prompt (the default TTFT run; `--ttft87 32768` only in a longer window).
   Expected: `ttft87: ... served 16384 from prompt end` and the fresh prefill's time beside it. The red binary with the
   same flags gives the pre-fix number, if a window allows a sixth run.
```bash
"$WT/scripts/ie-run-guarded" --mem 220G --timeout 600 "$WT/build/tools/ie-mimo26-cache-test" "$M" /tmp/mimo26_cache_text.txt --only87 --ttft87 16384 --ctx 40960
```
4. Green, the default arms (1-7, drafter on). Expected: `MIMO26 CACHE TEST: PASS`.
```bash
"$WT/scripts/ie-run-guarded" --mem 220G --timeout 600 "$WT/build/tools/ie-mimo26-cache-test" "$M" /tmp/mimo26_cache_text.txt
```
5. Green, drafter off. Expected: PASS.
```bash
"$WT/scripts/ie-run-guarded" --mem 220G --timeout 600 "$WT/build/tools/ie-mimo26-cache-test" "$M" /tmp/mimo26_cache_text.txt --no-dflash
```
`--big` (section 7: 17 minutes with the standard arms) does not fit the 9-minute rule and is not part of this runbook.

### 8.6a GPU results (2026-09-25, Dream closed, both B70s idle at 26 MiB before each run)

Binaries: red `build/tools/ie-mimo26-cache-test.red` md5 b16220668074d8ce0d86ba86dd80970d (built at 1bcfcd8); green
md5 1d20810bb7c34d8a0c50a1421647eef0 (at 9c17d6a). Runs 1-3 by the coordinator through `ie-run-guarded --timeout 570`
in the foreground; runs 4-5 by the owner in a terminal (`--timeout 2400`: the default arms exceed the tool's 10 minutes).

| Run | Command | Result |
|---|---|---|
| 1 red | `--only87` | exit 1, `FAILURE(S)`: arms 7/7b/7c "served 0 of N from none" and "1 slot(s), 728 MiB" -- red for the reason #87 describes; 4 m 48 s |
| 2 green | `--only87` | exit 0, 15 ok, `PASS`: served 8192 / 8188 / 8191 from prompt end; last logits, 16 greedy tokens and the drafter's bookkeeping, drafts and probabilities identical to the control; the snapshot 68.8 MiB taken in 38 ms, restored in 10 ms; 3 m 44 s |
| 3 green | `--only87 --ttft87 16384 --ctx 40960` | exit 0, `PASS`: after a discarded 2,708-token reply, TTFT 3,921 ms (16,384 served from prompt end) vs a fresh prefill of the same 16,884 tokens 44,837 ms (x11.4); 5 m 29 s |
| 4 green | default arms 1-7, drafter on | `MIMO26 CACHE TEST: PASS` (owner's terminal, ~15 min) |
| 5 green | `--no-dflash` | `MIMO26 CACHE TEST: PASS` (owner's terminal, 12:19-12:29): arms 7/7b/7c served from prompt end, 0 slots, logits and 16 tokens identical; without the drafter the snapshot is 48.8 MiB (51,118,080 B), taken in ~28 ms, restored in ~9 ms |

Not measured: TTFT at the incident's size (~91k tokens, where the fresh re-read took 319 s); the restore cost does not
grow with the context (the snapshot is the ring window, not the context), so the saving there is expected to be larger
-- an inference, not a measurement.

### 8.7 Residual risks

* **No GPU run yet.** Bit-identity after a snapshot restore rests on the same argument as the host slots (2.6) plus
  `hi_syn`; arm 7 is the check. Unverified until it runs.
* The snapshot costs a D2H copy per request of 1,024+ tokens (estimate 10-40 ms of TTFT, unmeasured).
* Only divergences within `snap_margin` (128) of the prompt's end are served. A client that drops more of the prompt
  (e.g. an earlier turn's reasoning) still re-reads; V4.1-style checkpoints remain the general fix (2.7).
* The snapshot is not taken with host slots off (`IE_MIMO26_PROMPT_CACHE_GIB=0`).
* A failed allocation of the snapshot buffer (`std::bad_alloc` in `prompt_done`) fails the request with
  `error: mimo_v2 prefix cache: the prompt-end snapshot's N MiB of host memory could not be allocated` and clears the
  live state, the same path as a failed copy. It has no CPU test: `Mimo26PrefixCache` needs SYCL queues and the
  forward, and the CPU test builds without SYCL; the path is two lines inside the existing try.
* #77: greedy cheapest-first can over-evict small states; a never-continued large one-shot state is evicted last. Edge:
  the order is by size, not age, so the owner's big first turn (one-shot until its first continuation) is evicted
  BEFORE a newer, larger one-shot state.
* (gate) On a swap, or a branch that keeps the live state (`keep_live`), the live conversation is kept as its
  post-reply state and the snapshot is dropped. Switching to another conversation right after a discarded long reply
  and coming back therefore re-reads it: the kept slot's rings hold the reply, not the prompt's end.
* (gate) When `prompt_done` fails (a D2H copy, or the snapshot's host allocation), the continued slot (`drop_id_`) has
  already been dropped and the live state is cleared: that conversation is re-read in full on the retry.
* #78: the store can be one slot over budget between a swap and `prompt_done` (or until the next admission if the
  prompt fails).
