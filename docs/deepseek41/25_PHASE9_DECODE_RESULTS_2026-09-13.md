# V4.1 port — Phase 9 results: decode on the resident runtime

**Criteria:** `19_PHASE9_DECODE_CRITERIA_2026-09-12.md`. **Design:** `24_PHASE9_DECODE_DESIGN_2026-09-13.md`.
**Test:** `ie-ds41-decode-test <model> <golden> <golden>/decode2 [ranking]` (`tools/ds41_decode_test.cpp`).
**Golden:** `golden_decode.py` extended to dump the post-prefill state (`d_pre_*`), the hashes as
integers (`p_hashes`, `d_hashes`, `s{k}_hashes`), and the router's decision per layer for the
prefill (`p_route_*`) and every step (`d_route_*`, `s{k}_route_*`), into `scratchpad/golden/decode2/`
(reference on CPU: prefill 337 s, steps 55-95 s each on a warm page cache).

## What was built

`Ds41Forward::forward(ids, T, pos0)` with per-card, per-layer state carried across calls: the
window ring (`slot = pos % 128`, RoPE'd rows), the compressed latents and index keys of
kv-source layers, the ratio-2 compressor's open half-group, and the token sequence for the
engram look-back. Two regimes: a prefill at `pos0 == 0` (fills the state) and a single-token
step at `pos0 == n_pos()`. The layer body is the Phase 7/8 body with the decode branches
inlined: the ring attended as one segment with a slot mask, the compressed segment read from
the source layer's persistent cache, the indexer's top-k over `nc` keys for one query, the
ratio-2 group opened at an even position and closed (pooled, normed, RoPE'd at `pos + 1 -
ratio`) at the odd one, the ratio-1 latent appended per token. `prefill` is `forward` at 0.
The state is 25 MiB per card at the 2048-position capacity.

## The runs (2026-09-13, two cards, held-out placement, golden prompt of 12 tokens)

**Run 1 (against the first golden set, no state/hash/routing dumps):** every decode-only
mechanism works on its first run — the four greedy continuation tokens equal the golden's
(" Berlin. The capital of"), step 0's top-5 exact, per-step VRAM flat (0 / 1 MiB held after
four steps). Per layer at step 0: 4.0e-4 at layer 0 rising to 7.7e-3 at layer 9 — then 5.3e-2
at layer 10, compounding to ~0.2 by layer 21. The negative control (RoPE position off by one)
puts all 40 layers over their bars, worst 0.82, so the small numbers through layer 9 mean
something and the jump at 10 needed an explanation.

**Run 2/3 (against `decode2`, routing dumped):** the explanation is Phase 7's: the engine's
own router flips a near-tie at layer 10 of the step (golden 6th-vs-7th margin 2.1e-3), and the
prefill's own-router flip at layer 7 (margin 6.9e-5) had already carried into every ring row
and latent of the later layers — which is why layer 20's state was 10% off when the prefill
ran with the engine's routing and matches when the prefill runs with the golden's.

| criterion | result |
|---|---|
| 1. state at the boundary (forced prefill) | layer 0 ring 3.7e-5; layer 2 ring 1.9e-3, latents 2.6e-3, keys 8.5e-4 (bar 3e-3); layer 20 ring 8.6e-3, latents 6.8e-3, keys 5.4e-3 (bar 1.2e-2: the Q8 grouped path's floor at depth, Phase 8's card-0 forced bar, stated in the test); 6 and 12 latents held; prompt hashes exact. **PASS** |
| 2. step 0 layer by layer | **routing forced** (prefill and step): all 40 layers within their bars — card 0 worst 9.4e-3, card 1 worst 1.6e-2; engram outs 4.0e-4 / 2.7e-3; logits 7.5e-3. **Own router**: judged to the first flip at layer 10 (a near-tie, margin 2.1e-3) — worst 7.7e-3 over layers 0-9. **PASS** |
| 3. step 0 token and top-5 set | "." (16), top-5 set equal. **PASS** |
| 4. N = 4 greedy steps | all four equal the golden's (margins 2.95, 0.35, 4.18); no divergence to excuse. **PASS** |
| 5. negative control | position off by one: 40 layers over, worst 0.82. **PASS** |
| 6. hashes as integers | prompt, step 0, steps 1-3: 0 differ. **PASS** |
| 7. no prefill regression | `ie-ds41-forward-test` own-router and forced PASS on the rewritten forward (flip at 7, margin 6.87e-5, " Berlin"; forced logits 1.269e-3); resident test PASS with the same digits (6.124e-3, flip 7, forced 9.02e-3 / 2.76e-2). **PASS** |
| 8. memory bounded per step | free VRAM after four steps within 0 / 1 MiB of before (the mmap tier's transient bank is persistent and grow-only since run K). **PASS** |

Bars, and where they come from: criterion 1's 3e-3 holds at layers 0 and 2 as written; at
layer 20 the state is the output of 20 layers of the Q8 grouped MoE and sits at that path's
floor (Phase 8 measured 8.2e-3 at layer 20 with forced routing), so the test uses Phase 8's
card-0 forced-pass bar 1.2e-2 there and says so. Step 0's per-layer bars are Phase 8's
forced-pass regression bars (1.2e-2 to layer 19, 3.5e-2 after); the logits bar 5e-2 likewise.
The measured values sit well under them (worst 1.6e-2 on card 1, logits 7.5e-3).

## Gate verdict (2026-09-13 08:51): PASS

A fresh evaluator rebuilt at `df32888`, reproduced every digit of the decode test, both
forward-test modes and the resident test, and went further with its own probe (linked against
the built library, repo untouched): steps 0-3 with the routing forced, judged per layer against
the golden's `s{k}_layer_out_*` and the state after each step against `d_step{k}_*` — dumps the
shipped test never reads. Its findings, all non-blocking, and what they mean:

- The ratio-2 CLOSE branch (positions 13 and 15) is correct per layer: closing latent 2.7e-3,
  seven latents after step 1, 0 layers over at steps 1 and 3. Step 2 (position 14, " The")
  reaches 1.4-1.8e-2 at layers 17-19 on the resident (Q8) path, over Phase 8's 1.2e-2 bar,
  while the fp16 streaming path meets Phase 7's tighter bars at every step (worst 3.25e-3 /
  1.16e-3 / 1.36e-3 / 9.1e-4; state at layer 20 5.96e-4). So the resident excesses are the Q8
  floor and Phase 8's bar came from one 12-token prefill; a bar for per-layer judging of later
  steps on the Q8 path needs more tokens behind it (finding 5). The shipped test covers steps
  1-3 by tokens and hashes only (finding 4) — stated in the test now.
- Criterion 1 at layer 20 is met at 8.55e-3 against the test's 1.2e-2 (disclosed), not the
  criterion's 3e-3, which the Q8 path cannot reach at depth by construction; the fp16 path
  meets 3e-3 there (finding 1). Criterion 2's bars are Phase 8's, stated (finding 2).
- Not exercised: a full or wrapped ring (positions >= 127), a prefill longer than the window
  reaching decode, an odd-length prompt (refused), a ratio-2 group across a card boundary
  (cannot occur at the 20/20 split: layer 20 is a source). The control shifts q/kv RoPE only;
  ring slot, mask and top-k positions are not shown load-bearing (finding 10).
- Criterion 8 is measured as net retained VRAM over six forwards (0 / 1 MiB), not per-step
  composition (finding 11).
- `all_ids_` was appended before the step ran, so an errored step would have desynced the
  engram window (finding 12) — fixed after the gate: rolled back unless the call succeeds.
- The streaming forward test sits 0.3 MiB under its 128 MiB teardown bar because the decode
  state (~35 MiB) is freed in the destructor, after the check (finding 13).

## What decode costs today (not this phase's criterion; the number is what it is)

A step on the resident runtime takes ~0.48 s at position 12 (the resident test's warm golden
forward is 1.2-1.5 s for 12 tokens). Nothing in the step is tuned: the MoE runs the Q8 grouped
prefill path at M = 1 row per expert, the six routed experts per layer that are not in VRAM are
fetched from the pinned arena or read from disk, and every op in the dense body launches with
a wait. V4's decode path (the GEMV route, the fetch pipeline at decode sizes) is the template
for Phase 10.

## Files

`include/ie/deepseek41_forward.hpp`, `src/model/deepseek41_forward.cpp` (forward with state;
prefill as a wrapper; read_state / last_hashes / set_rope_offset_diagnostic / reset_state),
`tools/ds41_decode_test.cpp`, `tools/CMakeLists.txt`, `tools/ds41_reference/golden_decode.py`.
