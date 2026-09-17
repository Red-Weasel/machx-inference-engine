# V4.1 port — Phase 13 gate criteria: prefill continuation at pos0 > 0 (written BEFORE the build)

**Starting line (Phase 12, ab8b578):** V4.1 is usable end to end. Two things the usable path
exposed, both traced to one missing runtime feature: (1) every chat turn re-renders and
re-prefills the whole history (the third turn of the docs/32 chat carried the first two; a
2,000-token history costs ~10 s per turn at 208 tok/s), because `Ds41Forward::forward` admits
T > 1 only at pos0 == 0; (2) docs/30's next speed lever, a batched verify, needs k > 1 rows at
pos0 > 0 for the same reason. One feature serves both: **a prefill of T rows appended at
pos0 > 0 that continues the state exactly as T single-token steps would.**

## What "exactly" means, and what the runtime must do

A decode step at position p already does everything a continuation row needs: the ring slot
p % 128, the ratio-2 half-group at even p closed at odd p, the ratio-1 latent per row, the
index key, the indexer's causal threshold at the absolute position, the compressed keys from
the source cache. The appended prefill does the same for T rows at once. The parts of the
prefill path that assume pos0 == 0 and must change: the sliding attention's keys (today `segs.a`
is the chunk's own rows; an appended chunk's row at position p must also see the ring's positions
p-127..pos0-1, so the keys become the ring's live slots plus the chunk, masked by position — the
exact, not truncated, SWA); the mask kernel's positions (segment-relative, docs/28: the keys'
row indices must be offset-consistent with the positions the mask compares against); the
compressor group alignment (pos0 even and T even for the ratio-2 layers, or the odd tail stepped
as the generate loop does today); `reset_state()` no longer implied by a prefill; the replay
mode: OFF for an appended chunk shorter than the window (it is exact by construction), and an
appended chunk longer than the window takes the same transition as a first prefill does today
(the decoder's rings end up holding the last 128 positions either way).

## Scope, in order; each step measured before the next

1. **The runtime feature.** `forward(ids, T, pos0 > 0)` with T > 1 admitted; the encoder/decoder
   split unchanged; the four kv sources append T/ratio latents; the ring receives T rows at
   (pos0 + r) % 128; the sliding keys are ring + chunk with a position mask. Gate: for the golden
   prompt and the pp2048 text, `prefill(T)` versus `prefill(T-k) + append(k)` for k in
   {2, 8, 32, 128, 130 (crosses the window), 512}, judged on the per-layer state (ring slot by
   slot, latents and index keys bit for bit — they are the same kernels on the same rows) and on
   the final logits against `prefill(T)` within the bar the continuity check of Phase 12 set
   (5e-2, same argmax), AND against `prefill(T-k) + k decode steps` (which must be closer, since
   an appended chunk and k steps are the same arithmetic in a different batch shape). A forced-
   routing version of the same comparisons pins the arithmetic without near-tie flips.
2. **The chat loop reuses the state.** `Ds41Generator` (and the engine's `chat()`) keeps the
   rendered-prompt ids of the previous turn; a new turn whose rendered prompt extends the previous
   one by k tokens appends k instead of re-prefilling; a turn that does not extend it (edited
   history, a different system prompt) re-prefills. Gate: the docs/32 three-turn chat reproduces
   the same three answers at temperature 0 with the second and third turns' prefill counted in
   the log as the appended tokens only; a fourth turn after an edited history re-prefills and
   still answers correctly.
3. **The batched verify's shape, measured only.** k rows at pos0 > 0 through the MoE: the tier's
   bytes per row for k = 1, 2, 4, 8 on the 2048-token context (the union of routed experts over
   the rows — docs/30's arithmetic said nearly k times the experts; this measures it), and the
   step time per row. No drafter in this phase: the number says whether Phase 14 is DSpark, a
   prompt-lookup drafter, or neither.

## Pass criteria

1. Append == prefill on the state (bit for bit) and on the logits (5e-2, same argmax), at every
   k above, on both texts; append closer to prefill than k steps are, at every k.
2. The three-turn chat: same answers, appended-only prefill on turns 2 and 3, correct re-prefill
   after an edit.
3. Bytes per row and ms per row for k = 1, 2, 4, 8, stated with the routed-expert union count.
4. **No regression:** every V4.1 test PASS (the forward's pos0 == 0 path untouched: the decode,
   forward, resident, replay, generate, tokenizer, prompt tests), unload unchanged.
5. Nothing faked: an append whose pos0 or T breaks the ratio-2 alignment is refused with a
   message, not silently rounded.

## Explicitly NOT in this phase

The drafter itself (DSpark or prompt lookup), tool calls, the prompt cache across sessions,
the FP8-resident dense set.
