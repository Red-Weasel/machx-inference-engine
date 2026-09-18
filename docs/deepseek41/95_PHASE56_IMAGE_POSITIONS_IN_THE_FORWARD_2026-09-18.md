# V4.1 — Phase 56: image positions in the forward

## The reference's three rules for an image-span token (model.py, engram.py)

1. **Its stream row** is the aligner's row (or a learned delimiter row), not the embedding table's.
2. **The engram skips it**: `NgramHashState` caches it as DEAD, so no n-gram spans it, and the engram gate is shut
   (`gate.masked_fill(~token_mask, 0)`): the row passes through the engram layers untouched.
3. **The MoE router selects by `bias_vl`** instead of `bias` (training's `noaux_tc_for_vl`); the weights stay the
   unbiased scores', normalised over the selected set.

## One encoding carries all three: a negative id

The engine gives every image position a NEGATIVE token id, a function of the image's bytes and the slot
(`ds41_image_id`, src/engine/ds41_engine.cpp). Everything downstream reads it off the id:

- `Ds41Forward::forward_impl`: a negative id takes its row from a vision span (`set_vision_span`) or asks the
  vision provider (`set_vision_provider`), else the call fails with the position named.
- `ds41_engram_hash`: a negative id blocks the look-back (the reference's DEAD); the gate kernel's output is replaced
  by its input at image rows (a device copy per run of image rows).
- The router: when the layer's rows hold image positions, a second `ds4_router_topk` pass with `bias_vl`, its rows
  taken at the image positions, host and device index/weight arrays kept equal.
- **The prefix cache needs nothing new**: the longest common prefix over ids now stops at the first differing image
  (different bytes -> different ids), and the same image at the same place is the same ids -- a cached turn with an
  image reuses the image's KV and never re-encodes it.
- The repeat penalty already skipped ids outside the vocabulary.

A text-only prompt has no negative id, so every one of these branches is dead for it.

## Gates

| gate | result |
|---|---|
| `ie-ds41-vision-test` engram check (golden: `tools/ds41_reference/golden_engram_mask.py`, 163 tokens, 4 image spans, a prefix + continuation split inside a span) | 0 of 7,824 hashes differ |
| `ie-ds41-decode-test` | 28/28 (a first run tripped the per-step VRAM bound at 77 MiB after an unattended-upgrade of desktop packages at 06:56; the rerun held 35 MiB; the historic range is 0-13) |
| `ie-ds41-cache-test` (CPU miss split off) | PASS |
| `ie-ds41-generate-test` | PASS |
