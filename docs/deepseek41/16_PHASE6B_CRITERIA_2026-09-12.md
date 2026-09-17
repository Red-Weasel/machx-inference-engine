# V4.1 port — Phase 6b gate criteria: compressed attention (written BEFORE the build)

**Scope:** the attention block of a **source** layer with `compress_ratio == 2` — layer 2, the
first of them — against the reference's own `Attention(2)`. This is the layer type that 18 of
the 40 layers are, and it adds every mechanism layer 0 lacks: the compressor, the indexer, the
shared index key, and attention over window + compressed positions.

## What changes at layer 2, read from `model.py:429-612, 700-790`

- **Compressor** (`Compressor.forward`): `kv = wkv(x)`, `score = wgate(x)`, both `[T, 512]`;
  group consecutive `ratio` tokens; `kv = (kv * softmax(score over the group)).sum` — the
  softmax is over the slot axis **independently per channel**; then `kv_norm`. No position
  bias tensor exists in V4.1 (the tensor census has no `compressor_ape`), unlike V4.
- **Index key** (`Indexer.forward`, owner branch): `k = k_norm(wk(latent))` on the RoPE-free
  latent, then RoPE at group positions `j * ratio` using the **compress** table
  (theta 160000, YaRN factor 16 over 65536).
- **Index score**: `sum_h relu(q_h . k) * weights_proj(x)_h * (index_head_dim^-0.5 * n_heads^-0.5)`,
  where the indexer's `q = wq_b(qr)` is a `[32 x 128]` projection of the SAME `qr` attention
  uses, RoPE'd on its trailing 64.
- **Causal mask**: compressed position `e` is visible to query `t` iff `e < (t+1) // ratio` —
  a group is visible once the query has passed its last token.
- **Top-k**: `min(index_topk=512, n_compressed)` by score, re-sorted into position order,
  unreachable → −1, reachable shifted by `offset` (the window's kv count) because the attention
  concatenates `[window_kv, compress_kv]`.
- **Attention**: `sparse_attn` over the concatenation, with the window indices from
  `get_window_topk_idxs` and the compressed indices above. Same sink, same conjugate rotation,
  same block-diagonal `wo_a` as layer 0.
- **Two-level candidate top-k** is owned by layer 20 and consumed by 24/28/32/36. Layer 2 is
  neither, and at any length below `2048 * 8` compressed positions the mask is all-true. Out of
  scope here; recorded so it is not forgotten at long context.

## The trap this phase is designed around

`topk = min(512, T // 2)`. **Below 1,024 tokens every compressed position is selected**, and
the indexer's ranking has no effect on the output at all: a test at T=64 would pass with the
scores replaced by zeros. This is the same trap `deepseek-v4-flash-status` recorded for V4
("a 512-token PPL cannot test an indexer change"). So:

1. **T = 1,536** tokens. Compressed length 768 > 512, so 256 positions per late query are
   *excluded* and the selection matters.
2. The test asserts, from the golden, that for the last query fewer than all reachable
   positions were selected — if that assertion fails the test is vacuous and must fail.

## Pass criteria

1. **Compressor output (RoPE-free latent) matches** at 3e-3 (two fp16 GEMMs + softmax pooling).
   Checked BEFORE RoPE and quantisation, on the latent the indexer consumes.
2. **Index keys match** at 3e-3 after `wk`, `k_norm` and RoPE at group positions.
3. **Index scores match** at 3e-3 on the `[T, 768]` score matrix, with the causal mask
   compared as a *pattern*: every `-inf` in the golden is `-inf` (or masked) in the engine and
   vice versa, exactly.
4. **Selected indices match as sets.** The engine emits descending-score order, the reference
   position order; both are sorted and compared **as integers, per query, for every query**. A
   discrete selection has no tolerance. Ties: the reference's `topk` tie-break is unspecified,
   so a mismatch is only accepted if the two differing indices have golden scores within 1e-6
   of each other, and the test must report how many such tie-cases it excused (expected: few
   or none).
5. **Attention output matches** at 3e-3 through the concatenated window + compressed KV, and
   the block output at 5e-3.
6. **The RoPE tables are the right ones.** Layer 2 uses the compress table for everything
   (theta 160000, YaRN); the test asserts that using the main table (theta 10000, no YaRN)
   instead fails criterion 2 — i.e. demonstrates the choice is load-bearing at T=1536.
7. **Reuse is real.** Kernels used: `ds4_compress_pool` (bias = 0, no overlap),
   `ds4_indexer_score`, `ds4_indexer_topk`, `ds4_block_bias_topk`, `ds4_attention_segs`,
   `ds4_rope_*` with the YaRN config, plus 5b's set. Any kernel added must be named in the
   result doc with the reason the existing one did not fit.

## Explicitly NOT in this phase

Consumer layers (3-7 read layer 2's caches); ratio-1 layers (20-39); the candidate top-k;
decode (start_pos > 0, the ring buffer, the partial-group state); the MoE half (already proved
at layer 0 and unchanged); engram (6a).

## AMENDED AFTER THE FIRST RUN — disclosed

The first run failed criteria 4 and 5: **30 of 1,536 queries** selected a different top-512
set, and the attention output landed at 7.1e-3 against 5e-3. Everything upstream matched
(scores 3.85e-4, mask pattern identical, keys and latent ~2-3e-4). Two diagnostics were added
before anything was changed:

- the worst golden-score gap between any swapped-in and swapped-out entry was **4.48e-4 of
  the query's score scale** — inside the measured 3.85e-4 score error. Every one of the 30 was
  a boundary flip between numerically tied entries.
- the output error over queries whose selection matched was **3.96e-4** — the same fp16 floor
  as layer 0. The entire excess sat in the 30 differing queries.

So the *criterion* was wrong, not the code: an exact set match with ties at 1e-6 is an
impossible bar for a discrete top-k over scores computed at different precisions. The
reference's own fp8/fp4 CUDA kernels would diverge from this fp32 golden in the same way.

Criterion 4 now reads: **selections may differ only on entries whose golden scores are within
3x the measured score error (of that query's scale) of each other**, and fewer than 5% of
queries may need that excuse; a swap beyond the band is a defect. Criterion 5 now applies the
5e-3 bar to queries with matching selections and a separate 2e-2 bound to the boundary-tie
queries, and reports both. The diagnostics stay in the test, so the numbers that justified
the amendment are printed on every run. Result after amendment: 0 queries beyond the band,
30 excused (2.0%), same-set output 3.96e-4, boundary-tie output 7.07e-3.

## Second amendment, from the Phase 6 gate: the excuse became a proof, and the band is fixed

The gate accepted the amendment as principled (its own analysis of the golden scores found 126
queries with a 512th-vs-513th margin inside the measured error and the engine flipped only 30,
none out of band) but asked for two things. Both done.

1. **The band no longer scales with the error it excuses.** It was `3 x g_c3_rel`, where
   `g_c3_rel` is the *measured* score error — so a regression that degraded the scores would
   have widened its own excuse (the gate showed a 2.5e-3 perturbation excusing 75 queries while
   still passing). It is now the absolute `3 x 3.85e-4` recorded here, is void when C3 itself
   fails, and the swapped entries are paired by golden score rather than by index.
2. **The 30 boundary-tie queries are proved, not excused.** `prove_ties.py` re-runs the
   reference's own `sparse_attn` with the **engine's** selected indices: over the 30 queries the
   reference's output moves by **7.077e-3** — the engine's error there was 7.074e-3 — and by
   2.5e-7 over the other 1,506 (fp32 summation-order noise). The entire excess is the selection
   itself, in the reference's own arithmetic; the engine's attention over that selection is at
   the fp16 floor.
