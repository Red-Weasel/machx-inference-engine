# V4.1 port — Phase 7 gate criteria: the layer loop (written BEFORE the build)

**Scope:** the engine's full text forward for a prefill chunk — embed, 40 layers with every
per-layer kind, engram at 1 and 14, the shared caches between source and consumer layers, the
final collapse, norm and head — checked **layer by layer** against the full-model golden
(`golden_model.py`, `m_layer_out_{L}`), ending at the logits and the first token.

This is a **correctness** phase. Weights are streamed per layer and freed; nothing is resident
across layers; there is no attempt at speed. The residency planner is Phase 8.

## What the loop must do, from `Transformer.forward` (model.py:1242-1272)

```
hashes  = engram_hash(ids)                    # both engram layers, before layer 0
h       = embed[ids] expanded to hc_mult copies
pre_mix = one-hot on copy 0
for L in 0..39:
    if L in {1, 14}:  h = engram_L(h, hashes[L])
    h, pre_mix = Block_L(h, pre_mix)          # attention kind by L; MoE
h       = collapse(h, pre_mix)                # the LAST layer's ffn_pre
logits  = head(norm(h))
```

Per-layer attention kind (all read from config, `Ds41LayerKind`):

| L | ratio | compressor | indexer | reads |
|---|---|---|---|---|
| 0, 1 | 0 | — | — | window only |
| 2, 8, 14 | 2 | wkv + gate + norm, pooled | wq_b, weights_proj, **wk + k_norm** | — |
| 3-7, 9-13, 15-19 | 2 | — | — | the latest source's compress_kv, index_k and top-k |
| 20 | 1 | wkv + norm (plain) | with wk + k_norm | — |
| 24, 28, 32, 36 | 1 | — | wq_b, weights_proj (own top-k) | layer 20's compress_kv and index_k |
| 21-23, 25-27, … | 1 | — | — | the latest index source's top-k, layer 20's caches |

`shared_attn` semantics (model.py:1166): one slot each for `compress_kv`, `index_k`,
`topk_idxs`, `candidates`; every source writes before its consumers read.

## Pass criteria

1. **Every layer's output stream matches the golden**, `[T, hc_mult, dim]`, at a tolerance that
   *grows* with depth as fp16 error compounds but is justified per layer: **5e-3 for layers
   0-19, 1e-2 for layers 20-38**, and layer 39 judged on the *normed* output (its raw stream
   reaches absmax 468 in the golden; the norm is what the head sees). A layer that steps out of
   the trend of its neighbours must be explained, not tolerated.

2. **The engram outputs at layers 1 and 14 match** (`m_engram_out_1`, `m_engram_out_14`) at
   5e-3, checked *before* those layers' blocks run — so an engram defect is attributed to
   engram, not to the block.

3. **Every layer kind is exercised and attributed.** The test prints the kind per layer and the
   per-layer error, so a consumer-layer defect (reading the wrong cache) shows at the first
   consumer, and a ratio-1 defect at layer 20, not as a blur at the logits.

4. **The logits match** at 1e-2 relative over `[T, vocab]`, and **the top-1 token at the last
   position is " Berlin"** with the same top-5 *set* as the golden. A different top-1 fails the
   phase regardless of the relative error.

5. **The engine uses its own routing**, `ds4_router_topk`, not the golden's indices. The test
   asserts the engine's per-layer routed sets equal the golden's for layer 0 (whose input is
   exactly the embedding, so no accumulated error can excuse a mismatch there), and reports
   the count of differing sets across all layers as information.

6. **Per-layer residency accounting.** For each layer the test reports experts uploaded (the
   union of the routed sets), bytes moved, and free VRAM before and after; **free VRAM at the
   end equals the start** within 128 MiB. This is the number the residency planner is built
   from.

7. **The candidate top-k is a no-op at this length and the test says so.** Layer 20 would
   build candidates over `2048 * 8 = 16,384` compressed positions; at T = 12 every block is
   kept. The test prints this and the doc records it as a **named gap for T > 16,384 with
   ratio 1**, not as implemented.

8. **No new kernel** beyond a trivial embedding gather / head GEMM if one is needed; the diff
   is the runtime, the test, CMakeLists and docs. Any kernel added is named with the reason.

## Explicitly NOT in this phase

Decode (start_pos > 0, ring buffers, partial compressor groups, the `-1e30` floor); residency
across forwards; the NVMe tier; performance; images; MTP/DSpark.

## AMENDED AFTER THE FIRST RUN — disclosed

The first run: layers 0-6 at 7.5e-5 to 4.7e-4, then **layer 7 at 3.4e-2** and everything after
it over tolerance, logits rel 0.19, top-5 set differing — with the top-1 still " Berlin". Before
changing anything, two diagnostics were added: per-layer routing dumps from the golden (indices,
weights, and each token's 6th-vs-7th score margin), and a `force_routing` mode that substitutes
the golden's routing for the engine's router.

- **Forced routing: all 40 layers pass** (worst 1.46e-3), logits rel **1.07e-3**, top-5
  identical in order. The arithmetic of the whole forward is correct.
- **Own router:** the first routing difference is at **layer 7, one token, at a golden margin
  of 6.87e-5** — inside the ~5e-4 stream error accumulated by then. One near-tie flip, then
  compounding. Layers 8 and 9 still route identically; flips accumulate from layer 11.

So criterion 1 as written — every layer within tolerance with the engine's router — was
impossible for the same reason 6b's C4 was: a discrete selection over inputs that differ by the
fp16 floor will flip near-ties, and the reference's own fp8/fp4 kernels would flip the same ones
against this fp32 golden. The amendment splits the criterion into what each mode can prove:

1. **Forced routing is the arithmetic contract**: every layer within its tolerance, logits at
   1e-2, top-5 set equal. (Passes: 0 layers over, 1.07e-3, top-5 exact.)
2. **Own router**: the per-layer bar applies *up to the first flip*; the first flip must be a
   near-tie (golden margin under 3e-3 in sqrtsoftplus+bias units, ~6x the accumulated error);
   the top-1 token must be the golden's. Later layers are reported as informational, because
   after a flip the golden's routing was computed on a different stream and the two runs are no
   longer the same computation. (Passes: first flip at layer 7, margin 6.9e-5, top-1 " Berlin".)

### Correction from the gate: the band's arithmetic

The amendment justified the 3e-3 band as "~6x the accumulated error", comparing a margin in
sqrtsoftplus+bias *score* units against 5e-4 of *stream-relative* error — different units. The
gate measured the conversion directly (64 trials of N(0, (5e-4·rms)²) noise on layer 7's gate
input): the 6th-vs-7th selection margin moves by **median 5.5e-5, p99 2.35e-4, max 3.4e-4**.
Layer 7's golden margin of 6.87e-5 sits *below the median* displacement — the flip is the
expected one, not merely an explicable one — and 3e-3 is ~9x the p99, conservative. The
conclusion stands; this is the arithmetic that supports it.
