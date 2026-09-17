# V4.1 port — Phase 10 gate criteria: Decoder SWA Bounded Replay (written BEFORE the build)

**Source:** the technical report §2.2 (CED) and §3.2.2 (Decoder SWA Bounded Replay); docs/26.
**Starting line:** the exact path, all 40 layers over every prompt token — pp2048 173.9 tok/s
cold on the held-out placement (docs/23, final chain).

## What the deployed model does, and what the engine will do

Under CED the decoder's global KV is projected from the encoder's final hidden state: in the
engine's terms, layer 20 (CSA2 Full, ratio 1) computes its latents and index keys for ALL T
prompt tokens from the stream after layer 19, and layers 21-39 reuse them. Nothing else in
layers 20-39 is needed for prompt tokens except their own sliding-window KV, which decode
reads for its first 128 steps. Deployment reconstructs that approximately: the decoder half
runs over only the last `n_win = 128` prompt tokens, "a query at position i attends to SWA
keys in [max(s, i - W + 1), i]" where s is the replay start, and the reconstructed decoder
SWA KV "is not mathematically equivalent to that from a full decoder forward pass"; the model
was post-trained with the same replay simulated and the report finds "only a negligible
impact on response quality". Prefill compute goes from O(NL) to O(NL/2 + n_win L/2).

The engine gets a prefill MODE, `bounded_replay` (default ON, as deployed; the exact path
stays and is what every existing golden test runs):

- Card 0 (layers 0-19) is unchanged: the whole prompt.
- Card 1, layer 20: the compressor path (`comp_wkv` GEMM, norm, index-key GEMM, norm, RoPE at
  every position) runs over ALL T tokens of the encoder output — the persistent latent and
  key caches hold T entries, exactly as in the exact path. Everything after it in layer 20,
  and all of layers 21-39, runs over the last `T_dec = min(T, 128)` tokens only: the hc
  stream, the pre-mix, the projections, attention (sliding keys = the segment's own rows,
  which is the truncated SWA; compressed keys = all T latents through the indexer's causal
  threshold at the query's absolute position), the MoE, the shared expert, the head.
- The decoder layers' rings end up holding exactly the segment (positions T-128..T-1 land in
  all 128 slots), so decode after a replayed prefill is the same decode as today.
- Prompts of T <= 128 tokens replay everything: the mode is then IDENTICAL to the exact path,
  bit for bit — a built-in self-check the tests use.

## Pass criteria

1. **The exact path is untouched.** With the mode off, `ie-ds41-forward-test` (both modes),
   `ie-ds41-resident-test` and `ie-ds41-decode-test` pass with the same digits as docs/23-25.
2. **Replay of a prompt that fits the window is bit-identical to the exact path.** The golden
   prompt (12 tokens) and a 128-token prompt: logits equal bit for bit, mode on vs off, and
   the decode test passes unchanged on the replayed state.
3. **Replay of longer prompts agrees with the exact path where the report says it should.**
   For five prefixes of the pp2048 text (256, 512, 1024, 1536, 2048 tokens): the next token
   (greedy) equals the exact path's, or, where it differs, the exact path's top-1/top-2
   margin is below the two paths' logit difference at that position (the Phase 9 margin
   rule); the logits' relative difference and the margin are printed for every prefix. The
   decoder's per-layer stream over the replayed segment is compared with the exact path's
   last 128 rows and printed per layer (informational: the report says these differ by
   design, so no bar — the numbers go in the results doc).
4. **The state after replay is the state decode needs.** Layer 20's latents and index keys
   equal the exact path's for all T entries (bit for bit: the same computation on the same
   input); the decoder rings hold positions T-128..T-1 at slots pos % 128 (asserted).
5. **Speed, measured on the same placement, mode on vs off:** pp512 and pp2048 cold and
   warm, the stage breakdown, and the decoder's share. The expectation from the report is
   "nearly halving" the prefill; the number is what it is, both are printed.
6. **Memory:** no new persistent allocation; the segment's scratch is a subset of the full
   pass's; free VRAM after load and the unload accounting unchanged.
7. **Decode on a replayed long prompt works end to end:** after replaying the 2048-token pp
   text, four greedy decode steps run, their tokens printed with the exact path's four for the
   same prompt (agreement expected but not required: the two paths' states differ by design;
   a disagreement is reported with both top-5 lists).

## Explicitly NOT in this phase

Encoder SWA Bounded Replay (prefix caching), chunked prefill, the candidate pool above 16K,
the FP4/FP8 caches, decode speed.
