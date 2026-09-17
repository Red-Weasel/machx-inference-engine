# V4.1 port — what the technical report changes about the roadmap (2026-09-13)

Source: `~/Desktop/DeepSeek_V41_Tech_Report.pdf` (51 pages; sections 2.2-2.4 and 3.2 read in full),
plus the founder's transcript of a talk on DeepSelect. Read against what is built: prefill and
decode on both cards, gate-passed through Phase 9 (docs/22-25).

## What the report confirms about the port

- The forward the engine implements (docs/24) is the reference's `Transformer.forward`: all 40
  layers over every token, the exact path. The report's CED description matches the config
  and the code the port was built from: layer 20 (the decoder's Full-mode CSA2) projects the
  decoder's global KV from the encoder's final hidden state (`C_l = H_{L/2} W^KV_l`), layers
  21-39 reuse it, only Reindex layers (24, 28, 32, 36) run their own indexer, Reuse layers
  neither index nor compress. The engine has all of that.
- The Hierarchical Sparse Indexer (§2.3.2) is the candidate pool the engine refuses above
  16,384 compressed positions: layer 20 keeps the 2,048 highest-scoring 8-blocks, Reindex
  layers score only inside them. Needed for contexts past 16K tokens; not before.
- Single-Pass mHC (§2.4.1) is what Phase 6 measured as "consumes `pre` one sublayer later";
  the deployed Mega-mHC kernel fuses the residual update, input mixing and coefficient
  prediction with the pre-norm and the FP8 cast into one kernel. The engine runs them as
  four (`ds41_hc_mixes`, `ds41_hc_collapse`, `ds4_rms_norm`, `ds4_hc_mix`).
- Engram (§2.4.2, §3.1.3): two modules at layers 1 and 14, n-gram orders {2, 3, 4}, 8 hash
  heads, ~16M rows per head, FP8 tables and projections, "deterministic addressing enables
  embeddings to be prefetched from host memory... overlapping computation in the first
  Transformer block". The engine gathers on the host from the mapping, now on parallel
  threads (docs/23); the prefetch overlap is the next step there.

## Two divergences from the deployed model, now documented

1. **Cache precision.** The deployed model stores the main (compressed) KV as FP4 — E2M1 with
   one E4M3 scale per 16 channels, quantised AFTER RoPE, trained with QAT in post-training
   (§2.4.4) — and the SWA KV as FP8. The reference harness the port was validated against
   replaces both quantisers with identity (`tools/ds41_reference/kernel.py`: `act_quant`,
   `fp4_act_quant`), so the goldens and the engine keep both caches in fp32. The engine is
   matching an unquantised variant of the model. Effect on quality: unknown here; the report
   says the FP4 format was chosen for "only marginal performance degradation" and an outside
   evaluation (transcript) scored the same with the approximation on and off. Effect on
   memory: 2 KB per compressed entry now against 288 B in deployment — irrelevant at 2K
   context, decisive at 1M. **Decision for the founder**: keep fp32 (more precise than
   deployment, larger cache) or implement the FP4/FP8 caches (deployment-faithful, 7x smaller,
   needs a quantised golden to validate against, which means giving the harness real
   quantisers).
2. **Decoder SWA Bounded Replay** (§2.2, §3.2.2). In deployment the decoder half runs over
   only the last `n_win = 128` prompt tokens at prefill: layer 20's global KV needs only the
   encoder output (available for every token after layers 0-19), and the decoder's own SWA
   states for the prompt are reconstructed approximately by replaying the last 128 tokens
   ("a query at position i attends to SWA keys in [max(s, i-W+1), i]"). This "nearly halves
   total prefill computation", is "not mathematically equivalent to a full decoder forward
   pass", has "only a negligible impact on response quality", and the model was post-trained
   with the same replay simulated. The engine runs the exact path: 40 layers over every
   prompt token.

## The roadmap, re-ordered by what the report says pays

1. **Decoder bounded replay as a prefill mode** (default on, exact mode kept for correctness
   tests). At pp2048 the decoder half is ~half of the MoE (layers 20-39 own 20 of 40 layers'
   experts) and half of the dense body; replaying 128 tokens instead of 2048 removes almost
   all of it: the pass would drop from ~22.6 s to roughly 12-13 s (estimate, not measured).
   Correctness contract: greedy tokens equal the exact path's on the golden prompts, logits
   within a stated bound, and the same decode test passing on the replayed state.
2. **The floating ~250 ms per layer** in the pp2048 profile (docs/23, runs O-R): the attention
   kernel executes in 8 ms cold and warm, yet the host stage takes 150 ms cold and the MoE
   stage doubles warm. Under test as this is written: whether the runtime executes host-device
   copies on the EUs (`UR_L0_USE_COPY_ENGINE`), which would make every DMA compete with every
   kernel. Whatever it is, it is ~10 s of a 22.6 s pass.
3. **Kernel count in the dense body.** The report: a Reuse-mode layer is 15 kernels at prefill
   and 11 at decode. The engine launches ~30 with a host wait after most; at decode that is
   the whole cost. Fusing the mHC chain (Mega-mHC's shape), the norm+cast pairs and the
   mask/attention/RoPE-conjugate sequence is the dense lever, after the floating cost above.
4. **Decode speed (Phase 10).** V4's GEMV expert path at M = 1 instead of the grouped prefill
   path; then DSpark (§2.4.3: a 3-block drafter with a 128-token sliding window, 5 draft
   positions per pass, a confidence head, the weights are in the checkpoint) as the
   speculative-decoding lever; then DeepSelect's threshold-filtered top-k (shuffled 1,024-blocks,
   a rising threshold, ~3% of scores ever sorted) for the indexer at long context.
5. **Long context**: the candidate pool (item above 16K), the FP4/FP8 caches (decision 1),
   the engram prefetch overlap, and only then chunked prefill across cards.

Not changed by the report: the residency design (three tiers, held-out profile), the
correctness methodology (forced routing to judge arithmetic, near-tie flips proven, negative
controls), or the Q8 grouped MoE path's floor.

## Roadmap status, 2026-09-16 07:30 (supersedes the ordering above; detail in the handoff)

Of the five items above: 4 is done as far as this hardware allows (decode kernels at their floors, docs/80; DSpark
measured a net loss twice, docs/59 and docs/80), 5's candidate pool and chunked prefill are built (docs/66-67), and
3's kernel count was measured not to matter at decode (the attention stage's gap is ~1.6 ms, docs/75). The current
order, by measured size:

1. **Prefill, DMA/compute overlap** — a 32k chunk's wall exceeds its kernels by ~3.6 s of unhidden expert streaming.
2. **Prefill at long context, the two O(NC) kernels** — the XMX attention's 64-key block walk (3.2 s of a 32k chunk)
   and the indexer score as a SIMT GEMV (0.95 s); both dominate at 224k.
3. **Founder decision: the XMX prefill expert route** (`IE_DS41_PREFILL_XMX`, +8-14 % prefill, two gates miss as
   written) — needs a perplexity gate first.
4. **Founder decision: expert parallel as a prefill-only mode** (+21 % prefill at 32k) — needs the two-device
   context's VRAM mirror fixed for the V4.1 tier first (host RAM fell to 42 GiB).
5. **Re-measure decode at 223k** with Phases 34-39 in.
6. The pre-existing FP8-mode abort in `ie-ds41-dense-cache-test`.

Decode at short and mid context is bound by host DRAM bandwidth for the expert miss bytes and at long context by
VRAM residency: neither has a software lever left on the measured record.

`docs/HANDOFF_2026-09-16_ds41-kernel-passes-and-prefill.md` is the entry point for the next session.

## Roadmap status, 2026-09-16 09:40 (supersedes the order above)

Items 1-2 of the 07:30 order are done for chunked prompts (docs/83 pipeline, docs/84 gathered attention, docs/85
candidate skip; 32k 165 -> 308-318, 223k 104 -> 319 tok/s), item 5 is measured (223k decode 4.71 tok/s), and decision 1
has its perplexity. The prefill now binds on each card's MoE (expert bytes + GEMMs). Ranked next work:
`docs/HANDOFF_2026-09-16b_ds41-prefill-pipeline.md` §6.
