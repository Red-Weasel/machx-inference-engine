# Hy4-flash routed experts: how to store them (open question, 2026-11-18)

Working notes for anyone reviewing the expert-format decision. Everything marked MEASURED was measured on Tencent's
released BF16 weights today; everything marked ESTIMATE or OPEN is not.

## The problem
- Model: Tencent Hy4-flash, 42 layers, 41 of them MoE. Per MoE layer: 512 routed experts, 16 picked per token (sigmoid
  router + bias, weights normalised and scaled by 3.998).
- An expert works in a LATENT space: `x = D . h` (shared `routed_expert_down_proj`, 3072 -> 1536, no norm after it),
  `y_e = W2_e . ( silu(min(W1_e x, 10)) * clamp(W3_e x, -10, 10) )` with W1, W3, W2 all 1536 x 1536, then
  `out = RMSNorm( U . sum_e w_e y_e + shared_expert(h) )` (shared `routed_expert_up_proj`, 1536 -> 3072).
- 7.08 M weights an expert, 20,992 experts, 148.6 B weights: 297 GB as released. One token reads 656 experts.
- Machine: 2 cards x 31.9 GiB VRAM, 249 GiB RAM. Decode is bound by the bytes of expert weights read per token
  (DeepSeek-V4.1 here: 4.6 GB a token = 17.7 tok/s). The rest of the model is 12.2 GiB as fp16.
- Goal: best quality-to-speed ratio, measured; full or near-lossless quality at the fewest bits; new kernels, a
  new expert cache or a new file format are all acceptable. Do not assume the engine's current formats.

## What exists
- A 4-bit pack in MXFP4 (4-bit float, one power-of-two scale per 32 weights), 74 GiB, built because the engine's expert
  tier (`Ds41ExpertTier`) computes on that format only. Chosen for speed of bring-up, not after a comparison.
- A full-precision path in the engine (`Hy4fOptions::exact_experts`) and a format EMULATION inside it
  (`Hy4fOptions::emulate`, `ie-hy4flash-run --emulate FMT`): each expert matrix is rounded to a candidate format on the
  card, then computed in fp16. One format = one ~10 minute run of 2,048 tokens. Formats so far: mxfp4, mxfp6, fp8, int8,
  q6s, q5a, q4a, q3a, q2a, t158, b1; `--emulate-hot` gives a usage-ranked subset another format.

## MEASURED
Relative RMS error per weight, real experts (layer 10, experts 0-7), plain rounding, against the rate-distortion floor
for independent Gaussian weights at the same bits (2^-bits):

| format | bits / weight | error | floor |
|---|---|---|---|
| int8, scale per 32 | 8.5 | 0.5 % | 0.3 % |
| fp8 (E4M3), power-of-two scale per 32 | 8.25 | 2.7 % | 0.3 % |
| 6-bit integer, scale per 16 (Q6_K shape) | 6.56 | 1.9 % | 1.1 % |
| 6-bit float (E2M3) | 6.25 | 2.8 % | 1.3 % |
| 5-bit integer, scale + offset per 32 (Q5_K shape) | 5.5 | 3.8 % | 2.2 % |
| 4-bit integer, scale + offset per 16 (Q4_K shape) | 4.5 | 6.5 % | 4.4 % |
| MXFP4 (the pack) | 4.25 | 11.2 % | 5.3 % |
| 3-bit integer | 3.5 | 13.9 % | 8.8 % |
| 2-bit integer | 2.5 | 32.7 % | 17.7 % |
| ternary | 1.83 | 42.6 % | 28.1 % |
| sign | 1.25 | 59.3 % | 42.0 % |

Structure of the weights (layer 10):
- Across the 512 experts of a layer: the top 64 principal directions hold 13.4 % of the energy (independent experts:
  12.5 %); the mean expert 0.2-0.3 %. The experts share nothing.
- One expert matrix: its top 256 singular directions hold 48 % of the energy (a random matrix: 39 %); kurtosis 3.03.
  Close to Gaussian noise with a faint low-rank excess.
- Row and column norms of an expert vary by 2-4 % (pure noise: 1.8 %); two experts do not agree on which input columns
  are large (correlation -0.03).
- The shared maps D and U: singular values spread 3-10x (largest / smallest), 50 % of the energy in 35-39 % of the
  directions, effective number of directions 1,200-1,400 of 1,536 (layer 1: 840-940).

Bits per weight above are IDEALISED for the integer shapes: the emulation keeps fp32 scales and offsets; as stored
(two 6-bit fields per 16 weights) the 2-, 3- and 4-bit shapes cost 2.75, 3.75 and 4.75 bits (an outside review, 2026-11-18).

### The engine is right on the real weights (the yardstick is Tencent's reference, not perplexity)
`scripts/hy4flash_oracle/real_oracle.py` runs the reference implementation itself on the released weights on the CPU
(fp32; experts and n-gram tables read in place, cast per use; about 15 ms per (layer, expert) used: 85 tokens in 39 s).
- Full 42 layers, 85 tokens of a chat exchange, the engine with full-precision experts and the reference's expert
  picks: every layer's streams within 2e-3, logits within 2e-3, top-1 equal on 85 of 85. First 4 layers on raw text: 4e-4.
- Free-running (its own picks): 249 of 3,485 expert picks differ (the 16th and 17th candidates are close on most
  tokens: median margin 0.002), KL(reference || engine) 0.0037 mean, top-1 equal on 84 of 85. That is the engine's floor.
- Perplexity on raw wikitext after BOS is NOT a usable yardstick for this model: outside its chat format it is
  over-confident and repeats (it gave " He" 98.8 % and restarted the previous sentence), 11 % of tokens score NLL > 10,
  and two runs differ by more than the effect being measured. An earlier "4-bit is 10 % better than full precision"
  reading came from that and means nothing.

### Formats against the reference (85 tokens; emulated formats = the full-precision path with rounded experts)
| experts as | bits (idealised) | weight error | top-1 equal (of 85) | KL(reference \|\| engine) mean | worst row | expert picks that differ (of 3,485) |
|---|---|---|---|---|---|---|
| full precision | 16 | 0 | 84 | 0.0037 | 0.10 | 249 |
| int8 | 8.5 | 0.5 % | 81 | 0.0114 | 0.16 | 712 |
| q6s | 6.56 | 1.9 % | 77 | 0.0287 | 0.36 | 1,196 |
| mxfp6 | 6.25 | 2.8 % | 78 | 0.0385 | 0.61 | 1,330 |
| q5a | 5.5 | 3.8 % | 81 | 0.0404 | 0.35 | 1,570 |
| fp8 | 8.25 | 2.7 % | 77 | 0.0465 | 0.78 | 1,378 |
| q4a | 4.5 | 6.5 % | 75 | 0.0869 | 1.36 | 1,888 |
| mxfp4 (emulated) | 4.25 | 11.2 % | 71 | 0.209 | 2.14 | 2,357 |
| MXFP4 pack through the expert tier | 4.25 | 11.2 % | 71 | 0.195 | 2.21 | 2,344 |
| q3a | 3.5 | 13.9 % | 70 | 0.237 | 2.34 | 2,512 |
| q2a | 2.5 | 32.7 % | 60 | 0.885 | 11.7 | 3,100 |
| t158 | 1.83 | 42.6 % | 50 | 1.85 | 16.8 | 3,340 |
| b1 | 1.25 | 59.3 % | 26 | 3.72 | 17.6 | 3,382 |

The pack with the reference's picks forced: KL 0.144 -- most of the damage is the weights themselves, not the picks.
KL grows about linearly with the weight error. The MXFP4 pack is not usable for anything that has to reflect the model
(it misspells: "atmosfere", "plethera"). 85 tokens is a small sample; a 554-token reference is the next run.

### Firm numbers and the calibration of "lossless" (554 tokens of a chat exchange, KL to the fp32 reference / top-1 equal)
| run | KL mean | top-1 equal of 554 |
|---|---|---|
| engine, full-precision experts | 0.0100 | 535 |
| engine, int8 experts (emulated) | 0.0116 | 530 |
| **the reference itself in BFLOAT16 (how the model is served)** | **0.0195** | **525** |
| engine, q6s | 0.0258 | 518 |
| engine, q5a | 0.0268 | 517 |
| engine, q4a | 0.0599 | 509 |
| engine, the MXFP4 pack | 0.125 | 476 |

Two correct implementations of this model differ by 0.01-0.02: the 16th and 17th expert candidates are close on most
tokens (median margin 0.002), so rounding anywhere moves picks. 8-bit integer experts are inside the vendor's own noise;
5 and 6 bits just outside; the MXFP4 pack six times outside. (The bf16 figure is the reference code on a CPU, not
Tencent's serving stack: the order of magnitude, not an exact bar. On the 85-token sample it is 0.060.)
In exact arithmetic (format_lab.py, no engine in the loop, 85 tokens): 8-bit 0.0141, 4-bit integer 0.114 -- the format's
own effect is what the engine numbers show; the engine's fp16 kernels add little.

`Hy4fQ8Store` (the engine's 8-bit expert path, commit 5319058): KL 0.015 on the 85 tokens; a 110-token greedy answer
equals the reference's first choices except at 4 coin-flip positions; the misspellings such answers contain ("critial",
"committment") are the reference's own first choices.

### What the low-bit ideas depend on (reference statistics, 2,602 tokens; ~/models/hy4flash-ablit/stats/)
- The experts' INPUTS: on average half of the energy in 22 of 1,536 directions, 90 % in 403, 99 % in 1,015 (per layer:
  layer 41 has half in ONE direction). Rotation gain by the high-rate formula: about 1.1 bits a weight at equal error
  (1.36 raw, 0.27 of it a sampling artefact at 2,602 samples for 1,536 dimensions).
- The summed OUTPUTS: half in 100 directions, 90 % in 616, 99 % in 1,198.
- A token's 16 routing weights: 13.6 / 10.5 / 8.7 / ... / 4.0 % of the total; the weakest 8 still carry 22 % of the
  squared weight -- "precision by routing weight" buys little and is dropped.
- Usage: the most-used 25 % of a layer's experts take 76 % of the picks (79 % of the weight mass), the top half 93 %;
  1,203 of 20,992 experts were never picked. (A narrow sample: one article and one chat.)

## OPEN (not measured)
1. The covariance spectrum of the experts' real inputs `x` per layer (`IE_HY4F_LAT_DUMP` records them).
2. How unequally experts are used (`--route-hist`), and how unequal the 16 weights of one token are.
3. Per-unit activity of an expert's 1,536 hidden units on real inputs.

## Ideas on the table (none built)
1. Rotate each layer's latent space to the principal axes of the real inputs and fold the rotation into D (exact, free at
   run time: nothing sits between D and the experts); then allot bits per input direction, zero included. The same on the
   output side through U. Needs a format with per-column-group bit widths.
2. Precision by routing weight: a low-bit copy of every expert in VRAM, a high-precision copy elsewhere, the latter used
   only for a token's strongest picks.
3. Bits by usage: hot experts at high precision, cold ones low.
4. Fit a low-bit (or smaller) network to each expert's OUTPUTS on real inputs -- a local fit per expert, no full-model
   training. Not bound by the weight rounding floor.
5. Entropy-coded or trellis codes decoded inside the kernel (fixed-rate integer codes sit about one bit above the floor).
6. Data-aware rounding with error feedback (GPTQ / EXL3 style) from the engine's own recorded inputs.

## Questions for a reviewer
- Which of the assumptions above are wrong or untested in a way that changes the choice?
- Is there a storage or computation scheme for these experts that the list misses -- in particular one that exploits the
  latent sandwich (shared D before, shared U after, RMSNorm after the sum), the 16-of-512 routing, or the SwiGLU clamp?
- What is the cheapest experiment that would falsify each idea, given the emulation harness?

## Serving at 8 bits: what a decode token costs (2026-11-18, evening; 2 cards, real weights)

MEASURED, 176 expert slots a layer, the big non-expert matrices 8-bit too, a 400-token answer on unseen text:
13.3-15.1 tok/s (66-75 ms a token). Where it goes:

| part | ms a token | bound by |
|---|---|---|
| missed experts on the CPU (23-31 % of a token's 656 picks, 0.12-0.14 ms each) | 20-33 | RAM bandwidth (~55 GB/s here) |
| weights read on the cards: non-expert 7 GB + resident experts 3.5 GB | ~22 | VRAM bandwidth (~550 GB/s) |
| launch and library overhead (~2,000 launches, ~250 oneDNN calls, ~130 host waits) | ~15 | engineering |
| n-gram rows, head, transfers | ~4 | |

Steps of the day (decode tok/s): per-expert chains 3.4 -> a token's experts in two launches + the stream-mixing kernel
fixed 7.9 -> misses on the CPU instead of 7.5 MB uploads 11.9 (120 slots) -> 160 slots 14.1 -> 176 slots + 8-bit
non-expert matrices 14.9. Prompt reading: 2.1 -> 72 tok/s (554 tokens, chunks of 128). Load: 10 min -> 11-15 s (the
8-bit pack is mapped from disk).

A token reads 12.3 GB of NON-expert weights as fp16 (attention's gate and output maps are 16,384 wide in 30 of the 42
layers) against 4.9 GB of experts at 8 bits; DeepSeek-V4.1 here reads 4.6 GB a token in all.

### How many picks the cards can serve, by expert format (recorded usage: 1.34 M picks of wiki text, 0.36 M of chat)
| experts a layer on the cards | format that fits | ranked on other text | residency matched to the text |
|---|---|---|---|
| 176 of 512 | 8-bit (now) | 77-84 % | 84-89 % |
| 234 | 6.5-bit | 85-91 % | 91-95 % |
| 270 | 5.5-bit | 89-93 % | 94-97 % |
| 330 | 4.5-bit | 93-97 % | 97-99 % |
| 352 | 4.25-bit | 95-97.5 % | 98-99.5 % |

Pooling the slots across layers instead of a fixed number per layer gains nothing (within 0.3 points everywhere).
Fixed residency from a ranking file served only 56 % on an unseen topic (8.8 tok/s); the usage-scored promotion rule
reached 76-81 % within 400 tokens (15.1 tok/s). So: the format decides the miss cost AND the bytes read on the cards --
51 of a token's ~70 ms.

### A low-noise yardstick: the reference's expert picks forced
Free-running KL has a floor of 0.01-0.02 from pick flips and jumps by +-0.005 between runs of the same format (a single
row at KL 4 moves the mean of 554 rows by 0.007). With the reference's picks forced (`--force`) only the rounding is
left: 8-bit experts KL 0.00020 (logits within 0.9 %, top-1 553 of 554); 8-bit experts + 8-bit non-expert matrices
0.00066 (1.8 %, 549). Use it to compare formats; use the free-running number to say what a user sees.

### Next experiment (running): b-bit experts through the 8-bit path
`pack_lowbit_trial.py`: a symmetric b-bit integer with an fp16 scale per 32 is a special case of the 8-bit slot, so a
trial format needs no kernel. Three layers (5, 20, 35) are rounded to 4 bits -- plainly, and with the rounding error of
each column pushed onto the later ones under the recorded input covariance (GPTQ) -- and the forced-pick KL is scaled
by 41 / 3.

### Result: low-bit experts with error feedback (2026-11-18 evening; forced picks, 554 tokens, baseline all-8-bit KL 0.00020 / logits within 0.90 %)
One layer (20) converted, the other 40 at 8 bits; "layer error" = sqrt(error^2 - baseline^2):

| layer 20's experts | bits a weight | KL | added KL | layer error |
|---|---|---|---|---|
| 4-bit, plain rounding | 4.25 | 0.00136 | +0.00116 | 1.71 % |
| 4-bit, gate / up with GPTQ (down plain) | 4.25 | 0.00057 | +0.00037 | 1.24 % |
| 5-bit, gate / up with GPTQ (down plain) | 5.5 | 0.00038 | +0.00018 | 0.63 % |
| gate / up 4-bit with GPTQ, down 8-bit | 5.67 | 0.00031 | +0.00011 | 0.63 % |

Three layers (5, 20, 35) at 4-bit GPTQ: KL 0.00203 (+0.00183: the first and last layers cost more than the middle).
- GPTQ under the recorded input covariance (2,602 tokens, damping 1 %) cuts the 4-bit cost about 3x; it raises the
  WEIGHT error (9.2 % -> 11.8 %) while lowering the output's -- weight error is the wrong thing to watch.
- `down` is the sensitive matrix under plain rounding: its input is an expert's own hidden activation, for which there
  are no statistics yet. Recording them (per layer, all experts pooled) would let GPTQ cover it too.
- Scaled to 41 layers: 4 / 4 / 8 or 5 / 5 / 5 about KL 0.005-0.007 (logits within ~4 %), 4 / 4 / 4 about 0.015-0.025 --
  against 0.0002 at 8 bits and the model's free-running noise of 0.01-0.02. The old MXFP4 pack: 0.144 (85 tokens).
- Kernels for 8 / 5 / 4 bits exist and are checked (commit 21bcf7e): a format run natively = the same integers through
  the 8-bit kernels, digit for digit. What a format buys in speed is in the table above (picks served on the cards).

### CORRECTION (2026-11-18, 20:00): the trial table above used statistics that contained the test text
The covariance behind the first GPTQ trials (`hy4_cov.npz`, 2,602 tokens) included the 554-token text the KL is
measured on. Recorded again with the engine (`ie-hy4flash-run --cov-out`, 6,861 tokens of prompts, answers and article
text, the test text excluded), layer 20, forced picks, baseline 0.00020 / 0.90 %:

| layer 20's experts | statistics | KL | added KL | layer error |
|---|---|---|---|---|
| gate / up 4-bit GPTQ, down 8-bit | with the test text (above) | 0.00031 | +0.00011 | 0.63 % |
| gate / up 4-bit GPTQ, down 8-bit | clean | 0.00046 | +0.00026 | 0.99 % |
| 4-bit, GPTQ on gate / up only | with the test text (above) | 0.00057 | +0.00037 | 1.24 % |
| 4-bit, GPTQ on all three (down under the pooled hidden-activation covariance) | clean | 0.00059 | +0.00039 | 1.35 % |

- Scaled to 41 layers with clean statistics: 4 / 4 / 8 about KL 0.010 (not 0.005), full 4-bit about 0.016 -- at the
  model's own free-running noise (0.01-0.02), not below it.
- GPTQ's gain depends on how well the input covariance is known: 6,861 samples for 1,536 dimensions is thin, and the
  leaked run shows what a well-matched covariance buys (0.63 % against 0.99 %). A larger calibration set is the lever
  (`--cov-out ... --cov-inputs-only` records the inputs at normal speed).
- A layer-pooled covariance of the hidden activations does not help `down` (1.35 % against about 1.2-1.3 % expected
  without it): each expert has its own hidden statistics, and 6,861 tokens give an expert about 200 samples.

## 8-bit serving, end of 2026-11-18 (no lower-bit pack; the 8-bit path made faster)
Unseen 400-token answers, both cards, 8-bit experts + 8-bit big non-expert matrices: **17.7-18.8 tok/s** (13.3 at
18:00; DeepSeek-V4.1 on this machine 17.7). Quality unchanged: forced-pick KL 0.00067 (549 of 554), free-running through
the decode kernels 0.017 (521 of 554) -- the band the model itself moves in.

What moved it, in order:
- The starting ranking was built from 2,602 tokens of article text and one chat: on 21,013 tokens of question-and-answer
  text it put the right experts on the cards for 56.7 % of picks; a ranking from that text reaches 75.2 %
  (`stats/hy4_rank_chat.u32`).
- After a prompt chunk each layer swaps in the experts the prompt used (`rebalance`, up to 32 a layer); during decode a
  missed expert is promoted on a score difference of 3. Misses on unseen answers 31-33 % -> 14-16 %.
- The rarely used experts that now go to the CPU must be in RAM: a background thread pages the mapped pack in at load
  (cold, the same miss rate ran at 11.9 tok/s instead of 16.3). Big reads of other files evict it again -- conversions
  and model downloads on this machine slow serving until the pack is back.
- The n-gram rows of a call are requested from the drive at its start (5.2 -> 1.4 ms a token on unseen text; the tables
  are on the slower drive).
- Slots a layer are sized from each card's free VRAM (186 / 173); fewer launches in attention.

A decode token now (about 53-56 ms): attention 20-21 (11 of it weight traffic, the rest ~20 launches a layer at ~11 us),
experts and the rest of the FFN on the cards 15-16, missed experts on the CPU 10-14 (0.12-0.16 ms each, 89 a token),
n-gram 0.5-1.4, head and transfers 2.

Measured dead ends: more promotions a token (1 / 2 / 4 / 8 / 16 all 18.3-18.8); removing host waits (the indexer's 12
read-backs a token, the stage-timing wait: sync points, not work); a next-token guess from the half-way state
(the output head on the first card's result equals the final token on 2 of 300 tokens) -- so running the two cards at
the same time needs a real predictor, i.e. the model's drafter (`model.dfly_layers.*`, no reference code).
Left at 8 bits: fuse more launches (each one a layer is ~0.46 ms a token), the remaining fp16 matrices to 8-bit
(~1.5 ms), one parallel region in the CPU leg (~1.5 ms), the drafter.

## Serving state, 2026-11-19 04:00 (the playbook pass and the first real-context measurements)
Everything before this section was measured at a 57-token prompt and ctx 2,048. The test machine's presets run ctx
200,000-500,000 and prompts of thousands of tokens; the numbers that describe real use are these (a desktop with other
jobs running, `ie serve` or `ie-hy4flash-run --cards 2 --q8 --q8-dense`):

| | before this pass | now |
|---|---|---|
| decode, short context (ctx 2,048) | 17.4 tok/s | 21-23.6 |
| decode at 3,900 tokens of context, ctx 262,144 | 14.3 (139 / 128 slots a layer) | 16.9-17.3 (169 / 156) |
| decode at 23,000 tokens of context | not measurable (cost grew with the context) | 14.9 |
| prompt read, 3,900 tokens | 43.6 tok/s | 113-120 |
| prompt read, 23,000 tokens | (quadratic) | 107.8 |

What moved them, by commit: c806636 (pinned hand-offs, four small kernels), c1badd5 (the CPU team's idle spin),
c5b5fa5 (work units of 16 rows), 4d2cc7d (a missed expert read once per chunk; the recall over a team), c84a57d (eight
blocks a step in the row kernel), 2ee7316 (key windows for the sliding layers, selected keys and on-card group scores
for the indexed layers).

Where a decode token goes at a short context (about 42 ms): the cards 27.8 ms, of which 24 is weight traffic at memory
speed (attention's gate / o / q_b 9, the experts 7.7, the other 8-bit maps 2); the CPU leg 11.5 ms, mostly inside the
cards' expert time; about 11 ms of host cost for 1,771 kernel launches. Each card is executing 43-50 % of the time (they
take turns).

Measured and ruled out (numbers in the commit messages): slot counts by layer spread; our own thread pool; huge pages
for the pack; heads on the sink (0.00 %); low rank in gate / o (3066 and 3068 of 3072); a cache fetch of predicted
misses on the calling thread; copying missed experts to the card (0.75 ms against 0.12-0.22 on the CPU).

Still to take, with my estimates: launch fusion (1,771 -> ~600 launches a token: 5-7 ms); the heads of one layer's
attention on both cards at once (about 5 ms; the prompt read's overlap has to be kept); the drafter (about +30 %: the
experts do not share across rows); the predicted-miss fetch with each core computing what it fetched. Bound on the
misses themselves: the best fixed set per conversation misses 9.6 % at 186 slots a layer, the engine's rule 10.6 %
once a reply is 200 tokens old (20 % in its first 200).
