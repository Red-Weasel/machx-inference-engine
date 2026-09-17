# V4.1 port — Phase 6a gate criteria: engram (written BEFORE the build)

**Scope:** the engram module at layer 1, checked against the reference's own `NgramHashState`
and `Engram` on real weights. Not deferrable: engram adds into the residual stream at layers 1
and 14, so without it every token after layer 1 is wrong.

## What engram is, from `inference/engram.py` and `model.py:296-368`

Per position, per engram layer:

1. **Hash.** Token ids are mapped through a *compressed* vocabulary (tokens that normalise alike
   — " The", "the", "THE" — collapse to one id; 129,280 → 99,092). The position and the 3 tokens
   before it are each multiplied by a per-layer odd int64 multiplier; the running XOR after `i`
   steps is the hash of the `(i+1)`-gram. Each of the 3 n-gram sizes (2,3,4) and 8 heads gets
   its own prime-sized bucket range, so a position yields **24 table indices**.
2. **Gather.** 24 rows of `[256]` FP8 with `[8]` E8M0 scales from a `[384,006,168, 256]` table —
   **6 KB per token from a 91.5 GB table.** The table stays FP8; rows dequantise on lookup.
3. **`wkv`.** The 24 rows flatten to 6144; a `[25600, 6144]` FP8 Linear gives `hc_mult` keys of
   `dim` plus one value of `dim`.
4. **Gate.** Per hc copy, a dot product of the stream against its key, each RMS-normalised
   separately, scaled by `dim^-0.5`, then **signed sqrt** then sigmoid.
5. **Residual.** `h + gate * value`.

Two facts that make this cheap at runtime and are load-bearing for the design:
- the hash depends **only on token ids**, so every lookup for a prefill chunk — and for a decode
  token — is known *before* layer 0 runs. The 6 KB gather can be issued at the top of the
  forward pass and overlap all of layers 0.
- the compressed-token map, primes and multipliers are functions of the tokenizer and config
  only. They are computed **once**, by the reference's own functions, and stored.

## Pass criteria

1. **The tables are the reference's, not a reimplementation.** `engram_tables.py` calls the
   shipped `build_compressed_token_map`, `EngramLayout.from_args` and
   `compute_hash_multipliers` and serialises their outputs. The compressed vocab size it
   produces must equal `config.json`'s `engram_compressed_vocab_size` (99,092) — the reference
   asserts this too, and a mismatch "would silently rehash the whole table".

2. **Hashing is bit-exact.** For a real tokenised text (≥ 64 tokens, including repeated
   n-grams), the engine's hash ids must equal the reference's `NgramHashState.forward` output
   **as integers, all of them** — `[L, 2 layers, 24]`. Not close: equal. Integer arithmetic has
   no tolerance.

3. **Look-back boundaries are tested.** The first three positions (where the n-gram would reach
   before the sequence start) use `pad_id` for the missing slots. The comparison must cover
   positions 0, 1, 2 explicitly, since a hash that is right everywhere except at the start
   would be right for most of a long prompt and wrong for every short one.

4. **The gather + dequant is bit-exact.** The 24 dequantised rows per position, from the engine's
   mmap read and the Phase-2-verified decode, must equal the reference's
   `ParallelEngramEmbedding.forward` as fp32 **exactly** (max |diff| 0.0): E4M3 x E8M0 is exact.
   The reference then rounds to bf16; the engine keeps fp32 — recorded as a deliberate
   precision difference, not a defect, and everything downstream is compared with that in mind.

5. **`wkv`, gate and residual match stage by stage** on the block-golden residual stream at
   tolerances justified from the arithmetic: `wkv` output at 3e-3 (one fp16 GEMM), the gate at
   3e-3, the final stream at 3e-3. The gate stage must be compared **on its own**, because a
   gate stuck at 0.5 (sigmoid of 0) would still give a plausible-looking residual.

6. **The signed sqrt is exercised.** The test asserts that the golden's pre-sigmoid values
   include both signs and that at least one has |dot| > clamp (1e-6) — otherwise criterion 5
   cannot distinguish `sigmoid(copysign(sqrt(|d|), d))` from `sigmoid(d)`.

7. **Table bytes are read from disk, not held in RAM.** Peak RSS under 4 GiB against the 91.5 GB
   table; the gather is 24 rows per position, not a slice of the table.

8. **No new kernel beyond the gate**, and the diff shows only: `engram_tables.py`,
   `golden_engram.py`, the engine's hash function + gate kernel, the test tool, CMakeLists.

## Explicitly NOT in this phase

Prefetch/overlap of the gather, the NVMe path for the table (it is mmap'd here and the OS pages
it), layer 14, image-span masking (`token_mask`), tokenisation in the engine (token ids come
from the golden — the tokenizer is its own phase).
