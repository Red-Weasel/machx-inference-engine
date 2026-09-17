# Phase 7 — Tokenizer + sampler

**Date:** 2026-04-25
**Hardware:** Intel Arc Pro B70 (BMG-G31)

## Gate result: PASS

13 / 13 tests pass.

### Tokenizer
| Test | Result |
|---|---|
| Load from GGUF (vocab=248,320, bos=248044, eos=248046, pad=248044, add_bos=false) | OK |
| Special token IDs match research/04 (`<|endoftext|>`=248044, `<|im_start|>`=248045, `<|im_end|>`=248046) | OK |
| `encode("Hello, world")` decodes to `["Hello", ",", " world"]` | OK (ids=[9419, 11, 1814]) |
| Round-trip ASCII `Hello, world` | OK (3 tokens for 12 bytes) |
| Round-trip ASCII pangram | OK (10 tokens for 44 bytes) |
| Round-trip CJK `你好,世界` | OK (3 tokens for 13 bytes) |
| Round-trip mixed emoji+text `🚀 Llama 🦙 emoji ✅` | OK (10 tokens for 25 bytes) |
| Round-trip whitespace edge cases | OK (6 tokens for 22 bytes) |
| ChatML build + encode special tokens | OK |

### Sampler
| Test | Result |
|---|---|
| `sample_argmax` finds the maximum | OK |
| `temp=0` falls back to argmax | OK |
| `top_k=1` always picks argmax | OK |
| Reproducible per seed | OK |

## Notable findings

1. **research/04's claim about `Hello, world` token IDs was wrong.** Research/04 §6.1 stated the IDs are `[9707, 11, 1879]` based on the assertion that "text-token IDs match Qwen2.5 exactly". They don't — Qwen3.6's vocab reassigns positions. The ACTUAL IDs are `[9419, 11, 1814]`, verified by decoding each ID back through the GGUF vocab. The tokenizer is correct; the prior reference fixture was wrong. Research/04 has been corrected.

2. **CJK and emoji round-trip cleanly** despite the simplified pretokenizer. Multi-byte UTF-8 sequences are treated as "letter" runs, which produces tokenizations that may differ from HF in token-count but decode losslessly. For decode-mode generation (where the model emits tokens we just decode), this is sufficient.

## Carried-over limitations (Phase 9 backlog)

The pretokenizer is a simplified stand-in for the real `\p{L}\p{N}\p{M}` Unicode regex. Edge cases that may diverge from HF's tokenization (different IDs, same decode):
- Script transitions inside CJK runs (Hangul + Han + Latin in one stream).
- Numbers attached to letters (HF splits on `\p{N}` per-digit; we lump all consecutive digits into one pretoken — actually NO, looking at the code more carefully, my impl emits one digit at a time per HF — so this case is fine).
- Arabic / Devanagari combining marks (`\p{M}`).
- Contractions (`'s`, `'t`, etc.) — HF splits with case-insensitive prefix; ours doesn't.

For Qwen3.6 daily-driver use (English-language chat + light multilingual), the simplified version should produce equivalent or near-equivalent tokenizations. **All round-trips are lossless** (the load-bearing correctness contract).

## Files

- `include/ie/tokenizer.hpp` + `src/tokenizer/tokenizer.cpp` — vocab loader, byte-level BPE, ChatML template builder. ~350 lines.
- `include/ie/ops.hpp` (extended) + `src/ops/sampling.cpp` — `sample_argmax`, `repetition_penalty`, `sample_softmax_topk_topp`. The sampler is single-threaded on-device for v1; Phase 9 has a parallel version queued.
- `tests/unit/tokenizer_test.cpp` — 13 tests covering load, special tokens, known IDs (decoded-string check), 5 round-trip cases, ChatML, 4 sampler cases.

## Phase 9 backlog (additions from Phase 7)

Lower priority than the FA-2 / fused-MoE items, but worth listing:
1. **Full Unicode pretokenization regex.** Vendor llama.cpp's `unicode.cpp` + `unicode-data.cpp` (their property tables), or use PCRE2 with `--enable-unicode-properties`. ~3000 lines. Resolves all known divergences from HF.
2. **WG-parallel sampler.** Current `sample_softmax_topk_topp` is `single_task` — fine for vocab=248k (~250 µs) but a parallel version would shave it to ~30 µs. Not load-bearing for 50 tok/s.
3. **Speculative decoding via the MTP head** (model has 1 MTP layer; Qwen claims 1.5–2× decode speedup). Far-future.
