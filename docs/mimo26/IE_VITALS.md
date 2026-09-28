# ie_vitals: per-token vital signs in the stream (opt-in)

Status 2026-09-25: engine half of VS-E phase 1. MiMo-V2.6 only. The streaming chat endpoint of `ie serve` supports it.

## What it is

A client that sends `"ie_vitals": true` as a top-level field of a `/v1/chat/completions` request with `"stream": true`
gets a few extra numbers in the SSE stream. They describe how sure the model was about the tokens it just wrote, and
how often the DFlash drafter guessed right.

It is **passive**. The sampler writes the numbers and nothing reads them back. The tokens, the sampling, the RNG
stream and the reply are the same with the field on or off.

Without the field (or with `false`), the stream is byte-for-byte what it was before.

## Wire format

Every chunk the server writes carries the window once it holds at least 16 committed tokens. The server also
reports it early at a tool-call boundary: when `<tool_call>` is detected, the chunk that flushes the text before it (or, if there is none, the next written chunk) carries whatever the window holds.
The finish chunk (the one with `finish_reason`) carries the rest, so the windows add up to the whole reply.
Chunks can be held back (reasoning tag, a forming tool call), so `n` can be more than 16.

```json
{"object":"chat.completion.chunk", "choices":[...],
 "ie_vitals":{"n":16,"H_mean":0.41,"H_max":2.9,"margin_min":0.03,"n_hi":1,"draft":{"offered":14,"accepted":9}}}
```

The usage chunk (just before `data: [DONE]`) carries the request summary:

```json
"ie_vitals_summary":{"tokens":256,"H_mean":0.38,"n_hi":7,"draft_offered":180,"draft_accepted":121,
                     "cached_tokens":40210,"cache_source":"prompt end","prefill_ms":1890.2,"restore_ms":38.1,"decode_tps":16.2}
```

Numbers are rounded to 4 decimals.

## What each number means

| field | plain meaning |
|---|---|
| `n` | tokens written since the last report |
| `H_mean`, `H_max` | how spread out the model's choice was, averaged over the window and at its worst. This is the entropy in nats of the model's distribution after the repeat penalty and temperature, before the top-p/min-p cut (the token is drawn after that cut). 0 = one obvious next token; ln 2 ≈ 0.69 = a coin flip between two; 2 ≈ seven equally likely options |
| `margin_min` | the closest call in the window: the probability of the top choice minus the second choice. Near 0 = two continuations were almost tied |
| `n_hi` | tokens in the window with an entropy above **2 nats**. This cut is a fixed diagnostic, not calibrated |
| `draft.offered` / `draft.accepted` | DFlash draft tokens sent to verification, and how many of them the model kept. A low ratio means the text is not what the drafter predicted. Prompt-lookup drafts (off by default in serve) are not counted |
| `tokens` | the whole reply's token count |
| `cached_tokens`, `cache_source` | prompt tokens restored instead of recomputed, and from where: `live`, `prompt end`, `slot N`, `none`. Omitted when the prefix cache is off |
| `prefill_ms`, `restore_ms` | prompt time, and the part of it spent restoring the cache |
| `decode_tps` | completion tokens per second of the decode loop. Omitted when there was no decode time |

The distribution that `H` and the margin describe is computed in double precision (under icpx's fast floating-point model):

- It is the temperature-scaled softmax over the kept candidates. At `top_k` 0 (Dream's setting) that is the whole
  vocabulary; with `top_k` N it is the top N.
- It is computed after the repeat penalty and before the top-p / min-p cut.

## When the numbers are null or missing

- **Greedy (`temperature` 0):** `H_mean`, `H_max` and `margin_min` are `null`, and the summary's `H_mean` is `null`.
  Greedy never computes the softmax, and the field does not add a vocabulary pass to get one.
- **Other models:** DeepSeek-V4.1 shares the sampler (`Ds41Generator::sample_row` takes the same optional stats
  pointer) but its generate loop is not wired. With `ie_vitals` true on any model other than MiMo-V2.6, the stream
  carries no extra fields at all.
- **Non-streaming requests:** these ignore the field.

## Cost

- **Off:** one pointer and one untaken branch per sampled row.
- **On:** one extra read-only pass over the kept logits per sampled row. It uses the `exp` and the sum the entropy
  needs, and leaves the sampler's own `z` loop untouched. Measured by `ie_vitals_test` on the CPU, at MiMo's
  152,576-entry vocabulary and Dream's setting: about +0.1 ms per sampled row (1.33 ms → 1.45 ms). A MiMo decode
  token takes ~45-60 ms.
- **Bandwidth:** about 150 bytes every 16 tokens.

## Code

| where | what |
|---|---|
| `include/ie/vitals.hpp` | `VitalsWindow`, the accumulator |
| `Ds41Generator::sample_row(..., Ds41SampleStats*)` | the entropy and margin |
| `mimo26_run_ids` | each committed token and each DFlash pass go into the window |
| `SamplingParams::vitals` | the pointer |
| `oai::sse_add_field`, `vitals_window_json`, `vitals_summary_json` | the JSON |
| the stream lambda in `openai_server.cpp` | when a report goes out |

Test: `tests/unit/ie_vitals_test.cpp` covers the entropy and margin against hand values and a long-double
reference, checks that the pick, the RNG and the row are unchanged with stats, and tests the window JSON and the parser.

## GPU verification (2026-09-25)

MiMo-UNCENSORED on port 11450, seed 1234, T 0.7, top-p 0.95, 256 tokens. With `ie_vitals` absent, the SSE bytes of a cold
and a warm request are identical to 4321c84's (id/created masked; 4321c84 run twice agrees with itself). With `ie_vitals`
on as the first request of a fresh server, the text and every frame (vitals fields removed) equal the request without it,
and the drafter's accepted count is the same (66 passes, 146). Decode 18.5-19.7 tok/s on both builds (noise). The windows'
accepted drafts sum to the summary, which equals the `[mimo26 dflash]` line. Note: a warm request whose predecessor left a
different reply generates a different text even with the same seed (the cache/drafter history differs), so compare vitals
on/off only from the same server history.
