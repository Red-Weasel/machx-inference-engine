# V4.1 — Phase 58: prompt-lookup speculation for agent traffic

## Why

The long agentic soak (docs/96) spent 7.5 of 18.4 minutes decoding `update_todos` calls: ~300 tokens each, nearly all
of it the previous todo list copied out again. Agent output is full of copies -- tool arguments that restate a list,
a file written back with one change, a quoted `old_string`, a report assembled from notes. Decode is at the hardware
bound (docs/90), so the lever is committing several tokens per forward where the next ones are predictable.

DSpark (docs/48-59) lost because its learned drafter was accepted 2.57 tokens per pass and cost 19.5 ms per draft.
A **prompt-lookup** draft costs nothing: the continuation after an earlier occurrence of the context's last tokens.

## Measured before building: the shadow replay and the policy search

- `IE_DS41_LOOKUP_SHADOW=1` replays each finished request as a lookup decoder would have run it (output untouched);
  `IE_DS41_LOOKUP_DUMP=<dir>` writes each request's token stream.
- `tools/ds41_reference/lookup_policy.py` replays the streams under candidate policies, priced with the measured step
  costs (one row 79 ms in the shipping configuration -- the CPU miss leg serves one-row steps only; a T-row verify
  131 / 171 / 292 ms at T = 2 / 3 / 6, ~41 ms per extra row, docs/59).

66 requests / 17,817 generated tokens (the long agent run, free-text explanations, a code rewrite, a list edit):

| policy | priced | losing requests |
|---|---|---|
| draft on any 2-gram match, <= 3 or 7 drafts | x1.05-1.09 | 18-29 of 66 (prose: x0.61-0.83) |
| copy >= 8 tokens, most recent occurrence | x1.19 | 7 |
| copy >= 12 tokens, most recent occurrence | x1.16 | 2 |
| **copy >= 12 tokens, the occurrence whose match reaches furthest back, <= 7 drafts** | **x1.24** | 5 (worst x0.90, a 21-token reply) |

Passes are bimodal: a copy either lands whole (7 of 7) or misses at once. Blind drafting loses on prose; requiring a
long copy keeps prose at exactly x1.00 (no draft is ever offered). "Furthest back" beats "most recent": rewriting a
list, the line just written shares every fixed field with the next one, and only the original line in the context
also shares its number.

## What was built (`src/model/deepseek41_generate.cpp`)

- `Ds41NgramIndex`: every 2..6-gram's last 16 ends; a draft copies from the earlier occurrence of the longest suffix
  whose backward-extended match is longest (ties: most recent), only when that match is >= `IE_DS41_LOOKUP_MIN`
  (12) tokens, up to `IE_DS41_LOOKUP_K` (7) drafts, never an image position.
- The loop: a pass feeds [id, drafts] as one multi-row step with every row's logits; row r samples with the plain
  sampler and the running `recent`; drafts are kept while they equal the sample; the first mismatch's sample is the
  next token; `rollback_to` keeps the rows of id and the accepted drafts. Sampling each row from p and accepting while
  equal is exact speculative sampling for a one-hot draft, so temperature > 0 stays distribution-exact. No draft: a
  plain one-row step (with the CPU miss leg). The prefix cache stays on (no drafter rings to checkpoint).
- `Ds41Forward::set_multi_row_decode` admits the 2..8-row steps without `IE_DS41_DECODE_MULTI`.
- `ie serve` turns it on (`Ds41Bundle::lookup`; `IE_DS41_LOOKUP=0` turns it off); elsewhere `IE_DS41_LOOKUP=1` or
  `Ds41Generator::set_lookup` per generator.

## Gate: `ie-ds41-lookup-test` 8/8 (`IE_DS41_CPU_MISS=0`, `~/ds41_work/p58/lookup_test.log`)

| prompt | plain vs lookup | tokens | speed |
|---|---|---|---|
| rewrite a 20-item list with one change | identical | 360 | x1.33 (47 passes, 6.2 drafts accepted each) |
| repeat a paragraph | identical | 98 | x1.24 |
| return code with a parameter renamed | identical | 98 | x1.10 |
| free text | identical | 150 | x1.03 (no draft offered) |
| temperature 0.7, seeded | repeats under its seed, 26 passes verified | 200 | -- |

## The served A/B: 42 recorded agent requests replayed byte-identical (`~/ds41_work/p58/replay.py`)

Each recorded request (full message history, images included) re-sent at temperature 0 with `max_tokens` = the
recording's length, to a plain server and a lookup server, both cold; decode time from the server's `[gen]` lines.

| run | decode | vs plain |
|---|---|---|
| plain (A) | 8,757 tokens, 106.0 ms/token (9.43 tok/s) | -- |
| lookup, CPU leg serving one-row steps only (B) | 101.6 ms/token | x1.04; outputs identical 42/42 |
| **lookup + the multi-row CPU leg (D, docs/98)** | **71.6 ms/token (13.96 tok/s)** | **x1.47 vs the mean of A and A2**; 40/42 identical |
| plain again (A2, drift control) | 104.8 ms/token | identical to A, 40/40 |

B's acceptance was already excellent (6.62 of 6.89 drafts per pass) but a verify row cost 97 % of a one-row step;
docs/98 found why (every pinned expert of the verify's union crossed PCIe) and fixed it. D is the shipped
configuration: **`ie serve` runs lookup ON** (`IE_DS41_LOOKUP=0` turns it off); the library default stays off, so the
golden-token tests keep their meaning. The two divergent outputs are greedy near-ties resolved differently by the
split (docs/98, Correctness); D decoded 8,193 tokens because of them, so ms/token is the comparison.

## Found on the way: a teardown deadlock

`~Ds41ExpertTier()` was `= default`, and the persistent mmap readers (docs/88's leak fix) were stopped only in
`free_storage()`. A process that destroyed the runtime without `free_resident()` -- the first lookup test, returning
on an error -- destroyed the readers' condition variables while the threads waited on them, and `pthread_cond_destroy`
blocked forever with ~215 GB pinned. The destructor now stops the readers and the CPU worker (no-ops after
`free_storage`). The stuck process was terminated with the founder's approval; kernel taint stayed clean (12288).
