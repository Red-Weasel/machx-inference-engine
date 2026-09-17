# V4.1 port — Phase 24 step 2 criteria: the chunked prefill, so a long prompt actually goes in (written BEFORE the build)

**The requirement (founder, 22:29):** "ideally id want to run 500k context at least but get 250k going solid first
then try for 500k as long as it doesnt cost decode speed too much... 500k ctx and 10+ decode is at least useable".

**Where step 1 left it (all measured tonight).** Any context now LOADS — 200k in 41 s, 250k, 500k — because the
per-call scratch is sized by `max_forward_tokens` instead of the context. What is still missing is a way to put a
long prompt IN: a call longer than that bound is refused, so the prompt ceiling is the bound, and raising the bound
costs exactly the thing the founder does not want to pay:

| configuration at ctx 500,000 | static expert slots / card | prefill | decode |
|---|---|---|---|
| `IE_DS41_MAX_FWD=2048` (the default) | **39 / 44** | — (prompts > 2,048 refused) | — |
| `IE_DS41_MAX_FWD=16384`, EP on | **14 / 16** | 12,637 tok at **117 tok/s** | **5.3 tok/s** |

The workspace at a 16k bound eats ~8 GiB per card, which comes straight out of the static expert tier, and decode
falls to 5.3 tok/s. **So the answer to "500k with 10+ decode" is a small forward bound plus a chunked prefill** —
keep the scratch at ~1 GiB and the static tier near 40 slots, and feed the prompt in 2,048-token pieces.

## What a continuation chunk has to do, and the one thing that does not scale

For a chunk of T rows at `pos0 > 0` the correct key set is: the window keys of the previous `WIN-1` positions plus
the chunk's own rows (causally), and the accumulated compressed latents plus the chunk's own new latents (causally).
Every piece of that bookkeeping — ring writes at their slots, latents appended at `nc`, the ratio-2 half-pair across
a boundary, per-row masks against the ring — was built and gate-verified bit-identically at T ≤ 8 by DSpark P2/P3
(docs/55, 56). Two things must change to lift it to chunk scale:

1. **The window keys.** The 128-slot ring cannot hold both the previous window and 2,048 new rows, so the chunk's
   window segment becomes a dense `[WIN + T, HD]` buffer — the live ring copied in position order, then the chunk's
   own keys — with a `[T, WIN + T]` sliding-causal mask. 4.5 MB at T = 2,048, and it reuses the prefill's own
   `ds4_attention_segs` exactly.
2. **The compressed bias does not scale, and this is the real work.** `mcomp` and the indexer's `scores` are
   `[T, NC]`, and NC is the ACCUMULATED latent count. At T = 2,048 and NC = 250,000 that is **1.9 GiB each**; at
   500,000, 3.8 GiB. The fix is the one the DS4 1M campaign already established for this exact shape
   (`deepseek4_attn.hpp:229-232`): **strip the query rows** — run the indexer's scoring, its top-k, the bias build
   and the attention over strips of S query rows, so every `[T, NC]` buffer becomes `[S, NC]`. At S = 64 and
   NC = 500,000 that is 122 MiB. `index_topk` is 512, so a query needs only 512 of those columns; stripping is what
   makes the dense bias affordable until a gather attention replaces it.

## Pass criteria

1. **Correctness at small scale first, against the path that already works**: a 4,096-token prompt fed as 2 × 2,048
   chunks must give the same next-token logits as one 4,096-token prefill — the existing logits bar (5e-2) and the
   SAME argmax, not bit-identity (the grouping differs). Then 8,192 as 4 chunks, and 16,384 as 8.
2. **250k, solid**: a ≥ 200,000-token prompt prefilled in 2,048-token chunks at the DEFAULT bound, with the static
   tier still ~40 slots per card, the prefill rate reported, and a correct short reply afterwards.
3. **Then 500k**, the same way, with **decode ≥ 10 tok/s measured after the long prefill** — the founder's bar. If
   decode lands below 10, report the term that costs it (the prime suspect is the indexer scoring NC entries per
   layer per token) before proposing anything.
4. **No regression at 2,048**: decode / resident / replay / forward / generate / multi (`IE_DS41_CPU_MISS=0`) /
   rollback / dspark all pass with today's counts, and the decode test's forced digits are unchanged.
5. **The chunking is invisible to callers**: the generator and the server feed chunks themselves, so `ie serve` and
   Dream need no flag beyond the context. A prompt longer than the context is still refused by name.
6. **The GPU survives**: zero `xe ... Timedout job` / reset lines across the phase, and every run shut down through
   `/admin/shutdown` (the 8a6ae6a drain is what makes that safe).
7. **Reported**: the prefill rate per chunk size, the VRAM the strip buffers cost, and the ingestion time for 250k
   and 500k, so the founder can judge the one-time cost of a long prompt.

## Explicitly NOT in this phase

A gather attention (the strip makes the dense bias affordable; replacing it is a later, larger kernel phase), prompt
caching across turns, concurrency, and any change to the decode path at T = 1.
