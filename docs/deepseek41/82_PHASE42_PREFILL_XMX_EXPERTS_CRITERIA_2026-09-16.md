# V4.1 — Phase 42: prefill, step 1 -- the expert GEMMs on XMX (criteria before the build)

## The chunk timeline, measured (the 32k profiled long test, `p24/p39_32k_prof.log`)

A 2,048-token chunk at 32k: **wall 13.9 s, named kernels 10.3 s** -- compute-bound, not DMA-bound as docs/80's
recommendation assumed (the pinned-expert streaming is the ~3.6 s the wall exceeds the kernels by).

| kernel, per chunk at 32k | ms | at NC 2,048 | grows with context? |
|---|---|---|---|
| `ds4_gemm_mxfp4` (int-dot, W4A8, M > 1) | **4,794** (4,478 calls) | 2,596 | no |
| `ds4_attention_xmx` (block-sparse prefill) | 3,249 (306 calls, 10.6 ms each) | 198 | yes: a 64-key block walk over NC per (token, tile) |
| `ds4_indexer_score` (SIMT "headlane") | 948 (64 calls, 14.8 ms) | 19 | yes: O(T x NC), a GEMM done as a GEMV |
| `ds41_hc_mixes_part` | 370 | 564 | no |
| `ds4_router_logits` | 244 (6.1 ms each: a [2048 x 5120] x [5120 x 384] on a GEMV kernel) | 119 | no |
| `ds4_indexer_topk` | 108 | — | yes |
| the rest (oneDNN q_b/o_b/o_a, masks, norms) | ~350 | ~200 | little |

At 224k the two O(NC) terms scale to ~10 + ~7 s of a ~24 s chunk (the 262k prefill ran 85 tok/s at the end);
at 32k the int-dot expert GEMM is the largest single term at every context.

## Step 1: the expert GEMMs on XMX for the prefill groups

The engine already has the route: `ds4_expert_gemm_xmx_grouped` (fused MXFP4 -> XMX, W4A16, default on for DS4's
model since 2026-08-09 on a PPL gate that measured it BETTER than int-dot -- no int8 activation quantisation --
and 11.5-32 TFLOP/s against the int-dot kernel's ~3). V4.1's tier never calls it: every prefill group runs
`ds4_expert_gemm_q8_grouped`. The change mirrors `deepseek4.cpp:3943-4030`: for groups with more than 8 rows
(prefill strips; T <= 8 keeps the int-dot M = 1 / per-row jobs so decode, DSpark and the CPU leg are untouched
bit for bit), gate/up as XMX jobs into fp32 `g_f`/`u_f`, the fp32 -> fp16 SwiGLU, down as XMX jobs into `y_f32`
straight from `h_h`, and the fp32 scatter. `IE_DS4_EXPERT_XMX=0` keeps the int-dot route.

## Criteria

1. **Decode step kernels untouched, bit for bit:** `multi` 38/38, `rollback` 74/74, `dspark` 153/153 (T <= 8 paths,
   a T-row step against its one-row steps after the same prefill). The decode test must PASS 28/28, but its
   criterion values may move within their bars: its 12-token prompt is itself a prefill (T > 8), so the KV state
   the decode steps read comes through the new route. (First draft said "identical criterion lines"; that was
   a wrong premise, corrected before the build.)
2. **Prefill numerics:** `ie-ds41-forward-test` and `ie-ds41-cont-test` PASS on their existing bars (the reference
   is the transformers model; W4A16 is expected at or inside the W4A8 route's error), and the 24k held-out run
   retrieves the needle with coherent text (the stream is expected to differ from the int-dot stream).
3. **Speed:** the 32k long test's per-chunk `ds4_gemm_mxfp4` term replaced by the XMX kernel at less than half its
   time (4.8 s -> < 2.4 s per chunk at 32k), and the chunk rate up accordingly; the 2k prefill (`pp2048`, the
   resident test's number) not slower.
4. Recorded either way; the int-dot route stays selectable.

## Results so far (build 2026-09-16 ~06:30; the route wired for T > 8 in `deepseek41_experts.cpp`)

**Speed (criterion 3):**

| 32k profiled long test | int-dot (Phase 39) | XMX prefill route |
|---|---|---|
| expert GEMM per chunk at 32k | `ds4_gemm_mxfp4` 4,794 ms | `ds4_gemm_mxfp4_xmx` **2,517 ms** (-47 %; the bar said < 2,400) |
| named kernels per chunk at 32k | 10,299 ms | 7,941 |
| chunk wall at 32k | 13.89 s | 12.79 s |
| 32k prefill | 199.3 s = 164.4 tok/s | **184.9 s = 177.2 tok/s** (+8 %) |
| 24k held-out prefill | 179-182 tok/s | **206 tok/s** (+14 %), needle retrieved, decode 10.48 tok/s |
| decode at 32k (profiled) | 67.5 ms/token | 68.9 (noise; the decode step is not on this route) |

The wall gains less than the kernels (-2.3 s of kernels, -1.1 s of wall): with the compute shorter, the
pinned-expert streaming that the wall exceeds the kernels by (~4.8 s per chunk now) shows through -- the
DMA/compute overlap is the next term, exactly as docs/80 first guessed and the profile then hid.

**Gates (criteria 1-2):** multi 38/38, rollback 74/74, dspark 153/153 (decode-step kernels untouched);
`ie-ds41-forward-test` 5/5 on both routes. `ie-ds41-decode-test` **26/28**: every per-layer bar IMPROVED (forced
routing worst 9.0e-3 -> 4.6e-3 on card 0, logits 8.0e-3 -> 4.9e-3, the W4A16 route's fp16 activations against
int-dot's int8), but the engine's own router flips at layer 12 where the golden's 6th-vs-7th margin is 4.566e-3
-- above the test's 3e-3 near-tie bar -- and the top-5 set then differs in its 5th entry (14 vs 1) with the next
token still the golden's. `ie-ds41-cont-test` **14/15**: chunks of 16 against one-token steps read logits rel
1.56e-1 against a 1.5x-the-control bar of ~6.2e-2; chunks of 32 and 64 pass. The reference steps are T = 1
(int-dot), the chunks are XMX; a deviation that grows as chunks shrink points at small-M jobs in the XMX route.

## Diagnosis of the two misses, and the verdict

- The fused MXFP4 -> XMX kernel is exact at every M: `deepseek4_mxfp4xmx_gate_test` (ragged M, 8-row tails,
  err/bound <= 0.010, no non-determinism) and `deepseek4_xmx_boundary_test` (152 boundary cases) PASS on this
  build. The chunks-of-16 miss is the cont test comparing a W4A16 chunked prefill against W4A8 one-token steps:
  perturbation 4.10 logits against the XMX arm's control 2.22 (bar 1.5x = 3.32); on int-dot both sides it is
  1.79 against 3.01 and the test is 15/15. It measures the distance between two activation-quantisation routes,
  not a chunking fault -- but the engine's chunked and one-token prefills of the same text now diverge by ~1.5x
  more than they did, which is a real property of having two routes chosen by T.
- The decode test's miss is one own-router flip at a 4.566e-3 golden margin (bar 3e-3) at layer 12 of a
  12-token prompt, with every arithmetic bar better than the int-dot route's and the next token the golden's.

**Verdict: NOT the default.** Both bars are the shipping gates as written and are not relaxed here. The route is
wired and OPT-IN: `IE_DS41_PREFILL_XMX=1` (with the engine-wide `IE_DS4_EXPERT_XMX` still on). What it buys when
on: +8 % at 32k prefill, +14 % at 24k, the expert GEMM term halved, and per-layer errors halved. What the founder
would be accepting: prefill routing that follows the W4A16 numerics at near-ties (a different top-5 in one of the
test's 12-token prompts), and chunked-vs-one-token prefill divergence ~1.5x today's. The forward test (against
the transformers reference) passes on both routes.

**Next prefill term, route-neutral:** the pinned-expert streaming that the chunk wall exceeds its kernels by
(~3.6 s per chunk on int-dot, ~4.8 s with the XMX route), i.e. the DMA/compute overlap inside the tier's group
loop -- the fetch pipeline issues group g+1 before group g's GEMMs, so the first thing to measure is how much of
a chunk's DMA is actually hidden.
