# DeepSeek-V4 decode: host-synchronization inventory and removal plan (2026-09-10)

Source: read-only analysis by a second Claude session (00-inference-engine-2e) at the campaign
lead's request, against the tree at commit 438e480. Line numbers refer to that tree. Nothing here is
implemented yet; every item is a change to eviction-safety or staging-lifetime logic and must be gated
by `deepseek4_residency_test` §15h plus NLL bit-identity, not throughput alone.

## Why this matters (measured 22:12, docs/hardening_campaign_2026-09-10.md)
One warm T=1 step at 4K context, two cards: unprofiled p50 52 ms/token; GPU kernels 39.5 ms (two-card
sum, ~20 ms per card), expert-miss DMA busy 39.1 ms, host_ms_est 52 ms. Attention kernels are ~3 ms per
card. The token is roughly half GPU-idle; the idle is expert-miss traffic plus ~10 host serialization
points per layer (~430 per token).

## Per-layer host-synchronous points at T=1 (default two-card config, split_non_expert on)
Per card (its own driver thread; every `drain()` = `q_->wait()` timed into `waits_`, deepseek4.cpp:3284-3289):
1. router readback: D2H h_ridx_/h_rw_ then drain(waits_.router) (4056-4058) — required (host decides residency).
2. MoE eviction barrier: drain(waits_.moe) after the T=1 GEMV chunk (4284; 3951 on the T>1 grouped route).
3. layer_forward_post: drain(waits_.post) (4382).
4. window-close drains: drain(waits_.other) after the stack-local `cp` H2D (3379 compressor, 3444 indexer) —
   every 4th token on the 21 CSA layers (two drains), every 128th on HCA layers.
Orchestrator (DeepSeek4TpRuntime::forward 5455-5570): three lockstep segments per layer, each a
run_cards + join_job (4777-4832, spin_until 4680-4694) = 3 joins; two host reductions per layer
(attention output 5471-5485; routed+shared experts 5545-5555) via Ds4TpReducer::reduce
(expert_stream.cpp:1128+): D2H from both cards, d2h[0][k].wait() + d2h[1][k].wait() (1197-1198),
threaded host add, H2D on aux queues not waited = 4 host-blocking event waits + 2 host sums per layer.

## What is provably not needed at T=1 with exactly one acquire chunk, and the minimal replacement
1. **MoE eviction barrier (4284)** — not needed while the router drain (4058) stays. The hazard
   (4279-4283; expert_stream.cpp:1694-1700) is a later acquire evicting a dev_[L] slot an earlier GEMV of
   the same layer still reads. Slots are per layer (dev_[L] + v*sb, 1754), so within a token the only later
   writers of dev_[L] are the next chunk of layer L (absent by hypothesis) and token t+1's acquire(L) (4208)
   or speculate(L) (4289) — both issued by the host strictly after token t+1's router drain of that layer,
   and that drain waits the in-order compute queue past every token-t layer-L GEMV. T>1 keeps its barrier.
   If the router drain is ever removed too: keep a per-layer "last reader" event (the final accum_f16 of the
   chunk, 4274) as last_read_ev_[L] and issue the eviction fill as
   `xq_->submit(h.depends_on(last_read_ev_[L]); h.memcpy(...))` at 1754 and 1798 — the mirror image of the
   slot_ev_/slot_seq_ bookkeeping acquire already keeps for fills (1727-1733, 1755-1756).
2. **layer_forward_post drain (4382)** — at T=1 it only still protects h_spos_ (3505-3513: a member buffer
   whose next host write is layer L+1's). The comment's other reasons (4376-4381) are covered elsewhere:
   the cache's geometric-growth frees drain themselves and only when they grow (deepseek4_cache.cpp:376,
   432, 492; rationale 359-373), so the argument at deepseek4.cpp:3341-3349 is redundant; the two-card
   reducer orders the compute queue after its aux-queue H2D device-side via ext_oneapi_submit_barrier
   (expert_stream.cpp:1201); ws_logits_ growth has its own wait (4521-4523); nothing in post's output is
   host-read. Replacement: wait on the h_spos_ memcpy event just before the next host write (normally
   complete), or double-buffer h_spos_ by layer parity, or at T=1 pass the single int32 as a kernel argument
   to ds4_sliding_causal_mask(_vis) and delete the H2D. Then layer L+1's dense prologue queues behind
   layer L's hc_mix with no bubble on either card.
3. **Router readback drain (4056-4058)** — needed (h_ridx_ drives the residency/uniq decision 4186-4195;
   h_rw_ is a kernel argument 4274). But the idle window can be filled: between the router launch
   (4021-4023) and the readback (4056) nothing is enqueued, and the shared expert (sliced 4312-4323;
   mirrored 4359-4370) depends only on ws_norm_ (ready at 3993-4000) and writes ws_shg_/ws_shu_/
   ds4_shared_out(T), which nothing else touches (2163, 3057, 3087, 4316-4323, 4363-4370). Hoisting those
   three launches to just before 4056 hides ~1.3 ms/token per card (72_: shexp 2.6 ms two-card sum) under
   the host's router stall + residency loop + acquire issue, every layer. Ordering preserved: reduction 2
   reads ws_moe_ + ds4_shared_out(T) as one range after the chunk loop (4306-4310); the mirrored add
   (4372) stays after the MoE accumulate.
4. **Window-close drains (3379, 3444)** — exist only because `cp` is stack-local (3376-3378). Replacement: a
   member staging buffer (like h_spos_, sized rate entries) or generate the positions on the device
   (first_window_position + i*ratio; ds4_rope_cos_sin could take base+stride). ~10 drains/token on average.
5. **Orchestrator segments (5455-5570)** — joins are only required where data crosses cards: before
   reduction 1 (5471-5485) and reduction 2 (5545-5555). The post segment (5568, L*3+2) produces ws_streams_
   with no cross-card consumer, so post(L) folds into pre(L+1): 2 joins per layer instead of 3. Candidate
   (a) — mirror attention at T=1 (each card computes all 64 heads of an ~8%-of-wall kernel) — removes
   reduction 1 entirely: 2 fewer D2H waits and one host sum per layer.

## Net per layer at T=1 (default two-card config)
Today: 3 drains per card (router, moe, post) + 3 joins + 4 D2H waits.
After items 1, 2, 4, 5: 1 drain per card (router, with the shared expert hidden under it), 2 joins,
4 D2H waits. With (a) as well: 1 drain, 2 joins, 2 D2H waits.

## Gate for any of this
`deepseek4_residency_test` §15h (eviction safety) + stream-mode NLL bit-identity (ie-ds4-ppl --mode
stream --dump-nll vs baseline) + the 8-length stress at ctx 4096, then throughput (ie-ds4-bench tg at
128/4096/16384) — in that order.

## Measured (2026-09-11 10:11-10:36; H1 = items 3, 4, 2, 1 on commit 13f68e7; the ggml-org MXFP4 file, two cards)
Correction to the inventory above: the DEFAULT MoE route for every T, including T = 1, is the expert-grouped loop
(`moe_expert_grouped`, drain between groups); the token-chunked loop the inventory called "the T=1 route" is the
opt-in reference (IE_DS4_MOE_TOKEN_MAJOR / _TOKEN_CHUNK). Item 1 was therefore applied to the grouped route: the drain
after the LAST group of a layer is skipped when t0 == 0 && nt == 1 (⟺ T == 1); between-group drains stay; the
reference loop is untouched. Item 2 = post drain skipped at T == 1; item 3 = shared expert launched right after
the router drain (both branches, under Ds4SiteScope kShExp); item 4 = member staging for the window-close
positions (two drains gone). The decode barrier census (deepseek4_sched_gate_test §2) now reads router = n_layers,
post = 0, moe = 0 at T = 1 with one group per layer.
Correctness (scratchpad ds4_h1_val.log; every comparison on data rows): abliterated file batch NLL byte-identical
(4.1393); ggml-org batch NLL byte-identical to the route build's dump (3.9476); stream NLL (the T=1 path) run twice,
both byte-identical to the route build's dump (3.9514); single-card (mirrored shared-expert branch) batch NLL
byte-identical OLD (13f68e7, worktree build) vs NEW (3.9297); 64 greedy tokens x3 identical (md5 688f407b7c);
8-length stress at ctx 4096 clean.
Throughput (same lines as the route build): pp 140.7 / 318.5 / 566.5 vs 141.6 / 319.5 / 571.1 tok/s at
128 / 512 / 4096; tg 32.97 / 25.96 vs 32.67 / 26.05 at 128 / 4096; serve 454 / 25.2 / 30.5 vs 463 / 25.3 / 30.5.
FLAT: the four removed host syncs were not on the decode critical path of this file at these depths. After H1
waits_.post and waits_.moe read 0 at T=1 and waits_.other no longer carries the window-close drains, so a host
partition from --kprofile must be read against that; the partition itself (scratchpad ds4_h1_kprof.log) is what says
where the remaining host time is — the router drain, the two reductions' D2H waits + host sums, and the three
per-layer joins (item 5 / candidate (a)) are what is left of the inventory.
Per-card arithmetic (gate, from the profile's counters; inference from code structure and counters, not a timeline):
kprof.gpu_ms 40.1 is the two-card sum (attention.calls 86 = 43 x 2), so per card GPU ~20.0 ms; DMA busy ~15.8 ms
profiled / ~14.6 unprofiled (789.6 MB/token split evenly at 26.97 GB/s); prefetch_issued = 0 (the transfer queue is
never idle, so speculation never fires) and a layer's miss DMA sits between its router drain and its expert GEMMs —
GPU and DMA alternate: ~20 + ~14.6 = ~34.6 of the 38.5 ms/token unprofiled, leaving <= ~4 ms/token (~90 us/layer) as
the ceiling for ANY host-sync removal. What moves the token is the 14.6 ms of expert-miss DMA (hit rate 0.77 at
4K): residency priority / cache size / prefetch that actually fires — a founder-level lever decision.
DISPOSITION (gate PASS on correctness, builder + gate concur): NOT committed. The final patch, criteria, validation
log, profile and verdict are archived under results/deepseek4-hostsync-h1-2026-09-11/; the tree is reverted to
13f68e7 for these files. Revisit after the DMA floor moves; the inventory correction above stands.
