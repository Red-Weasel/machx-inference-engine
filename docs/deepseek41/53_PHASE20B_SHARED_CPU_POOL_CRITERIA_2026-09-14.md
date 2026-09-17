# V4.1 port — Phase 20b gate criteria: one shared CPU pool for both tiers, assigned by cost (written BEFORE the build)

**The term (docs/52):** with the split under EP at q* 0.60 the step is 80.3 ms/token and the per-layer imbalance is
still 53.5 ms/token of "later" against 23.0 of "earlier" — each tier balances its own link against its OWN six
E-cores, so an expert on the CPU costs 0.90-1.01 ms (docs/35 measured 0.36 on all twelve, standalone) and barely
beats the link's ~0.75; the slower tier's excess stays bounded by half the CPU while the other half idles in
the layers where the faster tier has nothing. A layer has 0-3 misses per tier; in most layers ONE tier has them.

**The change, bounded to the CPU leg and the owner's dispatch.**
1. **One pool:** a single worker (twelve threads on the E-cores 8-19, its team pinned once) with a FIFO of
   requests; a request is (the activation, a list of (arena slot, destination row buffer)). Both tiers post
   into it and wait on their own request's completion; the owner's D2H of x serves both (the same layer, the
   same x). The per-tier workers and the 6/6 core partition go; the readers stay on the P-cores.
2. **Assignment by cost, not q*:** before the two tier calls the owner knows both tiers' misses for the layer
   (`h_idx` and each tier's `is_resident`). With c_link (the measured per-expert link time, ~0.75 ms at 26
   GB/s) and c_cpu (the pool's per-expert time, measured per run and fed back), it moves misses from the tier
   with the larger link time to the pool while the layer's wall — max(link_A, link_B, pool queue) — falls,
   and hands each tier its list (`set_cpu_experts`). The lone-miss case stays on the link unless the pool is
   idle and cheaper. The rounding rule and `IE_DS41_QSTAR` become the fallback when the assignment is off
   (`IE_DS41_CPU_POOL=0`).
3. The stats: the pool's experts per token, its queue wait, its per-expert time; the E[max] line.

**The arithmetic, stated before the build.** In a layer with misses (2, 0): today 1 link + 1 CPU on six cores =
max(0.75, 1.0) = 1.0 ms; with the pool 1 link + 1 CPU at 0.4 = 0.75, or 0 link + 2 CPU = 0.8. With (3, 1): today
2 link + 1 CPU vs 1 link → max(1.5, 1.0, 0.75) = 1.5; pool: A 1 link + 2 CPU, B 1 link → max(0.75, 0.8, 0.75) =
0.8. Over the ~40 layers the later-side sum should fall from 53.5 toward ~40 ms/token: **−10 to −13 ms/token**,
if the pool's per-expert time under the links' traffic is ~0.4-0.5 ms (measured, not assumed: docs/35's 0.36
was standalone; docs/52's 0.9-1.0 on six cores under traffic is the pessimistic anchor: then ~−5).

## Pass criteria

1. No crash; the pool's numbers reported (experts per token, ms per expert under traffic, queue wait).
2. Digits under the bars with the pool on (decode test, EP and off), the golden's greedy tokens; generate PASS;
   resident PASS (the split is T = 1 only: prefill unchanged).
3. Measured on one build against Phase 20's q* 0.60 (the same env otherwise): ≥ 3 ms/token or reverted; the
   E[max] line moved as the arithmetic says or the doc says why not; a second sample of the best.
4. The fallback (`IE_DS41_CPU_POOL=0`) reproduces Phase 20's numbers on the same build.

## Explicitly NOT in this phase

Warming a stream slot with a CPU-served expert (the lost hits of docs/52 item 3 — a follow-up with its own
criteria: it costs link bytes); DSpark; the drafter.

## Results (08:53, one build `4fc867b8…`, the gate-20 defaults: the split ON, q* 0.60 under EP) — REVERTED

| arm | ms/token | tok/s | CPU experts per token, ms per expert | link misses slower / faster | E[max] later / earlier |
|---|---|---|---|---|---|
| EP alone (`IE_DS41_CPU_MISS=0`) | 86.3 | 11.58 | — | 55.2 / 15.3 | 61.0 / 22.9 |
| the per-tier split, q* 0.60 (`IE_DS41_CPU_POOL=0`; two 12-thread teams on the same cores) | 80.6 | 12.41 | 19.5, 0.89 | 39.8 / 14.2 | 53.7 / 24.2 |
| **the shared pool, the cost rule** | **80.2** | 12.47 | 16.0, 0.85 | 40.2 / 15.9 | 53.4 / 24.3 |

Built as written (one worker, twelve threads on 8-19, a FIFO of requests, the owner's greedy cost assignment
with c_link 0.75 and the pool's measured mean as c_cpu) and measured: **0.4 ms/token against the per-tier
split — inside the spread, under the ≥ 3 ms bar: reverted** (criterion 3), the per-tier split with q* 0.60
stays. Why the arithmetic failed: the pool's per-expert time on twelve threads is 0.85 ms, not the 0.36-0.5
assumed — an expert is 18 MB read from the pinned arena while both links pull ~40 GB/s from the same host
memory, so the CPU sees ~20 GB/s and six more threads buy nothing; with c_cpu ≈ c_link the cost rule
reduces to the q* rule (a lone miss stays on the link, pairs split), which is what the counts show (16 vs
19.5 experts). The E[max] term is untouched (53 vs 24 ms/token): the CPU cannot absorb the slower side's
excess at this bandwidth. **What can:** moving the slower side's misses to the OTHER card's link — the two
links together carry ~52 GB/s and the imbalance leaves one of them idle half the time. That needs a tier to
fetch an expert of the other card's share (its arena reachable by both cards' DMA — the mmap staging's
driver import already does this for anonymous memory — or a duplicated hot subset) and to adopt it for the
layer. Phase 21, criteria first. Gate 20 finding 6 (the confinement masks let host threads onto the other
tier's E-cores) stands as a follow-up; the pool would have fixed it as a side effect.
