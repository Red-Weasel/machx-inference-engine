# V4.1 port — Phase 19 results: the expert-parallel helper's timeline — the join hypothesis falsified, the E[max] term measured

**Criteria:** `50_PHASE19_EP_HELPER_TIMELINE_CRITERIA_2026-09-14.md`. **Starting line:** c3eefb9 (gates 17 + 18 PASS,
the defaults on): 89.3 ms/token on the bare defaults (`IE_DS41_EP=1 IE_DS41_EXPERT_FILE`, nothing else). Logs
`~/ds41_work/p19/`.

## Step 0(a) — the EP timeline, measured (ms per token, summed over the 40 layers, from the MoE call's start)

| build | staging D2H | helper entered | remote tier in / out | rows on the host | owner tier in | import in / out | call end | ms/token |
|---|---|---|---|---|---|---|---|---|
| a `std::thread` per layer (c3eefb9's path; `IE_DS41_EP_THREAD=0` on the new build) | 0.3 | 1.0 | 2.0-2.2 / 39.1-39.3 | 39.8-40.0 | 0.7 | 36.8-37.2 / 56.5-56.9 | 56.8-57.3 | 88.8-89.3 |
| **the persistent helper** (this phase) | 0.3 | 0.7 | 0.9 / 36.9 | 37.4 | 0.4 | 37.0 / 55.3 | 55.7 | **87.7** |

The remote tier enters 50 µs per layer after the owner's staging — there is no late start. What the columns
DID say was true in the sum and false per layer: the remote's rows land at 37.4 and the owner reaches its
import at 37.0 (both summed over layers), yet the import returns at 55.3 — and the import's own copies are
0.7 ms/token (`staging + add` 1.0). docs/48's reading of that gap as the per-layer thread's join was the
first hypothesis and the persistent helper tested it: the join is gone and the gap remains (import in 37.0 →
out 55.3). **The gap is the per-layer E[max] geometry, now measured directly** (the decode bench's new line,
the full decode test's post-replay steps): the sum over layers of the LATER of (the owner at its import, the
remote's rows landed) is **66.6 ms/token**, the EARLIER **23.5** (per-layer thread: 71.2 / 27.0) — in a
typical layer one card finishes its share at ~0.6 ms and the other at ~1.7 ms, because a layer's ~1.75
misses split 2-0 or 1-0 between the parity halves far more often than 1-1. The MoE call pays the later side
in every layer: ~20 ms/token above an even split. That is the term Phase 20 (docs/49) is written for — the
E-cores absorbing the slower side's excess — not a threading fix.

The persistent helper (`Card::ep_thread`, parked on a condition variable, posted the same lambda per layer;
`IE_DS41_EP_THREAD=0` the per-layer thread) is **bit-identical** (the full decode test's dumps 0 of 200
differ, both arms PASS with the digits 9.871e-3 / 1.417e-2) and measured **−1.1 ms/token on the bench**
(88.8 → 87.7) and −7.8 on the full test's post-replay steps (99.7 → 91.9; that state has more misses and a
wider spread) — kept, ON by default.

## Step 0(b) — docs/48's P0: the routing union, measured (36 consecutive decode steps of the pp2048 text)

| T consecutive steps | union per layer (of 6T selections) | static | non-static (pinned + mmap) | shuffled steps' union | non-static per layer per TOKEN |
|---|---|---|---|---|---|
| 1 | 6.00 | 2.95 | 3.05 | 6.00 | 3.05 |
| 2 | 10.18 | 4.66 | 5.51 | 10.73 | 2.76 |
| 3 | 13.92 | 6.10 | 7.82 | 14.83 | 2.61 |
| 4 | 17.05 | 7.22 | 9.83 | 18.70 | 2.46 |
| 6 | **22.69** | 9.14 | 13.55 | 25.10 | **2.26** |

The non-static experts a verification pass would fetch fall from 3.05 per layer per token at T = 1 to 2.26
at T = 6 (consecutive; 2.51 shuffled — the routing correlation is worth ~10% on top of the within-pass
overlap). docs/48's central case assumed 1.79 → 2.81 link misses per layer per token; the measured
non-static count already sits below its T = 6 figure before the stream slots' cross-pass hits are counted,
so the fetch does NOT anti-amortise — DSpark's optimistic column (20 tok/s at an acceptance length ≈ 4 of 6)
is the one this measurement supports. (36 steps is one bench; P0 asks for ≥ 200 — the dump is cheap and the
DSpark phase's criteria will take a longer run.) `union.py` and the dump are beside the logs.

## Step 0(c) — the attention kernel at T = 1 vs T = 6: not run (the term was resolved without it; docs/48 F.3 stays open for the DSpark phase).

## Founder table

| | Phase 18 (docs/47) | **now** (defaults) | the goal |
|---|---|---|---|
| decode at 2,048 ctx | 88.1-88.6 ms (11.3) | **87.7 ms (11.4 tok/s)** | 50 ms (20) |
| prefill pp2048 | 407 / 414 | unchanged by construction (the EP block is the same code; not re-measured this phase) | 400+ (met) |

The next term is named and measured: ~20 ms/token of per-layer imbalance between the two cards' expert
shares. Phase 20 absorbs it on the E-cores (criteria docs/49); DSpark (docs/48) multiplies whatever remains.
