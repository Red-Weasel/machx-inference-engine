# V4.1 port — Phase 20 results: the CPU miss split under expert parallel, the CPU experts' rows exported

**Criteria:** `49_PHASE20_CPU_SPLIT_UNDER_EP_CRITERIA_2026-09-14.md`. **Starting line:** d3f0b88 (Phase 19: the
persistent helper), 87.7-88.3 ms/token on the bare defaults. Logs `~/ds41_work/p20/`.

## The change (deepseek41_experts.{hpp,cpp}, bounded to the CPU leg)

The CPU leg's product is no longer a weighted fp32 partial added into `y` (null in export mode: the
DEVICE_LOST of docs/47). Each CPU expert's output is written as its packed fp16 row (`CpuMiss::h_rows`,
pinned host, in work order; the worker converts `out` to half), its routing weight stays real, and after
the worker's join the rows are copied into `bws_.yp` at their packed positions — before an exporting
tier's `q.wait()` or the owner's import + scatter. From there a CPU row is a GPU row to the export
protocol and to the one scatter. The zeroing of the weight and of the row, `d_part`, the partial vector
and the `y += dp` add are gone; the export-mode guard of c771120 is gone with them (the path is valid).
Both tiers split their own misses on their own six E-cores (`split_cores`: 8-13 / 14-19); q* per tier
(`IE_DS41_QSTAR`, the PCIe share, rounded per layer).

## Step 0 — the guard build (c771120 + Phase 19) with the split under EP: not run separately; the rows
build ran directly and did not crash in any arm (five bench arms, the full decode test, generate).

## Measured, one build (`fe05d7a6…`; `IE_DS41_EP=1 IE_DS41_EXPERT_FILE IE_DS41_CPU_MISS=1`, the defaults of
gates 17-19 on, the 2,048-token context, steps 4-35; the remote tier's CPU experts are counted since this build)

| arm | ms/token | tok/s | link misses slower / faster | CPU experts per token, ms per expert, the leg's wall | stream hits | E[max] later / earlier |
|---|---|---|---|---|---|---|
| EP alone (this build) | 88.3 | 11.32 | 55.2 / 15.3 | — | 52.0 | 62.6 / 22.9 |
| q* 0.30 | 84.7 | 11.80 | 20.4 / 2.8 | 61.4, 0.92, 45.8 | 37.6 | 57.1 / 25.6 |
| q* 0.45 | 82.9 | 12.07 | 21.5 / 2.8 | 60.0, 0.90, 43.1 | 37.8 | 55.2 / 25.7 |
| **q* 0.60** | **80.3** | **12.45** | 39.8 / 14.2 | 19.5, 1.01, 22.6 | 48.7 | 53.5 / 23.0 |
| q* 0.75 | 85.6 | 11.69 | 50.7 / 15.6 | 4.7, 1.52, 9.6 | 51.3 | 59.2 / 23.3 |

**88.3 → 80.3 ms/token (−9%, 12.45 tok/s) at q* 0.60.** What the sweep says:

1. **The rounding makes q* coarse.** A tier sees 0-3 misses in a layer; `n_pcie = round(q* · m)` puts a
   lone miss on the CPU for q* < 0.5 and on the link for q* ≥ 0.5, splits a pair for 0.25 ≤ q* < 0.75,
   and so 0.30 and 0.45 are the same policy (61 vs 60 CPU experts). 0.60 is the rule the measured costs
   ask for — a lone miss on the link (0.75 ms) rather than the CPU (1.0 ms), pairs split, triples 2 + 1.
2. **The CPU leg is not free.** On six E-cores an expert takes 0.90-1.01 ms (docs/35: 0.36 on twelve,
   standalone); the wake-up adds ~0.15 ms per request. At q* 0.30 the CPU took 61 experts per token and
   its leg (46 ms) became the longer one; the links idled.
3. **CPU-served experts leave no stream slot behind.** The stream-slot hits fall 52.0 → 37.6 at q* 0.30
   (48.7 at 0.60): an expert computed on the CPU is not in VRAM for the next step, so the total of
   non-static work rises (link + CPU 81-84 per token at 0.30 against 70.5 misses before). The lever
   trades link bytes for CPU work AND for later hits.
4. **The per-layer imbalance is only partly absorbed** (E[max] 62.6/22.9 → 53.5/23.0): each tier
   balances its own link against its own six cores; the slower tier's excess is still bounded by its
   half of the CPU. A shared twelve-core pool serving whichever tier has the misses (0.36-0.5 ms per
   expert) with a cost rule instead of q* is the next bounded step (Phase 20b).

## Correctness (appended as the runs land)

- **The full decode test with the split at q* 0.60 (EP): PASS**, forced digits card 0 worst **1.055e-2** (bar 1.2e-2;
  9.871e-3 without the split — the CPU's fp32 activation against the GPU's Q8, docs/35, now on the owner's card
  too), card 1 **1.373e-2** (bar 3.5e-2; 1.417e-2 before), logits **6.008e-3** (bar 5e-2), the first own-router flip
  at layer 10 at the same near-tie, the four greedy tokens the golden's, hashes exact. Not bit-identical to the
  no-split path, as docs/49 said; the bars judge and hold — card 0's margin is now 12%.
- **Generate PASS** with the split: the golden's continuation, the continuity KL 0.0001 / 0.0000 / 0.0062 / 0.0016
  with the same argmax, the exact path's first four tokens, 83 ms/token over the greedy run.
- **Resident PASS** with the split (digits the no-split path's: 9.024e-3 / 2.755e-2 / 4.195e-3 — the split is T = 1 only),
  but the first sample read pp2048 **390.4 / 394.2** against 403-414 in the four runs before it: docs/49's scope 3 — the
  core partition confines the host thread and every thread it spawns to the P-cores, the prefill's readers included.
  Fixed as docs/49 said: at T > 1 the reader thread is spawned with every core allowed (the host thread's mask lifted
  around the spawn and restored), T = 1 keeps the partition. Re-measured: **pp2048 399.5 cold / 404.6 warm, pp512
  159.5 / 163.8** — within the spread. The decode bench on that build: **80.0 ms/token (12.51 tok/s)**; the second
  sample of q* 0.60 on the previous build 82.5 — the arm's spread is ~2 ms.

## Founder table

| | Phase 19 (docs/51) | **now (`IE_DS41_EP=1 IE_DS41_EXPERT_FILE IE_DS41_CPU_MISS=1 IE_DS41_QSTAR=0.60`)** | the goal |
|---|---|---|---|
| decode at 2,048 ctx | 87.7-88.3 ms (11.3-11.4) | **80.0-82.5 ms (12.1-12.5 tok/s)** | 50 ms (20) |
| prefill pp2048 | 403-414 | 399.5 / 404.6 (the readers unconfined at prefill) | 400+ |

Default: the split stays **opt-in** (`IE_DS41_CPU_MISS=1`, q* 0.60) until the gate — it moves the forced digits
(within the bars) and it costs the stream slots their CPU-served experts; the decision on the default is the gate's
and the founder's. What remains of the imbalance is 52.9 vs 23.1 ms/token — Phase 20b (docs/53): one shared
twelve-core pool assigned by cost.
