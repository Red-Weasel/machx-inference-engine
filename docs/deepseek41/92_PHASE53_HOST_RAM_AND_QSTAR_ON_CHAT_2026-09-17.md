# V4.1 — Phase 53: what more pinned host RAM buys on chat decode, and the CPU/PCIe split re-swept

Held-out set of docs/89 (6 Dream-style prompts, 90 tool schemas, ~18k context), `ie serve` at ctx 75,000, the chat
ranking and the expert file, a fresh process per arm, decode tok/s summed over the 6 replies.

## Pinned expert arena +10 GiB (`IE_DS41_PIN_CAP_GIB=196.8` against the live rule's 186.8)

+15 experts per layer move from the NVMe tier to pinned RAM (boundary 334/332); the expert file rebuilt at [331, 375)
for the arm (covers 840/1,020 and 880/1,060 = 82 % of the smaller disk tier), restored to [317, 361) after.

| arm | decode tok/s | host MemAvailable floor |
|---|---|---|
| default (186.8 GiB), two runs (docs/90) | 12.22 / 12.52 | ~38-39 GiB |
| **+10 GiB (196.8)** | **13.08** (+5-7 %) | **30 GiB** |

A founder decision, not a default change: the live rule reserves 66.84 GB of total memory against the pinned-RAM
livelock of 2026-08-27, and this spends 10 GiB of that margin (browser, editor and remote-desktop processes on this box
use ~20-24 GB). `IE_DS41_PIN_CAP_GIB` is the knob; the expert file's window must follow the boundary
(`ie-ds41-expert-file --cutoff <static + pinned>`).

## q* (the PCIe share of pinned misses; the rest go to the E-core CPU leg)

| q* | 0.15 | **0.30 (default)** | 0.45 |
|---|---|---|---|
| decode tok/s | 10.53 | **12.63** (12.22 / 12.52 earlier) | 12.05 |

The default holds on chat traffic with the new ranking and file. Closed.
