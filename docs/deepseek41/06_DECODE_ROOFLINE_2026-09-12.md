# V4.1-Flash decode roofline on this box (2026-09-12)

Arithmetic from **measured** bandwidths, not a measured tok/s. Written before the forward pass
exists so the port's ceiling is known before the kernel work is paid for.

## The one number that sets everything

`num_experts_per_tok = 6`, 40 layers, **17.93 MiB per expert** (w1+w3+w2 at FP4 plus scales,
measured from the shipped tensors). A decode token therefore touches

> **6 x 40 x 17.93 MiB = 4.20 GiB of routed-expert weights.**

Everything else in the text path — attention, shared experts, embed, head, norms, gates, mHC —
is **8.87 GiB total** and sits in VRAM permanently. It costs nothing per token. The decode
problem is entirely "how do 4.20 GiB of experts reach the GPU each token".

## Measured tier bandwidths

| tier | bandwidth | how measured |
|---|---:|---|
| VRAM-resident | — | no transfer |
| host RAM → VRAM (PCIe) | **26.5 GB/s** | prior DS4-Flash campaign on this box |
| NVMe (9100 PRO) → host | **10.3 GB/s** | `scratchpad/nvme_expert_bw.py`, this session |

The NVMe figure is new and is the one that matters for the third tier: random **5.625 MiB**
reads (one FP4 expert plane) with `O_DIRECT` across all 48 shards, so it is the drive, not the
page cache. It saturates at **4 threads** — 5.01 GB/s at 1 thread, 10.16 at 4, 10.32 at 16, no
gain past that. That is well under the drive's ~14 GB/s sequential headline, which is the right
lesson: expert streaming is a random-read workload, and 10.3 GB/s is what it actually gives.
**1,742 planes/s = 581 experts/s = 1.72 ms per expert.**

## Residency, from the measured census

268.95 GiB of routed experts, 15,360 of them.

| tier | capacity | experts | share |
|---|---:|---:|---:|
| VRAM (59.6 − 8.87 core − ~4 KV/workspace) | ~46.7 GiB | ~2,670 | **17.4%** |
| host RAM (~195 GiB usable of 215) | ~195 GiB | ~11,140 | **72.5%** |
| NVMe | remainder | ~1,550 | **10.1%** |

Note how close this is: the text core needs **277.8 GiB** against **274.6 GiB** of VRAM+RAM —
short by about **3 GiB**, not the 55 GiB the original port scope estimated. Nearly everything
can be resident; NVMe is a small third tier, not the main path.

## Calibrating against a model this engine already runs

DS4-Flash: 43 layers, 256 experts, 6 active, hidden 4096, expert_ffn 2048 → 25.17 M params per
expert → 12.75 MiB at MXFP4 → **3.21 GiB of expert weights per decode token**, remarkably close
to V4.1's 4.20. Measured decode on this box is **26.9 tok/s = 37.2 ms/token**, i.e. an
effective **92.7 GB/s** of expert weight delivery — **3.5x the PCIe link**. So most of those
bytes never cross PCIe.

Working backwards: PCIe at 26.5 GB/s can move at most 0.92 GiB in 37.2 ms, so **≤29% of DS4's
expert activations miss VRAM**, while only about **38% of its experts fit in VRAM**. Routing
skew plus the engine's measured `expert_priority` ordering is therefore worth roughly a
**1.9x amplification** of the naive residency fraction — and that is a lower bound, because the
CPU miss-split path carries some of the remainder too.

## Projection

| assumption | PCIe share of 4.20 GiB | NVMe share | ms/token | tok/s |
|---|---:|---:|---:|---:|
| uniform routing (floor) | 72.5% = 3.05 GiB | 10.1% = 0.42 GiB | 131–172 | **5.8–7.6** |
| DS4-like 1.9x skew amplification | ~67% = 2.84 GiB | ~5% = 0.21 GiB | 115–137 | **7.3–8.7** |

**Expect roughly 6–9 tok/s at decode, PCIe-bound.** Prefill is a different regime — batching
amortises each expert load across many tokens — and should land in DS4-Flash's range
(pp 686 tok/s at 2048), since the per-token expert traffic collapses when a batch shares loads.

## What this says about where to spend effort

1. **VRAM-resident expert fraction is the only lever that removes traffic rather than moving
   it.** Every point of VRAM hit rate is worth ~1.5 ms/token. Getting the core under 8.87 GiB
   and the KV budget tight directly buys expert slots.
2. **NVMe is not the bottleneck** at 10.1% of experts — 22–41 ms/token, overlappable. Do not
   over-engineer the third tier.
3. **PCIe is the wall.** 2.8–3.0 GiB/token over a 26.5 GB/s link is ~110 ms no matter how good
   the kernels are. Beating 9 tok/s needs either a higher VRAM hit rate or host-side expert
   compute (the FreeToken q* split already in the engine for GLM).
4. The 3 GiB shortfall is small enough that dropping vision (0.90) and MTP (7.39) from residency
   more than covers it.

**Status of these numbers:** the bandwidths are measured; the expert sizes and counts are read
from the checkpoint; the tok/s figures are arithmetic over those, with the skew factor inferred
from DS4-Flash's measured decode rate. No V4.1 forward pass exists yet, so no V4.1 tok/s has
been measured, and none is claimed.
