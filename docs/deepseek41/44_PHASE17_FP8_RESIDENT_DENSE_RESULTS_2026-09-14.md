# V4.1 port — Phase 17 results: the dense projections resident in their own FP8

**Criteria:** `43_PHASE17_FP8_RESIDENT_DENSE_CRITERIA_2026-09-14.md`. **Starting line:** the Phase 15
build on the re-baselined text with `IE_DS41_EP=1 IE_DS41_EXPERT_FILE=…` at the live cap: decode
110.7-111.3 ms/token (9.0 tok/s), pp2048 390 / 394 tok/s, dense weights 14.45 GiB of fp16 across
the cards.

## Step 1 — the kernel

`gemv_fp8_e4m3_f16` (src/ops/gemv_fp8.cpp): the Q8_0 SoA f16 leaf's shape over the checkpoint's own
bytes — one 16-lane subgroup per output column, each lane the 32-weight blocks strided by 16, two
128-bit loads per block, the E4M3 decode of `ds41_e4m3` (the load-time dequant's, so the weight
values are the same ones the fp16 path reads), 32 fp32 FMAs against the fp16 activation, the
block's E8M0 scale once per block, fp32 accumulation, a subgroup reduction, fp32 out (the oneDNN
path's output type). No repack: the checkpoint's [N, K] byte layout IS the per-column layout the
leaf wants, and the [N/32, K/32] scale table is read as is. `gemv_fp8_test` on the model's seven
shapes: **~1.1-1.4e-7** of the double reference over the decoded weights (bar 1e-5), **1.5-3.2e-7**
against `ds41_dense_dequant_f16` + the oneDNN fp16 GEMM at M = 1 (bar 2e-3) — fp32-exact
agreement, the two paths differ only in summation order; `ds41_e4m3` equals the E4M3 formula on
all 254 finite codes.

## Steps 2-3 — the cache and the forward

`Ds41DenseCache` with `IE_DS41_DENSE_FP8=1` keeps each GEMM-consumed matrix as its FP8 pair
(`Ds41Fp8Mat`: bytes + scales; `wq_a wq_b wkv wo_b sh_w1 sh_w3 sh_w2 idx_wq_b engram_wkv`; `wo_a`
stays fp16 because its consumer is the block-diagonal kernel, the BF16 head / embedding / router
and the fp32 norms are untouched) and a per-card fp16 scratch of the largest matrix serves the
T > 1 GEMMs (`f16()`: dequantise in order on the queue, return the scratch). The forward's nine
projection sites go through `dense_proj`: the FP8 GEMV at T = 1 (named by its own profiler entry),
the scratch + oneDNN at T > 1, the fp16 path when the pair is absent. Default off: byte-for-byte
the previous path.

## Step 4 — measured (one build, 09b11ef; `IE_DS41_EP=1 IE_DS41_EXPERT_FILE` at the live cap; the
2,048-token context; `~/ds41_work/g15fix_p17.sh`)

| | fp16 dense (the Phase 15 path) | **FP8-resident dense** |
|---|---|---|
| dense weights on the cards | 14.45 GiB (6.90 / 7.54) | **9.33 GiB (4.20 / 5.13)** — the head's 1.32 GB stays BF16 |
| static expert slots per layer (card 0 / 1) | 18 / 17 | **22 / 21** (the freed VRAM) |
| decode, steps 4-35 (two samples) | 110.7 (and 111.3 earlier) | **99.6, 103.2** = 10.0 / 9.7 tok/s |
| per token: static / link MiB / disk MiB | 105.3 / 1,373 / 36 | 116.3 / 1,263 / 22 |
| forced digits, decode test (bars 1.2e-2 / 3.5e-2; logits 5e-2) | 9.355e-3 / 1.486e-2 / 7.738e-3 | 9.871e-3 / 1.417e-2 / 6.663e-3 — moved through the Q8 trajectory as expected, all under the bars, the four greedy tokens the golden's |
| forced digits, resident test (T = 12 prefill = the scratch path) | 9.024e-3 / 2.755e-2 / 4.195e-3 | identical (the same math) |
| pp512 / pp2048 cold / warm | 156-158 / 158-161; 391 / 394-395 | 157.3 / 161.4; **390.0 / 394.3** (the per-pass dequant is invisible) |
| the nine converted projections' device time per token (the kernel table, 32 profiled steps) | 19.9 ms (q_b 5.8, o_b 5.7, sh_gate/up/down 5.5, q_a 1.1, engram 1.1, kv 0.6, idx_q 0.2) at ~580 GB/s | **17.9 ms** (`gemv_fp8_e4m3_f16`, 290 calls per token, 61.8 µs each) at ~325 GB/s |
| the fp16 terms that stay | o_a (block-diagonal) 4.7 ms, the BF16 head 2.2 ms | the same |

**The kernel is ALU-bound, not bandwidth-bound — the maturity step this phase leaves open.** Half
the bytes bought only 2 of the 10 ms: the FP8 GEMV moves ~325 GB/s against the fp16 GEMM's ~580,
because its per-byte E4M3 decode (`ds41_e4m3`: three field extractions, a subnormal table, a
select) costs ~8 ops per weight beside the one FMA, where the Q8 leaf it was copied from spends
~2. The other 8 ms came from the freed VRAM: 4 more static slots per layer per card, so the link
carries 110 MiB less per token and the disk 14 MiB less. A packed decode (two E4M3 bytes → one
fp16 pair by bit arithmetic, the subnormal codes fixed with one masked select per pair, the FMA
kept in fp32) is ~3 ops per weight and should take the kernel to the bandwidth floor, ~10 ms per
token — another ~8 ms — with the same 1e-5 unit-test bar (fp32 accumulation kept). That is
Phase 17b, criteria first.

**~8-10% off the decode step (110.7 → ~101 ms/token on the mean of two)** for exactly the reason
docs/43 gave — half the dense bytes per token — plus the freed VRAM's static slots (link bytes
−8%, disk −40%). The bars hold with margin; prefill is unchanged. Load +2 s (47 vs 45: the FP8
pairs upload without the dequant, the scratch allocates lazily).

## The founder-facing statement

| | before Phase 13 | now (`IE_DS41_EP=1`, the expert file, `IE_DS41_DENSE_FP8=1`, live cap) | the goal |
|---|---|---|---|
| prefill pp2048 | 280 tok/s | **390-395** | 400+ |
| pp512 | 110 | **157-161** | — |
| decode at 2,048 ctx | 200 ms (5.0 tok/s) | **~100 ms (10 tok/s); 95.9 with the 17b table decode** | 50 ms (20) |

Decode has doubled since Phase 12 on the same hardware: the block time (Phase 13), expert
parallel (14), the live cap and the expert file (15), the FP8-resident dense (17). What remains
on decode: the attention path at T = 1 (~7 ms of kernels but ~30 ms of wall in the attention
site), the slower card's share of the misses (E[max]), the CPU split under EP (drafted), the head
in Q8 (a precision decision); 20 tok/s single-stream has no path on this hardware. What remains on
prefill: the last 2-3% to 400 is P2P for the staging (an IPC-handle probe is written) or a bigger
tail window (the founder's disk).

Defaults: the FP8-resident dense is opt-in until gate 17 passes; the recommendation is on by
default for V4.1 (the bars hold, the memory is freed, prefill is unchanged).
