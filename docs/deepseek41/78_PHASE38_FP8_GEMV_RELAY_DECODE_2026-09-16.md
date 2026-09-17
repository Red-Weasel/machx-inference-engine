# V4.1 — Phase 38: the FP8 GEMV without its table (criteria, then the bench, then the gate)

## Criteria (before the build)

The dense FP8 GEMVs are the largest kernels on the serial attention stage: q_b 4.2, o_b 3.9, o_a 3.0 ms/token at
2k, at ~400 GB/s where the fp16 GEMV reached 578. The suspect is the per-weight decode: variants 1-3 look each
byte up in a 256-entry local-memory table, one SLM load per weight (4x the weight bytes in SLM traffic, random
banks). The change: an exact decode with no table, bit-identical by construction; chosen per shape by measurement.

1. Bit-identical: every variant equal to the scalar decode on random bytes covering all 256 codes (bench), and
   the profiled decode test's criterion lines identical to Phase 35's.
2. Faster where it is used: GB/s per shape on a bench that STREAMS from VRAM (a matrix re-read from L2 says
   nothing about a token that reads 2 GB once); the dispatch takes the winner per shape.
3. The 2k profiled decode test: q_b + o_b + o_a below 10 ms/token (from 11.2), wall recorded.

## The decode

An E4M3 byte's low seven bits shifted left by 7, with the sign at bit 15, is the fp16 bit pattern of value / 256:
for normals the exponent field carries over (bias 15 vs 7 is the 2^-8), and for subnormals E4M3's m * 2^-9 lands
as fp16's m * 2^-17 -- the same 2^-8, because both formats put the subnormal boundary at exponent field 0 with the
mantissa in the same relative place. So two bytes of a 32-bit word relay in one AND/shift/OR each (bytes 0/2, then
1/3), two half -> float conversions follow, and the block's E8M0 scale is multiplied by 256 (exact). Every product
and every partial sum is the table variant's scaled by a power of two -- the same bits. The two NaN codes (low
seven bits all ones) are detected per word and take the scalar decode; weights never hold them.

## The bench (`bench/ds41_fp8_gemv_bench.cpp`, 100 launches per point, rotating over >= 256 MB of copies)

| shape | MB | floor (plain read) | dec0 scalar | dec1 table fp16 (was default) | dec2 | dec3 | **dec4 relay** | identical? |
|---|---|---|---|---|---|---|---|---|
| q_b 1280x32768 | 41.9 | 289* | 185 | 364 | 380 | 380 | **416** | all |
| o_b 8192x5120 | 41.9 | 563 | 296 | 379 | 389 | 388 | **455** | all |
| o_a 4096x8192 (grouped) | 33.6 | 560 | 319 | 411 | 425 | 426 | **480** | all |
| sh_gate 5120x2304 | 11.8 | 447 | 203 | 270 | 279 | 281 | **306** | all |
| sh_down 2304x5120 | 11.8 | 530 | 248 | 336 | 350 | 350 | **391** | all |
| q_a 5120x1280 | 6.6 | 492 | 225 | **388** | 389 | 389 | 302 | all |

(*) the plain-read kernel is itself short-loop bound at K = 1280; the GEMV beats it. Without the rotation the
small shapes read 1.9-2.2 TB/s -- L2 -- which is the trap the bench first fell into.

The relay decode wins 13-20 % on every shape with >= 2304 columns and LOSES 22 % on q_a's 1280 columns, where
80 work-groups cannot hide its extra ALU work. **Dispatch: N >= 2048 -> variant 4, else variant 1**;
`IE_DS41_FP8_PACKED=<0..4>` still forces one variant everywhere for A/B.

## In situ (the profiled decode test, 2k prompt, `p24/p38_decode.log` vs `w35_decode.log`)

Criterion 1: 28/28 and every criterion line identical to Phase 35 -- bit-identical. Criterion 3: q_b + o_b + o_a
**9.0 ms/token from 11.2**.

| kernel | Phase 35, avg us | Phase 38 | change |
|---|---|---|---|
| q_b | 103.9 | **80.6** | -22 % |
| o_b | 98.8 | **77.0** | -22 % |
| o_a | 76.8 | **68.2** | -11 % |
| sh_gate / sh_up / sh_down | 39.8 / 39.9 / 32.3 | 31.1 / 30.2 / 25.5 | -21 to -24 % |
| q_a (table, by the rule) | 17.1 | 17.4 | — |
| named kernels, ms/token | 35.5 | **32.4** | -3.1 |
| wall, ms/token | 80.8 = 12.37 tok/s | **78.4 = 12.75** | |

Larger in situ than on the bench: the token's other kernels share the SLM the table lived in, and every
work-group filled a table before its first load.

Gates: multi **38/38**; the 32k long test (default mode) **64.2 ms/token = 15.58 tok/s** from 67.1 (Phase 35) -- the
shared-expert GEMVs sit partly on the critical path too. `ie-ds41-dense-cache-test` aborts in FP8 dense mode
("NULL pointer argument in memory copy") with the OLD default decode as well (`IE_DS41_FP8_PACKED=1`) and passes
6/6 with `IE_DS41_DENSE_FP8=0`: a pre-existing gap in that test (its fresh-dequant comparison reads an fp16 pointer
the FP8 mode leaves null), not this phase; noted, not fixed here. The bench lives in `bench/` (untracked in this
repo) with the attention probe.

## Falsified after the gate: two columns per sub-group (removed)

A relay-decode variant giving each sub-group TWO columns (both columns' block loads issued together, the
activation block staged once, separate accumulators -- bit-identical), meant for the short-K q_b where a lane holds
only 2-3 blocks. Streamed bench: q_b 419 -> 430 GB/s (noise), o_b 456 -> 355, o_a 480 -> 446, sh_down 391 -> 296,
q_a 305 -> 193. The doubled live state (two columns' loads and accumulators plus the 32 staged activations) costs
more than the extra loads in flight buy; removed the same hour. The single-column relay stays.
