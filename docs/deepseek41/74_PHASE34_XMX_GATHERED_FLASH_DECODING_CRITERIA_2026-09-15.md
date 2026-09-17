# V4.1 — Phase 34: XMX gathered flash-decoding for the decode attention (criteria before the build)

## Where the decode attention stands, and the probe that chose this

`ds4_attention` is 6.7 ms/token at the 2k prompt (167 us per call, 64 heads, 128 window + 512 picks live) and
7.3 at 32k. The fp32 split kernel is at its measured ladder optimum for this shape (SP = 64; the kernel's own
falsified list covers lane maps, SLM sharing of K, register caching, GRF modes). What it cannot avoid: every one
of the 64 heads re-reads the same 640 live K rows (MLA: K == V, one latent per key), 84 MB per layer per token.

`bench/ds41_attn_decode_probe.cpp` (synthetic data, the decode shape) measured the existing block-sparse XMX prefill
kernel called at decode with the heads as its 32-row tile:

| T, NC | fp32 split, gathered | fp32 split, dense | XMX (kernel only) |
|---|---|---|---|
| 1, 1024 | 194 us (mode 1) / 174 (mode 2) | 174 us | **198 us** |
| 4, 1024 | 658 | 663 | **199** |
| 1, 16384 | 296 | 210 | 2,355 |

XMX vs fp32 output: max rel 3.5e-4, relative L2 2.7e-4. The XMX kernel's time is FLAT in T and LINEAR in blocks:
a serial chain of ~9-11 us per 64-key block (18 blocks at NC 1024, 258 at 16k). One (row, 32-head tile) is one
work-group walking every block, so at T = 1 the card runs 2 work-groups. That is the shape to change, not the math:

1. **Gather** the row's live columns into a fixed 640-column fp16 axis: the 128 window slots in slot order, then
   the 512 picks ascending (sentinels as masked columns), with a compact [T, 640] mask. Per row, per layer:
   640 x 1 KB written, once -- 40 layers x 0.6 MB = 26 MB/token, against 84 MB x 40 read today. Context-independent:
   the 16k case above costs the same as the 2k case.
2. **Split-K**: work-group = (row, 32-head tile, key block); 10 blocks x 2 tiles = 20 work-groups per row, each
   one block deep, writing a partial (m, l, O[32 x 512]) to scratch.
3. **Combine** per (row, tile): rescale to the global max, sum, sinks, 1/l -- the fp32 split kernel's tree in a
   separate launch.

Expected: ~one block latency + combine + gather per layer, ~20-30 us, i.e. ~1 ms/token: **-5 to -6 ms/token at 2k**
(86 -> ~80 ms/token, ~12.4 tok/s) and the same absolute saving at 32k and 200k.

## Criteria

1. **Row identity (P2):** work-groups never mix rows -- `multi` 38/38, `rollback` 74/74, `dspark` 153/153.
2. **Decode gate:** `ie-ds41-decode-test` 28/28 with the XMX decode path on (every layer within its bar, next
   token the golden's, top-5 set equal). This path is fp16-operand attention at DECODE, which the fp32 path never
   was; prefill at T > 16 already is (the XMX prefill kernel is the default there), so the KV state was always
   built through it -- but the decode bar has to hold on its own.
3. **Text:** 120 real decode tokens at 24k of held-out text (`pp33_text_1.log` is the fp32 stream): the needle
   retrieved and the text coherent; identical text is NOT expected (fp16 operands) and is not the bar. Record
   where it diverges.
4. **Speed, like for like:** the profiled decode test's 2k prompt, `ds4_attention` (+ gather + combine, all
   profiled under one name or summed) below 3.0 ms/token (from 6.7); the 32k long test's decode attention below
   3.5 ms/token (from 7.29, docs/71).
5. **Kill switch:** `IE_DS41_ATTN_XMX=0` restores the fp32 split kernel exactly (bit-identical to today).

If criterion 2 fails on precision, the fp32 kernel stays the default and the numbers are recorded; a fp32-score
variant (fp16 V only) is the fallback design, not a weakened bar.

## Results

Build: `src/ops/deepseek41_attn_xmx.cpp` (gather / split-K partial / combine, helpers `ds41dx_`), wired at both
decode attention sites of the forward (the consumer layers with the picks, and the window-only layers with 2
blocks), `IE_DS41_ATTN_XMX=0` the kill switch. The probe first (synthetic, per call, 64 heads):

| T, NC | fp32 split (shipped) | Phase 34 | error vs fp32 (rel L2) |
|---|---|---|---|
| 1, 1024 | 194-264 us | **39.5 us** | 2.7e-4 |
| 4, 1024 (DSpark) | 651 | **59.6** | 2.6e-4 |
| 1, 16384 | 297 | **36.8** | 2.6e-4 |

Then the model (default configuration, the XMX decode path on):

| criterion | result |
|---|---|
| 1. row identity | multi **38/38**, rollback **74/74**, dspark **153/153** |
| 2. decode gate | **28/28**; logits within 8.0e-3 of the golden (bar 5e-2); worst layer 1.7e-2 (bar 3.5e-2; fp32 path 1.4e-2) |
| 3. 24k held-out text, 120 tokens | needle retrieved, coherent; diverges from the fp32 stream after 13 words (a heading choice); 9.74 tok/s (fp32 arm 9.73, one run each) |
| 4a. 2k prompt, profiled decode test | attention **2.05 ms/token** (part 1.32 + combine 0.45 + gather 0.28) from **6.69**; named kernels 40.1 -> **35.5** ms/token; wall 83.2 ms/token = **12.01 tok/s** (fp32 runs 86.1-91.1) |
| 5. kill switch | `IE_DS41_ATTN_XMX=0` takes the fp32 branches untouched |
| 4b. 32k long test, 512 decode steps (`p24/dx32k_1.log` / `dx32k_0.log`) | attention kernel **1.31 ms/token** (from 7.51); attention stage 27.6 -> 22.3 ms; wall **75.7 ms/token = 13.21 tok/s** from 81.9 = 12.21 (**+8.2 %**) |

A trap on the way: the first 32k A/B showed both arms identical to the microsecond because `ie-ds41-long-test` had
not been relinked after the kernel build (only the decode/multi/rollback/dspark tools and the runner were). Relink
every tool that measures before believing an A/B.

**Verdict: PASS on every criterion; the XMX gathered flash-decoding is the default decode attention.** The cost is
a precision class: decode attention now runs fp16 operands with fp32 accumulation, like the prefill has since
Phase K; the decode gate's worst layer moved 1.4e-2 -> 1.7e-2 against a 3.5e-2 bar and the 24k text diverges
from the fp32 stream at a heading choice with the needle intact. `IE_DS41_ATTN_XMX=0` is the fp32 path.
