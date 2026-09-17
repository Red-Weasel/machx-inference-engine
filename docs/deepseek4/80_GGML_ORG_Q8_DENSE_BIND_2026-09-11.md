# DeepSeek-V4-Flash: loading ggml-org's MXFP4 file (Q8_0 dense roles) — 2026-09-11

## Why
The founder wants the engine tuned on a NON-abliterated V4-Flash. The only complete one on this box is
ggml-org/DeepSeek-V4-Flash-0731-GGUF `MXFP4` (2 shards, 155 GB; shard 1 metadata-only, shard 2 holds the 1328
tensors), downloaded 2026-09-10/11 to `/media/<user>/Secondary Drive/models/DeepSeek-V4-Flash-0731-GGUF/MXFP4/`.
Its dtype map differs from the abliterated "NativePreserved F16-F32-MXFP4" file the DS4 port was built on:

| dtype | tensors | roles |
|---|---|---|
| MXFP4 | 129 | routed experts (gate/up/down, all layers) — unchanged |
| Q8_0 | 661 | attn_q_a/q_b/kv/output_a/output_b, ffn_*_shexp, hc_attn_fn, hc_ffn_fn, attn_compressor_{kv,gate,ape}, indexer_compressor_{kv,gate,ape}, indexer.proj, indexer.attn_q_b, token_embd, output, output_hc_fn |
| F32 | 492 | norms, biases, sinks, hc base/scale, ... |
| BF16 | 43 | (per-layer, unchanged roles) |
| I32 | 3 | hash-router lookup tables |

## What failed
`[ds4-note] bind: output_hc_fn.weight: expected dtype F32, file has Q8_0` (ie-ds4-bench run.status=bind_failed).
Every 2-D projection stored as Q8_0 already binds through `Binder::bind_dense` (whitelist incl. Q8_0) and runs on
the engine's PACKED Q8_0 kernels (`dense_packed<kQ8_0>`, `grouped_packed<kQ8_0>`, `gather_embd_packed<kQ8_0>`) —
the UD-Q3_K_XL file exercised that route. Five roles the engine consumes as F32 arrays were bound with a strict
`want = kF32` and uploaded by two lambdas (`f32vec`, `gvec`) that memcpy the file bytes as F32:
`output_hc_fn` (1), `blk.*.hc_attn_fn` (43), `blk.*.hc_ffn_fn` (43), `blk.*.attn_compressor_ape` (41),
`blk.*.indexer_compressor_ape` (21).

## Change (src/model/deepseek4.cpp, tests/unit/deepseek4_q8_f32_bind_test.cpp, tests/CMakeLists.txt)
- `Binder::bind_f32_src`: binds an F32 role from any source dtype the F32 row upload converts (F32, F16, BF16,
  Q8_0, Q6_K, Q4_K); shape/byte-size checks run against the file's dtype; `bind`'s block-geometry check covers
  Q8_0's K % 32. Used at exactly the five sites. Every other F32 role stays strict.
- `f32vec` / `gvec`: a non-F32 source is dequantized on the host through `dequant_rows` (which calls
  `ref::dequant_q8_0_buffer` etc., the bit-exact reference the packed kernels also follow) as ONE row of n
  elements — the tensor is contiguous rows of K/32 blocks — and the F32 image is uploaded. No kernel or runtime
  path changes; the rest of the engine sees exactly what the F32 file gave it.
- Unit test: a 2-D synthetic Q8_0 tensor dequantized as one row equals the per-row dequantization concatenated,
  and the formula y = d * qs on exactly representable scales (through `ref::dequant_q8_0_buffer`).
- Not a variable here: `IE_DS4_DENSE_Q8` (the SoA requantise route only sees F16/F32/BF16 sources).

## Measured (09:06-09:25, two cards, DS4_TP_GPUS=0,1, default config; scratchpad ds4_q8_val.log)
- Abliterated file on the new binary: batch NLL dump byte-identical to the pre-change dump (PPL 4.1393) — the F32
  path is untouched.
- ggml-org file: loads (run.status=ok). Batch PPL over the first 512 wikitext-2 test tokens **3.9170**
  (avg NLL 1.365336), run-to-run byte-identical; stream-mode (T=1 decode path) PPL 3.9579 (avg NLL 1.375713).
  Plausibility only: the abliterated file scores 4.1393 / 1.4205 on the same tokens, so the non-abliterated,
  Q8_0-dense file is 3.9% better in NLL — the expected direction and inside the pre-set +-15% band. No external
  reference exists on this box (local llama.cpp predates deepseek4); "correct" here rests on the untouched F32 path,
  the bit-exact reference dequantizer and determinism, not on an oracle.
- Serve, 14612-token prompt, 64 greedy tokens: coherent summary of the passage, identical across two requests
  (md5 688f407b7c); the second request hit the prompt cache (14610 cached).
- Throughput — THIS FILE'S OWN BASELINE, not comparable to the abliterated file's numbers: pp 73.9 / 92.9 / 66.4
  tok/s at 128 / 512 / 4096 (abliterated: 144 / 330 / 655), tg 21.8 / 18.9 at 128 / 4096 (abliterated: 33.8 / 25.0),
  serve prefill 65 tok/s at 14.6K, decode 18.4 (20.9 with the warm cache). Cause (by construction, not yet profiled):
  every 2-D projection of this file binds PACKED and runs on the packed Q8_0 decode/prefill kernels
  (`dense_packed<kQ8_0>` etc.), while the abliterated file's F16/BF16 dense weights take the oneDNN / XMX prefill
  route and the fp16 decode GEMVs the port was tuned on. Closing that gap (expand Q8_0 dense sources to F16 at
  load, or a packed-Q8_0 prefill GEMM) is the next DeepSeek phase; the host-sync phase (docs/deepseek4/79) follows
  it, with this file's dumps (`ds4nb_base_ppl.tsv`, `ds4nb_base_stream.tsv`) as its bit-identity reference.

## Route change (09:29-09:51): Q8_0 dense sources take the Q8 SoA route — 8.6x prefill, +38-50% decode
Change (src/model/deepseek4.cpp): `ds4_dense_requantises` also accepts Q8_0 sources (K % 32 == 0), and `upload_dense`
consults it BEFORE the packed branch, so a Q8_0 dense weight is dequantized row-wise by the bit-exact reference and
re-blocked into the int8 + fp16-scale SoA by the existing requantiser — the route the F16/BF16 files take. The
detour is exact: for a Q8_0 block (absmax element = +-127 by ggml's quantizer) amax/127 == d in fp32, half(d) == d,
and lround(d*qs / d) == qs, so the SoA image carries the file's own values. Q6_K keeps its packed path.
`IE_DS4_DENSE_Q8=0` keeps every requantisation off; on this file that flips exactly one thing — Q8_0 dense packed
versus SoA — and reproduces the packed baseline's NLL dump byte-for-byte in every data row (gate-verified; the
file's 43 BF16 tensors are the routers, `ffn_gate_inp.weight`, which never pass through `upload_dense` — they are
dequantized to fp32 into `rt.router_w` — and its 492 F32 tensors are all 1-D). Only the tool's header line, which
records the env value, differs between the two dumps; compare NLL dumps on `grep -v '^#'`.
Gate's round-trip scan of every Q8_0 block in the shard (226.9 M blocks): scale and qs recovered bit-exactly for
100% of the blocks on this route; the only non-byte-exact blocks (0.0089%, in hc_attn_fn / hc_ffn_fn, which are
F32-consumed and not on this route) have d = +0 with nonzero qs and decode to exactly 0 everywhere. So the +0.78%
batch / -0.16% stream shift is kernel arithmetic, and an "exact repack" would change nothing.
Measured (two cards, same lines as above; scratchpad ds4_q8route_val.log):
| | packed (before) | SoA route (after) |
|---|---|---|
| pp tok/s at 128 / 512 / 4096 | 73.9 / 92.9 / 66.4 | 141.6 / 319.5 / 571.1 |
| tg tok/s at 128 / 4096 | 21.8 / 18.9 | 32.7 / 26.1 |
| serve 14.6K prompt: prefill / decode / cached-decode | 65 / 18.4 / 20.9 | 463 / 25.3 / 30.5 |
| batch PPL (avg NLL) | 3.9170 (1.365336) | 3.9476 (1.373110), +0.78% |
| stream PPL (avg NLL) | 3.9579 (1.375713) | 3.9514 (1.374068), -0.16% |
| 64 greedy tokens, 14.6K prompt | md5 688f407b7c | md5 688f407b7c (identical) |
Batch and stream dumps run-to-run byte-identical on the new route; the abliterated file's dump byte-identical to
before (its route did not change). The batch shift is the prefill path's oneDNN s8 x fp16-activation matmul
versus the packed kernel's fp32 arithmetic (the same class as the abliterated file's batch-vs-stream gap); the
decode path is within 0.2%. 8-length stress at ctx 4096 (129/512/1023/2048/3000/512/64/4000): all ok, 95-593 tok/s,
no DEVICE_LOST/OOM/nan. This file now sits at the abliterated file's level (pp 571 vs 655 at 4096, tg 26.1 vs
25.0) and is the reference for the host-sync phase (its SoA dumps `ds4soa_ppl.tsv` / `ds4soa_stream.tsv`).
