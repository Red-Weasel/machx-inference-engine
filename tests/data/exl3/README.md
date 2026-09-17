# EXL3 decode oracle test vectors

CUDA-free ground truth for the EXL3 host/SYCL decode (plan
`docs/superpowers/plans/2026-06-19-qtip-quant-engine.md`, Phase B; format spec
`docs/exl3_format_notes.md`).

The binary vectors (`*.trellis`, `*.f16`) are **Llama-3.2-derived weight data** and are
**git-ignored** (this repo keeps model weights out of git; Meta license). Only
`onelayer.meta.json` (shapes/keys, no weights) is tracked. Regenerate locally:

```bash
# 1. one-time tooling (numpy + safetensors, no torch/CUDA)
python3 -m venv /tmp/exl3venv && /tmp/exl3venv/bin/pip install numpy safetensors

# 2. download the smallest EXL3 model (~0.9 GB; per-bpw HF branch)
hf download turboderp/Llama-3.2-1B-Instruct-exl3 --revision 3.0bpw \
   --local-dir ~/models/exl3-test/llama32-1b-3.0bpw

# 3. generate the one-layer oracle (k_proj: K=2048 N=512 bits=3 cb=0)
/tmp/exl3venv/bin/python tools/exl3/make_oracle.py \
   ~/models/exl3-test/llama32-1b-3.0bpw/model.safetensors \
   model.layers.0.self_attn.k_proj  tests/data/exl3
```

Produces in `tests/data/exl3/`:
| file | contents |
|---|---|
| `onelayer.trellis` | int16 LE `[tile_k, tile_n, 16*bits]` — kernel input |
| `onelayer.suh.f16` / `onelayer.svh.f16` | fp16 input/output sign+scale vectors |
| `onelayer.wrot.f16` | fp16 `[K,N]` row-major — **decode-only** oracle (Task 1: trellis decode + tensor-core un-permute, no Hadamard) |
| `onelayer.weight.f16` | fp16 `[K,N]` — full materialized weight (Task 2: + 128-Hadamard + suh/svh) |
| `onelayer.meta.json` | K, N, bits, cb, tile dims (tracked in git) |

The C++ host-decode test (`tests/unit/exl3_decode_test.cpp`) skips gracefully when the
binaries are absent (fresh checkout without regeneration).

**Sanity signal** (printed by the script): `W_rot` std ≈ 1.15 (QTIP unit-Gaussian
codebook); `W_full` std ≈ 0.05 with realistic ±0.6 range — the magnitude profile of a
real Llama `k_proj`, which a wrong decode/permute/Hadamard would not produce.
