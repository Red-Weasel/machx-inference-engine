# Performance baseline — 2026-06-09 (verified head-to-head vs llama.cpp Vulkan)

First same-day, same-hardware, same-GGUF comparison against llama.cpp's
**Vulkan** backend (all earlier docs compared against *reported* SYCL-backend
numbers). Headline: **the engine beats llama.cpp Vulkan on token generation
by ~18%**; prefill remains 4.4× behind and is the only open perf gap.

## Configuration
- Model: `/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf` (both engines, identical file)
- Device: Intel Arc Pro B70 (BMG-G31, Xe2-HPG), level_zero:0 / Vulkan device 0
- Engine: working tree at commit `8f72ac6` + uncommitted 2026-05-05/06 optimization set
  (XMX M_TILE=16, shared-expert prefill batching, etc.), KV fp16, all-GPU
- llama.cpp: Vulkan build b8902 (`/home/weezy/llama.cpp-vulkan/llama-b8902/`),
  `llama-bench -ngl 99 -p 512 -n 128 -r 3 -sm none -mg 0`
  (pinning to B70 only; default dual-device run gave the same numbers: 39.78 tg / 884.6 pp)
- Engine decode: `./build/tools/ie-bench-suite` (5-prompt fixed suite, greedy, 128 toks, 3 runs)
- Engine prefill: `./build/tools/ie-bench --ctx 1024 --prefill 512 --decode 0 --profile`

## Quality gate
- Built-in 511-tok PPL: **6.54** (`ie-perplexity`, fp16 KV) — exactly at the
  historic 6.51–6.55 baseline, ±0.03 noise floor.
- **Standing rule: every perf change must hold PPL ≤ 6.57 on this corpus
  before it is kept.**

## Head-to-head

| metric | engine | llama.cpp Vulkan b8902 | ratio |
|---|---:|---:|---:|
| token generation (tg128) | **46.6–46.9 tok/s** | 39.63 ± 0.16 | **118% ✓** |
| prefill (pp512) | **202.9 tok/s** | 885.94 ± 5.32 | 23% ✗ |

### Engine decode detail (ie-bench-suite, median of 3)

| Prompt | Prompt toks | Median tok/s | Spread |
|---|---:|---:|---:|
| short-chat | 51 | 46.91 | 0.1% |
| long-instruction | 80 | 46.89 | 0.2% |
| codegen | 78 | 46.84 | 0.0% |
| math-reasoning | 86 | 46.67 | 0.3% |
| long-context | 219 | 46.64 | 0.2% |

Decode top-5 kernels (stable across all prompts): `gemv_q6k_huge` 23%,
`gemv_q4k` 16.4%, `gemv_q6k_med` 12.4%, `moe_dec_gate_q4k` 9.1%,
`moe_dec_down_q6k` 6.8%. Remaining decode headroom is mostly the Q6_K
lm_head (`gemv_q6k_huge`); the PR #4 Q4_K-lmhead GGUF variant trades
+0.10 PPL for cutting it ~31%.

### Engine prefill profile (T=512, total 3469 ms)

| section | ms | % |
|---|---:|---:|
| attn_dn_block | 946.5 | 27.3% |
| moe_routed_multiexpert | 900.9 | 26.0% |
| moe_shared | 379.3 | 10.9% |
| dn_alpha_beta_gate | 326.6 | 9.4% |
| dn_qkv_proj | 308.6 | 8.9% |
| attn_full_block | 289.7 | 8.3% |
| dn_ssm_out | 256.5 | 7.4% |
| dn_recurrence | 45.6 | 1.3% |
| (rest) | ~15 | <0.5% |

## Deltas vs 2026-05-04 baseline
- Decode: 44.7 → 46.8 tok/s (+4.7%)
- Prefill pp512: ~150 → 202.9 tok/s (+35%, the 2026-05-05 optimization set)
- PPL: 6.54 → 6.54 (unchanged)

## Notes
- llama.cpp Vulkan pp512 (886) is substantially higher than the ~615 SYCL
  figure older docs cite — the true prefill gap is 4.4×, not 3×.
- `ie-bench --help` is not handled: unknown flags are ignored and the tool
  proceeds to a full ~20 GB model load. Two concurrent tools = 2 models =
  full 32 GB VRAM. Kill strays before benchmarking; idle VRAM is ~540 MiB.
