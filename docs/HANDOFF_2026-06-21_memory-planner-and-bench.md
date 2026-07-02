# Handoff — 2026-06-21 — gemma4 SoA-only + VRAM planner + bench caveat

## What shipped this session (all committed, tree clean)

1. **`78cd854` — gemma4 SoA-only resident layout (fixes 31B single-card OOM).**
   The fast Q4_0 path kept BOTH an AoS and a SoA copy of every dense projection
   (AoS for the onednn prefill dequant, SoA for decode) → ~2× projection VRAM. On
   31B that peaked **32.6 GB → DEVICE_LOST crash** on a 32 GB B70. Fix: build SoA
   from the HOST GGUF (`repack_q4_0_to_soa` reads `t->data`) and skip the AoS
   device upload; new `dequant_q4_0_soa_to_Bt` feeds the onednn prefill GEMM
   **bit-identically**. 31B now loads at **20.2 GB** (fits a 24 GB card), 26B at
   16.3 GB. `IE_GEMMA4_KEEP_AOS=1` restores both copies for A/B. crown 6.4527
   bit-exact. **Perf-neutral** (26B A/B: SoA 93.7 vs AoS 87.8 tok/s, back-to-back).

2. **`8483c36` — VRAM-aware placement planner (`plan_placement`).**
   Senses per-card VRAM (`global_mem_size×0.90`), estimates footprint (Σ tensor
   bytes + per-arch KV reserve + 1.5 GB ws), auto-picks single vs multi-GPU.
   `--gpus 0` (new CLI default) = auto; `--gpus N` forces. Logs one `[mem-plan]`
   line. Validated via `ie run` (no `--gpus`): gemma4-31B → "fits 1 card", 80B
   Coder-Next → "split across 2 GPUs", both coherent.

3. **`b336585` — CRITICAL multi-GPU fix: device filter `"0xe223"` → `"B70"`.**
   The hardcoded `"0xe223"` matched ZERO current-driver device names
   (`Intel(R) Arc(TM) Pro B70 Graphics`). Single-GPU survived via a fallback;
   `DeviceFleet` has none → **all multi-GPU was silently broken**. Now `"B70"` +
   `IE_GPU_FILTER` env override. See `memory/project_gpu_filter_drift.md`.

4. **Tier C (universal SoA-only) = NO-OP by investigation.** gemma4 was the ONLY
   arch with a redundant *device* AoS+SoA double. qwen35_dense sets `dq={}` when
   SoA built (line 340); crown qwen36 uploads only SoA from host mmap; dense/
   qwen3moe are single-copy. No changes to validated/gate arches.

5. **Tier B (gemma4 multi-GPU) = DEFERRED.** 31B fits one card with ~11 GB
   headroom + tiny SWA KV → never needs 2 cards in practice. Not worth a new
   layer-split class at publishing time.

## ✅ RESOLVED — NO regression; gemma4-31B beats llama both axes (clean re-bench)

The "3 tok/s decode regression" was **100% a thrashed box**, not the code. After a
reboot (swap 0, GPU idle-clocked 517 MHz, 25 GB RAM free):
- 31B **prefill 307.9 tok/s** (T=677) vs llama-Vulkan 203 = **1.52×**.
- 31B **decode 17.6 tok/s** (stable 17.55 / 17.60) vs llama 15.35 = **1.15×**.
- Both axes beat llama; numbers snap back to ~historical (367 / 20.5; slightly under
  only because the box was ~5 min post-boot, clocks still ramping).
- **SoA-only change is perf-NEUTRAL** (26B A/B + clean 31B). The earlier-flagged
  `gemv_q4_0_soa` "7% BW" was a thrashed-box artifact.
- The `q.wait()`-removal experiment was **reverted** (no benefit, even on a clean box).

**LESSON (added to memory):** never benchmark on a thrashed box. Verify `swap` used
≈ 0 and GPU idle-clocked (~500 MHz, low W) BEFORE trusting any number. Identical runs
varying >1.5× = box state, not signal.

## Clean-box bench vs CURRENT llama (b8902), single GPU — gemma is now CONTESTED

| | Ours | llama Vulkan b8902 | Verdict |
|---|---|---|---|
| 31B prefill (~T512) | ~262 | 254.9 | ≈ parity |
| 31B decode (tg128 sustained) | 16.9 | **19.0** | llama +12% |
| 26B prefill (pp512) | 585 | **1064** | llama +82% |
| 26B decode (sustained) | 42–47 | ~26 (noisy) | ours wins |

**No regression** — the "16%" was burst (single-step 20.4 ≈ historical 20.5) vs
sustained (128-step 16.9, heat-soak). llama's Vulkan simply got faster (newer build).
**The "gemma beats llama both axes" claim no longer holds vs current llama.**

### Decode optimization roadmap (clean profile, 31B, 49 ms/token GPU-bound)
- `gemv_q4_0_soa` = **81.9%** (40.2 ms, 410 launches) at **~67% BW** (15.3 GiB
  projections ÷ 40.2 ms ÷ 608 GB/s). Near its 57–80% design band — the main lever.
- Overhead ~18%: rms_norm 6.7% (421 launches), lm_head gemv_q6 5.6%, attn 2.5%,
  quant_q8_1 1.2%.
- **Tried + REVERTED (no benefit):** native `sycl::half` decode in the gemv hot loop
  — IGC already lowers the manual `q40_fp16_to_fp32` to native HW conv. The kernel is
  NOT ALU-limited on the fp16 decode.
- **The real lever (not yet done):** push `gemv_q4_0_soa` 67%→~80% BW via a
  coalesced memory layout — today a WG's 32 columns sit `K/2` apart in the SoA
  stream (32 scattered read streams). An **interleaved-column repack** (consecutive
  memory = the 32 columns a WG consumes) should lift BW → ~+12% decode → beats
  llama's 19. Real kernel+repack rewrite; cross-arch (also helps any Q4_0 SoA arch).
- Secondary: fuse rms_norm→quant_q8_1; reduce the 410→fewer gemv launches by fusing
  Q/K/V (concatenated-N) projections.
- Prefill (26B) gap is a separate problem: llama's Vulkan KHR_coopmat matrix-engine
  GEMM (1064) vs our onednn/dp4a (585) — needs better XMX utilization on the MoE
  expert GEMMs.

## NEXT (on a CLEAN, rebooted box)
1. One **warm `ie-bench`-style** pass per model (no cold back-to-back loads):
   gemma 26B/31B + crown + 27B + 80B vs llama SYCL/Vulkan.
2. If gemma decode is still low → **A/B the oneAPI compiler 2025.3 vs 2026.0**
   (both installed under `/opt/intel/oneapi/{compiler,dnnl}/`) on `gemv_q4_0_soa`.
   That's the cheapest test of the "kernel regressed with the toolchain" theory.
3. Stand up **OpenVINO** for the publishing comparison (per
   `docs/PUBLISHING_READINESS_2026-06-21.md` P0/P1 list).

## System state (2026-06-21)
kernel 6.17.0-35 · NEO/L0 GPU 26.18.38308.4 · IGC 2.32.7 · oneAPI compiler+dnnl
2026.0 (2025.3 also installed) · L0-V2 adapter (UR) · 30 GB RAM (the bottleneck) ·
2×B70 (32 GB each) via `xe`. libze 1.28.0 (1.28.2 available, minor).
