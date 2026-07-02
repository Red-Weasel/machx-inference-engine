# Known Bugs and Constraints

Last updated: 2026-06-10

## 1. Stochastic non-determinism in `deltanet_recurrence` on Xe2 BMG-G31

### Symptom
When `deltanet_recurrence` is called many times in succession with bit-identical inputs and bit-identical prior state (verified across two parallel chains), the kernel occasionally produces a single-byte difference in its output `state[]`. Once that difference exists, the recurrence amplifies it on every subsequent iter and the model's quality collapses (`--prefill-chunk 1024` PPL = ~74 vs baseline ~6.55).

The bug fires stochastically. Across 8 parallel chains (`ie-bug-monitor`) the first divergent chain typically appears between iter 7 and iter 500; the divergent DN layer index also varies (L=3, L=9, L=15, L=20, L=22 all observed in different runs).

### What this is NOT
- Not a quantization bug (PPL on short corpora matches historic baselines exactly).
- Not a tokenizer mismatch (tokens are byte-identical to llama.cpp's).
- Not a missing init / uninitialized-memory bug (Step 8 ruled this out for `ws_attn_out_`).
- Not an SLM bank-conflict / FMA-pipelining bug (Steps 18, 19 ruled out).
- Not a missing memory ordering primitive (Steps 21–23 with seq_cst+system atomics, atomic_fence, and L0 DMA round-trip all FAILED to fix it).
- Not a state[]-side cache coherency bug (Step 25: USM host bypasses ALL GPU caches; bug persists).
- Not arithmetic drift small enough to clamp (Step 26: fp16 round-trip on writeback amplifies the bug, doesn't fix).
- Not a long-internal-loop drift (Step 24 chunking T=64 made the cascade 4× worse).
- Not in the recurrence kernel's reads of its workspace inputs at the captured (DN=22, iter=4) target — verified bytewise via Step 27 trace.

### What it likely IS
HW-level FMA pipeline non-determinism on Xe2 BMG-G31, timing-coupled to kernel-launch microarchitectural state. Software-irreducible from the SYCL layer.

**Citation correction (2026-06-09):** llama.cpp issue [#21893](https://github.com/ggml-org/llama.cpp/issues/21893), previously cited here as a sister case, was **closed 2026-05-04 as a llama.cpp software bug** (Q8_0 reorder missing a reorder-aware dequantizer in the GEMM prefill path, fixed in PR #21638) — it is NOT evidence of BMG DPAS hardware non-determinism. Better third-party corroboration of systemic BMG-G31 SYCL numerical/stability issues: [ginkgo #2018](https://github.com/ginkgo-project/ginkgo/issues/2018) (preconditioner numerical failures on BMG-G31) and [OGL #170](https://github.com/hpsim/OGL/issues/170) (DEVICE_LOST in IC/ICT solvers on Arc Pro B70). Use those if filing the IGC issue. Also worth testing before filing: Intel's production-blessed B70 stack pins **NEO 26.09.37435.12 + GuC 70.60.0** ([llm-scaler vllm-1.4](https://github.com/intel/llm-scaler/releases/tag/vllm-1.4)) — different NEO than ours (26.14); a repro attempt on that combo (or NEO 26.18/IGC 2.34.4, mind the IGC #412 RA regression) would strengthen or kill the HW hypothesis.

### Production workaround
The bug only manifests when `deltanet_recurrence` is run for many iters in succession (long single-call prefill or long streaming decode). Production paths chunk prefill externally at T=256 and the bug does not fire. Built-in 511-tok PPL = 6.55, decode = 50 tok/s. Both are at expected baseline.

### Per-arch cap (2026-06-20 UPDATE): the WHOLE gated-DeltaNet family now runs at T=512
**As of 2026-06-20 the crown (`qwen36`/`kQwen35Moe`) and the 27B (`qwen35`/`kQwen35Dense`) are RAISED to T=512** (joining qwen3next), env-revert via `IE_QWEN35_PREFILL_CHUNK`. **The §1 bug is NOT REPRODUCIBLE on our current stack** — NEO **26.14 AND 26.18** (userspace updated 06-20; GuC firmware `bmg_guc_70.bin` Apr-26 + kernel `6.17.0-35` unchanged, i.e. the exact diagnosis stack). Evidence: `ie-bug-monitor --max-iters 1024` (8 chains) **×3 passes = 0 divergence / 24 clean chains**; crown PPL `--prefill-chunk 512` **×3 bit-identical (16.27)** + no collapse vs 256 (18.52→512 16.27→1024 13.43, the decrease is the scored-token artifact, not the bug); 27B-Q6 PPL `--prefill-chunk 512` **×3 bit-identical (15.68)**; crown `ie run` 563-tok prompt → 701.9 tok/s prefill, coherent. NOT a "26.18 fixed it" claim (it didn't fire on 26.14 either) — it is "no longer reproducible." **T≥1024 still NOT validated** (one clean run only; needs the repeatability treatment). This does NOT lift the §1 do-not-re-investigate-root-cause rule.

### (prior) Per-arch cap (2026-06-14): qwen3next is empirically clean at T=512
**`qwen3next` (Qwen3-Next-80B) is empirically clean at T=512** and uses it in production (`Engine::generate`, gated on `next_`; env override `IE_QWEN3NEXT_PREFILL_CHUNK`). Evidence (`ie-qwen3next-ppl --sweep --repeats 25`, 2×B70, 651-tok diverse corpus): **25/25 bit-identical** PPL at both chunk-256 and chunk-512, **no collapse, no run-to-run divergence** — each chunk-512 pass runs the recurrence over 512 successive steps × 24 linear layers/card (~hundreds of thousands of step-executions) with zero divergence. The 512 single-call prefill is **1.08–1.15× faster** than 2×256 on >256-tok prompts (`ie-qwen3next-bench --ab`: 512-tok 543→617, 1024-tok 532→608 tok/s), closing the pp512 gap vs llama.cpp SYCL, at zero decode/short-prompt cost. **T≥1024 is NOT yet validated** (the diverse corpus only reaches ~650 tokens) — raising the cap further needs a longer-corpus `--sweep`. Why qwen3next tolerates 512 when the crown caps at 256 was NOT investigated (per the §1 do-not-re-investigate rule); the cap is a validated empirical bound, not a claim the bug is gone.

### How to reproduce
```bash
./build/tools/ie-bug-monitor --max-iters 200
# → "✗ DIVERGENCE DETECTED"  with first divergent chain + DN layer
```
or
```bash
./build/tools/ie-bug-live --max-iters 1024
```

### History
- 28-step bisect documented in `docs/bisect_step25_26_summary.md` and `Hand-off.md`.
- All bisect-step source scaffolding (Steps 15–28) was removed from the engine in the 2026-05-04 cleanup. The source is now back to a minimal baseline kernel.
- A clean alternative kernel path (`IE_DN_RECURRENCE_REWRITE`, default OFF) is preserved in `src/ops/deltanet.cpp` as a structurally-cleaner option (double-buffered state via DMA).

### Do not
- Re-run Steps 21–28 in any form. They were exhaustive.
- Chase this bug with more SYCL-level coherency primitives.
- Add fp16 / int8 state quantization expecting it to clamp the divergence.

### Do (when warranted)
- File a vendor IGC issue with a minimal repro from `tools/ie_bug_monitor.cpp`.
- Or accept the workaround (chunk prefill at T≤256) as the long-term solution.

---

## 2. Prefill performance gap vs llama.cpp

### Status
Engine prefill at pp512 is ~150 tok/s; llama.cpp's SYCL backend on the same B70 reports ~615–800 tok/s (per `research/01_hardware.md` and the PMZFX benchmarks brief). This is a **separate** issue from the determinism bug above.

### Known opportunities (none yet exercised)
- F16 accumulation in fp16 paths (the `-DGGML_SYCL_F16=ON` analog).
- Tile-size re-tuning for Xe2's 256 KB unified L1+SLM (33% larger than Alchemist).
- Persistent zero-gap MoE kernel with atomic counters for dynamic expert load balancing.
- Ensure `-O3 -DNDEBUG` is honored at all build paths.

### Constraints
- Per repo rules, do NOT touch ESIMD / block2d / `Subgroup2DBlockLoad` / `SubgroupBlockRead` / `gemm_q4k_esimd` / tile-load smoke tests when chasing this. Two prior attempts (`IE_ENABLE_GEMM_Q4K_ESIMD`, `IE_ENABLE_FA2_PREFILL_TILED`) regressed prefill and remain dormant scaffolds (default OFF).

### Decode performance — at parity
Engine decode is ~50 tok/s, matching llama.cpp's SYCL backend on the same B70. **No gap to close on decode.**

---

## 3. Architectural footnotes

- One in-order SYCL queue (`DeviceAllocator::queue()`); all kernels and memcpys serialized through it.
- Workspace buffers (`ws_x_normed_`, `ws_qkv_`, `ws_q_fp32_`, …) are SHARED across all forward calls. To inspect them per-chain, getter methods on `QwenModel` would need to be added — not yet present.
- `dn.state_ptr()`, `dn.conv_state_ptr()`, `kv.k_ptr()`, `kv.v_ptr()` are exposed and per-instance. The bug-monitor tools rely on these for hash comparisons.

---

## 4. [RESOLVED 2026-06-10] Dense-path run-to-run PPL nondeterminism — `rope_partial` in-place race

qwen3-8b dense PPL varied 18.95–19.42 across identical runs (P2 T8 blocker).
**NOT** the §1 hardware class — it was a software bug: `rope_partial`'s
one-item-per-dim layout let the cos-side and sin-side work-items of a rotary
pair race on the in-place (`x == y`) calls all model paths make. At the dense
shape (n_rotary=128, half=64) the two sides land in different work-groups and
the race fired constantly; at the crown shape (n_rotary=64, half=32) both
sides share one WG and it never manifested (crown stayed bit-exact
throughout, before and after the fix: avg NLL 1.864495).

Fixed in `src/ops/elementwise.cpp` (one work-item owns the whole pair).
Dense PPL is now bit-exact across runs (avg NLL 2.940491, 10/10).
Full bisect + verification: `docs/dense_nondeterminism_2026-06-10.md`.
Tooling kept: `tools/ie_det_probe.cpp` (`ie-det-probe`), `IE_DENSE_HASH=1`.

Lesson for §1-style investigations: a "nondeterministic kernel" verdict needs
the in-place-call audit too — a race that is value-invisible at trivial
inputs (here: rotation angle 0) passes short smoke tests and then looks like
hardware flicker at depth.
