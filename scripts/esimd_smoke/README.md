# Step 0 — ESIMD subgroup block read smoke tests

One-shot HW characterization tests for `__spirv_SubgroupBlockReadINTEL`
on BMG-G31 stepping C0 (IP 20.2.0).  Run once per driver/firmware
combination, then stop (per `project_v2_phase1_freeze_diagnosis.md`
safe-iteration checklist).

## Driver/firmware tested 2026-05-02

| | version |
|---|---|
| GPU | Intel(R) Graphics [0xe223] BMG-G31 stepping C0, IP 20.2.0 |
| compute-runtime / NEO | 26.14.37833.4 |
| IGC | 2.32.7 |
| libze | 1.15.37833+4 |
| GuC firmware | 70.60.0 |
| Kernel | Linux 6.17.0-22-generic |

## Result

**All cooperative subgroup block read forms PASS on G31 C0.**

| test | layout | result |
|---|---|---|
| sgbr_smoke (N=1, single ushort/lane) | 8 rows × 16 ushorts | PASS, sum=8128/8128 |
| sgbr_smoke_multi (N=2, vec<ushort,2>/lane) | 8 rows × 32 ushorts (64 B) | PASS |
| sgbr_smoke_multi (N=4, vec<ushort,4>/lane) | 8 rows × 64 ushorts (128 B) | PASS |
| sgbr_smoke_multi (N=8, vec<ushort,8>/lane) | 8 rows × 128 ushorts (256 B) | PASS |

No GuC CT failures, no `xe` driver coredumps, no exec-queue resets.
Total wall time across all 4 tests: ~290 ms.

## What this resolves

The project memory file (`project_v2_phase1_freeze_diagnosis.md`) and
the existing `src/ops/gemm_q4k_esimd.cpp` source comment disagreed on
whether `__spirv_SubgroupBlockReadINTEL` works on G31 C0:

- **Memory file:** "1D `__spirv_SubgroupBlockReadINTEL` works perfectly
  on G31."
- **Source comment:** "`lsc_load.ugm (M1_NM,1) d32x8t
  (__spirv_SubgroupBlockReadINTEL)` causes a GuC CT failure."

Both correct.  The two are talking about **different** intrinsic forms:

- The **non-transposed cooperative form** (`sg.load(mptr)` →
  `__spirv_SubgroupBlockReadINTEL<T>` with `T` ∈ {ushort, vec<ushort,N>})
  is what these smoke tests just verified — it **works**.  This maps to
  contiguous-row LSC.UGM block reads.
- The **transposed form** (`d32x8t` = 32-bit element × 8 elements
  transposed, used by cutlass-sycl's `BlockTiledCopy_TransposeT_8u32x8`
  on B580) is what the source's earlier test exercised — it **hangs**
  on G31 with a GuC CT failure.

The non-transposed form is sufficient for ESIMD GEMM weight loads in a
row-major layout.  The transposed form (used for VNNI-permuted loads on
B580) is not needed if we lay out tiles correctly.

## Implication

**Step 1 (ESIMD GEMM with cooperative 1D block reads) is unblocked.**
Projected closure of the prefill gap: +50–80% on GEMM-bound shapes
(PDF P0 lever).

## To re-run

Compile and run each test ONCE:

```bash
cd /tmp
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O2 -std=c++20 \
    /home/weezy/00\ -\ Inference\ Engine/scripts/esimd_smoke/sgbr_smoke.cpp \
    -o sgbr_smoke
timeout --kill-after=2 10 ./sgbr_smoke
journalctl --since "30 sec ago" -k -g "xe|GuC|coredump"
```

Re-test after any GPU driver/firmware update, OR if any future ESIMD
kernel work shows unexpected hangs.
