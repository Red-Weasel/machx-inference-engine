# CUDA / multi-platform readiness audit (owner roadmap #3) — 2026-06-13

**Purpose:** the owner's launch must-have is "at least say it supports CUDA," plus
"survey what platforms we DON'T support yet." This is a **codebase-grounded** audit
of what running Mach X on NVIDIA actually takes. **No CUDA validation is possible on
this box** (Intel-only: 2× B70); validation HW is the Windows + RTX 2080 box.

## TL;DR

Mach X is SYCL/DPC++. **CUDA is reachable WITHOUT a kernel rewrite** because oneAPI
DPC++ has an NVIDIA backend (`-fsycl-targets=nvptx64-nvidia-cuda`, via the Codeplay
"oneAPI for NVIDIA GPUs" plugin) — the same SYCL source compiles to PTX. **But it is
NOT a free recompile.** The gating work, in effort order:

1. **Subgroup width 16→32** (the dominant cost). ~15 kernel files hardcode Intel's
   sub-group size of 16 (`[[sycl::reqd_sub_group_size(16)]]` + `constexpr SG_SIZE=16`)
   and the reductions / lane-walk indexing are width-specific (e.g. the int-dot MoE
   "lane walks q8 blocks {lane, lane+16, …}"). NVIDIA warps are 32; the nvptx backend
   does not provide native SG=16. Each hot kernel needs an SG=32 variant (parameterize
   `SG_SIZE`, widen the reductions, re-tile). Bounded and mechanical, but real.
2. **oneDNN → compile-optional. ✅ DONE 2026-06-13.** `src/CMakeLists.txt` was a HARD
   `find_package(DNNL CONFIG REQUIRED …)` build dep on Intel's library. Now behind
   `option(IE_ENABLE_ONEDNN ON)`: ON (default) = byte-identical Intel build; OFF =
   `find_package` skipped, `DNNL::dnnl` unlinked, and `ops/gemm_onednn.cpp` swapped for
   `ops/gemm_onednn_stub.cpp` (a `gemm_fp16_onednn` shim that routes to the in-house
   `gemm_fp16` — **same signature, so NO call-site edits**, crucially keeping the
   crown's `qwen36.cpp` UNTOUCHED). `dnnl::` symbols live ONLY in `gemm_onednn.cpp`, so
   the stub fully severs the dependency. qwen35-27B prefill loses its ~1.65× oneDNN
   lever and falls back to `gemm_fp16` (correctness unaffected; a cuBLAS path is a later
   perf item). Verified: default-ON build unchanged (ie_core not recompiled, crown
   binary identical); `-DIE_ENABLE_ONEDNN=OFF` configures with no DNNL + ie_core builds.
3. **Generalize device selection.** `DeviceAllocator::init` / `DeviceFleet::init` take
   a `name_filter` (defaults to the Intel "0xe223" string). Add a backend-aware default
   (CUDA name filter or `sycl::gpu_selector`); already a parameter, so a small change.
4. **Add the nvptx64 build target.** `IE_SYCL_TARGET` is already a CMake cache var
   (`spir64` today). A CUDA build sets `-fsycl-targets=nvptx64-nvidia-cuda` +
   `-Xsycl-target-backend --cuda-gpu-arch=sm_75` (RTX 2080 = Turing sm_75) and needs the
   CUDA toolkit + Codeplay plugin on the build box.
5. **ESIMD: exclude the TU (small) — but with macro-coupling care.** ESIMD
   (`gemm_q4k_esimd.cpp`, the only file with Intel `__spirv_…INTEL` intrinsics) is never
   *called* on the production path (its call site is in `qwen36.cpp` behind
   `#if IE_ENABLE_GEMM_Q4K_ESIMD`, default **0**), but the TU is in the `ie_core` source
   list **unconditionally** and won't compile for nvptx64. Gate the source behind a CMake
   option (same `${IE_ONEDNN_SRC}`-style swap) — but that option MUST also drive
   `IE_ENABLE_GEMM_Q4K_ESIMD` so the crown's `#if` call site stays compiled-out whenever
   the TU is excluded (otherwise a config that sets the call-site macro but excludes the TU
   gets a link error). Do NOT edit `qwen36.cpp` — the `#if` already guards the call;
   just drive its macro from the same option.

   ⚠ **XMX is the hard part — NOT a simple gate (corrected 2026-06-13).**
   `gemm_q4k_xmx.cpp` exports `gemm_q4_K_xmx` / `gemm_q6_K_xmx` /
   `moe_prefill_gate_up_silu_q4k_xmx`, and **`gemm_q4_K_xmx` is LOAD-BEARING on the default
   dense path** (`dense_dispatch.hpp:584-585` — "the dense GEMM path always uses
   gemm_q4_K_xmx"), not behind any flag. So it canNOT just be excluded for CUDA — doing so
   breaks the dense forward. It uses SYCL `joint_matrix` (Intel XMX/DPAS), which *does* map
   to NVIDIA tensor cores under nvptx64, but the dequant/tiling is Intel-tuned and needs a
   tensor-core validation pass OR a `gemm_fp16` fallback shim (oneDNN-stub pattern, same
   signature) for non-Intel builds. **This is part of the kernel port (item 1), not trivial
   build-config — it's why "the tree configures for CUDA" is bigger than gating two TUs.**
   Net: the cheap, testable-on-Intel build-config wins were items 2-4 (oneDNN ✅ done);
   ESIMD gating + the XMX fallback are best done on the RTX box where they can be compiled
   and validated, not blind on Intel hardware.

**Open verification items (need the RTX box):** (a) whether `reqd_sub_group_size(32)`
+ widened kernels produce bit-correct output on Turing; (b) `joint_matrix`/XMX usage —
`gemm_q4k_xmx.cpp` exists (Intel matrix); confirm it is NOT on the default path (crown
default is `gemm_fp16`) or provide a tensor-core/scalar fallback; (c) fp16 atomics /
`sycl::half` codegen parity on sm_75; (d) per-kernel perf retune (warp=32 occupancy).

## SYCL coupling footprint (why this isn't a quick port, why it isn't a rewrite either)

- 50 / 82 source+header files use `sycl::` directly; compute is SYCL-woven across
  `src/ops/*` (all gemv/gemm/MoE/DeltaNet/conv), `src/core/{allocator,kv_cache,
  deltanet_state}`, and every model forward. **All of this is portable SYCL** — it
  compiles for nvptx64 as-is, modulo the SG-width issue above.
- The **non-portable** surface is small and already isolated/optional: ESIMD (1 file,
  off), oneDNN (opt-in, has fallback), the Intel device-name default, Intel subgroup
  intrinsics (confined to the off ESIMD file). The hot kernels use the **portable**
  `sycl::sub_group` API, not `ext::intel` — so SG *width* is the issue, not Intel-only
  primitives.

## Platform support matrix (what we DO and DON'T support today)

| Platform / target | Today | Path to support |
|---|---|---|
| **Intel Arc (Battlemage B70, Linux)** | ✅ shipping, crown + 8 arch families | — |
| **Intel Arc (other Xe/Xe2, Linux)** | 🟡 likely runs (SPIR-V JIT) | smoke-test; device filter generalize |
| **NVIDIA (CUDA, Linux/Windows)** | ❌ | items 1-4 above; validate on RTX 2080 (sm_75) |
| **Windows (Intel Arc)** | ❌ not validated | P3c groundwork (`…-p3c-windows-groundwork.md`); only load-bearing POSIX is the gguf mmap → `MappedFile` |
| **AMD ROCm (HIP)** | ❌ | same SYCL story via `amdgcn` target (Codeplay AMD plugin); same SG-width work (AMD wavefront 32/64) |
| **Apple Metal** | ❌ | no SYCL Metal backend; would be a separate backend — out of scope |
| **CPU (SYCL host / OpenCL)** | 🟡 compiles, unoptimized | exists as a SYCL target; not a product path |

## Recommendation

For the launch claim, the credible and honest statement after items 1-4 + RTX
validation is: **"runs on NVIDIA via SYCL/CUDA (nvptx64)"** — true, demonstrable, and
not a rewrite. Do NOT claim CUDA before the SG-width kernels are validated bit-correct
on the RTX box (an SG=16 kernel silently mis-reducing on a warp=32 device would corrupt
output, not crash — exactly the kind of "half-wired" result to avoid). Suggested
sequencing: (1) make oneDNN/ESIMD compile-optional + add the nvptx64 target (doable
here, no NVIDIA HW — it's CMake + a toolchain) so the tree *builds* for CUDA; (2) on the
RTX box, bring up the SG=32 kernel variants one family at a time (start with the dense
`kLlama3` path — fewest kernels), gated by the same per-layer-cosine + PPL battery used
for every arch. Effort estimate: **CMake/target seam ~0.5 day; SG-width kernel port +
validation ~1-2 weeks** depending on how many of the ~15 kernels the launch path needs
(dense-only is far less than crown MoE + DeltaNet).
</content>
