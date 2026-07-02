# Intel GPU Programming Stack for Arc Pro B70 (Battlemage / Xe2)

**Target:** From-scratch C++ inference engine, XMX-capable kernels, single-card B70 (BMG-G31, 32 GB GDDR6, 256 XMX engines, 367 INT8 TOPS, 608 GB/s) [[B70 specs]](https://wccftech.com/big-battlemage-gpu-is-here-intel-arc-pro-b70-b65-32-gb-graphics-cards/) [[B70 launch]](https://www.thefpsreview.com/2026/03/25/intels-big-battlemage-finally-arrives-arc-pro-b70-and-b65-launched-today-with-32gb-of-vram-and-up-to-367-tops/).

---

## 1. API decision matrix

| API | Launch overhead | XMX access | Async submission | B-series maturity (2026) | Maintainability | Typical use |
|---|---|---|---|---|---|---|
| **Level Zero** | Lowest. Direct-to-metal, immediate command lists, signal events, no SYCL runtime cost [[L0 intro]](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/INTRO.html). | None directly; you compile SPIR-V kernels written in OpenCL-C / CM. XMX reached via SPV_INTEL_subgroup_matrix_multiply_accumulate or generated from SYCL. | Best — explicit command lists, fences, multi-queue, peer-to-peer. | Production on BMG since compute-runtime 25.27+; 26.14.37833.4 is the current shipping driver paired with IGC 2.32.7 [[CR releases]](https://github.com/intel/compute-runtime/releases). | High — verbose C API, long bring-up, but stable. | Use as the **runtime under the hood** if you want zero SYCL overhead. |
| **SYCL (DPC++)** | Higher than L0 but acceptable. Single-source C++. Backed by L0 by default on Linux. | First-class via `joint_matrix` (portable) or ESIMD `xmx::dpas` (Intel-only) [[joint_matrix Intel guide]](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/programming-intel-xmx-using-sycl-joint-matrix.html). | Out-of-order queues, in-order queues, USM, events. Less granular than L0. | Battlemage shapes added to `joint_matrix` 2025–2026; nightly `intel/llvm` ships daily, e.g. nightly-2026-03-22 [[intel/llvm releases]](https://github.com/intel/llvm/releases). | Best for productivity. C++20, templates work. | **Primary kernel language.** |
| **OpenCL** | Comparable to SYCL via the same compute-runtime stack [[CR repo]](https://github.com/intel/compute-runtime). | Via `cl_intel_subgroup_matrix_multiply_accumulate` builtins [[Khronos ext]](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html). | Command queues + events, in/out-of-order. | Mature, but Intel’s investment is moving to SYCL. | Mid — separate `.cl` source files, runtime compile. | Useful as a **portable fallback** and for handcrafted SPIR-V shenanigans. |
| **OpenVINO GPU plugin** | High — graph-level. Not a kernel API. | Indirect, picks XMX automatically through its compiled graph [[OV 2026 RN]](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html). | Async infer requests, but you don’t schedule kernels. | Production B70 support added in 2026 release (single-GPU inference of 20–30B LLMs) [[OV B70]](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html). | Lowest cost if you accept the IR/import workflow. | **Not** for an engine you’re writing yourself. |
| **oneDNN** | Library-level. SYCL or L0 engine. | Auto-selects XMX paths for matmul/SDPA on Xe2; supports int8/int4 weight decompression [[oneDNN matmul]](https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html). | SYCL queues + dependency events. | Production on Xe2 in 3.7+; SDPA with int4/int8 KV in 3.8 [[oneDNN 3.8]](https://www.phoronix.com/news/Intel-oneDNN-3.8). | Excellent, but you’re a library client, not the kernel author. | **Reference + escape hatch** for the GEMM you don’t want to hand-tune. |
| **XeTLA** | Header-only templates over ESIMD. | Direct, via DPAS templates. | SYCL queue-based. | **Archived Dec 2024**, no Battlemage support, replaced by sycl-tla [[XeTLA repo]](https://github.com/intel/xetla). | Don’t adopt new. | Read-only reference. |
| **sycl-tla / Cutlass-Fork** | Header-only template framework, like CUTLASS. | Direct, via SYCL `joint_matrix` and CuTe atoms; ships explicit BMG-G31 examples [[sycl-tla]](https://github.com/intel/sycl-tla). | SYCL queues. | Active. v0.8 (March 2026) adds BMG-G31 platform support, FlashAttention v2 hitting ~78% of peak [[sycl-tla v0.8]](https://github.com/intel/sycl-tla). | High. BSD-3, mirrors CUTLASS API. | **Strongest reference** for Xe2 GEMM/Attention skeletons. |

### Bottom line

- **Write the engine in SYCL (DPC++)** with `joint_matrix` for portability and `ESIMD xmx::dpas` for the hottest kernels you can’t coax out of the JIT. Use the open-source `intel/llvm` nightly toolchain rather than the gated oneAPI release for newest BMG fixes [[intel/llvm releases]](https://github.com/intel/llvm/releases).
- **Drop to OpenCL C** as the shader language only when SYCL JIT codegen is bad and you want to inspect/patch SPIR-V; this also gives you `cl_intel_subgroup_2d_block_io` and matrix-mad intrinsics in their canonical form [[2D block IO ext]](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html).
- **Use Level Zero as the runtime layer** if you need lowest-overhead submission, multi-queue scheduling, or want SPIR-V binaries you’ve cached/patched [[L0 spec]](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/INTRO.html).
- **oneDNN** for production GEMM/SDPA when correctness > novelty (it already has Xe2 paths and int4 weight decompression) [[oneDNN matmul]](https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html).
- Treat **OpenVINO** as out of scope.

---

## 2. XMX programming, three paths

### 2a. SYCL `joint_matrix` (portable, recommended default)

Header and namespace [[Intel XMX guide]](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/programming-intel-xmx-using-sycl-joint-matrix.html):

```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
using namespace sycl::ext::oneapi::experimental::matrix;
```

Type signatures (paraphrased from the upstream extension spec [[sycl_ext_oneapi_matrix]](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc)):

```cpp
template <typename Group, typename T, use Use, size_t M, size_t N, layout L>
class joint_matrix;

template <typename Group, typename T, use U, size_t M, size_t N, layout L,
          typename Ptr>
void joint_matrix_load(Group g, joint_matrix<Group,T,U,M,N,L>& res,
                       Ptr p, size_t stride);

template <typename Group, typename TA, typename TB, typename TC,
          size_t M, size_t K, size_t N, layout LA, layout LB>
joint_matrix<Group,TC,use::accumulator,M,N,layout::dynamic>
joint_matrix_mad(Group g,
                 joint_matrix<Group,TA,use::a,M,K,LA>  A,
                 joint_matrix<Group,TB,use::b,K,N,LB>  B,
                 joint_matrix<Group,TC,use::accumulator,M,N,layout::dynamic> C);

template <typename Group, typename T, size_t M, size_t N, layout L,
          typename Ptr>
void joint_matrix_store(Group g, joint_matrix<Group,T,use::accumulator,M,N,L>& mat,
                        Ptr p, size_t stride, layout L_run);
```

**Tile shapes on Xe2 / Battlemage** (subgroup size 16, M is a per-subgroup repeat count up to 8) [[Khronos cl_intel_subgroup_matrix_multiply_accumulate]](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html) [[Battlemage XMX]](https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/):

| Types Ta × Tb → Tc | M | N | K |
|---|---|---|---|
| fp16 × fp16 → fp32 | 1,2,4,8 | 16 | 16 |
| bf16 × bf16 → fp32 | 1,2,4,8 | 16 | 16 |
| int8 × int8 → int32 | 1,2,4,8 | 16 | 32 |
| int4 × int4 → int32 | 1,2,4,8 | 16 | 64 |
| int2 × int2 → int32 | 1,2,4,8 | 16 | 128 |

Use `matrix::matrix_type_query` / `get_info` at runtime to confirm legality [[joint_matrix spec]](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc).

### 2b. OpenCL builtins `cl_intel_subgroup_matrix_multiply_accumulate`

Function family — examples with verbatim signatures [[Khronos ext]](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html):

```c
// int8 * int8 -> int32, K=32, subgroup-size 16
int  intel_sub_group_i8_i8_matrix_mad_k32(int   a, int8 b, int   acc); // M=1
int2 intel_sub_group_i8_i8_matrix_mad_k32(int2  a, int8 b, int2  acc); // M=2
int4 intel_sub_group_i8_i8_matrix_mad_k32(int4  a, int8 b, int4  acc); // M=4
int8 intel_sub_group_i8_i8_matrix_mad_k32(int8  a, int8 b, int8  acc); // M=8

// fp16 * fp16 -> fp32, K=16
float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc);

// bf16 * bf16 -> fp32, K=16
float8 intel_sub_group_bf16_bf16_matrix_mad_k16(short8 a, int8 b, float8 acc);

// int4 * int4 -> int32, K=64
int8 intel_sub_group_i4_i4_matrix_mad_k64(int8 a, int8 b, int8 acc);
```

A is held packed in registers, B in VNNI-packed layout (4-byte rows interleaved), C in row-major fp32 / int32 across the 16-lane subgroup [[Khronos ext]](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html).

### 2c. ESIMD `sycl::ext::intel::esimd::xmx::dpas`

Header and namespace [[ESIMD docs]](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md):

```cpp
#include <sycl/ext/intel/esimd.hpp>
namespace xmx = sycl::ext::intel::esimd::xmx;
```

Signatures from the DPC++ headers [[ESIMD dpas.hpp]](https://intel.github.io/llvm-docs/doxygen/dpas_8hpp.html):

```cpp
template <int SystolicDepth, int RepeatCount,
          typename T, typename CT, typename BT, typename AT,
          xmx::dpas_argument_type BPrec = ...,
          xmx::dpas_argument_type APrec = ...,
          int N, int BN, int AN>
sycl::ext::intel::esimd::simd<T, N>
xmx::dpas(simd<CT,N> C, simd<BT,BN> B, simd<AT,AN> A);
```

`dpas_argument_type` enum: `fp16, bf16, tf32, u8, s8, u4, s4, u2, s2`. `SystolicDepth` is fixed at 8 on Intel hardware; `RepeatCount` ∈ {1,2,4,8} [[ESIMD ext]](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md). On Xe2 the execution width is 16 (same as PVC), not 8 like DG2 [[ESIMD ext]](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md).

### 2d. Working `int8 × int8 → int32` skeleton (preferred path: SYCL `joint_matrix`)

Most reliable on B-series today; `joint_matrix` codegen for BMG is exercised by sycl-tla and the SYCL conformance tests [[sycl-tla]](https://github.com/intel/sycl-tla).

```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
using namespace sycl;
namespace M = sycl::ext::oneapi::experimental::matrix;

constexpr int TM = 8, TN = 16, TK = 32;          // one XMX tile
constexpr int SG = 16;

void gemm_i8_tile(queue& q, const int8_t* A, const int8_t* B, int32_t* C,
                  int Mdim, int Ndim, int Kdim) {
  q.parallel_for(nd_range<2>({Mdim/TM, Ndim/TN*SG}, {1, SG}),
   [=](nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
      auto sg = it.get_sub_group();
      int gm = it.get_group(0), gn = it.get_group(1);
      M::joint_matrix<sub_group, int8_t,  M::use::a,           TM,TK, M::layout::row_major> a;
      M::joint_matrix<sub_group, int8_t,  M::use::b,           TK,TN, M::layout::ext_intel_packed> b;
      M::joint_matrix<sub_group, int32_t, M::use::accumulator, TM,TN, M::layout::row_major> c;
      M::joint_matrix_fill(sg, c, 0);
      for (int k = 0; k < Kdim; k += TK) {
        M::joint_matrix_load(sg, a, A + (gm*TM)*Kdim + k,        Kdim);
        M::joint_matrix_load(sg, b, B + k*Ndim     + gn*TN,      Ndim);
        c = M::joint_matrix_mad(sg, a, b, c);
      }
      M::joint_matrix_store(sg, c, C + (gm*TM)*Ndim + gn*TN, Ndim,
                            M::layout::row_major);
   });
}
```

Sources for shape/types: [[sycl_ext_oneapi_matrix]](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc), [[Khronos OpenCL matrix ext]](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html). The `ext_intel_packed` B layout is the VNNI variant the XMX expects; the loader will repack on transfer if you load from row-major using the proper API [[joint_matrix Intel guide]](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/programming-intel-xmx-using-sycl-joint-matrix.html).

---

## 3. Memory model

### 3.1 USM trade-offs on a discrete B70

Three allocation kinds [[Intel USM guide]](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/unified-shared-memory-allocations.html):

- **Device USM (`malloc_device`)** — lives in GDDR6, only the device can dereference. Fastest for kernels. Host must `q.memcpy()` to read it. Use for weights, KV cache, activations.
- **Host USM (`malloc_host`)** — pinned host RAM, device DMAs over PCIe each access. Useful for tiny streaming buffers (token IDs, logits-out small batches), murderous for hot tensors.
- **Shared USM (`malloc_shared`)** — page-migrated. On a discrete GPU each first-touch on the wrong side faults a page in. Acceptable for weights you load once and never read back; awful for ping-pong access [[USM perf]](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/performance-impact-of-usm-and-buffers.html).

**Rule:** weights, KV cache, all activations → `malloc_device`. Logits and token outputs → `malloc_host` if you read them every step. Avoid shared on the hot path.

### 3.2 SLM allocation and banking on Xe2

- Each Xe-core has **192 KB of L1/SLM on Lunar Lake/Xe2-LPG; 256 KB on Battlemage discrete** [[Battlemage L1]](https://chipsandcheese.com/p/intels-battlemage-architecture).
- Banking: **64 consecutive bytes across 16 banks at 4-byte granularity** (i.e. lane *i* in a SIMD16 subgroup hits bank *i* for a 4-byte access) [[SLM Intel doc]](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-0/shared-local-memory.html). Stride your 32-bit accesses by 64 B (one per lane) to avoid bank conflicts. For fp16, pack two halves into a 32-bit word and lane-stride in the same way.
- Allocate via `local_accessor<T,N>` in SYCL or `__local` in OpenCL.
- A workgroup occupies one Xe-core; SLM is per-workgroup, not shared across them.

### 3.3 2D block load — `cl_intel_subgroup_2d_block_io`

This is the extension you want for tiled GEMM/Attention. Built-ins look like [[2D block IO ext]](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html):

```c
void intel_sub_group_2d_block_read_8b_8r32x1c(  // 8b=8-bit, 8 rows, 32 cols, 1 column-block
    const __global void* base, int width_bytes, int height, int pitch_bytes,
    int2 coord, /*private*/ uint* dst);

void intel_sub_group_2d_block_read_16b_16r16x2c(...);   // 16-bit, 16r×16c, 2 col-blocks
void intel_sub_group_2d_block_read_transform_8b_32r16x1c(...); // produces VNNI-packed B for matrix_mad
void intel_sub_group_2d_block_read_transpose_*(...);    // for transposing on read
void intel_sub_group_2d_block_write_8b_*(...);
```

Restrictions [[2D block IO ext]](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html):

- per-subgroup `base_address` 64-byte aligned
- block width 64 ≤ B ≤ 224 bytes, multiple of 4 for 8/16-bit
- block height 1 ≤ H ≤ 224
- pitch ≥ width and a multiple of 16 bytes
- subgroup size must equal max subgroup size (16 on Xe2)
- `coord.x` multiple of 4 for 8-bit, 2 for 16-bit data

The `_transform_` variants do row→VNNI packing on the fly, which is exactly what you feed into `intel_sub_group_*_matrix_mad_*` for B. The `_transpose_` variants invert layout for a transposed B in attention’s `K^T`. Use these instead of doing it yourself in SLM.

In SYCL, identical hardware is reachable by extending with the `[[intel::enable_2d_block_io]]` annotation or by calling these builtins from a `sycl::ext::oneapi::experimental::cuda::ldmatrix`-like wrapper (currently not portable; expect to either drop into OpenCL-C kernel files or use sycl-tla’s `cute::copy` atoms which wrap them) [[sycl-tla]](https://github.com/intel/sycl-tla).

### 3.4 Subgroup ops + cost

Available on Xe2 via `cl_intel_subgroups` and SYCL `sub_group` [[cl_intel_subgroups]](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html):

| Op | Cost | Notes |
|---|---|---|
| `shuffle`, `shuffle_xor`, `shuffle_up/down` | 1 cycle | Cross-lane, no SLM round-trip |
| `broadcast` | 1 cycle | |
| `reduce_*` (add/min/max) | log₂(SG) cycles | Implemented via shuffles |
| `scan_*` (inclusive/exclusive) | log₂(SG) cycles | |
| `block_read/write` | DRAM-bound; DMA-style | Sequential across lanes |
| `2d_block_read/write` | DRAM-bound; coalesced 2D | Best for tile copies |
| `intel_sub_group_*_matrix_mad_*` | 1 cycle issue, 8 cycles latency | The XMX instruction |

Subgroup size is **16** on Xe2 — annotate with `[[sycl::reqd_sub_group_size(16)]]` to force it [[Intel sub-group guide]](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/sub-group.html).

### 3.5 Coalescing rules

- A subgroup of 16 lanes issues a single 64-byte cache-line transaction when each lane reads a 4-byte element at consecutive offsets [[SLM doc]](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-0/shared-local-memory.html).
- For 16-bit (fp16/bf16): pair lanes with 2-element vector loads (`half2`) to fill a 64-byte line.
- For 8-bit (int8): use `uchar4` per lane, or, better, the 2D block load.
- Strided / scattered access drops to per-lane gathers — kills throughput. Always restructure to row-major contiguous on the K dimension.

---

## 4. Toolchain & build

### 4.1 Compiler choice

- **Intel oneAPI 2026.x DPC++** (icpx): the productized release, validated, gated quarterly. Ships oneMKL/oneDNN/oneCCL.
- **`intel/llvm` upstream** (clang++ from a sycl-branch build): nightly tags such as `nightly-2026-03-22` [[intel/llvm releases]](https://github.com/intel/llvm/releases). Has the latest `joint_matrix` shapes and BMG fixes weeks earlier. Doesn’t auto-link MKL.

**Pick:** upstream `intel/llvm` for the kernels (you’re writing them anyway), keep an Intel oneAPI install for oneDNN/oneMKL .so’s. They co-exist; just compile your engine with the upstream `clang++ -fsycl` and link `-lonednn`.

### 4.2 AOT compile flag for Battlemage

Verified target names from `intel/llvm`’s `UsersManual.md` [[Users Manual]](https://github.com/intel/llvm/blob/sycl/sycl/doc/UsersManual.md):

```
intel_gpu_bmg_g31   →  alias intel_gpu_20_2_0   (Big Battlemage / B70/B65)
intel_gpu_bmg_g21   →  alias intel_gpu_20_1_4   (B580 / smaller die)
```

So for the B70:

```bash
clang++ -fsycl -O3 -std=c++20 \
    -fsycl-targets=intel_gpu_bmg_g31 \
    -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -ze-opt-large-register-file" \
    engine.cpp kernels.cpp -o engine
```

For both BMG dies in one fat binary: `-fsycl-targets=intel_gpu_bmg_g21,intel_gpu_bmg_g31` [[Users Manual]](https://github.com/intel/llvm/blob/sycl/sycl/doc/UsersManual.md). sycl-tla’s shorthand `-DDPCPP_SYCL_TARGET=bmg` does both.

### 4.3 CMake glue (single-source SYCL + plain C++)

```cmake
cmake_minimum_required(VERSION 3.25)
project(engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

# Force the SYCL compiler. Set CXX=clang++ from intel/llvm, or icpx.
if(NOT CMAKE_CXX_COMPILER MATCHES "icpx|clang\\+\\+")
  message(FATAL_ERROR "Use icpx or intel/llvm clang++")
endif()

add_library(kernels OBJECT
    src/gemm_xmx.cpp
    src/attention_xmx.cpp
    src/dequant_int4.cpp)
target_compile_options(kernels PRIVATE
    -fsycl
    -fsycl-targets=intel_gpu_bmg_g31
    -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -ze-opt-large-register-file"
    -O3 -ffp-model=precise)

add_executable(engine src/main.cpp src/runtime.cpp)
target_link_options(engine PRIVATE
    -fsycl -fsycl-targets=intel_gpu_bmg_g31)
target_link_libraries(engine PRIVATE kernels dnnl)
```

If you mix CUDA-style `.cu` filenames for muscle memory, set `set_source_files_properties(... PROPERTIES LANGUAGE CXX)` and use `-x c++` — DPC++ doesn’t care about the extension.

### 4.4 Driver and IGC pairing

| compute-runtime | IGC | Date | BMG status |
|---|---|---|---|
| 26.14.37833.4 | 2.32.7 | 2026-04-20 | Production [[CR releases]](https://github.com/intel/compute-runtime/releases) |
| 26.09.37435.1 | 2.30.1 | 2026-03-17 | Production |
| 26.05.37020.3 | 2.28.4 | 2026-02-11 | Adds joint-waveall vectorization default [[Phoronix CR 26.05]](https://www.phoronix.com/news/Intel-CR-26.05.37020.3) |
| 26.01.36711.4 | 2.27.10 | 2026-01-14 | Adds BMG-G31 device IDs initial [[BMG-G31 IGC]](https://www.phoronix.com/news/Intel-Graphics-Compiler-IGC-216) |

Running on B70: pin to compute-runtime ≥ 26.05 + IGC ≥ 2.28.4. Earlier IGCs miss BMG-G31 codegen patches. Check with `clinfo -l` and `level-zero --version`.

---

## 5. Reference implementations to mine

| Project | Where the GEMM / attention lives | License | Readability | Why use it |
|---|---|---|---|---|
| **llama.cpp / ggml-sycl** | `ggml/src/ggml-sycl/{ggml-sycl.cpp, dmmv.cpp, mmvq.cpp, dequantize.hpp, vecdotq.hpp}` [[issue 21517]](https://github.com/ggml-org/llama.cpp/issues/21517) | MIT | OK; quant-format-driven, lots of dispatch | Real shipping kernels for INT4/INT8 quantized × FP16 matmul; FA added 2026-03 [[SYCL backend]](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md). Heads-up: there is a known correctness bug on B70 unless `GGML_SYCL_DISABLE_OPT=1` [[issue 21893]](https://github.com/ggml-org/llama.cpp/issues/21893). |
| **ipex-llm** | `python/llm/src/ipex_llm/transformers/models/*` plus C++ XPU custom ops; uses oneDNN + XeTLA + custom int4 kernels [[ipex-llm]](https://github.com/intel/ipex-llm) | Apache 2.0 | Mid; lots of Python plumbing | Has B-series-tuned attention. INT4 dequant+matmul kernels are in the `csrc/` C++ trees. |
| **oneDNN** | `src/gpu/intel/jit/gemm/`, `src/gpu/intel/ocl/gemm/` for Xe matmul; `src/gpu/intel/jit/sdpa/` for SDPA [[oneDNN matmul]](https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html) | Apache 2.0 | High in JIT layer; templated nGEN code | Production-quality GEMM, INT4 weight decomp, SDPA — link as a library and read for ideas. |
| **intel/sycl-tla** (Cutlass-Fork) | `include/cutlass/gemm/`, `include/cute/`, `examples/00..13` Battlemage-specific [[sycl-tla]](https://github.com/intel/sycl-tla) | BSD-3 | Best for kernel writers familiar with CUTLASS | Active, BMG-G31 coverage in v0.8 (March 2026), FA v2 example, FP8/INT4 GEMM, grouped GEMM. **Primary reference**. |
| **OpenVINO GenAI** | `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/` (OpenCL-C) and oneDNN-backed primitives [[OV 2026]](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html) | Apache 2.0 | Mixed; large kernel zoo | Useful for fused attention + paged KV idioms, and the OCL-C kernels show 2D block-load patterns. |
| **intel/xetla (archived)** | `xetla/include/subgroup/`, `xetla/include/group/`, `xetla/examples/01_gemm_universal` [[XeTLA repo]](https://github.com/intel/xetla) | Apache 2.0 | Decent but ESIMD-heavy | **Read-only**. Archived Dec 2024, no BMG support, but the GEMM tile design choices still teach you DPAS scheduling. |

---

## 6. Three canonical kernel idioms (Xe2-tuned pseudocode)

### 6.1 XMX-tiled GEMM with double-buffered SLM and 2D block loads

Tile shape choice for B70 (Xe-core has 8 XMX, 256 KB SLM): per-workgroup 128×128, per-subgroup 32×64. K tile = 32 (int8) or 16 (bf16).

```text
WG tile     128 x 128   (M x N)  matches one Xe-core
SG layout   4 x 4 subgroups, each 32 x 32  (so 16 subgroups per WG)
Per-SG XMX  M=8, N=16, K=32 int8    -> 4 (M/8) x 2 (N/16) accumulators per SG
SLM         A_smem[2][128 x K_tile]  + B_smem[2][K_tile x 128]   (double buffer)
            128 * 32 * 1B * 2 + 32 * 128 * 1B * 2 = 16 KB         well under 256 KB
```

```text
kernel xmx_gemm_int8(A, B, C, M, N, K) [[reqd_sub_group_size(16)]]:
  wg_m = group_id(0) * 128;  wg_n = group_id(1) * 128
  sg_m = wg_m + sg_row*32;   sg_n = wg_n + sg_col*32
  acc[4][2] = 0   // int32 simd<16>
  // Stage 0: prefetch tile 0 via 2D block read into SLM[0]
  prefetch_2d(A_smem[0], A, wg_m, 0, 128, K_tile)
  prefetch_2d(B_smem[0], B, 0, wg_n, K_tile, 128)
  barrier
  for k0 in 0..K step K_tile:
      cur = (k0/K_tile) & 1
      nxt = cur ^ 1
      // issue async load of next tile while computing current
      if k0 + K_tile < K:
          prefetch_2d(A_smem[nxt], A, wg_m, k0+K_tile, 128, K_tile)
          prefetch_2d(B_smem[nxt], B, k0+K_tile, wg_n, K_tile, 128)
      // load A,B fragments from SLM into registers
      for m in 0..4:
        a[m] = block_read_from_slm(A_smem[cur], sg_m + m*8, 0)  // 8x32 int8
      for n in 0..2:
        b[n] = block_read_vnni_from_slm(B_smem[cur], 0, sg_n + n*16) // 32x16 packed
      // 8 XMX MADs per SG per iteration
      for m in 0..4:
        for n in 0..2:
          acc[m][n] = intel_sub_group_i8_i8_matrix_mad_k32(a[m], b[n], acc[m][n])
      barrier
  // store 32x32 int32 accumulator to C
  store_2d(C, acc, sg_m, sg_n)
```

Key choices: 2D block reads with the `_transform_` variant pack B into VNNI on the fly so the matrix-mad consumes it directly [[2D block IO]](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html). Double-buffer via two SLM regions and a barrier between stages. Use `-ze-opt-large-register-file` to prevent spills [[joint_matrix Intel guide]](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/programming-intel-xmx-using-sycl-joint-matrix.html). sycl-tla’s `examples/04_bmg_gemm_collective_builder` is the canonical reference [[sycl-tla]](https://github.com/intel/sycl-tla).

### 6.2 FlashAttention-2 along K (fused QKᵀ → softmax → PV)

```text
kernel flash_attn_xmx(Q, K, V, O, B, H, N, D)  [[reqd_sub_group_size(16)]]:
  // tile sizes: Br rows of Q per WG, Bc cols of K per inner iteration
  Br = 128; Bc = 64; D <= 128
  q_tile = 2D_block_read(Q, head, row_start, 0, Br, D)             // SLM
  m_i = -inf vec(Br); l_i = 0 vec(Br); O_i = 0 mat(Br,D)           // registers
  for kc in 0..N step Bc:
      k_tile = 2D_block_read_transpose(K, head, kc, 0, Bc, D)      // SLM, K^T layout
      // S = Q @ K^T using XMX bf16
      S = xmx_gemm_bf16(q_tile, k_tile)                            // Br x Bc fp32
      apply_causal_mask(S, row_start, kc)
      m_new = max(m_i, rowmax(S))
      P = exp(S - m_new)                                           // bf16 cast
      l_new = exp(m_i - m_new) * l_i + rowsum(P)
      v_tile = 2D_block_read(V, head, kc, 0, Bc, D)                // SLM
      // O = diag(exp(m_i-m_new)) @ O + P @ V
      O_i = scale_rows(O_i, exp(m_i - m_new))
      O_i = xmx_gemm_bf16_accum(P, v_tile, O_i)                    // accumulate fp32
      m_i = m_new; l_i = l_new
  O_i = O_i / l_i
  2D_block_write(O, O_i, head, row_start, 0, Br, D)
```

Notes: keep Q resident in SLM, stream K and V tiles. Use `joint_matrix` with bf16 (M=8,N=16,K=16) for the `S = Q K^T` and `O += P V` GEMMs. Watch register pressure — Codeplay’s blog reports up to 10% gain by sinking loads close to use to reduce liveness on the Q matrix [[Codeplay FA]](https://codeplay.com/portal/blogs/2025/09/02/improving-triton-flashattention-performance-on-intel-gpu). sycl-tla’s `examples/06_bmg_flash_attention` runs ~78% of peak [[sycl-tla v0.8]](https://github.com/intel/sycl-tla).

### 6.3 INT4-weight × FP16-activation: dequant-and-XMX without SLM round-trip

```text
kernel int4_fp16_gemv(W4, scales, zps, X_fp16, Y_fp16, M, N, K) [[reqd_sub_group_size(16)]]:
  // Per-subgroup tile: M=8 (output rows), N=16 (always SG), K=64 (one int4-XMX tile)
  acc = 0   // float8 across SG lanes
  for k0 in 0..K step 64:
      // 1. Load 8x64 INT4 weight tile = 8 x 32 bytes  via 2D block read
      w_packed = intel_sub_group_2d_block_read_8b_8r32x1c(W4, ...)      // u8 holds 2x int4
      // 2. Decode INT4 to bf16/fp16 IN REGISTERS using bit ops
      //    Each lane holds 32 nibbles -> 32 fp16 values (per row, per lane)
      for row in 0..8:
        lo  = (w_packed[row] & 0x0f) - zps[group(k0,row)]
        hi  = (w_packed[row] >>   4) - zps[group(k0,row)]
        a_bf16[row] = bf16(lo) * scales[group(k0,row)]
                   |  bf16(hi) * scales[group(k0,row)]    // 32 bf16 lanes
      // 3. Load activation tile X (64 x 16 bf16, VNNI-packed) directly from device USM
      x_vnni = intel_sub_group_2d_block_read_transform_16b_32r16x1c(X_fp16, k0, 0)
      // 4. XMX bf16 mad, K=16, repeat 4x to cover K=64
      acc = intel_sub_group_bf16_bf16_matrix_mad_k16(a_bf16[0..15],  x_vnni[0], acc)
      acc = intel_sub_group_bf16_bf16_matrix_mad_k16(a_bf16[16..31], x_vnni[1], acc)
      // ... (4 calls in total)
  // store acc back as fp16
  intel_sub_group_2d_block_write_16b_8r16x1c(Y_fp16, ..., acc)
```

The trick: weights and scales never leave registers between dequant and XMX. The activation comes in pre-VNNI’d via the `_transform_` 2D block read [[2D block IO]](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html). This is the layout oneDNN uses for `int4`-weight / `f16`-activation matmul on Xe2 [[oneDNN 3.8]](https://www.phoronix.com/news/Intel-oneDNN-3.8) and what ipex-llm’s low-bit kernels target [[ipex-llm BMG quickstart]](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/bmg_quickstart.md). Group-quantization scales (group size 32, 64, or 128) sit in constant memory or registers depending on M.

---

## Things to verify on actual B70 hardware

1. **BMG GEMM correctness gating.** Confirm whether your build is hit by the `GGML_SYCL_DISABLE_OPT=1` weight-corruption class of bugs (DPAS sync / tensor reorder) [[issue 21893]](https://github.com/ggml-org/llama.cpp/issues/21893). Run a known-good golden GEMM (oneDNN) vs. your hand kernel for INT8 and BF16, and compare bit-exact.
2. **`-ze-opt-large-register-file`.** Measure with and without; on B70 the register file budget is generous but spills can still happen with 128×128 tiles. Use `IGC_ShaderDumpEnable=1` to confirm no spill.
3. **2D block read alignment.** All your weight pointers must be 64-byte aligned and pitches must be a multiple of 16 bytes [[2D block IO]](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html). Verify by allocating with `sycl::aligned_alloc_device(64, ...)`.
4. **SLM size.** Confirm 256 KB on the BMG-G31 die (vs. 192 KB on Lunar Lake Xe2-LPG) by querying `device::get_info<info::device::local_mem_size>()` — sources differ on whether discrete BMG bumped to 256 KB [[Battlemage L1]](https://chipsandcheese.com/p/intels-battlemage-architecture). Plan tiles for the smaller number to be safe.
5. **`joint_matrix` shape support.** Use the runtime query API on the actual B70 to enumerate supported (Ta,Tb,Tc,M,N,K) tuples; the spec says shapes are device-introspected [[sycl_ext_oneapi_matrix]](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc).
6. **Subgroup size 16 (not 32).** Xe2 supports SIMD16 native and SIMD32 via packed [[Xe2 arch]](https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/), but the matrix-mad builtins only accept SG=16. Always require it.
7. **AOT vs. JIT first-launch latency.** AOT with `-fsycl-targets=intel_gpu_bmg_g31` cuts cold-start; verify in your engine’s startup path.
8. **Driver version match.** Validate runtime against the matrix in §4.4. If `level-zero` reports older than `26.05.37020.3` the BMG-G31 codegen path may not be present [[CR releases]](https://github.com/intel/compute-runtime/releases).
9. **L0 vs. OCL backend choice.** Set `SYCL_UI_BACKEND=level_zero` and benchmark a 70B-class workload vs. `opencl`. L0 typically wins on submission overhead; OCL wins occasionally for legacy kernels.
10. **PCIe Gen 4/5 host transfers.** `malloc_host` works fine for token streams, but profile the steady-state per-step device→host copy of logits — it should be < 50 µs at vocab 200k fp32. If not, you’re in shared-USM-fault land.
11. **Power and clock telemetry.** B70 is a $949 pro card with TBP ~225–250 W. Use `intel_gpu_top` to confirm sustained XMX clocks under 70B-class inference.
12. **Multi-engine occupancy.** B70 has 32 Xe2 cores → 256 XMX engines [[B70 specs]](https://wccftech.com/big-battlemage-gpu-is-here-intel-arc-pro-b70-b65-32-gb-graphics-cards/). Pick WG count ≥ 256 for full saturation; verify via `ZE_AFFINITY_MASK` and queue-fan-out experiments.

---

### Sources

- [Programming Intel XMX Using SYCL Joint Matrix Extension (Intel optimization guide 2025-0)](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/programming-intel-xmx-using-sycl-joint-matrix.html)
- [SYCL ext_oneapi_matrix specification (intel/llvm)](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc)
- [SYCL ext_intel_esimd specification (intel/llvm)](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md)
- [DPC++ esimd::xmx::dpas header reference](https://intel.github.io/llvm-docs/doxygen/dpas_8hpp.html)
- [cl_intel_subgroup_matrix_multiply_accumulate (Khronos)](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html)
- [cl_intel_subgroup_2d_block_io (Khronos)](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html)
- [cl_intel_subgroups (Khronos)](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html)
- [Level Zero spec (oneapi-src)](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/INTRO.html)
- [intel/llvm UsersManual.md (–fsycl-targets table)](https://github.com/intel/llvm/blob/sycl/sycl/doc/UsersManual.md)
- [intel/llvm releases (nightly tags)](https://github.com/intel/llvm/releases)
- [intel/compute-runtime releases](https://github.com/intel/compute-runtime/releases)
- [Phoronix — Intel Compute Runtime 26.05](https://www.phoronix.com/news/Intel-CR-26.05.37020.3)
- [Phoronix — IGC 2.16 BMG-G31](https://www.phoronix.com/news/Intel-Graphics-Compiler-IGC-216)
- [intel/sycl-tla (BSD-3 CUTLASS fork for Intel GPUs)](https://github.com/intel/sycl-tla)
- [intel/xetla (archived Dec 2024)](https://github.com/intel/xetla)
- [oneDNN 3.8 release coverage (Phoronix)](https://www.phoronix.com/news/Intel-oneDNN-3.8)
- [oneDNN matmul dev guide (uxlfoundation)](https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html)
- [llama.cpp SYCL backend docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- [llama.cpp issue #21517 (Q8_0 perf on B70)](https://github.com/ggml-org/llama.cpp/issues/21517)
- [llama.cpp issue #21893 (B70 weight corruption without DISABLE_OPT)](https://github.com/ggml-org/llama.cpp/issues/21893)
- [intel/ipex-llm](https://github.com/intel/ipex-llm)
- [ipex-llm BMG quickstart](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/bmg_quickstart.md)
- [OpenVINO 2026 release notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html)
- [Intel SLM optimization guide 2024-0](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-0/shared-local-memory.html)
- [Intel sub-groups optimization guide 2024-1](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/sub-group.html)
- [Intel USM allocations guide 2025-0](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/unified-shared-memory-allocations.html)
- [Intel USM perf guide 2023-0](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/performance-impact-of-usm-and-buffers.html)
- [Codeplay — Improving Triton FlashAttention on Intel GPU](https://codeplay.com/portal/blogs/2025/09/02/improving-triton-flashattention-performance-on-intel-gpu)
- [Battlemage architecture analysis (HWCooling)](https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/)
- [Battlemage architecture (Chips and Cheese)](https://chipsandcheese.com/p/intels-battlemage-architecture)
- [Arc Pro B70 launch (WCCFTech)](https://wccftech.com/big-battlemage-gpu-is-here-intel-arc-pro-b70-b65-32-gb-graphics-cards/)
- [Arc Pro B70 launch (FPS Review)](https://www.thefpsreview.com/2026/03/25/intels-big-battlemage-finally-arrives-arc-pro-b70-and-b65-launched-today-with-32gb-of-vram-and-up-to-367-tops/)
