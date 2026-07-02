cmake_minimum_required(VERSION 3.28)

# The icpx compiler must be selected BEFORE project() so CMake's compiler
# detection picks up the SYCL/DPC++ frontend. Caller is expected to
# `source scripts/env.sh` first; if not, fall back to /opt/intel/oneapi.
if(NOT CMAKE_CXX_COMPILER)
  if(DEFINED ENV{CXX} AND NOT "$ENV{CXX}" STREQUAL "")
    set(CMAKE_CXX_COMPILER "$ENV{CXX}")
  elseif(EXISTS "/opt/intel/oneapi/compiler/2026.0/bin/icpx")
    set(CMAKE_CXX_COMPILER "/opt/intel/oneapi/compiler/2026.0/bin/icpx")
  else()
    find_program(_ICPX icpx REQUIRED)
    set(CMAKE_CXX_COMPILER "${_ICPX}")
  endif()
endif()

project(inference_engine
  VERSION 0.0.1
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# SYCL build mode. JIT-via-spir64 with a Battlemage device hint.
# AOT (-fsycl-targets=intel_gpu_bmg_g31) is also supported.
set(IE_SYCL_TARGET      "spir64"   CACHE STRING "SYCL target spec (spir64 = JIT, intel_gpu_* = AOT)")
set(IE_SYCL_DEVICE_HINT "bmg_g31"  CACHE STRING "Device hint for ocloc when target=spir64 (matches B70 IP 20.2.0)")

# SYCL flags shared by every device-side TU
add_library(ie_sycl_flags INTERFACE)
target_compile_options(ie_sycl_flags INTERFACE
  -fsycl
  -fsycl-targets=${IE_SYCL_TARGET}
  -fno-sycl-instrument-device-code
  -Wall -Wextra -Wpedantic
  -Wno-unused-parameter
)
target_link_options(ie_sycl_flags INTERFACE
  -fsycl
  -fsycl-targets=${IE_SYCL_TARGET}
  # SPIR-V extensions used by the ordinary SYCL/XMX path only.
  # Do not enable or imply ESIMD tile-load support here: every ESIMD tile-load
  # variant tried on BMG-G31 C0 caused DEVICE_LOST / Xe reset and remains behind
  # IE_ENABLE_UNSAFE_BMG_ESIMD_EXPERIMENTS, default OFF.
  -Xspirv-translator
  -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_subgroup_matrix_multiply_accumulate
)
if(IE_SYCL_TARGET STREQUAL "spir64" AND NOT IE_SYCL_DEVICE_HINT STREQUAL "")
  target_link_options(ie_sycl_flags INTERFACE
    "SHELL:-Xs \"-device ${IE_SYCL_DEVICE_HINT}\""
  )
endif()

# Verify icpx
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  message(WARNING
    "Expected IntelLLVM (icpx); got ${CMAKE_CXX_COMPILER_ID}. SYCL builds may fail.")
endif()
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "SYCL target: ${IE_SYCL_TARGET}  (device hint: '${IE_SYCL_DEVICE_HINT}')")

# ESIMD tile-load experiments — disabled by default.
# Every variant tried on BMG-G31 C0 (IP 20.2.0) causes
# UR_RESULT_ERROR_DEVICE_LOST / Xe exec-queue reset:
#   - lsc_load_block2d.ugm (Subgroup2DBlockLoad)
#   - lsc_load.ugm (M1_NM,1) d32x8t (SubgroupBlockRead)
#   - per-lane global indexing via LocalInvocationId
#   - SLM-staged global copy (still triggers DEVICE_LOST)
# Re-enable only when the root cause is understood.
option(IE_ENABLE_UNSAFE_BMG_ESIMD_EXPERIMENTS
  "Build ESIMD tile-load experiments (UNSAFE on BMG-G31: causes DEVICE_LOST)"
  OFF)

# oneDNN (Intel DNNL) fp16 GEMM path. Default ON (Intel build, byte-identical).
# Set OFF for non-Intel targets (e.g. a CUDA/nvptx64 build) so the tree does not
# require Intel's DNNL — gemm_fp16_onednn then links a stub that falls back to the
# in-house gemm_fp16. See docs/cuda_multiplatform_readiness_2026-06-13.md.
option(IE_ENABLE_ONEDNN
  "Build the oneDNN (Intel DNNL) fp16 GEMM path (OFF for non-Intel builds)"
  ON)

target_compile_definitions(ie_sycl_flags INTERFACE
  $<$<BOOL:${IE_ENABLE_UNSAFE_BMG_ESIMD_EXPERIMENTS}>:IE_ENABLE_UNSAFE_BMG_ESIMD_EXPERIMENTS=1>
)

# Sub-dirs
add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(bench)

enable_testing()
add_subdirectory(tests)
