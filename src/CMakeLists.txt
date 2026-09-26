# P1b: oneDNN for the E1 prefill GEMM. Intel-only; gated for non-Intel (CUDA)
# builds — see docs/cuda_multiplatform_readiness_2026-06-13.md.
if(IE_ENABLE_ONEDNN)
  find_package(DNNL CONFIG REQUIRED PATHS /opt/intel/oneapi/dnnl/latest/lib/cmake)
  set(IE_ONEDNN_SRC ops/gemm_onednn.cpp)
else()
  set(IE_ONEDNN_SRC ops/gemm_onednn_stub.cpp)  # gemm_fp16_onednn -> gemm_fp16 fallback
endif()

add_library(ie_core
  core/dtype.cpp
  core/tensor.cpp
  core/allocator.cpp
  core/kv_cache.cpp
  core/deltanet_state.cpp
  core/prefix_cache.cpp
  core/fleet_prefix_cache.cpp
  core/exl3_decode.cpp
  core/expert_stream.cpp
  loaders/gguf_reader.cpp
  loaders/model_config.cpp
  loaders/memory_plan.cpp
  loaders/preflight.cpp
  loaders/safetensors_reader.cpp
  loaders/awq.cpp
  loaders/hf_import.cpp
  loaders/gguf_writer.cpp
  loaders/quantize_q6k.cpp
  loaders/quantize_q4k.cpp
  ops/dequant_kernels.cpp
  ops/elementwise.cpp
  ops/gemv_q4k.cpp
  ops/cpu_moe_gemv.cpp
  ops/cpu_moe_mxfp4.cpp
  ops/qwen4_vision_gpu.cpp
  ops/gemv_qk_grouped_moe.cpp
  ops/gemv_hy4_grouped_moe.cpp
  ops/gemv_exl3.cpp
  ops/hadamard.cpp
  ops/gemv_q4_0.cpp
  ops/gemv_q5_1.cpp
  ops/gemv_q5k.cpp
  ops/gemv_iq_raw.cpp
  ops/gemv_fp16.cpp
  ops/gemv_q8dot.cpp
  ops/gemv_fp8.cpp
  ${IE_ONEDNN_SRC}
  ops/gemv_q6k.cpp
  ops/gemv_q6_soa.cpp
  ops/gemv_q4_soa.cpp
  ops/gemv_q2_0_soa.cpp
  ops/gemv_q1_0_soa.cpp
  ops/gemv_mxfp4.cpp
  ops/gemv_iq3_xxs.cpp
  ops/gemm_fp16.cpp
  ops/gemm_q4k_xmx.cpp
  ops/gemm_q4k_dpas.cpp
  # gemm_q4k_esimd.cpp: gemm_q4_K_esimd (cooperative SG block reads,
  # safe on G31 C0 — Step 0 smoke verified) + the legacy
  # esimd_block2d_smoke_tile256 (kept for re-test plumbing, uses SLM
  # staging workaround, also safe).
  ops/gemm_q4k_esimd.cpp
  ops/deepseek4_ops.cpp
  ops/qwen4_hc.cpp
  ops/qwen4_ple.cpp
  ops/ds4_decode_gemv.cpp
  ops/deepseek4_experts.cpp
  ops/deepseek41_upload.cpp
  ops/deepseek41_ops.cpp
  ops/deepseek4_attn.cpp
  ops/deepseek4_attn_xmx.cpp
  ops/deepseek4_topk_radix.cpp
  ops/deepseek41_candidate.cpp
  ops/deepseek41_attn_xmx.cpp
  ops/deepseek41_topk_split.cpp
  ops/mimo26_ops.cpp
  ops/inkling_ops.cpp
  ops/laguna_ops.cpp
  ops/deepseek4_cache.cpp
  ops/attention.cpp
  ops/conv1d.cpp
  ops/deltanet.cpp
  ops/moe.cpp
  ops/moe_fused.cpp
  ops/moe_qwen3.cpp
  ops/moe_gemma4.cpp
  ops/moe_q8.cpp
  ops/sampling.cpp
  ops/embedding.cpp
  tokenizer/tokenizer.cpp
  tokenizer/tokenizer_hf_json.cpp
  model/qwen36.cpp
  model/dense_transformer.cpp
  model/qwen35_dense.cpp
  model/dspark_drafter.cpp
  model/qwen35_split.cpp
  model/qwen4exp.cpp
  model/qwen4_vision.cpp
  model/qwen4_image.cpp
  model/ds4_vision.cpp
  model/ds41_vision.cpp
  model/glm5next.cpp
  model/hyv4.cpp
  model/qwen35moe_split.cpp
  model/qwen35_tp.cpp
  model/dense_split.cpp
  model/dense_tp.cpp
  model/qwen3moe.cpp
  model/qwen3moe_split.cpp
  model/qwen3moe_tp.cpp
  model/qwen3moe_pack.cpp
  model/qwen3next.cpp
  model/gptoss.cpp
  model/gptoss_tp.cpp
  model/gemma4.cpp
  model/gemma4_assistant.cpp
  model/deepseek4.cpp
  model/deepseek41.cpp
  model/deepseek41_engram.cpp
  model/deepseek41_forward.cpp
  model/deepseek41_dspark.cpp
  model/deepseek41_prompt.cpp
  model/deepseek41_generate.cpp
  model/deepseek41_prefix_cache.cpp
  model/deepseek41_prefix_key.cpp
  model/deepseek41_residency.cpp
  model/deepseek41_weights.cpp
  model/deepseek41_experts.cpp
  model/mimo26.cpp
  model/mimo26_dequant.cpp
  model/mimo26_forward.cpp
  model/mimo26_dflash.cpp
  model/mimo26_host_rules.cpp
  model/mimo26_prefix_cache.cpp
  model/mimo26_vision.cpp
  model/mimo26_vision_tower.cpp
  engine/engine.cpp
  engine/ds41_engine.cpp
  engine/mimo26_engine.cpp
  engine/glm5_server.cpp
  server/openai_proto.cpp
  server/capabilities.cpp
  server/idle_spin.cpp
  server/openai_server.cpp
)
target_include_directories(ie_core
  PUBLIC ${CMAKE_SOURCE_DIR}/include
  PUBLIC ${CMAKE_SOURCE_DIR}/third_party
)
target_link_libraries(ie_core PUBLIC ie_sycl_flags
  $<$<BOOL:${IE_ENABLE_ONEDNN}>:DNNL::dnnl>)
set_source_files_properties(ops/cpu_moe_gemv.cpp PROPERTIES
  COMPILE_OPTIONS "-mavx2;-mfma;-fopenmp")
# The V4.1 decode miss path (docs/deepseek41/34): the CPU MXFP4 expert GEMV, AVX2 + FMA + an OpenMP team.
set_source_files_properties(ops/cpu_moe_mxfp4.cpp PROPERTIES
  COMPILE_OPTIONS "-mavx2;-mfma;-fopenmp")
# The V4.1 expert tier reads and packs experts on OpenMP reader threads (docs/deepseek41/23):
# without this the pragmas are silently ignored and every fill runs on one thread.
set_source_files_properties(model/deepseek41_experts.cpp PROPERTIES
  COMPILE_OPTIONS "-fopenmp")
# The vision tower's transient mode converts ~0.5 B BF16 weights into pinned f16 once (docs/deepseek41/94).
set_source_files_properties(model/ds4_vision.cpp PROPERTIES
  COMPILE_OPTIONS "-fopenmp")
# The V4.1 image plan must round exactly as Python does: icpx's default fast model turns x / 14 into x * (1 / 14),
# and floor(1134 / 14) came out 80 (docs/deepseek41/94).
set_source_files_properties(model/ds41_vision.cpp PROPERTIES
  COMPILE_OPTIONS "-fp-model=precise")
# the MiMo-V2.6 resize plan must round exactly as Python does (docs/mimo26/00_PORT_PLAN.md, P6.1)
set_source_files_properties(model/mimo26_vision.cpp PROPERTIES
  COMPILE_OPTIONS "-fp-model=precise")
set_source_files_properties(model/deepseek41_engram.cpp PROPERTIES
  COMPILE_OPTIONS "-fopenmp")
target_link_options(ie_core PUBLIC -fopenmp)

# ---- #48 (docs/deepseek41/103): the V4.1 disk prompt cache's numerics key -------------------------------------------
# The cache keys its disk entries by a NUMERICS MANIFEST instead of this executable's identity, so a rebuild that changes
# no arithmetic keeps the cache. The manifest is the SHA-256 of every file whose code can change a cached value
# (model/deepseek41_numerics_inputs.txt lists them, with the reasons), the compiler's version line and the build flags
# that reach those files -- the flags file below. model/deepseek41_numerics.cmake writes it into a generated header at
# build time. This block sits after every set_source_files_properties above: it records their per-source options.
set(IE_DS41_NUMERICS_LIST ${CMAKE_CURRENT_SOURCE_DIR}/model/deepseek41_numerics_inputs.txt)
set(IE_DS41_NUMERICS_DIR  ${CMAKE_CURRENT_BINARY_DIR}/generated)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${IE_DS41_NUMERICS_LIST})   # a new input: re-read the list
file(READ ${IE_DS41_NUMERICS_LIST} _ds41n)
string(REGEX REPLACE "[][;]" " " _ds41n "${_ds41n}")       # CMake list syntax: see deepseek41_numerics.cmake
string(REPLACE "\n" ";" _ds41n "${_ds41n}")
set(_ds41n_files)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _ds41n_bt)
get_target_property(_ds41n_sco ie_sycl_flags INTERFACE_COMPILE_OPTIONS)
get_target_property(_ds41n_slo ie_sycl_flags INTERFACE_LINK_OPTIONS)
get_target_property(_ds41n_scd ie_sycl_flags INTERFACE_COMPILE_DEFINITIONS)
get_target_property(_ds41n_clo ie_core INTERFACE_LINK_OPTIONS)
set(_ds41n_flags "compiler_id=${CMAKE_CXX_COMPILER_ID}\ncompiler_version=${CMAKE_CXX_COMPILER_VERSION}\n")
string(APPEND _ds41n_flags "build_type=${CMAKE_BUILD_TYPE}\ncxx_flags=${CMAKE_CXX_FLAGS}\ncxx_flags_build_type=${CMAKE_CXX_FLAGS_${_ds41n_bt}}\n")
string(APPEND _ds41n_flags "sycl_target=${IE_SYCL_TARGET}\nsycl_device_hint=${IE_SYCL_DEVICE_HINT}\n")
string(APPEND _ds41n_flags "onednn=${IE_ENABLE_ONEDNN}\nonednn_version_at_configure=${DNNL_VERSION}\nesimd_experiments=${IE_ENABLE_UNSAFE_BMG_ESIMD_EXPERIMENTS}\n")
string(APPEND _ds41n_flags "sycl_compile_options=${_ds41n_sco}\nsycl_link_options=${_ds41n_slo}\nsycl_compile_definitions=${_ds41n_scd}\ncore_link_options=${_ds41n_clo}\n")
foreach(_l IN LISTS _ds41n)
  string(REGEX REPLACE "#.*$" "" _l "${_l}")
  string(STRIP "${_l}" _l)
  if(_l STREQUAL "")
    continue()
  endif()
  list(APPEND _ds41n_files ${CMAKE_SOURCE_DIR}/${_l})
  if(_l MATCHES "^src/(.+\\.cpp)$")                          # the per-source options set above (-fopenmp, -mavx2 ...)
    get_source_file_property(_o ${CMAKE_MATCH_1} COMPILE_OPTIONS)
    if(_o)
      string(APPEND _ds41n_flags "source_options ${_l}=${_o}\n")
    endif()
  endif()
endforeach()
string(REPLACE ";" " " _ds41n_flags "${_ds41n_flags}")         # list-valued properties, flattened
file(MAKE_DIRECTORY ${IE_DS41_NUMERICS_DIR})
set(_ds41n_old "")
if(EXISTS ${IE_DS41_NUMERICS_DIR}/deepseek41_numerics_flags.txt)
  file(READ ${IE_DS41_NUMERICS_DIR}/deepseek41_numerics_flags.txt _ds41n_old)
endif()
if(NOT _ds41n_old STREQUAL _ds41n_flags)                         # rewritten only on a change: the stamp's DEPENDS below
  file(WRITE ${IE_DS41_NUMERICS_DIR}/deepseek41_numerics_flags.txt "${_ds41n_flags}")
endif()
add_custom_command(
  OUTPUT     ${IE_DS41_NUMERICS_DIR}/deepseek41_numerics.stamp
  BYPRODUCTS ${IE_DS41_NUMERICS_DIR}/deepseek41_numerics_manifest.h
  COMMAND ${CMAKE_COMMAND} -DROOT=${CMAKE_SOURCE_DIR} -DLIST=${IE_DS41_NUMERICS_LIST}
          -DFLAGS=${IE_DS41_NUMERICS_DIR}/deepseek41_numerics_flags.txt -DCXX=${CMAKE_CXX_COMPILER}
          -DOUT=${IE_DS41_NUMERICS_DIR}/deepseek41_numerics_manifest.h
          -DSTAMP=${IE_DS41_NUMERICS_DIR}/deepseek41_numerics.stamp
          -P ${CMAKE_CURRENT_SOURCE_DIR}/model/deepseek41_numerics.cmake
  DEPENDS ${_ds41n_files} ${IE_DS41_NUMERICS_LIST} ${IE_DS41_NUMERICS_DIR}/deepseek41_numerics_flags.txt
          ${CMAKE_CURRENT_SOURCE_DIR}/model/deepseek41_numerics.cmake ${CMAKE_CXX_COMPILER}
  COMMENT "DeepSeek-V4.1 numerics manifest (the disk prompt cache key, docs/deepseek41/103)"
  VERBATIM)
add_custom_target(ie_ds41_numerics DEPENDS ${IE_DS41_NUMERICS_DIR}/deepseek41_numerics.stamp)
add_dependencies(ie_core ie_ds41_numerics)
target_include_directories(ie_core PRIVATE ${IE_DS41_NUMERICS_DIR})

add_executable(ie cli/main.cpp)
target_link_libraries(ie PRIVATE ie_core ie_sycl_flags)
