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
  loaders/gguf_reader.cpp
  loaders/model_config.cpp
  loaders/memory_plan.cpp
  loaders/safetensors_reader.cpp
  loaders/awq.cpp
  loaders/hf_import.cpp
  loaders/gguf_writer.cpp
  loaders/quantize_q6k.cpp
  loaders/quantize_q4k.cpp
  ops/dequant_kernels.cpp
  ops/elementwise.cpp
  ops/gemv_q4k.cpp
  ops/gemv_exl3.cpp
  ops/hadamard.cpp
  ops/gemv_q4_0.cpp
  ops/gemv_fp16.cpp
  ops/gemv_q8dot.cpp
  ${IE_ONEDNN_SRC}
  ops/gemv_q6k.cpp
  ops/gemv_q6_soa.cpp
  ops/gemv_q4_soa.cpp
  ops/gemv_mxfp4.cpp
  ops/gemm_fp16.cpp
  ops/gemm_q4k_xmx.cpp
  # gemm_q4k_esimd.cpp: gemm_q4_K_esimd (cooperative SG block reads,
  # safe on G31 C0 — Step 0 smoke verified) + the legacy
  # esimd_block2d_smoke_tile256 (kept for re-test plumbing, uses SLM
  # staging workaround, also safe).
  ops/gemm_q4k_esimd.cpp
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
  model/qwen36.cpp
  model/dense_transformer.cpp
  model/qwen35_dense.cpp
  model/qwen35_split.cpp
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
  engine/engine.cpp
  server/openai_proto.cpp
  server/openai_server.cpp
)
target_include_directories(ie_core
  PUBLIC ${CMAKE_SOURCE_DIR}/include
  PUBLIC ${CMAKE_SOURCE_DIR}/third_party
)
target_link_libraries(ie_core PUBLIC ie_sycl_flags
  $<$<BOOL:${IE_ENABLE_ONEDNN}>:DNNL::dnnl>)

add_executable(ie cli/main.cpp)
target_link_libraries(ie PRIVATE ie_core ie_sycl_flags)
