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
  model/deepseek41_residency.cpp
  model/deepseek41_weights.cpp
  model/deepseek41_experts.cpp
  engine/engine.cpp
  engine/ds41_engine.cpp
  engine/glm5_server.cpp
  server/openai_proto.cpp
  server/capabilities.cpp
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
set_source_files_properties(model/deepseek41_engram.cpp PROPERTIES
  COMPILE_OPTIONS "-fopenmp")
target_link_options(ie_core PUBLIC -fopenmp)

add_executable(ie cli/main.cpp)
target_link_libraries(ie PRIVATE ie_core ie_sycl_flags)
