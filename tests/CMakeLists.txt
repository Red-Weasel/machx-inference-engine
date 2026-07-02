add_executable(dequant_test unit/dequant_test.cpp)
target_link_libraries(dequant_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME dequant_test COMMAND dequant_test)

add_executable(exl3_decode_test unit/exl3_decode_test.cpp)
target_link_libraries(exl3_decode_test PRIVATE ie_core ie_sycl_flags)
target_compile_definitions(exl3_decode_test PRIVATE
  EXL3_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data/exl3")
add_test(NAME exl3_decode_test COMMAND exl3_decode_test)

add_executable(dequant_bt_test unit/dequant_bt_test.cpp)
target_link_libraries(dequant_bt_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME dequant_bt_test COMMAND dequant_bt_test)

add_executable(elementwise_test unit/elementwise_test.cpp)
target_link_libraries(elementwise_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME elementwise_test COMMAND elementwise_test)

add_executable(gemv_q4k_test unit/gemv_q4k_test.cpp)
target_link_libraries(gemv_q4k_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_q4k_test COMMAND gemv_q4k_test)

# Gemma 4 Q4_0 GEMV/GEMM — validates against a real Gemma QAT tensor (SKIPs if absent).
add_executable(gemv_q4_0_test unit/gemv_q4_0_test.cpp)
target_link_libraries(gemv_q4_0_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_q4_0_test COMMAND gemv_q4_0_test)

# P2 Task 2: dense fp16 GEMV (qwen3 F16 attn_v). GPU + pure-random, no GGUF.
add_executable(gemv_fp16_test gemv_fp16_test.cpp)
target_link_libraries(gemv_fp16_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_fp16_test COMMAND gemv_fp16_test)

# P3b Task 1: K-tiled int-dot Q6_K GEMV (dense ffn_down decode). GPU +
# pure-random Q6_K, no GGUF; CPU double reference over the device q8 stream.
add_executable(gemv_q6k_ktiled_test gemv_q6k_ktiled_test.cpp)
target_link_libraries(gemv_q6k_ktiled_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_q6k_ktiled_test COMMAND gemv_q6k_ktiled_test)

# MXFP4 dequant gate (gpt-oss expert format): kernel vs llama-bit-exact host ref.
add_executable(mxfp4_dequant_test mxfp4_dequant_test.cpp)
target_link_libraries(mxfp4_dequant_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME mxfp4_dequant_test COMMAND mxfp4_dequant_test)

# P2 Task 4: DenseModel::forward GPU smoke — loads qwen3-8b, prefill + 4
# greedy decode steps, asserts finite logits.
# NOT registered with add_test (model-loading; GPU-exclusivity rule —
# concurrent ctest runs must not trigger a second model load; same
# precedent as engine_generate_test). Run manually:
#   ./build/tests/dense_forward_smoke
add_executable(dense_forward_smoke dense_forward_smoke.cpp)
target_link_libraries(dense_forward_smoke PRIVATE ie_core ie_sycl_flags)

add_executable(attention_test unit/attention_test.cpp)
target_link_libraries(attention_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME attention_test COMMAND attention_test)

add_executable(deltanet_test unit/deltanet_test.cpp)
target_link_libraries(deltanet_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deltanet_test COMMAND deltanet_test)

add_executable(moe_test unit/moe_test.cpp)
target_link_libraries(moe_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME moe_test COMMAND moe_test)

add_executable(tokenizer_test unit/tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME tokenizer_test COMMAND tokenizer_test)

if(IE_ENABLE_UNSAFE_BMG_ESIMD_EXPERIMENTS)
  # UNSAFE on BMG-G31: every ESIMD tile-load variant tried causes DEVICE_LOST.
  add_executable(esimd_block2d_smoke_test unit/esimd_block2d_smoke_test.cpp)
  target_link_libraries(esimd_block2d_smoke_test PRIVATE ie_core ie_sycl_flags)
  add_test(NAME esimd_block2d_smoke_test COMMAND esimd_block2d_smoke_test)
endif()

add_executable(sampler_test unit/sampler_test.cpp)
target_link_libraries(sampler_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME sampler_test COMMAND sampler_test)

add_executable(q8dot_test unit/q8dot_test.cpp)
target_link_libraries(q8dot_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME q8dot_test COMMAND q8dot_test)

add_executable(quant_soa_test unit/quant_soa_test.cpp)
target_link_libraries(quant_soa_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME quant_soa_test COMMAND quant_soa_test)

add_executable(engine_util_test unit/engine_util_test.cpp)
target_link_libraries(engine_util_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME engine_util_test COMMAND engine_util_test)

# Host-only (no GPU): P2 arch detection + DenseConfig metadata reader.
# Skips-with-warning when local GGUFs are absent. CI-safe.
add_executable(model_config_test model_config_test.cpp)
target_link_libraries(model_config_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME model_config_test COMMAND model_config_test)

add_executable(dense_unpermute_test dense_unpermute_test.cpp)
target_link_libraries(dense_unpermute_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME dense_unpermute_test COMMAND dense_unpermute_test)

# Host-only (no GPU, no SYCL link): Qwen3-MoE Task 2 — counting-sort MoE packer.
# Self-contained TU + packer source; does NOT link ie_core (pure host code).
add_executable(qwen3moe_pack_test
  unit/qwen3moe_pack_test.cpp
  ${CMAKE_SOURCE_DIR}/src/model/qwen3moe_pack.cpp)
target_link_libraries(qwen3moe_pack_test PRIVATE ie_sycl_flags)
target_include_directories(qwen3moe_pack_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
# Keep assert()s live: Release defines -DNDEBUG, which would no-op the test's
# checks. -UNDEBUG on the test TU makes the assertions actually validate.
target_compile_options(qwen3moe_pack_test PRIVATE -UNDEBUG)
add_test(NAME qwen3moe_pack_test COMMAND qwen3moe_pack_test)

# Wave-1: tekken pre-tokenizer encode-parity (Mistral-Nemo/Small-24B/Devstral).
# Host-only (NO GPU, NO model weights). Compiles the SYCL-free tokenizer +
# gguf_reader TUs directly (same precedent as qwen3moe_pack_test — does NOT
# link ie_core). Loads only the tokenizer metadata KVs from a real tekken GGUF
# (skips-with-warning if absent → CI-safe). -UNDEBUG keeps the asserts live.
add_executable(tekken_tokenizer_test
  unit/tekken_tokenizer_test.cpp
  ${CMAKE_SOURCE_DIR}/src/tokenizer/tokenizer.cpp
  ${CMAKE_SOURCE_DIR}/src/loaders/gguf_reader.cpp
  ${CMAKE_SOURCE_DIR}/src/core/dtype.cpp)
target_link_libraries(tekken_tokenizer_test PRIVATE ie_sycl_flags)
target_include_directories(tekken_tokenizer_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party)
target_compile_options(tekken_tokenizer_test PRIVATE -UNDEBUG)
add_test(NAME tekken_tokenizer_test COMMAND tekken_tokenizer_test)

# Phi-unblock Task 3: dbrx (cl100k / gpt2-tiktoken) pre-tokenizer encode-parity
# (Phi-4). Host-only (NO GPU, NO model weights). Compiles the SYCL-free
# tokenizer + gguf_reader + dtype TUs directly (same precedent as
# tekken_tokenizer_test — does NOT link ie_core). Loads only the tokenizer
# metadata KVs from a real dbrx GGUF (skips-with-warning if absent → CI-safe).
# -UNDEBUG keeps the asserts live.
add_executable(dbrx_pretok_test
  unit/dbrx_pretok_test.cpp
  ${CMAKE_SOURCE_DIR}/src/tokenizer/tokenizer.cpp
  ${CMAKE_SOURCE_DIR}/src/loaders/gguf_reader.cpp
  ${CMAKE_SOURCE_DIR}/src/core/dtype.cpp)
target_link_libraries(dbrx_pretok_test PRIVATE ie_sycl_flags)
target_include_directories(dbrx_pretok_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party)
target_compile_options(dbrx_pretok_test PRIVATE -UNDEBUG)
add_test(NAME dbrx_pretok_test COMMAND dbrx_pretok_test)

# Wave-1 Task 3/4: Mistral [INST] + DeepSeek-R1-Distill chat-template builders.
# Host-only (NO GPU, NO model weights) — pure string builders. Compiles the
# SYCL-free tokenizer + gguf_reader + dtype TUs directly (same precedent as
# tekken_tokenizer_test — does NOT link ie_core). -UNDEBUG keeps the asserts live.
add_executable(mistral_template_test
  unit/mistral_template_test.cpp
  ${CMAKE_SOURCE_DIR}/src/tokenizer/tokenizer.cpp
  ${CMAKE_SOURCE_DIR}/src/loaders/gguf_reader.cpp
  ${CMAKE_SOURCE_DIR}/src/core/dtype.cpp)
target_link_libraries(mistral_template_test PRIVATE ie_sycl_flags)
target_include_directories(mistral_template_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party)
target_compile_options(mistral_template_test PRIVATE -UNDEBUG)
add_test(NAME mistral_template_test COMMAND mistral_template_test)

# Wave-1 Task 1: arch-detect / template-family / SWA-guard PURE logic.
# Host-only (NO GPU, NO model weights). Compiles the SYCL-free model_config +
# gguf_reader + dtype TUs directly (same precedent as tekken_tokenizer_test —
# does NOT link ie_core). -UNDEBUG keeps the asserts live under Release.
add_executable(arch_template_test
  unit/arch_template_test.cpp
  ${CMAKE_SOURCE_DIR}/src/loaders/model_config.cpp
  ${CMAKE_SOURCE_DIR}/src/loaders/gguf_reader.cpp
  ${CMAKE_SOURCE_DIR}/src/core/dtype.cpp)
target_link_libraries(arch_template_test PRIVATE ie_sycl_flags)
target_include_directories(arch_template_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party)
target_compile_options(arch_template_test PRIVATE -UNDEBUG)
add_test(NAME arch_template_test COMMAND arch_template_test)

# Host-only (no GPU): P2 Task 6 — qwen3-8b dense vocab goldens vs
# llama-tokenize b8902 reference IDs. Skips-with-warning when the GGUF is
# absent. CI-safe.
add_executable(tokenizer_qwen3_golden_test tokenizer_qwen3_golden_test.cpp)
target_link_libraries(tokenizer_qwen3_golden_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME tokenizer_qwen3_golden_test COMMAND tokenizer_qwen3_golden_test)

# Host-only (no GPU): P3e Task 0 — safetensors reader against a synthetic file.
add_executable(safetensors_reader_test safetensors_reader_test.cpp)
target_link_libraries(safetensors_reader_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME safetensors_reader_test COMMAND safetensors_reader_test)

# Host-only (no GPU): P3e Task 1 — AWQ dequant interleave + (q-z)*s bit-exact.
add_executable(awq_dequant_test awq_dequant_test.cpp)
target_link_libraries(awq_dequant_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME awq_dequant_test COMMAND awq_dequant_test)

# Host-only (no GPU): P3e Task 2 — HF config parse + HF→GGUF name mapping.
add_executable(hf_import_test hf_import_test.cpp)
target_link_libraries(hf_import_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME hf_import_test COMMAND hf_import_test)

# Host-only (no GPU): P3e Task 2 — GGUF writer round-trips through GgufReader.
add_executable(gguf_writer_test gguf_writer_test.cpp)
target_link_libraries(gguf_writer_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gguf_writer_test COMMAND gguf_writer_test)

# Host-only (no GPU): P3e Task 2 — fp32->Q6_K encoder round-trip via ref decoder.
add_executable(quantize_q6k_test quantize_q6k_test.cpp)
target_link_libraries(quantize_q6k_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME quantize_q6k_test COMMAND quantize_q6k_test)

# Host-only (no GPU): pure OpenAI protocol translation. CI-safe.
add_executable(openai_proto_test unit/openai_proto_test.cpp)
target_link_libraries(openai_proto_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME openai_proto_test COMMAND openai_proto_test)

# Integration test: requires daily-driver GGUF + B70 GPU.
# NOT registered with add_test (model-dependent).
# Run manually: ./build/tests/engine_generate_test
add_executable(engine_generate_test integration/engine_generate_test.cpp)
target_link_libraries(engine_generate_test PRIVATE ie_core ie_sycl_flags)
