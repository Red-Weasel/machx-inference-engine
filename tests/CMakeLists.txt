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

add_executable(gemv_iq_raw_test unit/gemv_iq_raw_test.cpp)
target_link_libraries(gemv_iq_raw_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_iq_raw_test COMMAND gemv_iq_raw_test)

add_executable(elementwise_test unit/elementwise_test.cpp)
target_link_libraries(elementwise_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME elementwise_test COMMAND elementwise_test)

add_executable(gemv_q4k_test unit/gemv_q4k_test.cpp)
target_link_libraries(gemv_q4k_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_q4k_test COMMAND gemv_q4k_test)

add_executable(gemv_q4_dual_dispatch_test unit/gemv_q4_dual_dispatch_test.cpp)
target_link_libraries(gemv_q4_dual_dispatch_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_q4_dual_dispatch_test COMMAND gemv_q4_dual_dispatch_test)
set_tests_properties(gemv_q4_dual_dispatch_test PROPERTIES SKIP_RETURN_CODE 77)

add_executable(gemv_q5_dispatch_test unit/gemv_q5_dispatch_test.cpp)
target_link_libraries(gemv_q5_dispatch_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_q5_dispatch_test COMMAND gemv_q5_dispatch_test)
set_tests_properties(gemv_q5_dispatch_test PROPERTIES SKIP_RETURN_CODE 77)

# Batched Q4/Q5, including out-of-order event dependencies.
add_executable(gemv_qk_rows_test unit/gemv_qk_rows_test.cpp)
target_link_libraries(gemv_qk_rows_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_qk_rows_test COMMAND gemv_qk_rows_test)

add_executable(conv1d_state_test unit/conv1d_state_test.cpp)
target_link_libraries(conv1d_state_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME conv1d_state_test COMMAND conv1d_state_test)

# Host Q4_K/Q5_K GEMV vs ie::ref dequant (no GPU). GLM-5.3-Flash CPU-miss path.
add_executable(cpu_moe_gemv_test unit/cpu_moe_gemv_test.cpp)
target_link_libraries(cpu_moe_gemv_test PRIVATE ie_core)
add_test(NAME cpu_moe_gemv_test COMMAND cpu_moe_gemv_test)

# Dual W8A16 GEMV bit-identity vs two singles (GPU1).
add_executable(gemv_q8_soa_dual_test unit/gemv_q8_soa_dual_test.cpp)
target_link_libraries(gemv_q8_soa_dual_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_q8_soa_dual_test COMMAND gemv_q8_soa_dual_test)

add_executable(gemm_fp16_shape_test unit/gemm_fp16_shape_test.cpp)
target_link_libraries(gemm_fp16_shape_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemm_fp16_shape_test COMMAND gemm_fp16_shape_test)
set_tests_properties(gemm_fp16_shape_test PROPERTIES SKIP_RETURN_CODE 77)

# Grouped Q4_K gate/up vs P gemv_q4_K_dual (GPU1). GLM-5.3 decode MoE.
add_executable(moe_q4_grouped_test unit/moe_q4_grouped_test.cpp)
target_link_libraries(moe_q4_grouped_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME moe_q4_grouped_test COMMAND moe_q4_grouped_test)

# Grouped Q5_K down vs P gemv_q5_K (GPU1). GLM-5.3 decode MoE.
add_executable(moe_q5_grouped_test unit/moe_q5_grouped_test.cpp)
target_link_libraries(moe_q5_grouped_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME moe_q5_grouped_test COMMAND moe_q5_grouped_test)

# Router load-ahead vs frozen FP32 lane order, including out-of-order queues.
add_executable(g5_router_test unit/g5_router_test.cpp)
target_link_libraries(g5_router_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME g5_router_test COMMAND g5_router_test)
set_tests_properties(g5_router_test PROPERTIES SKIP_RETURN_CODE 77)

add_executable(g5_router_topk_test unit/g5_router_topk_test.cpp)
target_link_libraries(g5_router_topk_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME g5_router_topk_test COMMAND g5_router_topk_test)
set_tests_properties(g5_router_topk_test PROPERTIES SKIP_RETURN_CODE 77)


# Prefill Q4_K XMX vs rows at M=4/8/16/32 (GPU1).
add_executable(gemm_q4_xmx_prefill_test unit/gemm_q4_xmx_prefill_test.cpp)
target_link_libraries(gemm_q4_xmx_prefill_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemm_q4_xmx_prefill_test COMMAND gemm_q4_xmx_prefill_test)

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
# Same binary with the XMX decode dispatch OFF: the xmx-vs-dispatcher rows then
# compare the XMX kernel against the SIMD16 kernel instead of against itself.
add_test(NAME attention_test_simd16 COMMAND attention_test)
set_tests_properties(attention_test_simd16 PROPERTIES ENVIRONMENT "IE_FA2_DECODE_XMX=0")

add_executable(deltanet_test unit/deltanet_test.cpp)
target_link_libraries(deltanet_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deltanet_test COMMAND deltanet_test)

add_executable(moe_test unit/moe_test.cpp)
target_link_libraries(moe_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME moe_test COMMAND moe_test)

# DeepSeek-V4 Phase 2: the five compute primitives vs the reference parity
# blobs.  SKIPs (exit 0) when the blob directories are absent; override the
# defaults with $DS4_PARITY_DIR / $DS4_PARITY2_DIR.
add_executable(deepseek4_parity_test unit/deepseek4_parity_test.cpp)
target_link_libraries(deepseek4_parity_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_parity_test COMMAND deepseek4_parity_test)

# Phase 4 gate: expert compute path (device IQ3_XXS GEMV + mixed IQ3_XXS/MXFP4
# expert dispatch).  Needs the parity blobs for the host-reference check and the
# real UD-Q3_K_XL GGUF for the device checks; reports honestly when either is
# absent rather than exiting green.
add_executable(deepseek4_experts_test unit/deepseek4_experts_test.cpp)
target_link_libraries(deepseek4_experts_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_experts_test COMMAND deepseek4_experts_test)

# Numerical gate for the two dense projection kernels (dense_f16/grouped_f16),
# which carry ~66% of prefill and had no coverage before this.  Registered TWICE
# so BOTH dispatch routes are gated: the fp32-accumulate kernels and the oneDNN
# fp16-activation route, each against the same double-precision host reference
# under its own derived tolerance.  Needs no model file.
add_executable(deepseek4_dense_gate_test unit/deepseek4_dense_gate_test.cpp)
target_link_libraries(deepseek4_dense_gate_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_dense_gate_test COMMAND deepseek4_dense_gate_test)
set_tests_properties(deepseek4_dense_gate_test PROPERTIES ENVIRONMENT "IE_DS4_ONEDNN=0")
add_test(NAME deepseek4_dense_gate_onednn_test COMMAND deepseek4_dense_gate_test)
set_tests_properties(deepseek4_dense_gate_onednn_test PROPERTIES ENVIRONMENT "IE_DS4_ONEDNN=1")

# Gates for the 2026-08-03 decode-latency fleet.
add_executable(deepseek4_sched_gate_test unit/deepseek4_sched_gate_test.cpp)
target_link_libraries(deepseek4_sched_gate_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_sched_gate_test COMMAND deepseek4_sched_gate_test)

add_executable(deepseek4_decgemv_gate_test unit/deepseek4_decgemv_gate_test.cpp)
target_link_libraries(deepseek4_decgemv_gate_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_decgemv_gate_test COMMAND deepseek4_decgemv_gate_test)

add_executable(deepseek4_ops_gate_test unit/deepseek4_ops_gate_test.cpp)
target_link_libraries(deepseek4_ops_gate_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_ops_gate_test COMMAND deepseek4_ops_gate_test)

add_executable(deepseek4_swiglu_f16_test unit/deepseek4_swiglu_f16_test.cpp)
target_link_libraries(deepseek4_swiglu_f16_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_swiglu_f16_test COMMAND deepseek4_swiglu_f16_test)

add_executable(deepseek4_router_tile_test unit/deepseek4_router_tile_test.cpp)
target_link_libraries(deepseek4_router_tile_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_router_tile_test COMMAND deepseek4_router_tile_test)

add_executable(deepseek4_indexer_tile_test unit/deepseek4_indexer_tile_test.cpp)
target_link_libraries(deepseek4_indexer_tile_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_indexer_tile_test COMMAND deepseek4_indexer_tile_test)

# V4.1 Phase 45 (docs/deepseek41/85): the candidate-skip score equals score + apply bit for bit.
add_executable(ds41_cand_skip_test unit/ds41_cand_skip_test.cpp)
target_link_libraries(ds41_cand_skip_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME ds41_cand_skip_test COMMAND ds41_cand_skip_test)

add_executable(deepseek4_hc_mix_test unit/deepseek4_hc_mix_test.cpp)
target_link_libraries(deepseek4_hc_mix_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_hc_mix_test COMMAND deepseek4_hc_mix_test)

# Gates owned one-per-agent by the 2026-08-03 streaming/compute optimisation pass.
# Wired here up front so three concurrent agents never edit this file.
add_executable(deepseek4_stream_gate_test unit/deepseek4_stream_gate_test.cpp)
target_link_libraries(deepseek4_stream_gate_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_stream_gate_test COMMAND deepseek4_stream_gate_test)

add_executable(deepseek4_expertgemm_gate_test unit/deepseek4_expertgemm_gate_test.cpp)
target_link_libraries(deepseek4_expertgemm_gate_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_expertgemm_gate_test COMMAND deepseek4_expertgemm_gate_test)

add_executable(deepseek4_mxfp4xmx_gate_test unit/deepseek4_mxfp4xmx_gate_test.cpp)
target_link_libraries(deepseek4_mxfp4xmx_gate_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_mxfp4xmx_gate_test COMMAND deepseek4_mxfp4xmx_gate_test)

add_executable(deepseek4_xmx_queue_test unit/deepseek4_xmx_queue_test.cpp)
target_link_libraries(deepseek4_xmx_queue_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_xmx_queue_test COMMAND deepseek4_xmx_queue_test)

add_executable(deepseek4_attn_xmx_gate_test unit/deepseek4_attn_xmx_gate_test.cpp)
target_link_libraries(deepseek4_attn_xmx_gate_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_attn_xmx_gate_test COMMAND deepseek4_attn_xmx_gate_test)

add_executable(deepseek4_decode_gate_test unit/deepseek4_decode_gate_test.cpp)
target_link_libraries(deepseek4_decode_gate_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_decode_gate_test COMMAND deepseek4_decode_gate_test)

# Sequence-slot gate (Ds4Options::seq_slots — the per-sequence attention state
# that continuous batching is built on).  Bitwise: one slot must be the numbers
# the single-conversation path already ships, and two slots must not be able to
# see each other.  Needs no model file and no parity blobs; it forces
# IE_DS4_ONEDNN=0 itself, since every check is a memcmp.
add_executable(deepseek4_seqcache_gate_test unit/deepseek4_seqcache_gate_test.cpp)
target_link_libraries(deepseek4_seqcache_gate_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_seqcache_gate_test COMMAND deepseek4_seqcache_gate_test)

# Prompt-cache gate (Ds4LayerCache::snapshot_to/restore_from and the runtime's
# single-endpoint snapshot).  Bitwise: restoring at depth N and prefilling the
# suffix must equal prefilling [0..N) and the suffix, the snapshot must survive
# later generation, and a restore of a different conversation must DIFFER.
# Needs no model file and no parity blobs; forces IE_DS4_ONEDNN=0 itself.
add_executable(deepseek4_promptcache_gate_test unit/deepseek4_promptcache_gate_test.cpp)
target_link_libraries(deepseek4_promptcache_gate_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_promptcache_gate_test COMMAND deepseek4_promptcache_gate_test)

# Gate for ds4_attention's mask-first early skip.  Needs NO parity blobs and no
# model file: it drives the same kernel down the skip and the walk paths and
# requires bit-identity, then checks both against a double-precision host softmax.
# docs/deepseek4/73 Phase 1 gate: the two-segment kv axis (Ds4KvSegs) must be
# bit-identical to the contiguous one on every dispatch shape, XMX included.
add_executable(deepseek4_attn_segs_test unit/deepseek4_attn_segs_test.cpp)
target_link_libraries(deepseek4_attn_segs_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_attn_segs_test COMMAND deepseek4_attn_segs_test)

add_executable(deepseek4_attn_sparse_test unit/deepseek4_attn_sparse_test.cpp)
target_link_libraries(deepseek4_attn_sparse_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_attn_sparse_test COMMAND deepseek4_attn_sparse_test)

# Phase 3 gate: attention stack (RoPE, sinks, compressors, cache, indexer).
add_executable(deepseek4_attn_test unit/deepseek4_attn_test.cpp)
target_link_libraries(deepseek4_attn_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_attn_test COMMAND deepseek4_attn_test)

# V4.1 decode miss path (docs/deepseek41/34): the CPU MXFP4 kernel vs the scalar reference and the nibble table
add_executable(cpu_moe_mxfp4_test unit/cpu_moe_mxfp4_test.cpp)
target_link_libraries(cpu_moe_mxfp4_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME cpu_moe_mxfp4_test COMMAND cpu_moe_mxfp4_test)

add_executable(gemv_fp8_test unit/gemv_fp8_test.cpp)
target_link_libraries(gemv_fp8_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_fp8_test COMMAND gemv_fp8_test)
add_executable(gemv_rows_test unit/gemv_rows_test.cpp)
target_link_libraries(gemv_rows_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_rows_test COMMAND gemv_rows_test)
set_tests_properties(gemv_fp8_test PROPERTIES SKIP_RETURN_CODE 77)

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

add_executable(gemv_q2_0_soa_test unit/gemv_q2_0_soa_test.cpp)
target_link_libraries(gemv_q2_0_soa_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_q2_0_soa_test COMMAND gemv_q2_0_soa_test)

add_executable(gemv_q1_0_soa_test unit/gemv_q1_0_soa_test.cpp)
target_link_libraries(gemv_q1_0_soa_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gemv_q1_0_soa_test COMMAND gemv_q1_0_soa_test)

# Perf microbench (not a ctest): per-shape GB/s of the Q2_0 decode GEMV at the
# Bonsai production shapes — docs/q2_0_optimization campaign instrument.
add_executable(gemv_q2_0_shape_bench unit/gemv_q2_0_shape_bench.cpp)
target_link_libraries(gemv_q2_0_shape_bench PRIVATE ie_core ie_sycl_flags)

add_executable(engine_util_test unit/engine_util_test.cpp)
target_link_libraries(engine_util_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME engine_util_test COMMAND engine_util_test)

# Process-wide runtime defaults (SYCL_CACHE_PERSISTENT) — host-only, no GPU.
add_executable(runtime_defaults_test unit/runtime_defaults_test.cpp)
target_link_libraries(runtime_defaults_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME runtime_defaults_test COMMAND runtime_defaults_test)

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

# DeepSeek-V4-Flash Phase 6: the joyai-llm pre-tokenizer (token-for-token vs
# llama.cpp), the unknown-`pre` diagnostic, and the encoding_dsv4.py prompt
# encoder + output parser. Host-only (NO GPU, NO model weights) — reads only the
# tokenizer metadata KVs from the real GGUF's metadata-only shard 1, same
# precedent as dbrx_pretok_test; skips with exit 0 if the model is absent.
# -UNDEBUG keeps the asserts live under Release.
add_executable(deepseek4_tokenizer_test
  unit/deepseek4_tokenizer_test.cpp
  ${CMAKE_SOURCE_DIR}/src/tokenizer/tokenizer.cpp
  ${CMAKE_SOURCE_DIR}/src/loaders/gguf_reader.cpp
  ${CMAKE_SOURCE_DIR}/src/core/dtype.cpp)
target_link_libraries(deepseek4_tokenizer_test PRIVATE ie_sycl_flags)
target_include_directories(deepseek4_tokenizer_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party)
target_compile_options(deepseek4_tokenizer_test PRIVATE -UNDEBUG)
add_test(NAME deepseek4_tokenizer_test COMMAND deepseek4_tokenizer_test)

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

# Preflight Task 1: pure dtype-classification predicate (is_loadable_weight_dtype,
# Global Constraint #4 supported set). Host-only (no GPU, no GGUF) — links ie_core
# for the symbol, same pattern as openai_proto_test.
add_executable(preflight_test unit/preflight_test.cpp)
target_link_libraries(preflight_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME preflight_test COMMAND preflight_test)

add_executable(server_capabilities_test unit/server_capabilities_test.cpp)
target_link_libraries(server_capabilities_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME server_capabilities_test COMMAND server_capabilities_test)

add_executable(glm5_weight_plan_test unit/glm5_weight_plan_test.cpp)
target_link_libraries(glm5_weight_plan_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME glm5_weight_plan_test COMMAND glm5_weight_plan_test)

add_executable(glm5_memory_policy_test unit/glm5_memory_policy_test.cpp)
target_include_directories(glm5_memory_policy_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME glm5_memory_policy_test COMMAND glm5_memory_policy_test)

add_executable(glm5_server_test unit/glm5_server_test.cpp)
target_link_libraries(glm5_server_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME glm5_server_test COMMAND glm5_server_test)

add_executable(engine_options_validation_test unit/engine_options_validation_test.cpp)
target_link_libraries(engine_options_validation_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME engine_options_validation_test COMMAND engine_options_validation_test)

add_executable(stop_sequences_test unit/stop_sequences_test.cpp)
target_include_directories(stop_sequences_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME stop_sequences_test COMMAND stop_sequences_test)

add_executable(gpu_count_test unit/gpu_count_test.cpp)
target_link_libraries(gpu_count_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gpu_count_test COMMAND gpu_count_test)
set_tests_properties(gpu_count_test PROPERTIES SKIP_RETURN_CODE 77)

add_executable(server_controls_test unit/server_controls_test.cpp)
target_link_libraries(server_controls_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME server_controls_test COMMAND server_controls_test)

add_executable(serve_options_test unit/serve_options_test.cpp)
target_link_libraries(serve_options_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME serve_options_test COMMAND serve_options_test)

# Server admission (bounded wait queue, shutdown, device-fault latch) — host-only.
add_executable(server_admission_test unit/server_admission_test.cpp)
target_link_libraries(server_admission_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME server_admission_test COMMAND server_admission_test)

add_executable(sampling_penalties_test unit/sampling_penalties_test.cpp)
target_link_libraries(sampling_penalties_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME sampling_penalties_test COMMAND sampling_penalties_test)
set_tests_properties(sampling_penalties_test PROPERTIES SKIP_RETURN_CODE 77)

# VRAM planner hard gate Task 2: pure should_block_oom predicate (refuse an
# over-budget, non-forced load unless IE_ALLOW_OOM). Host-only (no GPU, no GGUF)
# — links ie_core for the symbol, same pattern as preflight_test.
add_executable(memory_plan_gate_test unit/memory_plan_gate_test.cpp)
target_link_libraries(memory_plan_gate_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME memory_plan_gate_test COMMAND memory_plan_gate_test)

# Increment-2 universal loader: bit-exact KAT for the host reference dequant of
# Q4_1/Q5_0/Q5_1/Q2_K/Q3_K vs ggml's own golden output. Standalone / header-only
# — needs only ie/dequant_ref.hpp + the sibling dequant_fixtures.inc, so it does
# NOT link ie_core/ie_sycl_flags (no SYCL, fast CPU-only build). CI-safe.
add_executable(dequant_ref_test unit/dequant_ref_test.cpp)
target_include_directories(dequant_ref_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME dequant_ref_test COMMAND dequant_ref_test)

# DeepSeek-V4-Flash Milestone 0: detect_arch + read_deepseek4_config against the
# real GGUF's metadata-only shard 1 (5 MB, 0 tensors). Host-only (mmap header
# parse, no GPU); skips with exit 0 if the model is absent, like model_config_test.
add_executable(deepseek4_config_test unit/deepseek4_config_test.cpp)
target_link_libraries(deepseek4_config_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_config_test COMMAND deepseek4_config_test)

add_executable(inkling_shortconv_test unit/inkling_shortconv_test.cpp)
target_link_libraries(inkling_shortconv_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME inkling_shortconv_test COMMAND inkling_shortconv_test
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(laguna_ops_test unit/laguna_ops_test.cpp)
target_link_libraries(laguna_ops_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME laguna_ops_test COMMAND laguna_ops_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(laguna_manifest_test unit/laguna_manifest_test.cpp)
target_link_libraries(laguna_manifest_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME laguna_manifest_test COMMAND laguna_manifest_test)

add_executable(laguna_config_test unit/laguna_config_test.cpp)
target_link_libraries(laguna_config_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME laguna_config_test COMMAND laguna_config_test)

add_executable(inkling_load_test unit/inkling_load_test.cpp)
target_link_libraries(inkling_load_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME inkling_load_test COMMAND inkling_load_test)

add_executable(inkling_layer_test unit/inkling_layer_test.cpp)
target_link_libraries(inkling_layer_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME inkling_layer_test COMMAND inkling_layer_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(inkling_manifest_test unit/inkling_manifest_test.cpp)
target_link_libraries(inkling_manifest_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME inkling_manifest_test COMMAND inkling_manifest_test)

add_executable(inkling_config_test unit/inkling_config_test.cpp)
target_link_libraries(inkling_config_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME inkling_config_test COMMAND inkling_config_test)

# GgufReader must leave kernel readahead ON over every shard mapping: MADV_RANDOM
# zeroes the VMA's ra_pages and turns each mmap fault into one blocking 4 KiB
# read, which is wrong for both the sequential header walk and the whole-tensor
# streaming every loader does. Asserts on VmFlags (exact, never timing); the
# throughput lines it prints are diagnostics only. Host-only, skips if absent.
add_executable(gguf_madvise_test unit/gguf_madvise_test.cpp)
target_link_libraries(gguf_madvise_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME gguf_madvise_test COMMAND gguf_madvise_test)

# DeepSeek-V4-Flash Phase 1: DeepSeek4Model::load() binds all 1328 tensors of the
# real 4-shard GGUF (opens shard 1; the reader merges 2-4). Host-only (mmap header
# parse, no tensor bytes read, no GPU); skips with exit 0 if the model is absent.
add_executable(deepseek4_model_test unit/deepseek4_model_test.cpp)
target_link_libraries(deepseek4_model_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_model_test COMMAND deepseek4_model_test)

# Integration test: requires daily-driver GGUF + B70 GPU.
# NOT registered with add_test (model-dependent).
# Run manually: ./build/tests/engine_generate_test
add_executable(engine_generate_test integration/engine_generate_test.cpp)
target_link_libraries(engine_generate_test PRIVATE ie_core ie_sycl_flags)

# Phase 5 gate: expert streaming (pinned arena, per-card slot cache, transfer
# queue, measured overlap) + the forward-assembly primitives vs the parity2
# blobs.  §1-§3 need no model file; §4 needs $DS4_PARITY2_DIR and FAILS LOUDLY
# if it is absent (a vanished blob dir must never look like a green pass);
# §5 reports honestly on the real 128 GB GGUF.
add_executable(deepseek4_forward_test unit/deepseek4_forward_test.cpp)
target_link_libraries(deepseek4_forward_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_forward_test COMMAND deepseek4_forward_test)

# Tiered VRAM/host residency: an expert held permanently in the VRAM arena must
# NOT also keep a pinned host copy (docs/deepseek4/33 §2.2).  §1-§3 are pure
# arithmetic over the REAL tensor tables of both quants (mmap header parse, no
# weight byte read, no GPU) and degrade to a loud SKIP if a model is absent;
# §4-§6 allocate ~35 MB pinned host + ~2 MB device.  It never loads a model.
add_executable(deepseek4_residency_test unit/deepseek4_residency_test.cpp)
target_link_libraries(deepseek4_residency_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_residency_test COMMAND deepseek4_residency_test)

# qwen4exp vision tower CPU reference vs numpy golden oracle (P1 gate,
# docs/qwen4/16_vision_port.md). Needs IE_VIS_GOLDEN; skips without it.
add_executable(qwen4_vision_cpu_test unit/qwen4_vision_cpu_test.cpp)
target_link_libraries(qwen4_vision_cpu_test PRIVATE ie_core)
add_test(NAME qwen4_vision_cpu_test COMMAND qwen4_vision_cpu_test)

# 3-stream interleaved M-RoPE kernel (vision): text bit-identity + host-ref parity.
add_executable(rope_imrope3_test unit/rope_imrope3_test.cpp)
target_link_libraries(rope_imrope3_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME rope_imrope3_test COMMAND rope_imrope3_test)

# qwen4exp vision tower GPU path vs CPU reference (P3 gate).
add_executable(qwen4_vision_gpu_test unit/qwen4_vision_gpu_test.cpp)
target_link_libraries(qwen4_vision_gpu_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME qwen4_vision_gpu_test COMMAND qwen4_vision_gpu_test)

# DeepSeek-V4-Flash-Vision-Exp tower + aligner + block layout vs the numpy
# oracle golden (P2 gate, docs/deepseek4/70_VISION_EXP_PORT_PLAN.md). Needs
# IE_DS4_VIS_GOLDEN; skips without it.
add_executable(ds4_vision_gpu_test unit/ds4_vision_gpu_test.cpp)
target_link_libraries(ds4_vision_gpu_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME ds4_vision_gpu_test COMMAND ds4_vision_gpu_test)

# B70 KDA register tuning: exact output/state, OOO dependencies and shape errors.
add_executable(kda_recurrence_test unit/kda_recurrence_test.cpp)
target_link_libraries(kda_recurrence_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME kda_recurrence_test COMMAND kda_recurrence_test)
set_tests_properties(kda_recurrence_test PROPERTIES SKIP_RETURN_CODE 77)

# Shape-specialized indexer sort: exact causal/padding/tie behavior on both B70s.
add_executable(indexer_topk_test unit/indexer_topk_test.cpp)
target_link_libraries(indexer_topk_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME indexer_topk_test COMMAND indexer_topk_test)
set_tests_properties(indexer_topk_test PROPERTIES SKIP_RETURN_CODE 77)

add_executable(deepseek4_xmx_boundary_test unit/deepseek4_xmx_boundary_test.cpp)
target_link_libraries(deepseek4_xmx_boundary_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_xmx_boundary_test COMMAND deepseek4_xmx_boundary_test)
add_test(NAME deepseek4_xmx_boundary_fallback_test COMMAND deepseek4_xmx_boundary_test)
set_tests_properties(deepseek4_xmx_boundary_fallback_test PROPERTIES ENVIRONMENT "IE_DS4_EXPERT_XMX_FUSED=0")
set_tests_properties(deepseek4_xmx_boundary_test PROPERTIES ENVIRONMENT "IE_DS4_EXPERT_XMX_FUSED=1")

add_executable(deepseek4_rope_test unit/deepseek4_rope_test.cpp)
target_link_libraries(deepseek4_rope_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_rope_test COMMAND deepseek4_rope_test)

add_executable(deepseek4_attn_xmx_prep_test unit/deepseek4_attn_xmx_prep_test.cpp)
target_link_libraries(deepseek4_attn_xmx_prep_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_attn_xmx_prep_test COMMAND deepseek4_attn_xmx_prep_test)

add_executable(deepseek4_rms_reuse_test unit/deepseek4_rms_reuse_test.cpp)
target_link_libraries(deepseek4_rms_reuse_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_rms_reuse_test COMMAND deepseek4_rms_reuse_test)

add_executable(quantize_q8_1_test unit/quantize_q8_1_test.cpp)
target_link_libraries(quantize_q8_1_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME quantize_q8_1_test COMMAND quantize_q8_1_test)

add_executable(deepseek4_q8_dequant_test unit/deepseek4_q8_dequant_test.cpp)
target_link_libraries(deepseek4_q8_dequant_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_q8_dequant_test COMMAND deepseek4_q8_dequant_test)

add_executable(deepseek4_moe_movement_test unit/deepseek4_moe_movement_test.cpp)
target_link_libraries(deepseek4_moe_movement_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_moe_movement_test COMMAND deepseek4_moe_movement_test)

# Qwen partial/M-RoPE and GLM KDA gate: exact baseline, OOO events, guards.
add_executable(qwen_glm_kernel_test unit/qwen_glm_kernel_test.cpp)
target_link_libraries(qwen_glm_kernel_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME qwen_glm_kernel_test COMMAND qwen_glm_kernel_test)
add_test(NAME conv1d_empty_dependency_test COMMAND qwen_glm_kernel_test empty)

add_executable(conv1d_f32_test unit/conv1d_f32_test.cpp)
target_link_libraries(conv1d_f32_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME conv1d_f32_test COMMAND conv1d_f32_test)

add_executable(kda_norm_test unit/kda_norm_test.cpp)
target_link_libraries(kda_norm_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME kda_norm_test COMMAND kda_norm_test)

add_executable(dn_prepare_test unit/dn_prepare_test.cpp)
target_link_libraries(dn_prepare_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME dn_prepare_test COMMAND dn_prepare_test)

add_executable(kernel_movement_test unit/kernel_movement_test.cpp)
target_link_libraries(kernel_movement_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME kernel_movement_test COMMAND kernel_movement_test)

# Host-only: the Q8_0 -> F32 layout contract behind bind_f32_src (2026-09-11).
add_executable(deepseek4_q8_f32_bind_test unit/deepseek4_q8_f32_bind_test.cpp)
target_link_libraries(deepseek4_q8_f32_bind_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_q8_f32_bind_test COMMAND deepseek4_q8_f32_bind_test)

# Host-only: the dense-weight placement policy (kAuto / kKeepWide / kWideF16).
add_executable(deepseek4_dense_route_test unit/deepseek4_dense_route_test.cpp)
target_link_libraries(deepseek4_dense_route_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME deepseek4_dense_route_test COMMAND deepseek4_dense_route_test)

# Host-only (no GPU): OCP FP8 E4M3 / E8M0 element decoders vs torch goldens, all 256
# codes each. Closes the coverage gap the Phase 2 gate found in the V4.1 dense dequant.
add_executable(fp8_decode_test unit/fp8_decode_test.cpp)
target_link_libraries(fp8_decode_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME fp8_decode_test COMMAND fp8_decode_test)

# V4.1 mHC coefficient math vs the engine own ds4_hyper_connection: the duplicated
# Sinkhorn in deepseek41_ops.cpp must not drift from the V4 kernel. Needs a GPU; SKIPs (77) without.
add_executable(ds41_hc_parity_test unit/ds41_hc_parity_test.cpp)
target_link_libraries(ds41_hc_parity_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME ds41_hc_parity_test COMMAND ds41_hc_parity_test)
set_tests_properties(ds41_hc_parity_test PROPERTIES SKIP_RETURN_CODE 77)

# Host-only: the V4.1 three-tier expert residency planner (VRAM / pinned RAM / mmap).
add_executable(ds41_residency_plan_test unit/ds41_residency_plan_test.cpp)
target_link_libraries(ds41_residency_plan_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME ds41_residency_plan_test COMMAND ds41_residency_plan_test)

# MiMo-V2.6 P1 (docs/mimo26/00_PORT_PLAN.md): config parse, bind over the real checkpoint, host dequant vs the llama.cpp
# converter's and gguf-py's values (tools/mimo26/ref_dump.py). SKIPs (77) without the checkpoint or the reference dump.
add_executable(mimo26_model_test unit/mimo26_model_test.cpp)
target_link_libraries(mimo26_model_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME mimo26_model_test COMMAND mimo26_model_test)
set_tests_properties(mimo26_model_test PROPERTIES SKIP_RETURN_CODE 77)
# MiMo-V2.6 P2 kernels vs host references (attention with K 192 / V 128, sinks, window). Needs a GPU; SKIPs (77) without.
# The host sampler shared by DeepSeek-V4.1 and MiMo-V2.6: the nucleus-first sample_row vs the full sort it replaced. CPU only.
add_executable(ds41_sampler_test unit/ds41_sampler_test.cpp)
target_link_libraries(ds41_sampler_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME ds41_sampler_test COMMAND ds41_sampler_test)
# ie_vitals (docs/mimo26/IE_VITALS.md): sampler entropy / margin vs hand values, the pick unchanged, the window JSON, the parser. CPU only.
add_executable(ie_vitals_test unit/ie_vitals_test.cpp)
target_link_libraries(ie_vitals_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME ie_vitals_test COMMAND ie_vitals_test)
add_executable(mimo26_ops_test unit/mimo26_ops_test.cpp)
target_link_libraries(mimo26_ops_test PRIVATE ie_core ie_sycl_flags)
add_test(NAME mimo26_ops_test COMMAND mimo26_ops_test)
set_tests_properties(mimo26_ops_test PROPERTIES SKIP_RETURN_CODE 77)
# Host-only (no GPU, no SYCL): MiMo-V2.6 serving rules (docs/mimo26/P7_FIX64_FIX70.md) -- the host slots' ring runs, servable
# prefix, continuation test and admission policy (#70), and the drafter's fp16 range check (#64). Self-contained TU + the
# rules' source; does NOT link ie_core.
add_executable(mimo26_host_rules_test
  unit/mimo26_host_rules_test.cpp
  ${CMAKE_SOURCE_DIR}/src/model/mimo26_host_rules.cpp)
target_include_directories(mimo26_host_rules_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME mimo26_host_rules_test COMMAND mimo26_host_rules_test)
# Host-only (no GPU, no SYCL): #48 (docs/deepseek41/103) -- the V4.1 disk prompt cache key (the derivation) and its input
# list (every listed file exists, the list is closed under #include, and -- nm over ie_core's objects -- every function a
# listed source calls is defined in a listed or a named-unlisted source). Self-contained TU + the key's source; the
# repository root, the list and ie_core's objects are its arguments (it does not link ie_core, but needs it built).
add_executable(ds41_prefix_key_test
  unit/ds41_prefix_key_test.cpp
  ${CMAKE_SOURCE_DIR}/src/model/deepseek41_prefix_key.cpp)
target_include_directories(ds41_prefix_key_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_dependencies(ds41_prefix_key_test ie_core)
add_test(NAME ds41_prefix_key_test
         COMMAND ds41_prefix_key_test ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src/model/deepseek41_numerics_inputs.txt
                 --objects $<TARGET_OBJECTS:ie_core>
         COMMAND_EXPAND_LISTS)
# Host-only (no GPU, no SYCL): #29 (docs/server_idle_spin_watchdog_2026-09-24.md) -- the idle-spin watchdog's detector on
# fake samples, its /proc readers against a thread it spins, and its report line. Self-contained TU + the watchdog's source.
find_package(Threads REQUIRED)
add_executable(idle_spin_test
  unit/idle_spin_test.cpp
  ${CMAKE_SOURCE_DIR}/src/server/idle_spin.cpp)
target_include_directories(idle_spin_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(idle_spin_test PRIVATE Threads::Threads)
add_test(NAME idle_spin_test COMMAND idle_spin_test)
