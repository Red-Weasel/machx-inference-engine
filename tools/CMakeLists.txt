add_executable(ie_inspect inspect_gguf.cpp)
target_link_libraries(ie_inspect PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_inspect PROPERTIES OUTPUT_NAME "ie-inspect")

add_executable(ie_qwen35_load_test qwen35_load_test.cpp)
target_link_libraries(ie_qwen35_load_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen35_load_test PROPERTIES OUTPUT_NAME "ie-qwen35-load-test")

add_executable(ie_gptoss_load_test gptoss_load_test.cpp)
target_link_libraries(ie_gptoss_load_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_gptoss_load_test PROPERTIES OUTPUT_NAME "ie-gptoss-load-test")

add_executable(ie_qwen3next_config_test qwen3next_config_test.cpp)
target_link_libraries(ie_qwen3next_config_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen3next_config_test PROPERTIES OUTPUT_NAME "ie-qwen3next-config-test")

add_executable(ie_qwen3next_load_test qwen3next_load_test.cpp)
target_link_libraries(ie_qwen3next_load_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen3next_load_test PROPERTIES OUTPUT_NAME "ie-qwen3next-load-test")

add_executable(ie_gemma4_load_test gemma4_load_test.cpp)
target_link_libraries(ie_gemma4_load_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_gemma4_load_test PROPERTIES OUTPUT_NAME "ie-gemma4-load-test")

add_executable(ie_gemma4_gen gemma4_gen.cpp)
target_link_libraries(ie_gemma4_gen PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_gemma4_gen PROPERTIES OUTPUT_NAME "ie-gemma4-gen")

add_executable(ie_gemv_q4_0_batched_bench gemv_q4_0_batched_bench.cpp)
target_link_libraries(ie_gemv_q4_0_batched_bench PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_gemv_q4_0_batched_bench PROPERTIES OUTPUT_NAME "ie-gemv-q4-0-batched-bench")

add_executable(ie_gemma4_assistant_load_test gemma4_assistant_load_test.cpp)
target_link_libraries(ie_gemma4_assistant_load_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_gemma4_assistant_load_test PROPERTIES OUTPUT_NAME "ie-gemma4-assistant-load-test")

add_executable(ie_gemma4_spec ie_gemma4_spec.cpp)
target_link_libraries(ie_gemma4_spec PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_gemma4_spec PROPERTIES OUTPUT_NAME "ie-gemma4-spec")

add_executable(ie_gemma4_tok_test gemma4_tok_test.cpp)
target_link_libraries(ie_gemma4_tok_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_gemma4_tok_test PROPERTIES OUTPUT_NAME "ie-gemma4-tok-test")

add_executable(ie_gemma4_ppl gemma4_ppl.cpp)
target_link_libraries(ie_gemma4_ppl PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_gemma4_ppl PROPERTIES OUTPUT_NAME "ie-gemma4-ppl")

add_executable(ie_qwen3next_fwd_test qwen3next_fwd_test.cpp)
target_link_libraries(ie_qwen3next_fwd_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen3next_fwd_test PROPERTIES OUTPUT_NAME "ie-qwen3next-fwd-test")

# Step 2.5a — greedy-gen smoke (tokenize + forward + decode) for the completed fwd.
add_executable(ie_qwen3next_gen qwen3next_gen.cpp)
target_link_libraries(ie_qwen3next_gen PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen3next_gen PROPERTIES OUTPUT_NAME "ie-qwen3next-gen")

add_executable(ie_qwen3next_ppl qwen3next_ppl.cpp)
target_link_libraries(ie_qwen3next_ppl PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen3next_ppl PROPERTIES OUTPUT_NAME "ie-qwen3next-ppl")

add_executable(ie_q8_0_repack_bench q8_0_repack_bench.cpp)
target_link_libraries(ie_q8_0_repack_bench PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_q8_0_repack_bench PROPERTIES OUTPUT_NAME "ie-q8-0-repack-bench")

add_executable(ie_qwen3next_bench qwen3next_bench.cpp)
target_link_libraries(ie_qwen3next_bench PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen3next_bench PROPERTIES OUTPUT_NAME "ie-qwen3next-bench")

# Per-layer residual dumper for the 27B (qwen35) — engine side of diff_layers.sh.
add_executable(ie_qwen35_dump qwen35_dump.cpp)
target_link_libraries(ie_qwen35_dump PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen35_dump PROPERTIES OUTPUT_NAME "ie-qwen35-dump")

# STEP 0 de-risk gate: native MTP (blk.64) greedy acceptance for the 27B.
add_executable(ie_qwen35_mtp_accept ie_qwen35_mtp_accept.cpp)
target_link_libraries(ie_qwen35_mtp_accept PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen35_mtp_accept PROPERTIES OUTPUT_NAME "ie-qwen35-mtp-accept")

# STEP 1b+2: full greedy MTP self-speculative decode loop (LOSSLESS gate).
add_executable(ie_qwen35_spec ie_qwen35_spec.cpp)
target_link_libraries(ie_qwen35_spec PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen35_spec PROPERTIES OUTPUT_NAME "ie-qwen35-spec")

add_executable(ie_qwen36_spec ie_qwen36_spec.cpp)
target_link_libraries(ie_qwen36_spec PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen36_spec PROPERTIES OUTPUT_NAME "ie-qwen36-spec")

# P-A multi-GPU foundation: DeviceFleet enumeration + cross-device round-trip.
add_executable(ie_multi_gpu_probe multi_gpu_probe.cpp)
target_link_libraries(ie_multi_gpu_probe PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_multi_gpu_probe PROPERTIES OUTPUT_NAME "ie-multi-gpu-probe")

# P-B multi-GPU equality gate: single-GPU vs 2-GPU-split logits must be bit-identical.
add_executable(ie_multi_gpu_equality_test multi_gpu_equality_test.cpp)
target_link_libraries(ie_multi_gpu_equality_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_multi_gpu_equality_test PROPERTIES OUTPUT_NAME "ie-multi-gpu-equality-test")

# P-C multi-GPU runner: load a dense GGUF split across N GPUs + generate (for
# models too big for one card, e.g. the 72B across 2x B70). Split-only (no baseline).
add_executable(ie_multi_gpu_run multi_gpu_run.cpp)
target_link_libraries(ie_multi_gpu_run PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_multi_gpu_run PROPERTIES OUTPUT_NAME "ie-multi-gpu-run")

add_executable(ie_multi_gpu_ppl multi_gpu_ppl.cpp)
target_link_libraries(ie_multi_gpu_ppl PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_multi_gpu_ppl PROPERTIES OUTPUT_NAME "ie-multi-gpu-ppl")

add_executable(ie_multi_gpu_tp_test multi_gpu_tp_test.cpp)
target_link_libraries(ie_multi_gpu_tp_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_multi_gpu_tp_test PROPERTIES OUTPUT_NAME "ie-multi-gpu-tp-test")

# Per-device oneDNN ctx map gate: gemm_fp16_onednn on EACH B70 interleaved must
# not DEVICE_LOST (the multi-card landmine that kept oneDNN off the split paths).
add_executable(ie_onednn_multidev_test onednn_multidev_test.cpp)
target_link_libraries(ie_onednn_multidev_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_onednn_multidev_test PROPERTIES OUTPUT_NAME "ie-onednn-multidev-test")

add_executable(ie_forward forward_test.cpp)
target_link_libraries(ie_forward PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_forward PROPERTIES OUTPUT_NAME "ie-forward")

add_executable(ie_dump_embedding dump_embedding.cpp)
target_link_libraries(ie_dump_embedding PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_dump_embedding PROPERTIES OUTPUT_NAME "ie-dump-embedding")

add_executable(ie_debug ie_debug.cpp)
target_link_libraries(ie_debug PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_debug PROPERTIES OUTPUT_NAME "ie-debug")

add_executable(ie_bench ie_bench.cpp)
target_link_libraries(ie_bench PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_bench PROPERTIES OUTPUT_NAME "ie-bench")

add_executable(ie_perplexity ie_perplexity.cpp)
target_link_libraries(ie_perplexity PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_perplexity PROPERTIES OUTPUT_NAME "ie-perplexity")

# P2 Task 7: engine-side residual dumper for the dense (qwen3) path —
# counterpart of ie-llama-dump for tools/diff_layers.sh.
add_executable(ie_dense_dump ie_dense_dump.cpp)
target_link_libraries(ie_dense_dump PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_dense_dump PROPERTIES OUTPUT_NAME "ie-dense-dump")

# Chunking-equivalence validator: confirms forward(T=1024) ==
# 4 × forward(T=256) by comparing per-layer activations at the last
# position.  Tool for the prefill-PPL bug bisect (2026-05-02).
add_executable(ie_validate_chunking ie_validate_chunking.cpp)
target_link_libraries(ie_validate_chunking PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_validate_chunking PROPERTIES OUTPUT_NAME "ie-validate-chunking")

add_executable(ie_attn_profile ie_attn_profile.cpp)
target_link_libraries(ie_attn_profile PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_attn_profile PROPERTIES OUTPUT_NAME "ie-attn-profile")

add_executable(ie_kernel_monitor ie_kernel_monitor.cpp)
target_link_libraries(ie_kernel_monitor PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_kernel_monitor PROPERTIES OUTPUT_NAME "ie-kernel-monitor")

add_executable(ie_bench_suite ie_bench_suite.cpp)
target_link_libraries(ie_bench_suite PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_bench_suite PROPERTIES OUTPUT_NAME "ie-bench-suite")

add_executable(ie_kv_scale ie_kv_scale.cpp)
target_link_libraries(ie_kv_scale PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_kv_scale PROPERTIES OUTPUT_NAME "ie-kv-scale")

# PR #1 — partial-stack acceptance-rate measurement (DeltaNet-as-draft viability)
add_executable(ie_partial_stack_eval ie_partial_stack_eval.cpp)
target_link_libraries(ie_partial_stack_eval PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_partial_stack_eval PROPERTIES OUTPUT_NAME "ie-partial-stack-eval")

# PR #3 — prefix-cache snapshot/restore correctness gate (stage 1)
add_executable(ie_prefix_cache_test ie_prefix_cache_test.cpp)
target_link_libraries(ie_prefix_cache_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_prefix_cache_test PROPERTIES OUTPUT_NAME "ie-prefix-cache-test")

# PR #3 — prefix-cache class end-to-end (stage 2: trie + LRU)
add_executable(ie_prefix_cache_e2e ie_prefix_cache_e2e.cpp)
target_link_libraries(ie_prefix_cache_e2e PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_prefix_cache_e2e PROPERTIES OUTPUT_NAME "ie-prefix-cache-e2e")

# Live terminal monitor of the DeltaNet recurrence non-determinism.
# Two parallel forward(T=1) chains with separate KvCache + DeltaNetState;
# per-iter hash compare; colored alert when divergence first fires.
add_executable(ie_bug_live ie_bug_live.cpp)
target_link_libraries(ie_bug_live PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_bug_live PROPERTIES OUTPUT_NAME "ie-bug-live")

# Comprehensive live dashboard: determinism + per-iter latency graph +
# top-kernel timings + memory + auto-focus recommendations.  Constantly
# refreshing single-screen terminal UI.
add_executable(ie_bug_monitor ie_bug_monitor.cpp)
target_link_libraries(ie_bug_monitor PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_bug_monitor PROPERTIES OUTPUT_NAME "ie-bug-monitor")

# llama_dump links against the Vulkan-build of llama.cpp at /home/weezy/llama.cpp/build-vk/.
# It is the reference dumper for layer-by-layer activation diff.
set(LLAMA_VK_BUILD "/home/weezy/llama.cpp/build-vk")
set(LLAMA_SRC      "/home/weezy/llama.cpp")
add_executable(ie_llama_dump llama_dump.cpp)
target_include_directories(ie_llama_dump PRIVATE
    ${LLAMA_SRC}/include
    ${LLAMA_SRC}/ggml/include
    ${LLAMA_SRC}/common
)
target_link_directories(ie_llama_dump PRIVATE ${LLAMA_VK_BUILD}/bin)
target_link_libraries(ie_llama_dump PRIVATE llama ggml ggml-base)
target_compile_options(ie_llama_dump PRIVATE -Wno-unused-parameter)
set_target_properties(ie_llama_dump PROPERTIES
    OUTPUT_NAME "ie-llama-dump"
    BUILD_RPATH "${LLAMA_VK_BUILD}/bin"
    INSTALL_RPATH "${LLAMA_VK_BUILD}/bin"
)

# 2026-06-10 dense nondeterminism bisect: kernel-level determinism probe
add_executable(ie_det_probe ie_det_probe.cpp)
target_link_libraries(ie_det_probe PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_det_probe PROPERTIES OUTPUT_NAME "ie-det-probe")

add_executable(ie_import ie_import.cpp)
target_link_libraries(ie_import PRIVATE ie_core ie_sycl_flags)

add_executable(ie_qwen3moe_test qwen3moe_test.cpp)
target_link_libraries(ie_qwen3moe_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen3moe_test PROPERTIES OUTPUT_NAME "ie-qwen3moe-test")

# EXL3 (QTIP) Phase-B Task-2 GPU prototype: gemv_exl3 + hadamard_transform
# correctness vs the host oracle + speed vs gemv_q4_K (the decision gate).
add_executable(ie_exl3_test ie_exl3_test.cpp)
target_link_libraries(ie_exl3_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_exl3_test PROPERTIES OUTPUT_NAME "ie-exl3-test")

# In-engine spec-decode A/B: lossless + speedup vs plain greedy via Engine::generate.
add_executable(ie_spec_engine_ab ie_spec_engine_ab.cpp)
target_link_libraries(ie_spec_engine_ab PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_spec_engine_ab PROPERTIES OUTPUT_NAME "ie-spec-engine-ab")

add_executable(ie_prompt_bench ie_prompt_bench.cpp)
target_link_libraries(ie_prompt_bench PRIVATE ie_core ie_sycl_flags)

set_target_properties(ie_prompt_bench PROPERTIES OUTPUT_NAME "ie-prompt-bench")

# Host-only SoA-Q4 repack correctness gate (no GPU): repack_q4_K_to_soa
# reconstruction must be BIT-IDENTICAL to the AoS Q4_K dequant.
add_executable(ie_q4soa_test q4_soa_test.cpp)
target_link_libraries(ie_q4soa_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_q4soa_test PROPERTIES OUTPUT_NAME "ie-q4soa-test")

# gpt-oss 120b decode-optimization Step-0: the ONLY multi-GPU decode-tok/s bench
# (ie-bench is single-GPU). Also prints the per-section comm/router/gemv split
# under IE_GPTOSS_TP_TIMING=1 — is 120b decode comm-bound or spill-bound?
add_executable(ie_gptoss_tp_bench ie_gptoss_tp_bench.cpp)
target_link_libraries(ie_gptoss_tp_bench PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_gptoss_tp_bench PROPERTIES OUTPUT_NAME "ie-gptoss-tp-bench")

add_executable(ie_qwen35moe_bench ie_qwen35moe_bench.cpp)
target_link_libraries(ie_qwen35moe_bench PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen35moe_bench PROPERTIES OUTPUT_NAME "ie-qwen35moe-bench")
