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

# dspark M1 gate: engine DsparkDrafter forward vs fork golden vectors.
add_executable(ie_dspark_m1_gate ie_dspark_m1_gate.cpp)
target_link_libraries(ie_dspark_m1_gate PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_dspark_m1_gate PROPERTIES OUTPUT_NAME "ie-dspark-m1-gate")

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

# llama_dump links against the Vulkan-build of llama.cpp at $ENV{HOME}/llama.cpp/build-vk/.
# It is the reference dumper for layer-by-layer activation diff.
set(LLAMA_VK_BUILD "$ENV{HOME}/llama.cpp/build-vk")
set(LLAMA_SRC      "$ENV{HOME}/llama.cpp")
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

# DeepSeek-V4-Flash baseline harness: TTFT, prefill at several prompt lengths,
# decode at several context depths with its distribution, the expert-streaming
# DMA/hit counters, per-kernel attribution, and the profile -> priority -> re-run
# residency loop.  Emits one greppable `[ds4-bench] key=value` per metric so a
# later run can be diffed mechanically (--baseline).
add_executable(ie_ds4_bench ie_ds4_bench.cpp)
target_link_libraries(ie_ds4_bench PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds4_bench PROPERTIES OUTPUT_NAME "ie-ds4-bench")

# DeepSeek-V4-Flash QUALITY GATE: per-position NLL and perplexity through the
# DS4 runtime (ie_perplexity's arch dispatch predates this architecture and
# exits 1 on a deepseek4 GGUF).  `--gate` self-verifies the scorer on host-only
# inputs with closed-form answers, including two oracle alignment gates.
# `--dump-nll` / `--baseline` give the per-token A/B delta distribution.
add_executable(ie_ds4_ppl ie_ds4_ppl.cpp)
target_link_libraries(ie_ds4_ppl PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds4_ppl PROPERTIES OUTPUT_NAME "ie-ds4-ppl")

# DeepSeek-V4-Flash T>1 DENSE ROUTE cost, at the real per-card shapes, WITHOUT
# loading the 151 GB model.  Exists because the stated explanation for the Q8
# prefill regression (one extra pass over each weight per chunk, ~23 ms/card)
# is 229x too small to be the observed +5,259 ms, so the route that actually
# runs had to be measured rather than read.  Also times oneDNN's s8
# weight-decompression path and PRINTS the implementation oneDNN chose, because
# a wrong operand combination is accepted and silently dispatched to a reference
# kernel.  `--check` gates the numerics with three negative controls.
add_executable(ie_ds4_dense_prefill_bench ie_ds4_dense_prefill_bench.cpp)
target_link_libraries(ie_ds4_dense_prefill_bench PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds4_dense_prefill_bench PROPERTIES
                      OUTPUT_NAME "ie-ds4-dense-prefill-bench")

# GLM-5.2 (`glm-dsa`) config + tensor-layout gate. Verifies the engine reads the
# real 196.77 GiB GGUF correctly before any forward exists.
add_executable(ie_glm52_config_test glm52_config_test.cpp)
target_link_libraries(ie_glm52_config_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_glm52_config_test PROPERTIES OUTPUT_NAME "ie-glm52-config-test")

# GLM-5.3-Flash (`glm5next`) config gate (port P0). Hparams, the per-layer
# KDA/MLA attention map, and the per-block tensor schedule on the real 6-shard
# UD-Q4_K_XL GGUF, before any forward exists.
add_executable(ie_glm5next_config_test glm5next_config_test.cpp)
target_link_libraries(ie_glm5next_config_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_glm5next_config_test PROPERTIES OUTPUT_NAME "ie-glm5next-config-test")

# GLM-5.2 end-to-end forward (MLA + MoE) on one B70.
add_executable(ie_glm52_run glm52_run.cpp)
target_link_libraries(ie_glm52_run PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_glm52_run PROPERTIES OUTPUT_NAME "ie-glm52-run")

# Qwen3.8-Flash-Next (`qwen4exp`) config + tensor-layout gate (campaign P0).
# Exact tensor-map equality against config-derived shapes on the real 4-shard
# UD-Q4_K_XL GGUF, before any forward exists.
add_executable(ie_qwen4exp_config_test qwen4exp_config_test.cpp)
target_link_libraries(ie_qwen4exp_config_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_config_test PROPERTIES OUTPUT_NAME "ie-qwen4exp-config-test")

# Qwen3.8-Flash-Next load + placement gate: real weights onto one B70 (v0
# F16-everything device residency; expert banks + PLE table host-mmap).
add_executable(ie_qwen4exp_load_test qwen4exp_load_test.cpp)
target_link_libraries(ie_qwen4exp_load_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_load_test PROPERTIES OUTPUT_NAME "ie-qwen4exp-load-test")

# glm5next KDA kernel gate: fp64 references for kda_l2norm/kda_gate/
# kda_recurrence + the bit-exact state-carry contract. No model file.
add_executable(ie_glm5next_kda_test glm5next_kda_test.cpp)
target_link_libraries(ie_glm5next_kda_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_glm5next_kda_test PROPERTIES OUTPUT_NAME "ie-glm5next-kda-test")

# GLM-5.3-Flash load + placement gate: non-expert set onto one B70 (v0
# F16-everything device residency; expert banks host-mmap; MTP bound-not-uploaded).
add_executable(ie_glm5next_load_test glm5next_load_test.cpp)
target_link_libraries(ie_glm5next_load_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_glm5next_load_test PROPERTIES OUTPUT_NAME "ie-glm5next-load-test")

# GLM-5.3-Flash end-to-end forward smoke + chunked PPL (port P2 gate).
add_executable(ie_glm5next_run glm5next_run.cpp)
target_link_libraries(ie_glm5next_run PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_glm5next_run PROPERTIES OUTPUT_NAME "ie-glm5next-run")

# qwen4exp Q5_1 GEMV/GEMM + BF16/Q5_K conversion gate (expert down banks are
# Q5_1 — native GPU path; correctness vs CPU refs, no model file).
add_executable(ie_qwen4_quant_test qwen4_quant_test.cpp)
target_link_libraries(ie_qwen4_quant_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4_quant_test PROPERTIES OUTPUT_NAME "ie-qwen4-quant-test")

# qwen4exp hyper-connections correctness gate: GPU mix/combine/expand kernels
# vs a CPU fp64 reference of docs/qwen4/11_hyper_connections.md. Self-contained.
add_executable(ie_qwen4_hc_test qwen4_hc_test.cpp)
target_link_libraries(ie_qwen4_hc_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4_hc_test PROPERTIES OUTPUT_NAME "ie-qwen4-hc-test")

# Phase 1b (agent serving): in-process slot-stash suspend/resume byte-exactness
# harness for the 27B 2-GPU split path (cross-restart greedy is nondeterministic,
# so this comparison must live inside one process — see
# docs/agent-serving-campaign.md).
add_executable(ie_slot_stash_test ie_slot_stash_test.cpp)
target_link_libraries(ie_slot_stash_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_slot_stash_test PROPERTIES OUTPUT_NAME "ie-slot-stash-test")

# Phase 2a (decode-throughput): batched-vs-solo equivalence for
# Qwen35SplitModel::forward_slots — noise floor, logit deviation, argmax, and
# final bank-state bit-compare (docs/decode-throughput-campaign.md).
add_executable(ie_batched_equiv_test ie_batched_equiv_test.cpp)
target_link_libraries(ie_batched_equiv_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_batched_equiv_test PROPERTIES OUTPUT_NAME "ie-batched-equiv-test")

# qwen4exp PLE n-gram machinery gate (no model file): host hash/gather vs
# hand-computed + independent references; device blk.1 PLE layer vs CPU fp64;
# chunked-vs-monolithic conv-state carry.
add_executable(ie_qwen4_ple_test qwen4_ple_test.cpp)
target_link_libraries(ie_qwen4_ple_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4_ple_test PROPERTIES OUTPUT_NAME "ie-qwen4-ple-test")

# Qwen3.8-Flash-Next first-forward smoke: prefill + greedy decode on the real
# GGUF, finiteness/determinism checks (parity vs the HF oracle is the next gate).
add_executable(ie_qwen4exp_fwd_test qwen4exp_fwd_test.cpp)
target_link_libraries(ie_qwen4exp_fwd_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_fwd_test PROPERTIES OUTPUT_NAME "ie-qwen4exp-fwd-test")

# Qwen3.8-Flash-Next greedy generation smoke (tokenize + forward + decode).
add_executable(ie_qwen4exp_gen qwen4exp_gen.cpp)
target_link_libraries(ie_qwen4exp_gen PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_gen PROPERTIES OUTPUT_NAME "ie-qwen4exp-gen")

# Qwen3.8-Flash-Next P2 parity gate: run_block_parity vs the HF oracle npz
# goldens (scripts/qwen4exp_oracle/), real weights, per-stage rel-L2.
add_executable(ie_qwen4exp_parity_test qwen4exp_parity_test.cpp)
target_link_libraries(ie_qwen4exp_parity_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_parity_test PROPERTIES OUTPUT_NAME "ie-qwen4exp-parity-test")

# Flash-Next decode profile: wall vs GPU-busy + kernel buckets (charter step 1).
add_executable(ie_qwen4exp_profile qwen4exp_profile.cpp)
target_link_libraries(ie_qwen4exp_profile PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_profile PROPERTIES OUTPUT_NAME "ie-qwen4exp-profile")

# Flash-Next 2-GPU layer-split pipeline + bit-equality gate vs single-GPU.
add_executable(ie_qwen4exp_run2 qwen4exp_run2.cpp)
target_link_libraries(ie_qwen4exp_run2 PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_run2 PROPERTIES OUTPUT_NAME "ie-qwen4exp-run2")

# Flash-Next perplexity: streaming-T=1 NLL (quality gate; certifies quant
# residency trades like IE_Q4E_DENSE_Q8).
add_executable(ie_qwen4exp_ppl qwen4exp_ppl.cpp)
target_link_libraries(ie_qwen4exp_ppl PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_ppl PROPERTIES OUTPUT_NAME "ie-qwen4exp-ppl")

# Flash-Next MTP self-speculative decode: LOSSLESS gate + acceptance/speed.
add_executable(ie_qwen4exp_bvs qwen4exp_bvs_test.cpp)
target_link_libraries(ie_qwen4exp_bvs PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_bvs PROPERTIES OUTPUT_NAME "ie-qwen4exp-bvs")

add_executable(ie_q4k_bench q4k_kernel_bench.cpp)
target_link_libraries(ie_q4k_bench PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_q4k_bench PROPERTIES OUTPUT_NAME "ie-q4k-bench")

add_executable(ie_qwen4exp_spec3 qwen4exp_spec3.cpp)
target_link_libraries(ie_qwen4exp_spec3 PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_spec3 PROPERTIES OUTPUT_NAME "ie-qwen4exp-spec3")

add_executable(ie_qwen4exp_spec2 qwen4exp_spec2.cpp)
target_link_libraries(ie_qwen4exp_spec2 PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_spec2 PROPERTIES OUTPUT_NAME "ie-qwen4exp-spec2")

add_executable(ie_qwen4exp_spec qwen4exp_spec.cpp)
target_link_libraries(ie_qwen4exp_spec PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_spec PROPERTIES OUTPUT_NAME "ie-qwen4exp-spec")

# qwen4exp per-slot suspend/resume byte-exactness gate (in-process).
add_executable(ie_qwen4exp_slot_test qwen4exp_slot_test.cpp)
target_link_libraries(ie_qwen4exp_slot_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_slot_test PROPERTIES OUTPUT_NAME "ie-qwen4exp-slot-test")

# CPU ground-truth arbiter for the glm5next EP-verify oracle (IE_G5_EPV_DUMP).
add_executable(ie_glm5next_epv_ref glm5next_epv_ref.cpp)
target_link_libraries(ie_glm5next_epv_ref PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_glm5next_epv_ref PROPERTIES OUTPUT_NAME "ie-glm5next-epv-ref")

# qwen4exp vision tower probe: image -> merged embeddings (CPU reference path).
add_executable(ie_qwen4_vision_probe qwen4_vision_probe.cpp)
target_link_libraries(ie_qwen4_vision_probe PRIVATE ie_core)
set_target_properties(ie_qwen4_vision_probe PROPERTIES OUTPUT_NAME "ie-qwen4-vision-probe")

# W8A8 SoA GEMV bandwidth probe at qwen4exp decode shapes (roofline instrument).
add_executable(ie_gemv_shapes_bench gemv_shapes_bench.cpp)
target_link_libraries(ie_gemv_shapes_bench PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_gemv_shapes_bench PROPERTIES OUTPUT_NAME "ie-gemv-shapes-bench")

# Decode determinism stress (race hunter): repeat identical forwards, bit-compare.
add_executable(ie_qwen4exp_det_stress qwen4exp_det_stress.cpp)
target_link_libraries(ie_qwen4exp_det_stress PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_qwen4exp_det_stress PROPERTIES OUTPUT_NAME "ie-qwen4exp-det-stress")

# Tencent Hy4-preview (`hyv4`) end-to-end runner — forked from the glm5next
# runner: load, greedy generation, --ppl/--chunk scoring, 2-GPU stage split.
add_executable(ie_hyv4_run hyv4_run.cpp)
target_link_libraries(ie_hyv4_run PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_hyv4_run PROPERTIES OUTPUT_NAME "ie-hyv4-run")

# DeepSeek-V4.1-Flash Phase 2 probe (host-only): read the shipped safetensors
# through the engine, dequantise one dense tensor and one routed expert.
add_executable(ie_ds41_probe ds41_probe.cpp)
target_link_libraries(ie_ds41_probe PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_probe PROPERTIES OUTPUT_NAME "ie-ds41-probe")

# DeepSeek-V4.1-Flash Phase 3: config parse + bind all 96,085 tensors (host-only).
add_executable(ie_ds41_load_test ds41_load_test.cpp)
target_link_libraries(ie_ds41_load_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_load_test PROPERTIES OUTPUT_NAME "ie-ds41-load-test")

# DeepSeek-V4.1-Flash Phase 4: expert upload + dense FP8 dequant, on the GPU.
add_executable(ie_ds41_device_test ds41_device_test.cpp)
target_link_libraries(ie_ds41_device_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_device_test PROPERTIES OUTPUT_NAME "ie-ds41-device-test")

# DeepSeek-V4.1-Flash Phase 5a: routed-expert block vs the reference golden.
add_executable(ie_ds41_moe_test ds41_moe_test.cpp)
target_link_libraries(ie_ds41_moe_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_moe_test PROPERTIES OUTPUT_NAME "ie-ds41-moe-test")

# DeepSeek-V4.1-Flash Phase 5b: attention block vs the reference golden, stage by stage.
add_executable(ie_ds41_attn_test ds41_attn_test.cpp)
target_link_libraries(ie_ds41_attn_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_attn_test PROPERTIES OUTPUT_NAME "ie-ds41-attn-test")

# DeepSeek-V4.1-Flash Phase 5c: a WHOLE layer vs the reference Block(0).
add_executable(ie_ds41_block_test ds41_block_test.cpp)
target_link_libraries(ie_ds41_block_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_block_test PROPERTIES OUTPUT_NAME "ie-ds41-block-test")

# DeepSeek-V4.1-Flash Phase 6a: engram vs the reference (hash integer-exact, gather bit-exact).
add_executable(ie_ds41_engram_test ds41_engram_test.cpp)
target_link_libraries(ie_ds41_engram_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_engram_test PROPERTIES OUTPUT_NAME "ie-ds41-engram-test")

# DeepSeek-V4.1-Flash Phase 6b: compressed attention at layer 2 (compressor, indexer, top-k, two-segment KV).
add_executable(ie_ds41_attn2_test ds41_attn2_test.cpp)
target_link_libraries(ie_ds41_attn2_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_attn2_test PROPERTIES OUTPUT_NAME "ie-ds41-attn2-test")

# DeepSeek-V4.1-Flash Phase 7: the whole text forward vs the full-model golden (layer by layer, then the token).
add_executable(ie_ds41_forward_test ds41_forward_test.cpp)
target_link_libraries(ie_ds41_forward_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_forward_test PROPERTIES OUTPUT_NAME "ie-ds41-forward-test")

# DeepSeek-V4.1-Flash Phase 8 step 1: the dense text path resident on a device.
add_executable(ie_ds41_dense_cache_test ds41_dense_cache_test.cpp)
target_link_libraries(ie_ds41_dense_cache_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_dense_cache_test PROPERTIES OUTPUT_NAME "ie-ds41-dense-cache-test")

# DeepSeek-V4.1-Flash Phase 8: experts through V4 residency machinery, three tiers, vs the MoE golden.
add_executable(ie_ds41_tier_test ds41_tier_test.cpp)
target_link_libraries(ie_ds41_tier_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_tier_test PROPERTIES OUTPUT_NAME "ie-ds41-tier-test")

# DeepSeek-V4.1-Flash Phase 8: the resident runtime -- correctness, bytes by tier, prefill tok/s, routing profile.
add_executable(ie_ds41_resident_test ds41_resident_test.cpp)
target_link_libraries(ie_ds41_resident_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_resident_test PROPERTIES OUTPUT_NAME "ie-ds41-resident-test")

# DeepSeek-V4.1-Flash Phase 9: decode on the resident runtime vs the reference's generate loop (docs/deepseek41/19, 24).
add_executable(ie_ds41_lookup_test ds41_lookup_test.cpp)
target_link_libraries(ie_ds41_lookup_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_lookup_test PROPERTIES OUTPUT_NAME "ie-ds41-lookup-test")

add_executable(ie_ds41_vision_test ds41_vision_test.cpp)
target_link_libraries(ie_ds41_vision_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_vision_test PROPERTIES OUTPUT_NAME "ie-ds41-vision-test")
add_executable(ie_ds41_vision_leak ds41_vision_leak.cpp)
target_link_libraries(ie_ds41_vision_leak PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_vision_leak PROPERTIES OUTPUT_NAME "ie-ds41-vision-leak")

add_executable(ie_ds41_decode_test ds41_decode_test.cpp)
target_link_libraries(ie_ds41_decode_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_decode_test PROPERTIES OUTPUT_NAME "ie-ds41-decode-test")
add_executable(ie_ds41_multi_test ds41_multi_test.cpp)
target_link_libraries(ie_ds41_multi_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_multi_test PROPERTIES OUTPUT_NAME "ie-ds41-multi-test")
add_executable(ie_ds41_rollback_test ds41_rollback_test.cpp)
target_link_libraries(ie_ds41_rollback_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_rollback_test PROPERTIES OUTPUT_NAME "ie-ds41-rollback-test")
add_executable(ie_ds41_dspark_test ds41_dspark_test.cpp)
target_link_libraries(ie_ds41_dspark_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_dspark_test PROPERTIES OUTPUT_NAME "ie-ds41-dspark-test")

# DeepSeek-V4.1-Flash Phase 12: the tokenizer.json loader vs the Python reference, token for token (docs/deepseek41/31).
add_executable(ie_ds41_tokenizer_test ds41_tokenizer_test.cpp)
target_link_libraries(ie_ds41_tokenizer_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_tokenizer_test PROPERTIES OUTPUT_NAME "ie-ds41-tokenizer-test")

# DeepSeek-V4.1-Flash Phase 12: the prompt format vs the checkpoint's encoding.py, byte for byte (docs/deepseek41/31).
add_executable(ie_ds41_prompt_test ds41_prompt_test.cpp)
target_link_libraries(ie_ds41_prompt_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_prompt_test PROPERTIES OUTPUT_NAME "ie-ds41-prompt-test")

# DeepSeek-V4.1-Flash Phase 12: the generate loop vs the decode and replay tests' tokens (docs/deepseek41/31).
add_executable(ie_ds41_generate_test ds41_generate_test.cpp)
target_link_libraries(ie_ds41_generate_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_generate_test PROPERTIES OUTPUT_NAME "ie-ds41-generate-test")
add_executable(ie_ds41_spec_test ds41_spec_test.cpp)
target_link_libraries(ie_ds41_spec_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_spec_test PROPERTIES OUTPUT_NAME "ie-ds41-spec-test")
add_executable(ie_ds41_candidate_test ds41_candidate_test.cpp)
target_link_libraries(ie_ds41_candidate_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_candidate_test PROPERTIES OUTPUT_NAME "ie-ds41-candidate-test")

add_executable(ie_ds41_long_test ds41_long_test.cpp)
target_link_libraries(ie_ds41_long_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_long_test PROPERTIES OUTPUT_NAME "ie-ds41-long-test")

add_executable(ie_ds41_cont_test ds41_cont_test.cpp)
target_link_libraries(ie_ds41_cont_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_cont_test PROPERTIES OUTPUT_NAME "ie-ds41-cont-test")

# DeepSeek-V4.1-Flash Phase 12: the CLI -- a prompt or a chat, streamed (docs/deepseek41/31).
add_executable(ie_ds41_run ds41_run.cpp)
target_link_libraries(ie_ds41_run PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_run PROPERTIES OUTPUT_NAME "ie-ds41-run")

# DeepSeek-V4.1-Flash Phase 43: perplexity on real text, pairwise across routes (docs/deepseek41/83).
add_executable(ie_ds41_ppl ds41_ppl.cpp)
target_link_libraries(ie_ds41_ppl PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_ppl PROPERTIES OUTPUT_NAME "ie-ds41-ppl")

# DeepSeek-V4.1-Flash Phase 13: the CPU MXFP4 expert kernel's rate on real experts, cold, per core set (docs/deepseek41/34).
add_executable(ie_ds41_cpu_expert_bench ds41_cpu_expert_bench.cpp)
target_link_libraries(ie_ds41_cpu_expert_bench PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_cpu_expert_bench PROPERTIES OUTPUT_NAME "ie-ds41-cpu-expert-bench")

# DeepSeek-V4.1-Flash Phase 10: Decoder SWA Bounded Replay vs the exact path (docs/deepseek41/27).
add_executable(ie_ds41_replay_test ds41_replay_test.cpp)
target_link_libraries(ie_ds41_replay_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_replay_test PROPERTIES OUTPUT_NAME "ie-ds41-replay-test")

# Phase 13 step 1: the residency placement bench (docs/deepseek41/34-35)
add_executable(ie_ds41_placement_bench ds41_placement_bench.cpp)
target_link_libraries(ie_ds41_placement_bench PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_placement_bench PROPERTIES OUTPUT_NAME "ie-ds41-placement-bench")

# Phase 14 step 0: the two PCIe links' H2D rate, alone and together (docs/deepseek41/39)
add_executable(ie_ds41_dma_pair_bench ds41_dma_pair_bench.cpp)
target_link_libraries(ie_ds41_dma_pair_bench PRIVATE ie_sycl_flags)
set_target_properties(ie_ds41_dma_pair_bench PROPERTIES OUTPUT_NAME "ie-ds41-dma-pair-bench")

# Phase 15 C2: the mmap tier's experts in slot layout on disk (docs/deepseek41/40)
add_executable(ie_ds41_expert_file ds41_expert_file.cpp)
target_link_libraries(ie_ds41_expert_file PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_expert_file PROPERTIES OUTPUT_NAME "ie-ds41-expert-file")

# the pp / corpus token files of the ds41 tests, reproducible from text
add_executable(ie_ds41_text_ids ds41_text_ids.cpp)
target_link_libraries(ie_ds41_text_ids PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_text_ids PROPERTIES OUTPUT_NAME "ie-ds41-text-ids")

# CPU-only bridge for checkpoint reference parity (no model load).
add_executable(ie_ds41_protocol_fixture ds41_protocol_fixture.cpp)
target_link_libraries(ie_ds41_protocol_fixture PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_protocol_fixture PROPERTIES OUTPUT_NAME "ie-ds41-protocol-fixture")

# DeepSeek-V4.1-Flash Phase 46: the prefix cache's bit-identity gate (docs/deepseek41/86).
add_executable(ie_ds41_cache_test ds41_cache_test.cpp)
target_link_libraries(ie_ds41_cache_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_ds41_cache_test PROPERTIES OUTPUT_NAME "ie-ds41-cache-test")

# MiMo-V2.6 P1 (docs/mimo26/00_PORT_PLAN.md): the engine's tokenizer.json path in llama-tokenize's --ids shape
add_executable(ie_mimo26_tokenize mimo26_tokenize.cpp)
target_link_libraries(ie_mimo26_tokenize PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_mimo26_tokenize PROPERTIES OUTPUT_NAME "ie-mimo26-tokenize")
# MiMo-V2.6 P6.1 (docs/mimo26/00_PORT_PLAN.md): the vision input and tower gate
add_executable(ie_mimo26_vision_test mimo26_vision_test.cpp)
target_link_libraries(ie_mimo26_vision_test PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_mimo26_vision_test PROPERTIES OUTPUT_NAME "ie-mimo26-vision-test")
# MiMo-V2.6 P2 (docs/mimo26/00_PORT_PLAN.md): greedy generation and llama-perplexity-protocol NLL against the oracle
add_executable(ie_mimo26_run mimo26_run.cpp)
target_link_libraries(ie_mimo26_run PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_mimo26_run PROPERTIES OUTPUT_NAME "ie-mimo26-run")
add_executable(ie_mimo26_ppl mimo26_ppl.cpp)
target_link_libraries(ie_mimo26_ppl PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_mimo26_ppl PROPERTIES OUTPUT_NAME "ie-mimo26-ppl")
add_executable(ie_mimo26_score mimo26_score.cpp)
target_link_libraries(ie_mimo26_score PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_mimo26_score PROPERTIES OUTPUT_NAME "ie-mimo26-score")
# MiMo-V2.6 P3b: the C++ chat template / completion parser for tools/mimo26/check_template.py
add_executable(ie_mimo26_chat_render mimo26_chat_render.cpp)
target_link_libraries(ie_mimo26_chat_render PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_mimo26_chat_render PROPERTIES OUTPUT_NAME "ie-mimo26-chat-render")
# MiMo-V2.6 P4: an expert-residency ranking from real chat traffic
add_executable(ie_mimo26_profile mimo26_profile.cpp)
target_link_libraries(ie_mimo26_profile PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_mimo26_profile PROPERTIES OUTPUT_NAME "ie-mimo26-profile")
# MiMo-V2.6 P4: the attention kernels timed in isolation
add_executable(ie_mimo26_attn_bench mimo26_attn_bench.cpp)
target_link_libraries(ie_mimo26_attn_bench PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_mimo26_attn_bench PROPERTIES OUTPUT_NAME "ie-mimo26-attn-bench")
# MiMo-V2.6 P3b gate (b): a follow-up turn on decode-written caches vs a cold prefill
add_executable(ie_mimo26_reuse mimo26_reuse.cpp)
target_link_libraries(ie_mimo26_reuse PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_mimo26_reuse PROPERTIES OUTPUT_NAME "ie-mimo26-reuse")
# MiMo-V2.6 P5 gate (d): the DFlash drafter on dumped context features vs tools/mimo26/dflash_accept.py
add_executable(ie_mimo26_dflash_check mimo26_dflash_check.cpp)
target_link_libraries(ie_mimo26_dflash_check PRIVATE ie_core ie_sycl_flags)
set_target_properties(ie_mimo26_dflash_check PROPERTIES OUTPUT_NAME "ie-mimo26-dflash-check")
