// include/ie/hf_import.hpp — HuggingFace checkpoint import (P3e).
//
// Converts a HF model directory (config.json + safetensors, optionally AWQ/GPTQ
// quantized) into the engine's native GGUF so the EXISTING loader/forward runs
// it unchanged — a path llama.cpp doesn't offer for AWQ/GPTQ int4 weights.
//
// This header covers the metadata layer (config.json → engine config + the GGUF
// KV values to emit) and the HF↔GGUF tensor-name mapping. The dequant math is
// in awq.hpp; the GGUF emitter is in gguf_writer.hpp.
#pragma once

#include "ie/model_config.hpp"
#include "ie/awq.hpp"

#include <cstdint>
#include <string>

namespace ie {

// Parsed HF config.json for a dense transformer.
struct HfModelMeta {
    DenseConfig cfg;                      // hidden/heads/ffn/vocab/rope/eps/n_layers
    std::string arch;                     // model_type, e.g. "qwen3"
    bool        tie_word_embeddings = false;
    bool        attention_bias      = false;   // qwen2 has it, qwen3 does not
    std::string torch_dtype;              // "float16" / "bfloat16"
    bool        quantized           = false;   // has a quantization_config
    AwqConfig   quant;                    // valid when quantized
};

// Parse a HF config.json body. Fills dense fields + arch + tie flag, and (if a
// quantization_config is present) sets quantized=true and parses `quant`.
// Returns "" on success, error text on a missing REQUIRED field.
std::string parse_hf_config(const std::string& config_json, HfModelMeta& out);

// Map a HF safetensors tensor name to the engine's GGUF name, e.g.
//   model.layers.0.self_attn.q_proj.<suffix>  -> blk.0.attn_q.<suffix>
//   model.layers.0.mlp.gate_proj.<suffix>     -> blk.0.ffn_gate.<suffix>
//   model.embed_tokens.weight                 -> token_embd.weight
//   model.norm.weight                         -> output_norm.weight
//   lm_head.weight                            -> output.weight
// `<suffix>` is preserved (e.g. ".weight", ".qweight"). Returns "" (empty) if
// the name is not recognized (caller decides whether that's fatal).
std::string hf_to_gguf_tensor_name(const std::string& hf_name);

// Convert a HF AWQ checkpoint directory (config.json + model.safetensors) into a
// GGUF the engine loads natively. Projections + embeddings are AWQ-dequantized
// and re-encoded to Q6_K; norms → F32. Tokenizer KVs are copied from
// `tokenizer_ref_gguf` (a same-family GGUF whose vocab matches — e.g. any Qwen3
// dense GGUF for a Qwen3 AWQ model). Optional `log` collects progress lines.
// Returns "" on success, error text otherwise. (v1: model_type "qwen3", single
// safetensors file, AWQ gemm; sharding/other arches are follow-ups.)
std::string import_awq_to_gguf(const std::string& hf_dir,
                               const std::string& out_gguf,
                               const std::string& tokenizer_ref_gguf,
                               std::string* log = nullptr);

// Convert an EXL3 (QTIP-based) HF checkpoint into a native engine GGUF. The EXL3
// quantized weights are carried VERBATIM (packed trellis tagged DType::kEXL3 + the
// F16 suh/svh side-vectors) and decoded natively on-GPU at inference (gemv_exl3) —
// nothing is dequantized/re-quantized. token_embd stays F16; norms → F32; the
// llama3 rope_freqs.weight is computed from config. Tokenizer KVs copied from
// `tokenizer_ref_gguf` (a same-vocab GGUF, e.g. any Llama-3.x dense GGUF). v1:
// model_type "llama", single safetensors, codebook cb=0. Returns "" on success.
std::string import_exl3_to_gguf(const std::string& hf_dir,
                                const std::string& out_gguf,
                                const std::string& tokenizer_ref_gguf,
                                std::string* log = nullptr);

// Convert an EXL3 Qwen3-Next (MoE + DeltaNet, model_type "qwen3_next") checkpoint
// into a native engine GGUF the `qwen3next` loader consumes. Like import_exl3_to_gguf,
// EXL3 weights ride VERBATIM (kEXL3 trellis + F16 suh/svh) and decode on-GPU. Unlike
// the dense path: (1) the 512 per-layer experts are FUSED into per-(layer,proj) banks
// `blk.L.ffn_{gate,up,down}_exps.{weight,suh,svh}` — trellis is 4D ne=[16*bits,N/16,K/16,E]
// (experts outermost = contiguous slab, stride nbytes/E), suh ne=[K,E], svh ne=[N,E];
// (2) DeltaNet `in_proj_qkvz` stays a SINGLE fused EXL3 `attn_qkv` (N=12288, sliced in
// the forward — a trellis cannot be split); (3) raw DeltaNet tensors (A_log, dt_bias,
// conv1d, in_proj_ba, norms) and the router/shared-gate ride F32/F16. EXL3 bit-width is
// VARIABLE per layer (uniform across a layer's 512 experts) → bits read per tensor. The
// write is STREAMED (one tensor/bank in RAM at a time) — the 80B is 43 GB. Tokenizer KVs
// copied from `tokenizer_ref_gguf` (a same-vocab qwen3next GGUF). Returns "" on success.
std::string import_exl3_qwen3next_to_gguf(const std::string& hf_dir,
                                          const std::string& out_gguf,
                                          const std::string& tokenizer_ref_gguf,
                                          std::string* log = nullptr);

}  // namespace ie
