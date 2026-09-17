// include/ie/gemma4_assistant.hpp — Gemma 4 MTP draft head (`gemma4-assistant`).
//
// A 930M, 4-layer EAGLE-style self-draft for the gemma-4-31B target. It embeds a
// token with the TARGET's tok_embd, concatenates the target's hidden state, and
// predicts the next token + a recurrence hidden (h_next) for autoregressive
// drafting. Its Q-only attention SHARES the target's last-2 layers' KV cache
// (SWA head layers → target L(n-2), global → target L(n-1)). Loaded on demand
// only when MTP spec-decode is requested (~0.5 GB Q8_0).
//
// Authoritative forward: ~/llama.cpp/src/models/gemma4-assistant.cpp.
#pragma once

#include "ie/dtype.hpp"
#include "ie/dense_transformer.hpp"   // DenseQuantPtr, dense::upload helpers
#include "ie/gguf.hpp"
#include "ie/allocator.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>

namespace ie {

struct GemmaAssistantConfig {
    uint32_t n_layers = 0;          // 4
    uint32_t hidden   = 0;          // 1024 (head backbone)
    uint32_t n_embd_backbone = 0;   // 5376 (target hidden; from proj shapes)
    uint32_t ffn      = 0;          // 8192
    uint32_t vocab    = 0;          // 262144
    uint32_t n_q_heads = 0;         // 32 (constant)
    float    rms_eps  = 1e-6f;
    uint32_t sliding_window = 1024;
    // per-layer geometry (length n_layers)
    std::vector<uint8_t>  is_swa;       // 1 = sliding-window layer
    std::vector<uint32_t> n_kv_heads;   // [16,16,16,4]
    std::vector<uint32_t> head_dim;     // 256 (swa) / 512 (global)
    std::vector<uint32_t> n_rot;        // 256 / 512
    std::vector<float>    rope_theta;   // 1e4 (swa) / 1e6 (global)
};

// Parse gemma4-assistant.* keys. n_embd_backbone is derived from the nextn
// projection tensor shapes. Returns "" on success or error text.
std::string read_gemma4_assistant_config(const GgufReader& g, GemmaAssistantConfig& out);

class Gemma4AssistantHead {
public:
    Gemma4AssistantHead() = default;
    ~Gemma4AssistantHead();
    Gemma4AssistantHead(const Gemma4AssistantHead&) = delete;
    Gemma4AssistantHead& operator=(const Gemma4AssistantHead&) = delete;

    // Load all head weights to device. Q8_0 matmuls + F32 norms.
    std::string load(DeviceAllocator& alloc, const GgufReader& g,
                     const GemmaAssistantConfig& cfg);
    const GemmaAssistantConfig& config() const noexcept { return cfg_; }
    bool loaded() const noexcept { return loaded_; }

private:
    struct Layer {
        bool     is_swa = true;
        uint32_t head_dim = 0, n_kv = 0, n_rot = 0;
        float    rope_theta = 1e4f;
        float* attn_norm = nullptr;        // F32 [hidden]
        float* attn_q_norm = nullptr;      // F32 [head_dim]
        float* post_attn_norm = nullptr;   // F32 [hidden]
        float* ffn_norm = nullptr;         // F32 [hidden]
        float* post_ffw_norm = nullptr;    // F32 [hidden]
        float* out_scale = nullptr;        // F32 [1]
        float* rope_freqs = nullptr;       // F32 [head_dim/2], global layers only
        DenseQuantPtr wq, wo;              // Q8_0
        DenseQuantPtr ffn_gate, ffn_up, ffn_down;  // Q8_0
    };

    GemmaAssistantConfig cfg_;
    bool loaded_ = false;
    DeviceAllocator* alloc_ = nullptr;
    std::vector<void*> owned_;

    void*  tok_embd_ = nullptr;        // Q8_0 [hidden, vocab] — head's output/logits
    DType  tok_embd_dt_ = DType::kCount;
    float* output_norm_ = nullptr;     // F32 [hidden]
    DenseQuantPtr pre_proj_;           // Q8_0 [2*backbone, hidden]
    DenseQuantPtr post_proj_;          // Q8_0 [hidden, backbone]
    std::vector<Layer> layers_;
};

}  // namespace ie
