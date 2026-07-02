// include/ie/dense_split.hpp — P-B: multi-GPU LAYER-SPLIT for the dense path.
//
// ADDITIVE: this never touches DenseModel (the validated single-GPU path stays
// byte-identical). It re-uses the same upload helpers (dense_dispatch.hpp) and
// leaf ops (ops.hpp), placing each transformer layer's weights on its assigned
// device and running the forward device-by-device with one residual copy at each
// boundary. Validated by a single-GPU-vs-2-GPU bit-identical logits equality test.
// Design: docs/superpowers/specs/2026-06-12-multi-gpu-layer-split-design.md.
//
// SCOPE (v1): the prefill path only (T>1), enough for the equality gate and for
// a 72B PPL/greedy bring-up. The int-dot decode fast-paths are single-GPU-only;
// adding them to the split path is a later optimization.
#pragma once

#include "ie/allocator.hpp"          // DeviceFleet
#include "ie/dense_transformer.hpp"  // DenseLayerWeights, DenseQuantPtr
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

// Maps each transformer layer + the top-level tensors to a device index.
struct LayerPlan {
    std::vector<uint32_t> dev_of_layer;   // size n_layers; dev for layer L
    uint32_t embed_dev = 0;               // token_embd + embedding lookup
    uint32_t head_dev  = 0;               // output_norm + lm_head (logits)

    // Contiguous even split: layer L → floor(L * n_dev / n_layers). embed on
    // dev 0, head on dev n_dev-1. Balanced for a uniform dense model.
    static LayerPlan contiguous(uint32_t n_layers, uint32_t n_dev);
    uint32_t n_dev() const;               // 1 + max device index used
};

class DenseModelSplit {
public:
    DenseModelSplit() = default;
    ~DenseModelSplit();
    DenseModelSplit(const DenseModelSplit&) = delete;
    DenseModelSplit& operator=(const DenseModelSplit&) = delete;

    // Load all weights, each layer onto plan.dev_of_layer[L]'s device; embed →
    // plan.embed_dev, lm_head → plan.head_dev; rope_freqs (if present) onto every
    // device that owns ≥1 layer. `fleet` and `g` must outlive this object.
    std::string load(DeviceFleet& fleet, const LayerPlan& plan,
                     const GgufReader& g, const DenseConfig& cfg);

    const DenseConfig& config() const noexcept { return cfg_; }

    // Forward of `T` tokens at `start_pos`, writing the last token's logits to
    // `out_logits_host` (host fp16 buffer, size vocab). Handles both the prompt
    // prefill (T>1, start_pos=0, reset_kv=true) and incremental decode (T=1,
    // start_pos=len, reset_kv=false → appends to the per-device KV). KV is
    // internal, per device. Uses the fp16 GEMV decode path (no int-dot q8), so
    // a split decode is bit-identical to a single-GPU IE_NO_Q8_DECODE=1 decode.
    std::string forward(const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        bool reset_kv, sycl::half* out_logits_host);

    // Per-device VRAM the weights occupy (bytes), for reporting.
    std::vector<uint64_t> device_bytes() const { return dev_bytes_; }

private:
    // Per-device prefill scratch (defined here so std::vector<Workspace> is a
    // complete type for callers that construct/destroy a DenseModelSplit).
    struct Workspace {
        uint32_t T = 0;
        sycl::half *x = nullptr, *x_normed = nullptr, *q = nullptr, *k = nullptr,
                   *v = nullptr, *attn_out = nullptr, *attn_block = nullptr,
                   *gate = nullptr, *up = nullptr, *h = nullptr;
        int32_t* positions = nullptr;
    };

    DeviceFleet* fleet_ = nullptr;
    LayerPlan    plan_;
    DenseConfig  cfg_;
    uint32_t     n_dev_ = 0;

    // Per-LAYER weights (global index L); the pointers live on dev_of_layer[L].
    std::vector<DenseLayerWeights> layers_;
    // Top-level.
    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;
    float* output_norm_ = nullptr;
    void*  output_ = nullptr;      DType output_dtype_ = DType::kCount;
    std::vector<float*> rope_freqs_per_dev_;   // [n_dev] (null if no scaling)

    // Per-device resources.
    std::vector<std::vector<void*>> owned_;    // [dev] → device ptrs to free
    std::vector<Workspace>          ws_;       // [dev]
    std::vector<KvCache>            kv_;        // [dev], sized to the dev's layer count
    std::vector<uint32_t>           local_idx_;// [global L] → local layer index on its dev
    std::vector<uint64_t>           dev_bytes_;// [dev] weight bytes (reporting)

    std::string ensure_ws(uint32_t dev, uint32_t max_T);
    void free_all();
};

}  // namespace ie
