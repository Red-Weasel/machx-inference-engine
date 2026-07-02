// include/ie/deltanet_state.hpp — Gated DeltaNet recurrent state cache.
//
// Qwen3.6-35B-A3B has 30 DeltaNet (linear-attention) layers (the layers where
// `i % 4 != 3`). Each layer carries:
//
//   recurrent state  : [n_v_heads × v_head_dim × k_head_dim] fp32
//                      = 32 × 128 × 128 × 4 B = 2 MiB / layer / batch
//                      total: 30 × 2 MiB = 60 MiB / batch  (constant in ctx)
//
//   conv1d state     : [(n_q + n_k) * k_head_dim + n_v * v_head_dim, kernel_dim - 1]
//                      = (16+16)·128 + 32·128 = 8192 channels × 3 = 24 KiB / layer
//                      total: 30 × 24 KiB ≈ 720 KiB
//
// State is stored fp32 per `mamba_ssm_dtype: float32` (Qwen quirk #7) — keeps
// long-context numerics stable.

#pragma once

#include "ie/allocator.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace ie {

struct DeltaNetStateConfig {
    uint32_t n_layers_linear;     // 30 for Qwen3.6
    uint32_t n_v_heads;           // 32
    uint32_t v_head_dim;          // 128
    uint32_t k_head_dim;          // 128
    uint32_t conv_channels;       // (2·n_k_heads + n_v_heads) · head_dim = 8192
    uint32_t conv_kernel;         // 4
};

class DeltaNetState {
public:
    DeltaNetState() = default;
    ~DeltaNetState() noexcept;
    DeltaNetState(const DeltaNetState&) = delete;
    DeltaNetState& operator=(const DeltaNetState&) = delete;
    DeltaNetState(DeltaNetState&&) noexcept;
    DeltaNetState& operator=(DeltaNetState&&) noexcept;

    std::string init(DeviceAllocator& alloc, const DeltaNetStateConfig& cfg);
    void        reset(sycl::queue& q);     // zero state + conv-state
    void        free_storage() noexcept;

    bool                          ready() const noexcept { return state_ != nullptr; }
    const DeltaNetStateConfig&    config() const noexcept { return cfg_; }
    float*                        state_ptr() const noexcept { return state_; }
    sycl::half*                   conv_state_ptr() const noexcept { return conv_state_; }

    uint64_t state_elems_per_layer() const noexcept {
        return uint64_t(cfg_.n_v_heads) * cfg_.v_head_dim * cfg_.k_head_dim;
    }
    uint64_t conv_elems_per_layer() const noexcept {
        return uint64_t(cfg_.conv_channels) * (cfg_.conv_kernel - 1);
    }

    // PR #3 — prefix cache support.  Copy entire DeltaNet state (recurrent +
    // conv) from `src` into this state.  Both must have matching config.
    // Single device-to-device memcpy per buffer (state and conv are both
    // flat fixed-size arrays — independent of context length).
    std::string copy_from(sycl::queue& q, const DeltaNetState& src);

private:
    DeviceAllocator* alloc_ = nullptr;
    float*           state_ = nullptr;       // fp32
    sycl::half*      conv_state_ = nullptr;  // fp16
    DeltaNetStateConfig cfg_{};
};

}  // namespace ie
