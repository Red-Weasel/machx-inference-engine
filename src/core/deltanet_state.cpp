// src/core/deltanet_state.cpp

#include "ie/deltanet_state.hpp"

#include <utility>

namespace ie {

DeltaNetState::~DeltaNetState() noexcept { free_storage(); }
DeltaNetState::DeltaNetState(DeltaNetState&& other) noexcept { *this = std::move(other); }
DeltaNetState& DeltaNetState::operator=(DeltaNetState&& other) noexcept {
    if (this != &other) {
        free_storage();
        alloc_       = other.alloc_;
        state_       = other.state_;
        conv_state_  = other.conv_state_;
        cfg_         = other.cfg_;
        other.alloc_      = nullptr;
        other.state_      = nullptr;
        other.conv_state_ = nullptr;
    }
    return *this;
}

void DeltaNetState::free_storage() noexcept {
    if (alloc_) {
        if (state_)      alloc_->free(state_);
        if (conv_state_) alloc_->free(conv_state_);
    }
    state_ = nullptr;
    conv_state_ = nullptr;
}

std::string DeltaNetState::init(DeviceAllocator& alloc, const DeltaNetStateConfig& cfg) {
    free_storage();
    alloc_ = &alloc;
    cfg_   = cfg;
    if (cfg.n_layers_linear == 0 || cfg.n_v_heads == 0 ||
        cfg.v_head_dim == 0 || cfg.k_head_dim == 0 ||
        cfg.conv_channels == 0 || cfg.conv_kernel == 0) {
        return "deltanet config has zero dim";
    }
    const uint64_t state_total =
        uint64_t(cfg.n_layers_linear) * state_elems_per_layer() * sizeof(float);
    const uint64_t conv_total =
        uint64_t(cfg.n_layers_linear) * conv_elems_per_layer() * sizeof(sycl::half);

    state_      = static_cast<float*>(alloc.malloc(state_total));
    conv_state_ = static_cast<sycl::half*>(alloc.malloc(conv_total));
    if (!state_ || !conv_state_) {
        free_storage();
        return "deltanet state allocation failed";
    }
    return {};
}

void DeltaNetState::reset(sycl::queue& q) {
    if (state_) {
        const uint64_t bytes =
            uint64_t(cfg_.n_layers_linear) * state_elems_per_layer() * sizeof(float);
        q.memset(state_, 0, bytes);
    }
    if (conv_state_) {
        const uint64_t bytes =
            uint64_t(cfg_.n_layers_linear) * conv_elems_per_layer() * sizeof(sycl::half);
        q.memset(conv_state_, 0, bytes);
    }
    q.wait();
}

// PR #3 — copy entire state from src.  Configs must match exactly.  Two
// flat memcpys (state and conv state are both fixed-size flat buffers,
// independent of context length).
std::string DeltaNetState::copy_from(sycl::queue& q, const DeltaNetState& src) {
    if (!ready() || !src.ready()) return "copy_from: state not ready";
    if (cfg_.n_layers_linear != src.cfg_.n_layers_linear ||
        cfg_.n_v_heads       != src.cfg_.n_v_heads       ||
        cfg_.v_head_dim      != src.cfg_.v_head_dim      ||
        cfg_.k_head_dim      != src.cfg_.k_head_dim      ||
        cfg_.conv_channels   != src.cfg_.conv_channels   ||
        cfg_.conv_kernel     != src.cfg_.conv_kernel) {
        return "copy_from: config mismatch";
    }
    const uint64_t state_bytes =
        uint64_t(cfg_.n_layers_linear) * state_elems_per_layer() * sizeof(float);
    const uint64_t conv_bytes =
        uint64_t(cfg_.n_layers_linear) * conv_elems_per_layer() * sizeof(sycl::half);
    q.memcpy(state_,      src.state_,      state_bytes);
    q.memcpy(conv_state_, src.conv_state_, conv_bytes);
    q.wait();
    return {};
}

}  // namespace ie
