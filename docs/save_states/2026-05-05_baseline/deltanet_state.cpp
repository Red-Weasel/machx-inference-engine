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

}  // namespace ie
