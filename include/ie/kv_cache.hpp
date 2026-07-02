// include/ie/kv_cache.hpp — full-attention KV cache.
//
// Qwen3.6-35B-A3B has hybrid attention: only 10 of 40 layers carry a KV cache
// (the full-attention layers). The other 30 (DeltaNet) carry a fixed-size
// recurrent state — that's a separate type, see deltanet_state.hpp (Phase 5).
//
// Layout (chosen for contiguous reads along ctx for one kv_head):
//   k[L_full, n_kv_h, max_ctx, d_head]   fp16
//   v[L_full, n_kv_h, max_ctx, d_head]   fp16
//
// Per-layer current length lives on the host. The cache is sized at init for
// `max_ctx`. For Qwen3.6 the production sizing is L_full=10, n_kv_h=2,
// d_head=256, so per-token cost is 10·2·256·2 B = 10 KiB/token (FP16) or
// 5 KiB/token (INT8 — phase-2 backlog).

#pragma once

#include "ie/allocator.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace ie {

struct KvCacheConfig {
    uint32_t n_layers_full;   // L_full = 10 for Qwen3.6
    uint32_t n_kv_heads;      // 2
    uint32_t max_ctx;         // 4096 / 32768 / 131072 / 262144
    uint32_t head_dim;        // 256
    // When true, an additional INT8 + per-row fp16-scale shadow of the cache
    // is allocated alongside the fp16 cache. The FA-2 decode path uses the
    // INT8 cache to halve KV bandwidth at long ctx; prefill / naive paths
    // keep using fp16. Memory cost: additional ~half of the fp16 cache size.
    bool     use_int8 = false;
};

class KvCache {
public:
    KvCache() = default;
    ~KvCache() noexcept;
    KvCache(const KvCache&) = delete;
    KvCache& operator=(const KvCache&) = delete;
    KvCache(KvCache&&) noexcept;
    KvCache& operator=(KvCache&&) noexcept;

    std::string init(DeviceAllocator& alloc, const KvCacheConfig& cfg);
    void        reset() noexcept;             // clear lengths (keeps memory)
    void        free_storage() noexcept;

    bool                 ready() const noexcept { return k_ != nullptr; }
    const KvCacheConfig& config() const noexcept { return cfg_; }
    bool                 is_int8() const noexcept { return cfg_.use_int8; }
    sycl::half*          k_ptr() const noexcept { return k_; }
    sycl::half*          v_ptr() const noexcept { return v_; }
    int8_t*              k_int8_ptr() const noexcept { return k_int8_; }
    int8_t*              v_int8_ptr() const noexcept { return v_int8_; }
    sycl::half*          k_scales_ptr() const noexcept { return k_scales_; }
    sycl::half*          v_scales_ptr() const noexcept { return v_scales_; }
    uint32_t             length(uint32_t layer) const noexcept { return lengths_[layer]; }
    void                 set_length(uint32_t layer, uint32_t n) noexcept { lengths_[layer] = n; }
    uint32_t             int8_length(uint32_t layer) const noexcept {
        return int8_lengths_.empty() ? 0u : int8_lengths_[layer];
    }
    void                 set_int8_length(uint32_t layer, uint32_t n) noexcept {
        if (!int8_lengths_.empty()) int8_lengths_[layer] = n;
    }
    // Post-quantizes fp16 prefill rows [start_pos, start_pos+T) into the INT8 shadow.
    // Call after full_attention() when use_int8=true. Updates int8_lengths_[layer] to
    // start_pos+T on the host before the kernel finishes — REQUIRES an in-order queue
    // so that subsequent kernels see INT8 data before the next dispatch.
    sycl::event          quantize_to_int8(sycl::queue& q, uint32_t layer,
                                          uint32_t start_pos, uint32_t T);
    uint64_t             bytes_per_layer() const noexcept {
        return uint64_t(cfg_.n_kv_heads) * cfg_.max_ctx * cfg_.head_dim * sizeof(sycl::half);
    }

    // Append T tokens of (k, v) at positions [start_pos, start_pos+T) of the
    // given full-attention layer. `k_in` / `v_in` are device pointers to
    // tensors of shape [T, n_kv_heads, head_dim] in row-major order.
    sycl::event append(sycl::queue& q, uint32_t layer,
                       uint32_t start_pos, uint32_t T,
                       const sycl::half* k_in, const sycl::half* v_in,
                       const std::vector<sycl::event>& deps = {});

    // PR #3 — prefix cache support.  Copy the first `length` tokens of every
    // full-attention layer's (K, V) slice from `src` into this cache.  The
    // strides of src and dst may differ (max_ctx mismatch is allowed) — one
    // memcpy per (layer, kv_head, K|V) slot, 4·n_layers_full·n_kv_heads
    // memcpys total.  Sets per-layer length to `length` on the host side.
    //
    // INT8 KV is not yet supported (caller must use fp16 caches on both
    // sides); when use_int8 differs between src and dst, returns an error
    // status string.  Both caches must have the same n_layers_full,
    // n_kv_heads, head_dim.
    std::string copy_prefix_from(sycl::queue& q,
                                 const KvCache& src, uint32_t length);

private:
    DeviceAllocator* alloc_ = nullptr;
    sycl::half*      k_ = nullptr;
    sycl::half*      v_ = nullptr;
    // INT8 shadow (only allocated when cfg_.use_int8). Same layout as the
    // fp16 cache: int8 [L, kv_h, max_ctx, head_dim] + scales [L, kv_h, max_ctx]
    // (one fp16 scale per (layer, kv_head, position) row of head_dim INT8).
    int8_t*          k_int8_   = nullptr;
    int8_t*          v_int8_   = nullptr;
    sycl::half*      k_scales_ = nullptr;
    sycl::half*      v_scales_ = nullptr;
    KvCacheConfig    cfg_{};
    std::vector<uint32_t> lengths_;
    std::vector<uint32_t> int8_lengths_;  // highest INT8-populated pos per layer
};

}  // namespace ie
