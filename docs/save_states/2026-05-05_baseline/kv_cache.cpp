// src/core/kv_cache.cpp

#include "ie/kv_cache.hpp"

#include <cstring>
#include "ie/kernel_profiler.hpp"

namespace ie {

KvCache::~KvCache() noexcept { free_storage(); }
KvCache::KvCache(KvCache&& other) noexcept { *this = std::move(other); }
KvCache& KvCache::operator=(KvCache&& other) noexcept {
    if (this != &other) {
        free_storage();
        alloc_     = other.alloc_;
        k_         = other.k_;
        v_         = other.v_;
        k_int8_    = other.k_int8_;
        v_int8_    = other.v_int8_;
        k_scales_  = other.k_scales_;
        v_scales_  = other.v_scales_;
        cfg_       = other.cfg_;
        lengths_      = std::move(other.lengths_);
        int8_lengths_ = std::move(other.int8_lengths_);
        other.alloc_     = nullptr;
        other.k_         = nullptr;
        other.v_         = nullptr;
        other.k_int8_    = nullptr;
        other.v_int8_    = nullptr;
        other.k_scales_  = nullptr;
        other.v_scales_  = nullptr;
    }
    return *this;
}

void KvCache::free_storage() noexcept {
    if (alloc_) {
        if (k_)        alloc_->free(k_);
        if (v_)        alloc_->free(v_);
        if (k_int8_)   alloc_->free(k_int8_);
        if (v_int8_)   alloc_->free(v_int8_);
        if (k_scales_) alloc_->free(k_scales_);
        if (v_scales_) alloc_->free(v_scales_);
    }
    k_         = nullptr;
    v_         = nullptr;
    k_int8_    = nullptr;
    v_int8_    = nullptr;
    k_scales_  = nullptr;
    v_scales_  = nullptr;
    lengths_.clear();
    int8_lengths_.clear();
}

void KvCache::reset() noexcept {
    for (auto& n : lengths_) n = 0;
    for (auto& n : int8_lengths_) n = 0u;
}

std::string KvCache::init(DeviceAllocator& alloc, const KvCacheConfig& cfg) {
    free_storage();
    alloc_ = &alloc;
    cfg_   = cfg;
    if (cfg.n_layers_full == 0 || cfg.n_kv_heads == 0 ||
        cfg.max_ctx == 0 || cfg.head_dim == 0) {
        return "kv-cache config has zero dim";
    }
    const uint64_t per_layer = bytes_per_layer();
    const uint64_t total     = per_layer * cfg.n_layers_full;
    k_ = static_cast<sycl::half*>(alloc.malloc(total));
    v_ = static_cast<sycl::half*>(alloc.malloc(total));
    if (!k_ || !v_) {
        free_storage();
        return "kv-cache allocation failed";
    }
    if (cfg.use_int8) {
        // Per-row symmetric INT8: head_dim INT8 elements + 1 fp16 scale per
        // (layer, kv_head, position) row.
        const uint64_t int8_total =
            uint64_t(cfg.n_layers_full) * cfg.n_kv_heads * cfg.max_ctx * cfg.head_dim;
        const uint64_t scale_total =
            uint64_t(cfg.n_layers_full) * cfg.n_kv_heads * cfg.max_ctx;
        k_int8_   = static_cast<int8_t*>(alloc.malloc(int8_total));
        v_int8_   = static_cast<int8_t*>(alloc.malloc(int8_total));
        k_scales_ = static_cast<sycl::half*>(alloc.malloc(scale_total * sizeof(sycl::half)));
        v_scales_ = static_cast<sycl::half*>(alloc.malloc(scale_total * sizeof(sycl::half)));
        if (!k_int8_ || !v_int8_ || !k_scales_ || !v_scales_) {
            free_storage();
            return "kv-cache INT8 shadow allocation failed";
        }
    }
    lengths_.assign(cfg.n_layers_full, 0);
    if (cfg.use_int8) int8_lengths_.assign(cfg.n_layers_full, 0u);
    return {};
}

// Layout: k[L, kv_h, ctx, d_head] row-major.
//   stride per layer  = n_kv_h * max_ctx * d_head
//   stride per kv_h   = max_ctx * d_head
//   stride per ctx    = d_head
//
// We append T contiguous positions of one layer. For each kv_h slice, the
// destination is contiguous along the inner two dims (ctx, d_head). The
// source `k_in` is shaped [T, n_kv_h, d_head], which interleaves kv_heads
// along the leading axis — so we issue one memcpy per kv_h to handle the
// strided destination.
sycl::event KvCache::append(sycl::queue& q, uint32_t layer,
                            uint32_t start_pos, uint32_t T,
                            const sycl::half* k_in, const sycl::half* v_in,
                            const std::vector<sycl::event>& deps) {
    return ie::ps(q, "kv_append", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint64_t d_head    = cfg_.head_dim;
        const uint64_t n_kv_h    = cfg_.n_kv_heads;
        const uint64_t max_ctx   = cfg_.max_ctx;
        const uint64_t per_layer = n_kv_h * max_ctx * d_head;
        sycl::half*    K = k_;
        sycl::half*    V = v_;
        h.parallel_for(sycl::nd_range<3>({T, n_kv_h, d_head}, {1, 1, std::min<size_t>(d_head, 256)}),
                       [=](sycl::nd_item<3> it) {
            const uint64_t t  = it.get_global_id(0);
            const uint64_t kv = it.get_global_id(1);
            const uint64_t d  = it.get_global_id(2);
            const uint64_t src = (t * n_kv_h + kv) * d_head + d;
            const uint64_t dst = uint64_t(layer) * per_layer
                                 + kv * max_ctx * d_head
                                 + (uint64_t(start_pos) + t) * d_head
                                 + d;
            K[dst] = k_in[src];
            V[dst] = v_in[src];
        });
    });
}

// Post-quantize fp16 prefill rows [start_pos, start_pos+T) into INT8 shadow.
// One work-group per (kv_head, row). All head_dim work-items in the group
// cooperate via reduce_over_group to find max-abs, then each writes one INT8.
// Layout: k[L, kv_h, max_ctx, d_head] — same for both fp16 and INT8.
sycl::event KvCache::quantize_to_int8(sycl::queue& q, uint32_t layer,
                                       uint32_t start_pos, uint32_t T) {
    if (!k_int8_ || !v_int8_) return sycl::event{};

    const uint64_t d_head    = cfg_.head_dim;
    const uint64_t n_kv_h    = cfg_.n_kv_heads;
    const uint64_t max_ctx   = cfg_.max_ctx;
    const uint64_t per_layer = n_kv_h * max_ctx * d_head;
    const uint64_t per_scale = n_kv_h * max_ctx;

    const sycl::half* K  = k_       + per_layer * layer;
    const sycl::half* V  = v_       + per_layer * layer;
    int8_t*           Ki = k_int8_  + per_layer * layer;
    int8_t*           Vi = v_int8_  + per_layer * layer;
    sycl::half*       Ks = k_scales_ + per_scale * layer;
    sycl::half*       Vs = v_scales_ + per_scale * layer;

    // Local size = d_head (all head_dim elements in one WG per row).
    // Valid for Qwen3.6 head_dim=256; over-large d_head would need splitting.
    const size_t lsz = std::min<size_t>(d_head, 256u);

    auto ev = ie::ps(q, "kv_quant_int8", [&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<3>({T, n_kv_h, d_head}, {1, 1, lsz}),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const uint64_t t   = it.get_global_id(0);
                const uint64_t kv  = it.get_global_id(1);
                const uint64_t d   = it.get_global_id(2);
                const uint64_t row = kv * max_ctx + (start_pos + t);
                const uint64_t idx = row * d_head + d;

                const float kv_k = float(K[idx]);
                const float kv_v = float(V[idx]);

                const float k_max = sycl::reduce_over_group(
                    it.get_group(), sycl::fabs(kv_k), sycl::maximum<float>{});
                sycl::group_barrier(it.get_group());
                const float v_max = sycl::reduce_over_group(
                    it.get_group(), sycl::fabs(kv_v), sycl::maximum<float>{});

                const float k_sc = (k_max > 0.0f) ? (k_max / 127.0f) : 1.0f;
                const float v_sc = (v_max > 0.0f) ? (v_max / 127.0f) : 1.0f;

                if (it.get_local_id(2) == 0) {
                    Ks[row] = sycl::half(k_sc);
                    Vs[row] = sycl::half(v_sc);
                }

                auto clamp127 = [](float x) -> int8_t {
                    float r = sycl::round(x);
                    return int8_t(r < -127.0f ? -127.0f : (r > 127.0f ? 127.0f : r));
                };
                Ki[idx] = clamp127(kv_k / k_sc);
                Vi[idx] = clamp127(kv_v / v_sc);
            });
    });
    int8_lengths_[layer] = start_pos + T;
    return ev;
}

}  // namespace ie
