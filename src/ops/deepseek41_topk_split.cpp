// src/ops/deepseek41_topk_split.cpp -- see the header. Kernel-launching helpers are prefixed ds41ts_ (the SYCL
// kernel-name collision trap). The key packing and the descending read-out are ds4_indexer_topk's, copied verbatim:
// "largest score, smallest index on a tie", pack_key(v, idx) = (ordered bits of v) << 32 | (0xFFFFFFFF - idx).
#include "ie/deepseek41_topk_split.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>
#include <cstdlib>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace ie {
namespace ds41ts {

constexpr uint32_t kChunk = 4096;          // keys per level-1 work-group
constexpr uint32_t kWG = 1024, kSG = 16;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

inline uint64_t ds41ts_pack_key(float v, uint32_t idx) {
    uint32_t u = sycl::bit_cast<uint32_t>(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (uint64_t(u) << 32) | uint64_t(0xFFFFFFFFu - idx);
}

// Bitonic sort of buf[0, N) ascending in SLM, N a power of two, WG lanes; the shape ds4_indexer_topk_bitonic uses
// (long exchanges through SLM, the last four distances in sub-group registers).
template <uint32_t N>
inline void ds41ts_sort(sycl::nd_item<1> it, sycl::local_accessor<uint64_t, 1> buf, uint32_t lid) {
    #pragma unroll
    for (uint32_t k = 2; k <= N; k <<= 1) {
        uint32_t j = k >> 1;
        #pragma unroll
        for (; j >= kSG; j >>= 1) {
            for (uint32_t i = lid; i < N; i += kWG) {
                const uint32_t ixj = i ^ j;
                if (ixj > i) {
                    const bool up = (i & k) == 0;
                    const uint64_t a = buf[i], b = buf[ixj];
                    if ((a > b) == up) { buf[i] = b; buf[ixj] = a; }
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        for (uint32_t i = lid; i < N; i += kWG) {
            uint64_t v = buf[i];
            #pragma unroll
            for (uint32_t d = kSG / 2; d; d >>= 1) {
                if (d > j) continue;
                const uint64_t other = sycl::permute_group_by_xor(it.get_sub_group(), v, d);
                const bool take_min = ((i & k) == 0) == ((i & d) == 0);
                v = take_min ? sycl::min(v, other) : sycl::max(v, other);
            }
            buf[i] = v;
        }
        it.barrier(sycl::access::fence_space::local_space);
    }
}

struct Ws { int32_t* cand = nullptr; uint64_t cap = 0; };
Ws& ds41ts_ws_for(sycl::queue& q) {
    static std::deque<std::pair<sycl::queue, Ws>> all; static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    for (auto& p : all) if (p.first == q) return p.second;
    all.emplace_back(q, Ws{}); return all.back().second;
}

}  // namespace ds41ts

bool ds41_topk_split_wanted(uint32_t T, uint32_t n_keys, uint32_t top_k) {
    static const bool on = [] { const char* e = std::getenv("IE_DS41_TOPK_SPLIT"); return !(e && *e && std::string(e) == "0"); }();
    // ... and only where the level-2 sort holds every chunk's winners: past (131,072 keys at top_k 512) the row takes
    // the single-work-group radix again (the same picks in the same order) -- the 223k run of 2026-09-16 threw here
    return on && T <= 8 && n_keys > 8192 && top_k <= 512 && uint64_t((n_keys + ds41ts::kChunk - 1) / ds41ts::kChunk) * top_k <= 16384;
}

sycl::event ds41_indexer_topk_split(sycl::queue& q, const float* scores, const int32_t* positions, int32_t* out,
                                    uint32_t T, uint32_t n_keys, uint32_t top_k, uint32_t compress_rate,
                                    const std::vector<sycl::event>& deps) {
    using namespace ds41ts;
    if (top_k > 512) throw std::invalid_argument("ds41_indexer_topk_split: top_k must be <= 512");
    const uint32_t G = (n_keys + kChunk - 1) / kChunk;          // chunks per row
    const uint32_t NC = G * top_k;                               // candidates per row
    if (NC > 16384) throw std::invalid_argument("ds41_indexer_topk_split: too many chunks for one level-2 sort (n_keys > 131,072 at top_k 512)");
    Ws& w = ds41ts_ws_for(q);
    const uint64_t need = uint64_t(T) * NC;
    if (need > w.cap) { if (w.cand) { q.wait(); sycl::free(w.cand, q); } w.cand = sycl::malloc_device<int32_t>(size_t(need), q); if (!w.cand) throw std::runtime_error("ds41_indexer_topk_split: workspace"); w.cap = need; }
    int32_t* cand = w.cand;

    // ---- level 1: (row t, chunk g) sorts its 4,096 keys; the chunk's top_k indices (global key numbers) go to cand.
    // Keys past the row's causal cutoff, or past n_keys, are -inf exactly as the single kernel packs them.
    auto e1 = ie::ps(q, "ds41_topk_split_l1", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint64_t, 1> buf(sycl::range<1>(kChunk), h);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * G * kWG, kWG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t tg = uint32_t(it.get_group(0)), t = tg / G, g = tg % G, lid = uint32_t(it.get_local_id(0));
            const int64_t thr = (int64_t(positions[t]) + 1) / int64_t(compress_rate);
            const uint32_t base = g * kChunk;
            for (uint32_t e = lid; e < kChunk; e += kWG) {
                const uint32_t idx = base + e;
                const float v = (idx < n_keys && int64_t(idx) < thr) ? scores[size_t(t) * n_keys + idx] : kNegInf;
                buf[e] = ds41ts_pack_key(v, idx);
            }
            it.barrier(sycl::access::fence_space::local_space);
            ds41ts_sort<kChunk>(it, buf, lid);
            for (uint32_t j = lid; j < top_k; j += kWG)
                cand[(size_t(t) * G + g) * top_k + j] = int32_t(0xFFFFFFFFu - uint32_t(buf[kChunk - 1 - j] & 0xFFFFFFFFu));
        });
    });

    // ---- level 2: one work-group per row sorts the G * top_k candidates (re-packed from their scores, the same
    // packing) and emits the row as ds4_indexer_topk does. The padded extent is the next power of two of NC.
    auto level2 = [&]<uint32_t N2>() {
        return ie::ps(q, "ds41_topk_split_l2", [&](sycl::handler& h) {
            h.depends_on(e1);
            sycl::local_accessor<uint64_t, 1> buf(sycl::range<1>(N2), h);
            h.parallel_for(sycl::nd_range<1>(size_t(T) * kWG, kWG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
                const uint32_t t = uint32_t(it.get_group(0)), lid = uint32_t(it.get_local_id(0));
                const int64_t thr = (int64_t(positions[t]) + 1) / int64_t(compress_rate);
                for (uint32_t e = lid; e < N2; e += kWG) {
                    uint64_t key = 0;                                        // pad: the smallest key, never read out at top_k <= NC
                    if (e < NC) {
                        const uint32_t idx = uint32_t(cand[size_t(t) * NC + e]);
                        const float v = (idx < n_keys && int64_t(idx) < thr) ? scores[size_t(t) * n_keys + idx] : kNegInf;
                        key = ds41ts_pack_key(v, idx);
                    }
                    buf[e] = key;
                }
                it.barrier(sycl::access::fence_space::local_space);
                ds41ts_sort<N2>(it, buf, lid);
                for (uint32_t j = lid; j < top_k; j += kWG) {
                    const int32_t idx = int32_t(0xFFFFFFFFu - uint32_t(buf[N2 - 1 - j] & 0xFFFFFFFFu));
                    out[size_t(t) * top_k + j] = (int64_t(idx) >= thr) ? -1 : idx;
                }
            });
        });
    };
    if (NC <= 1024)  return level2.template operator()<1024>();
    if (NC <= 2048)  return level2.template operator()<2048>();
    if (NC <= 4096)  return level2.template operator()<4096>();
    if (NC <= 8192)  return level2.template operator()<8192>();
    return level2.template operator()<16384>();
}

}  // namespace ie
