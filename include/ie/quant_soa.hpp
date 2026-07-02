// include/ie/quant_soa.hpp — per-expert SoA ("reorder") layout for MoE
// expert tensors, plus layout-generic weight views for the MoE kernels.
//
// Motivation (docs/prefill_crown_plan.md): llama.cpp SYCL's load-time
// reorder_qw_q4_k splits each tensor into struct-of-arrays streams
//   [ qs × nblocks | scales × nblocks | dm × nblocks ]
// so the 16 B block header no longer interrupts every 144 B of quant
// stream.  We apply the same split PER-EXPERT REGION so the existing
// `expert_base = W + e * expert_stride_bytes` arithmetic is unchanged —
// only the intra-region layout differs.  Same bits moved → PPL-free by
// construction.
//
// Q4_K region (nb blocks):  qs[128]×nb | scales[12]×nb | {d,dmin}[4]×nb
// Q6_K region (nb blocks):  ql[128]×nb | qh[64]×nb | scales[16]×nb | d[2]×nb
//
// Alignment: per-expert regions keep their AoS stride (nb×144 / nb×210
// bytes).  For the production shapes (nb = 4096) every SoA stream start
// is 16 B-aligned, which upgrades several formerly 2 B-aligned Q6_K loads
// to clean vector loads.

#pragma once

#include "ie/quant_blocks.hpp"

#include <cstdint>
#include <cstring>

namespace ie {

// ---------------------------------------------------------------------------
// SoA stream offsets within one expert region of `nb` blocks.
// ---------------------------------------------------------------------------
struct q4k_soa_offsets {
    uint64_t qs = 0;        // 128 B / block
    uint64_t scales;        // 12 B / block
    uint64_t dm;            // 4 B / block ({d, dmin} as raw fp16 bit pairs)
    explicit constexpr q4k_soa_offsets(uint64_t nb)
        : scales(nb * 128u), dm(nb * 140u) {}
};

struct q6k_soa_offsets {
    uint64_t ql = 0;        // 128 B / block
    uint64_t qh;            // 64 B / block
    uint64_t scales;        // 16 B / block (int8)
    uint64_t d;             // 2 B / block (raw fp16 bits)
    explicit constexpr q6k_soa_offsets(uint64_t nb)
        : qh(nb * 128u), scales(nb * 192u), d(nb * 208u) {}
};

// ---------------------------------------------------------------------------
// Layout-generic device views.  SOA=false reproduces today's AoS addresses
// exactly (the streams are the block fields, stride = sizeof(block)); the
// kernels' arithmetic is bit-identical in either instantiation — only load
// addresses change.
// ---------------------------------------------------------------------------
template <bool SOA>
struct q4k_wview {
    const uint8_t* qs;      // per-block stride QS_STRIDE
    const uint8_t* sc;      // per-block stride SC_STRIDE (12 B of packed scales)
    const uint8_t* dm;      // per-block stride DM_STRIDE (d at +0, dmin at +2)

    static constexpr uint32_t QS_STRIDE = SOA ? 128u : uint32_t(sizeof(block_q4_K));
    static constexpr uint32_t SC_STRIDE = SOA ? 12u  : uint32_t(sizeof(block_q4_K));
    static constexpr uint32_t DM_STRIDE = SOA ? 4u   : uint32_t(sizeof(block_q4_K));

    static inline q4k_wview at(const void* W, uint64_t expert_stride_bytes,
                               uint32_t e, uint64_t nblocks_per_expert) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(W) +
                           uint64_t(e) * expert_stride_bytes;
        if constexpr (SOA) {
            const q4k_soa_offsets off(nblocks_per_expert);
            return { p + off.qs, p + off.scales, p + off.dm };
        } else {
            (void)nblocks_per_expert;
            return { p + 16, p + 4, p };   // block_q4_K{d,dmin,scales[12],qs[128]}
        }
    }
    inline const uint8_t* qs_blk(uint64_t bi) const { return qs + bi * QS_STRIDE; }
    inline const uint8_t* sc_blk(uint64_t bi) const { return sc + bi * SC_STRIDE; }
    inline uint16_t d_bits   (uint64_t bi) const {
        return *reinterpret_cast<const uint16_t*>(dm + bi * DM_STRIDE);
    }
    inline uint16_t dmin_bits(uint64_t bi) const {
        return *reinterpret_cast<const uint16_t*>(dm + bi * DM_STRIDE + 2);
    }
};

template <bool SOA>
struct q6k_wview {
    const uint8_t* ql;      // per-block stride QL_STRIDE
    const uint8_t* qh;      // per-block stride QH_STRIDE
    const uint8_t* sc;      // per-block stride SC_STRIDE (int8[16])
    const uint8_t* d;       // per-block stride D_STRIDE (raw fp16 bits)

    static constexpr uint32_t QL_STRIDE = SOA ? 128u : uint32_t(sizeof(block_q6_K));
    static constexpr uint32_t QH_STRIDE = SOA ? 64u  : uint32_t(sizeof(block_q6_K));
    static constexpr uint32_t SC_STRIDE = SOA ? 16u  : uint32_t(sizeof(block_q6_K));
    static constexpr uint32_t D_STRIDE  = SOA ? 2u   : uint32_t(sizeof(block_q6_K));

    static inline q6k_wview at(const void* W, uint64_t expert_stride_bytes,
                               uint32_t e, uint64_t nblocks_per_expert) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(W) +
                           uint64_t(e) * expert_stride_bytes;
        if constexpr (SOA) {
            const q6k_soa_offsets off(nblocks_per_expert);
            return { p + off.ql, p + off.qh, p + off.scales, p + off.d };
        } else {
            (void)nblocks_per_expert;
            return { p, p + 128, p + 192, p + 208 };  // block_q6_K field offsets
        }
    }
    inline const uint8_t* ql_blk(uint64_t bi) const { return ql + bi * QL_STRIDE; }
    inline const uint8_t* qh_blk(uint64_t bi) const { return qh + bi * QH_STRIDE; }
    inline const int8_t*  sc_blk(uint64_t bi) const {
        return reinterpret_cast<const int8_t*>(sc + bi * SC_STRIDE);
    }
    inline uint16_t d_bits(uint64_t bi) const {
        return *reinterpret_cast<const uint16_t*>(d + bi * D_STRIDE);
    }
};

// ---------------------------------------------------------------------------
// Host-side load-time repack: AoS expert-major tensor → per-expert SoA.
// src and dst are full-tensor buffers (n_experts × nb blocks); dst must not
// alias src.  Pure byte moves — no value ever changes representation.
// ---------------------------------------------------------------------------
inline void repack_moe_q4k_soa_host(const uint8_t* src, uint8_t* dst,
                                    uint32_t n_experts, uint64_t nblocks_per_expert) {
    const uint64_t stride = nblocks_per_expert * sizeof(block_q4_K);
    const q4k_soa_offsets off(nblocks_per_expert);
    for (uint32_t e = 0; e < n_experts; ++e) {
        const auto* blocks = reinterpret_cast<const block_q4_K*>(src + uint64_t(e) * stride);
        uint8_t* base = dst + uint64_t(e) * stride;
        for (uint64_t b = 0; b < nblocks_per_expert; ++b) {
            std::memcpy(base + off.qs     + b * 128u, blocks[b].qs,     128u);
            std::memcpy(base + off.scales + b * 12u,  blocks[b].scales, 12u);
            std::memcpy(base + off.dm     + b * 4u,   &blocks[b].d,     2u);
            std::memcpy(base + off.dm     + b * 4u + 2u, &blocks[b].dmin, 2u);
        }
    }
}

inline void repack_moe_q6k_soa_host(const uint8_t* src, uint8_t* dst,
                                    uint32_t n_experts, uint64_t nblocks_per_expert) {
    const uint64_t stride = nblocks_per_expert * sizeof(block_q6_K);
    const q6k_soa_offsets off(nblocks_per_expert);
    for (uint32_t e = 0; e < n_experts; ++e) {
        const auto* blocks = reinterpret_cast<const block_q6_K*>(src + uint64_t(e) * stride);
        uint8_t* base = dst + uint64_t(e) * stride;
        for (uint64_t b = 0; b < nblocks_per_expert; ++b) {
            std::memcpy(base + off.ql     + b * 128u, blocks[b].ql,     128u);
            std::memcpy(base + off.qh     + b * 64u,  blocks[b].qh,     64u);
            std::memcpy(base + off.scales + b * 16u,  blocks[b].scales, 16u);
            std::memcpy(base + off.d      + b * 2u,   &blocks[b].d,     2u);
        }
    }
}

}  // namespace ie
