// src/model/qwen35moe_split.cpp — Qwen3.6-35B-A3B crown (`kQwen35Moe`) multi-GPU
// layer-split for all-Q8_0 GGUFs. See header. ADDITIVE: mirrors the 27B
// Qwen35SplitModel orchestration (qwen35_split.cpp) with the crown's per-layer math
// (qwen36.cpp) lifted into a device-by-device loop; the FFN is the crown MoE run as
// a Phase-1 per-active-expert Q8_0 int-dot loop. The single-GPU QwenModel is NEVER
// edited (PPL 6.4527 gate safe).
//
// PHASE-1 SCOPE: correctness-first, reuse-only. The MoE decode/prefill is a per-expert
// loop over the existing single-matrix gemv_q8_0_soa_q8 / dequant+gemm — no grouped
// Q8_0-expert kernel yet (Phase 2). Decode is host/comm-bound on a 2-card layer-split,
// so the launch-heavy loop is tolerable for validation.

#include "ie/qwen35moe_split.hpp"
#include "ie/decode_prof.hpp"      // P4 B22: IE_DECODE_PROF per-token decode breakdown
#include "ie/q35m_lanes.hpp"

#include "ie/dequant.hpp"
#include "ie/dequant_ref.hpp"   // ie::ref::dequant_q4_K/q5_K/q6_K_buffer (K-quant experts → Q8_0)
#include "ie/quant_soa.hpp"     // repack_moe_q4k/q6k_soa_host (native K-quant expert reorder)
#include "ie/gemv_q8_soa_v2.hpp"   // P4 B37: the v2 int-dot GEMV selectors
#include "ie/moe_q8.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include "dense_dispatch.hpp"   // dense::upload<T>, dense::upload_quant_dense_auto, dense::gemv_q_T
#include "q35_xmx_decode.hpp"   // P4 B23: IE_Q35_XMX_DECODE XMX decode attention at long context (default on, =0 off)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ie {
namespace {

// P4 B43: the prefill-attention threshold -- a full-attention prefill piece of T rows at start_pos takes the wide-tile FA2 kernel
// iff start_pos + T >= this, the naive kernel below. IE_Q35MOE_FA2_TILE_MINCTX (> 0); default 512 (B29-v0.2.6: 6144, the 27B's gate,
// never measured on the crown; `=6144` restores v0.2.6's bytes). Read once; printed by load().
uint32_t q35m_fa2_tile_minctx() {
    static const uint32_t v = []() -> uint32_t {
        const char* e = std::getenv("IE_Q35MOE_FA2_TILE_MINCTX");
        if (!e) return 512u;
        const int n = std::atoi(e);
        return n > 0 ? uint32_t(n) : 512u;
    }();
    return v;
}

DecodeProf g_dp35("35B qwen35moe_split");   // P4 B22 (off unless IE_DECODE_PROF=1)
#define DPE(DEV, C, ...) do { sycl::event dpe_ = (__VA_ARGS__); if (g_dp35.active()) g_dp35.tag((DEV), dpe_, DecodeProf::C); } while (0)

// P4 B22: the shared-expert gate g[t] = sigmoid(sum_h gw[h] * x[t][h]) in the SAME serial fp32 order as the old
// one-work-item loop (`gg += gw[h] * float(x[h])`, h = 0..H-1), so the result is bit-identical; only the loads moved. The
// old kernel ran ONE work-item per token with a dependent global load per step: ~110 us per layer (B22 PROFILE: 4.4 ms of
// a 16.5 ms 35B token). Here one sub-group per token loads 128 elements at a time, coalesced (the next 128 are in flight
// while this 128 is summed), and every lane walks the same serial chain on shuffled operands; lane 0 writes.
// IE_Q35MOE_SHEXP_GATE_V0=1 = the old kernel.
sycl::event shexp_gate_serial(sycl::queue& q, const sycl::half* xn, const float* gw, float* dng, uint32_t T, uint32_t HH) {
    static const bool v0 = [] { const char* s = std::getenv("IE_Q35MOE_SHEXP_GATE_V0"); return s && *s == '1'; }();
    if (v0)
        return q.parallel_for(sycl::range<1>(T), [=](sycl::id<1> ti) {
            const uint32_t t = uint32_t(ti); float gg = 0.f;
            for (uint32_t h = 0; h < HH; ++h) gg += gw[h] * float(xn[uint64_t(t) * HH + h]);
            dng[t] = 1.f / (1.f + sycl::exp(-gg));
        });
    constexpr uint32_t SG = 16, PER = 8, STEP = SG * PER;   // 128 elements a round, 8 per lane
    return q.parallel_for(sycl::nd_range<1>(uint64_t(T) * SG, SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
#pragma clang fp reassociate(off)
        const uint32_t t = uint32_t(it.get_group(0));
        const uint32_t lid = uint32_t(it.get_local_id(0));
        auto sg = it.get_sub_group();
        const sycl::half* xt = xn + uint64_t(t) * HH;
        const uint32_t full = HH / STEP * STEP;
        float gv[PER], xv[PER];
        if (full) {
#pragma unroll
            for (uint32_t e = 0; e < PER; ++e) { gv[e] = gw[lid * PER + e]; xv[e] = float(xt[lid * PER + e]); }
        }
        float gg = 0.f;
        for (uint32_t h0 = 0; h0 < full; h0 += STEP) {
            float gn[PER], xnx[PER];
            const uint32_t hn = h0 + STEP < full ? h0 + STEP : h0;   // prefetch the next round (last round: a harmless reload)
#pragma unroll
            for (uint32_t e = 0; e < PER; ++e) { gn[e] = gw[hn + lid * PER + e]; xnx[e] = float(xt[hn + lid * PER + e]); }
#pragma unroll
            for (uint32_t j = 0; j < STEP; ++j) {
                const float a = sycl::select_from_group(sg, gv[j % PER], j / PER);
                const float b = sycl::select_from_group(sg, xv[j % PER], j / PER);
                gg += a * b;
            }
#pragma unroll
            for (uint32_t e = 0; e < PER; ++e) { gv[e] = gn[e]; xv[e] = xnx[e]; }
        }
        for (uint32_t h = full; h < HH; ++h) gg += gw[h] * float(xt[h]);   // tail (H % 128), same serial order
        if (lid == 0) dng[t] = 1.f / (1.f + sycl::exp(-gg));
    });
}

// IE_Q35MOE_TIMING diagnostic clock (per-section prefill/decode attribution).
inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Requantize a non-Q8_0 3-D expert tensor (BF16/F16/F32) to the ggml Q8_0 block
// layout on the host, so the crown's uniform Q8_0 SoA expert path stays unchanged.
// Unsloth "dynamic"/XL quants keep a few sensitive expert layers at BF16 (higher
// precision than Q8_0); this makes them load. The tensor is [K, N, E] with each
// expert's rows contiguous; produces the exact byte layout a native-Q8_0 file has.
std::vector<block_q8_0> requant_exps_to_q8_0(const GgufTensorInfo* t) {
    const uint64_t K = t->shape[0], N = t->shape[1], E = t->shape[2];
    const uint32_t nb = uint32_t(K / 32);
    std::vector<block_q8_0> out(E * N * nb);
    const DType dt = t->dtype;

    // Q8_0-quantize a fully-dequantized expert slice ef[N*K] (row-major [N,K])
    // into out's e-th block. Same amax/127 scheme as the native float path below.
    auto quant_expert = [&](uint64_t e, const float* ef) {
        for (uint64_t n = 0; n < N; ++n)
            for (uint32_t bl = 0; bl < nb; ++bl) {
                const float* x = ef + n * K + uint64_t(bl) * 32;
                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) { float a = std::fabs(x[i]); if (a > amax) amax = a; }
                const float d  = amax / 127.0f;
                const float id = d > 0.0f ? 1.0f / d : 0.0f;
                block_q8_0& blk = out[(e * N + n) * nb + bl];
                blk.d = fp32_to_fp16(d);
                for (int i = 0; i < 32; ++i) blk.qs[i] = int8_t(std::lround(x[i] * id));
            }
    };

    // K-quant experts (a *-Q4_K_M crown: ffn_*_exps are Q4_K, ffn_down_exps mixes
    // Q6_K/Q5_K). Dequant each expert's contiguous [N,K] block slice to fp32, then
    // re-quantize to Q8_0 so the uniform Q8_0 SoA expert kernel stays unchanged.
    // Per-expert scratch keeps host RAM low (the box is 32 GB).
    if (dt == DType::kQ4_K || dt == DType::kQ5_K || dt == DType::kQ6_K) {
        const uint64_t expert_bytes = t->nbytes / E;   // K-quant block bytes per expert
        const auto* base = reinterpret_cast<const uint8_t*>(t->data);
        std::vector<float> scr(N * K);
        for (uint64_t e = 0; e < E; ++e) {
            const void* ep = base + e * expert_bytes;
            if (dt == DType::kQ4_K)      ie::ref::dequant_q4_K_buffer(ep, N * K, scr.data());
            else if (dt == DType::kQ5_K) ie::ref::dequant_q5_K_buffer(ep, N * K, scr.data());
            else                         ie::ref::dequant_q6_K_buffer(ep, N * K, scr.data());
            quant_expert(e, scr.data());
        }
        return out;
    }

    // Float experts (unsloth dynamic/XL BF16/F16/F32) — unchanged path.
    const auto* raw16 = reinterpret_cast<const uint16_t*>(t->data);
    const auto* raw32 = reinterpret_cast<const float*>(t->data);
    auto getf = [&](uint64_t idx) -> float {
        if (dt == DType::kF32) return raw32[idx];
        if (dt == DType::kF16) return fp16_to_fp32(raw16[idx]);
        uint32_t b = uint32_t(raw16[idx]) << 16;            // BF16 = top 16 bits of fp32
        float f; std::memcpy(&f, &b, sizeof(f)); return f;
    };
    for (uint64_t e = 0; e < E; ++e)
        for (uint64_t n = 0; n < N; ++n)
            for (uint32_t bl = 0; bl < nb; ++bl) {
                const uint64_t off = (e * N + n) * K + uint64_t(bl) * 32;
                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) { float a = std::fabs(getf(off + i)); if (a > amax) amax = a; }
                const float d  = amax / 127.0f;
                const float id = d > 0.0f ? 1.0f / d : 0.0f;
                block_q8_0& blk = out[(e * N + n) * nb + bl];
                blk.d = fp32_to_fp16(d);
                for (int i = 0; i < 32; ++i) blk.qs[i] = int8_t(std::lround(getf(off + i) * id));
            }
    return out;
}

// COPY of qwen35_split.cpp:upload_f32_proj_fp16 (ssm_alpha/ssm_beta → [K,Npad] fp16).
// copy-not-hoist discipline.
// Natural-order Q4_K row dequant (ggml dequantize_row_q4_K) for the Q4_K-quantized
// ssm_alpha/ssm_beta projections in *-Q4_K_M GGUFs (the split path previously only
// took F32/Q8_0). Load-time only; K % 256 == 0.
static void dequant_q4_K_row(const block_q4_K* blocks, float* out, uint64_t K) {
    const uint64_t nb = K / kQK_K;
    for (uint64_t i = 0; i < nb; ++i) {
        const float d    = fp16_to_fp32(blocks[i].d);
        const float dmin = fp16_to_fp32(blocks[i].dmin);
        const uint8_t* sc = blocks[i].scales;   // 12B packed 6-bit scales/mins
        const uint8_t* q  = blocks[i].qs;        // 128B 4-bit quants
        float* y = out + i * kQK_K;
        for (int is = 0, j = 0; j < int(kQK_K); j += 64, is += 2) {
            // get_scale_min_k4 for sub-blocks `is` and `is+1`
            uint8_t s1, m1, s2, m2;
            if (is < 4) { s1 = sc[is] & 63; m1 = sc[is + 4] & 63; }
            else { s1 = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4);
                   m1 = (sc[is + 4] >> 4)  | ((sc[is    ] >> 6) << 4); }
            const int k2 = is + 1;
            if (k2 < 4) { s2 = sc[k2] & 63; m2 = sc[k2 + 4] & 63; }
            else { s2 = (sc[k2 + 4] & 0xF) | ((sc[k2 - 4] >> 6) << 4);
                   m2 = (sc[k2 + 4] >> 4)  | ((sc[k2    ] >> 6) << 4); }
            const float d1 = d * float(s1), b1 = dmin * float(m1);
            const float d2 = d * float(s2), b2 = dmin * float(m2);
            for (int l = 0; l < 32; ++l) y[l]      = d1 * float(q[l] & 0xF) - b1;
            for (int l = 0; l < 32; ++l) y[l + 32] = d2 * float(q[l] >> 4)  - b2;
            y += 64; q += 32;
        }
    }
}

DenseQuantPtr upload_f32_proj_fp16(DeviceAllocator& alloc, const GgufTensorInfo* t,
                                   std::vector<void*>& owned, std::string& err,
                                   uint32_t Npad) {
    DenseQuantPtr out;
    if (!t) { err = "tensor not found"; return out; }
    if (t->n_dims != 2 || (t->dtype != DType::kF32 && t->dtype != DType::kQ8_0 &&
                           t->dtype != DType::kQ4_K && t->dtype != DType::kQ5_K && t->dtype != DType::kQ6_K)) {
        err = "ssm proj: expected F32, Q8_0, Q4_K, Q5_K, or Q6_K 2-D"; return out;
    }
    const uint64_t K = t->shape[0];
    const uint64_t N = t->shape[1];
    if (Npad < N) Npad = uint32_t(N);
    std::vector<sycl::half> staging(K * Npad, sycl::half(0.0f));
    if (t->dtype == DType::kF32) {
        const float* src = reinterpret_cast<const float*>(t->data);
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k)
                staging[k * Npad + n] = sycl::half(src[n * K + k]);
    } else if (t->dtype == DType::kQ8_0) {
        const uint64_t bpr = K / 32;
        const auto* blocks = reinterpret_cast<const block_q8_0*>(t->data);
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k) {
                const block_q8_0& b = blocks[n * bpr + (k >> 5)];
                staging[k * Npad + n] = sycl::half(fp16_to_fp32(b.d) * float(b.qs[k & 31]));
            }
    } else if (t->dtype == DType::kQ4_K) {  // Q4_K: dequant per row (natural order), then transpose into [K,Npad].
        if (K % kQK_K != 0) { err = "ssm proj Q4_K: K not a multiple of 256"; return out; }
        const uint64_t bpr = K / kQK_K;
        const auto* blocks = reinterpret_cast<const block_q4_K*>(t->data);
        std::vector<float> row(K);
        for (uint64_t n = 0; n < N; ++n) {
            dequant_q4_K_row(blocks + n * bpr, row.data(), K);
            for (uint64_t k = 0; k < K; ++k)
                staging[k * Npad + n] = sycl::half(row[k]);   // [N,K] → [K,Npad]
        }
    } else {  // Q6_K (e.g. OpenYourMind APEX ssm_alpha/ssm_beta): host-dequant the
              // contiguous [N,K] block span (reuses ie::ref::dequant_q6_K_buffer, same
              // at-load path as the Q4_K/expert dequant), then transpose into [K,Npad].
        if (K % kQK_K != 0) { err = "ssm proj Q6_K: K not a multiple of 256"; return out; }
        std::vector<float> deq(uint64_t(N) * K);
        if (t->dtype == DType::kQ5_K) ie::ref::dequant_q5_K_buffer(t->data, uint64_t(N) * K, deq.data());   // P4 B21
        else                          ie::ref::dequant_q6_K_buffer(t->data, uint64_t(N) * K, deq.data());
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k)
                staging[k * Npad + n] = sycl::half(deq[n * K + k]);   // [N,K] → [K,Npad]
    }
    void* d = alloc.malloc(K * Npad * sizeof(sycl::half));
    if (!d) { err = "malloc failed (ssm proj)"; return out; }
    alloc.queue().memcpy(d, staging.data(), K * Npad * sizeof(sycl::half)).wait();
    owned.push_back(d);
    out.p = d; out.dt = DType::kF16;
    return out;
}

// P4 B21: repack a Q6_K / Q5_K [K, Ncols] GGUF tensor (Ncols = N, or E*N for an expert bank) into the c32 streams on `a`.
C32Bank upload_c32(DeviceAllocator& a, const GgufTensorInfo* t, uint32_t K, uint64_t Ncols, std::vector<void*>& own,
                   std::string& e, uint64_t& bytes) {
    C32Bank b;
    const bool q6 = t->dtype == DType::kQ6_K;
    // bytes: lo 4 bits/elem; hi Q6 2 bits, Q5 1 bit; sc Q6 int8/16 = Q5 u16/32; d Q6 fp16/256, Q5 u32/256
    const uint64_t n = Ncols * K;
    const uint64_t n_lo = n / 2, n_hi = q6 ? n / 4 : n / 8, n_sc = n / 16, n_d = q6 ? n / 128 : n / 64;
    std::vector<uint8_t> lo(n_lo), hi(n_hi), sc(n_sc), dd(n_d);
    if (q6) repack_q6_K_to_c32(t->data, K, uint32_t(Ncols), lo.data(), hi.data(), reinterpret_cast<int8_t*>(sc.data()),
                               reinterpret_cast<uint16_t*>(dd.data()));
    else    repack_q5_K_to_c32(t->data, K, uint32_t(Ncols), lo.data(), reinterpret_cast<uint32_t*>(hi.data()),
                               reinterpret_cast<uint16_t*>(sc.data()), reinterpret_cast<uint32_t*>(dd.data()));
    void* p[4] = {a.malloc(n_lo), a.malloc(n_hi), a.malloc(n_sc), a.malloc(n_d)};
    for (void* x : p) if (x) own.push_back(x);
    if (!p[0] || !p[1] || !p[2] || !p[3]) { e = "K-quant c32 alloc"; return b; }
    a.queue().memcpy(p[0], lo.data(), n_lo).wait();
    a.queue().memcpy(p[1], hi.data(), n_hi).wait();
    a.queue().memcpy(p[2], sc.data(), n_sc).wait();
    a.queue().memcpy(p[3], dd.data(), n_d).wait();
    b.lo = static_cast<const uint8_t*>(p[0]); b.hi = p[1]; b.sc = p[2]; b.d = p[3]; b.kind = q6 ? 6 : 5;
    bytes += n_lo + n_hi + n_sc + n_d;
    return b;
}

// COPY of qwen35_split.cpp:extract_cols.
inline sycl::event extract_cols(sycl::queue& q, const sycl::half* src, sycl::half* dst,
                                uint32_t T, uint32_t nh, uint32_t src_stride) {
    return q.parallel_for(sycl::range<1>(uint64_t(T) * nh), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(i) / nh, h = uint32_t(i) % nh;
        dst[uint64_t(t) * nh + h] = src[uint64_t(t) * src_stride + h];
    });
}

// COPY of qwen35_split.cpp:dequant_q8_0_soa_to_Bt.
inline sycl::event dequant_q8_0_soa_to_Bt(sycl::queue& q, const int8_t* qs,
                                          const uint16_t* d, sycl::half* Bt,
                                          uint32_t K, uint32_t N) {
    const uint32_t bpc = K / 32;
    return q.parallel_for(sycl::range<2>(N, K), [=](sycl::id<2> id) {
        const uint32_t n = uint32_t(id[0]), k = uint32_t(id[1]);
        const float dv = float(sycl::bit_cast<sycl::half>(d[uint64_t(n) * bpc + (k >> 5)]));
        Bt[uint64_t(k) * N + n] = sycl::half(float(qs[uint64_t(n) * K + k]) * dv);
    });
}

// Repack a Q8_0 AoS [N,K] block matrix (n-major rows of K/32 block_q8_0) → SoA
// (qs[n*K+k] int8 + d[n*(K/32)] fp16). Operates on the e-th expert slice. Mirrors
// qwen35_split.cpp:build_split's Q8_0 branch, but writing into pre-allocated dst.
// blocks = pointer to this expert's first block_q8_0; N,K = matrix dims.
static void repack_q8_0_soa(const block_q8_0* blocks, int8_t* qs, uint16_t* dd,
                            uint32_t N, uint32_t K) {
    const uint32_t bpc = K / 32;
    for (uint64_t n = 0; n < N; ++n)
        for (uint32_t b = 0; b < bpc; ++b) {
            const block_q8_0& blk = blocks[n * bpc + b];
            dd[n * bpc + b] = *reinterpret_cast<const uint16_t*>(&blk.d);
            for (int i = 0; i < 32; ++i)
                qs[n * K + uint64_t(b) * 32 + i] = blk.qs[i];
        }
}

}  // namespace

Qwen35MoeSplitModel::~Qwen35MoeSplitModel() { free_all(); }

void Qwen35MoeSplitModel::free_all() {
    if (!fleet_) return;
    for (uint32_t d = 0; d < ws_.size(); ++d) free_ws(d);
    ws_.clear();
    for (uint32_t d = 0; d < owned_.size(); ++d)
        for (void* p : owned_[d]) if (p) fleet_->dev(d).free(p);
    owned_.clear();
    for (uint32_t d = 0; d < prefill_bt_.size(); ++d)
        if (prefill_bt_[d]) fleet_->dev(d).free(prefill_bt_[d]);
    for (uint32_t d = 0; d < prefill_dt_.size(); ++d)
        if (prefill_dt_[d]) fleet_->dev(d).free(prefill_dt_[d]);
    prefill_bt_.clear(); prefill_bt_cap_.clear(); act_q8_.clear();
    prefill_dt_.clear(); prefill_dt_cap_.clear();
    for (auto& s : dn_) s.free_storage();
    for (auto& c : kv_) c.free_storage();
    for (auto& l : lane_dn_) for (auto& s : l) s.free_storage();
    for (auto& l : lane_kv_) for (auto& c : l) c.free_storage();
    lane_dn_.clear(); lane_kv_.clear();
}

std::string Qwen35MoeSplitModel::load(DeviceFleet& fleet, const LayerPlan& plan,
                                      const GgufReader& g, const QwenConfig& cfg,
                                      uint32_t max_ctx, bool int8_kv) {
    fleet_ = &fleet; plan_ = plan; cfg_ = cfg;
    n_dev_ = plan.n_dev();
    if (n_dev_ == 0 || plan.dev_of_layer.empty()) return "qwen35moe_split: empty plan";

    // Config = the crown's QwenConfig defaults (same as the single-GPU QwenModel,
    // which this fine-tune already loads config-wise). [VERIFY if a fine-tune ever
    // changes vocab/expert_count — then read the gguf arch keys here.]
    dense::prefer_onednn() = false;

    owned_.assign(n_dev_, {});
    dev_bytes_.assign(n_dev_, 0);

    char buf[96];
    auto Ttop = [&](const char* n) { return g.find_tensor(n); };
    auto Tl   = [&](uint32_t L, const char* n) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n); return g.find_tensor(buf);
    };
    std::string err;

    // P4 B21: IE_QWEN35_SPLIT_KQ=0 keeps the old paths for Q6_K / Q5_K (dense fp fallbacks, experts requantized to Q8_0)
    static const bool split_kq = [] { const char* v = std::getenv("IE_QWEN35_SPLIT_KQ"); return !v || v[0] != '0'; }();
    // build a SplitW (2-D Q8_0 → SoA; Q6_K/Q5_K → c32; else fp fallback). COPY of qwen35_split.cpp.
    auto build_split = [](DeviceAllocator& a, const GgufTensorInfo* t,
                          std::vector<void*>& own, std::string& e, uint64_t& bytes) -> SplitW {
        SplitW w{};
        if (!t) { e = "tensor not found"; return w; }
        if (t->n_dims != 2) { e = "split weight: expected 2-D"; return w; }
        w.K = uint32_t(t->shape[0]); w.N = uint32_t(t->shape[1]);
        if (t->dtype == DType::kQ8_0) {
            const uint32_t K = w.K, N = w.N, bpc = K / 32;
            if (K % 32 != 0) { e = "Q8_0 SoA: K % 32 != 0"; return w; }
            std::vector<int8_t>   qs(uint64_t(N) * K);
            std::vector<uint16_t> dd(uint64_t(N) * bpc);
            repack_q8_0_soa(reinterpret_cast<const block_q8_0*>(t->data), qs.data(), dd.data(), N, K);
            auto* dqs = static_cast<int8_t*>(a.malloc(qs.size()));
            auto* ddd = static_cast<uint16_t*>(a.malloc(dd.size() * sizeof(uint16_t)));
            if (!dqs || !ddd) { e = "Q8_0 SoA alloc"; return w; }
            own.push_back(dqs); own.push_back(ddd);
            a.queue().memcpy(dqs, qs.data(), qs.size()).wait();
            a.queue().memcpy(ddd, dd.data(), dd.size() * sizeof(uint16_t)).wait();
            w.q8_qs = dqs; w.q8_d = ddd;
            bytes += qs.size() + dd.size() * sizeof(uint16_t);
        } else if ((t->dtype == DType::kQ6_K || t->dtype == DType::kQ5_K) && w.K % 256 == 0 && split_kq) {
            w.kq = upload_c32(a, t, w.K, w.N, own, e, bytes);   // P4 B21
        } else {
            w.fp = dense::upload_quant_dense_auto(a, t, own, e);
            if (e.empty()) bytes += t->nbytes;
        }
        return w;
    };

    // build an ExpertsW from a 3-D Q8_0 expert tensor. ggml MoE expert tensors are
    // [in, out, E] (shape[0]=K=in, shape[1]=N=out, shape[2]=E); per-expert data is
    // expert-major contiguous (expert e at block-offset e*N*(K/32)). Repack each
    // expert's [N,K] AoS slice → one contiguous SoA buffer.  [VERIFY shape order
    // against gguf vs qwen36.cpp expert loading — flagged.]
    auto build_experts = [](DeviceAllocator& a, const GgufTensorInfo* t, bool native, bool c32,
                            std::vector<void*>& own, std::string& e, uint64_t& bytes) -> ExpertsW {
        ExpertsW w{};
        if (!t) { e = "experts tensor not found"; return w; }
        if (t->n_dims != 3) { e = "experts: expected 3-D"; return w; }
        if (c32) {   // P4 B21: the layer's experts are all Q6_K / Q5_K -> the c32 banks (moe_*_c32), the GGUF's own bits
            w.dtype = t->dtype;
            w.K = uint32_t(t->shape[0]); w.N = uint32_t(t->shape[1]); w.E = uint32_t(t->shape[2]);
            w.c32 = upload_c32(a, t, w.K, uint64_t(w.E) * w.N, own, e, bytes);
            return w;
        }
        // Native Q4_K/Q6_K path (opt-in IE_Q35MOE_NATIVE_KQUANT): store the raw AoS
        // expert bank and let the crown fused MoE kernels (moe_{decode,prefill}_*_q{4,6}k,
        // soa=false) read it directly — no requant-to-Q8_0 VRAM inflation. A *-Q4_K_M
        // crown has Q4_K gate/up + Q4_K/Q6_K down, so the whole layer stays K-quant.
        if (native && (t->dtype == DType::kQ4_K || t->dtype == DType::kQ6_K)) {
            w.dtype = t->dtype;
            w.K = uint32_t(t->shape[0]); w.N = uint32_t(t->shape[1]); w.E = uint32_t(t->shape[2]);
            const size_t bs = (t->dtype == DType::kQ4_K) ? sizeof(block_q4_K) : sizeof(block_q6_K);
            if (bs == 0 || w.E == 0 || t->nbytes % (uint64_t(w.E) * bs) != 0) {
                e = "native experts: unexpected geometry"; return w;
            }
            const uint64_t nb_e = t->nbytes / w.E / bs;   // superblocks per expert
            w.expert_stride_bytes = t->nbytes / w.E;
            w.soa = true;   // per-expert SoA reorder — the layout the fused kernels expect
            std::vector<uint8_t> staging(t->nbytes);
            if (t->dtype == DType::kQ4_K)
                repack_moe_q4k_soa_host(static_cast<const uint8_t*>(t->data), staging.data(), w.E, nb_e);
            else
                repack_moe_q6k_soa_host(static_cast<const uint8_t*>(t->data), staging.data(), w.E, nb_e);
            void* dblob = a.malloc(t->nbytes);
            if (!dblob) { e = "experts native bank alloc"; return w; }
            own.push_back(dblob);
            a.queue().memcpy(dblob, staging.data(), t->nbytes).wait();
            w.blob = dblob;
            bytes += t->nbytes;
            return w;
        }
        std::vector<block_q8_0> requant;   // populated iff the tensor isn't native Q8_0
        const block_q8_0* base;
        if (t->dtype == DType::kQ8_0) {
            base = reinterpret_cast<const block_q8_0*>(t->data);
        } else if (t->dtype == DType::kBF16 || t->dtype == DType::kF16 ||
                   t->dtype == DType::kF32  || t->dtype == DType::kQ4_K ||
                   t->dtype == DType::kQ5_K || t->dtype == DType::kQ6_K) {
            requant = requant_exps_to_q8_0(t);   // dynamic/XL floats OR *-Q4_K_M K-quant experts → Q8_0
            base = requant.data();
        } else {
            e = std::string("experts: unsupported dtype ") + std::string(type_name(t->dtype)) +
                " (need Q8_0/Q4_K/Q5_K/Q6_K, or BF16/F16/F32)";
            return w;
        }
        w.K = uint32_t(t->shape[0]); w.N = uint32_t(t->shape[1]); w.E = uint32_t(t->shape[2]);
        const uint32_t K = w.K, N = w.N, E = w.E, bpc = K / 32;
        if (K % 32 != 0) { e = "experts Q8_0: K % 32 != 0"; return w; }
        w.qs_stride = uint64_t(N) * K;
        w.d_stride  = uint64_t(N) * bpc;
        std::vector<int8_t>   qs(uint64_t(E) * w.qs_stride);
        std::vector<uint16_t> dd(uint64_t(E) * w.d_stride);
        const uint64_t blocks_per_expert = uint64_t(N) * bpc;
        // `base` set above: native Q8_0 view, or the host-requantized copy.
        for (uint32_t ex = 0; ex < E; ++ex)
            repack_q8_0_soa(base + ex * blocks_per_expert,
                            qs.data() + ex * w.qs_stride, dd.data() + ex * w.d_stride, N, K);
        auto* dqs = static_cast<int8_t*>(a.malloc(qs.size()));
        auto* ddd = static_cast<uint16_t*>(a.malloc(dd.size() * sizeof(uint16_t)));
        if (!dqs || !ddd) { e = "experts SoA alloc"; return w; }
        own.push_back(dqs); own.push_back(ddd);
        a.queue().memcpy(dqs, qs.data(), qs.size()).wait();
        a.queue().memcpy(ddd, dd.data(), dd.size() * sizeof(uint16_t)).wait();
        w.qs = dqs; w.d = ddd;
        bytes += qs.size() + dd.size() * sizeof(uint16_t);
        return w;
    };

    // token_embd → embed_dev.
    {
        DeviceAllocator& ea = fleet.dev(plan.embed_dev);
        const auto* ti = Ttop("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype == DType::kBF16 || ti->dtype == DType::kF16) {
            // BF16/F16 token_embd (e.g. OpenYourMind APEX crowns keep the embedding
            // at BF16). Dequant to fp16 on the host and gather via embedding_lookup_f16.
            // The ggml embedding is [hidden, vocab] with shape[0]=hidden contiguous, i.e.
            // element (token,h) at flat token*hidden+h — exactly the [vocab,hidden]
            // row-major layout embedding_lookup_f16 expects, so NO transpose.
            const uint64_t n = ti->nbytes / sizeof(uint16_t);
            std::vector<uint16_t> f16(n);
            const auto* src = reinterpret_cast<const uint16_t*>(ti->data);
            if (ti->dtype == DType::kBF16)
                for (uint64_t i = 0; i < n; ++i) {
                    uint32_t b = uint32_t(src[i]) << 16;   // BF16 = top 16 bits of fp32
                    float f; std::memcpy(&f, &b, sizeof(f));
                    f16[i] = fp32_to_fp16(f);
                }
            else
                std::memcpy(f16.data(), src, n * sizeof(uint16_t));   // already fp16
            void* d = ea.malloc(n * sizeof(uint16_t));
            if (!d) return "token_embd: malloc failed";
            ea.queue().memcpy(d, f16.data(), n * sizeof(uint16_t)).wait();
            owned_[plan.embed_dev].push_back(d);
            token_embd_ = d;
            token_embd_dtype_ = DType::kF16;
            dev_bytes_[plan.embed_dev] += n * sizeof(uint16_t);
        } else if (ti->dtype == DType::kQ4_K || ti->dtype == DType::kQ5_K || ti->dtype == DType::kQ6_K ||
                   ti->dtype == DType::kQ8_0) {
            token_embd_dtype_ = ti->dtype;
            token_embd_ = dense::upload<void>(ea, ti, owned_[plan.embed_dev], err, ti->dtype);
            if (!err.empty()) return "token_embd: " + err;
            dev_bytes_[plan.embed_dev] += ti->nbytes;
        } else {
            return std::string("token_embd: unsupported dtype ") +
                   std::string(type_name(ti->dtype)) + " (need Q4_K/Q5_K/Q6_K/Q8_0/F16/BF16)";
        }
    }
    // output_norm + lm_head → head_dev.
    {
        DeviceAllocator& ha = fleet.dev(plan.head_dev);
        output_norm_ = dense::upload<float>(ha, Ttop("output_norm.weight"),
                                            owned_[plan.head_dev], err, DType::kF32);
        if (!err.empty()) return "output_norm: " + err;
        const auto* ti = Ttop("output.weight");
        if (!ti) ti = Ttop("token_embd.weight");
        output_ = build_split(ha, ti, owned_[plan.head_dev], err, dev_bytes_[plan.head_dev]);
        if (!err.empty()) return "output: " + err;
    }

    const uint32_t n_layers = cfg_.n_layers;   // 40
    layers_.assign(n_layers, {});
    dn_local_.assign(n_layers, 0);
    kv_local_.assign(n_layers, 0);

    for (uint32_t L = 0; L < n_layers; ++L) {
        const uint32_t dev = plan.dev_of_layer[L];
        DeviceAllocator& a = fleet.dev(dev);
        auto& own = owned_[dev];
        LayerW& w = layers_[L];
        // crown: full-attn every full_attn_interval-th layer (i % 4 == 3); else DeltaNet.
        w.is_linear = ((L + 1) % cfg_.full_attn_interval) != 0;

        auto LW = [&](const char* n) -> SplitW {
            return build_split(a, Tl(L, n), own, err, dev_bytes_[dev]);
        };
        // Native K-quant experts (DEFAULT-ON, opt-out IE_Q35MOE_NO_NATIVE_KQUANT):
        // eligible only when the WHOLE layer's MoE is K-quant the crown fused kernels
        // support — Q4_K gate + Q4_K up + Q4_K/Q6_K down. A Q5_K down (e.g. an MTP
        // block) or any float expert falls the layer back to the Q8_0-requant path.
        // Validated on 2 Q4_K_M crowns (2×B70): −41% VRAM (21 vs 36 GB), +31% prefill,
        // decode-PPL parity/better (4.86=4.86; 4.95≤5.00), coherent prefill+gen both.
        static const bool native_kq_env = std::getenv("IE_Q35MOE_NO_NATIVE_KQUANT") == nullptr;
        const auto* t_g = Tl(L, "ffn_gate_exps.weight");
        const auto* t_u = Tl(L, "ffn_up_exps.weight");
        const auto* t_d = Tl(L, "ffn_down_exps.weight");
        const bool layer_native = native_kq_env && t_g && t_u && t_d &&
            t_g->dtype == DType::kQ4_K && t_u->dtype == DType::kQ4_K &&
            (t_d->dtype == DType::kQ4_K || t_d->dtype == DType::kQ6_K);
        // P4 B21: a layer whose experts are all Q6_K / Q5_K (gate and up of one type, K % 256 == 0) takes the c32 banks
        auto kq3 = [](const GgufTensorInfo* t) {
            return t && (t->dtype == DType::kQ6_K || t->dtype == DType::kQ5_K) && t->n_dims == 3 && t->shape[0] % 256 == 0;
        };
        const bool layer_c32 = split_kq && !layer_native && kq3(t_g) && kq3(t_u) && kq3(t_d) && t_g->dtype == t_u->dtype;
        auto EW = [&](const char* n) -> ExpertsW {
            return build_experts(a, Tl(L, n), layer_native, layer_c32, own, err, dev_bytes_[dev]);
        };
        auto F32 = [&](const char* n, float*& dst) -> std::string {
            const auto* ti = Tl(L, n);
            dst = dense::upload<float>(a, ti, own, err, DType::kF32);
            if (!err.empty()) return std::string(n) + ": " + err;
            if (ti) dev_bytes_[dev] += ti->nbytes;
            return {};
        };
        auto le = [&](const char* what) {
            return "layer " + std::to_string(L) + " " + what + ": " + err;
        };

        if (auto m = F32("attn_norm.weight", w.attn_norm); !m.empty())
            return "layer " + std::to_string(L) + " " + m;
        if (auto m = F32("post_attention_norm.weight", w.post_attn_norm); !m.empty())
            return "layer " + std::to_string(L) + " " + m;

        if (w.is_linear) {
            w.attn_qkv  = LW("attn_qkv.weight");  if (!err.empty()) return le("attn_qkv");
            w.attn_gate = LW("attn_gate.weight"); if (!err.empty()) return le("attn_gate");
            if (auto m = F32("ssm_a", w.ssm_a); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            const uint32_t svh_pad = ((cfg_.ssm_n_v_heads + 63u) / 64u) * 64u;
            w.ssm_alpha = upload_f32_proj_fp16(a, Tl(L, "ssm_alpha.weight"), own, err, svh_pad);
            if (!err.empty()) return le("ssm_alpha");
            w.ssm_beta  = upload_f32_proj_fp16(a, Tl(L, "ssm_beta.weight"), own, err, svh_pad);
            if (!err.empty()) return le("ssm_beta");
            if (auto m = F32("ssm_dt.bias", w.ssm_dt_bias); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            {
                const auto* ti = Tl(L, "ssm_conv1d.weight");
                if (auto m = F32("ssm_conv1d.weight", w.ssm_conv1d); !m.empty()) return "layer " + std::to_string(L) + " " + m;
                const uint64_t n = ti->nbytes / sizeof(float);
                w.ssm_conv1d_fp16 = static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_conv1d_fp16) return le("ssm_conv1d fp16 malloc");
                own.push_back(w.ssm_conv1d_fp16); dev_bytes_[dev] += n * sizeof(sycl::half);
                cast_fp32_to_fp16(a.queue(), w.ssm_conv1d, w.ssm_conv1d_fp16, n).wait();
            }
            {
                const auto* ti = Tl(L, "ssm_norm.weight");
                if (auto m = F32("ssm_norm.weight", w.ssm_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
                const uint64_t n = ti->nbytes / sizeof(float);
                w.ssm_norm_fp16 = static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_norm_fp16) return le("ssm_norm fp16 malloc");
                own.push_back(w.ssm_norm_fp16); dev_bytes_[dev] += n * sizeof(sycl::half);
                cast_fp32_to_fp16(a.queue(), w.ssm_norm, w.ssm_norm_fp16, n).wait();
            }
            w.ssm_out = LW("ssm_out.weight");   if (!err.empty()) return le("ssm_out");
        } else {
            w.attn_q      = LW("attn_q.weight");      if (!err.empty()) return le("attn_q");
            w.attn_k      = LW("attn_k.weight");      if (!err.empty()) return le("attn_k");
            w.attn_v      = LW("attn_v.weight");      if (!err.empty()) return le("attn_v");
            w.attn_output = LW("attn_output.weight"); if (!err.empty()) return le("attn_output");
            if (auto m = F32("attn_q_norm.weight", w.attn_q_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            if (auto m = F32("attn_k_norm.weight", w.attn_k_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
        }

        // MoE FFN — router + experts + shared expert (every layer).
        if (auto m = F32("ffn_gate_inp.weight", w.ffn_gate_inp); !m.empty()) return "layer " + std::to_string(L) + " " + m;
        w.exp_gate = EW("ffn_gate_exps.weight"); if (!err.empty()) return le("ffn_gate_exps");
        w.exp_up   = EW("ffn_up_exps.weight");   if (!err.empty()) return le("ffn_up_exps");
        w.exp_down = EW("ffn_down_exps.weight"); if (!err.empty()) return le("ffn_down_exps");
        // shared expert (optional — present on the crown).
        if (Tl(L, "ffn_gate_inp_shexp.weight")) {
            if (auto m = F32("ffn_gate_inp_shexp.weight", w.ffn_gate_inp_shexp); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            w.sh_gate = LW("ffn_gate_shexp.weight"); if (!err.empty()) return le("ffn_gate_shexp");
            w.sh_up   = LW("ffn_up_shexp.weight");   if (!err.empty()) return le("ffn_up_shexp");
            w.sh_down = LW("ffn_down_shexp.weight"); if (!err.empty()) return le("ffn_down_shexp");
        }
    }

    // Per-card hybrid caches.
    std::vector<uint32_t> n_lin(n_dev_, 0), n_full(n_dev_, 0);
    for (uint32_t L = 0; L < n_layers; ++L) {
        const uint32_t dev = plan.dev_of_layer[L];
        if (layers_[L].is_linear) dn_local_[L] = n_lin[dev]++;
        else                      kv_local_[L] = n_full[dev]++;
    }
    dn_.resize(n_dev_); kv_.resize(n_dev_);
    cur_.assign(n_dev_, 0);
    max_ctx_ = max_ctx;
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (n_lin[dev]) {
            DeltaNetStateConfig dc{};
            dc.n_layers_linear = n_lin[dev];
            dc.n_v_heads       = cfg_.ssm_n_v_heads;       // 32
            dc.v_head_dim      = cfg_.ssm_head_dim;        // 128
            dc.k_head_dim      = cfg_.ssm_head_dim;        // 128
            dc.conv_channels   = cfg_.ssm_inner + 2u * cfg_.ssm_n_k_heads * cfg_.ssm_head_dim;  // 8192
            dc.conv_kernel     = cfg_.ssm_conv_kernel;     // 4
            if (auto m = dn_[dev].init(fleet.dev(dev), dc); !m.empty())
                return "dn cache dev " + std::to_string(dev) + ": " + m;
        }
        if (n_full[dev]) {
            KvCacheConfig kc{};
            kc.n_layers_full = n_full[dev];
            kc.n_kv_heads    = cfg_.n_kv_heads;            // 2
            kc.max_ctx       = max_ctx;
            kc.head_dim      = cfg_.head_dim;              // 256
            kc.use_int8      = int8_kv;                    // --int8-kv: halve KV (int8 shadow + fp16 scales)
            if (auto m = kv_[dev].init(fleet.dev(dev), kc); !m.empty())
                return "kv cache dev " + std::to_string(dev) + ": " + m;
        }
    }

    // Per-card int-dot activation scratch + prefill dequant scratch.
    const uint32_t Kmax = std::max(std::max(cfg_.hidden, cfg_.expert_ffn),
                                   std::max(cfg_.ssm_inner, cfg_.n_q_heads * cfg_.head_dim));
    act_q8_.assign(n_dev_, nullptr);
    prefill_bt_.assign(n_dev_, nullptr);
    prefill_bt_cap_.assign(n_dev_, 0);
    prefill_dt_.assign(n_dev_, nullptr);
    prefill_dt_cap_.assign(n_dev_, 0);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        void* p = fleet.dev(dev).malloc((uint64_t(Kmax) / 32) * sizeof(block_q8_1x));
        if (!p) return "act_q8 alloc dev " + std::to_string(dev);
        owned_[dev].push_back(p);
        act_q8_[dev] = p;
    }
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        std::fprintf(stderr, "[qwen35moe_split] card %u weights: %.2f GB\n",
                     dev, double(dev_bytes_[dev]) / 1e9);
    // P4 B43: the prefill-attention threshold, so a log says which bytes a run produced
    if (std::getenv("IE_Q35MOE_NO_FA2_TILE"))
        std::fprintf(stderr, "[qwen35moe_split] prefill attention: the naive kernel at every depth (IE_Q35MOE_NO_FA2_TILE)\n");
    else
        std::fprintf(stderr, "[qwen35moe_split] prefill attention: tiled (FA2 wide tile) from %u positions, naive below "
                             "(IE_Q35MOE_FA2_TILE_MINCTX; 6144 = v0.2.6's bytes; IE_Q35MOE_NO_FA2_TILE = naive everywhere)\n",
                     q35m_fa2_tile_minctx());
    return {};
}

void Qwen35MoeSplitModel::free_ws(uint32_t dev) {
    if (!fleet_ || dev >= ws_.size()) return;
    Workspace& w = ws_[dev];
    auto& alloc = fleet_->dev(dev);
    for (void* p : {static_cast<void*>(w.x), static_cast<void*>(w.x_normed),
                    static_cast<void*>(w.attn_block), static_cast<void*>(w.positions),
                    static_cast<void*>(w.positions3),
                    static_cast<void*>(w.ids), static_cast<void*>(w.logits),
                    static_cast<void*>(w.qg), static_cast<void*>(w.q), static_cast<void*>(w.gate),
                    static_cast<void*>(w.k), static_cast<void*>(w.v), static_cast<void*>(w.attn_out),
                    static_cast<void*>(w.attn_partials),
                    static_cast<void*>(w.ag.qh), static_cast<void*>(w.ag.s), static_cast<void*>(w.ag.p),
                    static_cast<void*>(w.ag.o), static_cast<void*>(w.ag.l),
                    static_cast<void*>(w.dn_qkv), static_cast<void*>(w.dn_conv), static_cast<void*>(w.dn_z),
                    static_cast<void*>(w.dn_qpre), static_cast<void*>(w.dn_kpre), static_cast<void*>(w.dn_vpre),
                    static_cast<void*>(w.dn_g), static_cast<void*>(w.dn_beta), static_cast<void*>(w.dn_out),
                    static_cast<void*>(w.dn_qrep), static_cast<void*>(w.dn_krep),
                    static_cast<void*>(w.dn_alpha_h), static_cast<void*>(w.dn_beta_h),
                    static_cast<void*>(w.dn_alpha64), static_cast<void*>(w.dn_beta64),
                    static_cast<void*>(w.topk_idx), static_cast<void*>(w.topk_w),
                    static_cast<void*>(w.gate_o), static_cast<void*>(w.up_o), static_cast<void*>(w.ffn_h),
                    static_cast<void*>(w.moe_y), static_cast<void*>(w.eo),
                    static_cast<void*>(w.moe_hp), static_cast<void*>(w.moe_yp),
                    w.x_q8, w.h_q8,
                    static_cast<void*>(w.moe_xp), w.xp_q8,
                    static_cast<void*>(w.moe_eoff), static_cast<void*>(w.moe_sidx),
                    static_cast<void*>(w.moe_sw), static_cast<void*>(w.moe_tk2pk)})
        if (p) alloc.free(p);
    w = Workspace{};
}

std::string Qwen35MoeSplitModel::ensure_ws(uint32_t dev, uint32_t max_T) {
    Workspace& w = ws_[dev];
    max_T = (max_T + kAttnGemmRowStep - 1) / kAttnGemmRowStep * kAttnGemmRowStep;   // P4 B49: whole row steps
    if (max_T <= w.T) return {};
    if (w.T != 0) free_ws(dev);
    auto& alloc = fleet_->dev(dev);
    auto ah = [&](size_t n) { return static_cast<sycl::half*>(alloc.malloc(n * sizeof(sycl::half))); };
    auto af = [&](size_t n) { return static_cast<float*>(alloc.malloc(n * sizeof(float))); };

    const uint64_t T   = max_T;
    const uint32_t H   = cfg_.hidden;                       // 2048
    const uint32_t N_q = cfg_.n_q_heads * cfg_.head_dim;    // 4096
    const uint32_t N_qg = N_q * 2u;                         // 8192
    const uint32_t N_kv = cfg_.n_kv_heads * cfg_.head_dim;  // 512
    const uint32_t EFF = cfg_.expert_ffn;                  // 512
    const uint32_t SI  = cfg_.ssm_inner;                    // 4096
    const uint32_t cvc = SI + 2u * cfg_.ssm_n_k_heads * cfg_.ssm_head_dim;  // 8192
    const uint32_t Vd  = cfg_.ssm_n_v_heads * cfg_.ssm_head_dim;            // 4096
    const uint32_t Nv  = cfg_.ssm_n_v_heads;               // 32
    const uint32_t Nvp = ((Nv + 63u) / 64u) * 64u;         // 64
    const uint32_t Kt  = cfg_.experts_topk;                // 8
    const uint64_t TK  = T * Kt;

    w.x = ah(T*H); w.x_normed = ah(T*H); w.attn_block = ah(T*H);
    w.positions = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));
    w.positions3 = static_cast<int32_t*>(alloc.malloc(3 * T * sizeof(int32_t)));   // P4 B45 (12 bytes a row)
    if (dev == plan_.embed_dev) {   // P4 B22: persistent, like the 27B split's
        w.ids = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));
        if (!w.ids) return "qwen35moe_split ids alloc failed on dev " + std::to_string(dev);
    }
    if (dev == plan_.head_dev) {
        w.logits = ah(cfg_.vocab);
        if (!w.logits) return "qwen35moe_split logits alloc failed on dev " + std::to_string(dev);
    }
    w.qg = ah(T*N_qg); w.q = ah(T*N_q); w.gate = ah(T*N_q);
    w.k = ah(T*N_kv); w.v = ah(T*N_kv); w.attn_out = ah(T*N_q);
    w.dn_qkv = ah(T*cvc); w.dn_conv = ah(T*cvc); w.dn_z = ah(T*SI);
    w.dn_qpre = af(T*Vd); w.dn_kpre = af(T*Vd); w.dn_vpre = af(T*Vd);
    w.dn_g = af(T*Nv); w.dn_beta = af(T*Nv); w.dn_out = af(T*Vd);
    w.dn_qrep = af(T*Vd); w.dn_krep = af(T*Vd);
    w.dn_alpha_h = ah(T*Nv); w.dn_beta_h = ah(T*Nv);
    w.dn_alpha64 = ah(T*Nvp); w.dn_beta64 = ah(T*Nvp);
    w.topk_idx = static_cast<int32_t*>(alloc.malloc(TK * sizeof(int32_t)));
    w.topk_w = ah(TK); w.gate_o = ah(T*EFF); w.up_o = ah(T*EFF); w.ffn_h = ah(T*EFF);
    w.moe_y = ah(T*H); w.eo = ah(T*H);
    w.moe_hp = ah(TK*EFF); w.moe_yp = ah(TK*H);
    w.x_q8 = alloc.malloc(uint64_t(T) * (H / 32) * sizeof(block_q8_1x));
    // h_q8/xp_q8 hold block_q8_1x (Q8 path) OR block_q8_1s (native Q4/Q6 prefill, 48B
    // > 40B) — size for the larger so both activation quantizers fit the same scratch.
    w.h_q8 = alloc.malloc(TK * (EFF / 32) * sizeof(block_q8_1s));
    // Expert-batched prefill scratch (sized to maxTK; tiny at decode T=1).
    const uint32_t E = cfg_.n_experts;                     // 256
    w.moe_xp = ah(TK*H);
    w.xp_q8 = alloc.malloc(TK * (H / 32) * sizeof(block_q8_1s));   // block_q8_1s (native) ≥ block_q8_1x
    w.moe_eoff = static_cast<uint32_t*>(alloc.malloc((uint64_t(E) + 1) * sizeof(uint32_t)));
    w.moe_sidx = static_cast<int32_t*>(alloc.malloc(TK * sizeof(int32_t)));
    w.moe_sw = ah(TK);
    w.moe_tk2pk = static_cast<uint32_t*>(alloc.malloc(TK * sizeof(uint32_t)));

    if (!w.x || !w.x_normed || !w.attn_block || !w.positions || !w.positions3 || !w.qg || !w.q ||
        !w.gate || !w.k || !w.v || !w.attn_out || !w.dn_qkv || !w.dn_conv || !w.dn_z ||
        !w.dn_qpre || !w.dn_kpre || !w.dn_vpre || !w.dn_g || !w.dn_beta || !w.dn_out ||
        !w.dn_qrep || !w.dn_krep || !w.dn_alpha_h || !w.dn_beta_h || !w.dn_alpha64 ||
        !w.dn_beta64 || !w.topk_idx || !w.topk_w || !w.gate_o || !w.up_o || !w.ffn_h ||
        !w.moe_y || !w.eo || !w.moe_hp || !w.moe_yp || !w.x_q8 || !w.h_q8 ||
        !w.moe_xp || !w.xp_q8 || !w.moe_eoff || !w.moe_sidx || !w.moe_sw || !w.moe_tk2pk)
        return "qwen35moe_split workspace alloc failed on dev " + std::to_string(dev);

    if (kv_[dev].ready()) {
        const uint32_t max_ctx = kv_[dev].config().max_ctx;
        constexpr uint32_t Bc_floor = 64;
        const uint32_t n_chunks_max = (max_ctx + Bc_floor - 1) / Bc_floor;
        const uint64_t n_floats = uint64_t(n_chunks_max) * cfg_.n_q_heads * (cfg_.head_dim + 2);
        w.attn_partials = static_cast<float*>(alloc.malloc(n_floats * sizeof(float)));
        if (!w.attn_partials) return "qwen35moe_split attn_partials alloc failed on dev " + std::to_string(dev);
        w.partials_ctx = max_ctx;
        // P4 B49: full_attention_prefill_gemm's scratch. Scores + weights: 32M elements (128 + 64 MB), or 16 rows at
        // this card's largest context when that is more.
        if (attn_gemm_on()) {
            const uint32_t gqa = cfg_.n_q_heads / cfg_.n_kv_heads;
            w.ag.sp_elems = std::max<uint64_t>(uint64_t(32) << 20, uint64_t(16) * max_ctx);
            w.ag.qh = ah(T * N_q);
            w.ag.s  = af(w.ag.sp_elems);
            w.ag.p  = ah(w.ag.sp_elems);
            w.ag.o  = af(T * gqa * cfg_.head_dim);
            w.ag.l  = af(T * gqa);
            if (!w.ag.qh || !w.ag.s || !w.ag.p || !w.ag.o || !w.ag.l)
                return "qwen35moe_split attention scratch alloc failed on dev " + std::to_string(dev);
        }
    }
    w.T = max_T;
    return {};
}

// P4 B49: the 27B split's reading levers (P4 B48, ~/ds41_work/p60/b48/BUILD.md) on the crown split. Each is on by
// default; =0 restores the old path.
//   IE_Q35MOE_ATTN_GEMM   a prefill piece's full attention = full_attention_prefill_gemm (two oneDNN matmuls + a softmax)
//   IE_Q35MOE_S8_PREFILL  a Q8_0 projection read in place by the matmul (gemm_nt_s8_f16_onednn), no expand-to-fp16 pass
//   IE_Q35MOE_DN_SCAN     the DeltaNet recurrence of a piece = deltanet_scan_prefill (state in registers, no barriers)
namespace {
bool q35m_b49_switch(const char* name) {
    const char* e = std::getenv(name);
    return !(e && *e && std::atoi(e) == 0);
}
}  // namespace
bool Qwen35MoeSplitModel::attn_gemm_on() {
    static const bool on = q35m_b49_switch("IE_Q35MOE_ATTN_GEMM") && onednn_available();
    return on;
}
bool Qwen35MoeSplitModel::s8_prefill_on() {
    static const bool on = q35m_b49_switch("IE_Q35MOE_S8_PREFILL") && onednn_available();
    return on;
}
bool Qwen35MoeSplitModel::dn_scan_on() {
    static const bool on = q35m_b49_switch("IE_Q35MOE_DN_SCAN");
    return on;
}

sycl::event Qwen35MoeSplitModel::sgemv(uint32_t dev, const sycl::half* A, const SplitW& w,
                                       sycl::half* out, uint32_t K, uint32_t N, uint32_t T) {
    auto& alloc = fleet_->dev(dev);
    auto& q = alloc.queue();
    if (w.kq.lo && T == 1) return kq_gemv(dev, act_q8_[dev], A, w, out, K, N, 1);   // P4 B21
    if (!w.int_dot()) return dense::gemv_q_T(q, A, w.fp, out, K, N, T);
    if (T == 1) {
        DPE(dev, kQuant, quantize_q8_1(q, A, act_q8_[dev], K));
        sycl::event e = gemv_q8_0_soa_q8_sel(q, act_q8_[dev], w.q8_qs, w.q8_d, out, K, N);
        if (g_dp35.active()) g_dp35.tag(dev, e, DecodeProf::kGemv);
        return e;
    }
    // P4 B49: a Q8_0 weight goes to the matmul in place (oneDNN weights decompression): no expand pass. The scale plane
    // is transposed per call (1/16 of the weight's bytes) into a scratch grown to the largest plane.
    if (w.q8_qs && s8_prefill_on()) {
        const uint64_t nd = uint64_t(N) * (K / 32u);
        if (nd > prefill_dt_cap_[dev]) {
            q.wait();   // a queued matmul may still read the old plane
            if (prefill_dt_[dev]) alloc.free(prefill_dt_[dev]);
            prefill_dt_[dev] = static_cast<sycl::half*>(alloc.malloc(nd * sizeof(sycl::half)));
            prefill_dt_cap_[dev] = prefill_dt_[dev] ? nd : 0;
        }
        if (prefill_dt_[dev]) {
            q8_scales_kb_major(q, w.q8_d, prefill_dt_[dev], N, K / 32u);
            sycl::event e;
            const uint32_t M = sgemv_pad_ ? std::min<uint32_t>((T + kAttnGemmRowStep - 1) / kAttnGemmRowStep * kAttnGemmRowStep,
                                                               ws_[dev].T) : T;
            if (gemm_nt_s8_f16_onednn(q, A, w.q8_qs, prefill_dt_[dev], out, M, N, K, 32, &e)) return e;
            static bool said = false;
            if (!said) {
                said = true;
                std::fprintf(stderr, "[qwen35moe_split] IE_Q35MOE_S8_PREFILL: no jitted oneDNN kernel (%s); the expand route\n",
                             onednn_nt_s8_f16_impl(q, T, N, K, 32).c_str());
            }
        }
    }
    const uint64_t need = uint64_t(K) * N;
    if (need > prefill_bt_cap_[dev]) {
        if (prefill_bt_[dev]) alloc.free(prefill_bt_[dev]);
        prefill_bt_[dev] = static_cast<sycl::half*>(alloc.malloc(need * sizeof(sycl::half)));
        prefill_bt_cap_[dev] = prefill_bt_[dev] ? need : 0;
    }
    if (!prefill_bt_[dev]) { std::fprintf(stderr, "qwen35moe_split: prefill_bt alloc failed\n"); return {}; }
    if (w.kq.kind == 6)
        dequant_q6k_c32_to_Bt(q, w.kq.lo, static_cast<const uint8_t*>(w.kq.hi), static_cast<const int8_t*>(w.kq.sc),
                              static_cast<const uint16_t*>(w.kq.d), prefill_bt_[dev], K, N);
    else if (w.kq.kind == 5)
        dequant_q5k_c32_to_Bt(q, w.kq.lo, static_cast<const uint32_t*>(w.kq.hi), static_cast<const uint16_t*>(w.kq.sc),
                              static_cast<const uint32_t*>(w.kq.d), prefill_bt_[dev], K, N);
    else
        dequant_q8_0_soa_to_Bt(q, w.q8_qs, w.q8_d, prefill_bt_[dev], K, N);
    return dense::gemv_q_T(q, A, DenseQuantPtr{prefill_bt_[dev], DType::kF16}, out, K, N, T);
}

// Per-expert Q8_0 GEMV (one expert slice e of an ExpertsW). T tokens of A[T,K] →
// out[T,N]. Decode (T==1) int-dot; prefill dequant-to-fp16 + gemm.
sycl::event Qwen35MoeSplitModel::egemv(uint32_t dev, const sycl::half* A, const ExpertsW& w,
                                       uint32_t e, sycl::half* out, uint32_t T) {
    auto& alloc = fleet_->dev(dev);
    auto& q = alloc.queue();
    const int8_t*   qs = w.qs + uint64_t(e) * w.qs_stride;
    const uint16_t* dd = w.d  + uint64_t(e) * w.d_stride;
    if (T == 1) {
        quantize_q8_1(q, A, act_q8_[dev], w.K);
        return gemv_q8_0_soa_q8_sel(q, act_q8_[dev], qs, dd, out, w.K, w.N);
    }
    const uint64_t need = uint64_t(w.K) * w.N;
    if (need > prefill_bt_cap_[dev]) {
        if (prefill_bt_[dev]) alloc.free(prefill_bt_[dev]);
        prefill_bt_[dev] = static_cast<sycl::half*>(alloc.malloc(need * sizeof(sycl::half)));
        prefill_bt_cap_[dev] = prefill_bt_[dev] ? need : 0;
    }
    if (!prefill_bt_[dev]) return {};
    dequant_q8_0_soa_to_Bt(q, qs, dd, prefill_bt_[dev], w.K, w.N);
    return dense::gemv_q_T(q, A, DenseQuantPtr{prefill_bt_[dev], DType::kF16}, out, w.K, w.N, T);
}

std::string Qwen35MoeSplitModel::forward(const int32_t* input_ids, uint32_t T,
                                         uint32_t start_pos, bool reset_kv,
                                         sycl::half* out_logits_host) {
    static const char* kStopped = "stopped at a layer boundary: the engine is stopping";   // (P4 B33)
    if (T == 0) return "T == 0";
    if (aborting(T)) return kStopped;
    const uint32_t H    = cfg_.hidden;                  // 2048
    const uint32_t V    = cfg_.vocab;

    if (ws_.size() != n_dev_) ws_.assign(n_dev_, {});
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (auto m = ensure_ws(dev, T); !m.empty()) return m;
    g_dp35.begin(T, start_pos);

    // positions on every card (+ the [3, T] M-RoPE streams while vision is staged, P4 B45); the reset on a fresh sequence
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (ws_[dev].positions) upload_positions(dev, start_pos, T);
        if (reset_kv) {
            if (dn_at(dev).ready()) dn_at(dev).reset(fleet_->dev(dev).queue());
            if (kv_at(dev).ready()) kv_at(dev).reset();
        }
    }

    // embedding → ws_[embed_dev].x (+ the image rows spliced over it, P4 B45)
    if (auto m = embed(input_ids, T, start_pos); !m.empty()) return m;
    g_dp35.phase(DecodeProf::kSetup);

    // IE_Q35MOE_TIMING: per-section wall-clock (adds q.wait barriers → relative
    // attribution only, not absolute tok/s). Answers "is prefill attn- or MoE-bound?".
    const bool dbg_timing = std::getenv("IE_Q35MOE_TIMING") != nullptr;
    double t_attn = 0.0, t_moe = 0.0;
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (dev > 0) {
            // Drain the SOURCE card's queue before the cross-card x copy. The GPU-resident
            // MoE removed the per-layer host-pull that used to drain it implicitly, so without
            // this the copy reads card (dev-1)'s x while its kernels are still writing → a
            // cross-card race that faults the device (UR_RESULT_ERROR_DEVICE_LOST). One barrier
            // per card boundary (n_dev-1 total) — compute stays fully on the GPU.
            fleet_->dev(dev - 1).queue().wait();
            fleet_->copy_across(dev - 1, ws_[dev].x, dev, ws_[dev - 1].x, uint64_t(T) * H * sizeof(sycl::half));
            g_dp35.phase(DecodeProf::kHand);
        }
        run_layers(dev, T, start_pos, dbg_timing, t_attn, t_moe);
        fleet_->dev(dev).queue().wait();
        if (aborting(T)) return kStopped;   // (P4 B33: the card's queue is drained; the next card and the head never start)
        g_dp35.phase(dev == 0 ? DecodeProf::kCard0 : DecodeProf::kCard1);
        g_dp35.sync(dev);
    }
    if (dbg_timing)
        std::fprintf(stderr, "[q35moe-timing T=%u] attn=%.2f ms  moe=%.2f ms  (moe %.0f%%)\n",
                     T, t_attn, t_moe, 100.0 * t_moe / (t_attn + t_moe + 1e-9));

    // final norm + lm_head on head_dev.
    {
        auto& alloc = fleet_->dev(plan_.head_dev);
        sycl::half* d_logits = ws_[plan_.head_dev].logits;   // P4 B22: persistent [V] (ensure_ws)
        if (!d_logits) return "logits alloc failed";
        head(T, d_logits);
        g_dp35.sync(plan_.head_dev);
        alloc.queue().memcpy(out_logits_host, d_logits, uint64_t(V) * sizeof(sycl::half)).wait();
        g_dp35.phase(DecodeProf::kHead);
    }
    g_dp35.end();
    return {};
}

std::string Qwen35MoeSplitModel::embed(const int32_t* input_ids, uint32_t T, uint32_t start_pos) {
    const uint32_t H = cfg_.hidden;
    auto& alloc = fleet_->dev(plan_.embed_dev);
    auto& q = alloc.queue();
    int32_t* d_ids = ws_[plan_.embed_dev].ids;   // P4 B22: persistent [T] (ensure_ws)
    if (!d_ids) return "d_ids alloc failed";
    q.memcpy(d_ids, input_ids, T * sizeof(int32_t)).wait();
    if (token_embd_dtype_ == DType::kQ4_K)
        embedding_lookup_q4k(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
    else if (token_embd_dtype_ == DType::kQ8_0)
        embedding_lookup_q8_0(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
    else if (token_embd_dtype_ == DType::kQ5_K)
        embedding_lookup_q5k(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
    else if (token_embd_dtype_ == DType::kF16)
        // NB: embedding_lookup_f16 takes (embd, tokens) — swapped vs the q* kernels.
        embedding_lookup_f16(q, static_cast<const sycl::half*>(token_embd_), d_ids,
                             ws_[plan_.embed_dev].x, T, H);
    else
        embedding_lookup_q6k(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
    // P4 B45 vision: the staged projector rows over the gathered rows (in-order: after the gather, before layer 0; a no-op
    // without spans -- the lanes' forward_stage refuses while vision is staged, so only forward() reaches this with any)
    splice_vision(q, ws_[plan_.embed_dev].x, start_pos, T);
    q.wait();
    return {};
}

// ---- P4 B45 step 2: image input (the header's note; the 27B split's set, qwen35_split.cpp). ---------------------------
std::string Qwen35MoeSplitModel::set_vision(const float* rows_f32, uint32_t t0, uint32_t n_rows) {
    if (!rows_f32 || !n_rows) return "qwen35moe_split set_vision: empty rows";
    if (uint64_t(t0) + n_rows > max_ctx_) return "qwen35moe_split set_vision: span past max_ctx";
    const uint32_t H = cfg_.hidden;
    const size_t row0 = vis_rows_.size() / H;
    vis_rows_.resize(vis_rows_.size() + size_t(n_rows) * H);
    for (size_t i = 0; i < size_t(n_rows) * H; ++i) vis_rows_[row0 * H + i] = sycl::half(rows_f32[i]);
    vis_spans_.push_back({t0, n_rows, uint32_t(row0)});
    return {};
}

void Qwen35MoeSplitModel::set_mrope(const int32_t* pos3, uint32_t n_total, int32_t delta) {
    mrope3_.assign(pos3, pos3 + size_t(3) * n_total);
    mrope_n_ = n_total;
    mrope_delta_ = delta;
}

void Qwen35MoeSplitModel::clear_vision() {
    vis_spans_.clear();
    vis_rows_.clear();
    mrope3_.clear();
    mrope_n_ = 0;
    mrope_delta_ = 0;
}

// The piece [start, start + T)'s positions onto card dev: the linear token positions (every path's KV index and, without
// vision, the rope) and, while vision is staged, the [3, T] M-RoPE slice the rope reads instead (one blocking copy each,
// as forward() did before B45).
void Qwen35MoeSplitModel::upload_positions(uint32_t dev, uint32_t start, uint32_t T) {
    Workspace& w = ws_[dev];
    auto& q = fleet_->dev(dev).queue();
    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start + t);
    q.memcpy(w.positions, pos.data(), T * sizeof(int32_t)).wait();
    if (mrope_n_) {
        std::vector<int32_t> p3(size_t(3) * T);
        qwen4_mrope3_slice(mrope3_.data(), mrope_n_, mrope_delta_, start, T, p3.data());
        q.memcpy(w.positions3, p3.data(), p3.size() * sizeof(int32_t)).wait();
    }
}

// The staged projector rows over the piece's gathered embedding rows (f16 row memcpys on the embed card's in-order queue).
void Qwen35MoeSplitModel::splice_vision(sycl::queue& q, sycl::half* x, uint32_t start, uint32_t T) {
    if (vis_spans_.empty()) return;
    const uint32_t H = cfg_.hidden;
    for (const Qwen4VisSplice& r : qwen4_vis_splice_ranges(vis_spans_, start, T))
        q.memcpy(x + uint64_t(r.dst) * H, vis_rows_.data() + uint64_t(r.src) * H, uint64_t(r.n) * H * sizeof(sycl::half));
}

// One card's layers of a T-row step at start_pos (the selected lane's state on that card): x → x in ws_[dev].
void Qwen35MoeSplitModel::run_layers(uint32_t dev, uint32_t T, uint32_t start_pos, bool dbg_timing,
                                     double& t_attn, double& t_moe) {
    const uint32_t H    = cfg_.hidden;                  // 2048
    const uint32_t HD   = cfg_.head_dim;                // 256
    const uint32_t N_q  = cfg_.n_q_heads  * HD;         // 4096
    const uint32_t N_qg = N_q * 2u;                     // 8192
    const uint32_t N_kv = cfg_.n_kv_heads * HD;         // 512
    const uint32_t EFF  = cfg_.expert_ffn;             // 512
    const uint32_t E    = cfg_.n_experts;               // 256
    const uint32_t Kt   = cfg_.experts_topk;            // 8
    const uint32_t rope_n = cfg_.rope_dim;              // 64
    const float    eps  = cfg_.rms_eps;
    const uint32_t n_layers = cfg_.n_layers;
    const uint32_t n_kv = cfg_.n_kv_heads;
    Workspace& w = ws_[dev];
    auto& q = fleet_->dev(dev).queue();
    const uint64_t per_layer_kv =
        uint64_t(n_kv) * (kv_at(dev).ready() ? kv_at(dev).config().max_ctx : 0u) * HD;
    struct PadGuard {   // P4 B49: sgemv's operands are workspace buffers for the length of this call
        bool& f;
        explicit PadGuard(bool& x) : f(x) { f = true; }
        ~PadGuard() { f = false; }
    } pad_guard(sgemv_pad_);

    for (uint32_t L = 0; L < n_layers; ++L) {
        if (plan_.dev_of_layer[L] != dev) continue;
        if (aborting(T)) return;   // P4 B33: the engine is stopping -- no new layer (the caller drains the queue)
        const LayerW& w_l = layers_[L];
        double _ta = 0.0; if (dbg_timing) { q.wait(); _ta = now_ms(); }
        DPE(dev, kNorm, rms_norm_f32w(q, w.x, w_l.attn_norm, w.x_normed, T, H, eps));

        if (w_l.is_linear) {
            const uint32_t SKH = cfg_.ssm_n_k_heads;          // 16
            const uint32_t SVH = cfg_.ssm_n_v_heads;          // 32
            const uint32_t SHD = cfg_.ssm_head_dim;           // 128
            const uint32_t SI  = cfg_.ssm_inner;              // 4096
            const uint32_t conv_ch = SI + 2u * SKH * SHD;     // 8192
            const uint32_t kw  = SKH * SHD;                   // 2048
            const uint32_t rep = SVH / SKH;                   // 2 (16→32) — crown TILE
            const float qscale = 1.0f / sycl::sqrt(float(SHD));

            sgemv(dev, w.x_normed, w_l.attn_qkv, w.dn_qkv, H, conv_ch, T);
            sycl::half* conv_state = dn_at(dev).conv_state_ptr() +
                uint64_t(dn_local_[L]) * dn_at(dev).conv_elems_per_layer();
            DPE(dev, kDnConv, depthwise_conv1d_causal(q, w.dn_qkv, w_l.ssm_conv1d_fp16, conv_state,
                                    w.dn_conv, T, conv_ch, cfg_.ssm_conv_kernel));
            DPE(dev, kDnMisc, cast_qkv_split_fp16_to_fp32(q, w.dn_conv, w.dn_qpre, w.dn_kpre, w.dn_vpre, T, kw, SI));
            DPE(dev, kDnMisc, l2_norm_scale(q, w.dn_qpre, w.dn_qpre, T * SKH, SHD, qscale, 1e-6f));
            DPE(dev, kDnMisc, l2_norm_scale(q, w.dn_kpre, w.dn_kpre, T * SKH, SHD, 1.0f,   1e-6f));
            DPE(dev, kDnMisc, repeat_interleave_heads(q, w.dn_qpre, w.dn_qrep, T, SKH, SHD, rep));
            DPE(dev, kDnMisc, repeat_interleave_heads(q, w.dn_kpre, w.dn_krep, T, SKH, SHD, rep));
            const uint32_t SVHp = ((SVH + 63u) / 64u) * 64u;
            if (T == 1 && w_l.ssm_alpha.dt == DType::kF16 && w_l.ssm_beta.dt == DType::kF16) {
                // P4 B22: both [H, 64] F16 projections in ONE launch; per (row, mat) the T = 1 gemv_fp16 body verbatim
                DPE(dev, kGemvSmall, gemv_fp16_rows_dual(q, w.x_normed, H,
                    static_cast<const sycl::half*>(w_l.ssm_alpha.p), w.dn_alpha64,
                    static_cast<const sycl::half*>(w_l.ssm_beta.p),  w.dn_beta64, SVHp, H, SVHp, 1));
            } else {
            DPE(dev, kGemvSmall, dense::gemv_q_T(q, w.x_normed, w_l.ssm_alpha, w.dn_alpha64, H, SVHp, T));
            DPE(dev, kGemvSmall, dense::gemv_q_T(q, w.x_normed, w_l.ssm_beta,  w.dn_beta64,  H, SVHp, T));
            }
            DPE(dev, kDnMisc, extract_cols(q, w.dn_alpha64, w.dn_alpha_h, T, SVH, SVHp));
            DPE(dev, kDnMisc, extract_cols(q, w.dn_beta64,  w.dn_beta_h,  T, SVH, SVHp));
            DPE(dev, kDnMisc, compute_g_beta_h16(q, w.dn_alpha_h, w.dn_beta_h, w_l.ssm_a, w_l.ssm_dt_bias,
                               w.dn_g, w.dn_beta, T, SVH));
            float* state_layer = dn_at(dev).state_ptr() +
                uint64_t(dn_local_[L]) * dn_at(dev).state_elems_per_layer();
            if (dn_scan_on() && T >= kAttnGemmMinT && SHD == 128)   // P4 B49
                deltanet_scan_prefill(q, w.dn_qrep, w.dn_krep, w.dn_vpre, w.dn_g, w.dn_beta, state_layer, w.dn_out, T, SVH);
            else
            DPE(dev, kDnRec, deltanet_recurrence(q, w.dn_qrep, w.dn_krep, w.dn_vpre, w.dn_g, w.dn_beta,
                                state_layer, w.dn_out, /*B=*/1, T, SVH, SHD, SHD));
            sgemv(dev, w.x_normed, w_l.attn_gate, w.dn_z, H, SI, T);
            DPE(dev, kDnMisc, gated_rms_norm(q, w.dn_out, w.dn_z, w_l.ssm_norm_fp16, w.dn_qkv, T * SVH, SHD, eps));
            sgemv(dev, w.dn_qkv, w_l.ssm_out, w.attn_block, SI, H, T);
        } else {
            sgemv(dev, w.x_normed, w_l.attn_q, w.qg, H, N_qg, T);
            DPE(dev, kAttnPre, split_q_gate_per_head(q, w.qg, w.q, w.gate, T, cfg_.n_q_heads, HD));
            sgemv(dev, w.x_normed, w_l.attn_k, w.k, H, N_kv, T);
            sgemv(dev, w.x_normed, w_l.attn_v, w.v, H, N_kv, T);
            DPE(dev, kAttnPre, rms_norm_f32w(q, w.q, w_l.attn_q_norm, w.q, T * cfg_.n_q_heads,  HD, eps));
            DPE(dev, kAttnPre, rms_norm_f32w(q, w.k, w_l.attn_k_norm, w.k, T * cfg_.n_kv_heads, HD, eps));
            if (mrope_n_) {   // P4 B45 vision: the 3-stream M-RoPE (== rope_partial when the streams are equal)
                rope_imrope3(q, w.q, w.positions3, w.q, T, cfg_.n_q_heads,  HD, rope_n, cfg_.rope_theta);
                rope_imrope3(q, w.k, w.positions3, w.k, T, cfg_.n_kv_heads, HD, rope_n, cfg_.rope_theta);
            } else {
            DPE(dev, kAttnPre, rope_partial(q, w.q, w.positions, w.q, T, cfg_.n_q_heads,  HD, rope_n, cfg_.rope_theta));
            DPE(dev, kAttnPre, rope_partial(q, w.k, w.positions, w.k, T, cfg_.n_kv_heads, HD, rope_n, cfg_.rope_theta));
            }
            auto& kvc = kv_at(dev);
            const uint32_t li = kv_local_[L];   // local full-attn layer index on this card
            sycl::half* kc = kvc.k_ptr() + per_layer_kv * li;
            sycl::half* vc = kvc.v_ptr() + per_layer_kv * li;
            const uint32_t max_ctx = kvc.config().max_ctx;
            if (T == 1 && w.attn_partials) {
                // INT8-KV decode (--int8-kv): halve KV BW+storage via the int8 shadow
                // when it's populated up to start_pos; else fp16 FA2. Mirrors qwen36.cpp.
                if (kvc.is_int8() && kvc.k_int8_ptr() && start_pos == kvc.int8_length(li)) {
                    const uint64_t i8pl = uint64_t(cfg_.n_kv_heads) * max_ctx * HD;
                    const uint64_t scpl = uint64_t(cfg_.n_kv_heads) * max_ctx;
                    full_attention_fa2_decode_int8(q, w.q, w.k, w.v,
                                                   kvc.k_int8_ptr() + i8pl * li,
                                                   kvc.v_int8_ptr() + i8pl * li,
                                                   kvc.k_scales_ptr() + scpl * li,
                                                   kvc.v_scales_ptr() + scpl * li,
                                                   nullptr, nullptr,
                                                   w.attn_out, w.attn_partials, start_pos,
                                                   cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx);
                    kvc.set_int8_length(li, start_pos + 1);   // inline quantize-on-write wrote start_pos
                } else if (q35::xmx_decode(start_pos, max_ctx)) {
                    // P4 B23: the XMX kernel from IE_Q35_XMX_DECODE_MIN context (default on, IE_Q35_XMX_DECODE=0 off;
                    // numerics change; q35_xmx_decode.hpp).
                    DPE(dev, kAttn, full_attention_fa2_decode_xmx(q, w.q, w.k, w.v, kc, vc, w.attn_out,
                                              w.attn_partials, start_pos,
                                              cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx));
                } else {
                    DPE(dev, kAttn, full_attention_fa2_decode(q, w.q, w.k, w.v, kc, vc, w.attn_out,
                                              w.attn_partials, start_pos,
                                              cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx));
                }
            } else {
                // P4 B29: long-ctx full-attn prefill through the Gemma wide-tile kernel at ctx >= minctx -- the gate of the 27B split
                // (qwen35_split.cpp) and the single-card crown (qwen36.cpp); naive O(T^2) re-reads the whole KV per query row and
                // was 80-90% of a 30-54K-deep swarm piece. Covers --parallel 1's prefill and the lanes' pieces (both run_layers;
                // the rows step is decode-only). Opt-out IE_Q35MOE_NO_FA2_TILE; tune IE_Q35MOE_FA2_TILE_MINCTX.
                // P4 B43: the default is 512, not the 27B's 6144 (B29 copied it, never measured on the crown). Measured on the GPU
                // (~/ds41_work/p60/fa2minctx/REPORT.md, v0.2.6, 11 jobs): below 6144 the naive kernel costs ~30 ms per 1K positions
                // per 512-row piece vs ~18 on the tile, with no tile overhead at depth 0; at 512 the swarm replay 132.0 -> 123.5 s
                // (-6.4 %), the workers phase -9 %, their TTFT p50 4.1 -> 3.0 s, decode p50 8.5 -> 9.2, a 2,038-token cold prefill
                // 1.41 -> 1.28 s; PPL on the chunks that change kernel within +-0.5 % either way, needles 6/6 with byte-identical
                // replies. It also removes a concurrency-dependent bit difference: a prompt crossing 6144 took naive or tile rows
                // depending on whether it was re-cut. IE_Q35MOE_FA2_TILE_MINCTX=6144 restores v0.2.6's bytes.
                static const bool no_tile = std::getenv("IE_Q35MOE_NO_FA2_TILE") != nullptr;
                static const uint32_t tile_minctx = q35m_fa2_tile_minctx();
                if (w.ag.s && T >= kAttnGemmMinT) {   // P4 B49
                    full_attention_prefill_gemm(q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                                                cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx, w.ag);
                } else if (!no_tile && HD == 256 && (start_pos + T) >= tile_minctx) {
                    full_attention_fa2_prefill_tile_gemma(q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                                                          cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx, 0 /*window: full causal*/);
                } else {
                    full_attention(q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                                   cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx);
                }
                // Post-quantize the fp16 prefill rows into the int8 shadow so the next
                // T=1 decode step can take the int8 path.
                if (kvc.is_int8()) kvc.quantize_to_int8(q, li, start_pos, T);
            }
            kvc.set_length(li, start_pos + T);
            DPE(dev, kAttnPre, sigmoid_gate(q, w.attn_out, w.gate, w.attn_out, uint64_t(T) * N_q));
            sgemv(dev, w.attn_out, w_l.attn_output, w.attn_block, N_q, H, T);
        }

        // residual + post-attn norm → MoE.
        DPE(dev, kNorm, residual_add_rms_norm_fused(q, w.x, w.attn_block, w_l.post_attn_norm, w.x_normed, T, H, eps));
        if (dbg_timing) { q.wait(); t_attn += now_ms() - _ta; _ta = now_ms(); }
        if (std::getenv("IE_MOE_DBG")) { std::fprintf(stderr,"[L%u T%u dev%u attn-done is_lin=%d]\n",L,T,dev,(int)w_l.is_linear); q.wait(); }

        // ---- MoE — FULLY GPU-RESIDENT (no host pull, no per-expert host loop):
        // router → quantize x → gate_up_silu_q8 → quantize h → down_q8 → reduce.
        // topk_idx/topk_w stay on device; the grouped kernels read the expert id
        // per token-slot from device memory. ----
        DPE(dev, kRouter, moe_router(q, w.x_normed, w_l.ffn_gate_inp, w.topk_idx, w.topk_w, T, H, E, Kt));
        if (std::getenv("IE_MOE_DBG") && dev > 0) {   // card-1 router input + topk sanity
            std::vector<sycl::half> _xn(H);
            q.memcpy(_xn.data(), w.x_normed, H * sizeof(sycl::half)).wait();
            int _bad = 0; float _mx = 0.f;
            for (auto hh : _xn) { float v = float(hh); if (v != v || v > 1e30f || v < -1e30f) _bad++; float a = v < 0 ? -v : v; if (a > _mx) _mx = a; }
            std::vector<int32_t> _ti(uint64_t(T) * Kt);
            q.memcpy(_ti.data(), w.topk_idx, _ti.size() * sizeof(int32_t)).wait();
            int _tmn = 1 << 30, _tmx = -(1 << 30); for (int32_t v : _ti) { if (v < _tmn) _tmn = v; if (v > _tmx) _tmx = v; }
            std::fprintf(stderr, "[card1 L%u T%u] x_normed bad=%d maxabs=%.1f | topk min=%d max=%d (E=%u)\n", L, T, _bad, _mx, _tmn, _tmx, E);
        }
        // The DECODE (T==1) per-token-slot kernels re-stream every expert's Q8
        // weights ~TK/E times — fine at T=1 (host/comm-bound) but ~80% of pp512
        // prefill. T>1 routes through the EXPERT-BATCHED (weight-stationary) GEMM
        // analog: host counting-sort of the TK slots by expert → expert-sorted
        // gather → batched gate_up/down reading each weight column once. Numerics
        // are bit-identical (same block_q8_1x activation, same dp4a/lane order,
        // weight folded at down) so PPL is exact. Opt out: IE_Q35MOE_NO_PREFILL_GEMM.
        static const bool no_prefill_gemm = std::getenv("IE_Q35MOE_NO_PREFILL_GEMM") != nullptr;
        // Native K-quant experts always take the batched (prefill) path for T>1 —
        // the crown moe_decode_*_q4k kernels are single-token (no T arg).
        const bool moe_c32 = w_l.exp_gate.c32.lo != nullptr;   // P4 B21: Q6_K / Q5_K experts
        const bool moe_native = !moe_c32 && w_l.exp_gate.dtype != DType::kQ8_0;
        if (T > 1 && (!no_prefill_gemm || moe_native)) {
            const uint32_t TOTAL = T * Kt;
            // Pull topk to host (the only blocking read), counting-sort by expert,
            // push the partition + gather index + reduce map back (mirror qwen36.cpp).
            std::vector<int32_t>    hidx(TOTAL);
            std::vector<sycl::half> htw(TOTAL);
            q.memcpy(hidx.data(), w.topk_idx, TOTAL * sizeof(int32_t));
            q.memcpy(htw.data(),  w.topk_w,   TOTAL * sizeof(sycl::half)).wait();
            std::vector<uint32_t> off(E + 1, 0);
            for (uint32_t i = 0; i < TOTAL; ++i) ++off[uint32_t(hidx[i]) + 1];
            for (uint32_t ee = 0; ee < E; ++ee) off[ee + 1] += off[ee];
            std::vector<int32_t>    sidx(TOTAL);
            std::vector<sycl::half> sw(TOTAL);
            std::vector<uint32_t>   tk2pk(TOTAL);
            std::vector<uint32_t>   cursor(E, 0);
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t kk = 0; kk < Kt; ++kk) {
                    const uint32_t ex  = uint32_t(hidx[t * Kt + kk]);
                    const uint32_t pos = off[ex] + cursor[ex]++;
                    sidx[pos]            = int32_t(t);
                    sw[pos]              = htw[t * Kt + kk];
                    tk2pk[t * Kt + kk]   = pos;
                }
            q.memcpy(w.moe_eoff,  off.data(),   (E + 1) * sizeof(uint32_t));
            q.memcpy(w.moe_sidx,  sidx.data(),  TOTAL * sizeof(int32_t));
            q.memcpy(w.moe_sw,    sw.data(),    TOTAL * sizeof(sycl::half));
            q.memcpy(w.moe_tk2pk, tk2pk.data(), TOTAL * sizeof(uint32_t)).wait();

            moe_gather_rows(q, w.x_normed, w.moe_sidx, w.moe_xp, TOTAL, H);
            if (moe_c32) {   // P4 B21: expert-batched c32 kernels, moe_prefill_*_q8's contract
                quantize_q8_1(q, w.moe_xp, static_cast<block_q8_1x*>(w.xp_q8), uint32_t(TOTAL) * H);
                moe_prefill_gate_up_silu_c32(q, w.xp_q8, w_l.exp_gate.c32, w_l.exp_up.c32, w.moe_eoff, w.moe_hp, E, H, EFF);
                quantize_q8_1(q, w.moe_hp, static_cast<block_q8_1x*>(w.h_q8), uint32_t(TOTAL) * EFF);
                moe_prefill_down_c32(q, w.h_q8, w_l.exp_down.c32, w.moe_eoff, w.moe_sw, w.moe_yp, E, H, EFF);
                moe_prefill_reduce_sum(q, w.moe_yp, w.moe_tk2pk, w.moe_y, T, Kt, H);
            } else if (moe_native) {
                // Native K-quant batched prefill (crown fused kernels). NOTE: the
                // Q4_K prefill kernels consume the block_q8_1s activation stream
                // (quantize_q8_1s, 48B) — NOT the block_q8_1x the Q8 kernels use.
                // The down kernel does NOT fold the routing weight, so the reduce
                // is the WEIGHTED moe_prefill_reduce (moe_sw), not _sum.
                quantize_q8_1s(q, w.moe_xp, w.xp_q8, uint64_t(TOTAL) * H);
                moe_prefill_gate_up_silu_q4k_q8(q, w.xp_q8, w_l.exp_gate.blob, w_l.exp_up.blob,
                                                w.moe_eoff, w.moe_hp, E, H, EFF,
                                                w_l.exp_gate.expert_stride_bytes, w_l.exp_gate.soa);
                quantize_q8_1s(q, w.moe_hp, w.h_q8, uint64_t(TOTAL) * EFF);
                if (w_l.exp_down.dtype == DType::kQ4_K)
                    moe_prefill_down_packed_q4k_q8(q, w.h_q8, w_l.exp_down.blob, w.moe_eoff,
                                                   w.moe_yp, E, H, EFF,
                                                   w_l.exp_down.expert_stride_bytes, w_l.exp_down.soa);
                else
                    moe_prefill_down_packed_q6k_q8(q, w.h_q8, w_l.exp_down.blob, w.moe_eoff,
                                                   w.moe_yp, E, H, EFF,
                                                   w_l.exp_down.expert_stride_bytes, w_l.exp_down.soa);
                // moe_prefill_reduce ACCUMULATES into moe_y (acc = y + Σ w·down),
                // unlike the Q8 moe_prefill_reduce_sum which writes — so moe_y (reused
                // per layer, holds stale data) MUST be zeroed first. In-order queue.
                q.memset(w.moe_y, 0, uint64_t(T) * H * sizeof(sycl::half));
                moe_prefill_reduce(q, w.moe_yp, w.moe_tk2pk, w.moe_sw, w.moe_y, T, Kt, H);
            } else {
            quantize_q8_1(q, w.moe_xp, static_cast<block_q8_1x*>(w.xp_q8), uint32_t(TOTAL) * H);
            moe_prefill_gate_up_silu_q8(q, w.xp_q8, w_l.exp_gate.qs, w_l.exp_gate.d,
                                        w_l.exp_up.qs, w_l.exp_up.d, w_l.exp_gate.qs_stride,
                                        w_l.exp_gate.d_stride, w.moe_eoff, w.moe_hp, E, H, EFF);
            quantize_q8_1(q, w.moe_hp, static_cast<block_q8_1x*>(w.h_q8), uint32_t(TOTAL) * EFF);
            moe_prefill_down_q8(q, w.h_q8, w_l.exp_down.qs, w_l.exp_down.d,
                                w_l.exp_down.qs_stride, w_l.exp_down.d_stride, w.moe_eoff,
                                w.moe_sw, w.moe_yp, E, H, EFF);
            moe_prefill_reduce_sum(q, w.moe_yp, w.moe_tk2pk, w.moe_y, T, Kt, H);
            }
        } else {
        // DECODE (and IE_Q35MOE_NO_PREFILL_GEMM A/B fallback) — per-token-slot
        // grouped kernels. Activation → block_q8_1x in ONE batched quantize.
        // H % 32 == 0 so the per-32 blocks are row-independent → quantizing all
        // T·H elements at once is identical to a per-row loop but a SINGLE enqueue
        // (the per-row loop overran the device command queue → DEVICE_LOST).
        if (moe_c32) {   // P4 B21: the per-slot c32 kernels (the rows path runs the same ones)
            DPE(dev, kQuant, quantize_q8_1(q, w.x_normed, static_cast<block_q8_1x*>(w.x_q8), uint32_t(T) * H));
            DPE(dev, kMoe, moe_gate_up_silu_c32(q, w.x_q8, w_l.exp_gate.c32, w_l.exp_up.c32, w.topk_idx, w.moe_hp, T, Kt, H, EFF));
            DPE(dev, kQuant, quantize_q8_1(q, w.moe_hp, static_cast<block_q8_1x*>(w.h_q8), uint32_t(T) * Kt * EFF));
            DPE(dev, kMoe, moe_down_c32(q, w.h_q8, w_l.exp_down.c32, w.topk_idx, w.topk_w, w.moe_yp, T, Kt, EFF, H));
            DPE(dev, kElem, moe_reduce_q8(q, w.moe_yp, w.moe_y, T, Kt, H));
        } else if (moe_native) {
            // Native K-quant decode (T==1; native forces T>1 to the prefill path).
            // gate/up: W4A16 (fp16 x_normed) → fp16 h. down: fp16 h_in, folds topk_w
            // AND reduces over K_top → moe_y directly (no separate moe_reduce).
            moe_decode_gate_up_silu_q4k(q, w.x_normed, w_l.exp_gate.blob, w_l.exp_up.blob,
                                        w.topk_idx, w.moe_hp, H, EFF, Kt,
                                        w_l.exp_gate.expert_stride_bytes, w_l.exp_gate.soa);
            if (w_l.exp_down.dtype == DType::kQ4_K)
                moe_decode_down_q4k(q, w.moe_hp, w_l.exp_down.blob, w.topk_idx, w.topk_w,
                                    w.moe_y, H, EFF, Kt,
                                    w_l.exp_down.expert_stride_bytes, w_l.exp_down.soa);
            else
                moe_decode_down_q6k(q, w.moe_hp, w_l.exp_down.blob, w.topk_idx, w.topk_w,
                                    w.moe_y, H, EFF, Kt,
                                    w_l.exp_down.expert_stride_bytes, w_l.exp_down.soa);
        } else {
        DPE(dev, kQuant, quantize_q8_1(q, w.x_normed, static_cast<block_q8_1x*>(w.x_q8), uint32_t(T) * H));
        DPE(dev, kMoe, moe_gate_up_silu_q8(q, w.x_q8, w_l.exp_gate.qs, w_l.exp_gate.d,
                            w_l.exp_up.qs, w_l.exp_up.d, w_l.exp_gate.qs_stride,
                            w_l.exp_gate.d_stride, w.topk_idx, w.moe_hp, T, Kt, H, EFF));
        DPE(dev, kQuant, quantize_q8_1(q, w.moe_hp, static_cast<block_q8_1x*>(w.h_q8), uint32_t(T) * Kt * EFF));
        DPE(dev, kMoe, moe_down_q8(q, w.h_q8, w_l.exp_down.qs, w_l.exp_down.d, w_l.exp_down.qs_stride,
                    w_l.exp_down.d_stride, w.topk_idx, w.topk_w, w.moe_yp, T, Kt, EFF, H));
        DPE(dev, kElem, moe_reduce_q8(q, w.moe_yp, w.moe_y, T, Kt, H));   // moe_y = Σ_k weighted experts
        }
        }
        // shared expert (always-on, sigmoid-gated).
        if (w_l.ffn_gate_inp_shexp) {
            sgemv(dev, w.x_normed, w_l.sh_gate, w.gate_o, H, EFF, T);
            sgemv(dev, w.x_normed, w_l.sh_up,   w.up_o,   H, EFF, T);
            DPE(dev, kElem, swiglu(q, w.gate_o, w.up_o, w.ffn_h, uint64_t(T) * EFF));
            sgemv(dev, w.ffn_h, w_l.sh_down, w.eo, EFF, H, T);
            // shared-expert gate g[t]=sigmoid(ffn_gate_inp_shexp · x_normed[t]);
            // moe_y += eo*g. [VERIFY gate semantics vs qwen36.cpp:1775-1829.]
            {
                float* dng = w.dn_g; const float* gw = w_l.ffn_gate_inp_shexp;
                const sycl::half* xn = w.x_normed; const sycl::half* eo = w.eo;
                sycl::half* my = w.moe_y; const uint32_t HH = H;
                DPE(dev, kShexpGate, shexp_gate_serial(q, xn, gw, dng, T, HH));   // P4 B22
                DPE(dev, kElem, q.parallel_for(sycl::range<1>(uint64_t(T) * HH), [=](sycl::id<1> i) {
                    const uint32_t t = uint32_t(uint64_t(i) / HH);
                    my[i] = sycl::half(float(my[i]) + float(eo[i]) * dng[t]);
                }));
            }
        }
        DPE(dev, kElem, residual_add(q, w.x, w.moe_y, w.x, uint64_t(T) * H));
        if (dbg_timing) { q.wait(); t_moe += now_ms() - _ta; }
        if (std::getenv("IE_MOE_DBG")) { std::fprintf(stderr,"[L%u T%u dev%u moe-done]\n",L,T,dev); q.wait(); }
    }
}

// Final norm + lm_head of the last row into d_logits (device, head_dev); returns when it is written.
void Qwen35MoeSplitModel::head(uint32_t T, sycl::half* d_logits) {
    const uint32_t H = cfg_.hidden, V = cfg_.vocab;
    const uint32_t hd = plan_.head_dev;
    Workspace& w = ws_[hd];
    auto& q = fleet_->dev(hd).queue();
    DPE(hd, kNorm, rms_norm_f32w(q, w.x, output_norm_, w.x_normed, T, H, cfg_.rms_eps));
    const sycl::half* last = w.x_normed + uint64_t(T - 1) * H;
    sgemv(hd, last, output_, d_logits, H, V, 1).wait();
}


// ---- P4 B10: request lanes (docs/lanes/LANES_SERVE.md) -----------------------------------------------------------

std::string Qwen35MoeSplitModel::init_lanes(uint32_t n_lanes, uint32_t lane_ctx, uint32_t max_rows, uint64_t reserve_bytes,
                                            LanesAutoFit* fit) {
    if (fit) fit->cards.clear();
    else if (n_lanes < 2) return {};
    if (!fleet_ || n_dev_ != 2 || plan_.embed_dev != 0 || plan_.head_dev != 1)
        return "request lanes need the two-card plan (embedding on card 0, head on card 1)";
    if (!lane_kv_.empty()) return "request lanes are already allocated";
    if (lane_ctx < 9 || lane_ctx > max_ctx_)
        return "lane context " + std::to_string(lane_ctx) + " outside [9, " + std::to_string(max_ctx_) + "]";
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (kv_[dev].ready() && kv_[dev].is_int8()) return "request lanes do not support --int8-kv";
    // the workspace for the largest step first (it only grows; the lanes then never meet a growth mid-pipe)
    if (ws_.size() != n_dev_) ws_.assign(n_dev_, {});
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        // (B14 gate #6: at least a whole rows group too, so forward_stage_rows never grows it on a stage thread)
        if (auto m = ensure_ws(dev, std::max<uint32_t>(max_rows, kMaxRows)); !m.empty()) return m;
    lane_bytes_.assign(n_dev_, 0);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        Q35mLaneShape sh;
        if (kv_[dev].ready()) {
            const auto& kc = kv_[dev].config();
            sh.n_full = kc.n_layers_full; sh.n_kv_heads = kc.n_kv_heads; sh.head_dim = kc.head_dim;
        }
        if (dn_[dev].ready()) {
            const auto& dc = dn_[dev].config();
            sh.n_lin = dc.n_layers_linear; sh.v_heads = dc.n_v_heads; sh.k_head_dim = dc.k_head_dim; sh.v_head_dim = dc.v_head_dim;
            sh.conv_channels = dc.conv_channels; sh.conv_kernel = dc.conv_kernel;
        }
        lane_bytes_[dev] = q35m_lane_bytes(sh, lane_ctx);
        const sycl::device dv = fleet_->dev(dev).device();
        if (!dv.has(sycl::aspect::ext_intel_free_memory)) return "request lanes: card " + std::to_string(dev) + " does not report free memory";
        const uint64_t free_now = dv.get_info<sycl::ext::intel::info::device::free_memory>();
        // B14 gate #5: the rows buffers allocated below (the G-row activation on each card, the G rows' logits on the head card)
        // are counted in the reserve's place: they come out of the free memory before the reserve is kept
        const uint32_t Kmax_r = std::max(std::max(cfg_.hidden, cfg_.expert_ffn), std::max(cfg_.ssm_inner, cfg_.n_q_heads * cfg_.head_dim));
        const uint64_t rows_b = uint64_t(kMaxRows) * (Kmax_r / 32) * sizeof(block_q8_1x) +
                                (dev == plan_.head_dev ? uint64_t(kMaxRows) * cfg_.vocab * sizeof(sycl::half) : 0ull) +
                                (kv_[dev].ready()   // P4 B26: the rows attention partials
                                     ? uint64_t(kMaxRows) * fa2_decode_rows_partials_floats(cfg_.n_q_heads, cfg_.n_kv_heads, cfg_.head_dim,
                                                                                           kv_[dev].config().max_ctx) * sizeof(float)
                                     : 0ull);
        if (fit) {   // P4 B30: measured here, picked below (a lane's DeltaNet part = its sticky checkpoint, P4 B29)
            Q35mLaneShape dn = sh;
            dn.n_full = 0;
            fit->cards.push_back({free_now, lane_bytes_[dev], fit->dn_checkpoint ? q35m_lane_bytes(dn, 0) : 0ull,
                                  reserve_bytes + rows_b + fit->keep});
            continue;
        }
        if (auto m = q35m_lanes_fit(free_now, lane_bytes_[dev], n_lanes, lane_ctx, reserve_bytes + rows_b, dev); !m.empty()) return m;
    }
    if (fit) {
        n_lanes = fit->n = lanes_auto_fit(fit->cards, fit->n_max);
        if (n_lanes < 2) { lane_bytes_.clear(); return {}; }   // one lane: nothing allocated (the workspace keeps its size)
    }
    lane_ctx_ = lane_ctx;
    lane_kv_.resize(n_lanes - 1);
    lane_dn_.resize(n_lanes - 1);
    for (uint32_t l = 0; l + 1 < n_lanes; ++l) {
        lane_kv_[l].resize(n_dev_);
        lane_dn_[l].resize(n_dev_);
        for (uint32_t dev = 0; dev < n_dev_; ++dev) {
            uint64_t got = 0;
            if (kv_[dev].ready()) {
                KvCacheConfig kc = kv_[dev].config();
                kc.max_ctx = lane_ctx;
                if (auto m = lane_kv_[l][dev].init(fleet_->dev(dev), kc); !m.empty())
                    return "lane " + std::to_string(l + 1) + " kv card " + std::to_string(dev) + ": " + m;
                got += 2 * lane_kv_[l][dev].bytes_per_layer() * kc.n_layers_full;
            }
            if (dn_[dev].ready()) {
                auto& d = lane_dn_[l][dev];
                if (auto m = d.init(fleet_->dev(dev), dn_[dev].config()); !m.empty())
                    return "lane " + std::to_string(l + 1) + " deltanet card " + std::to_string(dev) + ": " + m;
                got += d.config().n_layers_linear * (d.state_elems_per_layer() * sizeof(float) + d.conv_elems_per_layer() * sizeof(sycl::half));
            }
            if (got != lane_bytes_[dev])
                return "request lanes: internal: card " + std::to_string(dev) + " allocated " + std::to_string(got) +
                       " bytes per lane, the arithmetic says " + std::to_string(lane_bytes_[dev]);
        }
    }
    lane_logits_ = static_cast<sycl::half*>(fleet_->dev(plan_.head_dev).malloc(uint64_t(cfg_.vocab) * sizeof(sycl::half)));
    if (!lane_logits_) return "request lanes: logits buffer alloc failed";
    owned_[plan_.head_dev].push_back(lane_logits_);
    // P4 B14 rows: the G-row int-dot activation per card and the G rows' logits (~80 KiB + ~7.9 MB; inside the reserve)
    const uint32_t Kmax = std::max(std::max(cfg_.hidden, cfg_.expert_ffn), std::max(cfg_.ssm_inner, cfg_.n_q_heads * cfg_.head_dim));
    act_rows_.assign(n_dev_, nullptr);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        void* p = fleet_->dev(dev).malloc(uint64_t(kMaxRows) * (Kmax / 32) * sizeof(block_q8_1x));
        if (!p) return "request lanes: rows activation alloc failed on card " + std::to_string(dev);
        owned_[dev].push_back(p);
        act_rows_[dev] = p;
    }
    rows_logits_ = static_cast<sycl::half*>(fleet_->dev(plan_.head_dev).malloc(uint64_t(kMaxRows) * cfg_.vocab * sizeof(sycl::half)));
    if (!rows_logits_) return "request lanes: rows logits buffer alloc failed";
    owned_[plan_.head_dev].push_back(rows_logits_);
    // P4 B26: the rows step's decode-attention partials, one slice per row (sized by lane 0's max_ctx, the largest cache),
    // and its per-group row table
    rows_partials_.assign(n_dev_, nullptr);
    rows_table_.assign(n_dev_, nullptr);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (!kv_[dev].ready()) continue;
        rows_partials_floats_ = fa2_decode_rows_partials_floats(cfg_.n_q_heads, cfg_.n_kv_heads, cfg_.head_dim, kv_[dev].config().max_ctx);
        void* p = fleet_->dev(dev).malloc(uint64_t(kMaxRows) * rows_partials_floats_ * sizeof(float));
        void* t = fleet_->dev(dev).malloc(kFaDecodeRowsTableBytes);
        if (!p || !t) return "request lanes: rows attention partials alloc failed on card " + std::to_string(dev);
        owned_[dev].push_back(p);
        owned_[dev].push_back(t);
        rows_partials_[dev] = static_cast<float*>(p);
        rows_table_[dev] = t;
    }
    return {};
}

void Qwen35MoeSplitModel::select_lane(uint32_t lane) {
    for (uint32_t dev = 0; dev < n_dev_; ++dev) cur_[dev] = lane;
}

void Qwen35MoeSplitModel::reset_state() {
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (dn_at(dev).ready()) dn_at(dev).reset(fleet_->dev(dev).queue());
        if (kv_at(dev).ready()) kv_at(dev).reset();
    }
}

// P4 B39: reset_state for `lane` by index (cur_ untouched: a card pipe stage may be running another lane on either card)
void Qwen35MoeSplitModel::reset_lane_state(uint32_t lane) {
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (lane_dn(lane, dev).ready()) lane_dn(lane, dev).reset(fleet_->dev(dev).queue());
        if (lane_kv(lane, dev).ready()) lane_kv(lane, dev).reset();
    }
}

std::string Qwen35MoeSplitModel::forward_stage(uint32_t dev, uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0,
                                               sycl::half* x_host) {
    static const char* kStopped = "stopped at a layer boundary: the engine is stopping";   // (P4 B33)
    if (T == 0) return "T == 0";
    if (aborting(T)) return kStopped;
    if (lane >= n_lanes() || n_dev_ != 2 || dev >= 2 || !lane_logits_) return "forward_stage: no such lane/card (lanes not initialised?)";
    if (vision_active()) return "forward_stage: vision is staged but the lanes have no per-lane image state (P4 B45 step 3)";
    if (auto m = ensure_ws(dev, T); !m.empty()) return m;
    cur_[dev] = lane;
    auto& q = fleet_->dev(dev).queue();
    Workspace& w = ws_[dev];
    // forward()'s per-card order: positions, the reset at pos 0, the embedding (card 0) or the residual in (card 1), the
    // layers, then the residual out (card 0) or the head (card 1)
    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(pos0 + t);
    if (w.positions) q.memcpy(w.positions, pos.data(), T * sizeof(int32_t)).wait();
    if (pos0 == 0) {
        if (dn_at(dev).ready()) dn_at(dev).reset(q);
        if (kv_at(dev).ready()) kv_at(dev).reset();
    }
    const uint64_t xb = uint64_t(T) * cfg_.hidden * sizeof(sycl::half);
    if (dev == plan_.embed_dev) { if (auto m = embed(ids, T, pos0); !m.empty()) return m; }
    else q.memcpy(w.x, x_host, xb).wait();
    double t_attn = 0.0, t_moe = 0.0;
    run_layers(dev, T, pos0, false, t_attn, t_moe);
    if (aborting(T)) { q.wait(); return kStopped; }   // (P4 B33: nothing of the step still runs on the card)
    if (dev != plan_.head_dev) { q.memcpy(x_host, w.x, xb).wait(); return {}; }
    head(T, lane_logits_);
    return {};
}


// ---- P4 B14 phase 1b: a rows-mode group's decode step (G lanes, one row each) ------------------------------------------

std::string Qwen35MoeSplitModel::rows_off_reason() const {
    if (lane_kv_.empty() || act_rows_.size() != n_dev_ || !rows_logits_) return "the request lanes are not initialised";
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (kv_[dev].ready() && kv_[dev].is_int8()) return "--int8-kv";
    if (!output_.int_dot()) return "the head is not Q8_0/Q6_K/Q5_K";
    if (cfg_.n_experts != 256 || cfg_.experts_topk != 8)   // (B14 gate #4: moe_router_rows' shape; refuse at load, not per group)
        return "router shape " + std::to_string(cfg_.n_experts) + " experts top-" + std::to_string(cfg_.experts_topk) + " (rows: 256 / top-8 only)";
    for (uint32_t L = 0; L < layers_.size(); ++L) {
        const LayerW& l = layers_[L];
        const std::string at = " (layer " + std::to_string(L) + ")";
        if (l.is_linear) {
            if (!l.attn_qkv.int_dot() || !l.attn_gate.int_dot() || !l.ssm_out.int_dot())
                return "a DeltaNet projection is not Q8_0/Q6_K/Q5_K" + at;
        } else if (!l.attn_q.int_dot() || !l.attn_k.int_dot() || !l.attn_v.int_dot() || !l.attn_output.int_dot()) {
            return "an attention projection is not Q8_0/Q6_K/Q5_K" + at;
        }
        if (!l.exp_gate.c32.lo &&
            (l.exp_gate.dtype != DType::kQ8_0 || l.exp_up.dtype != DType::kQ8_0 || l.exp_down.dtype != DType::kQ8_0))
            return "native K-quant experts (single-token decode kernels)" + at;
        if (l.ffn_gate_inp_shexp && (!l.sh_gate.int_dot() || !l.sh_up.int_dot() || !l.sh_down.int_dot()))
            return "a shared-expert projection is not Q8_0/Q6_K/Q5_K" + at;
    }
    return {};
}

sycl::event Qwen35MoeSplitModel::srows(uint32_t dev, const sycl::half* A, const SplitW& w, sycl::half* out, uint32_t K,
                                       uint32_t N, uint32_t G) {
    if (w.kq.lo) return kq_gemv(dev, act_rows_[dev], A, w, out, K, N, G);   // P4 B21
    auto& q = fleet_->dev(dev).queue();
    quantize_q8_1(q, A, act_rows_[dev], G * K);
    return gemv_q8_0_soa_q8_batched_sel(q, act_rows_[dev], w.q8_qs, w.q8_d, out, K, N, G);
}

// P4 B21: the c32 GEMV over T rows of A (one kernel template for T = 1 and the G rows: a row's value does not depend on T)
sycl::event Qwen35MoeSplitModel::kq_gemv(uint32_t dev, void* act, const sycl::half* A, const SplitW& w, sycl::half* out,
                                         uint32_t K, uint32_t N, uint32_t T) {
    auto& q = fleet_->dev(dev).queue();
    DPE(dev, kQuant, quantize_q8_1(q, A, act, T * K));
    sycl::event e;
    if (w.kq.kind == 6)
        e = gemv_q6k_c32_q8(q, act, w.kq.lo, static_cast<const uint8_t*>(w.kq.hi), static_cast<const int8_t*>(w.kq.sc),
                            static_cast<const uint16_t*>(w.kq.d), out, K, N, T);
    else
        e = gemv_q5k_c32_q8(q, act, w.kq.lo, static_cast<const uint32_t*>(w.kq.hi), static_cast<const uint16_t*>(w.kq.sc),
                            static_cast<const uint32_t*>(w.kq.d), out, K, N, T);
    if (g_dp35.active()) g_dp35.tag(dev, e, DecodeProf::kGemv);
    return e;
}

// run_layers' T == 1 decode path for G rows (row i = lanes[i] at pos0[i]); run_layers is not touched. The shared kernels run
// once over the G rows; the per-lane state (conv, DeltaNet recurrence, KV) is addressed by lane index, never through cur_.
void Qwen35MoeSplitModel::run_layers_rows(uint32_t dev, std::span<const uint32_t> lanes, const uint32_t* pos0) {
    const uint32_t G    = uint32_t(lanes.size());
    const uint32_t H    = cfg_.hidden;
    const uint32_t HD   = cfg_.head_dim;
    const uint32_t N_q  = cfg_.n_q_heads * HD;
    const uint32_t N_qg = N_q * 2u;
    const uint32_t N_kv = cfg_.n_kv_heads * HD;
    const uint32_t EFF  = cfg_.expert_ffn;
    const uint32_t E    = cfg_.n_experts;
    const uint32_t Kt   = cfg_.experts_topk;
    const uint32_t rope_n = cfg_.rope_dim;
    const float    eps  = cfg_.rms_eps;
    const uint32_t n_kv = cfg_.n_kv_heads;
    Workspace& w = ws_[dev];
    auto& q = fleet_->dev(dev).queue();
    // P4 B26: the G rows' decode attention as one launch per pass per kernel kind (IE_Q35_ROWS_ATTN=0: the per-row loop
    // below). The plan is built once per group: row i keeps the kernel the loop would give it (XMX / the dispatcher) on
    // its own lane's cache, so the bytes are the loop's; each layer runs from the plan with its KV slot.
    FaDecodeRowsPlan rows_plan;
    bool rows_on = false;
    if (q35::rows_attn() && dev < rows_partials_.size() && rows_partials_[dev] && rows_table_[dev]) {
        FaDecodeRow rows[kMaxRows];
        for (uint32_t i = 0; i < G; ++i) {
            const KvCache& kvc = lane_kv(lanes[i], dev);
            const uint32_t max_ctx = kvc.config().max_ctx;
            rows[i].k_base = kvc.k_ptr();
            rows[i].v_base = kvc.v_ptr();
            rows[i].layer_stride = uint64_t(n_kv) * max_ctx * HD;
            rows[i].start_pos = pos0[i];
            rows[i].max_ctx = max_ctx;
            rows[i].kind = q35::xmx_decode(pos0[i], max_ctx) ? FaDecodeKind::kXmx : FaDecodeKind::kAuto;
        }
        rows_on = fa2_decode_rows_plan(q, rows, G, cfg_.n_q_heads, cfg_.n_kv_heads, HD, rows_partials_floats_, rows_table_[dev], rows_plan);
        static const bool logged = [&] {
            std::fprintf(stderr, "[qwen35moe] rows decode attention: %s (P4 B26: one launch per pass per kernel kind over the "
                         "group's rows; IE_Q35_ROWS_ATTN=0 = the per-row loop)\n", rows_on ? "batched" : "per-row (plan not batched)");
            return true;
        }();
        (void)logged;
    }

    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        if (plan_.dev_of_layer[L] != dev) continue;
        const LayerW& w_l = layers_[L];
        rms_norm_f32w(q, w.x, w_l.attn_norm, w.x_normed, G, H, eps);
        if (w_l.is_linear) {
            const uint32_t SKH = cfg_.ssm_n_k_heads;
            const uint32_t SVH = cfg_.ssm_n_v_heads;
            const uint32_t SHD = cfg_.ssm_head_dim;
            const uint32_t SI  = cfg_.ssm_inner;
            const uint32_t conv_ch = SI + 2u * SKH * SHD;
            const uint32_t kw  = SKH * SHD;
            const uint32_t rep = SVH / SKH;
            const uint32_t Vd  = SVH * SHD;
            const float qscale = 1.0f / sycl::sqrt(float(SHD));
            const uint32_t SVHp = ((SVH + 63u) / 64u) * 64u;

            srows(dev, w.x_normed, w_l.attn_qkv, w.dn_qkv, H, conv_ch, G);
            for (uint32_t i = 0; i < G; ++i) {
                DeltaNetState& d = lane_dn(lanes[i], dev);
                sycl::half* conv_state = d.conv_state_ptr() + uint64_t(dn_local_[L]) * d.conv_elems_per_layer();
                depthwise_conv1d_causal(q, w.dn_qkv + uint64_t(i) * conv_ch, w_l.ssm_conv1d_fp16, conv_state,
                                        w.dn_conv + uint64_t(i) * conv_ch, 1, conv_ch, cfg_.ssm_conv_kernel);
            }
            cast_qkv_split_fp16_to_fp32(q, w.dn_conv, w.dn_qpre, w.dn_kpre, w.dn_vpre, G, kw, SI);
            l2_norm_scale(q, w.dn_qpre, w.dn_qpre, G * SKH, SHD, qscale, 1e-6f);
            l2_norm_scale(q, w.dn_kpre, w.dn_kpre, G * SKH, SHD, 1.0f,   1e-6f);
            repeat_interleave_heads(q, w.dn_qpre, w.dn_qrep, G, SKH, SHD, rep);
            repeat_interleave_heads(q, w.dn_kpre, w.dn_krep, G, SKH, SHD, rep);
            // alpha/beta per ROW: the G-row F16 route is another kernel (oneDNN / XMX gemm), not the T == 1 gemv
            if (w_l.ssm_alpha.dt == DType::kF16 && w_l.ssm_beta.dt == DType::kF16) {
                // P4 B22: every row and both mats in ONE launch, the T = 1 decode's kernel (gemv_fp16_rows_dual)
                gemv_fp16_rows_dual(q, w.x_normed, H, static_cast<const sycl::half*>(w_l.ssm_alpha.p), w.dn_alpha64,
                                    static_cast<const sycl::half*>(w_l.ssm_beta.p), w.dn_beta64, SVHp, H, SVHp, G);
            } else
            for (uint32_t i = 0; i < G; ++i) {
                dense::gemv_q_T(q, w.x_normed + uint64_t(i) * H, w_l.ssm_alpha, w.dn_alpha64 + uint64_t(i) * SVHp, H, SVHp, 1);
                dense::gemv_q_T(q, w.x_normed + uint64_t(i) * H, w_l.ssm_beta,  w.dn_beta64  + uint64_t(i) * SVHp, H, SVHp, 1);
            }
            extract_cols(q, w.dn_alpha64, w.dn_alpha_h, G, SVH, SVHp);
            extract_cols(q, w.dn_beta64,  w.dn_beta_h,  G, SVH, SVHp);
            compute_g_beta_h16(q, w.dn_alpha_h, w.dn_beta_h, w_l.ssm_a, w_l.ssm_dt_bias, w.dn_g, w.dn_beta, G, SVH);
            for (uint32_t i = 0; i < G; ++i) {
                DeltaNetState& d = lane_dn(lanes[i], dev);
                float* state_layer = d.state_ptr() + uint64_t(dn_local_[L]) * d.state_elems_per_layer();
                deltanet_recurrence(q, w.dn_qrep + uint64_t(i) * Vd, w.dn_krep + uint64_t(i) * Vd, w.dn_vpre + uint64_t(i) * SI,
                                    w.dn_g + uint64_t(i) * SVH, w.dn_beta + uint64_t(i) * SVH, state_layer,
                                    w.dn_out + uint64_t(i) * Vd, /*B=*/1, /*T=*/1, SVH, SHD, SHD);
            }
            srows(dev, w.x_normed, w_l.attn_gate, w.dn_z, H, SI, G);
            gated_rms_norm(q, w.dn_out, w.dn_z, w_l.ssm_norm_fp16, w.dn_qkv, G * SVH, SHD, eps);
            srows(dev, w.dn_qkv, w_l.ssm_out, w.attn_block, SI, H, G);
        } else {
            srows(dev, w.x_normed, w_l.attn_q, w.qg, H, N_qg, G);
            split_q_gate_per_head(q, w.qg, w.q, w.gate, G, cfg_.n_q_heads, HD);
            srows(dev, w.x_normed, w_l.attn_k, w.k, H, N_kv, G);
            srows(dev, w.x_normed, w_l.attn_v, w.v, H, N_kv, G);
            rms_norm_f32w(q, w.q, w_l.attn_q_norm, w.q, G * cfg_.n_q_heads,  HD, eps);
            rms_norm_f32w(q, w.k, w_l.attn_k_norm, w.k, G * cfg_.n_kv_heads, HD, eps);
            rope_partial(q, w.q, w.positions, w.q, G, cfg_.n_q_heads,  HD, rope_n, cfg_.rope_theta);   // row i at pos0[i]
            rope_partial(q, w.k, w.positions, w.k, G, cfg_.n_kv_heads, HD, rope_n, cfg_.rope_theta);
            const uint32_t li = kv_local_[L];
            if (rows_on) {   // P4 B26 (the plan above)
                full_attention_fa2_decode_rows(q, rows_plan, w.q, w.k, w.v, w.attn_out, rows_partials_[dev], li);
                for (uint32_t i = 0; i < G; ++i) lane_kv(lanes[i], dev).set_length(li, pos0[i] + 1);
            } else
            for (uint32_t i = 0; i < G; ++i) {
                KvCache& kvc = lane_kv(lanes[i], dev);
                const uint32_t max_ctx = kvc.config().max_ctx;           // the lane's own capacity
                const uint64_t per_layer_kv = uint64_t(n_kv) * max_ctx * HD;
                if (q35::xmx_decode(pos0[i], max_ctx))   // P4 B23: per row, the T == 1 decode's choice
                    full_attention_fa2_decode_xmx(q, w.q + uint64_t(i) * N_q, w.k + uint64_t(i) * N_kv, w.v + uint64_t(i) * N_kv,
                                              kvc.k_ptr() + per_layer_kv * li, kvc.v_ptr() + per_layer_kv * li,
                                              w.attn_out + uint64_t(i) * N_q, w.attn_partials, pos0[i],
                                              cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx);
                else
                    full_attention_fa2_decode(q, w.q + uint64_t(i) * N_q, w.k + uint64_t(i) * N_kv, w.v + uint64_t(i) * N_kv,
                                          kvc.k_ptr() + per_layer_kv * li, kvc.v_ptr() + per_layer_kv * li,
                                          w.attn_out + uint64_t(i) * N_q, w.attn_partials, pos0[i],
                                          cfg_.n_q_heads, cfg_.n_kv_heads, HD, max_ctx);
                kvc.set_length(li, pos0[i] + 1);
            }
            sigmoid_gate(q, w.attn_out, w.gate, w.attn_out, uint64_t(G) * N_q);
            srows(dev, w.attn_out, w_l.attn_output, w.attn_block, N_q, H, G);
        }
        residual_add_rms_norm_fused(q, w.x, w.attn_block, w_l.post_attn_norm, w.x_normed, G, H, eps);

        // MoE: always the Q8_0 decode chain over the G rows (never the prefill expert-GEMM branch)
        moe_router_rows(q, w.x_normed, w_l.ffn_gate_inp, w.topk_idx, w.topk_w, G, H, E, Kt);   // == the T == 1 router per row
        quantize_q8_1(q, w.x_normed, static_cast<block_q8_1x*>(w.x_q8), uint32_t(G) * H);
        if (w_l.exp_gate.c32.lo) {   // P4 B21: the same per-slot c32 kernels as the T == 1 decode
            moe_gate_up_silu_c32(q, w.x_q8, w_l.exp_gate.c32, w_l.exp_up.c32, w.topk_idx, w.moe_hp, G, Kt, H, EFF);
            quantize_q8_1(q, w.moe_hp, static_cast<block_q8_1x*>(w.h_q8), uint32_t(G) * Kt * EFF);
            moe_down_c32(q, w.h_q8, w_l.exp_down.c32, w.topk_idx, w.topk_w, w.moe_yp, G, Kt, EFF, H);
        } else {
        moe_gate_up_silu_q8(q, w.x_q8, w_l.exp_gate.qs, w_l.exp_gate.d,
                            w_l.exp_up.qs, w_l.exp_up.d, w_l.exp_gate.qs_stride,
                            w_l.exp_gate.d_stride, w.topk_idx, w.moe_hp, G, Kt, H, EFF);
        quantize_q8_1(q, w.moe_hp, static_cast<block_q8_1x*>(w.h_q8), uint32_t(G) * Kt * EFF);
        moe_down_q8(q, w.h_q8, w_l.exp_down.qs, w_l.exp_down.d, w_l.exp_down.qs_stride,
                    w_l.exp_down.d_stride, w.topk_idx, w.topk_w, w.moe_yp, G, Kt, EFF, H);
        }
        moe_reduce_q8(q, w.moe_yp, w.moe_y, G, Kt, H);
        if (w_l.ffn_gate_inp_shexp) {
            srows(dev, w.x_normed, w_l.sh_gate, w.gate_o, H, EFF, G);
            srows(dev, w.x_normed, w_l.sh_up,   w.up_o,   H, EFF, G);
            swiglu(q, w.gate_o, w.up_o, w.ffn_h, uint64_t(G) * EFF);
            srows(dev, w.ffn_h, w_l.sh_down, w.eo, EFF, H, G);
            {   // run_layers' shared-expert gate, verbatim (row-wise)
                float* dng = w.dn_g; const float* gw = w_l.ffn_gate_inp_shexp;
                const sycl::half* xn = w.x_normed; const sycl::half* eo = w.eo;
                sycl::half* my = w.moe_y; const uint32_t HH = H;
                shexp_gate_serial(q, xn, gw, dng, G, HH);   // P4 B22
                q.parallel_for(sycl::range<1>(uint64_t(G) * HH), [=](sycl::id<1> i) {
                    const uint32_t t = uint32_t(uint64_t(i) / HH);
                    my[i] = sycl::half(float(my[i]) + float(eo[i]) * dng[t]);
                });
            }
        }
        residual_add(q, w.x, w.moe_y, w.x, uint64_t(G) * H);
    }
}

std::string Qwen35MoeSplitModel::forward_stage_rows(uint32_t dev, std::span<const uint32_t> lanes, const int32_t* ids,
                                                    const uint32_t* pos0, sycl::half* x_host) {
    const uint32_t G = uint32_t(lanes.size());
    if (G < 2 || G > kMaxRows) return "forward_stage_rows: " + std::to_string(G) + " rows (a group takes 2.." + std::to_string(kMaxRows) + ")";
    if (n_dev_ != 2 || dev >= 2 || act_rows_.size() != n_dev_ || !rows_logits_)
        return "forward_stage_rows: no such card, or the rows buffers are not initialised";
    for (uint32_t i = 0; i < G; ++i) {
        if (lanes[i] >= n_lanes()) return "forward_stage_rows: lane " + std::to_string(lanes[i]) + " of " + std::to_string(n_lanes());
        if (pos0[i] == 0) return "forward_stage_rows: lane " + std::to_string(lanes[i]) + " at position 0 (a group row never starts a sequence)";
    }
    if (dev == plan_.embed_dev && !ids) return "forward_stage_rows: no ids";
    if (vision_active()) return "forward_stage_rows: vision is staged but the lanes have no per-lane image state (P4 B45 step 3)";
    if (auto m = ensure_ws(dev, G); !m.empty()) return m;
    auto& q = fleet_->dev(dev).queue();
    Workspace& w = ws_[dev];
    if (kv_[dev].ready() && !w.attn_partials) return "forward_stage_rows: no decode attention scratch on card " + std::to_string(dev);
    int32_t pos[kMaxRows];
    for (uint32_t i = 0; i < G; ++i) pos[i] = int32_t(pos0[i]);
    q.memcpy(w.positions, pos, G * sizeof(int32_t)).wait();
    const uint64_t xb = uint64_t(G) * cfg_.hidden * sizeof(sycl::half);
    // (the G rows are one decode row each of different lanes: no piece position; the splice is a no-op -- guarded above)
    if (dev == plan_.embed_dev) { if (auto m = embed(ids, G, /*start_pos=*/0); !m.empty()) return m; }
    else q.memcpy(w.x, x_host, xb).wait();
    run_layers_rows(dev, lanes, pos0);
    if (dev != plan_.head_dev) { q.memcpy(x_host, w.x, xb).wait(); return {}; }
    // head(): the final norm, then the lm_head over the G rows (batched int-dot; row i == the one-row sgemv's)
    const uint32_t H = cfg_.hidden;
    rms_norm_f32w(q, w.x, output_norm_, w.x_normed, G, H, cfg_.rms_eps);
    srows(dev, w.x_normed, output_, rows_logits_, H, cfg_.vocab, G).wait();   // (Q8_0: the quantize + batched int-dot as before)
    return {};
}

}  // namespace ie
