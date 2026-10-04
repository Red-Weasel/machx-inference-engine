// src/model/qwen35_split.cpp — Qwen3.6-27B (`kQwen35Dense`) multi-GPU layer-split.
// See header. ADDITIVE: mirrors the 80B fleet scaffold (qwen3next.cpp) with the
// validated single-GPU 27B per-layer math (qwen35_dense.cpp) lifted verbatim into
// a device-by-device loop. The single-GPU Qwen35DenseModel is NEVER edited.

#include "ie/qwen35_split.hpp"

#include "ie/dequant.hpp"
#include "ie/decode_prof.hpp"      // P4 B22: IE_DECODE_PROF per-token decode breakdown
#include "ie/gemv_q8_soa_v2.hpp"   // P4 B37: the v2 int-dot GEMV selectors
#include "ie/kernel_profiler.hpp"   // ie::ps (named-kernel submit)
#include "ie/ops.hpp"
#include "ie/q35m_lanes.hpp"       // Q35mLaneShape / q35m_lane_bytes / q35m_lanes_fit (P4 B18 lanes)
#include "ie/quant_blocks.hpp"

#include "dense_dispatch.hpp"   // dense::upload<T>, dense::upload_quant_dense_auto, dense::gemv_q/gemv_q_T
#include "q35_xmx_decode.hpp"   // P4 B23: IE_Q35_XMX_DECODE XMX decode attention at long context (default on, =0 off)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace ie {
namespace {

DecodeProf g_dp27("27B qwen35split");   // P4 B22 (off unless IE_DECODE_PROF=1)

// IE_QWEN35_PFPROF=1 (P4 B48): where a pipelined prefill's time goes on each card. Every mark waits on the card's
// queue, so the two cards no longer overlap: read it as shares, not as a speed.
struct PfProf27 {
    enum Kind { kExpand, kGemm, kAttn, kDnScan, kDnMisc, kElem, kCount };
    using clk = std::chrono::steady_clock;
    const bool on = std::getenv("IE_QWEN35_PFPROF") != nullptr;
    bool armed[2] = {false, false};
    double ms[2][kCount] = {};
    clk::time_point last[2];
    void start(uint32_t dev, sycl::queue& q) {
        if (!on || dev > 1) return;
        q.wait(); last[dev] = clk::now(); armed[dev] = true;
    }
    void mark(uint32_t dev, sycl::queue& q, Kind k) {
        if (!on || dev > 1 || !armed[dev]) return;
        q.wait();
        const auto n = clk::now();
        ms[dev][k] += std::chrono::duration<double, std::milli>(n - last[dev]).count();
        last[dev] = n;
    }
    void stop(uint32_t dev) { if (dev <= 1) armed[dev] = false; }
    void report(uint32_t tokens) {
        if (!on) return;
        static const char* names[kCount] = {"expand to fp16", "matmul", "attention", "DeltaNet scan", "DeltaNet other",
                                            "norms + elementwise"};
        for (uint32_t dev = 0; dev < 2; ++dev) {
            double tot = 0;
            for (double v : ms[dev]) tot += v;
            std::fprintf(stderr, "[qwen35split PFPROF] card %u: %u tokens, %.1f ms (wait-bracketed)\n", dev, tokens, tot);
            for (int k = 0; k < kCount; ++k)
                std::fprintf(stderr, "  %-20s %9.1f ms  %5.1f%%\n", names[k], ms[dev][k], tot > 0 ? 100.0 * ms[dev][k] / tot : 0.0);
            for (double& v : ms[dev]) v = 0;
        }
    }
};
PfProf27 g_pf27;
#define DPE(DEV, C, ...) do { sycl::event dpe_ = (__VA_ARGS__); if (g_dp27.active()) g_dp27.tag((DEV), dpe_, DecodeProf::C); } while (0)

// Host-side canonical Q6_K row dequant — mirrors ggml's dequantize_row_q6_K
// (natural element order). COPY of qwen35_dense.cpp:dequant_q6_K_row
// (copy-not-hoist discipline). Load-time only; K % 256 == 0.
static void dequant_q6_K_row(const block_q6_K* blocks, float* out, uint64_t K) {
    const uint64_t nb = K / kQK_K;
    for (uint64_t i = 0; i < nb; ++i) {
        const float    d  = fp16_to_fp32(blocks[i].d);
        const uint8_t* ql = blocks[i].ql;
        const uint8_t* qh = blocks[i].qh;
        const int8_t*  sc = blocks[i].scales;
        float*         y  = out + i * kQK_K;
        for (int half = 0; half < kQK_K; half += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = int((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = int((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = int((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * float(sc[is + 0]) * float(q1);
                y[l + 32] = d * float(sc[is + 2]) * float(q2);
                y[l + 64] = d * float(sc[is + 4]) * float(q3);
                y[l + 96] = d * float(sc[is + 6]) * float(q4);
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

// Natural-order Q4_K row dequant (ggml dequantize_row_q4_K) for the Q4_K-quantized
// ssm_alpha/ssm_beta projections in *-Q4_K_M GGUFs (this split loader previously only
// took F32/Q8_0/Q6_K, so any Q4_K_M qwen35 failed to load on >1 GPU). COPY of
// qwen35_dense.cpp:dequant_q4_K_row (copy-not-hoist discipline). Load-time only; K % 256 == 0.
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

// ssm_alpha / ssm_beta projections are F32 (or Q8_0) [K=5120, N=48] in the GGUF
// (ggml shape[0]=K → [N,K] row-major). Transpose to [K,Npad] (cols ≥N zero-padded,
// Npad=64) + cast fp16 so they ride the BATCHED gemm_fp16 path. COPY of
// qwen35_dense.cpp:upload_f32_proj_fp16 (copy-not-hoist discipline).
DenseQuantPtr upload_f32_proj_fp16(DeviceAllocator& alloc, const GgufTensorInfo* t,
                                   std::vector<void*>& owned, std::string& err,
                                   uint32_t Npad) {
    DenseQuantPtr out;
    if (!t) { err = "tensor not found"; return out; }
    if (t->n_dims != 2 || (t->dtype != DType::kF32 && t->dtype != DType::kQ8_0 &&
                           t->dtype != DType::kQ6_K && t->dtype != DType::kQ5_K && t->dtype != DType::kQ4_K)) {
        err = "ssm proj: expected F32, Q8_0, Q6_K, Q5_K, or Q4_K 2-D"; return out;
    }
    const uint64_t K = t->shape[0];   // 5120 (in)
    const uint64_t N = t->shape[1];   // 48   (out)
    if (Npad < N) Npad = uint32_t(N);
    std::vector<sycl::half> staging(K * Npad, sycl::half(0.0f));   // zero-padded cols ≥N
    if (t->dtype == DType::kF32) {
        const float* src = reinterpret_cast<const float*>(t->data);
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k)
                staging[k * Npad + n] = sycl::half(src[n * K + k]);   // [N,K] → [K,Npad]
    } else if (t->dtype == DType::kQ8_0) {  // dequant on the fly
        const uint64_t bpr = K / 32;
        const auto* blocks = reinterpret_cast<const block_q8_0*>(t->data);
        for (uint64_t n = 0; n < N; ++n)
            for (uint64_t k = 0; k < K; ++k) {
                const block_q8_0& b = blocks[n * bpr + (k >> 5)];
                staging[k * Npad + n] = sycl::half(fp16_to_fp32(b.d) * float(b.qs[k & 31]));
            }
    } else if (t->dtype == DType::kQ6_K) {  // Q6_K: K-quant ssm proj (e.g. *-Q6_K GGUFs). Dequant per row, then transpose.
        if (K % kQK_K != 0) { err = "ssm proj Q6_K: K not a multiple of 256"; return out; }
        const uint64_t bpr = K / kQK_K;
        const auto* blocks = reinterpret_cast<const block_q6_K*>(t->data);
        std::vector<float> row(K);
        for (uint64_t n = 0; n < N; ++n) {
            dequant_q6_K_row(blocks + n * bpr, row.data(), K);
            for (uint64_t k = 0; k < K; ++k)
                staging[k * Npad + n] = sycl::half(row[k]);   // [N,K] → [K,Npad]
        }
    } else if (t->dtype == DType::kQ5_K) {  // Q5_K: the Q5_K_M 27B's ssm proj (P4 B21). ggml's dequant per row, then transpose.
        if (K % kQK_K != 0) { err = "ssm proj Q5_K: K not a multiple of 256"; return out; }
        std::vector<float> row(K);
        for (uint64_t n = 0; n < N; ++n) {
            ie::ref::dequant_q5_K_buffer(static_cast<const uint8_t*>(t->data) + n * (K / kQK_K) * sizeof(block_q5_K), K, row.data());
            for (uint64_t k = 0; k < K; ++k)
                staging[k * Npad + n] = sycl::half(row[k]);   // [N,K] → [K,Npad]
        }
    } else {  // Q4_K: K-quant ssm proj (e.g. *-Q4_K_M GGUFs). Dequant per row, then transpose.
        if (K % kQK_K != 0) { err = "ssm proj Q4_K: K not a multiple of 256"; return out; }
        const uint64_t bpr = K / kQK_K;
        const auto* blocks = reinterpret_cast<const block_q4_K*>(t->data);
        std::vector<float> row(K);
        for (uint64_t n = 0; n < N; ++n) {
            dequant_q4_K_row(blocks + n * bpr, row.data(), K);
            for (uint64_t k = 0; k < K; ++k)
                staging[k * Npad + n] = sycl::half(row[k]);   // [N,K] → [K,Npad]
        }
    }
    void* d = alloc.malloc(K * Npad * sizeof(sycl::half));
    if (!d) { err = "malloc failed (ssm proj)"; return out; }
    alloc.queue().memcpy(d, staging.data(), K * Npad * sizeof(sycl::half)).wait();
    owned.push_back(d);
    out.p = d; out.dt = DType::kF16;
    return out;
}

// Strided extract: dst[t,h]=src[t,h], src row stride=src_stride. COPY of
// qwen35_dense.cpp:extract_cols (compacts the N-padded [T,64] gemm out → [T,48]).
inline sycl::event extract_cols(sycl::queue& q, const sycl::half* src, sycl::half* dst,
                                uint32_t T, uint32_t nh, uint32_t src_stride) {
    return q.parallel_for(sycl::range<1>(uint64_t(T) * nh), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(i) / nh, h = uint32_t(i) % nh;
        dst[uint64_t(t) * nh + h] = src[uint64_t(t) * src_stride + h];
    });
}

// Dequant a SoA-Q8_0 weight (qs[n*K+k] int8, d[n*(K/32)+b] fp16) → fp16 Bt[K,N]
// (row-major, B for gemm_fp16: Bt[k*N+n]). Used on prefill so the packed weight
// rides the existing batched gemm. q is the owning card's queue.
//
// 32×32 SLM tile transpose (2026-08-15): the original per-element kernel read qs
// coalesced but WROTE Bt[k*N+n] as 2-byte stores at stride N — one cache-line
// transaction per element, the dominant cost of every prefill chunk (the whole
// weight is re-materialized per chunk). Tiling makes BOTH sides coalesced: reads
// stream 32 consecutive k per row, writes stream 32 consecutive n per row. Same
// per-element math → bit-identical Bt.
inline sycl::event dequant_q8_0_soa_to_Bt(sycl::queue& q, const int8_t* qs,
                                          const uint16_t* d, sycl::half* Bt,
                                          uint32_t K, uint32_t N) {
    const uint32_t bpc = K / 32;
    constexpr uint32_t TS = 32;                       // tile edge
    const uint32_t ntn = (N + TS - 1) / TS, ntk = (K + TS - 1) / TS;
    return ie::ps(q, "dequant_q8_soa_bt", [&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 2> tile(sycl::range<2>(TS, TS + 1), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(uint64_t(ntn) * 8, uint64_t(ntk) * TS),
                                         sycl::range<2>(8, TS)),
                       [=](sycl::nd_item<2> it) {
            const uint32_t ly = uint32_t(it.get_local_id(0));   // 0..7
            const uint32_t lx = uint32_t(it.get_local_id(1));   // 0..31
            const uint32_t n0 = uint32_t(it.get_group(0)) * TS;
            const uint32_t k0 = uint32_t(it.get_group(1)) * TS;
            #pragma unroll
            for (uint32_t r = 0; r < 4; ++r) {                  // read: coalesced along k
                const uint32_t n = n0 + ly + r * 8, k = k0 + lx;
                if (n < N && k < K) {
                    const float dv = float(sycl::bit_cast<sycl::half>(d[uint64_t(n) * bpc + (k >> 5)]));
                    tile[ly + r * 8][lx] = sycl::half(float(qs[uint64_t(n) * K + k]) * dv);
                }
            }
            sycl::group_barrier(it.get_group());
            #pragma unroll
            for (uint32_t r = 0; r < 4; ++r) {                  // write: coalesced along n
                const uint32_t k = k0 + ly + r * 8, n = n0 + lx;
                if (n < N && k < K)
                    Bt[uint64_t(k) * N + n] = tile[lx][ly + r * 8];
            }
        });
    });
}

}  // namespace

Qwen35SplitModel::~Qwen35SplitModel() { free_all(); }

void Qwen35SplitModel::free_all() {
    if (!fleet_) return;
    free_slot_banks();
    lane_kv_.clear(); lane_dn_.clear();   // (P4 B18 lanes; their logits buffers are in owned_)
    ss_ = SpecStep{};                     // (P4 B48: its checkpoints free through the fleet; its buffers are in owned_)
    vrows_partials_.clear(); vrows_table_.clear();
    lane_logits_ = rows_logits_ = nullptr;
    for (uint32_t d = 0; d < ws_.size(); ++d) free_ws(d);
    ws_.clear();
    for (uint32_t d = 0; d < owned_.size(); ++d)
        for (void* p : owned_[d]) if (p) fleet_->dev(d).free(p);
    owned_.clear();
    for (uint32_t d = 0; d < prefill_bt_.size(); ++d)   // not in owned_ (re-alloc on grow)
        if (prefill_bt_[d]) fleet_->dev(d).free(prefill_bt_[d]);
    for (uint32_t d = 0; d < prefill_dt_.size(); ++d)
        if (prefill_dt_[d]) fleet_->dev(d).free(prefill_dt_[d]);
    prefill_bt_.clear(); prefill_bt_cap_.clear(); act_q8_.clear();
    prefill_dt_.clear(); prefill_dt_cap_.clear();
    for (auto& s : dn_) s.free_storage();
    for (auto& c : kv_) c.free_storage();
}

std::string Qwen35SplitModel::load(DeviceFleet& fleet, const LayerPlan& plan,
                                   const GgufReader& g, const Qwen35Config& cfg,
                                   uint32_t max_ctx, bool int8_kv) {
    fleet_ = &fleet; plan_ = plan; cfg_ = cfg;
    n_dev_ = plan.n_dev();
    const DenseConfig& d = cfg.dense;

    if (n_dev_ == 0 || plan.dev_of_layer.empty()) return "qwen35split: empty plan";
    if (d.hidden == 0 || d.vocab == 0 || d.n_q_heads == 0 || d.n_kv_heads == 0 ||
        d.ffn == 0)                                return "qwen35split: zero dim";
    if (cfg.ssm_inner == 0 || cfg.ssm_n_v_heads == 0 || cfg.ssm_n_k_heads == 0 ||
        cfg.ssm_state == 0 || cfg.ssm_conv_kernel == 0)
        return "qwen35split: missing ssm dims";

    // oneDNN on the split prefill — DEFAULT ON since the 2026-08-15 A/B (Qwen3.8-27B
    // Q8_0, 2×B70): pp median 94.3 → 167.0 (+77%), long-ctx prompt 194 → 412; decode
    // neutral; PPL gate held. Only became visible after the tiled dequant fix — the
    // old stride-N dequant writes drowned the GEMM (which is why the first A/B was
    // flat). The DEVICE_LOST landmine is FIXED (gemm_onednn.cpp:ctx_for keeps one
    // engine/stream PER device, ie-onednn-multidev-test); the old "small-M MoE"
    // rationale was a copy-paste from the 80B split — THIS FFN is dense SwiGLU at
    // M=T, the regime where oneDNN won 1.65× single-GPU (authority §5 row 2).
    // Opt-out: IE_QWEN35_SPLIT_NO_ONEDNN=1 (kill switch → gemm_fp16+cast).
    dense::prefer_onednn() = std::getenv("IE_QWEN35_SPLIT_NO_ONEDNN") == nullptr;

    owned_.assign(n_dev_, {});
    dev_bytes_.assign(n_dev_, 0);

    char buf[64];
    auto Ttop = [&](const char* n) { return g.find_tensor(n); };
    auto Tl   = [&](uint32_t L, const char* n) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n); return g.find_tensor(buf);
    };
    std::string err;

    // Phase 2: build a SplitW from a GGUF matrix weight. Q8_0 → de-interleave AoS
    // block_q8_0 → SoA (int8 qs col-contiguous + raw fp16 d/32-block), kept PACKED
    // (no F16 doubling, bit-exact). Q6_K/Q5_K → the c32 repack (P4 B21, same bits).
    // Else (Q4_K packed, F16; Q6_K/Q5_K with IE_QWEN35_SPLIT_KQ=0) → `fp`
    // fallback via the existing auto-upload. `bytes` accumulates the chosen device
    // residency. SplitW is the model's nested type → this lambda lives in load().
    // P4 B21: IE_QWEN35_SPLIT_KQ=0 keeps the old dense fallbacks for Q6_K / Q5_K weights (Q6_K AoS, Q5_K -> F16) for A/B
    static const bool split_kq = [] { const char* v = std::getenv("IE_QWEN35_SPLIT_KQ"); return !v || v[0] != '0'; }();
    auto build_split = [](DeviceAllocator& a, const GgufTensorInfo* t,
                          std::vector<void*>& own, std::string& e, uint64_t& bytes) -> SplitW {
        SplitW w{};
        if (!t) { e = "tensor not found"; return w; }
        if (t->n_dims != 2) { e = "split weight: expected 2-D"; return w; }
        w.K = uint32_t(t->shape[0]); w.N = uint32_t(t->shape[1]);
        if (t->dtype == DType::kQ8_0) {
            const uint32_t K = w.K, N = w.N, bpc = K / 32;
            if (K % 32 != 0) { e = "Q8_0 SoA: K % 32 != 0"; return w; }
            const auto* blocks = reinterpret_cast<const block_q8_0*>(t->data);
            std::vector<int8_t>   qs(uint64_t(N) * K);
            std::vector<uint16_t> dd(uint64_t(N) * bpc);
            for (uint64_t n = 0; n < N; ++n)
                for (uint32_t b = 0; b < bpc; ++b) {
                    const block_q8_0& blk = blocks[n * bpc + b];
                    dd[n * bpc + b] = *reinterpret_cast<const uint16_t*>(&blk.d);
                    for (int i = 0; i < 32; ++i)
                        qs[n * K + uint64_t(b) * 32 + i] = blk.qs[i];
                }
            auto* dqs = static_cast<int8_t*>(a.malloc(qs.size()));
            auto* ddd = static_cast<uint16_t*>(a.malloc(dd.size() * sizeof(uint16_t)));
            if (!dqs || !ddd) { e = "Q8_0 SoA alloc"; return w; }
            own.push_back(dqs); own.push_back(ddd);
            a.queue().memcpy(dqs, qs.data(), qs.size()).wait();
            a.queue().memcpy(ddd, dd.data(), dd.size() * sizeof(uint16_t)).wait();
            w.q8_qs = dqs; w.q8_d = ddd;
            bytes += qs.size() + dd.size() * sizeof(uint16_t);
        } else if ((t->dtype == DType::kQ6_K || t->dtype == DType::kQ5_K) && w.K % 256 == 0 && split_kq) {
            // P4 B21: the c32 repack (gemv_kq_c32.cpp) -- the K-quant's own bytes, int-dot decode + rows, dequant prefill
            const uint64_t K = w.K, N = w.N;
            const bool q6 = t->dtype == DType::kQ6_K;
            // bytes: lo 4 bits/elem; hi Q6 2 bits, Q5 1 bit; sc Q6 int8/16 = Q5 u16/32; d Q6 fp16/256, Q5 u32/256
            const uint64_t n_lo = N * K / 2, n_hi = q6 ? N * K / 4 : N * K / 8, n_sc = N * K / 16, n_d = q6 ? N * K / 128 : N * K / 64;
            std::vector<uint8_t> lo(n_lo), hi(n_hi), sc(n_sc), dd(n_d);
            if (q6) repack_q6_K_to_c32(t->data, w.K, w.N, lo.data(), hi.data(), reinterpret_cast<int8_t*>(sc.data()),
                                       reinterpret_cast<uint16_t*>(dd.data()));
            else    repack_q5_K_to_c32(t->data, w.K, w.N, lo.data(), reinterpret_cast<uint32_t*>(hi.data()),
                                       reinterpret_cast<uint16_t*>(sc.data()), reinterpret_cast<uint32_t*>(dd.data()));
            void* p[4] = {a.malloc(n_lo), a.malloc(n_hi), a.malloc(n_sc), a.malloc(n_d)};
            for (void* x : p) if (x) own.push_back(x);   // freed with the model even when one allocation failed
            if (!p[0] || !p[1] || !p[2] || !p[3]) { e = "K-quant c32 alloc"; return w; }
            a.queue().memcpy(p[0], lo.data(), n_lo).wait();   // one copy at a time, as the Q8_0 branch does
            a.queue().memcpy(p[1], hi.data(), n_hi).wait();
            a.queue().memcpy(p[2], sc.data(), n_sc).wait();
            a.queue().memcpy(p[3], dd.data(), n_d).wait();
            w.kq = t->dtype;
            w.kq_lo = static_cast<uint8_t*>(p[0]); w.kq_hi = p[1]; w.kq_sc = p[2]; w.kq_d = p[3];
            bytes += n_lo + n_hi + n_sc + n_d;
        } else {
            w.fp = dense::upload_quant_dense_auto(a, t, own, e);
            if (e.empty()) bytes += t->nbytes;
        }
        return w;
    };

    // token_embd → embed_dev (Q4_K/Q6_K/Q8_0 — the cyber model is Q8_0).
    {
        DeviceAllocator& ea = fleet.dev(plan.embed_dev);
        const auto* ti = Ttop("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ5_K && ti->dtype != DType::kQ6_K && ti->dtype != DType::kQ8_0)
            return std::string("token_embd: unsupported dtype ") +
                   std::string(type_name(ti->dtype)) + " (need Q4_K/Q5_K/Q6_K/Q8_0)";
        token_embd_dtype_ = ti->dtype;
        token_embd_ = dense::upload<void>(ea, ti, owned_[plan.embed_dev], err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
        dev_bytes_[plan.embed_dev] += ti->nbytes;
    }
    // output_norm + lm_head → head_dev.
    {
        DeviceAllocator& ha = fleet.dev(plan.head_dev);
        output_norm_ = dense::upload<float>(ha, Ttop("output_norm.weight"),
                                            owned_[plan.head_dev], err, DType::kF32);
        if (!err.empty()) return "output_norm: " + err;
        // lm_head is always a single-vector GEMV (last token) → Q8_0-SoA int-dot
        // when packed. Untied = output.weight; tied = replicate token_embd onto
        // head_dev (a device ptr can't be read from another card).
        const auto* ti = Ttop("output.weight");
        if (!ti) ti = Ttop("token_embd.weight");
        output_ = build_split(ha, ti, owned_[plan.head_dev], err, dev_bytes_[plan.head_dev]);
        if (!err.empty()) return "output: " + err;
    }

    const uint32_t n_layers = cfg.n_transformer_layers();   // 64 (skip trailing NextN/MTP)
    layers_.assign(n_layers, {});
    dn_local_.assign(n_layers, 0);
    kv_local_.assign(n_layers, 0);

    for (uint32_t L = 0; L < n_layers; ++L) {
        const uint32_t dev = plan.dev_of_layer[L];
        DeviceAllocator& a = fleet.dev(dev);
        auto& own = owned_[dev];
        LayerW& w = layers_[L];
        w.is_linear = cfg.recurrent_layer(L);

        auto LW = [&](const char* n) -> SplitW {
            return build_split(a, Tl(L, n), own, err, dev_bytes_[dev]);
        };
        auto F32 = [&](const char* n, float*& dst) -> std::string {
            const auto* ti = Tl(L, n);
            dst = dense::upload<float>(a, ti, own, err, DType::kF32);
            if (!err.empty()) return std::string(n) + ": " + err;
            if (ti) dev_bytes_[dev] += ti->nbytes;
            return {};
        };
        auto layer_err = [&](const char* what) {
            return "layer " + std::to_string(L) + " " + what + ": " + err;
        };

        if (auto m = F32("attn_norm.weight", w.attn_norm); !m.empty())
            return "layer " + std::to_string(L) + " " + m;
        if (auto m = F32("post_attention_norm.weight", w.post_attn_norm); !m.empty())
            return "layer " + std::to_string(L) + " " + m;

        if (w.is_linear) {
            w.attn_qkv  = LW("attn_qkv.weight");  if (!err.empty()) return layer_err("attn_qkv");
            w.attn_gate = LW("attn_gate.weight"); if (!err.empty()) return layer_err("attn_gate");
            if (auto m = F32("ssm_a", w.ssm_a); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            const uint32_t svh_pad = ((cfg.ssm_n_v_heads + 63u) / 64u) * 64u;   // 48 → 64
            w.ssm_alpha = upload_f32_proj_fp16(a, Tl(L, "ssm_alpha.weight"), own, err, svh_pad);
            if (!err.empty()) return layer_err("ssm_alpha");
            w.ssm_beta  = upload_f32_proj_fp16(a, Tl(L, "ssm_beta.weight"), own, err, svh_pad);
            if (!err.empty()) return layer_err("ssm_beta");
            if (auto m = F32("ssm_dt.bias", w.ssm_dt_bias); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            {   // ssm_conv1d [4, conv_ch] → cast fp16
                const auto* ti = Tl(L, "ssm_conv1d.weight");
                if (auto m = F32("ssm_conv1d.weight", w.ssm_conv1d); !m.empty()) return "layer " + std::to_string(L) + " " + m;
                const uint64_t n = ti->nbytes / sizeof(float);
                w.ssm_conv1d_fp16 = static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_conv1d_fp16) return layer_err("ssm_conv1d fp16 malloc");
                own.push_back(w.ssm_conv1d_fp16); dev_bytes_[dev] += n * sizeof(sycl::half);
                cast_fp32_to_fp16(a.queue(), w.ssm_conv1d, w.ssm_conv1d_fp16, n).wait();
            }
            {   // ssm_norm [v_head_dim] → cast fp16
                const auto* ti = Tl(L, "ssm_norm.weight");
                if (auto m = F32("ssm_norm.weight", w.ssm_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
                const uint64_t n = ti->nbytes / sizeof(float);
                w.ssm_norm_fp16 = static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half)));
                if (!w.ssm_norm_fp16) return layer_err("ssm_norm fp16 malloc");
                own.push_back(w.ssm_norm_fp16); dev_bytes_[dev] += n * sizeof(sycl::half);
                cast_fp32_to_fp16(a.queue(), w.ssm_norm, w.ssm_norm_fp16, n).wait();
            }
            w.ssm_out = LW("ssm_out.weight");   if (!err.empty()) return layer_err("ssm_out");
        } else {
            w.attn_q      = LW("attn_q.weight");      if (!err.empty()) return layer_err("attn_q");
            w.attn_k      = LW("attn_k.weight");      if (!err.empty()) return layer_err("attn_k");
            w.attn_v      = LW("attn_v.weight");      if (!err.empty()) return layer_err("attn_v");
            w.attn_output = LW("attn_output.weight"); if (!err.empty()) return layer_err("attn_output");
            if (auto m = F32("attn_q_norm.weight", w.attn_q_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
            if (auto m = F32("attn_k_norm.weight", w.attn_k_norm); !m.empty()) return "layer " + std::to_string(L) + " " + m;
        }

        w.ffn_gate = LW("ffn_gate.weight"); if (!err.empty()) return layer_err("ffn_gate");
        w.ffn_up   = LW("ffn_up.weight");   if (!err.empty()) return layer_err("ffn_up");
        w.ffn_down = LW("ffn_down.weight"); if (!err.empty()) return layer_err("ffn_down");
    }

    // Per-card hybrid caches: count each card's linear/full-attn layers, assign
    // card-local indices, init one DeltaNetState + one KvCache per card.
    std::vector<uint32_t> n_lin(n_dev_, 0), n_full(n_dev_, 0);
    for (uint32_t L = 0; L < n_layers; ++L) {
        const uint32_t dev = plan.dev_of_layer[L];
        if (layers_[L].is_linear) dn_local_[L] = n_lin[dev]++;
        else                      kv_local_[L] = n_full[dev]++;
    }
    dn_.resize(n_dev_); kv_.resize(n_dev_);
    cur_.assign(n_dev_, 0);   // P4 B18: every card acts on lane 0 (the live state) until select_lane
    max_ctx_ = max_ctx;
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (n_lin[dev]) {
            DeltaNetStateConfig dc{};
            dc.n_layers_linear = n_lin[dev];
            dc.n_v_heads       = cfg.ssm_n_v_heads;       // 48
            dc.v_head_dim      = cfg.ssm_state;           // 128
            dc.k_head_dim      = cfg.ssm_state;           // 128
            dc.conv_channels   = cfg.ssm_inner + 2u * cfg.ssm_n_k_heads * cfg.ssm_state;  // 10240
            dc.conv_kernel     = cfg.ssm_conv_kernel;     // 4
            if (auto m = dn_[dev].init(fleet.dev(dev), dc); !m.empty())
                return "dn cache dev " + std::to_string(dev) + ": " + m;
        }
        if (n_full[dev]) {
            KvCacheConfig kc{};
            kc.n_layers_full = n_full[dev];
            kc.n_kv_heads    = d.n_kv_heads;              // 4
            kc.max_ctx       = max_ctx;
            kc.head_dim      = d.head_dim;                // 256
            kc.use_int8      = int8_kv;                   // --int8-kv: halve KV (int8 shadow + fp16 scales)
            if (auto m = kv_[dev].init(fleet.dev(dev), kc); !m.empty())
                return "kv cache dev " + std::to_string(dev) + ": " + m;
        }
    }

    // Phase-2 per-card scratch (allocated once; T-independent). act_q8 = the int-dot
    // decode activation (block_q8_1x [Kmax/32]); prefill_bt grows lazily in sgemv to
    // the largest weight. Kmax = max projection input dim across the model.
    const uint32_t Kmax = std::max(std::max(d.hidden, d.ffn),
                                   std::max(cfg.ssm_inner, d.n_q_heads * d.head_dim));
    act_q8_.assign(n_dev_, nullptr);
    prefill_bt_.assign(n_dev_, nullptr);
    prefill_bt_cap_.assign(n_dev_, 0);
    prefill_dt_.assign(n_dev_, nullptr);
    prefill_dt_cap_.assign(n_dev_, 0);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        // ×16 rows: row 0 = T=1 decode; rows 0..T-1 = spec-verify batched GEMV (T≤16).
        void* p = fleet.dev(dev).malloc((uint64_t(Kmax) / 32) * sizeof(block_q8_1x) * 16u);
        if (!p) return "act_q8 alloc dev " + std::to_string(dev);
        owned_[dev].push_back(p);
        act_q8_[dev] = p;
    }
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        std::fprintf(stderr, "[qwen35split] card %u weights: %.2f GB\n",
                     dev, double(dev_bytes_[dev]) / 1e9);
    return {};
}

void Qwen35SplitModel::free_ws(uint32_t dev) {
    if (!fleet_ || dev >= ws_.size()) return;
    Workspace& w = ws_[dev];
    auto& alloc = fleet_->dev(dev);
    for (void* p : {static_cast<void*>(w.x), static_cast<void*>(w.x_normed),
                    static_cast<void*>(w.attn_block), static_cast<void*>(w.positions),
                    static_cast<void*>(w.positions3),
                    static_cast<void*>(w.ids), static_cast<void*>(w.logits),
                    static_cast<void*>(w.qg), static_cast<void*>(w.q),
                    static_cast<void*>(w.gate), static_cast<void*>(w.k),
                    static_cast<void*>(w.v), static_cast<void*>(w.attn_out),
                    static_cast<void*>(w.attn_partials),
                    static_cast<void*>(w.ag.qh), static_cast<void*>(w.ag.s), static_cast<void*>(w.ag.p),
                    static_cast<void*>(w.ag.o), static_cast<void*>(w.ag.l),
                    static_cast<void*>(w.dn_qkv), static_cast<void*>(w.dn_conv),
                    static_cast<void*>(w.dn_z), static_cast<void*>(w.dn_qpre),
                    static_cast<void*>(w.dn_kpre), static_cast<void*>(w.dn_vpre),
                    static_cast<void*>(w.dn_g), static_cast<void*>(w.dn_beta),
                    static_cast<void*>(w.dn_out), static_cast<void*>(w.dn_qrep),
                    static_cast<void*>(w.dn_krep), static_cast<void*>(w.dn_alpha_h),
                    static_cast<void*>(w.dn_beta_h), static_cast<void*>(w.dn_alpha64),
                    static_cast<void*>(w.dn_beta64),
                    static_cast<void*>(w.ffn_gate), static_cast<void*>(w.ffn_up),
                    static_cast<void*>(w.ffn_h)})
        if (p) alloc.free(p);
    w = Workspace{};
}

std::string Qwen35SplitModel::ensure_ws(uint32_t dev, uint32_t max_T) {
    Workspace& w = ws_[dev];
    max_T = (max_T + kAttnGemmRowStep - 1) / kAttnGemmRowStep * kAttnGemmRowStep;   // P4 B48: whole row steps
    if (max_T <= w.T) return {};
    if (w.T != 0) free_ws(dev);
    auto& alloc = fleet_->dev(dev);
    auto ah = [&](size_t n) { return static_cast<sycl::half*>(alloc.malloc(n * sizeof(sycl::half))); };
    auto af = [&](size_t n) { return static_cast<float*>(alloc.malloc(n * sizeof(float))); };

    const DenseConfig& d = cfg_.dense;
    const uint64_t T   = max_T;
    const uint32_t H   = d.hidden;                          // 5120
    const uint32_t N_q = d.n_q_heads * d.head_dim;          // 6144
    const uint32_t N_qg = N_q * 2u;                         // 12288
    const uint32_t N_kv = d.n_kv_heads * d.head_dim;        // 1024
    const uint32_t F   = d.ffn;                             // 17408
    const uint32_t SI  = cfg_.ssm_inner;                    // 6144
    const uint32_t cvc = SI + 2u * cfg_.ssm_n_k_heads * cfg_.ssm_state;  // 10240
    const uint32_t Vd  = cfg_.ssm_n_v_heads * cfg_.ssm_state;            // 6144
    const uint32_t Nv  = cfg_.ssm_n_v_heads;               // 48
    const uint32_t Nvp = ((Nv + 63u) / 64u) * 64u;         // 64

    w.x          = ah(T * H);
    w.x_normed   = ah(T * H);
    w.attn_block = ah(T * H);
    w.positions  = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));
    w.positions3 = static_cast<int32_t*>(alloc.malloc(3 * T * sizeof(int32_t)));   // P4 B45 (12 bytes a row)
    // persistent per-forward buffers (were malloc/free'd EVERY token in forward):
    if (dev == plan_.embed_dev)
        w.ids    = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));
    if (dev == plan_.head_dev)
        w.logits = ah(cfg_.dense.vocab);
    w.qg         = ah(T * N_qg);
    w.q          = ah(T * N_q);
    w.gate       = ah(T * N_q);
    w.k          = ah(T * N_kv);
    w.v          = ah(T * N_kv);
    w.attn_out   = ah(T * N_q);
    w.dn_qkv     = ah(T * cvc);
    w.dn_conv    = ah(T * cvc);
    w.dn_z       = ah(T * SI);
    w.dn_qpre    = af(T * Vd);
    w.dn_kpre    = af(T * Vd);
    w.dn_vpre    = af(T * Vd);
    w.dn_g       = af(T * Nv);
    w.dn_beta    = af(T * Nv);
    w.dn_out     = af(T * Vd);
    w.dn_qrep    = af(T * Vd);
    w.dn_krep    = af(T * Vd);
    w.dn_alpha_h = ah(T * Nv);
    w.dn_beta_h  = ah(T * Nv);
    w.dn_alpha64 = ah(T * Nvp);
    w.dn_beta64  = ah(T * Nvp);
    w.ffn_gate   = ah(T * F);
    w.ffn_up     = ah(T * F);
    w.ffn_h      = ah(T * F);

    if (!w.x || !w.x_normed || !w.attn_block || !w.positions || !w.positions3 || !w.qg || !w.q ||
        !w.gate || !w.k || !w.v || !w.attn_out || !w.dn_qkv || !w.dn_conv ||
        !w.dn_z || !w.dn_qpre || !w.dn_kpre || !w.dn_vpre || !w.dn_g || !w.dn_beta ||
        !w.dn_out || !w.dn_qrep || !w.dn_krep || !w.dn_alpha_h || !w.dn_beta_h ||
        !w.dn_alpha64 || !w.dn_beta64 || !w.ffn_gate || !w.ffn_up || !w.ffn_h)
        return "qwen35split workspace alloc failed on dev " + std::to_string(dev);

    // FA-2 decode partials for cards holding ≥1 full-attn layer (have a KV cache).
    if (kv_[dev].ready()) {
        const uint32_t max_ctx = kv_[dev].config().max_ctx;
        constexpr uint32_t Bc_floor = 64;
        const uint32_t n_chunks_max = (max_ctx + Bc_floor - 1) / Bc_floor;
        const uint64_t n_floats = uint64_t(n_chunks_max) * d.n_q_heads * (d.head_dim + 2);
        w.attn_partials = static_cast<float*>(alloc.malloc(n_floats * sizeof(float)));
        if (!w.attn_partials)
            return "qwen35split attn_partials alloc failed on dev " + std::to_string(dev);
        w.partials_ctx = max_ctx;
        // P4 B48: full_attention_prefill_gemm's scratch. Scores + weights: 32M elements (128 + 64 MB), or 16 rows at
        // this card's largest context when that is more.
        if (attn_gemm_on()) {
            const uint32_t gqa = d.n_q_heads / d.n_kv_heads;
            w.ag.sp_elems = std::max<uint64_t>(uint64_t(32) << 20, uint64_t(16) * max_ctx);
            w.ag.qh = ah(T * N_q);
            w.ag.s  = af(w.ag.sp_elems);
            w.ag.p  = ah(w.ag.sp_elems);
            w.ag.o  = af(T * gqa * d.head_dim);
            w.ag.l  = af(T * gqa);
            if (!w.ag.qh || !w.ag.s || !w.ag.p || !w.ag.o || !w.ag.l)
                return "qwen35split attention scratch alloc failed on dev " + std::to_string(dev);
        }
    }
    w.T = max_T;
    return {};
}

// P4 B48: a prefill chunk's full attention through full_attention_prefill_gemm (two oneDNN matmuls and a softmax)
// from kAttnGemmMinT rows. On by default (gated 2026-10-04: ~/ds41_work/p60/b48/BUILD.md); IE_QWEN35_ATTN_GEMM=0 = the
// streaming kernels.
// P4 B48: Q8_0 weights read in place by the prefill matmul (gemm_nt_s8_f16_onednn). On by default (gated 2026-10-04);
// IE_QWEN35_S8_PREFILL=0 = the expand-to-fp16 route.
bool Qwen35SplitModel::s8_prefill_on() {
    static const bool on = [] {
        const char* e = std::getenv("IE_QWEN35_S8_PREFILL");
        return !(e && *e && std::atoi(e) == 0) && onednn_available();
    }();
    return on;
}

// P4 B48: a prefill chunk's DeltaNet recurrence through deltanet_scan_prefill (no per-step barriers), from
// kAttnGemmMinT rows at the 128 / 128 head dims. On by default (gated 2026-10-04); IE_QWEN35_DN_SCAN=0 =
// deltanet_recurrence.
bool Qwen35SplitModel::dn_scan_on() {
    static const bool on = [] {
        const char* e = std::getenv("IE_QWEN35_DN_SCAN");
        return !(e && *e && std::atoi(e) == 0);
    }();
    return on;
}

// P4 B48: the spec verify's rows through full_attention_fa2_decode_rows -- one launch per pass for the V rows instead of
// V serial single-row calls per layer (the rows kernels run the single-row kernels' code per row: the same bytes).
// On by default (gated 2026-10-04: text equal to plain decode); IE_QWEN35_SPEC_ROWS=0 = the per-row loop.
bool Qwen35SplitModel::spec_rows_on() {
    static const bool on = [] {
        const char* e = std::getenv("IE_QWEN35_SPEC_ROWS");
        return !(e && *e && std::atoi(e) == 0);
    }();
    return on;
}

bool Qwen35SplitModel::attn_gemm_on() {
    static const bool on = [] {
        const char* e = std::getenv("IE_QWEN35_ATTN_GEMM");
        return !(e && *e && std::atoi(e) == 0) && onednn_available();
    }();
    return on;
}

// Q8_0-SoA aware GEMV. Decode (T==1): quantize activation → int-dot W8A8 over the
// packed int8 weight (≈½ the F16 bandwidth → decode win). Prefill (T>1): dequant
// the SoA weight → fp16 scratch (grown to the largest weight, reused) → gemm_fp16.
// Non-Q8_0 weights (Q4_K/Q6_K packed, Q5_K/F16) → the existing dense path.
sycl::event Qwen35SplitModel::sgemv(uint32_t dev, const sycl::half* A, const SplitW& w,
                                    sycl::half* out, uint32_t K, uint32_t N, uint32_t T, bool act_ready) {
    auto& alloc = fleet_->dev(dev);
    auto& q = alloc.queue();
    if (w.kq_lo && (T == 1 || (spec_verify_gemv_ && T <= 16)))   // P4 B21: Q6_K/Q5_K decode + verify rows (c32 int-dot)
        return kq_gemv(dev, A, w, out, K, N, T, act_ready && T == 1);
    if (!w.int_dot())                               // non-SoA fallback
        return dense::gemv_q_T(q, A, w.fp, out, K, N, T);
    if (T == 1) {                                   // decode: int-dot W8A8
        if (!act_ready) DPE(dev, kQuant, quantize_q8_1(q, A, act_q8_[dev], K));
        // SLM-staged activation is the DEFAULT and the faster path: the 32 columns
        // per WG share ONE staged copy of the (reused) activation. The no-SLM `_g`
        // variant (IE_QWEN35_SOA_GMEM=1) was A/B-tested SLOWER (~10×) — each column
        // re-reads the activation from L2, and that traffic outweighs the occupancy
        // gain. Kept opt-in only so the negative result isn't re-discovered.
        static const bool soa_gmem = std::getenv("IE_QWEN35_SOA_GMEM") != nullptr;
        if (soa_gmem)
            return gemv_q8_0_soa_q8_g_sel(q, act_q8_[dev], w.q8_qs, w.q8_d, out, K, N);
        // v2 = coalesced-load lane remap (one 64 B line/SG, no SLM). A/B 2026-08-15
        // (Qwen3.8-27B Q8_0, 2×B70): REGRESSION 12.86 vs 15.96 tg (-19%), PPL held.
        // v1 already runs ~71% of peak BW — the 1-dp4a/iteration loop overhead
        // outweighs the coalescing win. Kept opt-in so the negative result isn't
        // re-discovered (same lesson as the _g variant above).
        static const bool q8_v2 = []{
            const char* e = std::getenv("IE_QWEN35_Q8_SOA_V2");
            return e && std::atoi(e) != 0;
        }();
        if (q8_v2)
            return gemv_q8_0_soa_q8_v2(q, act_q8_[dev], w.q8_qs, w.q8_d, out, K, N);
        sycl::event e = gemv_q8_0_soa_q8_sel(q, act_q8_[dev], w.q8_qs, w.q8_d, out, K, N);
        if (g_dp27.active()) g_dp27.tag(dev, e, DecodeProf::kGemv);
        return e;
    }
    // Spec-decode VERIFY (2 ≤ T ≤ 16, only while spec_verify_gemv_ is set): batched
    // int-dot — every weight block read ONCE and dotted against all T rows. The
    // dequant→gemm path below would re-materialize the whole weight per verify round
    // (~28.5 GB), destroying the amortization spec exists to buy.
    if (spec_verify_gemv_ && w.q8_qs && T >= 2 && T <= 16) {
        quantize_q8_1(q, A, act_q8_[dev], uint32_t(uint64_t(T) * K));
        return gemv_q8_0_soa_q8_batched_sel(q, act_q8_[dev], w.q8_qs, w.q8_d, out, K, N, T);
    }
    // P4 B48: a Q8_0 weight goes to the matmul in place (oneDNN weights decompression): no expand pass. The scale plane
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
            g_pf27.mark(dev, q, PfProf27::kExpand);
            sycl::event e;
            const uint32_t M = sgemv_pad_ ? std::min<uint32_t>((T + kAttnGemmRowStep - 1) / kAttnGemmRowStep * kAttnGemmRowStep,
                                                               ws_[dev].T) : T;
            if (gemm_nt_s8_f16_onednn(q, A, w.q8_qs, prefill_dt_[dev], out, M, N, K, 32, &e)) {
                g_pf27.mark(dev, q, PfProf27::kGemm);
                return e;
            }
            static bool said = false;
            if (!said) {
                said = true;
                std::fprintf(stderr, "[qwen35split] IE_QWEN35_S8_PREFILL: no jitted oneDNN kernel (%s); the expand route\n",
                             onednn_nt_s8_f16_impl(q, T, N, K, 32).c_str());
            }
        }
    }
    // prefill: dequant SoA → fp16 scratch, then the batched gemm via gemv_q_T(F16).
    const uint64_t need = uint64_t(K) * N;
    if (need > prefill_bt_cap_[dev]) {
        if (prefill_bt_[dev]) alloc.free(prefill_bt_[dev]);
        prefill_bt_[dev] = static_cast<sycl::half*>(alloc.malloc(need * sizeof(sycl::half)));
        prefill_bt_cap_[dev] = prefill_bt_[dev] ? need : 0;
    }
    if (!prefill_bt_[dev]) { std::fprintf(stderr, "qwen35split: prefill_bt alloc failed\n"); return {}; }
    if (w.kq == DType::kQ6_K)
        dequant_q6k_c32_to_Bt(q, w.kq_lo, static_cast<const uint8_t*>(w.kq_hi), static_cast<const int8_t*>(w.kq_sc),
                              static_cast<const uint16_t*>(w.kq_d), prefill_bt_[dev], K, N);
    else if (w.kq == DType::kQ5_K)
        dequant_q5k_c32_to_Bt(q, w.kq_lo, static_cast<const uint32_t*>(w.kq_hi), static_cast<const uint16_t*>(w.kq_sc),
                              static_cast<const uint32_t*>(w.kq_d), prefill_bt_[dev], K, N);
    else
        dequant_q8_0_soa_to_Bt(q, w.q8_qs, w.q8_d, prefill_bt_[dev], K, N);
    g_pf27.mark(dev, q, PfProf27::kExpand);
    sycl::event e = dense::gemv_q_T(q, A, DenseQuantPtr{prefill_bt_[dev], DType::kF16}, out, K, N, T);
    g_pf27.mark(dev, q, PfProf27::kGemm);
    return e;
}

std::string Qwen35SplitModel::forward(const int32_t* input_ids, uint32_t T,
                                      uint32_t start_pos, bool reset_kv,
                                      sycl::half* out_logits_host,
                                      sycl::half* all_logits_dev,
                                      sycl::half* hidden_pre_norm_dev,
                                      std::vector<Qwen35SpecCheckpoint>* ckpts) {
    if (T == 0) return "T == 0";
    // Checkpoint mode: per-card per-position DeltaNet snapshots (spec verify).
    const bool ckpt_mode = ckpts && ckpts->size() == n_dev_ && T >= 1 &&
                           !ckpts->empty() && (*ckpts)[0].K >= T;
    const DenseConfig& dc = cfg_.dense;
    const uint32_t H    = dc.hidden;                    // 5120
    const uint32_t HD   = dc.head_dim;                  // 256
    const uint32_t N_q  = dc.n_q_heads  * HD;           // 6144
    const uint32_t N_qg = N_q * 2u;                     // 12288
    const uint32_t N_kv = dc.n_kv_heads * HD;           // 1024
    const uint32_t V    = dc.vocab;
    const uint32_t F    = dc.ffn;                       // 17408
    const uint32_t rope_n = dc.rope_dim;                // 64 (partial)
    const float    eps  = dc.rms_eps;
    const uint32_t n_layers = cfg_.n_transformer_layers();
    const uint32_t n_kv = dc.n_kv_heads;

    // env-gated decode phase profiler (IE_QWEN35_PROFILE=1). Wait-bracketed →
    // absolute total runs slower than prod; read it as a RELATIVE breakdown.
    static const bool prof = std::getenv("IE_QWEN35_PROFILE") != nullptr;
    using pclk = std::chrono::steady_clock;
    static double pf_dn = 0, pf_attn = 0, pf_ffn = 0, pf_head = 0, pf_other = 0, pf_total = 0;
    static uint64_t pf_tok = 0, pf_calls = 0;
    auto pms = [](pclk::time_point a, pclk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    pclk::time_point pf_t0 = pclk::now();
    pclk::time_point pf_last;
    auto mark = [&](sycl::queue& qq, double& acc) {
        if (!prof) return;
        qq.wait();
        auto now = pclk::now();
        acc += pms(pf_last, now);
        pf_last = now;
    };

    if (ws_.size() != n_dev_) ws_.assign(n_dev_, {});
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (auto m = ensure_ws(dev, T); !m.empty()) return m;
    g_dp27.begin(T, start_pos);

    // positions on every card (+ the [3, T] M-RoPE streams while vision is staged, P4 B45); reset per-card hybrid
    // state on a fresh sequence.
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (ws_[dev].positions) upload_positions(dev, start_pos, T);
        if (reset_kv) {
            if (dn_[dev].ready()) dn_[dev].reset(fleet_->dev(dev).queue());
            if (kv_[dev].ready()) kv_[dev].reset();
        }
    }

    // embedding → ws_[embed_dev].x
    {
        auto& q = fleet_->dev(plan_.embed_dev).queue();
        int32_t* d_ids = ws_[plan_.embed_dev].ids;   // persistent [T] (ensure_ws)
        if (!d_ids) return "d_ids alloc failed";
        q.memcpy(d_ids, input_ids, T * sizeof(int32_t)).wait();
        if (token_embd_dtype_ == DType::kQ4_K)
            embedding_lookup_q4k(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
        else if (token_embd_dtype_ == DType::kQ8_0)
            embedding_lookup_q8_0(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
        else if (token_embd_dtype_ == DType::kQ5_K)
            embedding_lookup_q5k(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
        else
            embedding_lookup_q6k(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
        splice_vision(q, ws_[plan_.embed_dev].x, start_pos, T);   // P4 B45: the image rows (in-order: after the gather)
        q.wait();
        g_dp27.phase(DecodeProf::kSetup);
    }

    // device-by-device: copy residual in from prior card, run this card's layers.
    sgemv_pad_ = true;   // P4 B48: the layer loops' operands are workspace buffers (cleared before the head)
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        Workspace& w = ws_[dev];
        auto& q = fleet_->dev(dev).queue();
        if (dev > 0) {   // residual hand-off at the card boundary (ONE copy)
            fleet_->copy_across(dev - 1, w.x, dev, ws_[dev - 1].x,
                                uint64_t(T) * H * sizeof(sycl::half));
            g_dp27.phase(DecodeProf::kHand);
        }
        const uint64_t per_layer_kv =
            uint64_t(n_kv) * (kv_[dev].ready() ? kv_[dev].config().max_ctx : 0u) * HD;
        // P4 B48: the verify rows' attention plan (all rows on this card's cache, row i at start_pos + i with the kernel
        // the per-row loop would give it), built once per forward and run per layer with its KV slot.
        FaDecodeRowsPlan vplan;
        bool vrows = false;
        if (spec_verify_gemv_ && spec_rows_on() && T >= 2 && T <= kFaDecodeRowsMax && kv_[dev].ready() && w.attn_partials) {
            if (vrows_partials_.size() != n_dev_) { vrows_partials_.assign(n_dev_, nullptr); vrows_table_.assign(n_dev_, nullptr); }
            const uint32_t max_ctx = kv_[dev].config().max_ctx;
            if (!vrows_partials_[dev]) {
                vrows_partials_floats_ = fa2_decode_rows_partials_floats(dc.n_q_heads, dc.n_kv_heads, HD, max_ctx);
                void* pp = fleet_->dev(dev).malloc(uint64_t(kFaDecodeRowsMax) * vrows_partials_floats_ * sizeof(float));
                void* tt = fleet_->dev(dev).malloc(kFaDecodeRowsTableBytes);
                if (pp && tt) {
                    owned_[dev].push_back(pp); owned_[dev].push_back(tt);
                    vrows_partials_[dev] = static_cast<float*>(pp); vrows_table_[dev] = tt;
                } else {
                    if (pp) fleet_->dev(dev).free(pp);
                    if (tt) fleet_->dev(dev).free(tt);
                }
            }
            if (vrows_partials_[dev]) {
                static const bool dec_vec_p = []{
                    const char* e = std::getenv("IE_Q35_FA2_VEC");
                    return !e || std::atoi(e) != 0;
                }();
                FaDecodeRow rows[kFaDecodeRowsMax];
                for (uint32_t i = 0; i < T; ++i) {
                    rows[i].k_base = kv_[dev].k_ptr();
                    rows[i].v_base = kv_[dev].v_ptr();
                    rows[i].layer_stride = per_layer_kv;
                    rows[i].start_pos = start_pos + i;
                    rows[i].max_ctx = max_ctx;
                    rows[i].kind = q35::xmx_decode(start_pos + i, max_ctx) ? FaDecodeKind::kXmx
                                 : dec_vec_p ? FaDecodeKind::kVec : FaDecodeKind::kAuto;
                }
                vrows = fa2_decode_rows_plan(q, rows, T, dc.n_q_heads, dc.n_kv_heads, HD, vrows_partials_floats_,
                                             vrows_table_[dev], vplan);
            }
        }
        if (prof) { q.wait(); pf_last = pclk::now(); }

        for (uint32_t L = 0; L < n_layers; ++L) {
            if (plan_.dev_of_layer[L] != dev) continue;
            const LayerW& w_l = layers_[L];
            // pre-attn norm
            DPE(dev, kNorm, rms_norm_f32w(q, w.x, w_l.attn_norm, w.x_normed, T, H, eps));

            if (w_l.is_linear) {
                // ---- gated-DeltaNet (27B conventions: separate ssm_alpha/beta,
                // tile repeat 16→48). Lifted verbatim from qwen35_dense.cpp. ----
                const uint32_t SKH = cfg_.ssm_n_k_heads;          // 16
                const uint32_t SVH = cfg_.ssm_n_v_heads;          // 48
                const uint32_t SHD = cfg_.ssm_state;              // 128
                const uint32_t SI  = cfg_.ssm_inner;              // 6144
                const uint32_t conv_ch = SI + 2u * SKH * SHD;     // 10240
                const uint32_t kw  = SKH * SHD;                   // 2048
                const uint32_t rep = SVH / SKH;                   // 3 (16→48)
                const float qscale = 1.0f / sycl::sqrt(float(SHD));

                sgemv(dev, w.x_normed, w_l.attn_qkv, w.dn_qkv, H, conv_ch, T);
                sycl::half* conv_state = dn_[dev].conv_state_ptr() +
                    uint64_t(dn_local_[L]) * dn_[dev].conv_elems_per_layer();
                if (ckpt_mode) {
                    // Fused: ONE snapshot kernel + ONE batched conv, replacing the
                    // T×(conv(T=1) + state memcpy) loop. The conv state is raw
                    // input halfs shifted through ([kernel-1, ch]; row p after
                    // position s = token s-(kernel-2)+p, pre-state for negatives),
                    // so all T post-position states are pure index remaps of
                    // (pre-state, dn_qkv) — byte-identical to what the per-step
                    // memcpys captured. The batched conv itself is byte-identical
                    // to T streaming steps (causal), per the original note.
                    Qwen35SpecCheckpoint& ck = (*ckpts)[dev];
                    const uint64_t ce = dn_[dev].conv_elems_per_layer();
                    {
                        const uint32_t km1 = cfg_.ssm_conv_kernel - 1;
                        const uint32_t ch  = conv_ch;
                        const sycl::half* xin = w.dn_qkv;
                        const sycl::half* pst = conv_state;
                        sycl::half* ck_base = ck.ckpt_conv + uint64_t(dn_local_[L]) * ce;
                        const uint64_t ck_step = uint64_t(ck.n_lin) * ce;
                        q.submit([&](sycl::handler& h) {
                            h.parallel_for(sycl::range<3>(T, km1, ch), [=](sycl::id<3> id) {
                                const uint32_t s = uint32_t(id[0]);
                                const uint32_t p = uint32_t(id[1]);
                                const uint32_t c = uint32_t(id[2]);
                                const int32_t i = int32_t(s) - int32_t(km1 - 1) + int32_t(p);
                                const sycl::half v = (i >= 0)
                                    ? xin[uint64_t(i) * ch + c]
                                    : pst[uint64_t(uint32_t(i + int32_t(km1))) * ch + c];
                                ck_base[uint64_t(s) * ck_step + uint64_t(p) * ch + c] = v;
                            });
                        });
                    }
                    depthwise_conv1d_causal(q, w.dn_qkv, w_l.ssm_conv1d_fp16, conv_state,
                                            w.dn_conv, T, conv_ch, cfg_.ssm_conv_kernel);
                } else {
                DPE(dev, kDnConv, depthwise_conv1d_causal(q, w.dn_qkv, w_l.ssm_conv1d_fp16, conv_state,
                                        w.dn_conv, T, conv_ch, cfg_.ssm_conv_kernel));
                }
                DPE(dev, kDnMisc, cast_qkv_split_fp16_to_fp32(q, w.dn_conv, w.dn_qpre, w.dn_kpre, w.dn_vpre, T, kw, SI));
                DPE(dev, kDnMisc, l2_norm_scale(q, w.dn_qpre, w.dn_qpre, T * SKH, SHD, qscale, 1e-6f));
                DPE(dev, kDnMisc, l2_norm_scale(q, w.dn_kpre, w.dn_kpre, T * SKH, SHD, 1.0f,   1e-6f));
                // TILE repeat 16→48 (interleave=false, the VALIDATED 27B convention)
                DPE(dev, kDnMisc, repeat_interleave_heads(q, w.dn_qpre, w.dn_qrep, T, SKH, SHD, rep));
                DPE(dev, kDnMisc, repeat_interleave_heads(q, w.dn_kpre, w.dn_krep, T, SKH, SHD, rep));
                // separate N-padded ssm_alpha/ssm_beta projections → batched gemm → compact
                const uint32_t SVHp = ((SVH + 63u) / 64u) * 64u;   // 64
                if (((spec_verify_gemv_ && T >= 2 && T <= 16) || T == 1) &&   // P4 B22: T == 1 too (the same leaf, one launch)
                    w_l.ssm_alpha.dt == DType::kF16 && w_l.ssm_beta.dt == DType::kF16) {
                    // Spec VERIFY: per-row math is REQUIRED (the batched gemm's fp
                    // order differs in the last bit and breaks losslessness). The
                    // fused kernel runs the T=1 F16 leaf verbatim per (row, mat)
                    // in ONE launch instead of 2·T tiny N=64 launches.
                    DPE(dev, kGemvSmall, gemv_fp16_rows_dual(q, w.x_normed, H,
                        static_cast<const sycl::half*>(w_l.ssm_alpha.p), w.dn_alpha64,
                        static_cast<const sycl::half*>(w_l.ssm_beta.p),  w.dn_beta64,
                        SVHp, H, SVHp, T));
                } else if (spec_verify_gemv_ && T >= 2 && T <= 16) {
                    // Non-F16 alpha/beta (never seen for qwen35; kept as the exact
                    // per-row fallback for other dtypes).
                    for (uint32_t t = 0; t < T; ++t) {
                        dense::gemv_q_T(q, w.x_normed + uint64_t(t) * H, w_l.ssm_alpha,
                                        w.dn_alpha64 + uint64_t(t) * SVHp, H, SVHp, 1);
                        dense::gemv_q_T(q, w.x_normed + uint64_t(t) * H, w_l.ssm_beta,
                                        w.dn_beta64 + uint64_t(t) * SVHp, H, SVHp, 1);
                    }
                } else {
                DPE(dev, kGemvSmall, dense::gemv_q_T(q, w.x_normed, w_l.ssm_alpha, w.dn_alpha64, H, SVHp, T));
                DPE(dev, kGemvSmall, dense::gemv_q_T(q, w.x_normed, w_l.ssm_beta,  w.dn_beta64,  H, SVHp, T));
                }
                DPE(dev, kDnMisc, extract_cols(q, w.dn_alpha64, w.dn_alpha_h, T, SVH, SVHp));
                DPE(dev, kDnMisc, extract_cols(q, w.dn_beta64,  w.dn_beta_h,  T, SVH, SVHp));
                DPE(dev, kDnMisc, compute_g_beta_h16(q, w.dn_alpha_h, w.dn_beta_h, w_l.ssm_a, w_l.ssm_dt_bias,
                                   w.dn_g, w.dn_beta, T, SVH));
                float* state_layer = dn_[dev].state_ptr() +
                    uint64_t(dn_local_[L]) * dn_[dev].state_elems_per_layer();
                if (ckpt_mode) {
                    // Fused scan + per-position snapshot: ONE launch replaces the
                    // T×(recurrence(T=1) + D2D state memcpy) loop. Bit-identical:
                    // K×(T=1) == 1×(T=K) (sequential scan, fp32 state), and the
                    // in-kernel post-step store equals what the memcpy captured.
                    Qwen35SpecCheckpoint& ck = (*ckpts)[dev];
                    const uint64_t se = dn_[dev].state_elems_per_layer();
                    deltanet_recurrence_ckpt(q,
                        w.dn_qrep, w.dn_krep, w.dn_vpre, w.dn_g, w.dn_beta,
                        state_layer, w.dn_out,
                        ck.ckpt_state + uint64_t(dn_local_[L]) * se,
                        /*ckpt_step=*/uint64_t(ck.n_lin) * se,
                        T, SVH, SHD, SHD);
                } else {
                if (dn_scan_on() && T >= kAttnGemmMinT && SHD == 128)   // P4 B48
                    deltanet_scan_prefill(q, w.dn_qrep, w.dn_krep, w.dn_vpre, w.dn_g, w.dn_beta, state_layer, w.dn_out, T, SVH);
                else
                DPE(dev, kDnRec, deltanet_recurrence(q, w.dn_qrep, w.dn_krep, w.dn_vpre, w.dn_g, w.dn_beta,
                                    state_layer, w.dn_out, /*B=*/1, T, SVH, SHD, SHD));
                }
                // gated RMS-norm with z = attn_gate · x_normed (reuse dn_qkv as out)
                if (spec_verify_gemv_ && T >= 2 && T <= 16 && w_l.attn_gate.q8_qs) {
                    // Verify: act_q8_ still holds x_normed's quantization — the
                    // attn_qkv sgemv above quantized it and nothing since touches
                    // act_q8_ (αβ runs the F16 leaf; conv/recurrence don't quantize).
                    // Call the batched kernel directly, skipping the re-quantize.
                    gemv_q8_0_soa_q8_batched_sel(q, act_q8_[dev],
                        w_l.attn_gate.q8_qs, w_l.attn_gate.q8_d, w.dn_z, H, SI, T);
                } else {
                    sgemv(dev, w.x_normed, w_l.attn_gate, w.dn_z, H, SI, T);
                }
                DPE(dev, kDnMisc, gated_rms_norm(q, w.dn_out, w.dn_z, w_l.ssm_norm_fp16, w.dn_qkv,
                               T * SVH, SHD, eps));
                sgemv(dev, w.dn_qkv, w_l.ssm_out, w.attn_block, SI, H, T);
            } else {
                // ---- gated full-attention (lifted verbatim from qwen35_dense.cpp) ----
                if (spec_verify_gemv_ && T >= 2 && T <= 16 &&
                    w_l.attn_q.q8_qs && w_l.attn_k.q8_qs && w_l.attn_v.q8_qs) {
                    // Verify: ONE activation quantize for q/k/v (sgemv would run it
                    // 3×) and k+v in a dual launch (same [H, N_kv] shape). Same
                    // kernels, same op order → bit-identical.
                    quantize_q8_1(q, w.x_normed, act_q8_[dev],
                                  uint32_t(uint64_t(T) * H));
                    gemv_q8_0_soa_q8_batched_sel(q, act_q8_[dev],
                        w_l.attn_q.q8_qs, w_l.attn_q.q8_d, w.qg, H, N_qg, T);
                    split_q_gate_per_head(q, w.qg, w.q, w.gate, T, dc.n_q_heads, HD);
                    gemv_q8_0_soa_q8_batched_dual_sel(q, act_q8_[dev],
                        w_l.attn_k.q8_qs, w_l.attn_k.q8_d, w.k,
                        w_l.attn_v.q8_qs, w_l.attn_v.q8_d, w.v, H, N_kv, T);
                } else {
                    // P4 B21: at T == 1 k and v reuse q's quantized x_normed when q went through an int-dot route
                    const bool qa = T == 1 && w_l.attn_q.int_dot();
                    sgemv(dev, w.x_normed, w_l.attn_q, w.qg, H, N_qg, T);
                    DPE(dev, kAttnPre, split_q_gate_per_head(q, w.qg, w.q, w.gate, T, dc.n_q_heads, HD));
                    sgemv(dev, w.x_normed, w_l.attn_k, w.k, H, N_kv, T, qa);
                    sgemv(dev, w.x_normed, w_l.attn_v, w.v, H, N_kv, T, qa && w_l.attn_k.int_dot());
                }
                DPE(dev, kAttnPre, rms_norm_f32w(q, w.q, w_l.attn_q_norm, w.q, T * dc.n_q_heads,  HD, eps));
                DPE(dev, kAttnPre, rms_norm_f32w(q, w.k, w_l.attn_k_norm, w.k, T * dc.n_kv_heads, HD, eps));
                if (mrope_n_) {   // P4 B45 vision: the 3-stream M-RoPE (== rope_partial when the streams are equal)
                    rope_imrope3(q, w.q, w.positions3, w.q, T, dc.n_q_heads,  HD, rope_n, dc.rope_theta);
                    rope_imrope3(q, w.k, w.positions3, w.k, T, dc.n_kv_heads, HD, rope_n, dc.rope_theta);
                } else {
                DPE(dev, kAttnPre, rope_partial(q, w.q, w.positions, w.q, T, dc.n_q_heads,  HD, rope_n, dc.rope_theta));
                DPE(dev, kAttnPre, rope_partial(q, w.k, w.positions, w.k, T, dc.n_kv_heads, HD, rope_n, dc.rope_theta));
                }
                auto& kvc = kv_[dev];
                const uint32_t li = kv_local_[L];   // card-local full-attn layer index
                sycl::half* kc = kvc.k_ptr() + per_layer_kv * li;
                sycl::half* vc = kvc.v_ptr() + per_layer_kv * li;
                const uint32_t max_ctx = kvc.config().max_ctx;
                if (T == 1 && w.attn_partials) {
                    // INT8-KV decode (--int8-kv) when the int8 shadow is populated up to
                    // start_pos; else fp16 FA2. Mirrors qwen35_dense.cpp / the crown split.
                    if (kvc.is_int8() && kvc.k_int8_ptr() && start_pos == kvc.int8_length(li)) {
                        const uint64_t i8pl = uint64_t(dc.n_kv_heads) * max_ctx * HD;
                        const uint64_t scpl = uint64_t(dc.n_kv_heads) * max_ctx;
                        full_attention_fa2_decode_int8(q, w.q, w.k, w.v,
                                                       kvc.k_int8_ptr() + i8pl * li,
                                                       kvc.v_int8_ptr() + i8pl * li,
                                                       kvc.k_scales_ptr() + scpl * li,
                                                       kvc.v_scales_ptr() + scpl * li,
                                                       nullptr, nullptr,
                                                       w.attn_out, w.attn_partials, start_pos,
                                                       dc.n_q_heads, dc.n_kv_heads, HD, max_ctx);
                        kvc.set_int8_length(li, start_pos + 1);
                    } else {
                        // IE_Q35_FA2_VEC: same llama fattn-vec decode port the dense path
                        // runs DEFAULT ON (+47% long-ctx 16K decode, short-ctx neutral,
                        // PPL held — 2026-07-18 A/B). Same kernel, same buffers/partials
                        // format, per-card pointers here. IE_Q35_FA2_VEC=0 kills both
                        // paths back to v1.
                        static const bool dec_vec = []{
                            const char* e = std::getenv("IE_Q35_FA2_VEC");
                            return !e || std::atoi(e) != 0;
                        }();
                        // P4 B23: the XMX kernel from IE_Q35_XMX_DECODE_MIN context (default on, IE_Q35_XMX_DECODE=0 off;
                        // numerics change; q35_xmx_decode.hpp).
                        if (q35::xmx_decode(start_pos, max_ctx))
                            DPE(dev, kAttn, full_attention_fa2_decode_xmx(q, w.q, w.k, w.v, kc, vc, w.attn_out,
                                                          w.attn_partials, start_pos,
                                                          dc.n_q_heads, dc.n_kv_heads, HD, max_ctx));
                        else if (dec_vec)
                            DPE(dev, kAttn, full_attention_fa2_decode_vec(q, w.q, w.k, w.v, kc, vc, w.attn_out,
                                                          w.attn_partials, start_pos,
                                                          dc.n_q_heads, dc.n_kv_heads, HD, max_ctx));
                        else
                            DPE(dev, kAttn, full_attention_fa2_decode(q, w.q, w.k, w.v, kc, vc, w.attn_out,
                                                      w.attn_partials, start_pos,
                                                      dc.n_q_heads, dc.n_kv_heads, HD, max_ctx));
                    }
                } else if (spec_verify_gemv_ && T <= 16 && w.attn_partials) {
                    // Spec-decode VERIFY: LOOP the decode kernel over the T positions —
                    // bit-identical to T sequential T==1 decode steps (the prefill
                    // leaves below are NOT; same lesson as qwen35_dense.cpp's verify).
                    // Mirror the decode branch's kernel choice exactly (vec default).
                    static const bool dec_vec_v = []{
                        const char* e = std::getenv("IE_Q35_FA2_VEC");
                        return !e || std::atoi(e) != 0;
                    }();
                    if (vrows) {
                        // P4 B48: the T rows share this cache, and a row attends the rows before it: append them all first
                        // (the rows kernels append per kernel kind, and a group that straddles the XMX threshold would run
                        // its later rows before its earlier ones were appended), then one launch per pass. The XMX kind's
                        // append must not zero the group's own rows: the _shared entry point.
                        const sycl::half* k_in = w.k; const sycl::half* v_in = w.v;
                        const uint32_t nkv = dc.n_kv_heads;
                        ie::ps(q, "q27_verify_rows_append", [&](sycl::handler& h) {
                            h.parallel_for(sycl::range<1>(uint64_t(T) * nkv * HD), [=](sycl::id<1> id) {
                                const uint64_t idx = id[0];
                                const uint32_t d   = uint32_t(idx % HD);
                                const uint32_t kvh = uint32_t((idx / HD) % nkv);
                                const uint32_t t   = uint32_t(idx / (uint64_t(nkv) * HD));
                                const uint64_t o   = (uint64_t(kvh) * max_ctx + (start_pos + t)) * HD + d;
                                kc[o] = k_in[idx];
                                vc[o] = v_in[idx];
                            });
                        });
                        full_attention_fa2_decode_rows_shared(q, vplan, w.q, w.k, w.v, w.attn_out, vrows_partials_[dev], li,
                                                              start_pos + T);
                    } else
                    for (uint32_t t = 0; t < T; ++t) {
                        if (q35::xmx_decode(start_pos + t, max_ctx))   // P4 B23: same choice per position as decode
                            full_attention_fa2_decode_xmx(q,
                                w.q + uint64_t(t) * N_q, w.k + uint64_t(t) * N_kv,
                                w.v + uint64_t(t) * N_kv, kc, vc,
                                w.attn_out + uint64_t(t) * N_q, w.attn_partials,
                                start_pos + t, dc.n_q_heads, dc.n_kv_heads, HD, max_ctx);
                        else if (dec_vec_v)
                            full_attention_fa2_decode_vec(q,
                                w.q + uint64_t(t) * N_q, w.k + uint64_t(t) * N_kv,
                                w.v + uint64_t(t) * N_kv, kc, vc,
                                w.attn_out + uint64_t(t) * N_q, w.attn_partials,
                                start_pos + t, dc.n_q_heads, dc.n_kv_heads, HD, max_ctx);
                        else
                            full_attention_fa2_decode(q,
                                w.q + uint64_t(t) * N_q, w.k + uint64_t(t) * N_kv,
                                w.v + uint64_t(t) * N_kv, kc, vc,
                                w.attn_out + uint64_t(t) * N_q, w.attn_partials,
                                start_pos + t, dc.n_q_heads, dc.n_kv_heads, HD, max_ctx);
                    }
                    if (kvc.is_int8()) kvc.quantize_to_int8(q, li, start_pos, T);
                } else {
                    // Long-ctx full-attn prefill: route the hd256 layers through the Gemma
                    // wide-tile kernel at ctx >= minctx — same gate as qwen35_dense.cpp
                    // (argmax-bit-identical at hd256; naive O(T²) re-reads the whole KV T×
                    // and collapsed 16K prefill 4.35× on the dense path before this).
                    // Opt-out IE_QWEN35_NO_FA2_TILE; tune IE_QWEN35_FA2_TILE_MINCTX.
                    static const bool no_tile = std::getenv("IE_QWEN35_NO_FA2_TILE") != nullptr;
                    static const uint32_t tile_minctx = []() -> uint32_t {
                        const char* e = std::getenv("IE_QWEN35_FA2_TILE_MINCTX");
                        if (!e) return 6144u;
                        int v = std::atoi(e); return v > 0 ? uint32_t(v) : 6144u;
                    }();
                    if (w.ag.s && T >= kAttnGemmMinT) {   // P4 B48
                        full_attention_prefill_gemm(q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                                                    dc.n_q_heads, dc.n_kv_heads, HD, max_ctx, w.ag);
                    } else if (!no_tile && HD == 256 && (start_pos + T) >= tile_minctx) {
                        full_attention_fa2_prefill_tile_gemma(
                            q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                            dc.n_q_heads, dc.n_kv_heads, HD, max_ctx,
                            0 /*window: full causal*/);
                    } else {
                        full_attention(q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                                       dc.n_q_heads, dc.n_kv_heads, HD, max_ctx);
                    }
                    // Post-quantize the fp16 prefill rows into the int8 shadow so the next
                    // T=1 decode step can take the int8 path.
                    if (kvc.is_int8()) kvc.quantize_to_int8(q, li, start_pos, T);
                }
                kvc.set_length(li, start_pos + T);
                DPE(dev, kAttnPre, sigmoid_gate(q, w.attn_out, w.gate, w.attn_out, uint64_t(T) * N_q));
                sgemv(dev, w.attn_out, w_l.attn_output, w.attn_block, N_q, H, T);
            }
            mark(q, w_l.is_linear ? pf_dn : pf_attn);   // pre-norm + DeltaNet/attn block

            // residual + pre-FFN (post-attention) norm (27B fused order) → dense SwiGLU
            DPE(dev, kNorm, residual_add_rms_norm_fused(q, w.x, w.attn_block, w_l.post_attn_norm,
                                        w.x_normed, T, H, eps));
            if (spec_verify_gemv_ && T >= 2 && T <= 16 &&
                w_l.ffn_gate.q8_qs && w_l.ffn_up.q8_qs) {
                // Verify: gate+up in ONE dual launch off ONE activation quantize
                // (sgemv would quantize the same T×H rows twice and serialize
                // the two big GEMVs on the in-order queue). Bit-identical.
                quantize_q8_1(q, w.x_normed, act_q8_[dev],
                              uint32_t(uint64_t(T) * H));
                gemv_q8_0_soa_q8_batched_dual_sel(q, act_q8_[dev],
                    w_l.ffn_gate.q8_qs, w_l.ffn_gate.q8_d, w.ffn_gate,
                    w_l.ffn_up.q8_qs,   w_l.ffn_up.q8_d,   w.ffn_up,
                    H, F, T);
            } else {
                sgemv(dev, w.x_normed, w_l.ffn_gate, w.ffn_gate, H, F, T);
                sgemv(dev, w.x_normed, w_l.ffn_up,   w.ffn_up,   H, F, T, T == 1 && w_l.ffn_gate.int_dot());   // P4 B21
            }
            DPE(dev, kElem, swiglu(q, w.ffn_gate, w.ffn_up, w.ffn_h, uint64_t(T) * F));
            sgemv(dev, w.ffn_h, w_l.ffn_down, w.attn_block, F, H, T);
            DPE(dev, kElem, residual_add(q, w.x, w.attn_block, w.x, uint64_t(T) * H));
            mark(q, pf_ffn);   // post-norm + dense SwiGLU FFN
        }
        q.wait();   // finish this card before the boundary copy reads its ws.x
        g_dp27.phase(dev == 0 ? DecodeProf::kCard0 : DecodeProf::kCard1);
        g_dp27.sync(dev);
    }

    sgemv_pad_ = false;
    // final norm + lm_head on head_dev → last token's logits → host.
    {
        const uint32_t hd = plan_.head_dev;
        Workspace& w = ws_[hd];
        auto& q = fleet_->dev(hd).queue();
        if (prof) { q.wait(); pf_last = pclk::now(); }
        // Spec conditioning: final residual BEFORE output_norm, all T rows.
        if (hidden_pre_norm_dev)
            q.memcpy(hidden_pre_norm_dev, w.x, uint64_t(T) * H * sizeof(sycl::half));
        DPE(hd, kNorm, rms_norm_f32w(q, w.x, output_norm_, w.x_normed, T, H, eps));
        if (all_logits_dev) {
            // Spec verify: lm_head over ALL T rows (batched int-dot via
            // spec_verify_gemv_) → [T, vocab]; the host row is the last one.
            sgemv(hd, w.x_normed, output_, all_logits_dev, H, V, T).wait();
            q.memcpy(out_logits_host, all_logits_dev + uint64_t(T - 1) * V,
                     uint64_t(V) * sizeof(sycl::half)).wait();
        } else {
        const sycl::half* last = w.x_normed + uint64_t(T - 1) * H;
        sycl::half* d_logits = w.logits;   // persistent [V] (ensure_ws)
        if (!d_logits) return "logits alloc failed";
        // lm_head is a single-vector GEMV → sgemv with T=1 (Q8_0-SoA int-dot when packed).
        sgemv(hd, last, output_, d_logits, H, V, 1).wait();
        g_dp27.sync(hd);
        q.memcpy(out_logits_host, d_logits, uint64_t(V) * sizeof(sycl::half)).wait();
        }
        mark(q, pf_head);   // final norm + lm_head + logits bounce
        g_dp27.phase(DecodeProf::kHead);
    }
    g_dp27.end();

    if (prof) {
        pf_total += pms(pf_t0, pclk::now());
        pf_tok += T; pf_calls++;
        const double summed = pf_dn + pf_attn + pf_ffn + pf_head;
        pf_other = pf_total - summed;
        if (pf_calls % 32 == 0) {
            auto pct = [&](double x) { return pf_total > 0 ? 100.0 * x / pf_total : 0.0; };
            std::fprintf(stderr,
                "\n[qwen35split PROFILE] calls=%llu tok=%llu (wait-bracketed; relative)\n"
                "  total       %8.2f ms  (%.2f ms/tok = %.1f tok/s instrumented)\n"
                "  DeltaNet    %8.2f ms  %5.1f%%\n"
                "  full-attn   %8.2f ms  %5.1f%%\n"
                "  dense FFN   %8.2f ms  %5.1f%%\n"
                "  head+bounce %8.2f ms  %5.1f%%\n"
                "  other       %8.2f ms  %5.1f%%\n",
                (unsigned long long)pf_calls, (unsigned long long)pf_tok,
                pf_total, pf_total / double(pf_tok), 1000.0 * double(pf_tok) / pf_total,
                pf_dn, pct(pf_dn), pf_attn, pct(pf_attn), pf_ffn, pct(pf_ffn),
                pf_head, pct(pf_head), pf_other, pct(pf_other));
        }
    }
    return {};
}

// ===========================================================================
// SLOT BANKS + BATCHED MULTI-SEQUENCE DECODE (decode-throughput campaign 2a).
//
// A bank holds a full per-card (KvCache @ slot_ctx, DeltaNetState) pair.
// Prefill stays on the live kv_/dn_ singletons; bank_store() captures a
// finished prefill device-to-device; forward_slots() then decodes N slots per
// step reading/advancing state IN the banks. Kernel choices mirror the
// spec-verify configuration, whose per-row leaves are documented bit-identical
// to sequential T=1 steps — so a batched step equals N solo steps bit-for-bit
// on the Q8-packed model (the plain sgemv fallbacks below keep other dtypes
// functional without that guarantee).
// ===========================================================================
std::string Qwen35SplitModel::alloc_slot_banks(uint32_t n_slots, uint32_t slot_ctx) {
    free_slot_banks();
    if (n_slots == 0) return {};
    // ws_ scratch (FA2 partials etc.) is sized by the LIVE max_ctx; a deeper
    // slot_ctx would undersize it inside forward_slots (gate finding 2).
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (kv_[dev].ready() && slot_ctx > kv_[dev].config().max_ctx)
            return "alloc_slot_banks: slot_ctx " + std::to_string(slot_ctx) +
                   " exceeds live max_ctx " + std::to_string(kv_[dev].config().max_ctx);
    bank_kv_.resize(n_slots);
    bank_dn_.resize(n_slots);
    for (uint32_t s = 0; s < n_slots; ++s) {
        bank_kv_[s].resize(n_dev_);
        bank_dn_[s].resize(n_dev_);
        for (uint32_t dev = 0; dev < n_dev_; ++dev) {
            DeviceAllocator& a = fleet_->dev(dev);
            if (kv_[dev].ready()) {
                KvCacheConfig kc = kv_[dev].config();
                kc.max_ctx  = slot_ctx;
                kc.use_int8 = false;               // banks are fp16-only (v1)
                if (auto e = bank_kv_[s][dev].init(a, kc); !e.empty())
                    return "slot bank " + std::to_string(s) + " kv dev " +
                           std::to_string(dev) + ": " + e;
            }
            if (dn_[dev].ready()) {
                if (auto e = bank_dn_[s][dev].init(a, dn_[dev].config()); !e.empty())
                    return "slot bank " + std::to_string(s) + " dn dev " +
                           std::to_string(dev) + ": " + e;
            }
        }
    }
    {
        auto& q = fleet_->dev(plan_.head_dev).queue();
        d_slot_logits_ = sycl::malloc_device<sycl::half>(
            uint64_t(n_slots) * cfg_.dense.vocab, q);
        if (!d_slot_logits_) { free_slot_banks(); return "slot logits alloc failed"; }
    }
    n_banks_  = n_slots;
    slot_ctx_ = slot_ctx;
    return {};
}

void Qwen35SplitModel::free_slot_banks() {
    bank_kv_.clear();
    bank_dn_.clear();
    if (d_slot_logits_ && fleet_)
        sycl::free(d_slot_logits_, fleet_->dev(plan_.head_dev).queue());
    d_slot_logits_ = nullptr;
    n_banks_ = 0;
    slot_ctx_ = 0;
}

std::string Qwen35SplitModel::bank_store(uint32_t slot, uint32_t depth) {
    if (slot >= n_banks_)  return "bank_store: bad slot " + std::to_string(slot);
    if (depth > slot_ctx_) return "bank_store: depth " + std::to_string(depth) +
                                  " exceeds slot_ctx " + std::to_string(slot_ctx_);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        sycl::queue& q = fleet_->dev(dev).queue();
        if (kv_[dev].ready())
            if (auto e = bank_kv_[slot][dev].copy_prefix_from(q, kv_[dev], depth); !e.empty())
                return "bank_store kv dev " + std::to_string(dev) + ": " + e;
        if (dn_[dev].ready())
            if (auto e = bank_dn_[slot][dev].copy_from(q, dn_[dev]); !e.empty())
                return "bank_store dn dev " + std::to_string(dev) + ": " + e;
        q.wait();
    }
    return {};
}

std::string Qwen35SplitModel::bank_load(uint32_t slot, uint32_t depth) {
    if (slot >= n_banks_)  return "bank_load: bad slot " + std::to_string(slot);
    if (depth > slot_ctx_) return "bank_load: depth exceeds slot_ctx";
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        sycl::queue& q = fleet_->dev(dev).queue();
        if (kv_[dev].ready())
            if (auto e = kv_[dev].copy_prefix_from(q, bank_kv_[slot][dev], depth); !e.empty())
                return "bank_load kv dev " + std::to_string(dev) + ": " + e;
        if (dn_[dev].ready())
            if (auto e = dn_[dev].copy_from(q, bank_dn_[slot][dev]); !e.empty())
                return "bank_load dn dev " + std::to_string(dev) + ": " + e;
        q.wait();
    }
    return {};
}

// Per-card layer walk for one GROUP of decode slots. Pure ENQUEUE on the
// card's in-order queue (no host waits) so the caller can pipeline the two
// cards. Assumes ws_[dev].x holds the group's residual rows and
// ws_[dev].positions the group's per-slot positions. Kernel choices mirror
// spec-verify (per-row-exact leaves) — the caller holds spec_verify_gemv_.
std::string Qwen35SplitModel::stage_card_slots(uint32_t dev, uint32_t T,
                                               const uint32_t* positions,
                                               const uint32_t* slots) {
    // P4 B18: the walk is stage_card_rows over the slots' banks (a bank's KvCache max_ctx is slot_ctx_: the same KV layout and
    // attention arguments as before; sgemv_rows under spec_verify_gemv_ is sgemv)
    if (T == 0 || T > kMaxSlots) return "stage_card_slots: " + std::to_string(T) + " rows";
    KvCache* kv[kMaxSlots];
    DeltaNetState* dn[kMaxSlots];
    for (uint32_t i = 0; i < T; ++i) { kv[i] = &bank_kv_[slots[i]][dev]; dn[i] = &bank_dn_[slots[i]][dev]; }
    return stage_card_rows(dev, T, positions, kv, dn);
}

// P4 B18: the batched int-dot route forward_slots forces through spec_verify_gemv_ (sgemv's verify branch), without the
// model-wide flag: a lane pipe's other card may be running a prefill piece, whose T > 1 sgemv must keep the prefill route.
sycl::event Qwen35SplitModel::sgemv_rows(uint32_t dev, const sycl::half* A, const SplitW& w, sycl::half* out, uint32_t K,
                                         uint32_t N, uint32_t T) {
    if (w.q8_qs && T >= 2 && T <= 16) {
        auto& q = fleet_->dev(dev).queue();
        quantize_q8_1(q, A, act_q8_[dev], uint32_t(uint64_t(T) * K));
        return gemv_q8_0_soa_q8_batched_sel(q, act_q8_[dev], w.q8_qs, w.q8_d, out, K, N, T);
    }
    if (w.kq_lo && T <= 16) return kq_gemv(dev, A, w, out, K, N, T);
    return sgemv(dev, A, w, out, K, N, T);
}

// P4 B21: the c32 K-quant GEMV for T rows (T <= 16: act_q8_ holds 16 rows of Kmax/32 blocks). T = 1 (sgemv's decode) and
// T >= 2 (the rows / verify) are one kernel template, so a row's result does not depend on T.
sycl::event Qwen35SplitModel::kq_gemv(uint32_t dev, const sycl::half* A, const SplitW& w, sycl::half* out, uint32_t K,
                                      uint32_t N, uint32_t T, bool act_ready) {
    auto& q = fleet_->dev(dev).queue();
    if (!act_ready) DPE(dev, kQuant, quantize_q8_1(q, A, act_q8_[dev], uint32_t(uint64_t(T) * K)));
    sycl::event e;
    if (w.kq == DType::kQ6_K)
        e = gemv_q6k_c32_q8(q, act_q8_[dev], w.kq_lo, static_cast<const uint8_t*>(w.kq_hi),
                            static_cast<const int8_t*>(w.kq_sc), static_cast<const uint16_t*>(w.kq_d), out, K, N, T);
    else
        e = gemv_q5k_c32_q8(q, act_q8_[dev], w.kq_lo, static_cast<const uint32_t*>(w.kq_hi),
                            static_cast<const uint16_t*>(w.kq_sc), static_cast<const uint32_t*>(w.kq_d), out, K, N, T);
    if (g_dp27.active()) g_dp27.tag(dev, e, DecodeProf::kGemv);
    return e;
}

// P4 B18: stage_card_slots' walk (its body moved here verbatim but for the row state) over any rows' state: row i's conv /
// DeltaNet state is dn[i], its KV kv[i] (layout and attention bound by kv[i]'s own max_ctx), its position positions[i].
std::string Qwen35SplitModel::stage_card_rows(uint32_t dev, uint32_t T, const uint32_t* positions, KvCache* const* kv,
                                              DeltaNetState* const* dn) {
    const DenseConfig& dc = cfg_.dense;
    const uint32_t H     = dc.hidden;
    const uint32_t HD    = dc.head_dim;
    const uint32_t N_q   = dc.n_q_heads * HD;
    const uint32_t N_qg  = N_q * 2u;
    const uint32_t N_kv  = dc.n_kv_heads * HD;
    const uint32_t F     = dc.ffn;
    const uint32_t rope_n = dc.rope_dim;
    const float    eps   = dc.rms_eps;
    const uint32_t n_layers = cfg_.n_transformer_layers();
    Workspace& w = ws_[dev];
    auto& q = fleet_->dev(dev).queue();
    static const bool dec_vec = []{
        const char* e = std::getenv("IE_Q35_FA2_VEC");
        return !e || std::atoi(e) != 0;
    }();
    // P4 B26: the T rows' decode attention as one launch per pass per kernel kind (IE_Q35_ROWS_ATTN=0: the per-row loop
    // below). The plan is built once per group: row i keeps the kernel the loop would give it (XMX / vec / the dispatcher)
    // on its own bank, so the bytes are the loop's; each layer runs from the plan with its KV slot.
    FaDecodeRowsPlan rows_plan;
    bool rows_on = false;
    if (q35::rows_attn() && T >= 2 && T <= kMaxRows && dev < rows_partials_.size() && rows_partials_[dev] && rows_table_[dev]) {
        FaDecodeRow rows[kMaxRows];
        for (uint32_t i = 0; i < T; ++i) {
            const KvCache& bkv = *kv[i];
            const uint32_t row_ctx = bkv.config().max_ctx;
            rows[i].k_base = bkv.k_ptr();
            rows[i].v_base = bkv.v_ptr();
            rows[i].layer_stride = uint64_t(dc.n_kv_heads) * row_ctx * HD;
            rows[i].start_pos = positions[i];
            rows[i].max_ctx = row_ctx;
            rows[i].kind = q35::xmx_decode(positions[i], row_ctx) ? FaDecodeKind::kXmx
                         : dec_vec ? FaDecodeKind::kVec : FaDecodeKind::kAuto;
        }
        rows_on = fa2_decode_rows_plan(q, rows, T, dc.n_q_heads, dc.n_kv_heads, HD, rows_partials_floats_, rows_table_[dev], rows_plan);
        static const bool logged = [&] {
            std::fprintf(stderr, "[qwen35split] rows decode attention: %s (P4 B26: one launch per pass per kernel kind over the "
                         "group's rows; IE_Q35_ROWS_ATTN=0 = the per-row loop)\n", rows_on ? "batched" : "per-row (plan not batched)");
            return true;
        }();
        (void)logged;
    }

    for (uint32_t L = 0; L < n_layers; ++L) {
        if (plan_.dev_of_layer[L] != dev) continue;
        const LayerW& w_l = layers_[L];
        rms_norm_f32w(q, w.x, w_l.attn_norm, w.x_normed, T, H, eps);

        if (w_l.is_linear) {
            const uint32_t SKH = cfg_.ssm_n_k_heads;
            const uint32_t SVH = cfg_.ssm_n_v_heads;
            const uint32_t SHD = cfg_.ssm_state;
            const uint32_t SI  = cfg_.ssm_inner;
            const uint32_t conv_ch = SI + 2u * SKH * SHD;
            const uint32_t kw  = SKH * SHD;
            const uint32_t rep = SVH / SKH;
            const float qscale = 1.0f / sycl::sqrt(float(SHD));
            const uint64_t Vd  = uint64_t(SVH) * SHD;

            sgemv_rows(dev, w.x_normed, w_l.attn_qkv, w.dn_qkv, H, conv_ch, T);
            for (uint32_t i = 0; i < T; ++i) {
                DeltaNetState& bdn = *dn[i];
                sycl::half* conv_state = bdn.conv_state_ptr() +
                    uint64_t(dn_local_[L]) * bdn.conv_elems_per_layer();
                depthwise_conv1d_causal(q,
                    w.dn_qkv  + uint64_t(i) * conv_ch, w_l.ssm_conv1d_fp16,
                    conv_state,
                    w.dn_conv + uint64_t(i) * conv_ch,
                    /*T=*/1, conv_ch, cfg_.ssm_conv_kernel);
            }
            cast_qkv_split_fp16_to_fp32(q, w.dn_conv, w.dn_qpre, w.dn_kpre,
                                        w.dn_vpre, T, kw, SI);
            l2_norm_scale(q, w.dn_qpre, w.dn_qpre, T * SKH, SHD, qscale, 1e-6f);
            l2_norm_scale(q, w.dn_kpre, w.dn_kpre, T * SKH, SHD, 1.0f,   1e-6f);
            repeat_interleave_heads(q, w.dn_qpre, w.dn_qrep, T, SKH, SHD, rep);
            repeat_interleave_heads(q, w.dn_kpre, w.dn_krep, T, SKH, SHD, rep);
            const uint32_t SVHp = ((SVH + 63u) / 64u) * 64u;
            if (w_l.ssm_alpha.dt == DType::kF16 && w_l.ssm_beta.dt == DType::kF16) {
                gemv_fp16_rows_dual(q, w.x_normed, H,
                    static_cast<const sycl::half*>(w_l.ssm_alpha.p), w.dn_alpha64,
                    static_cast<const sycl::half*>(w_l.ssm_beta.p),  w.dn_beta64,
                    SVHp, H, SVHp, T);
            } else {
                for (uint32_t t = 0; t < T; ++t) {
                    dense::gemv_q_T(q, w.x_normed + uint64_t(t) * H, w_l.ssm_alpha,
                                    w.dn_alpha64 + uint64_t(t) * SVHp, H, SVHp, 1);
                    dense::gemv_q_T(q, w.x_normed + uint64_t(t) * H, w_l.ssm_beta,
                                    w.dn_beta64 + uint64_t(t) * SVHp, H, SVHp, 1);
                }
            }
            extract_cols(q, w.dn_alpha64, w.dn_alpha_h, T, SVH, SVHp);
            extract_cols(q, w.dn_beta64,  w.dn_beta_h,  T, SVH, SVHp);
            compute_g_beta_h16(q, w.dn_alpha_h, w.dn_beta_h, w_l.ssm_a,
                               w_l.ssm_dt_bias, w.dn_g, w.dn_beta, T, SVH);
            for (uint32_t i = 0; i < T; ++i) {
                DeltaNetState& bdn = *dn[i];
                float* state_slot = bdn.state_ptr() +
                    uint64_t(dn_local_[L]) * bdn.state_elems_per_layer();
                deltanet_recurrence(q,
                    w.dn_qrep + uint64_t(i) * Vd, w.dn_krep + uint64_t(i) * Vd,
                    w.dn_vpre + uint64_t(i) * Vd,
                    w.dn_g + uint64_t(i) * SVH,  w.dn_beta + uint64_t(i) * SVH,
                    state_slot, w.dn_out + uint64_t(i) * Vd,
                    /*B=*/1, /*T=*/1, SVH, SHD, SHD);
            }
            if (T >= 2 && T <= 16 && w_l.attn_gate.q8_qs) {
                gemv_q8_0_soa_q8_batched_sel(q, act_q8_[dev],
                    w_l.attn_gate.q8_qs, w_l.attn_gate.q8_d, w.dn_z, H, SI, T);
            } else {
                sgemv_rows(dev, w.x_normed, w_l.attn_gate, w.dn_z, H, SI, T);
            }
            gated_rms_norm(q, w.dn_out, w.dn_z, w_l.ssm_norm_fp16, w.dn_qkv,
                           T * SVH, SHD, eps);
            sgemv_rows(dev, w.dn_qkv, w_l.ssm_out, w.attn_block, SI, H, T);
        } else {
            if (T >= 2 && T <= 16 &&
                w_l.attn_q.q8_qs && w_l.attn_k.q8_qs && w_l.attn_v.q8_qs) {
                quantize_q8_1(q, w.x_normed, act_q8_[dev],
                              uint32_t(uint64_t(T) * H));
                gemv_q8_0_soa_q8_batched_sel(q, act_q8_[dev],
                    w_l.attn_q.q8_qs, w_l.attn_q.q8_d, w.qg, H, N_qg, T);
                split_q_gate_per_head(q, w.qg, w.q, w.gate, T, dc.n_q_heads, HD);
                gemv_q8_0_soa_q8_batched_dual_sel(q, act_q8_[dev],
                    w_l.attn_k.q8_qs, w_l.attn_k.q8_d, w.k,
                    w_l.attn_v.q8_qs, w_l.attn_v.q8_d, w.v, H, N_kv, T);
            } else {
                sgemv_rows(dev, w.x_normed, w_l.attn_q, w.qg, H, N_qg, T);
                split_q_gate_per_head(q, w.qg, w.q, w.gate, T, dc.n_q_heads, HD);
                sgemv_rows(dev, w.x_normed, w_l.attn_k, w.k, H, N_kv, T);
                sgemv_rows(dev, w.x_normed, w_l.attn_v, w.v, H, N_kv, T);
            }
            rms_norm_f32w(q, w.q, w_l.attn_q_norm, w.q, T * dc.n_q_heads,  HD, eps);
            rms_norm_f32w(q, w.k, w_l.attn_k_norm, w.k, T * dc.n_kv_heads, HD, eps);
            rope_partial(q, w.q, w.positions, w.q, T, dc.n_q_heads,  HD, rope_n, dc.rope_theta);
            rope_partial(q, w.k, w.positions, w.k, T, dc.n_kv_heads, HD, rope_n, dc.rope_theta);
            const uint32_t li = kv_local_[L];
            if (rows_on) {   // P4 B26 (the plan above)
                full_attention_fa2_decode_rows(q, rows_plan, w.q, w.k, w.v, w.attn_out, rows_partials_[dev], li);
                for (uint32_t i = 0; i < T; ++i) kv[i]->set_length(li, positions[i] + 1);
            } else
            for (uint32_t i = 0; i < T; ++i) {
                KvCache& bkv = *kv[i];
                const uint32_t row_ctx = bkv.config().max_ctx;   // (a bank's is the slot context)
                const uint64_t row_layer_kv = uint64_t(dc.n_kv_heads) * row_ctx * HD;
                sycl::half* kc = bkv.k_ptr() + row_layer_kv * li;
                sycl::half* vc = bkv.v_ptr() + row_layer_kv * li;
                if (q35::xmx_decode(positions[i], row_ctx))   // P4 B23: per row, the T == 1 decode's choice
                    full_attention_fa2_decode_xmx(q,
                        w.q + uint64_t(i) * N_q, w.k + uint64_t(i) * N_kv,
                        w.v + uint64_t(i) * N_kv, kc, vc,
                        w.attn_out + uint64_t(i) * N_q, w.attn_partials,
                        positions[i], dc.n_q_heads, dc.n_kv_heads, HD, row_ctx);
                else if (dec_vec)
                    full_attention_fa2_decode_vec(q,
                        w.q + uint64_t(i) * N_q, w.k + uint64_t(i) * N_kv,
                        w.v + uint64_t(i) * N_kv, kc, vc,
                        w.attn_out + uint64_t(i) * N_q, w.attn_partials,
                        positions[i], dc.n_q_heads, dc.n_kv_heads, HD, row_ctx);
                else
                    full_attention_fa2_decode(q,
                        w.q + uint64_t(i) * N_q, w.k + uint64_t(i) * N_kv,
                        w.v + uint64_t(i) * N_kv, kc, vc,
                        w.attn_out + uint64_t(i) * N_q, w.attn_partials,
                        positions[i], dc.n_q_heads, dc.n_kv_heads, HD, row_ctx);
                bkv.set_length(li, positions[i] + 1);
            }
            sigmoid_gate(q, w.attn_out, w.gate, w.attn_out, uint64_t(T) * N_q);
            sgemv_rows(dev, w.attn_out, w_l.attn_output, w.attn_block, N_q, H, T);
        }

        residual_add_rms_norm_fused(q, w.x, w.attn_block, w_l.post_attn_norm,
                                    w.x_normed, T, H, eps);
        if (T >= 2 && T <= 16 && w_l.ffn_gate.q8_qs && w_l.ffn_up.q8_qs) {
            quantize_q8_1(q, w.x_normed, act_q8_[dev],
                          uint32_t(uint64_t(T) * H));
            gemv_q8_0_soa_q8_batched_dual_sel(q, act_q8_[dev],
                w_l.ffn_gate.q8_qs, w_l.ffn_gate.q8_d, w.ffn_gate,
                w_l.ffn_up.q8_qs,   w_l.ffn_up.q8_d,   w.ffn_up,
                H, F, T);
        } else {
            sgemv_rows(dev, w.x_normed, w_l.ffn_gate, w.ffn_gate, H, F, T);
            sgemv_rows(dev, w.x_normed, w_l.ffn_up,   w.ffn_up,   H, F, T);
        }
        swiglu(q, w.ffn_gate, w.ffn_up, w.ffn_h, uint64_t(T) * F);
        sgemv_rows(dev, w.ffn_h, w_l.ffn_down, w.attn_block, F, H, T);
        residual_add(q, w.x, w.attn_block, w.x, uint64_t(T) * H);
    }
    return {};
}

std::string Qwen35SplitModel::forward_slots(uint32_t N, const int32_t* ids,
                                            const uint32_t* positions,
                                            const uint32_t* slots,
                                            sycl::half* out_logits_host) {
    if (N == 0)         return "forward_slots: N == 0";
    if (N > n_banks_)   return "forward_slots: N exceeds allocated banks";
    if (N > kMaxSlots)  return "forward_slots: N > 16 (batched int-dot leaf cap)";
    if (!d_slot_logits_) return "forward_slots: banks not allocated";
    for (uint32_t i = 0; i < N; ++i) {
        if (slots[i] >= n_banks_)
            return "forward_slots: bad slot index " + std::to_string(slots[i]);
        if (positions[i] >= slot_ctx_)
            return "forward_slots: slot position exceeds slot_ctx";
        for (uint32_t j = 0; j < i; ++j)
            if (slots[j] == slots[i])
                return "forward_slots: duplicate slot " + std::to_string(slots[i]);
    }
    const DenseConfig& dc = cfg_.dense;
    const uint32_t H = dc.hidden, V = dc.vocab;
    const float eps = dc.rms_eps;

    // Force sgemv's batched W8A8 int-dot dispatch for every projection (the
    // T>=2 non-verify default is the prefill dequant+fp16-GEMM route —
    // numerically different from solo decode and catastrophically slower).
    const bool saved_verify_ = spec_verify_gemv_;
    spec_verify_gemv_ = true;
    struct VerifyRestore { bool* f; bool v; ~VerifyRestore() { *f = v; } }
        verify_restore_{&spec_verify_gemv_, saved_verify_};

    if (ws_.size() != n_dev_) ws_.assign(n_dev_, {});
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (auto m = ensure_ws(dev, N); !m.empty()) return m;

    auto embed_group = [&](sycl::queue& q0, const int32_t* g_ids, uint32_t n) -> std::string {
        int32_t* d_ids = ws_[plan_.embed_dev].ids;
        if (!d_ids) return "d_ids alloc failed";
        q0.memcpy(d_ids, g_ids, n * sizeof(int32_t));
        if (token_embd_dtype_ == DType::kQ4_K)
            embedding_lookup_q4k(q0, d_ids, token_embd_, ws_[plan_.embed_dev].x, n, H);
        else if (token_embd_dtype_ == DType::kQ8_0)
            embedding_lookup_q8_0(q0, d_ids, token_embd_, ws_[plan_.embed_dev].x, n, H);
        else if (token_embd_dtype_ == DType::kQ5_K)
            embedding_lookup_q5k(q0, d_ids, token_embd_, ws_[plan_.embed_dev].x, n, H);
        else
            embedding_lookup_q6k(q0, d_ids, token_embd_, ws_[plan_.embed_dev].x, n, H);
        return {};
    };
    auto upload_pos = [&](sycl::queue& qq, uint32_t dev, const uint32_t* g_pos, uint32_t n) {
        std::vector<int32_t> pos(n);
        for (uint32_t t = 0; t < n; ++t) pos[t] = int32_t(g_pos[t]);
        qq.memcpy(ws_[dev].positions, pos.data(), n * sizeof(int32_t)).wait();
    };

    // ---- 2-card GROUP PIPELINE — MEASURED SLOWER, kept OPT-IN ONLY so the
    // negative result isn't re-discovered (2026-08-26: 128.1 vs 118.8 ms/step
    // at N=4/14K). Splitting the batch into groups makes each card read its
    // full weight set once PER GROUP (T=2 twice instead of T=4 once) — the
    // doubled weight traffic outweighs the card-overlap win while weights
    // dominate the step. IE_QWEN35_SLOTS_PIPELINE=1 enables. ----
    static const bool pipe_on = std::getenv("IE_QWEN35_SLOTS_PIPELINE") != nullptr;
    if (pipe_on && N >= 2 && n_dev_ == 2 && plan_.embed_dev == 0 && plan_.head_dev == 1) {
        auto& q0 = fleet_->dev(0).queue();
        auto& q1 = fleet_->dev(1).queue();
        const uint32_t nA = (N + 1) / 2;
        const uint32_t off[2] = {0, nA};
        const uint32_t cnt[2] = {nA, N - nA};
        static std::vector<sycl::half> hstage[2];
        sycl::event ev_feed[2];
        bool ev_valid[2] = {false, false};
        for (int g = 0; g < 2; ++g) {
            const uint32_t n = cnt[g];
            if (n == 0) break;
            const uint32_t o = off[g];
            if (hstage[g].size() < uint64_t(n) * H) hstage[g].resize(uint64_t(n) * H);
            upload_pos(q0, 0, positions + o, n);
            if (auto m = embed_group(q0, ids + o, n); !m.empty()) return m;
            if (auto m = stage_card_slots(0, n, positions + o, slots + o); !m.empty()) return m;
            if (ev_valid[g]) ev_feed[g].wait();
            q0.memcpy(hstage[g].data(), ws_[0].x, uint64_t(n) * H * sizeof(sycl::half)).wait();
            ev_feed[g] = q1.memcpy(ws_[1].x, hstage[g].data(), uint64_t(n) * H * sizeof(sycl::half));
            ev_valid[g] = true;
            upload_pos(q1, 1, positions + o, n);
            if (auto m = stage_card_slots(1, n, positions + o, slots + o); !m.empty()) return m;
            // head for this group (q1 in-order → after the group's layers)
            rms_norm_f32w(q1, ws_[1].x, output_norm_, ws_[1].x_normed, n, H, eps);
            sgemv(1, ws_[1].x_normed, output_, d_slot_logits_ + uint64_t(o) * V, H, V, n);
        }
        q1.wait();
        if (out_logits_host)
            q1.memcpy(out_logits_host, d_slot_logits_,
                      uint64_t(N) * V * sizeof(sycl::half)).wait();
        return {};
    }

    // ---- serial fallback (N==1, other topologies, or kill switch) ----
    upload_pos(fleet_->dev(0).queue(), 0, positions, N);
    if (n_dev_ > 1) upload_pos(fleet_->dev(1).queue(), 1, positions, N);
    {
        auto& q0 = fleet_->dev(plan_.embed_dev).queue();
        if (auto m = embed_group(q0, ids, N); !m.empty()) return m;
        q0.wait();
    }
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        auto& q = fleet_->dev(dev).queue();
        if (dev > 0)
            fleet_->copy_across(dev - 1, ws_[dev].x, dev, ws_[dev - 1].x,
                                uint64_t(N) * H * sizeof(sycl::half));
        if (auto m = stage_card_slots(dev, N, positions, slots); !m.empty()) return m;
        q.wait();
    }
    {
        const uint32_t hd = plan_.head_dev;
        Workspace& w = ws_[hd];
        auto& q = fleet_->dev(hd).queue();
        rms_norm_f32w(q, w.x, output_norm_, w.x_normed, N, H, eps);
        sgemv(hd, w.x_normed, output_, d_slot_logits_, H, V, N).wait();
        if (out_logits_host)
            q.memcpy(out_logits_host, d_slot_logits_,
                     uint64_t(N) * V * sizeof(sycl::half)).wait();
    }
    return {};
}

// ===========================================================================
// P4 B18: REQUEST LANES (docs/lanes/LANES_SERVE.md) -- the shared lanes module's view of the 27B split. A lane is one
// sequence's KV + DeltaNet state on every card (lane 0 = kv_/dn_). A lane step runs --parallel 1's launches for its piece: a
// prefill piece forward_pipelined's per-chunk walk (stage_card_prefill on the lane), a decode row forward_slots' walk
// (stage_card_rows), G decode rows of G lanes forward_slots' batched walk. Nothing here touches the slot banks.
// ===========================================================================
std::string Qwen35SplitModel::init_lanes(uint32_t n_lanes, uint32_t lane_ctx, uint32_t max_rows, uint64_t reserve_bytes,
                                         LanesAutoFit* fit) {
    if (fit) fit->cards.clear();
    else if (n_lanes < 2) return {};
    if (!fleet_ || n_dev_ != 2 || plan_.embed_dev != 0 || plan_.head_dev != 1)
        return "request lanes need the two-card plan (embedding on card 0, head on card 1)";
    if (!lane_kv_.empty()) return "request lanes are already allocated";
    if (n_lanes > kMaxRows) return "request lanes: " + std::to_string(n_lanes) + " lanes, at most " + std::to_string(kMaxRows);
    if (lane_ctx < 9 || lane_ctx > max_ctx_)
        return "lane context " + std::to_string(lane_ctx) + " outside [9, " + std::to_string(max_ctx_) + "]";
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (kv_[dev].ready() && kv_[dev].is_int8()) return "request lanes do not support --int8-kv";
    // the workspace for the largest step first (it only grows): a prefill piece and a whole rows group, so no stage thread
    // ever grows it mid-pipe (B14 gate #6's rule)
    if (ws_.size() != n_dev_) ws_.assign(n_dev_, {});
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (auto m = ensure_ws(dev, std::max<uint32_t>(max_rows, kMaxRows)); !m.empty()) return m;
    // and the prefill dequant scratch at its full size (sgemv grows it at a card's first prefill piece: never on a stage
    // thread mid-pipe, and counted in the free memory below)
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        uint64_t need = 0;
        for (uint32_t L = 0; L < layers_.size(); ++L) {
            if (plan_.dev_of_layer[L] != dev) continue;
            const LayerW& l = layers_[L];
            for (const SplitW* w : {&l.ffn_gate, &l.ffn_up, &l.ffn_down, &l.attn_q, &l.attn_k, &l.attn_v, &l.attn_output,
                                    &l.attn_qkv, &l.attn_gate, &l.ssm_out})
                if (w->q8_qs || w->kq_lo) need = std::max<uint64_t>(need, uint64_t(w->K) * w->N);   // (B21: c32 dequants too)
        }
        if (need > prefill_bt_cap_[dev]) {
            if (prefill_bt_[dev]) fleet_->dev(dev).free(prefill_bt_[dev]);
            prefill_bt_[dev] = static_cast<sycl::half*>(fleet_->dev(dev).malloc(need * sizeof(sycl::half)));
            prefill_bt_cap_[dev] = prefill_bt_[dev] ? need : 0;
            if (!prefill_bt_[dev]) return "request lanes: the prefill dequant scratch alloc failed on card " + std::to_string(dev);
        }
        std::fprintf(stderr, "[qwen35split] card %u prefill dequant scratch %.3f GiB (allocated before the lanes' free-memory check)\n",
                     dev, double(prefill_bt_cap_[dev]) * sizeof(sycl::half) / double(1ull << 30));
    }
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
        // the logits buffers allocated below (the head card) and the rows attention partials (P4 B26, every card with a KV
        // cache) come out of the free memory before the reserve is kept
        const uint64_t logits_b = dev == plan_.head_dev ? uint64_t(1 + kMaxRows) * cfg_.dense.vocab * sizeof(sycl::half) : 0ull;
        const uint64_t rows_partials_b = kv_[dev].ready()
            ? uint64_t(kMaxRows) * fa2_decode_rows_partials_floats(cfg_.dense.n_q_heads, cfg_.dense.n_kv_heads, cfg_.dense.head_dim,
                                                                  kv_[dev].config().max_ctx) * sizeof(float)
            : 0ull;
        if (fit) {   // P4 B30: measured here, picked below
            fit->cards.push_back({free_now, lane_bytes_[dev], 0, reserve_bytes + logits_b + rows_partials_b + fit->keep});
            continue;
        }
        if (auto m = q35m_lanes_fit(free_now, lane_bytes_[dev], n_lanes, lane_ctx, reserve_bytes + logits_b + rows_partials_b, dev); !m.empty()) return m;
    }
    if (fit) {
        n_lanes = fit->n = lanes_auto_fit(fit->cards, fit->n_max);
        if (n_lanes < 2) { lane_bytes_.clear(); return {}; }   // one lane: nothing allocated (the scratch and workspace keep their size)
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
    auto& hd = fleet_->dev(plan_.head_dev);
    lane_logits_ = static_cast<sycl::half*>(hd.malloc(uint64_t(cfg_.dense.vocab) * sizeof(sycl::half)));
    if (!lane_logits_) return "request lanes: logits buffer alloc failed";
    owned_[plan_.head_dev].push_back(lane_logits_);
    rows_logits_ = static_cast<sycl::half*>(hd.malloc(uint64_t(kMaxRows) * cfg_.dense.vocab * sizeof(sycl::half)));
    if (!rows_logits_) return "request lanes: rows logits buffer alloc failed";
    owned_[plan_.head_dev].push_back(rows_logits_);
    // P4 B26: the rows step's decode-attention partials, one slice per row (sized by lane 0's max_ctx, the largest cache),
    // and its per-group row table
    rows_partials_.assign(n_dev_, nullptr);
    rows_table_.assign(n_dev_, nullptr);
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (!kv_[dev].ready()) continue;
        rows_partials_floats_ = fa2_decode_rows_partials_floats(cfg_.dense.n_q_heads, cfg_.dense.n_kv_heads, cfg_.dense.head_dim,
                                                                kv_[dev].config().max_ctx);
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

void Qwen35SplitModel::select_lane(uint32_t lane) {
    for (uint32_t dev = 0; dev < n_dev_; ++dev) cur_[dev] = lane;
}

void Qwen35SplitModel::reset_state() {
    for (uint32_t dev = 0; dev < n_dev_; ++dev) {
        if (dn_at(dev).ready()) dn_at(dev).reset(fleet_->dev(dev).queue());
        if (kv_at(dev).ready()) kv_at(dev).reset();
    }
}

// forward() / forward_slots' embedding on the embed card: host ids in, n rows into ws_[embed_dev].x
std::string Qwen35SplitModel::embed_rows(const int32_t* ids, uint32_t n) {
    auto& q = fleet_->dev(plan_.embed_dev).queue();
    Workspace& w = ws_[plan_.embed_dev];
    if (!w.ids) return "d_ids alloc failed";
    q.memcpy(w.ids, ids, n * sizeof(int32_t)).wait();
    const uint32_t H = cfg_.dense.hidden;
    if (token_embd_dtype_ == DType::kQ4_K)      embedding_lookup_q4k(q, w.ids, token_embd_, w.x, n, H);
    else if (token_embd_dtype_ == DType::kQ8_0) embedding_lookup_q8_0(q, w.ids, token_embd_, w.x, n, H);
    else if (token_embd_dtype_ == DType::kQ5_K) embedding_lookup_q5k(q, w.ids, token_embd_, w.x, n, H);
    else                                        embedding_lookup_q6k(q, w.ids, token_embd_, w.x, n, H);
    return {};
}

// ---- P4 B45: image input (the header's note). ------------------------------------------------------------------------
std::string Qwen35SplitModel::set_vision(const float* rows_f32, uint32_t t0, uint32_t n_rows) {
    if (!rows_f32 || !n_rows) return "qwen35split set_vision: empty rows";
    if (uint64_t(t0) + n_rows > max_ctx_) return "qwen35split set_vision: span past max_ctx";
    const uint32_t H = cfg_.dense.hidden;
    const size_t row0 = vis_rows_.size() / H;
    vis_rows_.resize(vis_rows_.size() + size_t(n_rows) * H);
    for (size_t i = 0; i < size_t(n_rows) * H; ++i) vis_rows_[row0 * H + i] = sycl::half(rows_f32[i]);
    vis_spans_.push_back({t0, n_rows, uint32_t(row0)});
    return {};
}

void Qwen35SplitModel::set_mrope(const int32_t* pos3, uint32_t n_total, int32_t delta) {
    mrope3_.assign(pos3, pos3 + size_t(3) * n_total);
    mrope_n_ = n_total;
    mrope_delta_ = delta;
}

void Qwen35SplitModel::clear_vision() {
    vis_spans_.clear();
    vis_rows_.clear();
    mrope3_.clear();
    mrope_n_ = 0;
    mrope_delta_ = 0;
}

// The piece [start, start + T)'s positions onto card dev: the linear token positions (every path's KV index and, without
// vision, the rope) and, while vision is staged, the [3, T] M-RoPE slice the rope reads instead. One blocking copy each,
// as forward() did before B45.
void Qwen35SplitModel::upload_positions(uint32_t dev, uint32_t start, uint32_t T) {
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

// The staged projector rows over the piece's gathered embedding rows (f16 row memcpys on the embed card's in-order
// queue: after the gather, before layer 0). vis_rows_ outlives the copies (cleared at the next request's clear_vision).
void Qwen35SplitModel::splice_vision(sycl::queue& q, sycl::half* x, uint32_t start, uint32_t T) {
    if (vis_spans_.empty()) return;
    const uint32_t H = cfg_.dense.hidden;
    for (const Qwen4VisSplice& r : qwen4_vis_splice_ranges(vis_spans_, start, T))
        q.memcpy(x + uint64_t(r.dst) * H, vis_rows_.data() + uint64_t(r.src) * H, uint64_t(r.n) * H * sizeof(sycl::half));
}

std::string Qwen35SplitModel::forward_stage(uint32_t dev, uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0,
                                            sycl::half* x_host, bool pk) {
    if (T == 0) return "T == 0";
    if (lane >= n_lanes() || n_dev_ != 2 || dev >= 2 || !lane_logits_) return "forward_stage: no such lane/card (lanes not initialised?)";
    if (vision_active()) return "forward_stage: vision is staged but the lanes have no per-lane image state (P4 B45 step 3)";
    if (uint64_t(pos0) + T > lane_ctx(lane))
        return "forward_stage: rows [" + std::to_string(pos0) + ", " + std::to_string(pos0 + T) + ") past lane " +
               std::to_string(lane) + "'s " + std::to_string(lane_ctx(lane)) + " positions";
    if (auto m = ensure_ws(dev, T); !m.empty()) return m;
    cur_[dev] = lane;
    auto& q = fleet_->dev(dev).queue();
    Workspace& w = ws_[dev];
    const uint32_t H = cfg_.dense.hidden;
    // forward()'s per-card order: positions, the reset at pos 0, the embedding (card 0) or the residual in (card 1), the
    // layers, then the residual out (card 0) or the head on the last row (card 1)
    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(pos0 + t);
    q.memcpy(w.positions, pos.data(), T * sizeof(int32_t)).wait();
    if (pos0 == 0) {
        if (dn_at(dev).ready()) dn_at(dev).reset(q);
        if (kv_at(dev).ready()) kv_at(dev).reset();
    }
    const uint64_t xb = uint64_t(T) * H * sizeof(sycl::half);
    if (dev == plan_.embed_dev) { if (auto m = embed_rows(ids, T); !m.empty()) return m; }
    else q.memcpy(w.x, x_host, xb).wait();
    if (T > 1 || pk) {
        stage_card_prefill(dev, T, pos0);
    } else {
        KvCache* kv = &kv_at(dev);
        DeltaNetState* dn = &dn_at(dev);
        if (auto m = stage_card_rows(dev, 1, &pos0, &kv, &dn); !m.empty()) return m;
    }
    if (dev != plan_.head_dev) { q.memcpy(x_host, w.x, xb).wait(); return {}; }
    rms_norm_f32w(q, w.x + uint64_t(T - 1) * H, output_norm_, w.x_normed, 1, H, cfg_.dense.rms_eps);
    sgemv(dev, w.x_normed, output_, lane_logits_, H, cfg_.dense.vocab, 1).wait();
    return {};
}

std::string Qwen35SplitModel::forward_stage_rows(uint32_t dev, std::span<const uint32_t> lanes, const int32_t* ids,
                                                 const uint32_t* pos0, sycl::half* x_host) {
    const uint32_t G = uint32_t(lanes.size());
    if (G < 2 || G > kMaxRows) return "forward_stage_rows: " + std::to_string(G) + " rows (2.." + std::to_string(kMaxRows) + ")";
    if (n_dev_ != 2 || dev >= 2 || !rows_logits_) return "forward_stage_rows: no such card (lanes not initialised?)";
    for (uint32_t i = 0; i < G; ++i) {
        if (lanes[i] >= n_lanes()) return "forward_stage_rows: no lane " + std::to_string(lanes[i]);
        if (pos0[i] == 0 || pos0[i] >= lane_ctx(lanes[i]))
            return "forward_stage_rows: lane " + std::to_string(lanes[i]) + " at " + std::to_string(pos0[i]) + " (a decode row needs 1.." +
                   std::to_string(lane_ctx(lanes[i]) - 1) + ")";
        for (uint32_t j = 0; j < i; ++j)
            if (lanes[j] == lanes[i]) return "forward_stage_rows: lane " + std::to_string(lanes[i]) + " twice";
    }
    if (auto m = ensure_ws(dev, G); !m.empty()) return m;
    auto& q = fleet_->dev(dev).queue();
    Workspace& w = ws_[dev];
    const uint32_t H = cfg_.dense.hidden, V = cfg_.dense.vocab;
    int32_t pos[kMaxRows];
    for (uint32_t i = 0; i < G; ++i) pos[i] = int32_t(pos0[i]);
    q.memcpy(w.positions, pos, G * sizeof(int32_t)).wait();
    const uint64_t xb = uint64_t(G) * H * sizeof(sycl::half);
    if (dev == plan_.embed_dev) { if (auto m = embed_rows(ids, G); !m.empty()) return m; }
    else q.memcpy(w.x, x_host, xb).wait();
    KvCache* kv[kMaxRows];
    DeltaNetState* dn[kMaxRows];
    for (uint32_t i = 0; i < G; ++i) { kv[i] = &lane_kv(lanes[i], dev); dn[i] = &lane_dn(lanes[i], dev); }
    if (auto m = stage_card_rows(dev, G, pos0, kv, dn); !m.empty()) return m;
    if (dev != plan_.head_dev) { q.memcpy(x_host, w.x, xb).wait(); return {}; }
    // forward_slots' head: the G rows normalised, the batched lm_head
    rms_norm_f32w(q, w.x, output_norm_, w.x_normed, G, H, cfg_.dense.rms_eps);
    sgemv_rows(dev, w.x_normed, output_, rows_logits_, H, V, G).wait();
    return {};
}

std::string Qwen35SplitModel::rows_off_reason() const {
    if (lane_kv_.empty() || !rows_logits_) return "the request lanes are not initialised";
    if (!output_.int_dot()) return "the head is not Q8_0/Q6_K/Q5_K";
    // a projection without the batched int-dot (Q8_0-SoA or the c32 K-quants) would run its G rows through the prefill GEMM
    // (another rounding than one row's GEMV)
    for (uint32_t L = 0; L < layers_.size(); ++L) {
        const LayerW& l = layers_[L];
        const std::string at = " (layer " + std::to_string(L) + ")";
        if (l.is_linear) {
            if (!l.attn_qkv.int_dot() || !l.attn_gate.int_dot() || !l.ssm_out.int_dot())
                return "a DeltaNet projection is not Q8_0/Q6_K/Q5_K" + at;
        } else if (!l.attn_q.int_dot() || !l.attn_k.int_dot() || !l.attn_v.int_dot() || !l.attn_output.int_dot()) {
            return "an attention projection is not Q8_0/Q6_K/Q5_K" + at;
        }
        if (!l.ffn_gate.int_dot() || !l.ffn_up.int_dot() || !l.ffn_down.int_dot())
            return "an FFN projection is not Q8_0/Q6_K/Q5_K" + at;
    }
    return {};
}

// ===========================================================================
// PIPELINED PREFILL (2 cards): the serial layer-split leaves each card idle
// while the other computes. For prefill, chunk c+1's card-0 pass can overlap
// chunk c's card-1 pass (each card's in-order queue serializes ITS chunks; the
// only cross-card dependency is the residual handoff). Host staging is
// ping-ponged so the host never blocks on card 1. Same per-chunk math as
// forward() — a lean prefill-only stage (no prof/ckpt/int8-decode branches).
// ===========================================================================
void Qwen35SplitModel::stage_card_prefill(uint32_t dev, uint32_t T, uint32_t start_pos) {
    Workspace& w = ws_[dev];
    auto& q = fleet_->dev(dev).queue();
    const DenseConfig& dc = cfg_.dense;
    const uint32_t H    = dc.hidden;
    const uint32_t HD   = dc.head_dim;
    const uint32_t N_q  = dc.n_q_heads  * HD;
    const uint32_t N_qg = N_q * 2u;
    const uint32_t N_kv = dc.n_kv_heads * HD;
    const uint32_t F    = dc.ffn;
    const uint32_t rope_n = dc.rope_dim;
    const float    eps  = dc.rms_eps;
    const uint32_t n_layers = cfg_.n_transformer_layers();
    const uint32_t n_kv = dc.n_kv_heads;
    const uint64_t per_layer_kv =
        uint64_t(n_kv) * (kv_at(dev).ready() ? kv_at(dev).config().max_ctx : 0u) * HD;

    g_pf27.start(dev, q);
    sgemv_pad_ = true;
    for (uint32_t L = 0; L < n_layers; ++L) {
        if (plan_.dev_of_layer[L] != dev) continue;
        const LayerW& w_l = layers_[L];
        rms_norm_f32w(q, w.x, w_l.attn_norm, w.x_normed, T, H, eps);
        g_pf27.mark(dev, q, PfProf27::kElem);
        if (w_l.is_linear) {
            const uint32_t SKH = cfg_.ssm_n_k_heads;
            const uint32_t SVH = cfg_.ssm_n_v_heads;
            const uint32_t SHD = cfg_.ssm_state;
            const uint32_t SI  = cfg_.ssm_inner;
            const uint32_t conv_ch = SI + 2u * SKH * SHD;
            const uint32_t kw  = SKH * SHD;
            const uint32_t rep = SVH / SKH;
            const float qscale = 1.0f / sycl::sqrt(float(SHD));
            sgemv(dev, w.x_normed, w_l.attn_qkv, w.dn_qkv, H, conv_ch, T);
            sycl::half* conv_state = dn_at(dev).conv_state_ptr() +
                uint64_t(dn_local_[L]) * dn_at(dev).conv_elems_per_layer();
            depthwise_conv1d_causal(q, w.dn_qkv, w_l.ssm_conv1d_fp16, conv_state,
                                    w.dn_conv, T, conv_ch, cfg_.ssm_conv_kernel);
            cast_qkv_split_fp16_to_fp32(q, w.dn_conv, w.dn_qpre, w.dn_kpre, w.dn_vpre, T, kw, SI);
            l2_norm_scale(q, w.dn_qpre, w.dn_qpre, T * SKH, SHD, qscale, 1e-6f);
            l2_norm_scale(q, w.dn_kpre, w.dn_kpre, T * SKH, SHD, 1.0f,   1e-6f);
            repeat_interleave_heads(q, w.dn_qpre, w.dn_qrep, T, SKH, SHD, rep);
            repeat_interleave_heads(q, w.dn_kpre, w.dn_krep, T, SKH, SHD, rep);
            const uint32_t SVHp = ((SVH + 63u) / 64u) * 64u;
            dense::gemv_q_T(q, w.x_normed, w_l.ssm_alpha, w.dn_alpha64, H, SVHp, T);
            dense::gemv_q_T(q, w.x_normed, w_l.ssm_beta,  w.dn_beta64,  H, SVHp, T);
            extract_cols(q, w.dn_alpha64, w.dn_alpha_h, T, SVH, SVHp);
            extract_cols(q, w.dn_beta64,  w.dn_beta_h,  T, SVH, SVHp);
            compute_g_beta_h16(q, w.dn_alpha_h, w.dn_beta_h, w_l.ssm_a, w_l.ssm_dt_bias,
                               w.dn_g, w.dn_beta, T, SVH);
            g_pf27.mark(dev, q, PfProf27::kDnMisc);
            float* state_layer = dn_at(dev).state_ptr() +
                uint64_t(dn_local_[L]) * dn_at(dev).state_elems_per_layer();
            if (dn_scan_on() && T >= kAttnGemmMinT && SHD == 128)   // P4 B48
                deltanet_scan_prefill(q, w.dn_qrep, w.dn_krep, w.dn_vpre, w.dn_g, w.dn_beta, state_layer, w.dn_out, T, SVH);
            else
            deltanet_recurrence(q, w.dn_qrep, w.dn_krep, w.dn_vpre, w.dn_g, w.dn_beta,
                                state_layer, w.dn_out, /*B=*/1, T, SVH, SHD, SHD);
            g_pf27.mark(dev, q, PfProf27::kDnScan);
            sgemv(dev, w.x_normed, w_l.attn_gate, w.dn_z, H, SI, T);
            gated_rms_norm(q, w.dn_out, w.dn_z, w_l.ssm_norm_fp16, w.dn_qkv,
                           T * SVH, SHD, eps);
            g_pf27.mark(dev, q, PfProf27::kDnMisc);
            sgemv(dev, w.dn_qkv, w_l.ssm_out, w.attn_block, SI, H, T);
        } else {
            sgemv(dev, w.x_normed, w_l.attn_q, w.qg, H, N_qg, T);
            split_q_gate_per_head(q, w.qg, w.q, w.gate, T, dc.n_q_heads, HD);
            sgemv(dev, w.x_normed, w_l.attn_k, w.k, H, N_kv, T);
            sgemv(dev, w.x_normed, w_l.attn_v, w.v, H, N_kv, T);
            rms_norm_f32w(q, w.q, w_l.attn_q_norm, w.q, T * dc.n_q_heads,  HD, eps);
            rms_norm_f32w(q, w.k, w_l.attn_k_norm, w.k, T * dc.n_kv_heads, HD, eps);
            if (mrope_n_) {   // P4 B45 vision: the 3-stream M-RoPE (== rope_partial when the streams are equal)
                rope_imrope3(q, w.q, w.positions3, w.q, T, dc.n_q_heads,  HD, rope_n, dc.rope_theta);
                rope_imrope3(q, w.k, w.positions3, w.k, T, dc.n_kv_heads, HD, rope_n, dc.rope_theta);
            } else {
            rope_partial(q, w.q, w.positions, w.q, T, dc.n_q_heads,  HD, rope_n, dc.rope_theta);
            rope_partial(q, w.k, w.positions, w.k, T, dc.n_kv_heads, HD, rope_n, dc.rope_theta);
            }
            g_pf27.mark(dev, q, PfProf27::kElem);
            auto& kvc = kv_at(dev);
            const uint32_t li = kv_local_[L];
            sycl::half* kc = kvc.k_ptr() + per_layer_kv * li;
            sycl::half* vc = kvc.v_ptr() + per_layer_kv * li;
            const uint32_t max_ctx = kvc.config().max_ctx;
            // prefill leaves only (same gates as forward): tile at ctx≥minctx.
            static const bool no_tile = std::getenv("IE_QWEN35_NO_FA2_TILE") != nullptr;
            static const uint32_t tile_minctx = []() -> uint32_t {
                const char* e = std::getenv("IE_QWEN35_FA2_TILE_MINCTX");
                if (!e) return 6144u;
                int v = std::atoi(e); return v > 0 ? uint32_t(v) : 6144u;
            }();
            if (w.ag.s && T >= kAttnGemmMinT) {   // P4 B48
                full_attention_prefill_gemm(q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                                            dc.n_q_heads, dc.n_kv_heads, HD, max_ctx, w.ag);
            } else if (!no_tile && HD == 256 && (start_pos + T) >= tile_minctx) {
                full_attention_fa2_prefill_tile_gemma(
                    q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                    dc.n_q_heads, dc.n_kv_heads, HD, max_ctx, 0);
            } else {
                full_attention(q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                               dc.n_q_heads, dc.n_kv_heads, HD, max_ctx);
            }
            if (kvc.is_int8()) kvc.quantize_to_int8(q, li, start_pos, T);
            kvc.set_length(li, start_pos + T);
            g_pf27.mark(dev, q, PfProf27::kAttn);
            sigmoid_gate(q, w.attn_out, w.gate, w.attn_out, uint64_t(T) * N_q);
            g_pf27.mark(dev, q, PfProf27::kElem);
            sgemv(dev, w.attn_out, w_l.attn_output, w.attn_block, N_q, H, T);
        }
        residual_add_rms_norm_fused(q, w.x, w.attn_block, w_l.post_attn_norm,
                                    w.x_normed, T, H, eps);
        g_pf27.mark(dev, q, PfProf27::kElem);
        sgemv(dev, w.x_normed, w_l.ffn_gate, w.ffn_gate, H, F, T);
        sgemv(dev, w.x_normed, w_l.ffn_up,   w.ffn_up,   H, F, T);
        swiglu(q, w.ffn_gate, w.ffn_up, w.ffn_h, uint64_t(T) * F);
        g_pf27.mark(dev, q, PfProf27::kElem);
        sgemv(dev, w.ffn_h, w_l.ffn_down, w.attn_block, F, H, T);
        residual_add(q, w.x, w.attn_block, w.x, uint64_t(T) * H);
        g_pf27.mark(dev, q, PfProf27::kElem);
    }
    sgemv_pad_ = false;
    g_pf27.stop(dev);
}

std::string Qwen35SplitModel::forward_pipelined(const int32_t* ids, uint32_t T_total,
                                                uint32_t start_pos, bool reset_kv,
                                                uint32_t chunk,
                                                sycl::half* out_logits_host,
                                                sycl::half* hidden_last_dev) {
    if (T_total == 0) return "pipeline: T == 0";
    if (n_dev_ != 2 || plan_.embed_dev != 0 || plan_.head_dev != 1 || chunk == 0)
        return "pipeline: unsupported topology";
    const DenseConfig& dc = cfg_.dense;
    const uint32_t H = dc.hidden, V = dc.vocab;
    const float eps = dc.rms_eps;
    const uint32_t max_T = std::min<uint32_t>(chunk, T_total);
    if (ws_.size() != n_dev_) ws_.assign(n_dev_, {});   // same guard as forward()
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (auto m = ensure_ws(dev, max_T); !m.empty()) return m;
    if (reset_kv) {
        for (uint32_t dev = 0; dev < n_dev_; ++dev) {
            if (dn_[dev].ready()) dn_[dev].reset(fleet_->dev(dev).queue());
            if (kv_[dev].ready()) kv_[dev].reset();
        }
    }
    std::vector<int32_t> pos_all(T_total);
    for (uint32_t t = 0; t < T_total; ++t) pos_all[t] = int32_t(start_pos + t);
    // P4 B45 vision: each chunk's [3, n] M-RoPE slice, host-side, alive until q1.wait() below (the copies are enqueued
    // without a wait like the positions above: the host must not wait on card 1's compute).
    std::vector<std::vector<int32_t>> p3_chunks;

    auto& q0 = fleet_->dev(0).queue();
    auto& q1 = fleet_->dev(1).queue();
    // Ping-pong host staging for the residual handoff — the host waits only on
    // card 0 (fast) and on card 1's FEED of two chunks ago (frees the buffer),
    // never on card 1's compute → card 0 runs chunk c+1 while card 1 runs c.
    std::vector<sycl::half> hstage[2];
    hstage[0].resize(uint64_t(max_T) * H);
    hstage[1].resize(uint64_t(max_T) * H);
    sycl::event ev_feed[2];
    bool ev_valid[2] = {false, false};

    uint32_t pos = 0, c = 0, last_n = 0;
    while (pos < T_total) {
        const uint32_t n  = std::min<uint32_t>(chunk, T_total - pos);
        last_n = n;
        const uint32_t sp = start_pos + pos;
        // card 0: ids + positions + embed + its layers (enqueue, in-order).
        q0.memcpy(ws_[0].ids, ids + pos, n * sizeof(int32_t));
        q0.memcpy(ws_[0].positions, pos_all.data() + pos, n * sizeof(int32_t));
        const int32_t* p3 = nullptr;
        if (mrope_n_) {
            p3_chunks.emplace_back(size_t(3) * n);
            qwen4_mrope3_slice(mrope3_.data(), mrope_n_, mrope_delta_, sp, n, p3_chunks.back().data());
            p3 = p3_chunks.back().data();
            q0.memcpy(ws_[0].positions3, p3, size_t(3) * n * sizeof(int32_t));
        }
        if (token_embd_dtype_ == DType::kQ4_K)
            embedding_lookup_q4k(q0, ws_[0].ids, token_embd_, ws_[0].x, n, H);
        else if (token_embd_dtype_ == DType::kQ8_0)
            embedding_lookup_q8_0(q0, ws_[0].ids, token_embd_, ws_[0].x, n, H);
        else if (token_embd_dtype_ == DType::kQ5_K)
            embedding_lookup_q5k(q0, ws_[0].ids, token_embd_, ws_[0].x, n, H);
        else
            embedding_lookup_q6k(q0, ws_[0].ids, token_embd_, ws_[0].x, n, H);
        splice_vision(q0, ws_[0].x, sp, n);   // P4 B45: the image rows (in-order: after the gather)
        stage_card_prefill(0, n, sp);
        // residual → host stage (waits card 0 only; in-order queue orders the copy
        // after the layer ops). Reusing stage s requires card 1's feed of chunk
        // c-2 to have completed (ping-pong invariant).
        const int s = int(c & 1);
        if (ev_valid[s]) ev_feed[s].wait();
        q0.memcpy(hstage[s].data(), ws_[0].x, uint64_t(n) * H * sizeof(sycl::half)).wait();
        // card 1: feed + positions + its layers — ENQUEUE ONLY (the overlap:
        // the host loops on to card 0's next chunk while card 1 computes).
        ev_feed[s] = q1.memcpy(ws_[1].x, hstage[s].data(), uint64_t(n) * H * sizeof(sycl::half));
        ev_valid[s] = true;
        q1.memcpy(ws_[1].positions, pos_all.data() + pos, n * sizeof(int32_t));
        if (p3) q1.memcpy(ws_[1].positions3, p3, size_t(3) * n * sizeof(int32_t));
        stage_card_prefill(1, n, sp);
        pos += n; ++c;
    }
    q1.wait();
    // final norm + lm_head on head_dev(=1): last token's logits → host. Only the
    // LAST row of the final chunk matters (same math as forward()'s tail on that row).
    {
        Workspace& w = ws_[1];
        const sycl::half* xlast = w.x + uint64_t(last_n - 1) * H;
        if (hidden_last_dev)   // spec conditioning: pre-norm hidden of the last token
            q1.memcpy(hidden_last_dev, xlast, uint64_t(H) * sizeof(sycl::half));
        rms_norm_f32w(q1, xlast, output_norm_, w.x_normed, 1, H, eps);
        sycl::half* d_logits = w.logits;   // persistent [V] (ensure_ws, head_dev)
        if (!d_logits) return "pipeline: logits scratch missing";
        sgemv(1, w.x_normed, output_, d_logits, H, V, 1).wait();
        q1.memcpy(out_logits_host, d_logits, uint64_t(V) * sizeof(sycl::half)).wait();
    }
    g_pf27.report(T_total);
    return {};
}

// ===========================================================================
// MTP self-speculative decode on the split path (--spec, greedy/lossless).
// Same draft→verify→accept→commit machinery as Qwen35DenseModel::spec_loop_,
// with per-card KV/DeltaNet bookkeeping. The NextN head lives on head_dev.
// ===========================================================================
std::string Qwen35SplitModel::load_mtp_head(const GgufReader& g, uint32_t max_ctx) {
    if (!fleet_) return "load_mtp_head: model not loaded";
    if (auto e = mtp_.load(fleet_->dev(plan_.head_dev), g, cfg_, max_ctx); !e.empty()) return e;
    // Borrow the model's Q8_0-SoA lm_head (head_dev = the MTP head's device):
    // halves the draft's dominant per-step head-GEMV bytes.
    if (output_.q8_qs) mtp_.use_soa_lm_head(output_.q8_qs, output_.q8_d);
    return {};
}

// ===========================================================================
// P4 B48: speculation as a step of a normal decode loop (the header's note).
// ===========================================================================
std::string Qwen35SplitModel::spec_step_init(uint32_t K) {
    if (!mtp_.loaded) return "spec_step: MTP head not loaded";
    const uint32_t V = std::min<uint32_t>(16u, std::max<uint32_t>(2u, K));
    SpecStep& s = ss_;
    if (s.V == V) return {};
    if (s.V != 0) return "spec_step: already initialised for another K";
    const uint32_t H = cfg_.dense.hidden, vocab = cfg_.dense.vocab, hd = plan_.head_dev;
    auto& ha = fleet_->dev(hd);
    s.ckpts.clear();
    s.ckpts.resize(n_dev_);
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (dn_[dev].ready())
            if (auto e = s.ckpts[dev].init(fleet_->dev(dev), dn_[dev], V); !e.empty())
                return "spec_step ckpt dev " + std::to_string(dev) + ": " + e;
    s.kvsnap.assign(n_dev_, {});
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (kv_[dev].ready()) s.kvsnap[dev].resize(kv_[dev].config().n_layers_full);
    s.d_hid  = static_cast<sycl::half*>(ha.malloc(uint64_t(V) * H * sizeof(sycl::half)));
    s.d_all  = static_cast<sycl::half*>(ha.malloc(uint64_t(V) * vocab * sizeof(sycl::half)));
    s.h_last = static_cast<sycl::half*>(ha.malloc(uint64_t(H) * sizeof(sycl::half)));
    for (void* p : {static_cast<void*>(s.d_hid), static_cast<void*>(s.d_all), static_cast<void*>(s.h_last)})
        if (p) owned_[hd].push_back(p);
    if (!s.d_hid || !s.d_all || !s.h_last) return "spec_step scratch alloc failed";
    s.vin.assign(V, 0);
    s.V = V;
    return {};
}

void Qwen35SplitModel::spec_step_reset() {
    static const bool prof = std::getenv("IE_QWEN35_PROFILE") != nullptr;
    if (prof && (ss_.rounds || ss_.plain))
        std::fprintf(stderr, "[qwen35split SPEC-STEP] rounds=%llu rows answered without a forward=%llu plain steps=%llu "
                     "(%.3f tokens a forward) cooldowns=%llu\n", (unsigned long long)ss_.rounds, (unsigned long long)ss_.hits,
                     (unsigned long long)ss_.plain,
                     double(ss_.rounds + ss_.hits + ss_.plain) / double(ss_.rounds + ss_.plain),
                     (unsigned long long)ss_.cooldowns);
    ss_.have_h = false;
    ss_.n_rows = ss_.next = 0;
    ss_.rounds = ss_.hits = ss_.plain = ss_.cooldowns = 0;
    ss_.win_fwd = ss_.win_tok = ss_.cooldown = 0;
}

std::string Qwen35SplitModel::spec_step_flush() {
    SpecStep& s = ss_;
    if (s.next < s.n_rows) {   // rows [next, n_rows) were not followed: the state goes back to base + next
        const uint32_t accepted = s.next;
        for (uint32_t dev = 0; dev < n_dev_; ++dev) {
            if (dn_[dev].ready())
                if (auto e = s.ckpts[dev].commit_to_n(fleet_->dev(dev).queue(), dn_[dev], accepted); !e.empty())
                    return "spec_step commit dev " + std::to_string(dev) + ": " + e;
            for (uint32_t l = 0; l < uint32_t(s.kvsnap[dev].size()); ++l)
                kv_[dev].set_length(l, s.kvsnap[dev][l] + accepted);
        }
        const uint32_t H = cfg_.dense.hidden;
        fleet_->dev(plan_.head_dev).queue().memcpy(s.h_last, s.d_hid + uint64_t(accepted - 1) * H,
                                                   uint64_t(H) * sizeof(sycl::half)).wait();
        s.have_h = true;
        s.h_pos = s.base + accepted - 1;
    }
    s.n_rows = s.next = 0;
    return {};
}

std::string Qwen35SplitModel::spec_step(int32_t token, uint32_t pos, sycl::half* logits_host,
                                        const SpecDraftSampling* ds) {
    SpecStep& s = ss_;
    if (s.V == 0) return "spec_step: not initialised";
    const uint32_t H = cfg_.dense.hidden, vocab = cfg_.dense.vocab;
    auto& qh = fleet_->dev(plan_.head_dev).queue();
    const uint64_t row_bytes = uint64_t(vocab) * sizeof(sycl::half);

    // A verified row for exactly this position and input token: its logits are already there.
    if (s.next < s.n_rows && pos == s.base + s.next && token == s.vin[s.next]) {
        qh.memcpy(logits_host, s.d_all + uint64_t(s.next) * vocab, row_bytes).wait();
        ++s.next; ++s.hits; ++s.win_tok;
        if (s.next == s.n_rows) {   // the round is used up: the state is at pos + 1, as after a plain step
            qh.memcpy(s.h_last, s.d_hid + uint64_t(s.n_rows - 1) * H, uint64_t(H) * sizeof(sycl::half)).wait();
            s.have_h = true; s.h_pos = pos;
            s.n_rows = s.next = 0;
        }
        return {};
    }
    if (auto e = spec_step_flush(); !e.empty()) return e;

    // A round costs a draft and V rows where a plain step costs one row (73 against 59 ms at V = 3): below about 1.25
    // tokens a forward speculation loses. Every kSpecWindow forwards the window's rate is checked, and a rate under
    // kSpecMinRate runs kSpecCooldown plain steps (each exports its hidden, so the next round starts at once).
    // (24 / 1.3 / 64 fired on ordinary swings and cost 5-11 % on two of four server requests, 2026-10-04.)
    constexpr uint32_t kSpecWindow = 64, kSpecCooldown = 32;
    constexpr double   kSpecMinRate = 1.2;
    if (s.win_fwd >= kSpecWindow) {
        if (double(s.win_tok) / double(s.win_fwd) < kSpecMinRate) { s.cooldown = kSpecCooldown; ++s.cooldowns; }
        s.win_fwd = s.win_tok = 0;
    }
    const uint32_t max_pos = kv_[plan_.head_dev].ready() ? kv_[plan_.head_dev].config().max_ctx : 0u;
    if (!s.have_h || s.h_pos + 1 != pos || pos + s.V > max_pos || s.cooldown > 0) {
        if (s.cooldown > 0) --s.cooldown;
        // No hidden for position pos - 1 (the first step after a prefill or a restore), or no room for a round: one
        // plain step, which exports its hidden.
        if (auto e = forward(&token, 1, pos, /*reset_kv=*/pos == 0, logits_host, nullptr, s.d_hid, nullptr); !e.empty())
            return e;
        qh.memcpy(s.h_last, s.d_hid, uint64_t(H) * sizeof(sycl::half)).wait();
        s.have_h = true; s.h_pos = pos;
        ++s.plain;
        return {};
    }

    // Draft V - 1 tokens from (h_last, token); verify [token, drafts] in one forward (checkpoint mode).
    s.drafted.clear();
    if (ds && ds->temperature > 0.f)
        mtp_.draft_device_sampled(qh, s.h_last, token, /*p_base=*/0, s.V, s.drafted,
                                  ds->temperature, ds->top_k, ds->top_p, ds->min_p, ds->seed);
    else
        mtp_.draft_device_argmax(qh, s.h_last, token, /*p_base=*/0, s.V, s.drafted);
    if (s.drafted.size() + 1 < s.V) return "spec_step: draft too short";
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        for (uint32_t l = 0; l < uint32_t(s.kvsnap[dev].size()); ++l)
            s.kvsnap[dev][l] = kv_[dev].length(l);
    s.vin[0] = token;
    for (uint32_t j = 1; j < s.V; ++j) s.vin[j] = s.drafted[j - 1];
    spec_verify_gemv_ = true;    // sgemv T in [2, 16] -> the batched int-dot (the weights read once)
    const std::string e = forward(s.vin.data(), s.V, pos, /*reset_kv=*/false, logits_host, s.d_all, s.d_hid, &s.ckpts);
    spec_verify_gemv_ = false;
    if (!e.empty()) return e;
    qh.memcpy(logits_host, s.d_all, row_bytes).wait();   // row 0 = this position
    s.base = pos; s.n_rows = s.V; s.next = 1;
    ++s.rounds; ++s.win_fwd; ++s.win_tok;
    return {};
}

namespace {
inline int32_t split_argmax_row(const sycl::half* row, uint32_t n) {
    int32_t best_i = 0; float best = float(row[0]);
    for (uint32_t i = 1; i < n; ++i) {
        const float x = float(row[i]);
        if (x > best) { best = x; best_i = int32_t(i); }
    }
    return best_i;
}
}  // namespace

std::string Qwen35SplitModel::spec_generate(const int32_t* ids, uint32_t P0,
                                            uint32_t pf_chunk, uint32_t max_new,
                                            uint32_t K, const SpecEmit& emit,
                                            double* prefill_ms_out) {
    if (!mtp_.loaded) return "spec_generate: MTP head not loaded";
    if (K < 2 || K > 16) return "spec_generate: K must be in [2,16]";
    if (P0 == 0) return "spec_generate: empty prompt";
    if (pf_chunk == 0) pf_chunk = 256;
    const uint32_t H     = cfg_.dense.hidden;
    const uint32_t vocab = cfg_.dense.vocab;
    const uint32_t V     = K;                 // verify length (tn + K-1 drafts)
    const uint32_t hd    = plan_.head_dev;
    auto& ha = fleet_->dev(hd);
    auto& qh = ha.queue();

    // Per-card per-position DeltaNet checkpoints (cards that own linear layers).
    std::vector<Qwen35SpecCheckpoint> ckpts(n_dev_);
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (dn_[dev].ready())
            if (auto e = ckpts[dev].init(fleet_->dev(dev), dn_[dev], V); !e.empty())
                return "spec ckpt dev " + std::to_string(dev) + ": " + e;

    // head_dev scratch: conditioning hidden (per chunk / per verify), all-row
    // verify logits, and the live h_last conditioning vector.
    const uint32_t hid_rows = std::max<uint32_t>(V, pf_chunk);
    auto* d_hid  = static_cast<sycl::half*>(ha.malloc(uint64_t(hid_rows) * H * sizeof(sycl::half)));
    auto* d_all  = static_cast<sycl::half*>(ha.malloc(uint64_t(V) * vocab * sizeof(sycl::half)));
    auto* h_last = static_cast<sycl::half*>(ha.malloc(uint64_t(H) * sizeof(sycl::half)));
    auto* d_targ = static_cast<int32_t*>(ha.malloc(uint64_t(V) * sizeof(int32_t)));
    auto free_scratch = [&]() {
        if (d_hid)  ha.free(d_hid);
        if (d_all)  ha.free(d_all);
        if (h_last) ha.free(h_last);
        if (d_targ) ha.free(d_targ);
    };
    if (!d_hid || !d_all || !h_last || !d_targ) { free_scratch(); return "spec scratch alloc failed"; }

    std::vector<sycl::half> row(vocab);                       // host logits (last row)

    // ---- chunked spec prefill (fresh sequence; hidden exported per chunk) ----
    const auto t_pf0 = std::chrono::steady_clock::now();
    static const bool pipeline_on = []{
        // default ON (mirrors engine prefill_to; see the 2026-08-15 A/B there).
        if (std::getenv("IE_QWEN35_NO_PIPELINE")) return false;
        const char* e = std::getenv("IE_QWEN35_PIPELINE");
        return !e || std::atoi(e) != 0;
    }();
    bool prefilled = false;
    if (pipeline_on && P0 > pf_chunk) {
        if (auto e = forward_pipelined(ids, P0, /*start_pos=*/0, /*reset_kv=*/true,
                                       pf_chunk, row.data(), h_last); e.empty())
            prefilled = true;
        else
            std::fprintf(stderr, "[spec pipeline] %s — serial prefill\n", e.c_str());
    }
    if (!prefilled) {
        uint32_t pos = 0; bool first = true;
        while (pos < P0) {
            const uint32_t n = std::min<uint32_t>(pf_chunk, P0 - pos);
            if (auto e = forward(ids + pos, n, pos, /*reset_kv=*/first, row.data(),
                                 /*all_logits=*/nullptr, /*hidden=*/d_hid,
                                 /*ckpts=*/nullptr); !e.empty()) {
                free_scratch(); return "spec prefill: " + e;
            }
            qh.memcpy(h_last, d_hid + uint64_t(n - 1) * H,
                      uint64_t(H) * sizeof(sycl::half)).wait();
            pos += n; first = false;
        }
    }
    if (prefill_ms_out)
        *prefill_ms_out = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_pf0).count();
    int32_t tn = split_argmax_row(row.data(), vocab);   // target token @ last prompt pos
    // IE_SPEC_DEBUG: mirror of the TP dump (differential draft debugging).
    static const bool spec_dbg = std::getenv("IE_SPEC_DEBUG") != nullptr;
    if (spec_dbg) {
        std::vector<sycl::half> hh(H);
        qh.memcpy(hh.data(), h_last, H * sizeof(sycl::half)).wait();
        double s = 0; for (uint32_t i = 0; i < H; ++i) s += double(float(hh[i]));
        std::fprintf(stderr, "[spec-dbg split] tn=%d h_last[0..3]=%.6f %.6f %.6f %.6f sum=%.4f\n",
                     tn, float(hh[0]), float(hh[1]), float(hh[2]), float(hh[3]), s);
    }

    // ---- draft → verify(T=V, checkpoint mode) → accept → commit → emit ----
    spec_verify_gemv_ = true;   // sgemv T∈[2,16] → batched int-dot (weight read ×1)
    std::string ret;
    uint32_t emitted = 0;
    uint32_t p = P0;
    bool first_round = true, abort = false;
    static const bool prof = std::getenv("IE_QWEN35_PROFILE") != nullptr;
    static const bool tree_probe = std::getenv("IE_SPEC_TREE_PROBE") != nullptr;
    uint32_t probe_rounds = 0, probe_miss0 = 0, probe_rec = 0;
    double ms_draft = 0, ms_verify = 0, ms_rest = 0, ms_vfwd = 0;
    uint32_t prof_rounds = 0;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms  = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::vector<int32_t> drafted, vin(V), targ(V);
    std::vector<std::vector<uint32_t>> kvsnap(n_dev_);
    for (uint32_t dev = 0; dev < n_dev_; ++dev)
        if (kv_[dev].ready()) kvsnap[dev].resize(kv_[dev].config().n_layers_full);

    while (emitted < max_new && !abort) {
        // 1. DRAFT V tokens from (h_last, tn) — the loop consumes g_1..g_{V-1}.
        drafted.clear();
        const auto t0 = now();
        static const bool host_draft = std::getenv("IE_SPEC_HOST_DRAFT") != nullptr;
        if (host_draft)
            mtp_.draft(qh, h_last, tn, /*p_base=*/0, V, drafted);
        else
            mtp_.draft_device_argmax(qh, h_last, tn, /*p_base=*/0, V, drafted);
        const auto t1 = now();
        if (drafted.size() + 1 < V) { ret = "spec: draft too short"; break; }

        // 2. VERIFY: split forward(T=V) in checkpoint mode; snapshot KV lengths.
        for (uint32_t dev = 0; dev < n_dev_; ++dev)
            for (uint32_t l = 0; l < uint32_t(kvsnap[dev].size()); ++l)
                kvsnap[dev][l] = kv_[dev].length(l);
        vin[0] = tn;
        for (uint32_t j = 1; j < V; ++j) vin[j] = drafted[j - 1];
        const auto t1f0 = now();
        if (auto e = forward(vin.data(), V, p, /*reset_kv=*/false, row.data(),
                             /*all_logits=*/d_all, /*hidden=*/d_hid, &ckpts);
            !e.empty()) { ret = "spec verify: " + e; break; }
        ms_vfwd += ms(t1f0, now());
        for (uint32_t j = 0; j < V; ++j)
            MtpHead::argmax_row_device(qh, d_all + uint64_t(j) * vocab, vocab,
                                       d_targ + j);
        qh.memcpy(targ.data(), d_targ, uint64_t(V) * sizeof(int32_t)).wait();
        const auto t2 = now();

        uint32_t n = 0;                     // accepted drafts (g_1..g_n)
        for (uint32_t j = 1; j < V; ++j) {
            if (drafted[j - 1] == targ[j - 1]) ++n; else break;
        }
        if (tree_probe) {
            ++probe_rounds;
            if (drafted[0] != targ[0]) {
                ++probe_miss0;
                if (mtp_.top2_last == targ[0]) ++probe_rec;
            }
        }
        const int32_t bonus = targ[n];
        const uint32_t accepted = n + 1;    // verify rows kept (tn + g_1..g_n)
        if (spec_dbg && p < P0 + 8)
            std::fprintf(stderr, "[spec-dbg split] p=%u tn=%d drafted=[%d,%d] targ=[%d,%d,%d] acc=%u\n",
                         p, vin[0], V > 1 ? drafted[0] : -1, V > 2 ? drafted[1] : -1,
                         targ[0], V > 1 ? targ[1] : -1, V > 2 ? targ[2] : -1, n);

        // 3. COMMIT per-card state to p+accepted (no re-forward).
        if (accepted < V) {
            for (uint32_t dev = 0; dev < n_dev_ && !abort; ++dev) {
                if (dn_[dev].ready()) {
                    if (auto e = ckpts[dev].commit_to_n(fleet_->dev(dev).queue(),
                                                        dn_[dev], accepted); !e.empty()) {
                        ret = "spec commit dev " + std::to_string(dev) + ": " + e;
                        abort = true;
                    }
                }
                for (uint32_t l = 0; l < uint32_t(kvsnap[dev].size()); ++l)
                    kv_[dev].set_length(l, kvsnap[dev][l] + accepted);
            }
        }
        // else: all V rows accepted → per-card state already at p+V.

        // 4. EMIT committed tokens in order (first round also emits tn).
        auto do_emit = [&](int32_t id) -> bool {
            if (emitted >= max_new) return false;
            ++emitted;
            if (!emit(id)) { abort = true; return false; }
            return true;
        };
        if (!abort && first_round) { if (!do_emit(tn)) {} }
        for (uint32_t j = 0; j < n && !abort && emitted < max_new; ++j) do_emit(drafted[j]);
        if (!abort && emitted < max_new) do_emit(bonus);

        // 5. Next round: condition on the last ACCEPTED verify row's hidden.
        tn = bonus;
        qh.memcpy(h_last, d_hid + uint64_t(accepted - 1) * H,
                  uint64_t(H) * sizeof(sycl::half)).wait();
        p += accepted;
        first_round = false;
        if (prof) {
            ms_draft  += ms(t0, t1);
            ms_verify += ms(t1, t2);
            ms_rest   += ms(t2, now());
            ++prof_rounds;
        }
    }
    if (tree_probe && probe_rounds)
        std::fprintf(stderr,
            "[spec TREE-PROBE] rounds=%u miss0=%u top2-recovers=%u (%.0f%% of misses)\n",
            probe_rounds, probe_miss0, probe_rec,
            probe_miss0 ? 100.0 * probe_rec / probe_miss0 : 0.0);
    if (prof && prof_rounds)
        std::fprintf(stderr,
            "[qwen35split SPEC] rounds=%u draft %.2f ms/rd (%.2f/fwd) "
            "verify %.2f ms/rd (fwd %.2f) rest %.2f ms/rd\n",
            prof_rounds, ms_draft / prof_rounds, ms_draft / prof_rounds / V,
            ms_verify / prof_rounds, ms_vfwd / prof_rounds,
            ms_rest / prof_rounds);
    spec_verify_gemv_ = false;
    free_scratch();
    return ret;
}

}  // namespace ie
