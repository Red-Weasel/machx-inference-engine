// src/model/gptoss_tp.cpp — gpt-oss (OpenAI MoE) TENSOR-PARALLEL forward.
// See header. PHASE 1: MoE expert-TP only — attention runs REPLICATED on every
// card (full weights, full heads, from the replicated residual x → attn_block
// bit-identical → x stays in sync with NO attention all-reduce, the Qwen35TpModel
// Phase-0 pattern). Only the MXFP4 experts are sharded (gate/up column, down row,
// 32-block-aligned), one host-bounce all-reduce/layer on the MoE-down partial.
// The single-GPU GptOssModel is NEVER edited; the per-layer math is lifted verbatim.

#include "ie/gptoss_tp.hpp"

#include "ie/dequant.hpp"        // dequant_mxfp4_soa_to_Bt
#include "ie/ops.hpp"            // rms_norm_f32w, FA kernels, swiglu_oai, gemv_mxfp4_soa_q8, ...
#include "ie/quant_blocks.hpp"   // block_mxfp4, block_q8_0, block_q8_1x, kQK_MXFP4

#include "dense_dispatch.hpp"    // dense::upload<T>, dense::upload_quant_dense_auto, dense::gemv_q_T

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace ie {
namespace {

// COPY of gptoss.cpp:route_from_logits (copy-not-hoist discipline). gpt-oss router =
// SOFTMAX_WEIGHT: logits (router bias already added) → top-k by raw logit → softmax
// over the k selected, renormalized. Output sorted ascending by expert id.
void route_from_logits(const float* logit, uint32_t E, uint32_t k,
                       std::vector<std::pair<uint32_t, float>>& out) {
    float m = logit[0];
    for (uint32_t e = 1; e < E; ++e) m = std::max(m, logit[e]);
    double sum = 0.0;
    std::vector<float> prob(E);
    for (uint32_t e = 0; e < E; ++e) { prob[e] = std::exp(logit[e] - m); sum += prob[e]; }
    for (uint32_t e = 0; e < E; ++e) prob[e] = float(prob[e] / sum);
    std::vector<uint32_t> idx(E);
    for (uint32_t e = 0; e < E; ++e) idx[e] = e;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](uint32_t a, uint32_t b) { return prob[a] > prob[b]; });
    float wsum = 0.f;
    for (uint32_t j = 0; j < k; ++j) wsum += prob[idx[j]];
    out.clear();
    for (uint32_t j = 0; j < k; ++j) out.emplace_back(idx[j], prob[idx[j]] / wsum);
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
}

// MXFP4 expert COLUMN slice (gate/up: split the output dim N=EF). The on-disk MXFP4
// expert blocks are laid out [E][N][Bpc] (Bpc=K/32 blocks along the quantized K dim).
// Column nn∈[0,efc) of this card maps to source column n=ef0+nn. The result is the
// SoA repack of GptOssModel (qs nibble plane + e exponent plane) restricted to those
// efc columns — feeds gemv_mxfp4_soa_q8 (decode) / dequant_mxfp4_soa_to_Bt (prefill)
// with K unchanged, N=efc. No block-alignment constraint (K stays whole).
// Device-or-host-USM alloc: device while the card's expert residency is under budget,
// else host USM (the gemv kernel reads it over PCIe — the cold-expert "~10% to RAM"
// spill that keeps VRAM under the cap so the display card never maxes / freezes).
inline void* spill_alloc(DeviceAllocator& a, size_t sz, uint64_t budget,
                         uint64_t& dev_used, uint64_t& host_used) {
    // Under budget → device; if device is physically full anyway, fall through to host
    // (never fail the load — the whole point of the spill is to not OOM).
    if (dev_used + sz <= budget) { if (void* p = a.malloc(sz)) { dev_used += sz; return p; } }
    void* p = sycl::malloc_host(sz, a.queue()); if (p) host_used += sz; return p;
}

std::string repack_exps_col(DeviceAllocator& alloc, const GgufTensorInfo* ti,
                            uint32_t E, uint32_t K, uint32_t N, uint32_t ef0, uint32_t efc,
                            std::vector<void*>& own,
                            uint8_t*& qs_dev, uint8_t*& e_dev,
                            uint64_t& qs_stride, uint64_t& e_stride,
                            uint64_t budget, uint64_t& dev_used, uint64_t& host_used) {
    if (!ti) return "col exps: not found";
    if (ti->dtype != DType::kMXFP4) return "col exps: must be MXFP4";
    if (uint64_t(ef0) + efc > N) return "col exps: bad slice";
    const uint32_t Bpc = K / kQK_MXFP4;                  // blocks/column along K
    const uint64_t bpe = uint64_t(N) * Bpc;              // source blocks/expert
    if (ti->nbytes != uint64_t(E) * bpe * sizeof(block_mxfp4))
        return "col exps: size mismatch";
    qs_stride = uint64_t(efc) * (uint64_t(K) / 2);       // efc * Bpc * 16
    e_stride  = uint64_t(efc) * Bpc;
    std::vector<uint8_t> qs_host(uint64_t(E) * qs_stride);
    std::vector<uint8_t> e_host(uint64_t(E) * e_stride);
    const auto* src = reinterpret_cast<const block_mxfp4*>(ti->data);
    // Parallel fill: each expert ex writes its own disjoint [ex*qs_stride] / [ex*e_stride]
    // region, so there is NO data race across the E experts. Byte layout BIT-IDENTICAL to
    // the serial path (col: out[di*16] within di=j*Bpc+b). alloc/upload stay after the join.
    {
        uint8_t* qs_base = qs_host.data();
        uint8_t* e_base  = e_host.data();
        const auto fill = [&](uint32_t e_begin, uint32_t e_end) {
            for (uint32_t ex = e_begin; ex < e_end; ++ex) {
                uint8_t* qcol = qs_base + uint64_t(ex) * qs_stride;
                uint8_t* ecol = e_base  + uint64_t(ex) * e_stride;
                for (uint32_t j = 0; j < efc; ++j) {
                    const uint64_t n = ef0 + j;
                    for (uint32_t b = 0; b < Bpc; ++b) {
                        const uint64_t si = uint64_t(ex) * bpe + n * Bpc + b;
                        const uint64_t di = uint64_t(j) * Bpc + b;
                        std::memcpy(qcol + di * 16, src[si].qs, 16);
                        ecol[di] = src[si].e;
                    }
                }
            }
        };
        unsigned hw = std::thread::hardware_concurrency();
        unsigned nthreads = std::min<unsigned>(std::max(1u, hw ? hw : 1u), 8u);
        nthreads = std::min<unsigned>(nthreads, E ? E : 1u);   // never more threads than experts
        if (nthreads <= 1) {
            fill(0, E);
        } else {
            const uint32_t per = (E + nthreads - 1) / nthreads;
            std::vector<std::thread> pool;
            pool.reserve(nthreads);
            for (unsigned t = 0; t < nthreads; ++t) {
                const uint32_t b = t * per;
                const uint32_t e = std::min<uint32_t>(b + per, E);
                if (b >= e) break;
                pool.emplace_back(fill, b, e);
            }
            for (auto& th : pool) th.join();
        }
    }
    // Spill BOTH planes together (decided by the combined size) so an expert is wholly
    // device or wholly host — no split per-expert read.
    {
        const bool to_host = dev_used + qs_host.size() + e_host.size() > budget;
        const uint64_t b = to_host ? 0 : budget;
        qs_dev = static_cast<uint8_t*>(spill_alloc(alloc, qs_host.size(), b, dev_used, host_used));
        e_dev  = static_cast<uint8_t*>(spill_alloc(alloc, e_host.size(),  b, dev_used, host_used));
    }
    if (!qs_dev || !e_dev) return "col exps: SoA alloc failed";
    own.push_back(qs_dev); own.push_back(e_dev);
    alloc.queue().memcpy(qs_dev, qs_host.data(), qs_host.size());
    alloc.queue().memcpy(e_dev,  e_host.data(),  e_host.size()).wait();
    return {};
}

// MXFP4 expert ROW slice (down: split the contraction dim K=EF). On-disk blocks are
// [E][N=H][Bpc=EF/32]. Keep the block range [ef0/32, (ef0+efc)/32) of EVERY output
// column n∈[0,H). ef0 and efc MUST be 32-aligned (MXFP4 block). Result feeds the
// down GEMV with K=efc, N=H → a PARTIAL [T,H] that the all-reduce sums.
std::string repack_exps_row(DeviceAllocator& alloc, const GgufTensorInfo* ti,
                            uint32_t E, uint32_t K, uint32_t N, uint32_t ef0, uint32_t efc,
                            std::vector<void*>& own,
                            uint8_t*& qs_dev, uint8_t*& e_dev,
                            uint64_t& qs_stride, uint64_t& e_stride,
                            uint64_t budget, uint64_t& dev_used, uint64_t& host_used) {
    if (!ti) return "row exps: not found";
    if (ti->dtype != DType::kMXFP4) return "row exps: must be MXFP4";
    if ((ef0 % kQK_MXFP4) || (efc % kQK_MXFP4)) return "row exps: K-slice not 32-aligned";
    if (uint64_t(ef0) + efc > K) return "row exps: bad slice";
    const uint32_t Bpc = K / kQK_MXFP4;                  // source blocks/column (=EF/32)
    const uint64_t bpe = uint64_t(N) * Bpc;
    if (ti->nbytes != uint64_t(E) * bpe * sizeof(block_mxfp4))
        return "row exps: size mismatch";
    const uint32_t bk0 = ef0 / kQK_MXFP4, nbk = efc / kQK_MXFP4;
    qs_stride = uint64_t(N) * (uint64_t(efc) / 2);       // H * nbk * 16
    e_stride  = uint64_t(N) * nbk;
    std::vector<uint8_t> qs_host(uint64_t(E) * qs_stride);
    std::vector<uint8_t> e_host(uint64_t(E) * e_stride);
    const auto* src = reinterpret_cast<const block_mxfp4*>(ti->data);
    // Parallel fill: each expert ex writes its own disjoint [ex*qs_stride] / [ex*e_stride]
    // region, so there is NO data race across the E experts. Byte layout BIT-IDENTICAL to
    // the serial path (row: out[di*16] within di=n*nbk+bb). alloc/upload stay after the join.
    {
        uint8_t* qs_base = qs_host.data();
        uint8_t* e_base  = e_host.data();
        const auto fill = [&](uint32_t e_begin, uint32_t e_end) {
            for (uint32_t ex = e_begin; ex < e_end; ++ex) {
                uint8_t* qcol = qs_base + uint64_t(ex) * qs_stride;
                uint8_t* ecol = e_base  + uint64_t(ex) * e_stride;
                for (uint32_t n = 0; n < N; ++n)
                    for (uint32_t bb = 0; bb < nbk; ++bb) {
                        const uint64_t si = uint64_t(ex) * bpe + uint64_t(n) * Bpc + (bk0 + bb);
                        const uint64_t di = uint64_t(n) * nbk + bb;
                        std::memcpy(qcol + di * 16, src[si].qs, 16);
                        ecol[di] = src[si].e;
                    }
            }
        };
        unsigned hw = std::thread::hardware_concurrency();
        unsigned nthreads = std::min<unsigned>(std::max(1u, hw ? hw : 1u), 8u);
        nthreads = std::min<unsigned>(nthreads, E ? E : 1u);   // never more threads than experts
        if (nthreads <= 1) {
            fill(0, E);
        } else {
            const uint32_t per = (E + nthreads - 1) / nthreads;
            std::vector<std::thread> pool;
            pool.reserve(nthreads);
            for (unsigned t = 0; t < nthreads; ++t) {
                const uint32_t b = t * per;
                const uint32_t e = std::min<uint32_t>(b + per, E);
                if (b >= e) break;
                pool.emplace_back(fill, b, e);
            }
            for (auto& th : pool) th.join();
        }
    }
    {
        const bool to_host = dev_used + qs_host.size() + e_host.size() > budget;
        const uint64_t b = to_host ? 0 : budget;
        qs_dev = static_cast<uint8_t*>(spill_alloc(alloc, qs_host.size(), b, dev_used, host_used));
        e_dev  = static_cast<uint8_t*>(spill_alloc(alloc, e_host.size(),  b, dev_used, host_used));
    }
    if (!qs_dev || !e_dev) return "row exps: SoA alloc failed";
    own.push_back(qs_dev); own.push_back(e_dev);
    alloc.queue().memcpy(qs_dev, qs_host.data(), qs_host.size());
    alloc.queue().memcpy(e_dev,  e_host.data(),  e_host.size()).wait();
    return {};
}

// gate/up bias EF-slice: F32 [E,EF] on disk → device [E,efc] (each expert's
// [ef0,ef0+efc) slice). Added to the column-parallel gate/up output (local, no
// double-count).
float* upload_bias_ef_slice(DeviceAllocator& alloc, const GgufTensorInfo* ti,
                            uint32_t E, uint32_t EF, uint32_t ef0, uint32_t efc,
                            std::vector<void*>& own, std::string& err) {
    if (!ti) { err = "ef bias: not found"; return nullptr; }
    if (ti->dtype != DType::kF32) { err = "ef bias: expected F32"; return nullptr; }
    if (ti->nbytes != uint64_t(E) * EF * sizeof(float)) { err = "ef bias: size mismatch"; return nullptr; }
    const float* src = reinterpret_cast<const float*>(ti->data);
    std::vector<float> host(uint64_t(E) * efc);
    for (uint32_t e = 0; e < E; ++e)
        std::memcpy(host.data() + uint64_t(e) * efc, src + uint64_t(e) * EF + ef0, efc * sizeof(float));
    void* d = alloc.malloc(host.size() * sizeof(float));
    if (!d) { err = "ef bias: alloc failed"; return nullptr; }
    own.push_back(d);
    alloc.queue().memcpy(d, host.data(), host.size() * sizeof(float)).wait();
    return static_cast<float*>(d);
}

// Attention F16 weight slicers (Phase 2 head-shard). The single-GPU path dequants the
// FULL weight via dequant_q8_0_to_Bt → fp16 **Bt = [K,N]** (out[k*N+n], the GEMM-
// stationary orientation gemm_fp16_onednn / gemv_q_T consume). We MUST emit the SAME
// Bt layout for the slice. On-disk Q8_0 [K,N] is stored N rows × K/32 blocks (output-
// major): value(input k, output n) = d_{n,k/32} · qs[k%32].
//   COLUMN slice (q/k/v): keep output cols [n0,n0+nc) → Bt [K,nc], out[k*nc + j].
//   ROW slice (o-proj):   keep input rows [k0,k0+kc) → Bt [kc,N], out[kl*N + n]
//     (the contraction split → PARTIAL [T,H] → all-reduce; k0/kc 32-aligned for Q8_0).
DenseQuantPtr upload_attn_col_f16(DeviceAllocator& a, const GgufTensorInfo* ti,
                                  uint32_t n0, uint32_t nc, std::vector<void*>& own,
                                  std::string& err, uint64_t& bytes) {
    DenseQuantPtr out;
    if (!ti) { err = "attn col: not found"; return out; }
    const uint32_t K = uint32_t(ti->shape[0]), N = uint32_t(ti->shape[1]);
    if (ti->n_dims != 2 || uint64_t(n0) + nc > N) { err = "attn col: bad slice"; return out; }
    std::vector<sycl::half> host(uint64_t(K) * nc);   // Bt [K, nc]
    if (ti->dtype == DType::kQ8_0) {
        const uint32_t bpc = K / 32;
        const auto* blk = reinterpret_cast<const block_q8_0*>(ti->data);
        for (uint32_t j = 0; j < nc; ++j) {
            const uint32_t n = n0 + j;
            for (uint32_t b = 0; b < bpc; ++b) {
                const float d = fp16_to_fp32(blk[uint64_t(n) * bpc + b].d);
                const int8_t* qs = blk[uint64_t(n) * bpc + b].qs;
                for (int i = 0; i < 32; ++i)
                    host[uint64_t(b * 32 + i) * nc + j] = sycl::half(d * float(qs[i]));
            }
        }
    } else if (ti->dtype == DType::kF16) {   // on-disk mem out[n*K+k] → Bt out[k*nc+j]
        const auto* src = reinterpret_cast<const sycl::half*>(ti->data);
        for (uint32_t j = 0; j < nc; ++j)
            for (uint32_t k = 0; k < K; ++k)
                host[uint64_t(k) * nc + j] = src[uint64_t(n0 + j) * K + k];
    } else { err = "attn col: dtype unsupported (Q8_0/F16)"; return out; }
    void* d = a.malloc(host.size() * sizeof(sycl::half));
    if (!d) { err = "attn col: alloc"; return out; }
    own.push_back(d);
    a.queue().memcpy(d, host.data(), host.size() * sizeof(sycl::half)).wait();
    bytes += host.size() * sizeof(sycl::half);
    out.p = d; out.dt = DType::kF16; return out;
}

DenseQuantPtr upload_attn_row_f16(DeviceAllocator& a, const GgufTensorInfo* ti,
                                  uint32_t k0, uint32_t kc, std::vector<void*>& own,
                                  std::string& err, uint64_t& bytes) {
    DenseQuantPtr out;
    if (!ti) { err = "attn row: not found"; return out; }
    const uint32_t K = uint32_t(ti->shape[0]), N = uint32_t(ti->shape[1]);
    if (ti->n_dims != 2 || uint64_t(k0) + kc > K) { err = "attn row: bad slice"; return out; }
    std::vector<sycl::half> host(uint64_t(kc) * N);   // Bt [kc, N]
    if (ti->dtype == DType::kQ8_0) {
        if ((k0 % 32) || (kc % 32)) { err = "attn row: K-slice not 32-aligned"; return out; }
        const uint32_t bpc = K / 32, bk0 = k0 / 32, nbk = kc / 32;
        const auto* blk = reinterpret_cast<const block_q8_0*>(ti->data);
        for (uint32_t n = 0; n < N; ++n)
            for (uint32_t bb = 0; bb < nbk; ++bb) {
                const float d = fp16_to_fp32(blk[uint64_t(n) * bpc + bk0 + bb].d);
                const int8_t* qs = blk[uint64_t(n) * bpc + bk0 + bb].qs;
                for (int i = 0; i < 32; ++i)
                    host[uint64_t(bb * 32 + i) * N + n] = sycl::half(d * float(qs[i]));
            }
    } else if (ti->dtype == DType::kF16) {   // on-disk mem out[n*K+k] → Bt out[kl*N+n]
        const auto* src = reinterpret_cast<const sycl::half*>(ti->data);
        for (uint32_t n = 0; n < N; ++n)
            for (uint32_t kl = 0; kl < kc; ++kl)
                host[uint64_t(kl) * N + n] = src[uint64_t(n) * K + (k0 + kl)];
    } else { err = "attn row: dtype unsupported (Q8_0/F16)"; return out; }
    void* d = a.malloc(host.size() * sizeof(sycl::half));
    if (!d) { err = "attn row: alloc"; return out; }
    own.push_back(d);
    a.queue().memcpy(d, host.data(), host.size() * sizeof(sycl::half)).wait();
    bytes += host.size() * sizeof(sycl::half);
    out.p = d; out.dt = DType::kF16; return out;
}

}  // namespace

GptOssTpModel::~GptOssTpModel() { free_all(); }

void GptOssTpModel::free_all() {
    if (!fleet_) return;
    for (uint32_t d = 0; d < ws_.size(); ++d) free_ws(d);
    ws_.clear();
    for (uint32_t d = 0; d < owned_.size(); ++d)
        for (void* p : owned_[d]) if (p) fleet_->dev(d).free(p);
    owned_.clear();
    for (auto& c : kv_) c.free_storage();
}

std::string GptOssTpModel::load(DeviceFleet& fleet, const GgufReader& g,
                                const GptOssConfig& cfg, uint32_t max_ctx) {
    fleet_ = &fleet; cfg_ = cfg;
    n_dev_ = fleet.size();
    const auto& d = cfg.dense;
    const uint32_t E = cfg.n_experts, EF = cfg.expert_ffn, H = d.hidden;

    if (n_dev_ == 0) return "gptoss-tp: empty fleet";
    if (H == 0 || d.vocab == 0 || d.n_q_heads == 0 || d.n_kv_heads == 0 || EF == 0 || E == 0)
        return "gptoss-tp: zero dim";
    // MXFP4 down row-split lands on 32-element blocks → EF must split into n_dev
    // 32-aligned chunks. (EF=2880 = 90 blocks → 2 cards = 45 blocks = 1440 each.)
    if (EF % (kQK_MXFP4 * n_dev_)) {
        // allow uneven block split as long as each chunk is 32-aligned (handled below);
        // only reject if EF itself isn't a multiple of 32.
        if (EF % kQK_MXFP4) return "gptoss-tp: expert_ffn not 32-aligned";
    }

    // Per-card expert-ffn split into 32-block-aligned contiguous ranges (gate/up
    // column split point == down row split point).
    shard_.assign(n_dev_, {});
    {
        const uint32_t nb = EF / kQK_MXFP4;     // total 32-blocks
        uint32_t acc = 0;
        for (uint32_t c = 0; c < n_dev_; ++c) {
            const uint32_t my_nb = nb / n_dev_ + (c < (nb % n_dev_) ? 1u : 0u);
            shard_[c].ef0 = acc; shard_[c].efc = my_nb * kQK_MXFP4; acc += shard_[c].efc;
        }
        if (acc != EF) return "gptoss-tp: expert_ffn block split did not cover EF";
    }

    // Phase 2: attention head-shard (both cards compute attn + halve KV/card). Needs
    // n_q_heads/n_kv_heads divisible by n_dev and the o-proj K-slice (q_head0*HD,Nq)
    // 32-aligned (Q8_0 block) — true for HD=64. opt-out → Phase-1 replicate-attn.
    // Phase 1 (replicate attn) DROPS the attn all-reduce — ~34% of TP prefill, which
    // is COMM-bound (host-bounced all-reduce, P2P HW-blocked) → ~1.5× faster prefill
    // (442→679 tok/s measured on the 120b). It DOUBLES KV/card (full heads, no shard),
    // so it only fits at moderate ctx. AUTO: replicate when the full KV fits
    // (max_ctx ≤ kReplicateMaxCtx — validated to fit + stay PPL-correct on 120b 2×B70),
    // head-shard (Phase 2, halved KV) for long ctx. Force: IE_GPTOSS_TP_REPLICATE_ATTN
    // / IE_GPTOSS_TP_HEAD_SHARD.
    constexpr uint32_t kReplicateMaxCtx = 16384;
    attn_shard_ = (max_ctx > kReplicateMaxCtx);
    if (std::getenv("IE_GPTOSS_TP_REPLICATE_ATTN")) attn_shard_ = false;
    if (std::getenv("IE_GPTOSS_TP_HEAD_SHARD"))     attn_shard_ = true;
    if (attn_shard_ && (d.n_q_heads % n_dev_ || d.n_kv_heads % n_dev_)) {
        std::fprintf(stderr, "[gptoss-tp] heads not divisible by %u — replicating attn\n", n_dev_);
        attn_shard_ = false;
    }
    for (uint32_t c = 0; c < n_dev_; ++c) {
        Shard& s = shard_[c];
        if (attn_shard_) {
            const uint32_t nqc = d.n_q_heads / n_dev_, nkvc = d.n_kv_heads / n_dev_;
            s.q_head0 = c * nqc; s.nq = nqc; s.kv_head0 = c * nkvc; s.nkv = nkvc;
        } else {
            s.q_head0 = 0; s.nq = d.n_q_heads; s.kv_head0 = 0; s.nkv = d.n_kv_heads;
        }
        s.Nq = s.nq * d.head_dim; s.Nkv = s.nkv * d.head_dim;
        if (attn_shard_ && ((uint64_t(s.q_head0) * d.head_dim) % 32 || uint64_t(s.Nq) % 32))
            return "gptoss-tp: o-proj K-slice not 32-aligned";
    }

    owned_.assign(n_dev_, {});
    dev_bytes_.assign(n_dev_, 0);

    // Per-card VRAM budget for the experts (the bulk). Beyond it, expert planes spill
    // to host USM (read over PCIe) so the card never maxes — critical on card 0 (the
    // display GPU; a maxed display GPU freezes the desktop). Reserve more on card 0.
    // Override: IE_GPTOSS_TP_RESERVE_GB (applies to every card). Default 5 (card 0) / 3.
    std::vector<uint64_t> budget(n_dev_), host_bytes(n_dev_, 0);
    {
        const char* renv = std::getenv("IE_GPTOSS_TP_RESERVE_GB");
        for (uint32_t c = 0; c < n_dev_; ++c) {
            const uint64_t gmem = fleet.dev(c).device().get_info<sycl::info::device::global_mem_size>();
            double reserve_gb = renv ? std::atof(renv) : (c == 0 ? 5.0 : 3.0);
            const uint64_t res = uint64_t(reserve_gb * 1e9);
            budget[c] = gmem > res ? gmem - res : gmem / 2;
        }
    }

    char buf[64];
    auto T  = [&](const char* n) { return g.find_tensor(n); };
    auto Tl = [&](uint32_t L, const char* n) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n); return g.find_tensor(buf);
    };
    std::string err;

    // ---- globals on card 0 (embedding + final norm + lm_head; x replicated) ----
    {
        DeviceAllocator& a = fleet.dev(0);
        const auto* ti = T("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype != DType::kQ8_0 && ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K)
            return "token_embd: unsupported dtype (want Q8_0/Q4_K/Q6_K)";
        token_embd_dtype_ = ti->dtype;
        token_embd_ = dense::upload<void>(a, ti, owned_[0], err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
        dev_bytes_[0] += ti->nbytes;

        output_norm_ = dense::upload<float>(a, T("output_norm.weight"), owned_[0], err, DType::kF32);
        if (!err.empty()) return "output_norm: " + err;

        const auto* lo = T("output.weight");
        if (!lo) return "output.weight: not found";
        if (lo->dtype != DType::kQ8_0 || lo->n_dims != 2) return "output: expected Q8_0 2-D";
        const uint32_t K = uint32_t(lo->shape[0]), N = uint32_t(lo->shape[1]);
        if (K % 32) return "output: K not %32";
        const uint32_t bpc = K / 32;
        const auto* blocks = reinterpret_cast<const block_q8_0*>(lo->data);
        std::vector<int8_t>   qs(uint64_t(N) * K);
        std::vector<uint16_t> dd(uint64_t(N) * bpc);
        for (uint64_t n = 0; n < N; ++n)
            for (uint32_t b = 0; b < bpc; ++b) {
                const block_q8_0& blk = blocks[n * bpc + b];
                dd[n * bpc + b] = *reinterpret_cast<const uint16_t*>(&blk.d);
                for (int i = 0; i < 32; ++i)
                    qs[n * uint64_t(K) + uint64_t(b) * 32 + i] = blk.qs[i];
            }
        lmhead_qs_ = static_cast<int8_t*>(a.malloc(qs.size()));
        lmhead_d_  = static_cast<uint16_t*>(a.malloc(dd.size() * sizeof(uint16_t)));
        if (!lmhead_qs_ || !lmhead_d_) return "output: SoA alloc failed";
        owned_[0].push_back(lmhead_qs_); owned_[0].push_back(lmhead_d_);
        a.queue().memcpy(lmhead_qs_, qs.data(), qs.size());
        a.queue().memcpy(lmhead_d_, dd.data(), dd.size() * sizeof(uint16_t)).wait();
        lmhead_K_ = K; lmhead_N_ = N;
        dev_bytes_[0] += qs.size() + dd.size() * sizeof(uint16_t);
    }

    const uint32_t n_layers = d.n_layers;
    dlayers_.assign(n_dev_, std::vector<LayerW>(n_layers));

    for (uint32_t c = 0; c < n_dev_; ++c) {
        DeviceAllocator& a = fleet.dev(c);
        auto& own = owned_[c];
        const Shard& s = shard_[c];
        for (uint32_t L = 0; L < n_layers; ++L) {
            LayerW& w = dlayers_[c][L];
            w.is_swa = (L % 2 == 0);

            auto F32 = [&](const char* nm, float*& dst) -> std::string {
                const auto* ti = Tl(L, nm);
                if (!ti) return std::string(nm) + ": not found";
                if (ti->dtype != DType::kF32) return std::string(nm) + ": expected F32";
                dst = dense::upload<float>(a, ti, own, err, DType::kF32);
                if (!err.empty()) return std::string(nm) + ": " + err;
                dev_bytes_[c] += ti->nbytes;
                return {};
            };
            auto lerr = [&](const std::string& m) {
                return "card " + std::to_string(c) + " L" + std::to_string(L) + " " + m;
            };

            // norms (replicated)
            if (auto m = F32("attn_norm.weight", w.attn_norm); !m.empty()) return lerr(m);
            if (auto m = F32("post_attention_norm.weight", w.post_attn_norm); !m.empty()) return lerr(m);

            // attention weights + biases. PHASE 2 (attn_shard_): q/k/v column-parallel
            // (output heads), o-proj row-parallel (input heads → PARTIAL + all-reduce),
            // biases + sinks head-sliced, o-bias RANK-0-ONLY. PHASE 1: full/replicated.
            const Shard& sa = shard_[c];
            const uint32_t HD = d.head_dim;
            if (attn_shard_) {
                w.attn_q = upload_attn_col_f16(a, Tl(L, "attn_q.weight"), sa.q_head0 * HD, sa.Nq, own, err, dev_bytes_[c]);
                if (!err.empty()) return lerr("attn_q: " + err);
                w.attn_k = upload_attn_col_f16(a, Tl(L, "attn_k.weight"), sa.kv_head0 * HD, sa.Nkv, own, err, dev_bytes_[c]);
                if (!err.empty()) return lerr("attn_k: " + err);
                w.attn_v = upload_attn_col_f16(a, Tl(L, "attn_v.weight"), sa.kv_head0 * HD, sa.Nkv, own, err, dev_bytes_[c]);
                if (!err.empty()) return lerr("attn_v: " + err);
                w.attn_output = upload_attn_row_f16(a, Tl(L, "attn_output.weight"), sa.q_head0 * HD, sa.Nq, own, err, dev_bytes_[c]);
                if (!err.empty()) return lerr("attn_output: " + err);
                // head-sliced F32 biases + sinks (q/k/v biases column-parallel, local).
                auto F32S = [&](const char* nm, uint32_t off, uint32_t len, float*& dst) -> std::string {
                    const auto* ti = Tl(L, nm);
                    if (!ti) return std::string(nm) + ": not found";
                    if (ti->dtype != DType::kF32) return std::string(nm) + ": expected F32";
                    if (uint64_t(off) + len > ti->nbytes / sizeof(float)) return std::string(nm) + ": slice oob";
                    std::vector<float> hv(len);
                    std::memcpy(hv.data(), reinterpret_cast<const float*>(ti->data) + off, uint64_t(len) * sizeof(float));
                    void* p = a.malloc(uint64_t(len) * sizeof(float));
                    if (!p) return std::string(nm) + ": alloc";
                    own.push_back(p); dev_bytes_[c] += uint64_t(len) * sizeof(float);
                    a.queue().memcpy(p, hv.data(), uint64_t(len) * sizeof(float)).wait();
                    dst = static_cast<float*>(p); return {};
                };
                if (auto m = F32S("attn_q.bias", sa.q_head0 * HD, sa.Nq, w.attn_q_bias); !m.empty()) return lerr(m);
                if (auto m = F32S("attn_k.bias", sa.kv_head0 * HD, sa.Nkv, w.attn_k_bias); !m.empty()) return lerr(m);
                if (auto m = F32S("attn_v.bias", sa.kv_head0 * HD, sa.Nkv, w.attn_v_bias); !m.empty()) return lerr(m);
                if (auto m = F32S("attn_sinks.weight", sa.q_head0, sa.nq, w.attn_sinks); !m.empty()) return lerr(m);
                if (c == 0) { if (auto m = F32("attn_output.bias", w.attn_o_bias); !m.empty()) return lerr(m); }
            } else {
                w.attn_q = dense::upload_quant_dense_auto(a, Tl(L, "attn_q.weight"), own, err);
                if (!err.empty()) return lerr("attn_q: " + err);
                w.attn_k = dense::upload_quant_dense_auto(a, Tl(L, "attn_k.weight"), own, err);
                if (!err.empty()) return lerr("attn_k: " + err);
                w.attn_v = dense::upload_quant_dense_auto(a, Tl(L, "attn_v.weight"), own, err);
                if (!err.empty()) return lerr("attn_v: " + err);
                w.attn_output = dense::upload_quant_dense_auto(a, Tl(L, "attn_output.weight"), own, err);
                if (!err.empty()) return lerr("attn_output: " + err);
                if (auto m = F32("attn_q.bias", w.attn_q_bias); !m.empty()) return lerr(m);
                if (auto m = F32("attn_k.bias", w.attn_k_bias); !m.empty()) return lerr(m);
                if (auto m = F32("attn_v.bias", w.attn_v_bias); !m.empty()) return lerr(m);
                if (auto m = F32("attn_output.bias", w.attn_o_bias); !m.empty()) return lerr(m);
                if (auto m = F32("attn_sinks.weight", w.attn_sinks); !m.empty()) return lerr(m);
            }

            // router — card 0 only (routing is deterministic + replicated).
            if (c == 0) {
                const auto* ti = Tl(L, "ffn_gate_inp.weight");
                if (!ti) return lerr("ffn_gate_inp: not found");
                if (ti->dtype != DType::kF32) return lerr("ffn_gate_inp: expected F32");
                if (ti->nbytes != uint64_t(E) * H * sizeof(float)) return lerr("ffn_gate_inp: size");
                std::vector<sycl::half> rt(uint64_t(H) * E);
                const float* rw = reinterpret_cast<const float*>(ti->data);
                for (uint32_t e = 0; e < E; ++e)
                    for (uint32_t h = 0; h < H; ++h)
                        rt[uint64_t(h) * E + e] = sycl::half(rw[uint64_t(e) * H + h]);
                auto* rd = static_cast<sycl::half*>(a.malloc(uint64_t(H) * E * sizeof(sycl::half)));
                if (!rd) return lerr("router_w_dev alloc");
                a.queue().memcpy(rd, rt.data(), uint64_t(H) * E * sizeof(sycl::half)).wait();
                own.push_back(rd); w.router_w_dev = rd;
                if (auto m = F32("ffn_gate_inp.bias", w.router_bias); !m.empty()) return lerr(m);
            }

            // experts (MXFP4) — EF-sliced. gate/up column (K=H, N=EF→efc); down row
            // (K=EF→efc, N=H).
            if (auto m = repack_exps_col(a, Tl(L, "ffn_gate_exps.weight"), E, H, EF, s.ef0, s.efc,
                                         own, w.gate_qs, w.gate_e, w.gate_qs_stride, w.gate_e_stride,
                                         budget[c], dev_bytes_[c], host_bytes[c]); !m.empty()) return lerr("gate " + m);
            if (auto m = repack_exps_col(a, Tl(L, "ffn_up_exps.weight"), E, H, EF, s.ef0, s.efc,
                                         own, w.up_qs, w.up_e, w.up_qs_stride, w.up_e_stride,
                                         budget[c], dev_bytes_[c], host_bytes[c]); !m.empty()) return lerr("up " + m);
            if (auto m = repack_exps_row(a, Tl(L, "ffn_down_exps.weight"), E, EF, H, s.ef0, s.efc,
                                         own, w.down_qs, w.down_e, w.down_qs_stride, w.down_e_stride,
                                         budget[c], dev_bytes_[c], host_bytes[c]); !m.empty()) return lerr("down " + m);
            // gate/up bias EF-sliced; down bias RANK 0 ONLY (row-parallel → would double).
            w.gate_bias = upload_bias_ef_slice(a, Tl(L, "ffn_gate_exps.bias"), E, EF, s.ef0, s.efc, own, err);
            if (!err.empty()) return lerr("gate_bias: " + err);
            w.up_bias = upload_bias_ef_slice(a, Tl(L, "ffn_up_exps.bias"), E, EF, s.ef0, s.efc, own, err);
            if (!err.empty()) return lerr("up_bias: " + err);
            if (c == 0) {
                const auto* ti = Tl(L, "ffn_down_exps.bias");
                if (!ti) return lerr("ffn_down_exps.bias: not found");
                if (ti->dtype != DType::kF32) return lerr("ffn_down_exps.bias: expected F32");
                w.down_bias = dense::upload<float>(a, ti, own, err, DType::kF32);
                if (!err.empty()) return lerr("down_bias: " + err);
                dev_bytes_[c] += ti->nbytes;
            }
        }
    }

    // Per-card KV: head-sharded (nkv kv-heads/card) when attn_shard_ → halves KV/card;
    // else full (replicated attn). All layers (gpt-oss caches full KV for SWA too).
    kv_.resize(n_dev_);
    for (uint32_t c = 0; c < n_dev_; ++c) {
        KvCacheConfig kc{};
        kc.n_layers_full = n_layers;
        kc.n_kv_heads    = shard_[c].nkv;   // == d.n_kv_heads when not head-sharded
        kc.max_ctx       = max_ctx;
        kc.head_dim      = d.head_dim;
        if (auto m = kv_[c].init(fleet.dev(c), kc); !m.empty())
            return "kv dev " + std::to_string(c) + ": " + m;
    }

    ws_.assign(n_dev_, {});
    for (uint32_t c = 0; c < n_dev_; ++c)
        std::fprintf(stderr, "[gptoss-tp] card %u: %.2f GB VRAM + %.2f GB host-spill "
                     "(budget %.1f GB, ef0=%u efc=%u, attn %s nq=%u nkv=%u)\n",
                     c, double(dev_bytes_[c]) / 1e9, double(host_bytes[c]) / 1e9,
                     double(budget[c]) / 1e9, shard_[c].ef0, shard_[c].efc,
                     attn_shard_ ? "sharded" : "replicated", shard_[c].nq, shard_[c].nkv);
    return {};
}

void GptOssTpModel::free_ws(uint32_t dev) {
    if (!fleet_ || dev >= ws_.size()) return;
    Workspace& w = ws_[dev];
    auto& a = fleet_->dev(dev);
    for (void* p : {static_cast<void*>(w.x), static_cast<void*>(w.xn), static_cast<void*>(w.block),
                    static_cast<void*>(w.q), static_cast<void*>(w.attn_out), static_cast<void*>(w.k),
                    static_cast<void*>(w.v), static_cast<void*>(w.positions),
                    static_cast<void*>(w.moe_logits), static_cast<void*>(w.moe_xp),
                    static_cast<void*>(w.moe_h), static_cast<void*>(w.moe_h2),
                    static_cast<void*>(w.moe_out), static_cast<void*>(w.moe_y),
                    static_cast<void*>(w.moe_wpk), static_cast<void*>(w.moe_btf16),
                    static_cast<void*>(w.moe_sorted), static_cast<void*>(w.moe_tk2pk),
                    w.q8_x, w.q8_h, static_cast<void*>(w.attn_partials)})
        if (p) a.free(p);
    w = Workspace{};
}

std::string GptOssTpModel::ensure_ws(uint32_t dev, uint32_t max_T) {
    Workspace& w = ws_[dev];
    if (max_T <= w.T) return {};
    if (w.T != 0) free_ws(dev);
    auto& a = fleet_->dev(dev);
    auto ah = [&](uint64_t n) { return static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half))); };

    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, HD = d.head_dim;
    const uint32_t N_q = shard_[dev].Nq, N_kv = shard_[dev].Nkv;   // sharded heads (full when replicate)
    const uint32_t nq = shard_[dev].nq;
    const uint32_t E = cfg_.n_experts, K = cfg_.n_experts_used;
    const uint32_t efc = shard_[dev].efc;
    const uint64_t T = max_T, TK = uint64_t(max_T) * K;

    w.x        = ah(T * H);
    w.xn       = ah(T * H);
    w.block    = ah(T * H);
    w.q        = ah(T * N_q);
    w.attn_out = ah(T * N_q);
    w.k        = ah(T * N_kv);
    w.v        = ah(T * N_kv);
    w.positions = static_cast<int32_t*>(a.malloc(T * sizeof(int32_t)));
    w.moe_logits = ah(T * E);
    // +32-row slack on the gather/down inputs: the small-M pad-GEMM reads up to 32
    // rows from an expert's slice; rows past TK are read-safe and their output discarded.
    w.moe_xp   = ah(uint64_t(TK + 32) * H);
    w.moe_h    = ah(uint64_t(TK + 32) * efc);
    w.moe_h2   = ah(uint64_t(TK) * efc);
    w.moe_out  = ah(uint64_t(TK) * H);
    w.moe_y    = ah(T * H);
    w.moe_wpk  = ah(TK);
    w.moe_btf16 = ah(uint64_t(H) * efc);
    w.moe_pout = ah(uint64_t(32) * H);   // small-M pad-GEMM output scratch [32, max(efc,H)=H]
    w.moe_sorted = static_cast<int32_t*>(a.malloc(TK * sizeof(int32_t)));
    w.moe_tk2pk  = static_cast<int32_t*>(a.malloc(TK * sizeof(int32_t)));
    w.q8_x = a.malloc(uint64_t(H   / 32) * sizeof(block_q8_1x));
    w.q8_h = a.malloc(uint64_t(efc / 32) * sizeof(block_q8_1x));

    if (!w.x || !w.xn || !w.block || !w.q || !w.attn_out || !w.k || !w.v || !w.positions ||
        !w.moe_logits || !w.moe_xp || !w.moe_h || !w.moe_h2 || !w.moe_out || !w.moe_y ||
        !w.moe_wpk || !w.moe_btf16 || !w.moe_pout || !w.moe_sorted || !w.moe_tk2pk || !w.q8_x || !w.q8_h)
        return "gptoss-tp workspace alloc failed on dev " + std::to_string(dev);

    // FA-2 decode partials [n_chunks_max, nq(this card's heads), head_dim+2] (Bc_floor=64).
    const uint32_t max_ctx = kv_[dev].config().max_ctx;
    constexpr uint32_t Bc_floor = 64;
    const uint32_t n_chunks = (max_ctx + Bc_floor - 1) / Bc_floor;
    const uint64_t np = uint64_t(n_chunks) * nq * (d.head_dim + 2);
    w.attn_partials = static_cast<float*>(a.malloc(np * sizeof(float)));
    if (!w.attn_partials) return "gptoss-tp attn_partials alloc failed on dev " + std::to_string(dev);
    w.partials_ctx = max_ctx;
    w.T = max_T;
    return {};
}

// Per-card MoE (EF-sliced). Routing (pk_) already computed on card 0 this layer.
void GptOssTpModel::moe_ffn_card(uint32_t c, uint32_t L, uint32_t T) {
    Workspace& ws = ws_[c];
    const LayerW& w = dlayers_[c][L];
    auto& q = fleet_->dev(c).queue();
    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, E = cfg_.n_experts, K = cfg_.n_experts_used;
    const uint32_t efc = shard_[c].efc;
    const uint32_t TK = T * K;

    static const bool mxfp4_gemv = std::getenv("IE_GPTOSS_NO_MXFP4_GEMV") == nullptr;
    static const bool mxfp4_f16  = std::getenv("IE_GPTOSS_MXFP4_F16") != nullptr;
    static const bool x2_fuse    = std::getenv("IE_GPTOSS_NO_MXFP4_X2") == nullptr;  // fuse gate+up gemv+bias (decode)
    const bool decode_gemv = mxfp4_gemv && T == 1;
    // T>1 prefill MoE: default = oneDNN XMX GEMM with small-M experts padded to
    // kMoeMinM (out of the BMG small-M corruption regime, weights read once). Opt-out
    // to the proven per-row int-dot gemv (correct but launch-bound) via this env.
    static const bool gemv_prefill = std::getenv("IE_GPTOSS_TP_GEMV_PREFILL") != nullptr;
    constexpr uint32_t kMoeMinM = 32;

    // upload this layer's packing (identical on all cards) + gather replicated xn.
    q.memcpy(ws.moe_sorted, pk_.sorted_idx.data(),   TK * sizeof(int32_t));
    q.memcpy(ws.moe_tk2pk,  pk_.tk_to_packed.data(), TK * sizeof(int32_t));
    std::vector<sycl::half> wpk(TK);
    for (uint32_t i = 0; i < TK; ++i) wpk[i] = sycl::half(pk_.weights_packed[i]);
    q.memcpy(ws.moe_wpk, wpk.data(), TK * sizeof(sycl::half)).wait();   // wpk is local → wait
    moe_gather_rows(q, ws.xn, ws.moe_sorted, ws.moe_xp, TK, H);

    const auto& off = pk_.expert_offsets;
    if (decode_gemv && !mxfp4_f16) quantize_q8_1(q, ws.xn, ws.q8_x, H);

    // PREFILL (T>1): per-PACKED-ROW W4A8 int-dot gemv per expert (the validated
    // decode kernel), NOT a per-expert oneDNN f16 gemm. ROOT CAUSE of the 120b
    // batched-prefill corruption (PPL 42 vs 12.91): the small-M (n_e≈1–8 at
    // E=128/top-4) f16 XMX matmul on BMG corrupted local rows≥1 while row 0 stayed
    // correct; the STABLE counting sort makes token 0 always row 0 of every expert,
    // so token 0 was bit-exact at every layer and every other token garbled. Decode
    // (gemv) and the 20b (E=32 → large-M) never exercised the small-M gemm. Per-row
    // gemv is bit-faithful to the decode path; ws.q8_x/ws.q8_h reused per row
    // (in-order queue serializes quantize→gemv→next quantize). Speed-optimal pad-M
    // gemm is a later follow-up (correctness first).
    // gate + up (column-parallel: N=efc)
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t o = off[e], n_e = off[e + 1] - o;
        if (n_e == 0) continue;
        const sycl::half* x_e = ws.moe_xp + uint64_t(o) * H;
        sycl::half* g_o = ws.moe_h  + uint64_t(o) * efc;
        sycl::half* u_o = ws.moe_h2 + uint64_t(o) * efc;
        const uint8_t* gqs = w.gate_qs + uint64_t(e) * w.gate_qs_stride;
        const uint8_t* ge  = w.gate_e  + uint64_t(e) * w.gate_e_stride;
        const uint8_t* uqs = w.up_qs   + uint64_t(e) * w.up_qs_stride;
        const uint8_t* ue  = w.up_e    + uint64_t(e) * w.up_e_stride;
        bool fused_bias = false;
        if (decode_gemv) {
            if (mxfp4_f16) {
                gemv_mxfp4_soa_f16(q, x_e, gqs, ge, g_o, H, efc, {});
                gemv_mxfp4_soa_f16(q, x_e, uqs, ue, u_o, H, efc, {});
            } else if (x2_fuse) {
                // FUSED gate+up gemv + bias in ONE launch (decode is host-launch-bound):
                // both share ws.q8_x (K=H), N=efc; biases folded post-reduce. Bit-identical.
                gemv_mxfp4_soa_q8_x2(q, ws.q8_x, gqs, ge, uqs, ue,
                                     w.gate_bias + uint64_t(e) * efc,
                                     w.up_bias   + uint64_t(e) * efc,
                                     g_o, u_o, H, efc, {});
                fused_bias = true;
            } else {
                gemv_mxfp4_soa_q8(q, ws.q8_x, gqs, ge, g_o, H, efc, {});
                gemv_mxfp4_soa_q8(q, ws.q8_x, uqs, ue, u_o, H, efc, {});
            }
        } else if (gemv_prefill) {
            // opt-out: proven per-row int-dot gemv (bit-identical, but launch-bound).
            for (uint32_t r = 0; r < n_e; ++r) {
                quantize_q8_1(q, x_e + uint64_t(r) * H, ws.q8_x, H);
                gemv_mxfp4_soa_q8(q, ws.q8_x, gqs, ge, g_o + uint64_t(r) * efc, H, efc, {});
                gemv_mxfp4_soa_q8(q, ws.q8_x, uqs, ue, u_o + uint64_t(r) * efc, H, efc, {});
            }
        } else if (n_e >= kMoeMinM) {
            // large-M: direct XMX GEMM (safe regime, weights read once). moe_btf16
            // reuse gate→up is serialized by the in-order queue (gemma precedent).
            auto dg = dequant_mxfp4_soa_to_Bt(q, gqs, ge, ws.moe_btf16, H, efc, {});
            gemm_fp16_onednn(q, x_e, ws.moe_btf16, g_o, n_e, efc, H, {dg});
            auto du = dequant_mxfp4_soa_to_Bt(q, uqs, ue, ws.moe_btf16, H, efc, {});
            gemm_fp16_onednn(q, x_e, ws.moe_btf16, u_o, n_e, efc, H, {du});
        } else {
            // small-M (the BMG XMX corruption regime): run the GEMM at PADDED M=kMoeMinM
            // (reads slack rows of moe_xp — read-safe, output discarded) and copy the
            // first n_e rows back. Keeps the matmul out of the buggy small-M regime AND
            // on XMX (weights read once) instead of the launch-bound per-row gemv.
            auto dg = dequant_mxfp4_soa_to_Bt(q, gqs, ge, ws.moe_btf16, H, efc, {});
            gemm_fp16_onednn(q, x_e, ws.moe_btf16, ws.moe_pout, kMoeMinM, efc, H, {dg});
            q.memcpy(g_o, ws.moe_pout, uint64_t(n_e) * efc * sizeof(sycl::half));
            auto du = dequant_mxfp4_soa_to_Bt(q, uqs, ue, ws.moe_btf16, H, efc, {});
            gemm_fp16_onednn(q, x_e, ws.moe_btf16, ws.moe_pout, kMoeMinM, efc, H, {du});
            q.memcpy(u_o, ws.moe_pout, uint64_t(n_e) * efc * sizeof(sycl::half));
        }
        if (!fused_bias) {
            add_bias(q, g_o, w.gate_bias + uint64_t(e) * efc, n_e, efc);
            add_bias(q, u_o, w.up_bias   + uint64_t(e) * efc, n_e, efc);
        }
    }
    // clamped gated-SwiGLU (OAI) over the efc slice (element-wise → correct per-slice).
    swiglu_oai(q, ws.moe_h, ws.moe_h2, ws.moe_h, uint64_t(TK) * efc, 1.702f, 7.0f);

    // down (row-parallel: K=efc) → PARTIAL [.,H]; down_bias on rank 0 only.
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t o = off[e], n_e = off[e + 1] - o;
        if (n_e == 0) continue;
        const sycl::half* h_e = ws.moe_h + uint64_t(o) * efc;
        sycl::half* d_o = ws.moe_out + uint64_t(o) * H;
        const uint8_t* dqs = w.down_qs + uint64_t(e) * w.down_qs_stride;
        const uint8_t* de  = w.down_e  + uint64_t(e) * w.down_e_stride;
        if (decode_gemv) {
            if (mxfp4_f16) {
                gemv_mxfp4_soa_f16(q, h_e, dqs, de, d_o, efc, H, {});
            } else {
                quantize_q8_1(q, h_e, ws.q8_h, efc);
                gemv_mxfp4_soa_q8(q, ws.q8_h, dqs, de, d_o, efc, H, {});
            }
        } else if (gemv_prefill) {
            for (uint32_t r = 0; r < n_e; ++r) {
                quantize_q8_1(q, h_e + uint64_t(r) * efc, ws.q8_h, efc);
                gemv_mxfp4_soa_q8(q, ws.q8_h, dqs, de, d_o + uint64_t(r) * H, efc, H, {});
            }
        } else if (n_e >= kMoeMinM) {
            auto dd = dequant_mxfp4_soa_to_Bt(q, dqs, de, ws.moe_btf16, efc, H, {});
            gemm_fp16_onednn(q, h_e, ws.moe_btf16, d_o, n_e, H, efc, {dd});
        } else {
            auto dd = dequant_mxfp4_soa_to_Bt(q, dqs, de, ws.moe_btf16, efc, H, {});
            gemm_fp16_onednn(q, h_e, ws.moe_btf16, ws.moe_pout, kMoeMinM, H, efc, {dd});
            q.memcpy(d_o, ws.moe_pout, uint64_t(n_e) * H * sizeof(sycl::half));
        }
        if (w.down_bias) add_bias(q, d_o, w.down_bias + uint64_t(e) * H, n_e, H);
    }
    // weighted reduce → PARTIAL ws.moe_y [T,H] (all-reduce sums across cards).
    q.memset(ws.moe_y, 0, uint64_t(T) * H * sizeof(sycl::half));
    moe_prefill_reduce(q, ws.moe_out, reinterpret_cast<const uint32_t*>(ws.moe_tk2pk),
                       ws.moe_wpk, ws.moe_y, T, K, H);
}

std::string GptOssTpModel::forward(const int32_t* input_ids, uint32_t T,
                                   uint32_t start_pos, bool reset_kv,
                                   sycl::half* out_logits_host) {
    if (T == 0) return "T == 0";
    // Batched T>1 prefill is now CORRECT and the DEFAULT (the per-expert MoE prefill
    // small-M XMX bug that corrupted it is FIXED — see moe_ffn_card; 120b batched PPL
    // 15.1985 == T=1 15.1985, NLL bit-identical 2.721199). Opt-out IE_GPTOSS_TP_T1_PREFILL
    // falls back to the slow-but-correct T=1-sequential path (one full forward per token).
    static const bool t1_prefill = std::getenv("IE_GPTOSS_TP_T1_PREFILL") != nullptr;
    if (T > 1 && t1_prefill) {
        for (uint32_t t = 0; t < T; ++t)
            if (auto m = forward(input_ids + t, 1, start_pos + t, reset_kv && t == 0,
                                 out_logits_host); !m.empty()) return m;
        return {};
    }
    const auto& d = cfg_.dense;
    const uint32_t H = d.hidden, HD = d.head_dim;
    const uint32_t E = cfg_.n_experts, V = d.vocab;
    const float eps = d.rms_eps;
    const uint32_t n_layers = d.n_layers;

    static const bool proj_gemm = std::getenv("IE_GPTOSS_NO_PROJ_GEMM") == nullptr;
    static const bool fa2_decode_on  = std::getenv("IE_GPTOSS_NO_FA2_DECODE") == nullptr;
    static const bool fa2_prefill_on = std::getenv("IE_GPTOSS_NO_FA2_PREFILL") == nullptr;
    static const bool swa_on = std::getenv("IE_GPTOSS_NO_SWA_WINDOW") == nullptr;
    static const bool use_yarn = std::getenv("IE_GPTOSS_YARN") != nullptr;
    static const bool router_gemm = std::getenv("IE_GPTOSS_NO_PROJ_GEMM") == nullptr;
    // IE_GPTOSS_TP_TIMING: gated per-section wall-time attribution (Step-0). When
    // unset, none of the timing blocks below run → forward() byte-identical.
    static const bool tp_timing = std::getenv("IE_GPTOSS_TP_TIMING") != nullptr;
    using _tpclk = std::chrono::steady_clock;
    auto _tpms = [](_tpclk::time_point _t0) {
        return std::chrono::duration<double, std::milli>(_tpclk::now() - _t0).count(); };
    // IE_GPTOSS_TP_NANCHECK: localize a NaN/overflow in the residual stream (ws[0].x).
    // IE_GPTOSS_TP_XDUMP: token-0 per-layer checksum — token 0 is identical in the
    //   batched (T>1) and T=1-sequential paths, so any layer where the two diverge
    //   is the bug. Only dumps for the first chunk (start_pos==0).
    static const bool nancheck = std::getenv("IE_GPTOSS_TP_NANCHECK") != nullptr;
    static const bool xdump    = std::getenv("IE_GPTOSS_TP_XDUMP") != nullptr;
    auto chk = [&](const char* where, int L) {
        if (!nancheck && !xdump) return;
        auto& q0 = fleet_->dev(0).queue();
        std::vector<sycl::half> h(uint64_t(T) * H);
        q0.memcpy(h.data(), ws_[0].x, h.size() * sizeof(sycl::half)).wait();
        float maxabs = 0.f; bool bad = false;
        for (auto v : h) { float f = float(v);
            if (std::isnan(f) || std::isinf(f)) bad = true;
            maxabs = std::max(maxabs, std::fabs(f)); }
        if (xdump && start_pos == 0) {
            double xs = 0.0; for (uint32_t i = 0; i < H; ++i) xs += std::fabs(double(float(h[i])));
            std::fprintf(stderr, "[xdump] %-5s L=%-2d tok0_xsum=%.5f maxabs=%g\n", where, L, xs, maxabs);
        }
        if (nancheck && (bad || maxabs > 1e4f))
            std::fprintf(stderr, "[nancheck] pos=%u tok=%d %s L=%d maxabs=%g bad=%d\n",
                         start_pos, int(input_ids[0]), where, L, maxabs, int(bad));
    };

    if (ws_.size() != n_dev_) ws_.assign(n_dev_, {});
    for (uint32_t c = 0; c < n_dev_; ++c)
        if (auto m = ensure_ws(c, T); !m.empty()) return m;

    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
    for (uint32_t c = 0; c < n_dev_; ++c) {
        fleet_->dev(c).queue().memcpy(ws_[c].positions, pos.data(), T * sizeof(int32_t)).wait();
        if (reset_kv) kv_[c].reset();
    }

    // embedding on card 0 → ws[0].x, broadcast replicated x to all cards.
    _tpclk::time_point _tmisc; if (tp_timing) _tmisc = _tpclk::now();
    {
        auto& a = fleet_->dev(0);
        auto& q = a.queue();
        int32_t* d_ids = static_cast<int32_t*>(a.malloc(T * sizeof(int32_t)));
        if (!d_ids) return "d_ids alloc failed";
        q.memcpy(d_ids, input_ids, T * sizeof(int32_t)).wait();
        embedding_lookup_q8_0(q, d_ids, token_embd_, ws_[0].x, T, H);
        q.wait();
        a.free(d_ids);
    }
    for (uint32_t c = 1; c < n_dev_; ++c)
        fleet_->copy_across(0, ws_[c].x, c, ws_[0].x, uint64_t(T) * H * sizeof(sycl::half));
    if (tp_timing) { for (uint32_t c = 0; c < n_dev_; ++c) fleet_->dev(c).queue().wait();
                     tt_misc_ += _tpms(_tmisc); }
    chk("EMBED", -1);

    std::vector<sycl::half*> moe_y_bufs(n_dev_), attn_block_bufs(n_dev_);
    for (uint32_t c = 0; c < n_dev_; ++c) { moe_y_bufs[c] = ws_[c].moe_y; attn_block_bufs[c] = ws_[c].block; }

    for (uint32_t L = 0; L < n_layers; ++L) {
        // ===== ATTENTION =====
        // Phase 2 (attn_shard_): q/k/v column-parallel by head, o-proj row-parallel →
        // PARTIAL [T,H] → all-reduce #1. Phase 1: full/replicated (block is full, the
        // all-reduce is skipped → x stays in lock-step exactly as before).
        for (uint32_t c = 0; c < n_dev_; ++c) {
            Workspace& ws = ws_[c];
            const LayerW& w = dlayers_[c][L];
            const Shard& s = shard_[c];
            auto& q = fleet_->dev(c).queue();
            const uint32_t sNq = s.Nq, sNkv = s.Nkv, snq = s.nq, snkv = s.nkv;
            auto proj = [&](const sycl::half* A, const DenseQuantPtr& W, sycl::half* y,
                            uint32_t Kp, uint32_t Np) {
                if (proj_gemm && T > 1 && W.dt == DType::kF16 && (Np % 64 == 0))
                    gemm_fp16_onednn(q, A, static_cast<const sycl::half*>(W.p), y, T, Np, Kp, {});
                else
                    dense::gemv_q_T(q, A, W, y, Kp, Np, T);
            };
            rms_norm_f32w(q, ws.x, w.attn_norm, ws.xn, T, H, eps);
            proj(ws.xn, w.attn_q, ws.q, H, sNq);
            proj(ws.xn, w.attn_k, ws.k, H, sNkv);
            proj(ws.xn, w.attn_v, ws.v, H, sNkv);
            add_bias(q, ws.q, w.attn_q_bias, T, sNq);
            add_bias(q, ws.k, w.attn_k_bias, T, sNkv);
            add_bias(q, ws.v, w.attn_v_bias, T, sNkv);
            if (use_yarn) {
                rope_yarn(q, ws.q, ws.positions, ws.q, T, snq, HD, d.rope_dim,
                          d.rope_theta, cfg_.rope_freq_scale, cfg_.rope_orig_ctx,
                          cfg_.rope_ext_factor, 1.0f, 32.0f, 1.0f);
                rope_yarn(q, ws.k, ws.positions, ws.k, T, snkv, HD, d.rope_dim,
                          d.rope_theta, cfg_.rope_freq_scale, cfg_.rope_orig_ctx,
                          cfg_.rope_ext_factor, 1.0f, 32.0f, 1.0f);
            } else {
                rope_partial(q, ws.q, ws.positions, ws.q, T, snq, HD, d.rope_dim, d.rope_theta);
                rope_partial(q, ws.k, ws.positions, ws.k, T, snkv, HD, d.rope_dim, d.rope_theta);
            }
            const uint64_t per_layer = uint64_t(snkv) * kv_[c].config().max_ctx * HD;
            sycl::half* kc = kv_[c].k_ptr() + per_layer * L;
            sycl::half* vc = kv_[c].v_ptr() + per_layer * L;
            const uint32_t mc = kv_[c].config().max_ctx;
            const uint32_t window = (swa_on && w.is_swa) ? cfg_.sliding_window : 0u;
            if (T == 1 && fa2_decode_on && window == 0 && ws.attn_partials) {
                full_attention_fa2_decode_gptoss(q, ws.q, ws.k, ws.v, kc, vc, ws.attn_out,
                                                 ws.attn_partials, start_pos, snq,
                                                 snkv, HD, mc, w.attn_sinks, {});
            } else if (T > 1 && fa2_prefill_on) {
                full_attention_gptoss_prefill_tile(q, ws.q, ws.k, ws.v, kc, vc, ws.attn_out,
                                                   T, start_pos, snq, snkv,
                                                   mc, window, w.attn_sinks, {});
            } else {
                full_attention_gptoss(q, ws.q, ws.k, ws.v, kc, vc, ws.attn_out, T, start_pos,
                                      snq, snkv, HD, mc, window, w.attn_sinks);
            }
            kv_[c].set_length(L, start_pos + T);
            proj(ws.attn_out, w.attn_output, ws.block, sNq, H);   // PARTIAL [T,H] when sharded
            if (w.attn_o_bias) add_bias(q, ws.block, w.attn_o_bias, T, H);  // rank-0-only when sharded
        }
        if (tp_timing) { auto _t = _tpclk::now();
            for (uint32_t c = 0; c < n_dev_; ++c) fleet_->dev(c).queue().wait();
            tt_attn_ += _tpms(_t); }
        if (attn_shard_) {
            _tpclk::time_point _t; if (tp_timing) _t = _tpclk::now();
            fleet_->all_reduce_sum_fp16(attn_block_bufs, uint64_t(T) * H);
            if (tp_timing) tt_ar_attn_ += _tpms(_t);
        }
        for (uint32_t c = 0; c < n_dev_; ++c) {
            auto& q = fleet_->dev(c).queue();
            residual_add(q, ws_[c].x, ws_[c].block, ws_[c].x, uint64_t(T) * H);
            rms_norm_f32w(q, ws_[c].x, dlayers_[c][L].post_attn_norm, ws_[c].xn, T, H, eps);
        }
        chk("attn", int(L));

        // ===== MoE: route once on card 0, sharded experts, ONE all-reduce =====
        {
            _tpclk::time_point _tr; if (tp_timing) _tr = _tpclk::now();
            auto& q0 = fleet_->dev(0).queue();
            const LayerW& w0 = dlayers_[0][L];
            if (router_gemm && T > 1)
                gemm_fp16_onednn(q0, ws_[0].xn, w0.router_w_dev, ws_[0].moe_logits, T, E, H, {});
            else
                dense::gemv_q_T(q0, ws_[0].xn, DenseQuantPtr{w0.router_w_dev, DType::kF16},
                                ws_[0].moe_logits, H, E, T);
            add_bias(q0, ws_[0].moe_logits, w0.router_bias, T, E);
            std::vector<sycl::half> hl(uint64_t(T) * E);
            q0.memcpy(hl.data(), ws_[0].moe_logits, uint64_t(T) * E * sizeof(sycl::half)).wait();
            host_logits_.resize(uint64_t(T) * E);
            for (uint64_t i = 0; i < uint64_t(T) * E; ++i) host_logits_[i] = float(hl[i]);
            host_routes_.resize(T);
            for (uint32_t t = 0; t < T; ++t)
                route_from_logits(host_logits_.data() + uint64_t(t) * E, E,
                                  cfg_.n_experts_used, host_routes_[t]);
            build_moe_packing(host_routes_, E, cfg_.n_experts_used, pk_);
            if (tp_timing) tt_router_ += _tpms(_tr);
        }
        for (uint32_t c = 0; c < n_dev_; ++c) moe_ffn_card(c, L, T);
        if (tp_timing) { auto _t = _tpclk::now();
            for (uint32_t c = 0; c < n_dev_; ++c) fleet_->dev(c).queue().wait();
            tt_moe_ += _tpms(_t); }
        {
            _tpclk::time_point _t; if (tp_timing) _t = _tpclk::now();
            fleet_->all_reduce_sum_fp16(moe_y_bufs, uint64_t(T) * H);
            if (tp_timing) tt_ar_moe_ += _tpms(_t);
        }
        for (uint32_t c = 0; c < n_dev_; ++c)
            residual_add(fleet_->dev(c).queue(), ws_[c].x, ws_[c].moe_y, ws_[c].x, uint64_t(T) * H);
        chk("moe", int(L));
    }
    for (uint32_t c = 0; c < n_dev_; ++c) fleet_->dev(c).queue().wait();

    // final norm + lm_head on card 0 (x replicated).
    _tpclk::time_point _tlm; if (tp_timing) _tlm = _tpclk::now();
    {
        Workspace& ws = ws_[0];
        auto& a = fleet_->dev(0);
        auto& q = a.queue();
        rms_norm_f32w(q, ws.x, output_norm_, ws.xn, T, H, eps);
        const sycl::half* last = ws.xn + uint64_t(T - 1) * H;
        sycl::half* d_logits = static_cast<sycl::half*>(a.malloc(uint64_t(V) * sizeof(sycl::half)));
        if (!d_logits) return "logits alloc failed";
        quantize_q8_1(q, last, ws.q8_x, H);
        gemv_q8_0_soa_q8(q, ws.q8_x, lmhead_qs_, lmhead_d_, d_logits, H, V).wait();
        q.memcpy(out_logits_host, d_logits, uint64_t(V) * sizeof(sycl::half)).wait();
        a.free(d_logits);
    }
    if (tp_timing) tt_lmhead_ += _tpms(_tlm);
    return {};
}

void GptOssTpModel::reset_tp_timing() noexcept {
    tt_attn_ = tt_ar_attn_ = tt_router_ = tt_moe_ = tt_ar_moe_ = tt_lmhead_ = tt_misc_ = 0;
}

void GptOssTpModel::report_tp_timing(uint32_t steps) const {
    if (steps == 0) steps = 1;
    const double tot = tt_attn_ + tt_ar_attn_ + tt_router_ + tt_moe_
                     + tt_ar_moe_ + tt_lmhead_ + tt_misc_;
    auto row = [&](const char* n, double ms) {
        std::printf("    %-22s %10.3f ms  %9.4f ms/step  %6.1f%%\n",
                    n, ms, ms / steps, tot > 0 ? 100.0 * ms / tot : 0.0); };
    std::printf("\n  --- IE_GPTOSS_TP_TIMING decode breakdown (%u steps; timing mode "
                "adds per-layer barriers) ---\n", steps);
    row("attention compute", tt_attn_);
    row("attn all-reduce",   tt_ar_attn_);
    row("router round-trip", tt_router_);
    row("MoE gemv (+spill)", tt_moe_);
    row("MoE all-reduce",    tt_ar_moe_);
    row("lm_head",           tt_lmhead_);
    row("embed/bcast/misc",  tt_misc_);
    std::printf("    %-22s %10.3f ms  %9.4f ms/step  100.0%%\n",
                "TOTAL (instrumented)", tot, tot / steps);
    std::printf("    (true tok/s = the NO-timing run; comm = attn + MoE all-reduce "
                "+ router round-trip)\n");
}

}  // namespace ie
