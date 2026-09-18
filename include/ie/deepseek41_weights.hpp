// include/ie/deepseek41_weights.hpp — V4.1 dense weights, resident on a device across forwards.
//
// Phase 7 uploaded every layer's dense set on every forward and freed it (27.8 GiB moved for a
// 12-token prefill). Phase 8's first step is to upload the 8.87 GiB dense text path ONCE per
// card and keep it: this cache owns those device pointers. The dequant is Phase 4's
// ds41_dense_dequant_f16 (FP8 x 32x32 E8M0 -> fp16, bit-exact against the host) and the
// BF16/F32 conversions are the same ones the Phase 7 runtime used per forward, moved here so
// there is one copy of that logic and it runs once.
#pragma once

#include "ie/deepseek41.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

// One layer's resident dense set. fp16 for anything a GEMM consumes, fp32 for norms, sinks,
// hyper-connection tensors and the router. Null where the layer's kind has no such tensor.
// A dense matrix kept in the checkpoint's own FP8 (E4M3 bytes [N, K], one E8M0 scale per 32 x 32
// block [N/32, K/32]) -- Phase 17 (docs/deepseek41/43): read by gemv_fp8_e4m3_f16 at T = 1, dequantised
// into the cache's fp16 scratch for the T > 1 GEMMs.
struct Ds41Fp8Mat { uint8_t* w = nullptr; uint8_t* s = nullptr; uint32_t N = 0, K = 0; };

struct Ds41LayerDense {
    sycl::half *wq_a = nullptr, *wq_b = nullptr, *wkv = nullptr, *wo_a = nullptr, *wo_b = nullptr;
    // IE_DS41_DENSE_FP8=1: these matrices stay FP8 (the fp16 pointers above stay null for them). wo_a joined
    // them in Phase 32 (docs/deepseek41/72): the block-diagonal FP8 GEMV takes a column group now, so the
    // 2.68 GB/token of fp16 wo_a reads become 1.34 GB -- the same weight values, only the summation order
    // differs (the checkpoint stores wo_a as FP8 + 32x32 scales; the fp16 copy was a load-time dequant).
    Ds41Fp8Mat f8_wq_a, f8_wq_b, f8_wkv, f8_wo_a, f8_wo_b, f8_sh_w1, f8_sh_w3, f8_sh_w2, f8_idx_wq_b, f8_engram_wkv;
    sycl::half *sh_w1 = nullptr, *sh_w3 = nullptr, *sh_w2 = nullptr;
    sycl::half *comp_wkv = nullptr, *comp_wgate = nullptr;               // kv-source layers
    sycl::half *idx_wq_b = nullptr, *idx_weights = nullptr, *idx_wk = nullptr;   // index-source layers
    sycl::half *engram_wkv = nullptr;                                    // engram layers
    float *n_q = nullptr, *n_kv = nullptr, *n_at = nullptr, *n_ff = nullptr, *sinks = nullptr;
    float *n_c = nullptr, *n_ik = nullptr, *engram_qk = nullptr;         // q_weight * k_weight, [hc, dim]
    float *a_fn = nullptr, *a_bs = nullptr, *a_sc = nullptr, *f_fn = nullptr, *f_bs = nullptr, *f_sc = nullptr;
    float *g_w = nullptr, *g_b = nullptr;                                // router weight [E, dim] fp32, bias [E]
    float *g_b_vl = nullptr;                                             // the bias image-span tokens route by [E]
    uint64_t bytes = 0;                                                  // device bytes this layer holds
    std::vector<void*> owned;                                            // everything to free
};

class Ds41DenseCache {
public:
    // Upload layer L's dense set to `q`'s device if not already there. Idempotent. "" on success.
    std::string upload_layer(sycl::queue& q, const DeepSeek41Model& m, uint32_t L);
    // Upload embed (kept host-side: it is a gather), final norm, and the head to `q`'s device.
    // fp8_head (Phase 54, docs/deepseek41/93, founder-approved LOSSY): the BF16 head quantised at load to E4M3 with one E8M0
    // exponent per 32x32 block -- the checkpoint's own dense FP8 layout, read by the FP8 GEMVs -- instead of an fp16 copy:
    // half the bytes per decode token and 0.66 GB of VRAM returned to the expert tier. head() is then null; head_fp8().
    std::string upload_head(sycl::queue& q, const DeepSeek41Model& m, bool fp8_head = false);

    const Ds41LayerDense* layer(uint32_t L) const { return L < layers_.size() && layers_[L].bytes ? &layers_[L] : nullptr; }
    const float*      final_norm() const { return final_norm_; }
    const sycl::half* head() const { return head_; }
    const Ds41Fp8Mat& head_fp8() const { return head8_; }
    bool fp8() const { return fp8_; }                       // the FP8-resident mode (env IE_DS41_DENSE_FP8=1)
    // T > 1: dequantise an FP8 matrix into this cache's fp16 scratch (in order on q; the scratch is
    // reused per call, so the GEMM that reads it is enqueued before the next call) and return it.
    const sycl::half* f16(sycl::queue& q, const Ds41Fp8Mat& m);
    uint64_t bytes() const;                       // total resident device bytes
    void free_all(sycl::queue& q);                // every pointer this cache handed out is dead after this

private:
    std::vector<Ds41LayerDense> layers_;
    float*      final_norm_ = nullptr;
    sycl::half* head_       = nullptr;
    Ds41Fp8Mat  head8_;
    uint64_t    head_bytes_ = 0;
    bool        fp8_ = false;
    sycl::half* scratch_ = nullptr; size_t scratch_halves_ = 0; sycl::queue* scratch_q_ = nullptr;
};

}  // namespace ie
