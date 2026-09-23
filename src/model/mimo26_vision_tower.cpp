// src/model/mimo26_vision_tower.cpp — MiMo-V2.6's mimovl vision tower on the GPU (docs/mimo26/00_PORT_PLAN.md, P6.1b).
//
// Reference: tools/mimo26/vision_ref.py (the checkpoint's modeling_mimo_v2.py:585-911, P6.0 gate: bit-identical to it).
// Per block: RMSNorm -> qkv (+b) -> 2-D rope -> attention (32 q / 8 kv heads x 64, full or a +-64 band over the
// CURRENT order, a per-head bias on key 0's logit in the banded blocks) -> proj (+b) -> RMSNorm -> SwiGLU (+b each).
// Runs of kind-1 blocks work in column order: whole 2x2 merge blocks permuted in, and back after the run.
// Merger: LayerNorm (weight only) -> [N/4, 5120] -> Linear -> GELU (erf) -> Linear -> [N/4, 4096], no biases.
//
// GEMMs ride the XMX gemm_fp16 (A[M,K] f16 x Bt[K,N] f16 -> C[M,N] f32); every C buffer tolerates round_up(M, 8)
// rows (gemm_fp16.cpp CONTRACT). The f16 operands stay far inside f16's range (norm outputs are bounded by
// max|w| * sqrt(1280) = 94; q/k/v/att measured <= 64 on the fixtures) EXCEPT the SwiGLU product of block 27, where
// channel 572's massive activation is born: intermediate 4226 reaches 8.5e4 (fixture 0) / 1.4e5 (fixture 1) on a few
// rows, past f16's 65,504. mimov_swiglu_f16 scales such rows by a power of two before the f16 store and the down
// GEMM's f32 output is multiplied back (exact). Kernel-launching helpers carry the `mimov_` prefix: SYCL names device kernels after
// the enclosing function, and a name shared with another TU runs the wrong code (docs: sycl-kernel-name trap).
#include "ie/mimo26_vision.hpp"

#include "ie/allocator.hpp"
#include "ie/ops.hpp"
#include "ie/safetensors.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ie {

namespace {

constexpr uint32_t kD = 1280, kHeads = 32, kKvHeads = 8, kHd = 64, kQ = kHeads * kHd, kKv = kKvHeads * kHd;
constexpr uint32_t kQkv = kQ + 2 * kKv, kFfn = 4608, kOut = 4096, kMergeIn = 4 * kD, kDepth = 28, kRot = 32;
constexpr uint32_t kWindow = 64;           // visual_token_window_size
constexpr float kEps = 1e-6f, kTheta = 1e4f;
// vit_window_attn_types: -1 full attention (== fullatt_block_indexes 0, 9, 18, 27), 0 band in row order,
// 1 band in column order. load_from() checks the sinks tensors agree (they exist exactly on the banded blocks).
constexpr int kKind[kDepth] = {-1, 0, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 1, 1, 1, 1, -1};

inline uint32_t round8(uint32_t n) { return (n + 7u) & ~7u; }

}  // namespace

// One device buffer of the encode block: a weight (src set; in_d != 0 -> f16 Bt [in_d, out_d], else f32 [out_d])
// or scratch (src null).
struct MimovSlot { void** pp; size_t bytes; const SafeTensorInfo* src; uint32_t out_d, in_d; size_t off; size_t dev_off; int block; bool streamed; };
// off: the slot's offset in the pinned host copy (weights only); dev_off: its offset in the encode block, or, for a streamed
// block matrix, its offset inside the two ring slots; block: the tower block a streamed matrix belongs to (-1 otherwise).

struct MimoVision::Impl {
    ~Impl();
    std::function<const SafeTensorInfo*(const std::string&)> find;
    MimoVisionOptions opt;
    bool open = false;
    DeviceAllocator* alloc = nullptr;
    void* hw = nullptr;                 // pinned: the converted weights, copied into the block per encode
    size_t hw_bytes = 0, block_bytes = 0;
    size_t ring_bytes = 0, ring_dev_off[2] = {0, 0};   // streamed block weights: two device slots, one block each
    uint64_t hw_hash = 0;
    std::vector<MimovSlot> slots;
    void collect_slots();
    void fill_host();
    struct BlockW {
        float *n1 = nullptr, *n2 = nullptr, *qkv_b = nullptr, *proj_b = nullptr;
        float *gate_b = nullptr, *up_b = nullptr, *down_b = nullptr, *sinks = nullptr;
        sycl::half *qkv = nullptr, *proj = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
    };
    sycl::half* patch = nullptr;        // Bt [1536, 1280]
    BlockW blk[kDepth];
    float* ln_q = nullptr;              // [1280]
    sycl::half *m0 = nullptr, *m2 = nullptr;   // Bt [5120, 5120], [5120, 4096]
    // scratch: Mp = max_patches (multiple of 8), Nc = round_up(Mp, 128), R = mlp_rows, Tq = attn_tile
    sycl::half* px = nullptr;           // [Mp, 1536]
    float *res = nullptr, *res2 = nullptr, *Cd = nullptr;   // [Mp, 1280]
    sycl::half* xh = nullptr;           // [Mp, 1280]
    float* Cq = nullptr;                // [Mp, 3072]
    sycl::half* qkv = nullptr;          // [Mp, 3072]
    sycl::half* att = nullptr;          // [Mp, 2048]
    float *Cg = nullptr, *Cu = nullptr; // [R, 4608]
    sycl::half* ffn = nullptr;          // [R, 4608]
    float* rs = nullptr;                // [R] per-row 2^e the f16 ffn was divided by; the down GEMM's output is multiplied back
    float* tab = nullptr;               // cos/sin, row then column order: 4 x [Mp, 32]
    uint32_t* perm = nullptr;           // column order, then its inverse: 2 x [Mp / 4]
    sycl::half *qh = nullptr, *kT = nullptr, *vh = nullptr, *P = nullptr;   // [Mp,64] [64,Nc] [Nc,64] [Tq,Nc]
    float *S = nullptr, *O = nullptr;   // [Tq, Nc], [Tq, 64]
    float* Cm = nullptr;                // [Mp / 4, 5120]
    sycl::half* mh = nullptr;           // [Mp / 4, 5120]
    float* Co = nullptr;                // [Mp / 4, 4096]
    // pinned host staging
    sycl::half* h_px = nullptr;
    float *h_tab = nullptr, *h_out = nullptr, *h_cap = nullptr;
    uint32_t* h_perm = nullptr;
};

namespace {

// RMSNorm rows of f32 src -> f16 dst: y = w * x * rsqrt(mean(x^2) + eps)
sycl::event mimov_rms_norm_f16(sycl::queue& q, const float* src, const float* w, sycl::half* dst, uint32_t M, uint32_t N) {
    constexpr uint32_t WG = 256;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(WG, h);
        h.parallel_for(sycl::nd_range<1>(size_t(M) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t row = uint32_t(it.get_group(0)), l = uint32_t(it.get_local_id(0));
            const float* x = src + size_t(row) * N;
            float s = 0;
            for (uint32_t j = l; j < N; j += WG) s += x[j] * x[j];
            red[l] = s;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] += red[l + st];
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float inv = sycl::rsqrt(red[0] / float(N) + kEps);
            sycl::half* y = dst + size_t(row) * N;
            for (uint32_t j = l; j < N; j += WG) y[j] = sycl::half(w[j] * (x[j] * inv));
        });
    });
}

// LayerNorm rows (weight only, no bias) of f32 src -> f16 dst
sycl::event mimov_layer_norm_f16(sycl::queue& q, const float* src, const float* w, sycl::half* dst, uint32_t M, uint32_t N) {
    constexpr uint32_t WG = 256;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(WG, h);
        h.parallel_for(sycl::nd_range<1>(size_t(M) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t row = uint32_t(it.get_group(0)), l = uint32_t(it.get_local_id(0));
            const float* x = src + size_t(row) * N;
            float s = 0;
            for (uint32_t j = l; j < N; j += WG) s += x[j];
            red[l] = s;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] += red[l + st];
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float mean = red[0] / float(N);
            it.barrier(sycl::access::fence_space::local_space);
            float v = 0;
            for (uint32_t j = l; j < N; j += WG) v += (x[j] - mean) * (x[j] - mean);
            red[l] = v;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] += red[l + st];
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float inv = sycl::rsqrt(red[0] / float(N) + kEps);
            sycl::half* y = dst + size_t(row) * N;
            for (uint32_t j = l; j < N; j += WG) y[j] = sycl::half(w[j] * ((x[j] - mean) * inv));
        });
    });
}

// y[i, j] = f16(C[i, j] + b[j])
sycl::event mimov_bias_f16(sycl::queue& q, const float* C, const float* b, sycl::half* y, uint32_t M, uint32_t N) {
    return q.parallel_for(sycl::range<1>(size_t(M) * N), [=](sycl::id<1> i) { y[i] = sycl::half(C[i] + b[uint32_t(i % N)]); });
}

// res[i, j] += C[i, j] * rs[i] + b[j]; rs may be null (no per-row scale)
sycl::event mimov_bias_add_res(sycl::queue& q, const float* C, const float* b, const float* rs, float* res, uint32_t M,
                               uint32_t N) {
    return q.parallel_for(sycl::range<1>(size_t(M) * N), [=](sycl::id<1> i) {
        const float c = rs ? C[i] * rs[uint32_t(i / N)] : C[i];
        res[i] += c + b[uint32_t(i % N)];
    });
}

// ffn = f16(silu(g + gb) * (u + ub) * 2^-e), rs[row] = 2^e: one work-group per row; e = 0 unless the row's largest
// product is >= 32,768, then it lands in [2^14, 2^15) (block 27, intermediate 4226, reaches 1.4e5 on fixture 1; f16
// ends at 65,504). A power of two is exact, so the scaled rows round exactly as unscaled ones would.
sycl::event mimov_swiglu_f16(sycl::queue& q, const float* g, const float* gb, const float* u, const float* ub,
                             sycl::half* y, float* rs, uint32_t M) {
    constexpr uint32_t WG = 256, PER = kFfn / WG;
    static_assert(kFfn % WG == 0);
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(WG, h);
        h.parallel_for(sycl::nd_range<1>(size_t(M) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t row = uint32_t(it.get_group(0)), l = uint32_t(it.get_local_id(0));
            const size_t base = size_t(row) * kFfn;
            float p[PER], mx = 0.f;
            for (uint32_t k = 0; k < PER; ++k) {
                const uint32_t j = l + k * WG;
                const float a = g[base + j] + gb[j], v = u[base + j] + ub[j];
                p[k] = a / (1.0f + sycl::exp(-a)) * v;
                mx = sycl::fmax(mx, sycl::fabs(p[k]));
            }
            red[l] = mx;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] = sycl::fmax(red[l], red[l + st]);
                it.barrier(sycl::access::fence_space::local_space);
            }
            mx = red[0];
            const int e = mx >= 32768.f ? sycl::ilogb(mx) - 14 : 0;
            const float inv = sycl::ldexp(1.0f, -e);
            for (uint32_t k = 0; k < PER; ++k) y[base + l + k * WG] = sycl::half(p[k] * inv);
            if (l == 0) rs[row] = sycl::ldexp(1.0f, e);
        });
    });
}

// y = f16(gelu_erf(C))
sycl::event mimov_gelu_f16(sycl::queue& q, const float* C, sycl::half* y, size_t n) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const float v = C[i];
        y[i] = sycl::half(0.5f * v * (1.0f + sycl::erf(v * 0.70710678118654752f)));
    });
}

// 2-D rope over the 32 q heads and 8 k heads inside qkv [N, 3072]: pair (r, r + 32) turns by angle table[t][r]
// (r < 16 the h frequencies, r >= 16 the w frequencies): the reference's cat(freqs, freqs) + rotate_half, fp32 math.
sycl::event mimov_rope2d(sycl::queue& q, sycl::half* qkv, const float* cs, const float* sn, uint32_t N) {
    constexpr uint32_t slots = kHeads + kKvHeads;
    return q.parallel_for(sycl::range<1>(size_t(N) * slots * kRot), [=](sycl::id<1> id) {
        uint32_t i = uint32_t(id);
        const uint32_t r = i % kRot;   i /= kRot;
        const uint32_t s = i % slots;  i /= slots;
        const uint32_t t = i;
        const uint32_t off = s < kHeads ? s * kHd : kQ + (s - kHeads) * kHd;
        sycl::half* v = qkv + size_t(t) * kQkv + off;
        const float c = cs[size_t(t) * kRot + r], sv = sn[size_t(t) * kRot + r];
        const float a = float(v[r]), b = float(v[r + kRot]);
        v[r] = sycl::half(a * c - b * sv);
        v[r + kRot] = sycl::half(b * c + a * sv);
    });
}

// K^T [64, Nc] and V [Nc, 64] of kv head g (zero past N): shared by q heads 4g .. 4g+3 (repeat_interleave(4))
sycl::event mimov_gather_kv(sycl::queue& q, const sycl::half* qkv, sycl::half* kT, sycl::half* vh, uint32_t N,
                            uint32_t Nc, uint32_t g) {
    return q.parallel_for(sycl::range<1>(size_t(Nc) * kHd), [=](sycl::id<1> id) {
        const uint32_t i = uint32_t(id), t = i / kHd, d = i % kHd;
        const sycl::half zero(0.f);
        if (t < N) {
            const sycl::half* row = qkv + size_t(t) * kQkv;
            kT[size_t(d) * Nc + t] = row[kQ + g * kHd + d];
            vh[i] = row[kQ + kKv + g * kHd + d];
        } else {
            kT[size_t(d) * Nc + t] = zero;
            vh[i] = zero;
        }
    });
}

// Q [N, 64] of q head h
sycl::event mimov_gather_q(sycl::queue& q, const sycl::half* qkv, sycl::half* qh, uint32_t N, uint32_t h) {
    return q.parallel_for(sycl::range<1>(size_t(N) * kHd), [=](sycl::id<1> id) {
        const uint32_t i = uint32_t(id), t = i / kHd, d = i % kHd;
        qh[i] = qkv[size_t(t) * kQkv + h * kHd + d];
    });
}

// P[row, k] = pscale * softmax_k(scale * S[row, k] + bias) over the keys row0 + row may see; 0 elsewhere.
// window 0 = every key k < N; else |t - k| <= window in the current order. sink (may be null): its head's value is
// added to key 0's logit, as the reference adds it to the attention mask (a masked key 0 stays masked).
sycl::event mimov_softmax_band(sycl::queue& q, const float* S, sycl::half* P, uint32_t rows, uint32_t row0, uint32_t N,
                               uint32_t Nc, float scale, float pscale, uint32_t window, const float* sink, uint32_t head) {
    constexpr uint32_t WG = 256;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(WG, h);
        h.parallel_for(sycl::nd_range<1>(size_t(rows) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t row = uint32_t(it.get_group(0)), l = uint32_t(it.get_local_id(0));
            const uint32_t t = row0 + row;
            const uint32_t lo = window ? (t > window ? t - window : 0u) : 0u;
            const uint32_t hi = window ? sycl::min(N, t + window + 1u) : N;       // keys [lo, hi)
            const float bias0 = sink ? sink[head] : 0.f;
            const float* s = S + size_t(row) * Nc;
            sycl::half* p = P + size_t(row) * Nc;
            auto logit = [&](uint32_t k) { return s[k] * scale + (k == 0 ? bias0 : 0.f); };
            float mx = -1e30f;
            for (uint32_t k = lo + l; k < hi; k += WG) mx = sycl::fmax(mx, logit(k));
            red[l] = mx;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] = sycl::fmax(red[l], red[l + st]);
                it.barrier(sycl::access::fence_space::local_space);
            }
            mx = red[0];
            it.barrier(sycl::access::fence_space::local_space);
            float sm = 0.f;
            for (uint32_t k = lo + l; k < hi; k += WG) sm += sycl::exp(logit(k) - mx);
            red[l] = sm;
            it.barrier(sycl::access::fence_space::local_space);
            for (uint32_t st = WG / 2; st; st >>= 1) {
                if (l < st) red[l] += red[l + st];
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float inv = pscale / red[0];
            for (uint32_t k = l; k < Nc; k += WG)
                p[k] = (k >= lo && k < hi) ? sycl::half(sycl::exp(logit(k) - mx) * inv) : sycl::half(0.f);
        });
    });
}

// att[row0 + i, h*64 + d] = f16(O[i, d] * mul)
sycl::event mimov_attn_scatter(sycl::queue& q, const float* O, sycl::half* att, uint32_t rows, uint32_t h, float mul) {
    return q.parallel_for(sycl::range<1>(size_t(rows) * kHd), [=](sycl::id<1> id) {
        const uint32_t i = uint32_t(id), t = i / kHd, d = i % kHd;
        att[size_t(t) * kQ + h * kHd + d] = sycl::half(O[i] * mul);
    });
}

// dst[b*4 + j] = src[perm[b]*4 + j] for whole rows of width D (the reference's apply_index over 2x2 merge blocks)
sycl::event mimov_permute_blocks(sycl::queue& q, const float* src, float* dst, const uint32_t* perm, uint32_t nb, uint32_t D) {
    return q.parallel_for(sycl::range<1>(size_t(nb) * 4 * D), [=](sycl::id<1> id) {
        const size_t i = id;
        const uint32_t c = uint32_t(i % D);
        const uint32_t r = uint32_t(i / D), b = r / 4, j = r % 4;
        dst[i] = src[(size_t(perm[b]) * 4 + j) * D + c];
    });
}

}  // namespace

namespace {

inline float mimov_bf16_at(const uint8_t* p, size_t i) {
    uint16_t h;
    std::memcpy(&h, p + i * 2, 2);
    const uint32_t u = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// 4 MiB sampled across the pinned weights: catches anything that overwrote them (docs/deepseek41/101)
uint64_t mimov_sample_hash(const void* p, size_t bytes) {
    const auto* b = static_cast<const uint8_t*>(p);
    const size_t chunk = 4096, want = 4u << 20, chunks = bytes / chunk, take = std::min(chunks, want / chunk);
    if (!chunks) return 0;
    const size_t stride = std::max<size_t>(1, chunks / std::max<size_t>(1, take));
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t c = 0; c < chunks; c += stride)
        for (size_t i = 0; i < chunk; i += 64) { h ^= b[c * chunk + i]; h *= 0x100000001B3ull; }
    return h;
}

std::string mimov_check(const std::function<const SafeTensorInfo*(const std::string&)>& find, const std::string& name,
                        std::vector<int64_t> shape) {
    const SafeTensorInfo* t = find(name);
    if (!t) return name + ": not found";
    if (t->dtype_str != "BF16") return name + ": expected BF16, found " + t->dtype_str;
    if (t->shape != shape) return name + ": unexpected shape";
    return {};
}

}  // namespace

MimoVision::Impl::~Impl() {
    if (!alloc) return;
    sycl::queue& q = alloc->queue();
    for (void* p : {hw, static_cast<void*>(h_px), static_cast<void*>(h_tab), static_cast<void*>(h_out),
                    static_cast<void*>(h_cap), static_cast<void*>(h_perm)})
        if (p) sycl::free(p, q);
}

MimoVision::MimoVision() : impl_(std::make_unique<Impl>()) {}
MimoVision::~MimoVision() = default;
bool MimoVision::loaded() const { return impl_->open; }

std::string MimoVision::load_from(std::function<const SafeTensorInfo*(const std::string&)> find, const MimoVisionOptions& opt) {
    Impl& I = *impl_;
    if (opt.max_patches < 8 || opt.max_patches % 8 || !opt.attn_tile || opt.attn_tile % 8 || !opt.mlp_rows || opt.mlp_rows % 8)
        return "mimo26 vision: options must be positive multiples of 8";
    const int64_t D = kD, F = kFfn;
    std::vector<std::pair<std::string, std::vector<int64_t>>> want = {
        {"visual.patch_embed.proj.weight", {D, 3, 2, 16, 16}}, {"visual.merger.ln_q.weight", {D}},
        {"visual.merger.mlp.0.weight", {kMergeIn, kMergeIn}}, {"visual.merger.mlp.2.weight", {kOut, kMergeIn}}};
    for (uint32_t L = 0; L < kDepth; ++L) {
        const std::string p = "visual.blocks." + std::to_string(L) + ".";
        for (const char* n : {"norm1.weight", "norm2.weight", "attn.proj.bias", "mlp.down_proj.bias"}) want.push_back({p + n, {D}});
        want.push_back({p + "attn.qkv.weight", {kQkv, D}});        want.push_back({p + "attn.qkv.bias", {kQkv}});
        want.push_back({p + "attn.proj.weight", {D, kQ}});
        want.push_back({p + "mlp.gate_proj.weight", {F, D}});      want.push_back({p + "mlp.gate_proj.bias", {F}});
        want.push_back({p + "mlp.up_proj.weight", {F, D}});        want.push_back({p + "mlp.up_proj.bias", {F}});
        want.push_back({p + "mlp.down_proj.weight", {D, F}});
        const bool banded = kKind[L] != -1, has_sinks = find(p + "attn.sinks") != nullptr;
        if (banded != has_sinks) return p + "attn.sinks: present on a full-attention block or missing on a banded one";
        if (banded) want.push_back({p + "attn.sinks", {kHeads}});
    }
    for (const auto& [n, s] : want)
        if (std::string e = mimov_check(find, n, s); !e.empty()) return "mimo26 vision: " + e;
    for (const char* n : {"visual.merger.ln_q.bias", "visual.merger.mlp.0.bias", "visual.merger.mlp.2.bias"})
        if (find(n)) return std::string("mimo26 vision: ") + n + " is present; this tower reads the merger biases as zero";
    I.find = std::move(find);
    I.opt = opt;
    I.open = true;
    I.collect_slots();
    return {};
}

// Every device buffer in one fixed order (weights first) with its offset in the encode block.
void MimoVision::Impl::collect_slots() {
    slots.clear();
    int cur_block = -1;   // the block whose matrices W2 is collecting (streamed when the option is on)
    auto W2 = [&](sycl::half*& ptr, const std::string& n, uint32_t out_d, uint32_t in_d) {
        const bool streamed = opt.stream_weights && cur_block >= 0;
        slots.push_back({reinterpret_cast<void**>(&ptr), size_t(out_d) * in_d * 2, find(n), out_d, in_d, 0, 0, streamed ? cur_block : -1, streamed});
    };
    auto W1 = [&](float*& ptr, const std::string& n, uint32_t len) {
        slots.push_back({reinterpret_cast<void**>(&ptr), size_t(len) * 4, find(n), len, 0, 0, 0, -1, false});
    };
    W2(patch, "visual.patch_embed.proj.weight", kD, kMimoVisPatchDim);
    for (uint32_t L = 0; L < kDepth; ++L) {
        const std::string p = "visual.blocks." + std::to_string(L) + ".";
        BlockW& b = blk[L];
        cur_block = int(L);
        W1(b.n1, p + "norm1.weight", kD);                  W1(b.n2, p + "norm2.weight", kD);
        W2(b.qkv, p + "attn.qkv.weight", kQkv, kD);        W1(b.qkv_b, p + "attn.qkv.bias", kQkv);
        W2(b.proj, p + "attn.proj.weight", kD, kQ);        W1(b.proj_b, p + "attn.proj.bias", kD);
        W2(b.gate, p + "mlp.gate_proj.weight", kFfn, kD);  W1(b.gate_b, p + "mlp.gate_proj.bias", kFfn);
        W2(b.up, p + "mlp.up_proj.weight", kFfn, kD);      W1(b.up_b, p + "mlp.up_proj.bias", kFfn);
        W2(b.down, p + "mlp.down_proj.weight", kD, kFfn);  W1(b.down_b, p + "mlp.down_proj.bias", kD);
        if (kKind[L] != -1) W1(b.sinks, p + "attn.sinks", kHeads);
    }
    cur_block = -1;
    W1(ln_q, "visual.merger.ln_q.weight", kD);
    W2(m0, "visual.merger.mlp.0.weight", kMergeIn, kMergeIn);
    W2(m2, "visual.merger.mlp.2.weight", kOut, kMergeIn);
    const size_t n_weights = slots.size();
    const size_t Mp = opt.max_patches, Nc = (Mp + 127u) & ~size_t(127), R = opt.mlp_rows, Tq = opt.attn_tile;
    const size_t Bp = round8(uint32_t(Mp / 4));
    auto Sc = [&](auto*& ptr, size_t bytes) { slots.push_back({reinterpret_cast<void**>(&ptr), bytes, nullptr, 0, 0, 0, 0, -1, false}); };
    Sc(px, Mp * kMimoVisPatchDim * 2);
    Sc(res, Mp * kD * 4);  Sc(res2, Mp * kD * 4);  Sc(Cd, Mp * kD * 4);  Sc(xh, Mp * kD * 2);
    Sc(Cq, Mp * kQkv * 4);  Sc(qkv, Mp * kQkv * 2);  Sc(att, Mp * kQ * 2);
    Sc(Cg, R * kFfn * 4);  Sc(Cu, R * kFfn * 4);  Sc(ffn, R * kFfn * 2);  Sc(rs, R * 4);
    Sc(tab, 4 * Mp * kRot * 4);  Sc(perm, 2 * (Mp / 4) * 4);
    Sc(qh, Mp * kHd * 2);  Sc(kT, kHd * Nc * 2);  Sc(vh, Nc * kHd * 2);  Sc(P, Tq * Nc * 2);
    Sc(S, Tq * Nc * 4);  Sc(O, Tq * kHd * 4);
    Sc(Cm, Bp * kMergeIn * 4);  Sc(mh, Bp * kMergeIn * 2);  Sc(Co, Bp * kOut * 4);
    auto pad = [](size_t b) { return (b + 4095) & ~size_t(4095); };
    size_t off = 0;                                        // the pinned host copy: every weight, in slot order
    for (size_t i = 0; i < n_weights; ++i) { slots[i].off = off; off += pad(slots[i].bytes); }
    hw_bytes = off;
    size_t dev = 0;                                        // the encode block: resident weights, the two ring slots, scratch
    for (size_t i = 0; i < n_weights; ++i) if (!slots[i].streamed) { slots[i].dev_off = dev; dev += pad(slots[i].bytes); }
    // the streamed matrices are pushed in the same order for every block (qkv, proj, gate, up, down): the k-th one of a
    // block sits at the k-th offset inside a ring slot
    std::vector<size_t> ring_off_k;
    ring_bytes = 0;
    for (size_t i = 0; i < n_weights; ++i) if (slots[i].streamed && slots[i].block == 0) { ring_off_k.push_back(ring_bytes); ring_bytes += pad(slots[i].bytes); }
    {
        int cur = -1; size_t k = 0;
        for (size_t i = 0; i < n_weights; ++i) if (slots[i].streamed) {
            if (slots[i].block != cur) { cur = slots[i].block; k = 0; }
            slots[i].dev_off = ring_off_k.at(k++);
        }
    }
    ring_dev_off[0] = dev; ring_dev_off[1] = dev + ring_bytes; dev += 2 * ring_bytes;
    for (size_t i = n_weights; i < slots.size(); ++i) { slots[i].dev_off = dev; dev += pad(slots[i].bytes); }
    block_bytes = dev;
}

void MimoVision::Impl::fill_host() {
    auto* base = static_cast<uint8_t*>(hw);
    for (const auto& sl : slots) {
        if (!sl.src) break;                                // weights come first
        const uint8_t* src = sl.src->data;
        if (sl.in_d) {                                     // BF16 [out, in] -> f16 Bt [in, out]
            auto* bt = reinterpret_cast<sycl::half*>(base + sl.off);
            const uint32_t out_d = sl.out_d, in_d = sl.in_d;
            #pragma omp parallel for schedule(static)
            for (uint32_t o = 0; o < out_d; ++o)
                for (uint32_t k = 0; k < in_d; ++k)
                    bt[size_t(k) * out_d + o] = sycl::half(mimov_bf16_at(src, size_t(o) * in_d + k));
        } else {
            auto* f = reinterpret_cast<float*>(base + sl.off);
            for (uint32_t i = 0; i < sl.out_d; ++i) f[i] = mimov_bf16_at(src, i);
        }
    }
    hw_hash = mimov_sample_hash(hw, hw_bytes);
}

uint64_t MimoVision::encode_bytes() const { return impl_->open ? impl_->block_bytes : 0; }

std::string MimoVision::stage_host(DeviceAllocator& alloc) {
    Impl& I = *impl_;
    if (!I.open) return "mimo26 vision: not loaded";
    if (I.alloc) return I.alloc == &alloc ? std::string{} : "mimo26 vision: allocator changed between calls";
    sycl::queue& q = alloc.queue();
    const size_t Mp = I.opt.max_patches;
    I.alloc = &alloc;                                      // the destructor frees the pinned buffers through it
    I.hw = sycl::malloc_host(I.hw_bytes, q);
    I.h_px = sycl::malloc_host<sycl::half>(Mp * kMimoVisPatchDim, q);
    I.h_tab = sycl::malloc_host<float>(4 * Mp * kRot, q);
    I.h_perm = sycl::malloc_host<uint32_t>(2 * (Mp / 4), q);
    I.h_out = sycl::malloc_host<float>((Mp / 4) * kOut, q);
    I.h_cap = sycl::malloc_host<float>(Mp * kD, q);
    if (!I.hw || !I.h_px || !I.h_tab || !I.h_perm || !I.h_out || !I.h_cap) return "mimo26 vision: pinned host allocation failed";
    for (auto& sl : I.slots) *sl.pp = nullptr;
    I.fill_host();
    return {};
}

std::string MimoVision::encode_gpu(DeviceAllocator& alloc, const std::vector<float>& pv, const MimoVisGeom& g,
                                   std::vector<float>& merged, const std::vector<uint32_t>& capture,
                                   std::vector<std::vector<float>>* captured) {
    Impl& I = *impl_;
    if (!I.open) return "mimo26 vision: not loaded";
    if (std::string e = stage_host(alloc); !e.empty()) return e;
    const uint32_t N = g.patches();
    if (!N || g.grid_h % 2 || g.grid_w % 2) return "mimo26 vision: the patch grid must be non-empty and even";
    if (pv.size() != size_t(N) * kMimoVisPatchDim) return "mimo26 vision: pixel_values do not match the grid";
    if (N > I.opt.max_patches)
        return "mimo26 vision: " + std::to_string(N) + " patches exceed the cap of " + std::to_string(I.opt.max_patches);
    const uint32_t NB = N / 4, lh = g.grid_h / 2, lw = g.grid_w / 2;
    sycl::queue& q = alloc.queue();
    struct Lease {
        sycl::queue& q; Impl& I; void* p = nullptr;
        ~Lease() {
            if (!p) return;
            try { q.wait(); } catch (...) {}
            sycl::free(p, q);
            for (auto& sl : I.slots) *sl.pp = nullptr;
        }
    } lease{q, I};
    lease.p = sycl::malloc_device(I.block_bytes, q);
    if (!lease.p) return "mimo26 vision: no device memory for the " + std::to_string(I.block_bytes >> 20) + " MiB encode block";
    for (auto& sl : I.slots) *sl.pp = sl.streamed ? nullptr : static_cast<uint8_t*>(lease.p) + sl.dev_off;
    if (I.hw_hash && mimov_sample_hash(I.hw, I.hw_bytes) != I.hw_hash) {
        std::fprintf(stderr, "[mimo26-vis] the staged tower weights changed since staging -- restaging\n");
        I.fill_host();
    }
    auto* hwb = static_cast<const uint8_t*>(I.hw);
    for (const auto& sl : I.slots)                                    // the resident weights (patch embed, vectors, merger)
        if (sl.src && !sl.streamed) q.memcpy(static_cast<uint8_t*>(lease.p) + sl.dev_off, hwb + sl.off, sl.bytes);
    // a block's five matrices ride into ring slot L % 2 just before its kernels (the in-order queue puts the copy after the
    // kernels of block L - 2, the slot's previous user)
    auto stream_block = [&](uint32_t L) {
        uint8_t* ring = static_cast<uint8_t*>(lease.p) + I.ring_dev_off[L % 2];
        for (auto& sl : I.slots)
            if (sl.streamed && sl.block == int(L)) { q.memcpy(ring + sl.dev_off, hwb + sl.off, sl.bytes); *sl.pp = ring + sl.dev_off; }
    };

    // host prep: f16 pixels; rope angle tables [N, 32] (h freqs, then w) in row order and in column order;
    // the column permutation (reference get_window_index_1d, col=True) and its inverse (argsort)
    for (size_t i = 0; i < pv.size(); ++i) I.h_px[i] = sycl::half(pv[i]);
    float inv[kRot / 2];
    for (uint32_t j = 0; j < kRot / 2; ++j) inv[j] = 1.0f / std::pow(kTheta, float(2 * j) / float(kRot));
    float *cs_row = I.h_tab, *sn_row = cs_row + size_t(N) * kRot, *cs_col = sn_row + size_t(N) * kRot,
          *sn_col = cs_col + size_t(N) * kRot;
    for (uint32_t n = 0; n < N; ++n) {
        const uint32_t b = n / 4, j = n % 4, hp = (b / lw) * 2 + j / 2, wp = (b % lw) * 2 + j % 2;
        for (uint32_t r = 0; r < kRot; ++r) {
            const float a = r < kRot / 2 ? float(hp) * inv[r] : float(wp) * inv[r - kRot / 2];
            cs_row[size_t(n) * kRot + r] = std::cos(a);
            sn_row[size_t(n) * kRot + r] = std::sin(a);
        }
    }
    uint32_t *col = I.h_perm, *back = I.h_perm + NB;
    for (uint32_t c = 0; c < lw; ++c)
        for (uint32_t r = 0; r < lh; ++r) col[c * lh + r] = r * lw + c;
    for (uint32_t k = 0; k < NB; ++k) back[col[k]] = k;
    for (uint32_t k = 0; k < NB; ++k)
        for (uint32_t j = 0; j < 4; ++j) {
            std::memcpy(&cs_col[(size_t(k) * 4 + j) * kRot], &cs_row[(size_t(col[k]) * 4 + j) * kRot], kRot * 4);
            std::memcpy(&sn_col[(size_t(k) * 4 + j) * kRot], &sn_row[(size_t(col[k]) * 4 + j) * kRot], kRot * 4);
        }
    q.memcpy(I.px, I.h_px, pv.size() * 2);
    q.memcpy(I.tab, I.h_tab, 4 * size_t(N) * kRot * 4);
    q.memcpy(I.perm, I.h_perm, 2 * size_t(NB) * 4);

    // ---- forward (in-order queue) ------------------------------------------------------------------------------
    gemm_fp16(q, I.px, I.patch, I.res, N, kD, kMimoVisPatchDim);             // the Conv3d as a GEMM, no bias
    const float scale = 1.0f / std::sqrt(float(kHd)), pscale = I.opt.prob_scale;
    const uint32_t Nc = (N + 127u) & ~127u, Tq = std::min(I.opt.attn_tile, N), R = I.opt.mlp_rows;
    float *res = I.res, *res2 = I.res2;
    bool in_col = false;
    for (uint32_t L = 0; L < kDepth; ++L) {
        const int kind = kKind[L];
        if (kind == 1 && (L == 0 || kKind[L - 1] != 1)) {                      // into column order
            mimov_permute_blocks(q, res, res2, I.perm, NB, kD);
            std::swap(res, res2);
            in_col = true;
        }
        if (L > 0 && kind != 1 && kKind[L - 1] == 1) {                          // back to row order
            mimov_permute_blocks(q, res, res2, I.perm + NB, NB, kD);
            std::swap(res, res2);
            in_col = false;
        }
        const float* cs = I.tab + (kind == 1 ? 2 : 0) * size_t(N) * kRot;
        const float* sn = cs + size_t(N) * kRot;
        auto& b = I.blk[L];
        stream_block(L);
        mimov_rms_norm_f16(q, res, b.n1, I.xh, N, kD);
        gemm_fp16(q, I.xh, b.qkv, I.Cq, N, kQkv, kD);
        mimov_bias_f16(q, I.Cq, b.qkv_b, I.qkv, N, kQkv);
        mimov_rope2d(q, I.qkv, cs, sn, N);
        const uint32_t window = kind == -1 ? 0u : kWindow;
        for (uint32_t kv = 0; kv < kKvHeads; ++kv) {
            mimov_gather_kv(q, I.qkv, I.kT, I.vh, N, Nc, kv);
            for (uint32_t h = kv * (kHeads / kKvHeads); h < (kv + 1) * (kHeads / kKvHeads); ++h) {
                mimov_gather_q(q, I.qkv, I.qh, N, h);
                for (uint32_t r0 = 0; r0 < N; r0 += Tq) {
                    const uint32_t n = std::min(Tq, N - r0);
                    gemm_fp16(q, I.qh + size_t(r0) * kHd, I.kT, I.S, n, Nc, kHd);
                    mimov_softmax_band(q, I.S, I.P, n, r0, N, Nc, scale, pscale, window, b.sinks, h);
                    gemm_fp16(q, I.P, I.vh, I.O, n, kHd, Nc);
                    mimov_attn_scatter(q, I.O, I.att + size_t(r0) * kQ, n, h, 1.0f / pscale);
                }
            }
        }
        gemm_fp16(q, I.att, b.proj, I.Cd, N, kD, kQ);
        mimov_bias_add_res(q, I.Cd, b.proj_b, nullptr, res, N, kD);
        mimov_rms_norm_f16(q, res, b.n2, I.xh, N, kD);
        for (uint32_t c0 = 0; c0 < N; c0 += R) {
            const uint32_t rows = std::min(R, N - c0);
            gemm_fp16(q, I.xh + size_t(c0) * kD, b.gate, I.Cg, rows, kFfn, kD);
            gemm_fp16(q, I.xh + size_t(c0) * kD, b.up, I.Cu, rows, kFfn, kD);
            mimov_swiglu_f16(q, I.Cg, b.gate_b, I.Cu, b.up_b, I.ffn, I.rs, rows);
            gemm_fp16(q, I.ffn, b.down, I.Cd + size_t(c0) * kD, rows, kD, kFfn);
            mimov_bias_add_res(q, I.Cd + size_t(c0) * kD, b.down_b, I.rs, res + size_t(c0) * kD, rows, kD);
        }
        if (captured && std::find(capture.begin(), capture.end(), L) != capture.end()) {
            const float* src = res;
            if (in_col) {                                                      // the gate compares in row order
                mimov_permute_blocks(q, res, res2, I.perm + NB, NB, kD);
                src = res2;
            }
            q.memcpy(I.h_cap, src, size_t(N) * kD * 4).wait();
            captured->emplace_back(I.h_cap, I.h_cap + size_t(N) * kD);
        }
    }
    // block 27 is a full-attention block: the sequence is back in row order for the merger
    mimov_layer_norm_f16(q, res, I.ln_q, I.xh, N, kD);
    gemm_fp16(q, I.xh, I.m0, I.Cm, NB, kMergeIn, kMergeIn);                   // xh read as [N/4, 5120]
    mimov_gelu_f16(q, I.Cm, I.mh, size_t(NB) * kMergeIn);
    gemm_fp16(q, I.mh, I.m2, I.Co, NB, kOut, kMergeIn);
    merged.resize(size_t(NB) * kOut);
    q.memcpy(I.h_out, I.Co, merged.size() * 4).wait();
    std::memcpy(merged.data(), I.h_out, merged.size() * 4);
    return {};
}

}  // namespace ie
