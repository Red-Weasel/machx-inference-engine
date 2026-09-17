// tests/unit/deepseek4_attn_segs_test.cpp — gate for the two-segment kv axis
// (Ds4KvSegs, docs/deepseek4/73 Phase 1).
//
// THE CLAIM: ds4_attention_segs over (a = kv[0, n_a) fp32, b = kv[n_a, n_kv)
// fp16) is BIT-IDENTICAL to ds4_attention over the contiguous fp32 kv whose b
// rows were rounded through fp16 first, for every dispatch shape (decode split
// kernels at T <= 16, the plain prefill kernel, and the XMX prefill kernel at
// head_dim 512 with a multiple of 32 heads), including a second XMX call with
// `kv_prepared = true` reusing the first call's staging.  The kernels read the
// same values in the same column order; the only thing that changed is WHERE
// column i's row lives and at what width, so nothing short of memcmp-equality
// is the right bar.  No model file, no parity blobs.  (Rounding b through fp16
// on the host is what the Phase 2 cache does at append time, with the same
// saturating cast; the XMX kernel converted the contiguous rows the same way.)
#undef NDEBUG
#include "ie/deepseek4_attn.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::printf("  FAIL: %s\n", what); ++g_fail; }
}

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// Sliding causal band over the first n_sl columns, -inf block bias with top_k
// random live columns over the rest — the real mask's shape.
std::vector<float> make_mask(uint32_t T, uint32_t n_sl, uint32_t n_c, uint32_t top_k,
                             uint64_t seed) {
    const uint32_t n_kv = n_sl + n_c;
    std::vector<float> m(size_t(T) * n_kv, kNegInf);
    std::mt19937_64 rng(seed);
    const float lowest = std::numeric_limits<float>::lowest();
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t i = 0; i < n_sl; ++i)
            m[size_t(t) * n_kv + i] = ((i <= t + 64) && (t + 64 - i < 128)) ? 0.f : lowest;
        std::vector<uint32_t> idx(n_c);
        for (uint32_t i = 0; i < n_c; ++i) idx[i] = i;
        std::shuffle(idx.begin(), idx.end(), rng);
        for (uint32_t j = 0; j < std::min(top_k, n_c); ++j)
            m[size_t(t) * n_kv + n_sl + idx[j]] = 0.f;
    }
    return m;
}

struct Dev {
    float *q = nullptr, *kv = nullptr, *mask = nullptr, *sinks = nullptr, *y1 = nullptr, *y2 = nullptr;
};

void gate(sycl::queue& q, const char* label, uint32_t T, uint32_t H, uint32_t D,
          uint32_t n_a, uint32_t n_b, uint32_t top_k, bool xmx, uint64_t seed, bool b16 = true) {
    const uint32_t n_kv = n_a + n_b;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hq(size_t(T) * H * D), hkv(size_t(n_kv) * D), hs(H);
    for (auto& v : hq)  v = nd(rng);
    for (auto& v : hkv) v = nd(rng);
    // b rows: what the fp16 cache holds is sat16(x); the contiguous fp32
    // reference must see the same values.
    std::vector<sycl::half> hb(size_t(std::max(n_b, 1u)) * D);
    for (size_t i = 0; b16 && i < size_t(n_b) * D; ++i) {
        float& x = hkv[size_t(n_a) * D + i];
        x = std::fmin(std::fmax(x, -65504.f), 65504.f);
        hb[i] = sycl::half(x);
        x = float(hb[i]);
    }
    for (auto& v : hs)  v = nd(rng);
    const std::vector<float> hm = make_mask(T, n_a, n_b, top_k, seed + 1);

    Dev d;
    d.q     = sycl::malloc_device<float>(hq.size(), q);
    d.kv    = sycl::malloc_device<float>(hkv.size(), q);
    d.mask  = sycl::malloc_device<float>(hm.size(), q);
    d.sinks = sycl::malloc_device<float>(hs.size(), q);
    d.y1    = sycl::malloc_device<float>(hq.size(), q);
    d.y2    = sycl::malloc_device<float>(hq.size(), q);
    q.memcpy(d.q, hq.data(), hq.size() * 4).wait();
    q.memcpy(d.kv, hkv.data(), hkv.size() * 4).wait();
    q.memcpy(d.mask, hm.data(), hm.size() * 4).wait();
    q.memcpy(d.sinks, hs.data(), hs.size() * 4).wait();
    q.memset(d.y1, 0, hq.size() * 4).wait();
    q.memset(d.y2, 0xFF, hq.size() * 4).wait();   // NaN-filled: a row the kernel skips shows

    const float scaling = 1.f / std::sqrt(float(D));
    // Segment b lives at a DIFFERENT address than kv + n_a*D would, so a kernel
    // that silently indexed past segment a into the contiguous buffer is caught.
    // 64 zeroed pad rows past n_b, as the cache guarantees (the XMX kernel's
    // last block may read into them).
    // A fp32 b segment (IE_DS4_KV16=0) is the raw rows at another address.
    const size_t eb = b16 ? 2 : 4;
    void* db = sycl::malloc_device<uint8_t>((size_t(n_b) + 64) * D * eb, q);
    q.memset(db, 0, (size_t(n_b) + 64) * D * eb).wait();
    if (b16) q.memcpy(db, hb.data(), size_t(n_b) * D * 2).wait();
    else     q.memcpy(db, hkv.data() + size_t(n_a) * D, size_t(n_b) * D * 4).wait();
    const ie::Ds4KvSegs segs{d.kv, n_a, n_b ? db : nullptr, n_b, b16};

    std::vector<float> o1(hq.size()), o2(hq.size());
    if (xmx) {
        ie::ds4_attention_xmx(q, d.q, d.kv, d.mask, d.sinks, d.y1, T, H, D, n_kv, scaling).wait();
        ie::ds4_attention_xmx_segs(q, d.q, segs, d.mask, d.sinks, d.y2, T, H, D, scaling, false).wait();
    } else {
        ie::ds4_attention(q, d.q, d.kv, d.mask, d.sinks, d.y1, T, H, D, n_kv, scaling).wait();
        ie::ds4_attention_segs(q, d.q, segs, d.mask, d.sinks, d.y2, T, H, D, scaling).wait();
    }
    q.memcpy(o1.data(), d.y1, o1.size() * 4).wait();
    q.memcpy(o2.data(), d.y2, o2.size() * 4).wait();
    const bool same = std::memcmp(o1.data(), o2.data(), o1.size() * 4) == 0;
    bool finite = true;
    for (float v : o1) finite = finite && std::isfinite(v);
    std::printf("  %-34s T=%-3u H=%-2u D=%-3u n_a=%-4u n_b=%-5u  %s\n", label, T, H, D, n_a, n_b,
                same ? "bit-identical" : "DIFFERS");
    check(same, "two-segment output differs from the contiguous one");
    check(finite, "contiguous output is not finite");

    if (xmx && T > 16) {
        // Second strip of the same layer: kv_prepared reuses the staged fp16 kv
        // axis.  Must equal a cold call on the same rows.
        q.memset(d.y2, 0xFF, hq.size() * 4).wait();
        ie::ds4_attention_xmx_segs(q, d.q, segs, d.mask, d.sinks, d.y2, T, H, D, scaling, true).wait();
        q.memcpy(o2.data(), d.y2, o2.size() * 4).wait();
        const bool same2 = std::memcmp(o1.data(), o2.data(), o1.size() * 4) == 0;
        std::printf("  %-34s kv_prepared reuse                              %s\n", "",
                    same2 ? "bit-identical" : "DIFFERS");
        check(same2, "kv_prepared reuse differs from the cold call");
    }

    sycl::free(d.q, q); sycl::free(d.kv, q); sycl::free(d.mask, q); sycl::free(d.sinks, q);
    sycl::free(d.y1, q); sycl::free(d.y2, q); sycl::free(db, q);
}

}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::puts("§1 fp32 kernels: decode split shapes, prefill plain shape");
    gate(q, "decode T=1 (split 64)",        1, 8, 512, 131, 1024, 512, false, 11);
    gate(q, "decode T=4 (split 16)",        4, 8, 512, 131, 1024, 512, false, 12);
    gate(q, "decode T=16 (split 16)",      16, 8, 512, 130, 2048, 512, false, 13);
    gate(q, "prefill T=64 (plain)",        64, 8, 512, 191, 1024, 512, false, 14);
    gate(q, "indexer head_dim 128",         4, 8, 128, 129,  777, 512, false, 15);
    gate(q, "sliding-only (n_b = 0)",      20, 8, 512, 147,    0,   0, false, 16);
    gate(q, "n_a = 0 (all compressed)",     3, 4, 512,   0,  600, 512, false, 17);

    std::puts("\n§2 XMX prefill kernel (head_dim 512, 32 heads), odd n_a straddling a key block");
    gate(q, "xmx T=20",                    20, 32, 512, 147, 1024, 512, true, 21);
    gate(q, "xmx T=64",                    64, 32, 512, 191, 2049, 512, true, 22);
    gate(q, "xmx T=33 n_a = 64 (aligned)", 33, 32, 512,  64,  700, 512, true, 23);
    gate(q, "xmx sliding-only",            40, 32, 512, 167,    0,   0, true, 24);

    std::puts("\n§3 fp32 b segment (IE_DS4_KV16=0): the phase-1 contract, raw rows in place");
    gate(q, "decode T=1 fp32 b",             1, 8, 512, 131, 1024, 512, false, 31, false);
    gate(q, "prefill T=64 fp32 b",          64, 8, 512, 191, 1024, 512, false, 32, false);
    gate(q, "xmx T=20 fp32 b (staged)",     20, 32, 512, 147, 1024, 512, true,  33, false);

    if (g_fail) { std::printf("\nFAILED: %d check(s)\n", g_fail); return 1; }
    std::puts("\nOK — the two-segment kv axis is bit-identical to the contiguous one on every shape.");
    return 0;
}
