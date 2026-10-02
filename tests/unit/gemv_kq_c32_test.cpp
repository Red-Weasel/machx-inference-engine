// tests/unit/gemv_kq_c32_test.cpp -- P4 B21: the c32 Q6_K / Q5_K GEMV (src/ops/gemv_kq_c32.cpp).
//
//  1. Host, no device: repack_q{6,5}_K_to_c32 of random GGUF blocks, then a host dequant written from the documented c32
//     layout, must equal ggml's dequant (ie::ref::dequant_q{6,5}_K_buffer) bit for bit: the repack is a pure layout move.
//  2. Device: gemv_q{6,5}k_c32_q8 at T = 1 against a double-precision reference Σ_k W[n,k] * (d8 * q8[k]) over the SAME
//     quantized activation (the int-dot's only approximation is the activation's int8 rounding, which both sides share).
//  3. Device: every T in {2, 3, 4, 5, 8, 16, 19}: row t of the T-row call == the T = 1 call on row t, bit for bit (the
//     request lanes' same-reduction-order rule).
//  4. Device: dequant_q{6,5}k_c32_to_Bt and embedding_lookup_q5k against fp16(ggml dequant) (<= 1 ulp: device fma).
// `--bench` (not a ctest): GB/s per 27B decode shape at T = 1 for c32 Q6_K / Q5_K, the Q8_0-SoA kernel and the old Q6 SoA.
// Any SYCL device: ONEAPI_DEVICE_SELECTOR=opencl:cpu runs it without a GPU.
#include "ie/dequant_ref.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ie;

namespace {
int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("[FAIL] " __VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

float h2f(uint16_t h) { return float(sycl::bit_cast<sycl::half>(h)); }

void rand_q6(std::mt19937& r, std::vector<block_q6_K>& b) {
    std::uniform_int_distribution<int> byte(0, 255), sc(-64, 63);
    std::uniform_real_distribution<float> d(0.0005f, 0.004f);
    for (auto& x : b) {
        for (auto& v : x.ql) v = uint8_t(byte(r));
        for (auto& v : x.qh) v = uint8_t(byte(r));
        for (auto& v : x.scales) v = int8_t(sc(r));
        x.d = fp32_to_fp16(d(r));
    }
}
void rand_q5(std::mt19937& r, std::vector<block_q5_K>& b) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> d(0.0005f, 0.004f);
    for (auto& x : b) {
        for (auto& v : x.qs) v = uint8_t(byte(r));
        for (auto& v : x.qh) v = uint8_t(byte(r));
        for (auto& v : x.scales) v = uint8_t(byte(r));
        x.d = fp32_to_fp16(d(r));
        x.dmin = fp32_to_fp16(d(r));
    }
}

// Host dequant of one column from the c32 streams, straight from the layout comment in gemv_kq_c32.cpp.
void c32_col_q6(const uint8_t* lo, const uint8_t* hi, const int8_t* sc, const uint16_t* d, uint32_t K, uint32_t n, float* out) {
    for (uint32_t k = 0; k < K; ++k) {
        const uint32_t c = k / 32, e = k % 32, hh = e / 16, j = e % 16, s = j / 4, b = j % 4;
        const uint8_t lb = lo[uint64_t(n) * (K / 2) + 16 * c + j];
        const uint32_t nib = hh ? (lb >> 4) : (lb & 0xF);
        const uint32_t h2 = (hi[uint64_t(n) * (K / 4) + 8 * c + 4 * hh + b] >> (2 * s)) & 3;
        const int q = int(nib | (h2 << 4)) - 32;
        out[k] = h2f(d[uint64_t(n) * (K / 256) + k / 256]) * float(sc[uint64_t(n) * (K / 16) + k / 16]) * float(q);
    }
}
void c32_col_q5(const uint8_t* lo, const uint32_t* hi, const uint16_t* sm, const uint32_t* dm, uint32_t K, uint32_t n, float* out) {
    for (uint32_t k = 0; k < K; ++k) {
        const uint32_t c = k / 32, e = k % 32, hh = e / 16, j = e % 16;
        const uint8_t lb = lo[uint64_t(n) * (K / 2) + 16 * c + j];
        const uint32_t nib = hh ? (lb >> 4) : (lb & 0xF);
        const uint32_t h1 = (hi[uint64_t(n) * (K / 32) + c] >> (8 * (e % 4) + e / 4)) & 1;
        const uint16_t s = sm[uint64_t(n) * (K / 32) + c];
        const uint32_t w = dm[uint64_t(n) * (K / 256) + k / 256];
        const float d1 = h2f(uint16_t(w & 0xFFFF)) * float(s & 0xFF), m1 = h2f(uint16_t(w >> 16)) * float(s >> 8);
        out[k] = d1 * float(nib | (h1 << 4)) - m1;
    }
}

struct Streams {   // host c32 streams of one weight (either type)
    std::vector<uint8_t> lo, hi, sc, d;
};

Streams repack(bool q6, const void* W, uint32_t K, uint32_t N) {
    Streams s;
    s.lo.assign(uint64_t(N) * K / 2, 0);
    s.hi.assign(q6 ? uint64_t(N) * K / 4 : uint64_t(N) * K / 8, 0);
    s.sc.assign(uint64_t(N) * K / 16, 0);
    s.d.assign(q6 ? uint64_t(N) * K / 128 : uint64_t(N) * K / 64, 0);
    if (q6) repack_q6_K_to_c32(W, K, N, s.lo.data(), s.hi.data(), reinterpret_cast<int8_t*>(s.sc.data()),
                               reinterpret_cast<uint16_t*>(s.d.data()));
    else    repack_q5_K_to_c32(W, K, N, s.lo.data(), reinterpret_cast<uint32_t*>(s.hi.data()),
                               reinterpret_cast<uint16_t*>(s.sc.data()), reinterpret_cast<uint32_t*>(s.d.data()));
    return s;
}

void host_col(bool q6, const Streams& s, uint32_t K, uint32_t n, float* out) {
    if (q6) c32_col_q6(s.lo.data(), s.hi.data(), reinterpret_cast<const int8_t*>(s.sc.data()),
                       reinterpret_cast<const uint16_t*>(s.d.data()), K, n, out);
    else    c32_col_q5(s.lo.data(), reinterpret_cast<const uint32_t*>(s.hi.data()),
                       reinterpret_cast<const uint16_t*>(s.sc.data()), reinterpret_cast<const uint32_t*>(s.d.data()), K, n, out);
}

struct Dev {   // device copies
    sycl::queue& q;
    void* p[4] = {};
    Dev(sycl::queue& qq, const Streams& s) : q(qq) {
        const std::vector<uint8_t>* v[4] = {&s.lo, &s.hi, &s.sc, &s.d};
        for (int i = 0; i < 4; ++i) { p[i] = sycl::malloc_device(v[i]->size(), q); q.memcpy(p[i], v[i]->data(), v[i]->size()); }
        q.wait();
    }
    ~Dev() { for (void* x : p) sycl::free(x, q); }
    sycl::event gemv(bool q6, const void* xq8, sycl::half* y, uint32_t K, uint32_t N, uint32_t T) {
        if (q6) return gemv_q6k_c32_q8(q, xq8, static_cast<uint8_t*>(p[0]), static_cast<uint8_t*>(p[1]),
                                       static_cast<int8_t*>(p[2]), static_cast<uint16_t*>(p[3]), y, K, N, T);
        return gemv_q5k_c32_q8(q, xq8, static_cast<uint8_t*>(p[0]), static_cast<uint32_t*>(p[1]),
                               static_cast<uint16_t*>(p[2]), static_cast<uint32_t*>(p[3]), y, K, N, T);
    }
    sycl::event deq(bool q6, sycl::half* Bt, uint32_t K, uint32_t N) {
        if (q6) return dequant_q6k_c32_to_Bt(q, static_cast<uint8_t*>(p[0]), static_cast<uint8_t*>(p[1]),
                                             static_cast<int8_t*>(p[2]), static_cast<uint16_t*>(p[3]), Bt, K, N);
        return dequant_q5k_c32_to_Bt(q, static_cast<uint8_t*>(p[0]), static_cast<uint32_t*>(p[1]),
                                     static_cast<uint16_t*>(p[2]), static_cast<uint32_t*>(p[3]), Bt, K, N);
    }
};

int ulp16(sycl::half a, sycl::half b) {   // distance in fp16 steps (sign-magnitude -> ordered)
    auto ord = [](sycl::half h) { const int u = sycl::bit_cast<uint16_t>(h); return (u & 0x8000) ? -(u & 0x7FFF) : u; };
    return std::abs(ord(a) - ord(b));
}

void run_type(sycl::queue& q, bool q6, uint32_t K, uint32_t N, uint32_t seed) {
    const char* tn = q6 ? "Q6_K" : "Q5_K";
    std::mt19937 r(seed);
    const uint64_t nblk = uint64_t(N) * (K / 256);
    std::vector<uint8_t> raw;
    std::vector<float> ref(uint64_t(N) * K);
    if (q6) {
        std::vector<block_q6_K> b(nblk); rand_q6(r, b);
        raw.assign(reinterpret_cast<uint8_t*>(b.data()), reinterpret_cast<uint8_t*>(b.data() + nblk));
        ie::ref::dequant_q6_K_buffer(b.data(), ref.size(), ref.data());
    } else {
        std::vector<block_q5_K> b(nblk); rand_q5(r, b);
        raw.assign(reinterpret_cast<uint8_t*>(b.data()), reinterpret_cast<uint8_t*>(b.data() + nblk));
        ie::ref::dequant_q5_K_buffer(b.data(), ref.size(), ref.data());
    }
    const Streams s = repack(q6, raw.data(), K, N);

    // 1. repack == ggml dequant, bit for bit
    std::vector<float> col(K);
    uint64_t bad = 0;
    for (uint32_t n = 0; n < N; ++n) {
        host_col(q6, s, K, n, col.data());
        for (uint32_t k = 0; k < K; ++k) bad += std::memcmp(&col[k], &ref[uint64_t(n) * K + k], 4) != 0;
    }
    CHECK(bad == 0, "%s [%u,%u] repack: %llu elements differ from ggml's dequant", tn, K, N, (unsigned long long)bad);
    if (bad == 0) std::printf("[ ok ] %s [%u,%u] repack == ggml dequant (%llu elements)\n", tn, K, N,
                              (unsigned long long)(uint64_t(N) * K));

    // 2./3. GEMV at T = 1 vs double reference, rows vs T = 1
    constexpr uint32_t TMAX = 19;
    std::uniform_real_distribution<float> xf(-1.f, 1.f);
    std::vector<sycl::half> x(uint64_t(TMAX) * K);
    for (auto& v : x) v = sycl::half(xf(r));
    Dev dv(q, s);
    auto* dx  = sycl::malloc_device<sycl::half>(x.size(), q);
    void* dq8 = sycl::malloc_device(uint64_t(TMAX) * (K / 32) * sizeof(block_q8_1x), q);
    auto* y1  = sycl::malloc_device<sycl::half>(uint64_t(TMAX) * N, q);
    auto* yT  = sycl::malloc_device<sycl::half>(uint64_t(TMAX) * N, q);
    q.memcpy(dx, x.data(), x.size() * 2).wait();
    quantize_q8_1(q, dx, dq8, uint32_t(uint64_t(TMAX) * K)).wait();
    std::vector<block_q8_1x> hq8(uint64_t(TMAX) * (K / 32));
    q.memcpy(hq8.data(), dq8, hq8.size() * sizeof(block_q8_1x)).wait();
    const uint64_t row_bytes = uint64_t(K / 32) * sizeof(block_q8_1x);
    for (uint32_t t = 0; t < TMAX; ++t)
        dv.gemv(q6, static_cast<uint8_t*>(dq8) + t * row_bytes, y1 + uint64_t(t) * N, K, N, 1);
    q.wait();
    std::vector<sycl::half> h1(uint64_t(TMAX) * N), hT(uint64_t(TMAX) * N);
    q.memcpy(h1.data(), y1, h1.size() * 2).wait();
    std::vector<double> want(3ull * N);
    double worst = 0, sum_abs = 0;
    for (uint32_t t = 0; t < 3; ++t)
        for (uint32_t n = 0; n < N; ++n) {
            double acc = 0;
            for (uint32_t k = 0; k < K; ++k) {
                const block_q8_1x& b = hq8[uint64_t(t) * (K / 32) + k / 32];
                acc += double(ref[uint64_t(n) * K + k]) * double(b.d) * double(b.qs[k % 32]);
            }
            want[uint64_t(t) * N + n] = acc;
            const double got = double(float(h1[uint64_t(t) * N + n]));
            worst = std::max(worst, std::fabs(got - acc) / (std::fabs(acc) + 1e-3));
            sum_abs += std::fabs(acc);
        }
    const double mean_abs = sum_abs / (3.0 * N);
    uint64_t far = 0;
    for (uint64_t i = 0; i < 3ull * N; ++i) {
        // fp16 output rounding (2^-11 relative) + fp32 accumulation over K; the Q5 min term uses the rounded s8
        if (std::fabs(double(float(h1[i])) - want[i]) > std::fabs(want[i]) / 1024.0 + 2e-3 * mean_abs) ++far;
    }
    CHECK(far == 0, "%s [%u,%u] T=1 vs double reference: %llu of %u outputs off (worst rel %.3g)", tn, K, N,
          (unsigned long long)far, 3 * N, worst);
    if (far == 0) std::printf("[ ok ] %s [%u,%u] T=1 == reference (mean |y| %.3g)\n", tn, K, N, mean_abs);

    for (uint32_t T : {2u, 3u, 4u, 5u, 8u, 16u, 19u}) {
        q.memset(yT, 0xFF, uint64_t(TMAX) * N * 2).wait();
        dv.gemv(q6, dq8, yT, K, N, T).wait();
        q.memcpy(hT.data(), yT, uint64_t(T) * N * 2).wait();
        const bool same = std::memcmp(hT.data(), h1.data(), uint64_t(T) * N * 2) == 0;
        CHECK(same, "%s [%u,%u] T=%u rows differ from the T=1 calls", tn, K, N, T);
        if (same) std::printf("[ ok ] %s [%u,%u] T=%u rows == T=1 bit for bit\n", tn, K, N, T);
    }

    // 4. prefill dequant
    auto* Bt = sycl::malloc_device<sycl::half>(uint64_t(K) * N, q);
    dv.deq(q6, Bt, K, N).wait();
    std::vector<sycl::half> hb(uint64_t(K) * N);
    q.memcpy(hb.data(), Bt, hb.size() * 2).wait();
    uint64_t off1 = 0, offmore = 0;
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t k = 0; k < K; ++k) {
            const int u = ulp16(hb[uint64_t(k) * N + n], sycl::half(ref[uint64_t(n) * K + k]));
            off1 += u == 1; offmore += u > 1;
        }
    CHECK(offmore == 0, "%s [%u,%u] dequant: %llu elements > 1 ulp from fp16(ggml)", tn, K, N, (unsigned long long)offmore);
    if (offmore == 0) std::printf("[ ok ] %s [%u,%u] dequant == fp16(ggml) (%llu at 1 ulp)\n", tn, K, N, (unsigned long long)off1);
    sycl::free(Bt, q); sycl::free(dx, q); sycl::free(dq8, q); sycl::free(y1, q); sycl::free(yT, q);
}

void test_embed_q5(sycl::queue& q) {
    const uint32_t H = 512, V = 37;
    std::mt19937 r(99);
    std::vector<block_q5_K> b(uint64_t(V) * (H / 256)); rand_q5(r, b);
    std::vector<float> ref(uint64_t(V) * H);
    ie::ref::dequant_q5_K_buffer(b.data(), ref.size(), ref.data());
    const std::vector<int32_t> ids = {0, 36, 5, 17, 5};
    auto* dW = sycl::malloc_device(b.size() * sizeof(block_q5_K), q);
    auto* di = sycl::malloc_device<int32_t>(ids.size(), q);
    auto* dy = sycl::malloc_device<sycl::half>(ids.size() * H, q);
    q.memcpy(dW, b.data(), b.size() * sizeof(block_q5_K));
    q.memcpy(di, ids.data(), ids.size() * 4).wait();
    embedding_lookup_q5k(q, di, dW, dy, uint32_t(ids.size()), H).wait();
    std::vector<sycl::half> hy(ids.size() * H);
    q.memcpy(hy.data(), dy, hy.size() * 2).wait();
    uint64_t off = 0;
    for (size_t t = 0; t < ids.size(); ++t)
        for (uint32_t k = 0; k < H; ++k) off += ulp16(hy[t * H + k], sycl::half(ref[uint64_t(ids[t]) * H + k])) > 1;
    CHECK(off == 0, "embedding_lookup_q5k: %llu elements > 1 ulp", (unsigned long long)off);
    if (off == 0) std::printf("[ ok ] embedding_lookup_q5k == fp16(ggml)\n");
    sycl::free(dW, q); sycl::free(di, q); sycl::free(dy, q);
}

// MoE c32 kernels: a bank of E experts [K, N, E] (expert-major, so repacked over E*N columns). Per-slot gate_up / down at
// T = 1 vs a double reference, the T = 4 rows call == four T = 1 calls bit for bit, and the expert-batched prefill kernels
// vs the reference.
void test_moe(sycl::queue& q, bool q6gu, bool q6d, uint32_t seed) {
    const uint32_t H = 2048, EFF = 512, E = 5, Kt = 3, T = 4;
    std::mt19937 r(seed);
    auto make = [&](bool q6, uint32_t K, uint32_t Ncols, std::vector<uint8_t>& raw, std::vector<float>& ref) {
        const uint64_t nblk = uint64_t(Ncols) * (K / 256);
        ref.assign(uint64_t(Ncols) * K, 0.f);
        if (q6) { std::vector<block_q6_K> b(nblk); rand_q6(r, b); raw.assign((uint8_t*)b.data(), (uint8_t*)(b.data() + nblk));
                  ie::ref::dequant_q6_K_buffer(b.data(), ref.size(), ref.data()); }
        else    { std::vector<block_q5_K> b(nblk); rand_q5(r, b); raw.assign((uint8_t*)b.data(), (uint8_t*)(b.data() + nblk));
                  ie::ref::dequant_q5_K_buffer(b.data(), ref.size(), ref.data()); }
    };
    std::vector<uint8_t> rg, ru, rd; std::vector<float> fg, fu, fd;
    make(q6gu, H, E * EFF, rg, fg); make(q6gu, H, E * EFF, ru, fu); make(q6d, EFF, E * H, rd, fd);
    const Streams sg = repack(q6gu, rg.data(), H, E * EFF), su = repack(q6gu, ru.data(), H, E * EFF),
                  sd = repack(q6d, rd.data(), EFF, E * H);
    Dev dg(q, sg), du(q, su), dd(q, sd);
    auto bank = [](Dev& d, bool q6) {
        C32Bank b; b.lo = static_cast<const uint8_t*>(d.p[0]); b.hi = d.p[1]; b.sc = d.p[2]; b.d = d.p[3]; b.kind = q6 ? 6 : 5;
        return b;
    };
    const C32Bank bg = bank(dg, q6gu), bu = bank(du, q6gu), bd = bank(dd, q6d);
    const char* tn = q6gu ? (q6d ? "Q6/Q6" : "Q6/Q5") : (q6d ? "Q5/Q6" : "Q5/Q5");

    std::uniform_real_distribution<float> xf(-0.1f, 0.1f);   // small enough that silu(gate)*up*W stays inside fp16
    std::vector<sycl::half> x(uint64_t(T) * H);
    for (auto& v : x) v = sycl::half(xf(r));
    std::vector<int32_t> idx(T * Kt);
    std::vector<sycl::half> tw(T * Kt);
    for (uint32_t i = 0; i < T * Kt; ++i) { idx[i] = int32_t((i * 7 + seed) % E); tw[i] = sycl::half(0.1f + 0.1f * float(i % 5)); }
    auto* dx = sycl::malloc_device<sycl::half>(x.size(), q);
    auto* dxq = sycl::malloc_device(uint64_t(T) * (H / 32) * sizeof(block_q8_1x), q);
    auto* didx = sycl::malloc_device<int32_t>(idx.size(), q);
    auto* dtw = sycl::malloc_device<sycl::half>(tw.size(), q);
    auto* dh1 = sycl::malloc_device<sycl::half>(uint64_t(T) * Kt * EFF, q);
    auto* dhT = sycl::malloc_device<sycl::half>(uint64_t(T) * Kt * EFF, q);
    auto* dhq = sycl::malloc_device(uint64_t(T) * Kt * (EFF / 32) * sizeof(block_q8_1x), q);
    auto* dy1 = sycl::malloc_device<sycl::half>(uint64_t(T) * Kt * H, q);
    auto* dyT = sycl::malloc_device<sycl::half>(uint64_t(T) * Kt * H, q);
    q.memcpy(dx, x.data(), x.size() * 2);
    q.memcpy(didx, idx.data(), idx.size() * 4);
    q.memcpy(dtw, tw.data(), tw.size() * 2).wait();
    quantize_q8_1(q, dx, dxq, T * H).wait();
    const uint64_t xrow = uint64_t(H / 32) * sizeof(block_q8_1x);
    for (uint32_t t = 0; t < T; ++t)
        moe_gate_up_silu_c32(q, static_cast<uint8_t*>(dxq) + t * xrow, bg, bu, didx + t * Kt, dh1 + uint64_t(t) * Kt * EFF,
                             1, Kt, H, EFF);
    moe_gate_up_silu_c32(q, dxq, bg, bu, didx, dhT, T, Kt, H, EFF);
    q.wait();
    std::vector<sycl::half> h1(uint64_t(T) * Kt * EFF), hT(h1.size());
    q.memcpy(h1.data(), dh1, h1.size() * 2); q.memcpy(hT.data(), dhT, hT.size() * 2).wait();
    bool same = std::memcmp(h1.data(), hT.data(), h1.size() * 2) == 0;
    CHECK(same, "moe %s gate_up: T=%u slots differ from the T=1 calls", tn, T);
    // reference (on the quantized activation) for the gate/up silu
    std::vector<block_q8_1x> hx(uint64_t(T) * (H / 32));
    q.memcpy(hx.data(), dxq, hx.size() * sizeof(block_q8_1x)).wait();
    uint64_t far = 0; double mean = 0;
    std::vector<double> want(uint64_t(T) * Kt * EFF);
    for (uint32_t tk = 0; tk < T * Kt; ++tk)
        for (uint32_t n = 0; n < EFF; ++n) {
            const uint64_t col = uint64_t(idx[tk]) * EFF + n;
            double ga = 0, ua = 0;
            for (uint32_t k = 0; k < H; ++k) {
                const block_q8_1x& b = hx[uint64_t(tk / Kt) * (H / 32) + k / 32];
                const double xv = double(b.d) * double(b.qs[k % 32]);
                ga += double(fg[col * H + k]) * xv; ua += double(fu[col * H + k]) * xv;
            }
            want[uint64_t(tk) * EFF + n] = ga / (1.0 + std::exp(-ga)) * ua;
            mean += std::fabs(want[uint64_t(tk) * EFF + n]);
        }
    mean /= double(want.size());
    for (uint64_t i = 0; i < want.size(); ++i)
        if (std::fabs(double(float(h1[i])) - want[i]) > std::fabs(want[i]) / 256.0 + 2e-3 * mean) ++far;
    CHECK(far == 0, "moe %s gate_up vs reference: %llu of %zu off", tn, (unsigned long long)far, want.size());
    if (same && far == 0) std::printf("[ ok ] moe %s gate_up: T=1 == reference, T=%u == T=1 bit for bit\n", tn, T);

    // down: activation = quantized h (T*Kt rows)
    quantize_q8_1(q, dh1, dhq, T * Kt * EFF).wait();
    const uint64_t hrow = uint64_t(EFF / 32) * sizeof(block_q8_1x);
    for (uint32_t t = 0; t < T; ++t)
        moe_down_c32(q, static_cast<uint8_t*>(dhq) + uint64_t(t) * Kt * hrow, bd, didx + t * Kt, dtw + t * Kt,
                     dy1 + uint64_t(t) * Kt * H, 1, Kt, EFF, H);
    moe_down_c32(q, dhq, bd, didx, dtw, dyT, T, Kt, EFF, H);
    q.wait();
    std::vector<sycl::half> y1(uint64_t(T) * Kt * H), yT(y1.size());
    q.memcpy(y1.data(), dy1, y1.size() * 2); q.memcpy(yT.data(), dyT, yT.size() * 2).wait();
    same = std::memcmp(y1.data(), yT.data(), y1.size() * 2) == 0;
    CHECK(same, "moe %s down: T=%u slots differ from the T=1 calls", tn, T);
    std::vector<block_q8_1x> hh(uint64_t(T) * Kt * (EFF / 32));
    q.memcpy(hh.data(), dhq, hh.size() * sizeof(block_q8_1x)).wait();
    far = 0; mean = 0;
    std::vector<double> wy(uint64_t(T) * Kt * H);
    for (uint32_t tk = 0; tk < T * Kt; ++tk)
        for (uint32_t n = 0; n < H; ++n) {
            const uint64_t col = uint64_t(idx[tk]) * H + n;
            double a = 0;
            for (uint32_t k = 0; k < EFF; ++k) {
                const block_q8_1x& b = hh[uint64_t(tk) * (EFF / 32) + k / 32];
                a += double(fd[col * EFF + k]) * double(b.d) * double(b.qs[k % 32]);
            }
            wy[uint64_t(tk) * H + n] = double(float(tw[tk])) * a;
            mean += std::fabs(wy[uint64_t(tk) * H + n]);
        }
    mean /= double(wy.size());
    double worst = 0, worst_want = 0, worst_got = 0;
    for (uint64_t i = 0; i < wy.size(); ++i) {
        const double err = std::fabs(double(float(y1[i])) - wy[i]);
        if (err > std::fabs(wy[i]) / 256.0 + 2e-3 * mean) ++far;
        if (err > worst) { worst = err; worst_want = wy[i]; worst_got = double(float(y1[i])); }
    }
    CHECK(far == 0, "moe %s down vs reference: %llu of %zu off (worst |err| %.4g: want %.6g got %.6g; mean |y| %.4g)", tn,
          (unsigned long long)far, wy.size(), worst, worst_want, worst_got, mean);
    if (same && far == 0) std::printf("[ ok ] moe %s down: T=1 == reference, T=%u == T=1 bit for bit\n", tn, T);

    // prefill: sort the T*Kt slots by expert (host), gather x rows, run the batched kernels, compare to the per-slot ones
    std::vector<uint32_t> off(E + 1, 0);
    for (int32_t e : idx) ++off[uint32_t(e) + 1];
    for (uint32_t e = 0; e < E; ++e) off[e + 1] += off[e];
    std::vector<uint32_t> cur(E, 0), slot_of(T * Kt);
    for (uint32_t tk = 0; tk < T * Kt; ++tk) slot_of[off[idx[tk]] + cur[idx[tk]]++] = tk;
    std::vector<block_q8_1x> xp(uint64_t(T) * Kt * (H / 32)), hp(uint64_t(T) * Kt * (EFF / 32));
    std::vector<sycl::half> sw(T * Kt);
    for (uint32_t pos = 0; pos < T * Kt; ++pos) {
        const uint32_t tk = slot_of[pos];
        std::memcpy(&xp[uint64_t(pos) * (H / 32)], &hx[uint64_t(tk / Kt) * (H / 32)], (H / 32) * sizeof(block_q8_1x));
        std::memcpy(&hp[uint64_t(pos) * (EFF / 32)], &hh[uint64_t(tk) * (EFF / 32)], (EFF / 32) * sizeof(block_q8_1x));
        sw[pos] = tw[tk];
    }
    auto* doff = sycl::malloc_device<uint32_t>(off.size(), q);
    auto* dxp = sycl::malloc_device(xp.size() * sizeof(block_q8_1x), q);
    auto* dhp = sycl::malloc_device(hp.size() * sizeof(block_q8_1x), q);
    auto* dsw = sycl::malloc_device<sycl::half>(sw.size(), q);
    auto* dho = sycl::malloc_device<sycl::half>(uint64_t(T) * Kt * EFF, q);
    auto* dyo = sycl::malloc_device<sycl::half>(uint64_t(T) * Kt * H, q);
    q.memcpy(doff, off.data(), off.size() * 4); q.memcpy(dxp, xp.data(), xp.size() * sizeof(block_q8_1x));
    q.memcpy(dhp, hp.data(), hp.size() * sizeof(block_q8_1x)); q.memcpy(dsw, sw.data(), sw.size() * 2).wait();
    moe_prefill_gate_up_silu_c32(q, dxp, bg, bu, doff, dho, E, H, EFF);
    moe_prefill_down_c32(q, dhp, bd, doff, dsw, dyo, E, H, EFF);
    q.wait();
    std::vector<sycl::half> ho(uint64_t(T) * Kt * EFF), yo(uint64_t(T) * Kt * H);
    q.memcpy(ho.data(), dho, ho.size() * 2); q.memcpy(yo.data(), dyo, yo.size() * 2).wait();
    uint64_t diff = 0;
    for (uint32_t pos = 0; pos < T * Kt; ++pos) {
        const uint32_t tk = slot_of[pos];
        diff += std::memcmp(&ho[uint64_t(pos) * EFF], &h1[uint64_t(tk) * EFF], EFF * 2) != 0;
        diff += std::memcmp(&yo[uint64_t(pos) * H], &y1[uint64_t(tk) * H], H * 2) != 0;
    }
    CHECK(diff == 0, "moe %s prefill: %llu sorted rows differ from the per-slot kernels", tn, (unsigned long long)diff);
    if (diff == 0) std::printf("[ ok ] moe %s prefill (expert-batched) == per-slot bit for bit\n", tn);
    for (void* p : {(void*)dx, dxq, (void*)didx, (void*)dtw, (void*)dh1, (void*)dhT, dhq, (void*)dy1, (void*)dyT, (void*)doff,
                    dxp, dhp, (void*)dsw, (void*)dho, (void*)dyo}) sycl::free(p, q);
}

void bench(sycl::queue& q) {
    struct S { uint32_t K, N; const char* name; };
    const S shapes[] = {{5120, 10240, "(warm-up)"}, {5120, 10240, "attn_qkv"}, {5120, 6144, "attn_gate"}, {6144, 5120, "ssm_out/attn_output"},
                        {5120, 12288, "attn_q"}, {5120, 1024, "attn_k/v"}, {5120, 17408, "ffn_gate/up"},
                        {17408, 5120, "ffn_down"}, {5120, 248320, "head"}};
    std::mt19937 r(7);
    auto time_it = [&](auto&& f) {
        for (int i = 0; i < 5; ++i) f();
        q.wait();
        const int it = 40;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i) f();
        q.wait();
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / it;
    };
    std::printf("%-20s %8s %8s | %s\n", "shape", "K", "N", "GB/s (us) per kernel at T=1; rows T=4 / T=16 us");
    for (const S& s : shapes) {
        const uint32_t K = s.K, N = s.N;
        std::vector<sycl::half> x(16ull * K);
        for (auto& v : x) v = sycl::half(float(r() % 2000) / 1000.f - 1.f);
        auto* dx = sycl::malloc_device<sycl::half>(x.size(), q);
        void* dq8 = sycl::malloc_device(16ull * (K / 32) * sizeof(block_q8_1x), q);
        auto* y = sycl::malloc_device<sycl::half>(16ull * N, q);
        q.memcpy(dx, x.data(), x.size() * 2).wait();
        quantize_q8_1(q, dx, dq8, 16 * K).wait();
        std::string line;
        char buf[160];
        for (int ty = 0; ty < 2; ++ty) {
            const bool q6 = ty == 0;
            const uint64_t nblk = uint64_t(N) * (K / 256);
            std::vector<uint8_t> raw;
            if (q6) { std::vector<block_q6_K> b(nblk); rand_q6(r, b); raw.assign((uint8_t*)b.data(), (uint8_t*)(b.data() + nblk)); }
            else    { std::vector<block_q5_K> b(nblk); rand_q5(r, b); raw.assign((uint8_t*)b.data(), (uint8_t*)(b.data() + nblk)); }
            const Streams st = repack(q6, raw.data(), K, N);
            Dev dv(q, st);
            const double bytes = double(st.lo.size() + st.hi.size() + st.sc.size() + st.d.size());
            const double t1 = time_it([&] { dv.gemv(q6, dq8, y, K, N, 1); });
            const double t2 = time_it([&] { dv.gemv(q6, dq8, y, K, N, 2); });
            const double t4 = time_it([&] { dv.gemv(q6, dq8, y, K, N, 4); });
            const double t8 = time_it([&] { dv.gemv(q6, dq8, y, K, N, 8); });
            const double t16 = time_it([&] { dv.gemv(q6, dq8, y, K, N, 16); });
            std::snprintf(buf, sizeof buf, " %s c32 %.0f (%.0f) T2 %.0f T4 %.0f T8 %.0f T16 %.0f |", q6 ? "Q6" : "Q5",
                          bytes / t1 / 1e9, t1 * 1e6, t2 * 1e6, t4 * 1e6, t8 * 1e6, t16 * 1e6);
            line += buf;
            if (q6) {   // the old SoA-Q6 v1 kernel (the dense 1-card path) on the same weights
                std::vector<uint8_t> lo(uint64_t(N) * K / 2, 0), hi(uint64_t(N) * K / 4, 0);
                std::vector<int8_t> sc(uint64_t(N) * K / 16);
                std::vector<uint16_t> d(uint64_t(N) * K / 256);
                repack_q6_K_to_soa(raw.data(), K, N, lo.data(), hi.data(), sc.data(), d.data());
                auto* a = sycl::malloc_device<uint8_t>(lo.size(), q); auto* b = sycl::malloc_device<uint8_t>(hi.size(), q);
                auto* c = sycl::malloc_device<int8_t>(sc.size(), q);  auto* e = sycl::malloc_device<uint16_t>(d.size(), q);
                q.memcpy(a, lo.data(), lo.size()); q.memcpy(b, hi.data(), hi.size());
                q.memcpy(c, sc.data(), sc.size()); q.memcpy(e, d.data(), d.size() * 2).wait();
                const double to = time_it([&] { gemv_q6_soa_q8(q, dq8, a, b, c, e, y, K, N); });
                std::snprintf(buf, sizeof buf, " Q6 soa-v1 %.0f (%.0f) |", bytes / to / 1e9, to * 1e6);
                line += buf;
                sycl::free(a, q); sycl::free(b, q); sycl::free(c, q); sycl::free(e, q);
            }
        }
        {   // Q8_0-SoA (today's split kernel) on random int8
            auto* qs = sycl::malloc_device<int8_t>(uint64_t(N) * K, q);
            auto* dd = sycl::malloc_device<uint16_t>(uint64_t(N) * K / 32, q);
            q.memset(qs, 3, uint64_t(N) * K); q.fill(dd, uint16_t(0x1400), uint64_t(N) * K / 32).wait();
            const double bytes = double(N) * K * (1 + 2.0 / 32);
            const double t1 = time_it([&] { gemv_q8_0_soa_q8(q, dq8, qs, dd, y, K, N); });
            const double t2 = time_it([&] { gemv_q8_0_soa_q8_batched(q, dq8, qs, dd, y, K, N, 2); });
            const double t4 = time_it([&] { gemv_q8_0_soa_q8_batched(q, dq8, qs, dd, y, K, N, 4); });
            const double t8 = time_it([&] { gemv_q8_0_soa_q8_batched(q, dq8, qs, dd, y, K, N, 8); });
            const double t16 = time_it([&] { gemv_q8_0_soa_q8_batched(q, dq8, qs, dd, y, K, N, 16); });
            std::snprintf(buf, sizeof buf, " Q8 soa %.0f (%.0f) T2 %.0f T4 %.0f T8 %.0f T16 %.0f", bytes / t1 / 1e9, t1 * 1e6,
                          t2 * 1e6, t4 * 1e6, t8 * 1e6, t16 * 1e6);
            line += buf;
            sycl::free(qs, q); sycl::free(dd, q);
        }
        std::printf("%-20s %8u %8u |%s\n", s.name, K, N, line.c_str());
        sycl::free(dx, q); sycl::free(dq8, q); sycl::free(y, q);
    }
}
}  // namespace

int main(int argc, char** argv) {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};   // like the engine's queues
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    if (argc > 1 && std::string(argv[1]) == "--bench") { bench(q); return 0; }
    // the 27B's K values (5120, 6144, 17408) and small N so the double reference stays quick; N not a multiple of 16
    run_type(q, true, 5120, 67, 11);
    run_type(q, false, 5120, 67, 12);
    run_type(q, true, 17408, 33, 13);
    run_type(q, false, 17408, 33, 14);
    run_type(q, true, 6144, 48, 15);
    run_type(q, false, 6144, 48, 16);
    test_embed_q5(q);
    test_moe(q, true, true, 21);
    test_moe(q, false, true, 22);
    test_moe(q, false, false, 23);
    std::printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
