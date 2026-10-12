// tools/mimo26_fp8_bench.cpp -- B66: the FP8 dense GEMV (mimo26_gemv_fp8, src/ops/mimo26_ops.cpp) and candidate variants
// at MiMo's qkv shapes, timed with profiling events and compared bit for bit with the production kernel.
//
//   ie-mimo26-fp8-bench [--reps N]
//
// Variants (template parameters of one kernel, so the per-row arithmetic is written once):
//   NR  output rows per sub-group (1 = production): the activation loads are shared by the NR rows
//   LUT E4M3 decoded through the 256-entry fp16 table in SLM (production) or by bits: the E4M3 magnitude placed in an
//       fp16's exponent/mantissa field is the value times 2^-8 (normals and subnormals), times 256 exactly; 0x7F/0xFF
//       (the NaN code) is NaN -- the table's values, so the products are the same
// Each variant's output must equal the production kernel's bit for bit; the timing is the median of --reps launches.
#include "ie/deepseek41_upload.hpp"
#include "ie/mimo26_ops.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

inline float e4m3_bits(uint32_t v) {
    const uint16_t h = uint16_t(((v & 0x80u) << 8) | ((v & 0x7Fu) << 7));
    const float f = float(sycl::bit_cast<sycl::half>(h)) * 256.f;
    return (v & 0x7Fu) == 0x7Fu ? sycl::bit_cast<float>(0x7FC00000u | ((v & 0x80u) << 24)) : f;
}

template <int M, int NC, int NR, bool LUT>
sycl::event gemv(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const uint8_t* w, const float* srow, float* y,
                 uint32_t K, uint32_t N) {
    constexpr int SG_SIZE = 16, WG_ITEMS = NC * SG_SIZE;
    const uint32_t blocks = K / 32, SK = K / 128;
    const uint32_t rows_per_wg = NC * NR, n_wgs = (N + rows_per_wg - 1) / rows_per_wg;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> lut16(sycl::range<1>(256u), h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), wgid = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE, n0 = (wgid * NC + sg_id) * NR;
            if constexpr (LUT) {
                for (uint32_t i = lid; i < 256u; i += WG_ITEMS) lut16[i] = sycl::half(ie::ds41_e4m3(uint8_t(i)));
                sycl::group_barrier(it.get_group());
            }
            if (n0 >= N) return;
            auto dec = [&](uint32_t v) { if constexpr (LUT) return float(lut16[v]); else return e4m3_bits(v); };
            float acc[NR][M];
            #pragma unroll
            for (int r = 0; r < NR; ++r)
                #pragma unroll
                for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
            for (uint32_t b = lane; b < blocks; b += SG_SIZE) {
                sycl::vec<sycl::half, 8> xq[M][4];
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const auto* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(x + uint64_t(m) * x_stride + uint64_t(b) * 32);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) xq[m][j] = xv[j];
                }
                #pragma unroll
                for (int r = 0; r < NR; ++r) {
                    const uint32_t n = n0 + uint32_t(r);
                    if (n >= N) break;
                    const auto* w128 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(w + uint64_t(n) * K + uint64_t(b) * 32);
                    const sycl::vec<uint32_t, 4> wa = w128[0], wb = w128[1];
                    float bacc[M];
                    #pragma unroll
                    for (int m = 0; m < M; ++m) bacc[m] = 0.f;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t v = wa[j], u = wb[j];
                        const float d0 = dec(v & 0xFFu), d1 = dec((v >> 8) & 0xFFu), d2 = dec((v >> 16) & 0xFFu), d3 = dec(v >> 24);
                        const float e0 = dec(u & 0xFFu), e1 = dec((u >> 8) & 0xFFu), e2 = dec((u >> 16) & 0xFFu), e3 = dec(u >> 24);
                        #pragma unroll
                        for (int m = 0; m < M; ++m) {
                            bacc[m] += d0 * float(xq[m][(4 * j + 0) >> 3][(4 * j + 0) & 7]);
                            bacc[m] += d1 * float(xq[m][(4 * j + 1) >> 3][(4 * j + 1) & 7]);
                            bacc[m] += d2 * float(xq[m][(4 * j + 2) >> 3][(4 * j + 2) & 7]);
                            bacc[m] += d3 * float(xq[m][(4 * j + 3) >> 3][(4 * j + 3) & 7]);
                            bacc[m] += e0 * float(xq[m][(16 + 4 * j + 0) >> 3][(16 + 4 * j + 0) & 7]);
                            bacc[m] += e1 * float(xq[m][(16 + 4 * j + 1) >> 3][(16 + 4 * j + 1) & 7]);
                            bacc[m] += e2 * float(xq[m][(16 + 4 * j + 2) >> 3][(16 + 4 * j + 2) & 7]);
                            bacc[m] += e3 * float(xq[m][(16 + 4 * j + 3) >> 3][(16 + 4 * j + 3) & 7]);
                        }
                    }
                    const float sc = srow[uint64_t(n) * SK + (b >> 2)];
                    #pragma unroll
                    for (int m = 0; m < M; ++m) acc[r][m] += sc * bacc[m];
                }
            }
            #pragma unroll
            for (int r = 0; r < NR; ++r) {
                const uint32_t n = n0 + uint32_t(r);
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const float red = sycl::reduce_over_group(it.get_sub_group(), acc[r][m], sycl::plus<float>());
                    if (lane == 0 && n < N) y[uint64_t(m) * N + n] = red;
                }
            }
        });
    });
}

// Weights-first: the block's 32 weights decoded once (by bits), then each activation row in turn -- per row the same
// products in the same order as the production kernel (k = 4j..4j+3, then 16+4j..16+4j+3, j = 0..3), into its own
// block sum. XF: the activations as fp32 (each half's exact value), else fp16 converted at use.
template <int M, int NC, bool XF>
sycl::event gemv_wf(sycl::queue& q, const void* xv_, uint32_t x_stride, const uint8_t* w, const float* srow, float* y,
                    uint32_t K, uint32_t N) {
    constexpr int SG_SIZE = 16, WG_ITEMS = NC * SG_SIZE;
    const uint32_t blocks = K / 32, SK = K / 128, n_wgs = (N + NC - 1) / NC;
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), wgid = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE, n = wgid * NC + sg_id;
            if (n >= N) return;
            float acc[M];
            #pragma unroll
            for (int m = 0; m < M; ++m) acc[m] = 0.f;
            for (uint32_t b = lane; b < blocks; b += SG_SIZE) {
                const auto* w128 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(w + uint64_t(n) * K + uint64_t(b) * 32);
                const sycl::vec<uint32_t, 4> wa = w128[0], wb = w128[1];
                float dw[32];   // dw[k] = the weight of column k of the block
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t v = wa[j], u = wb[j];
                    dw[4 * j + 0] = e4m3_bits(v & 0xFFu); dw[4 * j + 1] = e4m3_bits((v >> 8) & 0xFFu);
                    dw[4 * j + 2] = e4m3_bits((v >> 16) & 0xFFu); dw[4 * j + 3] = e4m3_bits(v >> 24);
                    dw[16 + 4 * j + 0] = e4m3_bits(u & 0xFFu); dw[16 + 4 * j + 1] = e4m3_bits((u >> 8) & 0xFFu);
                    dw[16 + 4 * j + 2] = e4m3_bits((u >> 16) & 0xFFu); dw[16 + 4 * j + 3] = e4m3_bits(u >> 24);
                }
                const float sc = srow[uint64_t(n) * SK + (b >> 2)];
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    float xm[32];
                    if constexpr (XF) {
                        const auto* xv = reinterpret_cast<const sycl::vec<float, 8>*>(static_cast<const float*>(xv_) + uint64_t(m) * x_stride + uint64_t(b) * 32);
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) { const sycl::vec<float, 8> t = xv[i];
                            #pragma unroll
                            for (int e = 0; e < 8; ++e) xm[8 * i + e] = t[e]; }
                    } else {
                        const auto* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(static_cast<const sycl::half*>(xv_) + uint64_t(m) * x_stride + uint64_t(b) * 32);
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) { const sycl::vec<sycl::half, 8> t = xv[i];
                            #pragma unroll
                            for (int e = 0; e < 8; ++e) xm[8 * i + e] = float(t[e]); }
                    }
                    float bacc = 0.f;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        bacc += dw[4 * j + 0] * xm[4 * j + 0];
                        bacc += dw[4 * j + 1] * xm[4 * j + 1];
                        bacc += dw[4 * j + 2] * xm[4 * j + 2];
                        bacc += dw[4 * j + 3] * xm[4 * j + 3];
                        bacc += dw[16 + 4 * j + 0] * xm[16 + 4 * j + 0];
                        bacc += dw[16 + 4 * j + 1] * xm[16 + 4 * j + 1];
                        bacc += dw[16 + 4 * j + 2] * xm[16 + 4 * j + 2];
                        bacc += dw[16 + 4 * j + 3] * xm[16 + 4 * j + 3];
                    }
                    acc[m] += sc * bacc;
                }
            }
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const float red = sycl::reduce_over_group(it.get_sub_group(), acc[m], sycl::plus<float>());
                if (lane == 0) y[uint64_t(m) * N + n] = red;
            }
        });
    });
}

// Weights-first with NR output rows per sub-group: each converted activation block feeds NR rows' products (per row the
// production order). Registers: NR x 32 weights + 32 activations + M x NR sums.
template <int M, int NC, int NR>
sycl::event gemv_wfr(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const uint8_t* w, const float* srow, float* y,
                     uint32_t K, uint32_t N) {
    constexpr int SG_SIZE = 16, WG_ITEMS = NC * SG_SIZE;
    const uint32_t blocks = K / 32, SK = K / 128, rows_per_wg = NC * NR, n_wgs = (N + rows_per_wg - 1) / rows_per_wg;
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0)), wgid = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE, lane = lid % SG_SIZE, n0 = (wgid * NC + sg_id) * NR;
            if (n0 >= N) return;
            float acc[NR][M];
            #pragma unroll
            for (int r = 0; r < NR; ++r)
                #pragma unroll
                for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
            for (uint32_t b = lane; b < blocks; b += SG_SIZE) {
                float dw[NR][32], sc[NR];
                #pragma unroll
                for (int r = 0; r < NR; ++r) {
                    const uint32_t n = std::min(n0 + uint32_t(r), N - 1);   // a tail row repeats the last real row (not stored)
                    const auto* w128 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(w + uint64_t(n) * K + uint64_t(b) * 32);
                    const sycl::vec<uint32_t, 4> wa = w128[0], wb = w128[1];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t v = wa[j], u = wb[j];
                        dw[r][4 * j + 0] = e4m3_bits(v & 0xFFu); dw[r][4 * j + 1] = e4m3_bits((v >> 8) & 0xFFu);
                        dw[r][4 * j + 2] = e4m3_bits((v >> 16) & 0xFFu); dw[r][4 * j + 3] = e4m3_bits(v >> 24);
                        dw[r][16 + 4 * j + 0] = e4m3_bits(u & 0xFFu); dw[r][16 + 4 * j + 1] = e4m3_bits((u >> 8) & 0xFFu);
                        dw[r][16 + 4 * j + 2] = e4m3_bits((u >> 16) & 0xFFu); dw[r][16 + 4 * j + 3] = e4m3_bits(u >> 24);
                    }
                    sc[r] = srow[uint64_t(n) * SK + (b >> 2)];
                }
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const auto* xv = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(x + uint64_t(m) * x_stride + uint64_t(b) * 32);
                    float xm[32];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) { const sycl::vec<sycl::half, 8> t = xv[i];
                        #pragma unroll
                        for (int e = 0; e < 8; ++e) xm[8 * i + e] = float(t[e]); }
                    #pragma unroll
                    for (int r = 0; r < NR; ++r) {
                        float bacc = 0.f;
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            bacc += dw[r][4 * j + 0] * xm[4 * j + 0];
                            bacc += dw[r][4 * j + 1] * xm[4 * j + 1];
                            bacc += dw[r][4 * j + 2] * xm[4 * j + 2];
                            bacc += dw[r][4 * j + 3] * xm[4 * j + 3];
                            bacc += dw[r][16 + 4 * j + 0] * xm[16 + 4 * j + 0];
                            bacc += dw[r][16 + 4 * j + 1] * xm[16 + 4 * j + 1];
                            bacc += dw[r][16 + 4 * j + 2] * xm[16 + 4 * j + 2];
                            bacc += dw[r][16 + 4 * j + 3] * xm[16 + 4 * j + 3];
                        }
                        acc[r][m] += sc[r] * bacc;
                    }
                }
            }
            #pragma unroll
            for (int r = 0; r < NR; ++r)
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const float red = sycl::reduce_over_group(it.get_sub_group(), acc[r][m], sycl::plus<float>());
                    if (lane == 0 && n0 + uint32_t(r) < N) y[uint64_t(m) * N + n0 + r] = red;
                }
        });
    });
}

double ms_of(const sycl::event& e) {
    return double(e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                  e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
}

template <typename F>
double median_ms(sycl::queue& q, int reps, F&& launch) {
    std::vector<double> t;
    for (int i = 0; i < reps + 3; ++i) { sycl::event e = launch(); e.wait(); if (i >= 3) t.push_back(ms_of(e)); }
    std::sort(t.begin(), t.end()); return t[t.size() / 2];
}

template <int M>
int run_shape(sycl::queue& q, uint32_t K, uint32_t N, int reps) {
    std::mt19937 rng(1234 + M + N);
    std::vector<uint8_t> hw(size_t(N) * K);
    for (auto& b : hw) { b = uint8_t(rng() & 0xFF); if ((b & 0x7F) == 0x7F) b ^= 1; }   // real weights carry no NaN code
    std::vector<float> hs(size_t(N) * (K / 128)); for (auto& s : hs) s = std::ldexp(1.f + float(rng() % 1000) / 1000.f, int(rng() % 8) - 10);
    std::vector<sycl::half> hx(size_t(M) * K); for (auto& v : hx) v = sycl::half(float(int(rng() % 4001) - 2000) / 1000.f);
    auto* dw = sycl::malloc_device<uint8_t>(hw.size(), q); auto* ds = sycl::malloc_device<float>(hs.size(), q);
    auto* dx = sycl::malloc_device<sycl::half>(hx.size(), q);
    std::vector<float> hxf(hx.size()); for (size_t i = 0; i < hx.size(); ++i) hxf[i] = float(hx[i]);
    auto* dxf = sycl::malloc_device<float>(hxf.size(), q);
    q.memcpy(dxf, hxf.data(), hxf.size() * 4).wait();
    auto* yref = sycl::malloc_device<float>(size_t(M) * N, q); auto* ytry = sycl::malloc_device<float>(size_t(M) * N, q);
    q.memcpy(dw, hw.data(), hw.size()).wait(); q.memcpy(ds, hs.data(), hs.size() * 4).wait(); q.memcpy(dx, hx.data(), hx.size() * 2).wait();
    std::vector<float> ref(size_t(M) * N), got(size_t(M) * N);
    // the production kernel through its public entry, then its in-bench copy (NR 1, LUT): the copy must equal it
    ie::mimo26_gemv_fp8(q, dx, K, dw, ds, yref, M, K, N).wait();
    q.memcpy(ref.data(), yref, ref.size() * 4).wait();
    const double gb = double(N) * K / 1e9;
    const double t_prod = median_ms(q, reps, [&] { return ie::mimo26_gemv_fp8(q, dx, K, dw, ds, yref, M, K, N); });
    std::printf("M %d K %u N %u: production %.4f ms (%.0f GB/s)\n", M, K, N, t_prod, gb / (t_prod * 1e-3));
    int bad = 0;
    auto variant = [&](const char* name, auto launch) {
        q.memset(ytry, 0xFF, size_t(M) * N * 4).wait();
        launch().wait();
        q.memcpy(got.data(), ytry, got.size() * 4).wait();
        const bool same = std::memcmp(got.data(), ref.data(), got.size() * 4) == 0;
        const double t = median_ms(q, reps, launch);
        std::printf("  %-12s %.4f ms (%.0f GB/s, x%.2f)  bits %s\n", name, t, gb / (t * 1e-3), t_prod / t, same ? "IDENTICAL" : "DIFFER");
        bad += !same;
    };
    constexpr int NC = 16;
    variant("NR1 LUT", [&] { return gemv<M, NC, 1, true>(q, dx, K, dw, ds, ytry, K, N); });
    variant("NR1 bits", [&] { return gemv<M, NC, 1, false>(q, dx, K, dw, ds, ytry, K, N); });
    variant("WF half", [&] { return gemv_wf<M, NC, false>(q, dx, K, dw, ds, ytry, K, N); });
    variant("WF2", [&] { return gemv_wfr<M, NC, 2>(q, dx, K, dw, ds, ytry, K, N); });
    variant("WF2 NC8", [&] { return gemv_wfr<M, 8, 2>(q, dx, K, dw, ds, ytry, K, N); });
    variant("WF4 NC8", [&] { return gemv_wfr<M, 8, 4>(q, dx, K, dw, ds, ytry, K, N); });
    for (auto* p : {static_cast<void*>(dw), static_cast<void*>(ds), static_cast<void*>(dx), static_cast<void*>(dxf), static_cast<void*>(yref), static_cast<void*>(ytry)}) sycl::free(p, q);
    return bad;
}

// --checksums: FNV-1a of the decode-shape outputs of the FP8 GEMV (M 1..8, qkv N), the router logits (T 1..8) and the
// split-K decode attention (T 1..8 at positions 37, 6001, 30001; MiMo's SWA layer with sinks and the ring, and its full
// layer with V padded to hd), each
// at fixed random inputs. Run once per setting of a kernel switch (IE_MIMO26_FP8_LUT): equal lines = bit-identical outputs.
uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

int checksums(sycl::queue& q) {
    std::mt19937 rng(4242);
    auto rnd_h = [&](size_t n, float amp) { std::vector<sycl::half> v(n); for (auto& x : v) x = sycl::half(amp * float(int(rng() % 20001) - 10000) / 10000.f); return v; };
    // FP8 GEMV
    { const uint32_t K = 4096;
      for (const uint32_t N : {14848u, 13568u}) {
          std::vector<uint8_t> hw(size_t(N) * K); for (auto& b : hw) { b = uint8_t(rng() & 0xFF); if ((b & 0x7F) == 0x7F) b ^= 1; }
          std::vector<float> hs(size_t(N) * (K / 128)); for (auto& v : hs) v = std::ldexp(1.f + float(rng() % 1000) / 1000.f, int(rng() % 8) - 10);
          auto hx = rnd_h(size_t(8) * K, 2.f);
          auto* dw = sycl::malloc_device<uint8_t>(hw.size(), q); auto* ds = sycl::malloc_device<float>(hs.size(), q);
          auto* dx = sycl::malloc_device<sycl::half>(hx.size(), q); auto* dy = sycl::malloc_device<float>(size_t(8) * N, q);
          q.memcpy(dw, hw.data(), hw.size()).wait(); q.memcpy(ds, hs.data(), hs.size() * 4).wait(); q.memcpy(dx, hx.data(), hx.size() * 2).wait();
          std::vector<float> y(size_t(8) * N);
          for (uint32_t M = 1; M <= 8; ++M) {
              ie::mimo26_gemv_fp8(q, dx, K, dw, ds, dy, M, K, N).wait();
              q.memcpy(y.data(), dy, size_t(M) * N * 4).wait();
              std::printf("fp8 N %u M %u: %016llx\n", N, M, (unsigned long long)fnv(y.data(), size_t(M) * N * 4));
          }
          for (auto* p : {static_cast<void*>(dw), static_cast<void*>(ds), static_cast<void*>(dx), static_cast<void*>(dy)}) sycl::free(p, q);
      } }
    // router
    { constexpr uint32_t H = 4096, E = 256;
      std::vector<float> hx(8 * H), hw(size_t(E) * H);
      for (auto& v : hx) v = float(int(rng() % 4001) - 2000) / 997.f;
      for (auto& v : hw) v = float(int(rng() % 4001) - 2000) / 50021.f;
      auto* dx = sycl::malloc_device<float>(hx.size(), q); auto* dw = sycl::malloc_device<float>(hw.size(), q); auto* dl = sycl::malloc_device<float>(8 * E, q);
      q.memcpy(dx, hx.data(), hx.size() * 4).wait(); q.memcpy(dw, hw.data(), hw.size() * 4).wait();
      std::vector<float> l(8 * E);
      for (uint32_t T = 1; T <= 8; ++T) {
          ie::mimo26_router_logits(q, dx, dw, dl, T, H, E).wait();
          q.memcpy(l.data(), dl, size_t(T) * E * 4).wait();
          std::printf("router T %u: %016llx\n", T, (unsigned long long)fnv(l.data(), size_t(T) * E * 4));
      }
      for (auto* p : {dx, dw, dl}) sycl::free(p, q); }
    // decode attention: SWA (n_kv 8, window 128, sinks, ring 2240) and full (n_kv 4, V padded to 192); cap 32768
    { constexpr uint32_t n_q = 64, hd = 192, hdv = 128, cap = 32768;
      auto* partials = sycl::malloc_device<float>(size_t(8) * n_q * ie::mimo26_decode_max_splits() * (hdv + 2), q);
      auto* sinks = sycl::malloc_device<float>(n_q, q);
      { std::vector<float> hs(n_q); for (auto& v : hs) v = float(int(rng() % 2001) - 1000) / 400.f; q.memcpy(sinks, hs.data(), n_q * 4).wait(); }
      for (const bool swa : {true, false}) {
          const uint32_t n_kv = swa ? 8 : 4, window = swa ? 128 : 0, ring = swa ? 2240 : 0, vst = swa ? hdv : hd, slots = ring ? ring : cap;
          auto hk = rnd_h(size_t(n_kv) * slots * hd, 1.f), hv = rnd_h(size_t(n_kv) * slots * vst, 1.f);
          auto* kc = sycl::malloc_device<sycl::half>(hk.size(), q); auto* vc = sycl::malloc_device<sycl::half>(hv.size(), q);
          auto hq = rnd_h(size_t(8) * n_q * hd, 1.f), hkn = rnd_h(size_t(8) * n_kv * hd, 1.f), hvn = rnd_h(size_t(8) * n_kv * vst, 1.f);
          auto* dQ = sycl::malloc_device<sycl::half>(hq.size(), q); auto* dK = sycl::malloc_device<sycl::half>(hkn.size(), q);
          auto* dV = sycl::malloc_device<sycl::half>(hvn.size(), q); auto* dy = sycl::malloc_device<sycl::half>(size_t(8) * n_q * hdv, q);
          q.memcpy(dQ, hq.data(), hq.size() * 2).wait(); q.memcpy(dK, hkn.data(), hkn.size() * 2).wait(); q.memcpy(dV, hvn.data(), hvn.size() * 2).wait();
          std::vector<sycl::half> y(size_t(8) * n_q * hdv);
          for (const uint32_t pos0 : {37u, 6001u, 30001u}) for (uint32_t T = 1; T <= 8; ++T) {
              q.memcpy(kc, hk.data(), hk.size() * 2).wait(); q.memcpy(vc, hv.data(), hv.size() * 2).wait();   // the same caches every call
              ie::mimo26_attention_decode(q, dQ, dK, dV, kc, vc, dy, partials, T, pos0, n_q, n_kv, hd, hdv, cap, window,
                                          swa ? sinks : nullptr, {}, ring, swa ? 0u : hd).wait();
              q.memcpy(y.data(), dy, size_t(T) * n_q * hdv * 2).wait();
              std::printf("attn %s pos0 %u T %u: %016llx\n", swa ? "swa" : "full", pos0, T, (unsigned long long)fnv(y.data(), size_t(T) * n_q * hdv * 2));
          }
          for (auto* p : {kc, vc, dQ, dK, dV, dy}) sycl::free(p, q);
      }
      sycl::free(partials, q); sycl::free(sinks, q); }
    return 0;
}

// --attn-time: the decode attention's time (median of --reps) at MiMo's full layer (n_kv 4, V padded) and SWA layer, at
// positions 6001 and 30001, T 1 and 8 -- where a long context's decode time goes
int attn_time(sycl::queue& q, int reps) {
    std::mt19937 rng(77);
    constexpr uint32_t n_q = 64, hd = 192, hdv = 128, cap = 32768;
    auto rnd_h = [&](size_t n) { std::vector<sycl::half> v(n); for (auto& x : v) x = sycl::half(float(int(rng() % 20001) - 10000) / 10000.f); return v; };
    auto* partials = sycl::malloc_device<float>(size_t(8) * n_q * ie::mimo26_decode_max_splits() * (hdv + 2), q);
    auto* sinks = sycl::malloc_device<float>(n_q, q); q.memset(sinks, 0, n_q * 4).wait();
    for (const bool swa : {false, true}) {
        const uint32_t n_kv = swa ? 8 : 4, window = swa ? 128 : 0, ring = swa ? 2240 : 0, vst = swa ? hdv : hd, slots = ring ? ring : cap;
        auto hk = rnd_h(size_t(n_kv) * slots * hd), hv = rnd_h(size_t(n_kv) * slots * vst);
        auto* kc = sycl::malloc_device<sycl::half>(hk.size(), q); auto* vc = sycl::malloc_device<sycl::half>(hv.size(), q);
        q.memcpy(kc, hk.data(), hk.size() * 2).wait(); q.memcpy(vc, hv.data(), hv.size() * 2).wait();
        auto hq = rnd_h(size_t(8) * n_q * hd), hkn = rnd_h(size_t(8) * n_kv * hd), hvn = rnd_h(size_t(8) * n_kv * vst);
        auto* dQ = sycl::malloc_device<sycl::half>(hq.size(), q); auto* dK = sycl::malloc_device<sycl::half>(hkn.size(), q);
        auto* dV = sycl::malloc_device<sycl::half>(hvn.size(), q); auto* dy = sycl::malloc_device<sycl::half>(size_t(8) * n_q * hdv, q);
        q.memcpy(dQ, hq.data(), hq.size() * 2).wait(); q.memcpy(dK, hkn.data(), hkn.size() * 2).wait(); q.memcpy(dV, hvn.data(), hvn.size() * 2).wait();
        for (const uint32_t pos0 : {6001u, 30001u}) for (const uint32_t T : {1u, 8u}) {
            std::vector<double> t;
            for (int i = 0; i < reps + 3; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                ie::mimo26_attention_decode(q, dQ, dK, dV, kc, vc, dy, partials, T, pos0, n_q, n_kv, hd, hdv, cap, window,
                                            swa ? sinks : nullptr, {}, ring, swa ? 0u : hd).wait();
                if (i >= 3) t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(t.begin(), t.end());
            std::printf("attn %s pos0 %u T %u: %.3f ms (append + split pass + merge, host wall)\n", swa ? "swa " : "full", pos0, T, t[t.size() / 2]);
        }
        for (auto* p : {kc, vc, dQ, dK, dV, dy}) sycl::free(p, q);
    }
    sycl::free(partials, q); sycl::free(sinks, q);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 50; bool sums = false;
    for (int i = 1; i < argc; ++i) { if (std::string(argv[i]) == "--reps" && i + 1 < argc) reps = std::atoi(argv[i + 1]); if (std::string(argv[i]) == "--checksums") sums = true; }
    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    if (sums) return checksums(q);
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--attn-time") return attn_time(q, reps);
    int bad = 0;
    // MiMo's qkv rows: SWA layers 64 x 192 + 8 x 192 + 8 x 128, full layers 64 x 192 + 4 x 192 + 4 x 128; K = 4096
    for (const uint32_t N : {14848u, 13568u}) {
        bad += run_shape<1>(q, 4096, N, reps);
        bad += run_shape<4>(q, 4096, N, reps);
        bad += run_shape<8>(q, 4096, N, reps);
    }
    std::printf("%s\n", bad ? "SOME VARIANTS DIFFER" : "all variants bit-identical to production");
    return 0;
}
