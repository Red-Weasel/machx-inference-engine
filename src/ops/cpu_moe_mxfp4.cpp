// src/ops/cpu_moe_mxfp4.cpp — see include/ie/cpu_moe_mxfp4.hpp. Compiled with -mavx2 -mfma
// -fopenmp (src/CMakeLists.txt); the scalar reference needs none of them.
#include "ie/cpu_moe_mxfp4.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <omp.h>

namespace ie {

int cpu_mxfp4_nibble_int(uint32_t nb) {
    const int mag = int((0xC8643210u >> ((nb & 7u) * 4u)) & 0xFu);   // {0,1,2,3,4,6,8,12}
    return (nb & 8u) ? -mag : mag;
}

float cpu_mxfp4_e8m0_half(uint8_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : (uint32_t(e - 1u) << 23);
    float f; std::memcpy(&f, &bits, 4); return f;
}

void cpu_dequant_mxfp4_ref(const uint8_t* qs, const uint8_t* e, float* w, uint32_t K, uint32_t N) {
    const uint32_t nb = K / 32;
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t b = 0; b < nb; ++b) {
            const uint8_t* blk = qs + (size_t(n) * nb + b) * 16;
            const float s = cpu_mxfp4_e8m0_half(e[size_t(n) * nb + b]);
            for (uint32_t j = 0; j < 16; ++j) {
                w[size_t(n) * K + b * 32 + j]      = float(cpu_mxfp4_nibble_int(blk[j] & 0xFu)) * s;
                w[size_t(n) * K + b * 32 + 16 + j] = float(cpu_mxfp4_nibble_int(blk[j] >> 4)) * s;
            }
        }
}

namespace {
// B66: software prefetch, 2 KB (one MiMo weight row) past the block being read -- streamed experts -7 % on 12 E-cores
// (cpu_moe_mxfp4_test --bench: 0.229 -> 0.213 ms); addresses only, no arithmetic. IE_CPU_MXFP4_PF=<bytes>, 0 = none.
int pf_dist() { static const int d = [] { const char* v = std::getenv("IE_CPU_MXFP4_PF"); return v ? std::max(0, std::atoi(v)) : 2048; }(); return d; }

#if defined(__AVX2__)
// The block's 32 weights as floats, w[0..3] = k 0..7, 8..15, 16..23, 24..31 (the 16 bytes hold k = j in the low nibble,
// k = 16 + j in the high one). Two forms of the same values:
//   F16 = false (v0.2.21): the signed int8 table {0,1,2,3,4,6,8,12,0,-1,...,-12} by pshufb, sign-extended to int32, to float
//   F16 = true (B66): the same magnitudes' fp16 high bytes by pshufb (every low byte is 0), widened to fp16 words and
//     converted by vcvtph2ps -- exact for all 16 codes; code 8 (int 0) is +0, as the int route's 0 becomes +0.0f
template <bool F16>
inline void block_weights(__m128i raw, __m256 w[4]) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i ilo = _mm_and_si128(raw, m4), ihi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
    if constexpr (F16) {
        const __m128i th = _mm_setr_epi8(0x00, 0x3C, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4A,
                                         0x00, char(0xBC), char(0xC0), char(0xC2), char(0xC4), char(0xC6), char(0xC8), char(0xCA));
        const __m128i z = _mm_setzero_si128();
        const __m128i hl = _mm_shuffle_epi8(th, ilo), hh = _mm_shuffle_epi8(th, ihi);
        w[0] = _mm256_cvtph_ps(_mm_unpacklo_epi8(z, hl)); w[1] = _mm256_cvtph_ps(_mm_unpackhi_epi8(z, hl));
        w[2] = _mm256_cvtph_ps(_mm_unpacklo_epi8(z, hh)); w[3] = _mm256_cvtph_ps(_mm_unpackhi_epi8(z, hh));
    } else {
        // the 16-entry signed table applied with pshufb to 16 nibbles at once; the block's 32 weights are integers
        // times one scale, so the scale multiplies the block's dot once
        const __m128i tbl = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
        const __m128i lo = _mm_shuffle_epi8(tbl, ilo), hi = _mm_shuffle_epi8(tbl, ihi);
        w[0] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)); w[1] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
        w[2] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)); w[3] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
    }
}
#endif

// one row's dot product over its nb blocks
template <bool F16>
inline float row_dot(const float* x, const uint8_t* qs_row, const uint8_t* e_row, uint32_t nb, int pf = 0) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    if (pf) _mm_prefetch(reinterpret_cast<const char*>(e_row) + pf / 16, _MM_HINT_T0);
    for (uint32_t b = 0; b < nb; ++b) {
        if (pf && (b & 3) == 0) _mm_prefetch(reinterpret_cast<const char*>(qs_row) + size_t(b) * 16 + pf, _MM_HINT_T0);
        __m256 w[4]; block_weights<F16>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(qs_row + size_t(b) * 16)), w);
        const float* xb = x + size_t(b) * 32;
        __m256 d = _mm256_setzero_ps();
        d = _mm256_fmadd_ps(w[0], _mm256_loadu_ps(xb), d);
        d = _mm256_fmadd_ps(w[1], _mm256_loadu_ps(xb + 8), d);
        d = _mm256_fmadd_ps(w[2], _mm256_loadu_ps(xb + 16), d);
        d = _mm256_fmadd_ps(w[3], _mm256_loadu_ps(xb + 24), d);
        acc = _mm256_fmadd_ps(d, _mm256_set1_ps(cpu_mxfp4_e8m0_half(e_row[b])), acc);
    }
    alignas(32) float tmp[8]; _mm256_store_ps(tmp, acc);
    return ((tmp[0] + tmp[4]) + (tmp[1] + tmp[5])) + ((tmp[2] + tmp[6]) + (tmp[3] + tmp[7]));
#else
    float acc = 0.f;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = qs_row + size_t(b) * 16; const float* xb = x + size_t(b) * 32;
        float d = 0.f;
        for (uint32_t j = 0; j < 16; ++j) { d += float(cpu_mxfp4_nibble_int(blk[j] & 0xFu)) * xb[j]; d += float(cpu_mxfp4_nibble_int(blk[j] >> 4)) * xb[16 + j]; }
        acc += d * cpu_mxfp4_e8m0_half(e_row[b]);
    }
    return acc;
#endif
}

// R dot products of one weight row with R activation vectors: per row exactly row_dot's sequence, the block's
// nibbles and scale decoded once for all R
template <bool F16>
inline void rows_dot(const float* const* xs, uint32_t R, const uint8_t* qs_row, const uint8_t* e_row, uint32_t nb, float* res, int pf = 0) {
#if defined(__AVX2__)
    __m256 acc[kCpuExpertRows];
    for (uint32_t r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    if (pf) _mm_prefetch(reinterpret_cast<const char*>(e_row) + pf / 16, _MM_HINT_T0);
    for (uint32_t b = 0; b < nb; ++b) {
        if (pf && (b & 3) == 0) _mm_prefetch(reinterpret_cast<const char*>(qs_row) + size_t(b) * 16 + pf, _MM_HINT_T0);
        __m256 w[4]; block_weights<F16>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(qs_row + size_t(b) * 16)), w);
        const __m256 sc = _mm256_set1_ps(cpu_mxfp4_e8m0_half(e_row[b]));
        for (uint32_t r = 0; r < R; ++r) {
            const float* xb = xs[r] + size_t(b) * 32;
            __m256 d = _mm256_setzero_ps();
            d = _mm256_fmadd_ps(w[0], _mm256_loadu_ps(xb), d);
            d = _mm256_fmadd_ps(w[1], _mm256_loadu_ps(xb + 8), d);
            d = _mm256_fmadd_ps(w[2], _mm256_loadu_ps(xb + 16), d);
            d = _mm256_fmadd_ps(w[3], _mm256_loadu_ps(xb + 24), d);
            acc[r] = _mm256_fmadd_ps(d, sc, acc[r]);
        }
    }
    for (uint32_t r = 0; r < R; ++r) {
        alignas(32) float tmp[8]; _mm256_store_ps(tmp, acc[r]);
        res[r] = ((tmp[0] + tmp[4]) + (tmp[1] + tmp[5])) + ((tmp[2] + tmp[6]) + (tmp[3] + tmp[7]));
    }
#else
    for (uint32_t r = 0; r < R; ++r) res[r] = row_dot<F16>(xs[r], qs_row, e_row, nb);
#endif
}

// B66: the F16 block decoder (block_weights<true>), on by default: the same floats with fewer instructions (a cached
// expert -15 %, two-row experts -9 %); IE_CPU_MXFP4_F16=0 = the int8 route
bool f16_dec() { static const bool on = [] { const char* v = std::getenv("IE_CPU_MXFP4_F16"); return !(v && *v == '0'); }(); return on; }

void gemv_mxfp4_f32_rows(const float* const* xs, uint32_t R, const uint8_t* qs, const uint8_t* e, float* const* ys, uint32_t K, uint32_t N, int nthreads) {
    const uint32_t nb = K / 32;
    const int pf = pf_dist();
    auto run = [&](auto f16) {
        #pragma omp parallel for schedule(static) num_threads(nthreads > 0 ? nthreads : omp_get_max_threads())
        for (int64_t n = 0; n < int64_t(N); ++n) {
            float res[kCpuExpertRows];
            rows_dot<decltype(f16)::value>(xs, R, qs + size_t(n) * nb * 16, e + size_t(n) * nb, nb, res, pf);
            for (uint32_t r = 0; r < R; ++r) ys[r][n] = res[r];
        }
    };
    if (f16_dec()) run(std::true_type{}); else run(std::false_type{});
}
}  // namespace

void cpu_gemv_mxfp4_f32(const float* x, const uint8_t* qs, const uint8_t* e, float* y, uint32_t K, uint32_t N, int nthreads) {
    const uint32_t nb = K / 32;
    const int pf = pf_dist();
    auto run = [&](auto f16) {
        #pragma omp parallel for schedule(static) num_threads(nthreads > 0 ? nthreads : omp_get_max_threads())
        for (int64_t n = 0; n < int64_t(N); ++n)
            y[n] = row_dot<decltype(f16)::value>(x, qs + size_t(n) * nb * 16, e + size_t(n) * nb, nb, pf);
    };
    if (f16_dec()) run(std::true_type{}); else run(std::false_type{});
}

namespace {
// The SwiGLU of n entries, in place in g: the GPU's ds4_swiglu_clamped_h, in fp32. One function for the serial and the
// team form, so both compile to the same element code.
inline void swiglu_span(float* g, const float* u, uint32_t n, float swiglu_limit) {
    for (uint32_t i = 0; i < n; ++i) {
        float gg = g[i], uu = u[i];
        if (swiglu_limit > 0.f) { gg = std::min(gg, swiglu_limit); uu = std::max(std::min(uu, swiglu_limit), -swiglu_limit); }
        g[i] = (gg / (1.0f + std::exp(-gg))) * uu;
    }
}
}  // namespace

bool cpu_mxfp4_team() {   // B66: the leg's elementwise work (SwiGLU, the row stores) split over the team; IE_CPU_MXFP4_TEAM=0 = serial
    static const bool on = [] { const char* v = std::getenv("IE_CPU_MXFP4_TEAM"); return !(v && *v == '0'); }();
    return on;
}

namespace {
// SwiGLU over EF entries: serial (v0.2.21), or spans of 64 entries over the team -- each span the same function on the
// same entries, so no entry depends on the split
void swiglu(float* g, const float* u, uint32_t EF, float swiglu_limit, int nthreads) {
    if (!cpu_mxfp4_team() || EF % 64) { swiglu_span(g, u, EF, swiglu_limit); return; }
    #pragma omp parallel for schedule(static) num_threads(nthreads > 0 ? nthreads : omp_get_max_threads())
    for (int64_t s = 0; s < int64_t(EF / 64); ++s) swiglu_span(g + s * 64, u + s * 64, 64, swiglu_limit);
}
}  // namespace

void cpu_expert_mxfp4(const void* slot, const Ds4SlotLayout& lay, const float* x, float* scratch, float* out, float swiglu_limit, int nthreads) {
    const auto* base = static_cast<const uint8_t*>(slot);
    const uint32_t H = lay.gate.K, EF = lay.gate.N;
    float* g = scratch; float* u = scratch + EF;
    cpu_gemv_mxfp4_f32(x, base + lay.gate.off0, base + lay.gate.off1, g, H, EF, nthreads);
    cpu_gemv_mxfp4_f32(x, base + lay.up.off0,   base + lay.up.off1,   u, H, EF, nthreads);
    swiglu(g, u, EF, swiglu_limit, nthreads);
    cpu_gemv_mxfp4_f32(g, base + lay.down.off0, base + lay.down.off1, out, EF, H, nthreads);
}

void cpu_expert_mxfp4_rows(const void* slot, const Ds4SlotLayout& lay, const float* const* xs, uint32_t R, float* scratch, float* out, float swiglu_limit, int nthreads) {
    const auto* base = static_cast<const uint8_t*>(slot);
    const uint32_t H = lay.gate.K, EF = lay.gate.N;
    float* g[kCpuExpertRows]; float* u[kCpuExpertRows]; float* o[kCpuExpertRows]; const float* gc[kCpuExpertRows];
    for (uint32_t r = 0; r < R; ++r) { g[r] = scratch + size_t(r) * 2 * EF; u[r] = g[r] + EF; o[r] = out + size_t(r) * H; gc[r] = g[r]; }
    gemv_mxfp4_f32_rows(xs, R, base + lay.gate.off0, base + lay.gate.off1, g, H, EF, nthreads);
    gemv_mxfp4_f32_rows(xs, R, base + lay.up.off0,   base + lay.up.off1,   u, H, EF, nthreads);
    for (uint32_t r = 0; r < R; ++r) swiglu(g[r], u[r], EF, swiglu_limit, nthreads);   // cpu_expert_mxfp4's SwiGLU, row by row
    gemv_mxfp4_f32_rows(gc, R, base + lay.down.off0, base + lay.down.off1, o, EF, H, nthreads);
}

namespace {
// rows [n0, n1) of one MXFP4 GEMV on the calling thread: R = 1 by row_dot (cpu_gemv_mxfp4_f32's), R > 1 by rows_dot
// (gemv_mxfp4_f32_rows')
template <bool F16>
void gemv_range(const float* const* xs, uint32_t R, const uint8_t* qs, const uint8_t* e, float* const* ys, uint32_t K,
                uint32_t n0, uint32_t n1, int pf) {
    const uint32_t nb = K / 32;
    if (R == 1) { for (uint32_t n = n0; n < n1; ++n) ys[0][n] = row_dot<F16>(xs[0], qs + size_t(n) * nb * 16, e + size_t(n) * nb, nb, pf); return; }
    for (uint32_t n = n0; n < n1; ++n) {
        float res[kCpuExpertRows];
        rows_dot<F16>(xs, R, qs + size_t(n) * nb * 16, e + size_t(n) * nb, nb, res, pf);
        for (uint32_t r = 0; r < R; ++r) ys[r][n] = res[r];
    }
}

template <bool F16>
void experts_fused(const CpuExpertJob* jobs, size_t n_jobs, const Ds4SlotLayout& lay, float* scratch, float swiglu_limit,
                   int nthreads, CpuRowStore store, void* ctx) {
    const uint32_t H = lay.gate.K, EF = lay.gate.N;
    const int pf = pf_dist();
    float* const out = scratch + size_t(2) * kCpuExpertRows * 2 * EF;   // kCpuExpertRows x H: each thread its own columns
    #pragma omp parallel num_threads(nthreads > 0 ? nthreads : omp_get_max_threads())
    {
        const uint32_t t = uint32_t(omp_get_thread_num()), nt = uint32_t(omp_get_num_threads());
        const uint32_t a0 = uint32_t(uint64_t(EF) * t / nt), a1 = uint32_t(uint64_t(EF) * (t + 1) / nt);
        const uint32_t b0 = uint32_t(uint64_t(H) * t / nt), b1 = uint32_t(uint64_t(H) * (t + 1) / nt);
        for (size_t i = 0; i < n_jobs; ++i) {
            const CpuExpertJob& j = jobs[i];
            const auto* base = static_cast<const uint8_t*>(j.slot);
            float* const sh = scratch + (i & 1) * size_t(kCpuExpertRows) * 2 * EF;
            float* g[kCpuExpertRows]; float* u[kCpuExpertRows]; float* o[kCpuExpertRows]; const float* gc[kCpuExpertRows];
            for (uint32_t r = 0; r < j.R; ++r) { g[r] = sh + size_t(r) * 2 * EF; u[r] = g[r] + EF; o[r] = out + size_t(r) * H; gc[r] = g[r]; }
            gemv_range<F16>(j.xs, j.R, base + lay.gate.off0, base + lay.gate.off1, g, H, a0, a1, pf);
            gemv_range<F16>(j.xs, j.R, base + lay.up.off0,   base + lay.up.off1,   u, H, a0, a1, pf);
            for (uint32_t r = 0; r < j.R; ++r) swiglu_span(g[r] + a0, u[r] + a0, a1 - a0, swiglu_limit);
            #pragma omp barrier
            gemv_range<F16>(gc, j.R, base + lay.down.off0, base + lay.down.off1, o, EF, b0, b1, pf);
            for (uint32_t r = 0; r < j.R; ++r) store(ctx, j.out_row + r, b0, b1, o[r] + b0);
        }
    }
}
}  // namespace

void cpu_experts_mxfp4_fused(const CpuExpertJob* jobs, size_t n, const Ds4SlotLayout& lay, float* scratch,
                             float swiglu_limit, int nthreads, CpuRowStore store, void* ctx) {
    if (f16_dec()) experts_fused<true>(jobs, n, lay, scratch, swiglu_limit, nthreads, store, ctx);
    else experts_fused<false>(jobs, n, lay, scratch, swiglu_limit, nthreads, store, ctx);
}

}  // namespace ie
