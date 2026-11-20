// src/ops/hy4flash_q8_cpu.cpp — Hy4-flash: a decode token's MISSED experts computed on the CPU from their RAM slots
// (include/ie/hy4flash_q8.hpp). Profile of 2026-11-18: about a quarter of a token's experts are not in VRAM, and copying
// them up (1.2 GB a token) was 58 % of the token. Here they never leave RAM: W8A16 with integer dot products --
// the activation scaled to int16 (one scale per vector), 16 int8 weights sign-extended and multiplied per AVX2 madd,
// the block's fp16 scale applied to each 32-weight block sum. Built with -mavx2 -mfma -mf16c -fopenmp.
#include "ie/hy4flash_q8.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <vector>

namespace ie {

namespace {
// bit i of x -> byte i = 0xFF (ggml's bytes_from_bits_32)
inline __m256i bytes_from_bits_32(uint32_t x) {
    __m256i bytes = _mm256_set1_epi32(int(x));
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0x0000000000000000LL);
    bytes = _mm256_shuffle_epi8(bytes, shuf);
    bytes = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL));
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}
// the 32 weights of block b of a row stored at B bits (Hy4fQFmt's layouts) as two vectors of 16 int16
template <int B> inline void load_w(const int8_t* row, uint32_t b, uint32_t K, __m256i& w0, __m256i& w1) {
    if constexpr (B == 8) {
        w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + size_t(b) * 32)));
        w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + size_t(b) * 32 + 16)));
    } else {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + size_t(b) * 16));
        const __m128i m4 = _mm_set1_epi8(0x0F);
        const __m128i lo = _mm_and_si128(raw, m4), hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
        __m256i v = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));        // weights 0..31 as bytes
        if constexpr (B == 5) {
            uint32_t hb; std::memcpy(&hb, row + (K >> 1) + size_t(b) * 4, 4);
            v = _mm256_or_si256(v, _mm256_and_si256(bytes_from_bits_32(hb), _mm256_set1_epi8(0x10)));
            v = _mm256_sub_epi8(v, _mm256_set1_epi8(16));
        } else v = _mm256_sub_epi8(v, _mm256_set1_epi8(8));
        w0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
        w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
    }
}
// sum_b d[b] * sum_{i < 32} q[b*32 + i] * xi[b*32 + i]   (K a multiple of 32; |xi| <= 32767, |q| <= 127)
// Eight blocks a step: their integer sums are reduced side by side into one vector (block i in lane i), converted once
// and multiplied by the eight fp16 scales -- one convert / multiply / add per 256 weights instead of per 32, and no
// 4-cycle add chain per block. One fast core, data in its cache: 44.9 -> 72.2 GB/s; the team on bursts of random
// experts from the pack: 59.6 -> 69.0 GB/s (tmp bw/kern.c, 2026-11-19). The integer sums are exact; the float sum over
// the blocks is taken in a different order (differences of 4e-6 relative). IE_HY4F_CPU_KERNEL=0 = the first form.
inline bool wide_rows() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_CPU_KERNEL"); return !(v && *v == '0'); }(); return on; }
inline __m256i block_sum8(const int8_t* w, const int16_t* xi, uint32_t b) {
    const __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + size_t(b) * 32)));
    const __m256i w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + size_t(b) * 32 + 16)));
    return _mm256_add_epi32(_mm256_madd_epi16(w0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xi + size_t(b) * 32))),
                            _mm256_madd_epi16(w1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xi + size_t(b) * 32 + 16))));
}
inline float row_dot8_wide(const int8_t* w, const uint16_t* d, const int16_t* xi, uint32_t K) {      // 8-bit rows, K % 256 == 0
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t b = 0; b < K / 32; b += 8) {
        const __m256i h01 = _mm256_hadd_epi32(block_sum8(w, xi, b), block_sum8(w, xi, b + 1)), h23 = _mm256_hadd_epi32(block_sum8(w, xi, b + 2), block_sum8(w, xi, b + 3));
        const __m256i h45 = _mm256_hadd_epi32(block_sum8(w, xi, b + 4), block_sum8(w, xi, b + 5)), h67 = _mm256_hadd_epi32(block_sum8(w, xi, b + 6), block_sum8(w, xi, b + 7));
        const __m256i q0 = _mm256_hadd_epi32(h01, h23), q1 = _mm256_hadd_epi32(h45, h67);            // each 128-bit half: partial sums of blocks 0..3 | 4..7
        const __m256i sums = _mm256_add_epi32(_mm256_permute2x128_si256(q0, q1, 0x20), _mm256_permute2x128_si256(q0, q1, 0x31));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sums), _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(d + b))), acc);
    }
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
// Four rows against one 8-bit weight row (K % 256 == 0): each weight block is widened once and multiplied into all four
// rows with VNNI (vpdpwssd = the same int32 lane sums as madd + add: the products are exact and summed in the same lanes),
// then each row takes row_dot8_wide's float steps unchanged -- out[r] is bit-identical to row_dot8_wide(w, d, x[r], K).
__attribute__((target("avx2,fma,f16c,avxvnni")))
inline void row_dot8_x4(const int8_t* w, const uint16_t* d, const int16_t* const x[4], uint32_t K, float out[4]) {
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (uint32_t b = 0; b < K / 32; b += 8) {
        __m256i bs[4][8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + size_t(b + k) * 32)));
            const __m256i w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + size_t(b + k) * 32 + 16)));
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r] + size_t(b + k) * 32));
                const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r] + size_t(b + k) * 32 + 16));
                bs[r][k] = _mm256_dpwssd_avx_epi32(_mm256_madd_epi16(w0, x0), w1, x1);
            }
        }
        const __m256 sc = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(d + b)));
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const __m256i h01 = _mm256_hadd_epi32(bs[r][0], bs[r][1]), h23 = _mm256_hadd_epi32(bs[r][2], bs[r][3]);
            const __m256i h45 = _mm256_hadd_epi32(bs[r][4], bs[r][5]), h67 = _mm256_hadd_epi32(bs[r][6], bs[r][7]);
            const __m256i q0 = _mm256_hadd_epi32(h01, h23), q1 = _mm256_hadd_epi32(h45, h67);
            const __m256i sums = _mm256_add_epi32(_mm256_permute2x128_si256(q0, q1, 0x20), _mm256_permute2x128_si256(q0, q1, 0x31));
            acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sums), sc, acc[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc[r]), _mm256_extractf128_ps(acc[r], 1));
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        out[r] = _mm_cvtss_f32(lo);
    }
}
inline bool x4_ok(uint32_t bits, uint32_t K) {
    static const bool on = [] { const char* v = std::getenv("IE_HY4F_CPU_X4"); return !(v && *v == '0') && __builtin_cpu_supports("avxvnni"); }();
    return on && bits == 8 && (K & 255) == 0 && wide_rows();
}

template <int B> inline float row_dot_t(const int8_t* w, const uint16_t* d, const int16_t* xi, uint32_t K) {
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t b = 0; b < K / 32; ++b) {
        __m256i w0, w1; load_w<B>(w, b, K, w0, w1);
        const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xi + size_t(b) * 32));
        const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xi + size_t(b) * 32 + 16));
        const __m256i sm = _mm256_add_epi32(_mm256_madd_epi16(w0, x0), _mm256_madd_epi16(w1, x1));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sm), _mm256_set1_ps(_cvtsh_ss(d[b])), acc);
    }
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
inline float row_dot(uint32_t bits, const int8_t* w, const uint16_t* d, const int16_t* xi, uint32_t K) {
    if (bits == 8 && (K & 255) == 0 && wide_rows()) return row_dot8_wide(w, d, xi, K);
    return bits == 8 ? row_dot_t<8>(w, d, xi, K) : bits == 5 ? row_dot_t<5>(w, d, xi, K) : row_dot_t<4>(w, d, xi, K);
}
// x [n] fp32 -> int16 with one scale: xi = round(x * s), s = 32767 / max|x|; returns 1 / s (0 for an all-zero vector)
inline float to_i16(const float* x, uint32_t n, int16_t* xi) {
    float amax = 0.f;
    for (uint32_t i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
    if (!(amax > 0.f)) { std::memset(xi, 0, size_t(n) * 2); return 0.f; }
    const float s = 32767.0f / amax;
    for (uint32_t i = 0; i < n; ++i) xi[i] = int16_t(std::lrintf(x[i] * s));
    return 1.0f / s;
}
// 4 of the allowed CPUs stay free for the thread that drives the cards and the driver's own (measured on 18 allowed CPUs,
// 200 tokens: 6 threads 7.9 s in this leg, 10 6.8 s, 14 6.4 s, 18 9.5 s)
inline int default_threads() { return std::max(1, std::min(omp_get_max_threads() - 4, 16)); }

// The decode leg's team size, chosen from its own measured speed (IE_HY4F_CPU_THREADS fixes it): the test desktop runs
// other jobs (a test suite's headless browsers took the leg from 63.5 GB/s at 12 threads to 46 at 14, 2026-11-19), and a
// thread descheduled inside a region holds the region up. Candidates around the default; each call's time per missed
// expert feeds that size's running mean; every 64th call tries the next size. The sums are the same at any size (a
// row is one thread's dot product; the jobs are added in a fixed order).
struct LegTeam {
    int cand[4] = {0, 0, 0, 0}; int nc = 0; double mean[4] = {0, 0, 0, 0}; uint32_t seen[4] = {0, 0, 0, 0}; uint64_t calls = 0; int cur = 0;
    int pick() {
        if (!nc) { const int d = default_threads(); const int c0[4] = {d - 4, d - 2, d, d + 2}; for (int x : c0) if (x >= 4 && x <= omp_get_max_threads()) cand[nc++] = x; if (!nc) cand[nc++] = d; }
        ++calls;
        if (calls % 64 == 0 || seen[cur] < 4) {                       // explore: the next size (each one 4 calls at first)
            for (int i = 0; i < nc; ++i) if (seen[i] < 4) { cur = i; return cand[cur]; }
            cur = int((calls / 64) % uint64_t(nc)); return cand[cur];
        }
        int b = 0; for (int i = 1; i < nc; ++i) if (mean[i] < mean[b]) b = i;
        cur = b; return cand[cur];
    }
    void note(double us_per_job) {      // (a call the scheduler stalled counts at most twice the size's mean: one stall must not bury a size)
        if (seen[cur] >= 4) us_per_job = std::min(us_per_job, 2.0 * mean[cur]);
        const double a = seen[cur] < 4 ? 1.0 / double(seen[cur] + 1) : 0.1; mean[cur] += a * (us_per_job - mean[cur]); ++seen[cur];
    }
};
static LegTeam g_team;
inline bool adaptive_team() { static const bool on = [] { const char* v = std::getenv("IE_HY4F_CPU_ADAPT"); return !(v && *v == '0'); }(); return on; }

// How long this leg's idle workers spin before they sleep. Intel's OpenMP default is 200 ms: between two layers of a
// decode token that is the whole team spinning on the cores the card-driving thread and the driver's threads need.
// Measured on the test desktop with other jobs running (load average 12-17, 200 unseen tokens, three interleaved
// repeats): 200 ms 19.9-21.8 tok/s, 0 ms 22.3-22.6, 1 ms 22.5-23.1. The value is the calling thread's for the
// regions of one call (workers take it at the fork) and is put back after, so no other leg of the engine changes.
// IE_HY4F_CPU_SPIN_MS=<ms> (default 1; -1 = leave the runtime's value).
struct LegSpin {
    int old = -1;
    LegSpin() {
        static const int ms = [] { const char* v = std::getenv("IE_HY4F_CPU_SPIN_MS"); return v && *v ? std::atoi(v) : 1; }();
        if (ms >= 0) { old = kmp_get_blocktime(); kmp_set_blocktime(ms); }
    }
    ~LegSpin() { if (old >= 0) kmp_set_blocktime(old); }
};

// Rows of one expert matrix a unit of the decode leg computes. 96 (the first form: 42 units for 14 threads at 2.6 missed
// experts a call) left the last units on the slow cores while the fast ones had none -- 7.7 of 14 threads busy on
// average. At 16: 242-253 -> 220-223 us (gate / up) and 129-135 -> 118 us (down) a call, 55-58 GB/s against the
// 60-75 this machine's RAM gives the same access pattern (A/B on one binary, 2026-11-19). IE_HY4F_CPU_ROWS_UNIT.
inline uint32_t rows_unit() { static const uint32_t n = [] { const char* v = std::getenv("IE_HY4F_CPU_ROWS_UNIT"); const int k = v ? std::atoi(v) : 0; return uint32_t(k > 0 ? k : 16); }(); return n; }

}  // namespace

void hy4f_prefault(const uint8_t* const* addrs, size_t n, size_t bytes, int threads) {
    if (!n) return;
    if (threads <= 0) threads = 16;                     // waiting on the drive, not computing
    unsigned sink = 0;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads) reduction(+ : sink) if (n > 1)
    for (size_t i = 0; i < n; ++i) sink += unsigned(addrs[i][0]) + unsigned(addrs[i][bytes ? bytes - 1 : 0]);
    static volatile unsigned keep; keep = sink;
}

// IE_HY4F_CPU_LEG_DIAG=1 (a measurement): where a decode call of the leg spends its time. Printed at exit.
struct LegDiag {
    bool on = [] { const char* v = std::getenv("IE_HY4F_CPU_LEG_DIAG"); return v && *v == '1'; }();
    uint64_t calls = 0, jobs = 0; double t_pre = 0, t_gu = 0, t_mid = 0, t_dn = 0, t_tail = 0, work_gu = 0, work_dn = 0; int threads = 0;
    ~LegDiag() {
        if (!calls) return;
        const double c = double(calls);
        std::fprintf(stderr, "[leg] %llu calls, %.2f jobs a call, %d threads; us a call: setup %.1f, gate/up region %.1f, between %.1f, down region %.1f, sum-up %.1f\n",
                     (unsigned long long)calls, jobs / c, threads, 1e6 * t_pre / c, 1e6 * t_gu / c, 1e6 * t_mid / c, 1e6 * t_dn / c, 1e6 * t_tail / c);
        if (g_team.nc) { std::fprintf(stderr, "[leg] team sizes (us a missed expert):"); for (int i = 0; i < g_team.nc; ++i) std::fprintf(stderr, " %d: %.1f (%u calls)", g_team.cand[i], g_team.mean[i], g_team.seen[i]); std::fprintf(stderr, "\n"); }
        std::fprintf(stderr, "[leg] thread-time inside the loops, us a call: gate/up %.1f (= %.1f threads busy over its region), down %.1f (= %.1f)\n",
                     1e6 * work_gu / c, work_gu / std::max(t_gu, 1e-12), 1e6 * work_dn / c, work_dn / std::max(t_dn, 1e-12));
    }
};
static LegDiag g_leg;

void hy4f_q8_moe_cpu(const float* x, const Hy4fQ8Store::Planes* pl, const float* w, uint32_t n_jobs, float* y,
                     uint32_t LAT, uint32_t MI, float lim, int threads, Hy4fQFmt fmt) {
    const uint32_t bg = fmt.gu, bd = fmt.dn; const size_t rg = Hy4fQFmt::row_bytes(bg, LAT), rd = Hy4fQFmt::row_bytes(bd, MI);
    if (!n_jobs) return;
    const bool adapt = threads <= 0 && adaptive_team();
    if (threads <= 0) threads = adapt ? g_team.pick() : default_threads();
    const double t_call0 = adapt ? omp_get_wtime() : 0;
    static thread_local std::vector<int16_t> xi; static thread_local std::vector<float> h; static thread_local std::vector<int16_t> hi; static thread_local std::vector<float> yj;
    xi.resize(LAT); h.resize(size_t(n_jobs) * MI); hi.resize(size_t(n_jobs) * MI); yj.resize(size_t(n_jobs) * LAT);
    const bool dg = g_leg.on; double w_gu = 0, w_dn = 0;
    const double t0 = dg ? omp_get_wtime() : 0;
    const float inv_x = to_i16(x, LAT, xi.data());
    const int16_t* xip = xi.data(); float* hp = h.data();
    const double t1 = dg ? omp_get_wtime() : 0;
    // gate, up and the clamped SwiGLU of every missed expert: (job, row) units over the team
    // (dynamic: the machine mixes fast and slow cores; equal shares would wait for the slow ones)
    const LegSpin spin; const int ru_omp = int(rows_unit());
    #pragma omp parallel num_threads(threads) reduction(+ : w_gu)
    {
    const double tw = dg ? omp_get_wtime() : 0;
    #pragma omp for collapse(2) schedule(dynamic, ru_omp) nowait
    for (uint32_t j = 0; j < n_jobs; ++j)
        for (uint32_t n = 0; n < MI; ++n) {
            const float g = std::min(row_dot(bg, pl[j].gate_qs + size_t(n) * rg, pl[j].gate_d + size_t(n) * (LAT / 32), xip, LAT) * inv_x, lim);
            const float u = std::min(std::max(row_dot(bg, pl[j].up_qs + size_t(n) * rg, pl[j].up_d + size_t(n) * (LAT / 32), xip, LAT) * inv_x, -lim), lim);
            hp[size_t(j) * MI + n] = g / (1.0f + std::exp(-g)) * u;
        }
    if (dg) w_gu += omp_get_wtime() - tw;
    }
    const double t2 = dg ? omp_get_wtime() : 0;
    std::vector<float> inv_h(n_jobs);
    for (uint32_t j = 0; j < n_jobs; ++j) inv_h[j] = to_i16(hp + size_t(j) * MI, MI, hi.data() + size_t(j) * MI);
    const int16_t* hip = hi.data(); float* yjp = yj.data(); const float* invh = inv_h.data();
    const double t3 = dg ? omp_get_wtime() : 0;
    #pragma omp parallel num_threads(threads) reduction(+ : w_dn)
    {
    const double tw = dg ? omp_get_wtime() : 0;
    #pragma omp for collapse(2) schedule(dynamic, ru_omp) nowait
    for (uint32_t j = 0; j < n_jobs; ++j)
        for (uint32_t n = 0; n < LAT; ++n)
            yjp[size_t(j) * LAT + n] = row_dot(bd, pl[j].down_qs + size_t(n) * rd, pl[j].down_d + size_t(n) * (MI / 32), hip + size_t(j) * MI, MI) * invh[j] * w[j];
    if (dg) w_dn += omp_get_wtime() - tw;
    }
    const double t4 = dg ? omp_get_wtime() : 0;
    for (uint32_t j = 0; j < n_jobs; ++j)
        for (uint32_t n = 0; n < LAT; ++n) y[n] += yjp[size_t(j) * LAT + n];
    if (adapt) g_team.note(1e6 * (omp_get_wtime() - t_call0) / double(n_jobs));
    if (dg) {
        const double t5 = omp_get_wtime();
        ++g_leg.calls; g_leg.jobs += n_jobs; g_leg.threads = threads;
        g_leg.t_pre += t1 - t0; g_leg.t_gu += t2 - t1; g_leg.t_mid += t3 - t2; g_leg.t_dn += t4 - t3; g_leg.t_tail += t5 - t4; g_leg.work_gu += w_gu; g_leg.work_dn += w_dn;
    }
}

void hy4f_q8_moe_cpu_rows(const float* x, const Hy4fQ8Store::Planes* pl, const int32_t* job_row, const float* w, uint32_t n_jobs,
                          uint32_t T, float* y, uint32_t LAT, uint32_t MI, float lim, int threads, Hy4fQFmt fmt, float* h_all) {
    const uint32_t bg = fmt.gu, bd = fmt.dn; const size_t rg = Hy4fQFmt::row_bytes(bg, LAT), rd = Hy4fQFmt::row_bytes(bd, MI);
    if (!n_jobs) return;
    if (threads <= 0) threads = default_threads();
    const LegSpin spin;
    static thread_local std::vector<int16_t> xi; static thread_local std::vector<float> inv_x, yj;
    xi.resize(size_t(T) * LAT); inv_x.resize(T); yj.resize(size_t(n_jobs) * LAT);
    for (uint32_t t = 0; t < T; ++t) inv_x[t] = to_i16(x + size_t(t) * LAT, LAT, xi.data() + size_t(t) * LAT);
    const int16_t* xip = xi.data(); const float* invx = inv_x.data(); float* yjp = yj.data();
    // The jobs of one expert are consecutive (the caller sorts by expert). A GROUP is one expert with all the rows of
    // the chunk that picked it: its 7.5 MB go through a core once and every weight row is used for each of the group's
    // rows while it is in the cache. (The first form ran each (expert, row) job on its own and read the expert once per
    // row: a 3,900-token prompt spent 60 of its 90 s there, at the RAM's full speed.) IE_HY4F_CPU_GROUPS=0 = that form.
    // Each job's arithmetic and the order of the final sums are the first form's: the same bits.
    static const bool grouped = [] { const char* v = std::getenv("IE_HY4F_CPU_GROUPS"); return !(v && *v == '0'); }();
    std::vector<uint32_t> g0;                               // group starts, then n_jobs
    for (uint32_t j = 0; j < n_jobs; ++j) if (!grouped || j == 0 || pl[j].gate_qs != pl[j - 1].gate_qs) g0.push_back(j);
    const uint32_t n_groups = uint32_t(g0.size()); g0.push_back(n_jobs);
    std::vector<uint32_t> order(n_groups);                  // the largest groups first: the team's tail is then small work
    for (uint32_t g = 0; g < n_groups; ++g) order[g] = g;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return g0[a + 1] - g0[a] > g0[b + 1] - g0[b]; });
    #pragma omp parallel num_threads(threads)
    {
        std::vector<float> h, ih; std::vector<int16_t> hi;
        #pragma omp for schedule(dynamic, 1)
        for (uint32_t gi = 0; gi < n_groups; ++gi) {
            const uint32_t j0 = g0[order[gi]], R = g0[order[gi] + 1] - j0;
            const Hy4fQ8Store::Planes& P = pl[j0];
            h.resize(size_t(R) * MI); hi.resize(size_t(R) * MI); ih.resize(R);
            for (uint32_t n = 0; n < MI; ++n) {
                const int8_t* gq = P.gate_qs + size_t(n) * rg; const uint16_t* gd = P.gate_d + size_t(n) * (LAT / 32);
                const int8_t* uq = P.up_qs + size_t(n) * rg;   const uint16_t* ud = P.up_d + size_t(n) * (LAT / 32);
                uint32_t r = 0;
                if (x4_ok(bg, LAT))                // four rows a weight row: each weight widened once (the same bits)
                    for (; r + 4 <= R; r += 4) {
                        const int16_t* xs[4]; float gs[4], us[4];
                        for (int k = 0; k < 4; ++k) xs[k] = xip + size_t(job_row[j0 + r + k]) * LAT;
                        row_dot8_x4(gq, gd, xs, LAT, gs); row_dot8_x4(uq, ud, xs, LAT, us);
                        for (int k = 0; k < 4; ++k) {
                            const float ix = invx[job_row[j0 + r + k]];
                            const float g = std::min(gs[k] * ix, lim), u = std::min(std::max(us[k] * ix, -lim), lim);
                            h[size_t(r + k) * MI + n] = g / (1.0f + std::exp(-g)) * u;
                        }
                    }
                for (; r < R; ++r) {
                    const int32_t row = job_row[j0 + r]; const int16_t* xr = xip + size_t(row) * LAT; const float ix = invx[row];
                    const float g = std::min(row_dot(bg, gq, gd, xr, LAT) * ix, lim);
                    const float u = std::min(std::max(row_dot(bg, uq, ud, xr, LAT) * ix, -lim), lim);
                    h[size_t(r) * MI + n] = g / (1.0f + std::exp(-g)) * u;
                }
            }
            for (uint32_t r = 0; r < R; ++r) {
                if (h_all) std::memcpy(h_all + size_t(j0 + r) * MI, h.data() + size_t(r) * MI, size_t(MI) * 4);
                ih[r] = to_i16(h.data() + size_t(r) * MI, MI, hi.data() + size_t(r) * MI) * w[j0 + r];
            }
            for (uint32_t n = 0; n < LAT; ++n) {
                const int8_t* dq = P.down_qs + size_t(n) * rd; const uint16_t* dd = P.down_d + size_t(n) * (MI / 32);
                uint32_t r = 0;
                if (x4_ok(bd, MI))
                    for (; r + 4 <= R; r += 4) {
                        const int16_t* hs[4] = {hi.data() + size_t(r) * MI, hi.data() + size_t(r + 1) * MI, hi.data() + size_t(r + 2) * MI, hi.data() + size_t(r + 3) * MI};
                        float o[4]; row_dot8_x4(dq, dd, hs, MI, o);
                        for (int k = 0; k < 4; ++k) yjp[size_t(j0 + r + k) * LAT + n] = o[k] * ih[r + k];
                    }
                for (; r < R; ++r) yjp[size_t(j0 + r) * LAT + n] = row_dot(bd, dq, dd, hi.data() + size_t(r) * MI, MI) * ih[r];
            }
        }
    }
    for (uint32_t j = 0; j < n_jobs; ++j) {
        float* yr = y + size_t(job_row[j]) * LAT; const float* s = yjp + size_t(j) * LAT;
        for (uint32_t n = 0; n < LAT; ++n) yr[n] += s[n];
    }
}

void hy4f_gram_add(const float* R, size_t n, uint32_t K, double* A, int threads) {
    if (!n) return;
    if (threads <= 0) threads = default_threads();
    // rows of A in parallel; a row's sums in fp32 over blocks of 256 samples, folded into fp64
    #pragma omp parallel num_threads(threads)
    {
        std::vector<float> acc(K);
        #pragma omp for schedule(dynamic, 8)
        for (uint32_t i = 0; i < K; ++i) {
            double* Ai = A + size_t(i) * K;
            for (size_t n0 = 0; n0 < n; n0 += 256) {
                const size_t n1 = std::min(n, n0 + 256);
                std::fill(acc.begin(), acc.end(), 0.f);
                for (size_t r = n0; r < n1; ++r) {
                    const float* row = R + r * K; const float a = row[i];
                    for (uint32_t j = 0; j < K; ++j) acc[j] += a * row[j];
                }
                for (uint32_t j = 0; j < K; ++j) Ai[j] += double(acc[j]);
            }
        }
    }
}

}  // namespace ie
