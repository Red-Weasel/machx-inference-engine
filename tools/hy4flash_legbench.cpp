// ie-hy4flash-legbench: the Hy4-flash decode CPU leg (hy4f_q8_moe_cpu) alone on synthetic 8-bit experts laid out like the
// RAM pack, chosen at random from a pool far larger than the caches -- its RAM read rate without loading the model.
//   ie-hy4flash-legbench [jobs a call = 5] [calls = 3000] [pool GiB = 16] [threads = 0 (the leg's default)]
#include "ie/hy4flash_q8.hpp"

#include <omp.h>
#include <sys/mman.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const uint32_t jobs = argc > 1 ? uint32_t(std::atoi(argv[1])) : 5, calls = argc > 2 ? uint32_t(std::atoi(argv[2])) : 3000;
    const size_t pool_gib = argc > 3 ? size_t(std::atoi(argv[3])) : 16; const int threads = argc > 4 ? std::atoi(argv[4]) : 0;
    const uint32_t LAT = 1536, MI = 1536;
    const size_t qb = size_t(MI) * LAT, sb = size_t(MI) * (LAT / 32) * 2, ex = 3 * (qb + sb);
    const size_t n_ex = (pool_gib << 30) / ex;
    char* p = static_cast<char*>(mmap(nullptr, n_ex * ex, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (p == MAP_FAILED) { std::fprintf(stderr, "mmap failed\n"); return 1; }
    madvise(p, n_ex * ex, MADV_HUGEPAGE);
    #pragma omp parallel for
    for (size_t e = 0; e < n_ex; ++e) {
        std::mt19937 r{static_cast<uint32_t>(e)};
        int8_t* q = reinterpret_cast<int8_t*>(p + e * ex);
        for (size_t i = 0; i < 3 * qb; i += 4) { const uint32_t v = r(); std::memcpy(q + i, &v, 4); }
        uint16_t* s = reinterpret_cast<uint16_t*>(p + e * ex + 3 * qb);
        for (size_t i = 0; i < 3 * sb / 2; ++i) s[i] = 0x2000;     // fp16 ~0.0078
    }
    auto planes = [&](size_t e) {
        char* b = p + e * ex; ie::Hy4fQ8Store::Planes pl;
        pl.gate_qs = reinterpret_cast<int8_t*>(b); pl.up_qs = reinterpret_cast<int8_t*>(b + qb); pl.down_qs = reinterpret_cast<int8_t*>(b + 2 * qb);
        pl.gate_d = reinterpret_cast<uint16_t*>(b + 3 * qb); pl.up_d = reinterpret_cast<uint16_t*>(b + 3 * qb + sb); pl.down_d = reinterpret_cast<uint16_t*>(b + 3 * qb + 2 * sb);
        return pl;
    };
    std::vector<float> x(LAT), y(LAT), w(jobs, 0.1f);
    std::mt19937 r(7); std::normal_distribution<float> nd;
    for (auto& v : x) v = nd(r);
    std::vector<ie::Hy4fQ8Store::Planes> pl(jobs);
    double best = 0, sum = 0; uint32_t reps = 0;
    for (uint32_t rep = 0; rep < 3; ++rep) {
        double t = 0;
        for (uint32_t c = 0; c < calls; ++c) {
            for (uint32_t j = 0; j < jobs; ++j) pl[j] = planes(r() % n_ex);
            std::fill(y.begin(), y.end(), 0.f);
            const auto t0 = std::chrono::steady_clock::now();
            ie::hy4f_q8_moe_cpu(x.data(), pl.data(), w.data(), jobs, y.data(), LAT, MI, 7.f, threads, ie::Hy4fQFmt{});
            t += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            // (the card-driving thread does other work between two calls of a token: ~0.5 ms here)
            const auto tw = std::chrono::steady_clock::now(); while (std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count() < 2e-4) {}
        }
        const double gbs = double(calls) * jobs * ex / t / 1e9; best = std::max(best, gbs); sum += gbs; ++reps;
        std::printf("rep %u: %.1f us a call, %.1f GB/s\n", rep, 1e6 * t / calls, gbs);
    }
    std::printf("legbench jobs %u: mean %.1f GB/s, best %.1f (y[0] %.4f)\n", jobs, sum / reps, best, y[0]);
    return 0;
}
