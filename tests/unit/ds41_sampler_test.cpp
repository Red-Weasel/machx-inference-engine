// tests/unit/ds41_sampler_test.cpp -- Ds41Generator::sample_row (the host sampler of DeepSeek-V4.1 and MiMo-V2.6) against
// the full-sort implementation it replaced (2026-09-22: a sort of the whole 152,576-entry vocabulary per sampled row,
// 12 ms each with top_k 0): the same pick for the same RNG state on peaked random rows, across top-k / top-p / min-p /
// temperature settings, and the new one faster. CPU only.
#include "ie/deepseek41_generate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

uint64_t next_u64(uint64_t& s) { uint64_t z = (s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
double uniform01(uint64_t& s) { return double(next_u64(s) >> 11) * (1.0 / 9007199254740992.0); }

// the previous sample_row after the repeat penalty, verbatim
int32_t reference_row(const float* lg, size_t V, const ie::Ds41SampleParams& sp, uint64_t& rng) {
    if (sp.temperature <= 0.f) return int32_t(std::max_element(lg, lg + V) - lg);
    std::vector<int32_t> idx(V); std::iota(idx.begin(), idx.end(), 0);
    const size_t keep = sp.top_k && sp.top_k < V ? sp.top_k : V;
    std::partial_sort(idx.begin(), idx.begin() + std::ptrdiff_t(keep), idx.end(), [&](int32_t a, int32_t b) { return lg[size_t(a)] > lg[size_t(b)]; });
    idx.resize(keep);
    std::vector<double> p(keep); double mx = lg[size_t(idx[0])] / sp.temperature, z = 0;
    for (size_t i = 0; i < keep; ++i) { p[i] = std::exp(double(lg[size_t(idx[i])]) / sp.temperature - mx); z += p[i]; }
    for (auto& v : p) v /= z;
    size_t n = keep;
    if (sp.top_p < 1.f) { double c = 0; n = 0; while (n < keep) { c += p[n]; ++n; if (c >= sp.top_p) break; } }
    if (sp.min_p > 0.f) { const double floor = sp.min_p * p[0]; size_t m = 0; while (m < n && p[m] >= floor) ++m; n = std::max<size_t>(1, m); }
    double zz = 0; for (size_t i = 0; i < n; ++i) zz += p[i];
    const double r = uniform01(rng) * zz; double c = 0;
    for (size_t i = 0; i < n; ++i) { c += p[i]; if (r < c) return idx[i]; }
    return idx[n - 1];
}

}  // namespace

int main() {
    const size_t V = 152576;
    std::mt19937 g(20260922);
    std::vector<std::vector<float>> rows;
    for (int r = 0; r < 120; ++r) {   // an LM-like row: a broad tail plus 1..12 strong candidates
        std::vector<float> lg(V);
        std::normal_distribution<float> nd(0.f, 2.5f);
        for (auto& v : lg) v = nd(g);
        for (int k = 0; k < 1 + r % 12; ++k) lg[g() % V] += 10.f + float(g() % 1000) / 100.f;
        rows.push_back(std::move(lg));
    }
    struct Case { const char* name; float temp, top_p, min_p; uint32_t top_k; };
    const Case cases[] = {{"Dream: temp 1, top-p 0.95, top-k 0", 1.f, 0.95f, 0.f, 0}, {"temp 0.7, top-p 0.8", 0.7f, 0.8f, 0.f, 0},
                          {"top-k 40, top-p 0.95", 1.f, 0.95f, 0.f, 40}, {"min-p 0.05", 1.f, 1.f, 0.05f, 0},
                          {"no cut (top-p 1)", 1.f, 1.f, 0.f, 0}, {"temp 1.5, top-p 0.99, top-k 2000", 1.5f, 0.99f, 0.f, 2000}};
    int fails = 0;
    double t_new = 0, t_old = 0;
    const std::vector<int32_t> recent;
    for (const Case& c : cases) {
        ie::Ds41SampleParams sp; sp.temperature = c.temp; sp.top_p = c.top_p; sp.min_p = c.min_p; sp.top_k = c.top_k;
        int same = 0;
        for (size_t r = 0; r < rows.size(); ++r) {
            uint64_t s1 = 0x5eed0000ull + r * 7919, s2 = s1;
            std::vector<float> a = rows[r];
            const auto t0 = std::chrono::steady_clock::now();
            const int32_t x = ie::Ds41Generator::sample_row(a.data(), V, recent, sp, s1);
            const auto t1 = std::chrono::steady_clock::now();
            const int32_t y = reference_row(rows[r].data(), V, sp, s2);
            const auto t2 = std::chrono::steady_clock::now();
            t_new += std::chrono::duration<double, std::milli>(t1 - t0).count();
            t_old += std::chrono::duration<double, std::milli>(t2 - t1).count();
            same += x == y && s1 == s2;
        }
        std::printf("  %-38s %3d / %zu identical picks\n", c.name, same, rows.size());
        fails += same != int(rows.size());
    }
    const double n = double(rows.size() * std::size(cases));
    std::printf("sample_row %.2f ms/row vs the full sort %.2f ms/row\n", t_new / n, t_old / n);
    if (t_new >= t_old) { std::printf("FAIL: not faster\n"); ++fails; }
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails ? 1 : 0;
}
