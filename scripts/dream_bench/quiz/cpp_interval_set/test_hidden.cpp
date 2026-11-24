#include "solution.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "check failed line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)
using quiz_V_ = std::vector<std::pair<int64_t, int64_t>>;

int main(int argc, char** argv) {
    { IntervalSet s; CHECK(s.intervals().empty() && s.covered() == 0 && !s.contains(0));
      s.add(1, 3); s.add(3, 5); CHECK((s.intervals() == quiz_V_{{1, 5}}));
      s.add(7, 7); s.add(9, 8); CHECK((s.intervals() == quiz_V_{{1, 5}}));
      s.add(-2, 0); CHECK((s.intervals() == quiz_V_{{-2, 0}, {1, 5}}) && s.covered() == 6);
      s.add(0, 1); CHECK((s.intervals() == quiz_V_{{-2, 5}}));
      s.remove(1, 2); CHECK((s.intervals() == quiz_V_{{-2, 1}, {2, 5}}) && s.covered() == 6 && !s.contains(1) && s.contains(2) && !s.contains(5));
      s.remove(-10, 10); CHECK(s.intervals().empty() && s.covered() == 0);
      s.add(-1000000000000000LL, 1000000000000000LL); CHECK(s.covered() == 2000000000000000LL);
      s.remove(5, 5); CHECK(s.covered() == 2000000000000000LL); }
    // random against a brute-force model over [-50, 50)
    std::mt19937 rng(7);
    for (int round = 0; round < 40; ++round) {
        IntervalSet s; std::vector<bool> in(100, false);
        for (int op = 0; op < 300; ++op) {
            int64_t a = int64_t(rng() % 100) - 50, b = std::min<int64_t>(a + int64_t(rng() % 20) - 3, 50);
            bool add = rng() % 3 != 0;
            if (add) s.add(a, b); else s.remove(a, b);
            for (int64_t x = std::max<int64_t>(a, -50); x < std::min<int64_t>(b, 50); ++x) in[x + 50] = add;
            quiz_V_ want; int64_t cov = 0;
            for (int i = 0; i < 100; ++i) if (in[i]) {
                ++cov;
                if (!want.empty() && want.back().second == i - 50) want.back().second = i - 49; else want.push_back({i - 50, i - 49});
            }
            CHECK(s.intervals() == want); CHECK(s.covered() == cov);
            int64_t q = int64_t(rng() % 100) - 50; CHECK(s.contains(q) == bool(in[q + 50]));
        }
    }
    // speed
    auto t0 = std::chrono::steady_clock::now();
    IntervalSet big; int64_t hits = 0;
    for (int i = 0; i < 100000; ++i) { int64_t a = int64_t(rng() % 1000000000) * 4; big.add(a, a + 2); }
    for (int i = 0; i < 100000; ++i) {
        int64_t a = int64_t(rng() % 4000000000LL);
        if (i % 4 == 0) big.remove(a, a + 3); else if (i % 4 == 1) big.add(a, a + 1); else hits += big.contains(a) + (big.covered() > 0);
    }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(big.covered() > 0 && hits >= 0);
    if (dt > 15.0) { std::fprintf(stderr, "200,000 operations took %.1f s (not O(log n))\n", dt); return 1; }
    std::printf("%s\n", argc > 1 ? argv[1] : "");
}
