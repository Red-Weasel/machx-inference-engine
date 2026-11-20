// tests/unit/spec_accept_test.cpp — speculative sampling's one guarantee: the token it returns follows the model's
// distribution p, whatever the drafter's q is. Monte Carlo over small vocabularies: drafts drawn from q, the step applied,
// the returned tokens counted against p; and the share of drafts kept against sum of min(p, q). Host only.
#include "ie/spec_accept.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
struct Case { const char* name; std::vector<int32_t> p_ids; std::vector<double> p; std::vector<double> q; };   // q over ids 0 .. q.size() - 1

int run(const Case& c, uint64_t seed) {
    std::mt19937_64 g(seed); std::uniform_real_distribution<double> U(0.0, 1.0);
    const size_t V = c.q.size(); const int N = 600000;
    std::vector<double> full_p(V, 0.0);
    for (size_t i = 0; i < c.p_ids.size(); ++i) full_p[size_t(c.p_ids[i])] = c.p[i];
    double want_acc = 0; for (size_t v = 0; v < V; ++v) want_acc += std::min(full_p[v], c.q[v]);
    std::vector<uint64_t> hist(V, 0); uint64_t acc = 0;
    for (int n = 0; n < N; ++n) {
        double u = U(g), cum = 0; int32_t d = int32_t(V) - 1;
        for (size_t v = 0; v < V; ++v) { cum += c.q[v]; if (u < cum) { d = int32_t(v); break; } }
        const ie::SpecStep s = ie::spec_accept(c.p_ids, c.p, d, [&](int32_t id) { return c.q[size_t(id)]; }, [&] { return U(g); });
        ++hist[size_t(s.token)]; acc += s.accepted ? 1 : 0;
    }
    double worst = 0;
    for (size_t v = 0; v < V; ++v) worst = std::max(worst, std::fabs(double(hist[v]) / N - full_p[v]));
    const double got_acc = double(acc) / N;
    const bool ok = worst < 0.004 && std::fabs(got_acc - want_acc) < 0.004;      // (3 sigma of a share near 0.5 at this N is 0.002)
    std::printf("%-44s largest |share - p| %.4f, drafts kept %.4f (sum of min(p, q) = %.4f)  %s\n", c.name, worst, got_acc, want_acc, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
}  // namespace

int main() {
    const std::vector<Case> cases = {
        {"q close to p", {0, 1, 2, 3}, {0.5, 0.3, 0.15, 0.05}, {0.45, 0.35, 0.1, 0.1}},
        {"q far from p", {0, 1, 2, 3}, {0.7, 0.2, 0.05, 0.05}, {0.05, 0.05, 0.2, 0.7}},
        {"p truncated (top-p): q has mass p excludes", {2, 0, 4}, {0.6, 0.3, 0.1}, {0.2, 0.2, 0.2, 0.15, 0.15, 0.1}},
        {"p one token (greedy), q spread", {3}, {1.0}, {0.1, 0.2, 0.3, 0.4}},
        {"q one token p excludes", {0, 1}, {0.5, 0.5}, {0.0, 0.0, 1.0}},
        {"q equal to p", {0, 1, 2}, {0.2, 0.3, 0.5}, {0.2, 0.3, 0.5}},
    };
    int bad = 0; uint64_t seed = 20261119;
    for (const auto& c : cases) bad += run(c, seed++);
    std::printf("spec_accept_test: %d of %zu cases failed\n", bad, cases.size());
    return bad ? 1 : 0;
}
