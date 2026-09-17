// tests/unit/sampler_test.cpp — WG-parallel top-k/top-p sampler validation.
//
// The sampler contract (set 2026-06-09 with the parallel rewrite):
//   temperature → softmax over the FULL vocab → min-p floor → extract
//   candidates in descending probability order (≤ top_k, ≤ K_CAP=1024,
//   stopping when cumulative prob ≥ top_p) → renormalize kept set →
//   inversion draw in DESCENDING-prob order with xorshift64(seed^golden).
//
// Tests use distributions with decisive gaps so fp noise cannot flip
// outcomes, plus a CPU golden for exact pick-matching.

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

bool g_fail = false;
void report(const char* name, bool ok, const char* extra = "") {
    std::printf("  %s%-48s%s %s%s\n", ok ? "\033[32m" : "\033[31m", name,
                "\033[0m", ok ? "OK" : "FAIL", extra);
    if (!ok) g_fail = true;
}

uint64_t xorshift64(uint64_t& s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}
float to_unit_float(uint64_t r) {
    return float(r >> 40) * (1.0f / float(1u << 24));
}

// CPU golden of the kernel contract.
int32_t cpu_sample(std::vector<float> logits, float temperature,
                   uint32_t top_k, float top_p, float min_p, uint64_t seed) {
    const uint32_t vocab = uint32_t(logits.size());
    const uint32_t K_CAP = 1024;
    const uint32_t k_eff = (top_k > 0 && top_k < K_CAP) ? top_k : K_CAP;

    float lmax = -1e30f;
    for (float v : logits) lmax = std::max(lmax, v);
    const float inv_t = 1.0f / temperature;
    double z = 0.0;
    for (float v : logits) z += std::exp(double((v - lmax) * inv_t));
    const float l_floor = (min_p > 0.f) ? lmax + temperature * std::log(min_p)
                                        : -1e30f;
    std::vector<int32_t> kidx; std::vector<float> kp;
    float cum = 0.f;
    for (uint32_t kk = 0; kk < k_eff; ++kk) {
        float best = -1e30f; int32_t bi = -1;
        for (uint32_t i = 0; i < vocab; ++i)
            if (logits[i] >= l_floor && logits[i] > best) { best = logits[i]; bi = int32_t(i); }
        if (bi < 0) break;
        const float p = float(std::exp(double((best - lmax) * inv_t)) / z);
        kidx.push_back(bi); kp.push_back(p);
        logits[bi] = -1e30f;
        cum += p;
        if (top_p < 1.0f && cum >= top_p) break;
    }
    float zk = 0.f; for (float p : kp) zk += p;
    if (kp.empty()) return 0;
    int32_t picked = kidx[0];
    if (zk > 0.f) {
        uint64_t state = seed ^ 0x9E3779B97F4A7C15ull;
        (void)xorshift64(state);
        const float r = to_unit_float(xorshift64(state)) * zk;
        float c = 0.f;
        for (size_t i = 0; i < kp.size(); ++i) {
            c += kp[i];
            if (kp[i] > 0.f && c > r) { picked = kidx[i]; break; }
        }
    }
    return picked;
}

int32_t gpu_sample(sycl::queue& q, const std::vector<float>& logits_f,
                   float temperature, uint32_t top_k, float top_p,
                   float min_p, uint64_t seed) {
    const uint32_t vocab = uint32_t(logits_f.size());
    std::vector<sycl::half> hl(vocab);
    for (uint32_t i = 0; i < vocab; ++i) hl[i] = sycl::half(logits_f[i]);
    auto* dl = sycl::malloc_device<sycl::half>(vocab, q);
    auto* dout = sycl::malloc_device<int32_t>(1, q);
    q.memcpy(dl, hl.data(), vocab * sizeof(sycl::half)).wait();
    ie::sample_softmax_topk_topp(q, dl, dout, vocab, temperature, top_k,
                                 top_p, min_p, seed).wait();
    int32_t out = -1;
    q.memcpy(&out, dout, sizeof(int32_t)).wait();
    sycl::free(dl, q); sycl::free(dout, q);
    return out;
}

// Quantize through fp16 so the CPU golden sees the same values the GPU does.
std::vector<float> as_fp16(const std::vector<float>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = float(sycl::half(v[i]));
    return o;
}

}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property::queue::in_order{}};
    std::printf("Device: %s\n\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    // Synthetic vocab with decisive structure: token 7 dominant, token 42
    // second, token 99 third, the rest a flat low tail.
    auto make_logits = [&](uint32_t vocab) {
        std::vector<float> l(vocab, -4.0f);
        uint64_t s = 12345;
        for (uint32_t i = 0; i < vocab; ++i)
            l[i] += 0.25f * to_unit_float(xorshift64(s));
        l[7]  = 8.0f;
        l[42] = 6.0f;
        l[99] = 5.0f;
        return as_fp16(l);
    };

    for (uint32_t vocab : {1024u, 248320u}) {
        char tag[96];
        const auto L = make_logits(vocab);

        // 1. top_k=1 must always pick the argmax regardless of seed.
        {
            bool ok = true;
            for (uint64_t s = 1; s <= 16; ++s)
                ok &= (gpu_sample(q, L, 0.8f, 1, 1.0f, 0.f, s) == 7);
            std::snprintf(tag, sizeof(tag), "top_k=1 == argmax (vocab=%u)", vocab);
            report(tag, ok);
        }
        // 2. min_p=0.5 leaves only the dominant token.
        {
            bool ok = true;
            for (uint64_t s = 1; s <= 16; ++s)
                ok &= (gpu_sample(q, L, 1.0f, 0, 1.0f, 0.5f, s) == 7);
            std::snprintf(tag, sizeof(tag), "min_p=0.5 isolates argmax (vocab=%u)", vocab);
            report(tag, ok);
        }
        // 3. top_k=3: every draw lands in {7, 42, 99}.
        {
            bool ok = true;
            for (uint64_t s = 1; s <= 64; ++s) {
                const int32_t t = gpu_sample(q, L, 1.5f, 3, 1.0f, 0.f, s);
                ok &= (t == 7 || t == 42 || t == 99);
            }
            std::snprintf(tag, sizeof(tag), "top_k=3 stays in top set (vocab=%u)", vocab);
            report(tag, ok);
        }
        // 4. Exact pick match vs CPU golden across seeds and configs.
        {
            bool ok = true;
            int mismatches = 0;
            struct Cfg { float t; uint32_t k; float p; float mp; };
            const Cfg cfgs[] = {{0.8f, 40, 0.95f, 0.f},
                                {1.2f, 0,  0.90f, 0.f},
                                {1.0f, 8,  1.00f, 0.05f},
                                {2.0f, 100, 0.99f, 0.f}};
            for (const auto& c : cfgs) {
                for (uint64_t s = 1; s <= 24; ++s) {
                    const int32_t g = gpu_sample(q, L, c.t, c.k, c.p, c.mp, s);
                    const int32_t r = cpu_sample(L, c.t, c.k, c.p, c.mp, s);
                    if (g != r) { ok = false; ++mismatches; }
                }
            }
            char extra[48];
            std::snprintf(extra, sizeof(extra), "  (%d mismatches/96)", mismatches);
            std::snprintf(tag, sizeof(tag), "exact match vs CPU golden (vocab=%u)", vocab);
            report(tag, ok, ok ? "" : extra);
        }
        // 5. Determinism: same seed → same pick.
        {
            const int32_t a = gpu_sample(q, L, 1.0f, 40, 0.9f, 0.f, 777);
            const int32_t b = gpu_sample(q, L, 1.0f, 40, 0.9f, 0.f, 777);
            std::snprintf(tag, sizeof(tag), "deterministic per seed (vocab=%u)", vocab);
            report(tag, a == b);
        }
        // 6. temperature<=0 falls back to argmax.
        {
            std::snprintf(tag, sizeof(tag), "T=0 -> argmax (vocab=%u)", vocab);
            report(tag, gpu_sample(q, L, 0.0f, 0, 1.0f, 0.f, 5) == 7);
        }
    }

    std::printf("\n%s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
