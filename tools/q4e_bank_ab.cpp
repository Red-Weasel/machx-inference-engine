// tools/q4e_bank_ab.cpp -- ie-q4e-bank-ab: the two ways the engine serves a non-Q4_K / non-Q5_1 expert bank, side by side on
// the real file's banks, against an fp64 CPU reference (2026-11-07, Qwen 4 readiness). For a layer's gate / up / down bank
// and one expert: (a) the raw A16 bank kernel (qwen4_bank.hpp) on the GGUF bytes, (b) the host F16 copy exactly as
// prepare_bank_views builds it (dequant_ref -> fp32 -> [K, N] transposed -> f16) through gemv_fp16, (c) the fp64 dot of the
// dequant_ref weights. Prints the max relative error of (a) and (b) against (c), and (a) against (b). Decides which path
// deviates when a model-level number moves between IE_Q4E_RAW_KQ=0 and 1. The max rel is floored at 1 % of the largest
// output (so the small outputs' noise shows); the rel-L2 is the aggregate view.
// usage: ie-q4e-bank-ab <model.gguf> [layer=2] [expert=0] [gpu=0]
#include "ie/dequant_ref.hpp"
#include "ie/gguf.hpp"
#include "ie/ops.hpp"
#include "ie/qwen4_bank.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

using namespace ie;

namespace {

std::string dequant(const GgufTensorInfo* t, uint64_t e, uint32_t K, uint32_t N, std::vector<float>& out) {
    // one expert's [N rows of K] slice, GGUF element order
    const uint64_t slice_bytes = t->nbytes / t->shape[2];
    const uint8_t* src = t->data + e * slice_bytes;
    out.assign(uint64_t(K) * N, 0.f);
    switch (t->dtype) {
        case DType::kQ8_0: ref::dequant_q8_0_buffer(src, uint64_t(K) * N, out.data()); break;
        case DType::kQ5_K: ref::dequant_q5_K_buffer(src, uint64_t(K) * N, out.data()); break;
        case DType::kQ6_K: ref::dequant_q6_K_buffer(src, uint64_t(K) * N, out.data()); break;
        case DType::kQ5_0: ref::dequant_q5_0_buffer(src, uint64_t(K) * N, out.data()); break;
        case DType::kQ4_0: ref::dequant_q4_0_buffer(src, uint64_t(K) * N, out.data()); break;
        case DType::kQ4_K: ref::dequant_q4_K_buffer(src, uint64_t(K) * N, out.data()); break;
        case DType::kQ5_1: ref::dequant_q5_1_buffer(src, uint64_t(K) * N, out.data()); break;
        default: return std::string("no reference dequant for ") + std::string(type_name(t->dtype));
    }
    return {};
}

double rel(double got, double want, double floor) { return std::fabs(got - want) / std::max(std::fabs(want), floor); }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [layer=2] [expert=0] [gpu=0]\n", argv[0]); return 2; }
    const uint32_t L = argc > 2 ? uint32_t(std::atoi(argv[2])) : 2;
    const uint64_t e = argc > 3 ? uint64_t(std::atoi(argv[3])) : 0;
    const int gpu = argc > 4 ? std::atoi(argv[4]) : 0;
    GgufReader g;
    if (auto err = g.open(argv[1]); !err.empty()) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    std::vector<sycl::device> gpus;
    for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) gpus.push_back(d);
    if (gpus.empty()) { std::fprintf(stderr, "no GPU\n"); return 1; }
    sycl::queue q(gpus[size_t(gpu) % gpus.size()], sycl::property::queue::in_order{});
    std::printf("ie-q4e-bank-ab on %s, layer %u expert %llu\n", q.get_device().get_info<sycl::info::device::name>().c_str(), L, (unsigned long long)e);
    std::mt19937 rng(7u);
    std::uniform_real_distribution<float> ux(-1.f, 1.f);
    int fails = 0;
    for (const char* which : {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"}) {
        char name[96]; std::snprintf(name, sizeof name, "blk.%u.%s.weight", L, which);
        const GgufTensorInfo* t = g.find_tensor(name);
        if (!t) { std::printf("  %-14s missing\n", which); continue; }
        const uint32_t K = uint32_t(t->shape[0]), N = uint32_t(t->shape[1]);
        BankQ kind;
        const bool kq = qwen4_bank_kind(t->dtype, K, kind);
        std::printf("  %-14s %s [K=%u N=%u E=%llu]%s\n", which, std::string(type_name(t->dtype)).c_str(), K, N,
                    (unsigned long long)t->shape[2], kq ? "  (raw bank kernel)" : "  (not a bank-kernel dtype: reference and F16 copy only)");
        std::vector<float> w;   // [N][K]
        if (auto err = dequant(t, e, K, N, w); !err.empty()) { std::printf("    %s\n", err.c_str()); continue; }
        // activation
        std::vector<sycl::half> xh(K);
        for (uint32_t k = 0; k < K; ++k) xh[k] = sycl::half(ux(rng));
        // (c) fp64 reference
        std::vector<double> ref(N); double maxabs = 0.0;
        for (uint32_t n = 0; n < N; ++n) {
            double s = 0.0;
            for (uint32_t k = 0; k < K; ++k) s += double(float(xh[k])) * double(w[uint64_t(n) * K + k]);
            ref[n] = s; maxabs = std::max(maxabs, std::fabs(s));
        }
        const double floor = 0.01 * maxabs;
        auto* x = static_cast<sycl::half*>(sycl::malloc_shared(K * sizeof(sycl::half), q));
        std::memcpy(x, xh.data(), K * sizeof(sycl::half));
        auto* y = static_cast<sycl::half*>(sycl::malloc_shared(N * sizeof(sycl::half), q));
        // (b) the host F16 copy, transposed [K, N] exactly as prepare_bank_views does
        auto* wf16 = static_cast<sycl::half*>(sycl::malloc_shared(uint64_t(K) * N * sizeof(sycl::half), q));
        for (uint32_t nn = 0; nn < N; ++nn)
            for (uint32_t k = 0; k < K; ++k) wf16[uint64_t(k) * N + nn] = sycl::half(w[uint64_t(nn) * K + k]);
        gemv_fp16(q, x, wf16, y, K, N).wait();
        // rel-L2 = ||got - ref|| / ||ref|| over the N outputs (the aggregate view beside the floored max)
        const auto rel_l2 = [&](const std::vector<double>& got) {
            double num = 0.0, den = 0.0;
            for (uint32_t n = 0; n < N; ++n) { num += (got[n] - ref[n]) * (got[n] - ref[n]); den += ref[n] * ref[n]; }
            return std::sqrt(num / std::max(den, 1e-300));
        };
        double rb = 0.0; std::vector<double> yb(N);
        for (uint32_t n = 0; n < N; ++n) { yb[n] = double(float(y[n])); rb = std::max(rb, rel(yb[n], ref[n], floor)); }
        std::printf("    (b) F16 copy + gemv_fp16   vs fp64 ref: max rel %.3e  rel-L2 %.3e\n", rb, rel_l2(yb));
        if (kq) {
            const uint64_t slice_bytes = t->nbytes / t->shape[2];
            auto* raw = static_cast<uint8_t*>(sycl::malloc_shared(slice_bytes, q));
            std::memcpy(raw, t->data + e * slice_bytes, slice_bytes);
            qwen4_bank_gemv(q, kind, x, raw, y, K, N).wait();
            double ra = 0.0, rab = 0.0; std::vector<double> ya_v(N);
            for (uint32_t n = 0; n < N; ++n) { const double ya = double(float(y[n])); ya_v[n] = ya; ra = std::max(ra, rel(ya, ref[n], floor)); rab = std::max(rab, rel(ya, yb[n], floor)); }
            std::printf("    (a) raw %s bank kernel vs fp64 ref: max rel %.3e  rel-L2 %.3e;  (a) vs (b): max rel %.3e\n", qwen4_bank_name(kind), ra, rel_l2(ya_v), rab);
            if (!(ra <= 2e-3)) ++fails;
            sycl::free(raw, q);
        }
        if (!(rb <= 2e-3)) ++fails;
        sycl::free(x, q); sycl::free(y, q); sycl::free(wf16, q);
    }
    std::printf("%s\n", fails ? "A/B: a path is off (see above)" : "A/B: both paths within 2e-3 of the fp64 reference");
    return fails ? 1 : 0;
}
