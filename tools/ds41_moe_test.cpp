// tools/ds41_moe_test.cpp — V4.1 Phase 5a: the routed-expert block on device vs the reference.
//
// The golden comes from tools/ds41_reference/golden_moe.py, which runs DeepSeek's OWN MoE(0)
// with layer 0's real weights (see docs/deepseek41/09). This checks two things:
//   (1) the gate, recomputed here from the bound tensors, reproduces the reference's routing
//       EXACTLY in indices and closely in weights — i.e. sqrtsoftplus, bias-selects-only,
//       renormalise, route_scale are all understood correctly;
//   (2) the engine's existing ds4_experts_forward, fed the reference's own routing, reproduces
//       the reference's routed-expert output.
//
// Splitting (1) from (2) is deliberate: if they were tested together, a routing bug and an
// expert bug could cancel, and a single end-to-end number would not say which was wrong.
#include "ie/deepseek4_ops.hpp"
#include "ie/deepseek41_upload.hpp"
#include "ie/fp8.hpp"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("%s%s%s\n", ok ? "[ ok ] " : "[FAIL] ", what.c_str(),
                detail.empty() ? "" : ("  (" + detail + ")").c_str());
    if (!ok) ++g_fail;
}

template <class T>
std::vector<T> read_bin(const std::string& p, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(1); }
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T)));
    if (size_t(f.gcount()) != n * sizeof(T)) {
        std::fprintf(stderr, "%s: short read\n", p.c_str()); std::exit(1);
    }
    return v;
}

// BF16 -> fp32 (the gate weight ships BF16).
inline float bf16(uint16_t h) { const uint32_t b = uint32_t(h) << 16; float f; std::memcpy(&f, &b, 4); return f; }

struct Err { double max_abs, rel, rms; };
Err compare(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0, s = 0, ref = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(double(a[i]) - double(b[i]));
        m = std::max(m, d); s += d * d; ref = std::max(ref, std::fabs(double(b[i])));
    }
    return {m, ref > 0 ? m / ref : 0.0, std::sqrt(s / double(a.size()))};
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : "/tmp/ds41_golden";

    ie::DeepSeek41Model m;
    if (const auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    const auto& c = m.config();
    const auto& L0 = m.layers()[0];
    const uint32_t H = c.dim, EF = c.moe_inter_dim, E = c.n_routed_experts, TK = c.n_activated_experts;

    // 8 tokens, matching golden_moe.py's default.
    const uint32_t T = 8;
    auto x      = read_bin<float>(gd + "/moe_in.f32", size_t(T) * H);
    auto g_w    = read_bin<float>(gd + "/gate_weights.f32", size_t(T) * TK);
    auto g_i    = read_bin<int32_t>(gd + "/gate_indices.i32", size_t(T) * TK);
    auto g_out  = read_bin<float>(gd + "/moe_routed_out.f32", size_t(T) * H);

    // ---- (1) the gate, recomputed from the bound tensors ---------------------------------
    std::printf("=== gate (host recompute vs the reference's routing) ===\n");
    {
        const auto* gw = L0.gate_w.w;          // BF16 [E, H]
        const auto* gb = L0.gate_bias.w;       // F32  [E]
        const auto* W  = reinterpret_cast<const uint16_t*>(gw->data);
        const auto* B  = reinterpret_cast<const float*>(gb->data);

        size_t idx_bad = 0; double w_err = 0;
        for (uint32_t t = 0; t < T; ++t) {
            std::vector<float> sc(E);
            for (uint32_t e = 0; e < E; ++e) {
                double acc = 0;
                for (uint32_t k = 0; k < H; ++k) acc += double(x[size_t(t) * H + k]) * bf16(W[size_t(e) * H + k]);
                // score_func "sqrtsoftplus": sqrt(softplus(s)), gate_temp 1.0
                sc[e] = std::sqrt(std::log1p(std::exp(-std::fabs(acc))) + std::max(acc, 0.0));
            }
            // the bias steers SELECTION only; the weights come from the unbiased scores
            std::vector<uint32_t> ord(E);
            std::iota(ord.begin(), ord.end(), 0u);
            std::partial_sort(ord.begin(), ord.begin() + TK, ord.end(),
                              [&](uint32_t a, uint32_t b) { return sc[a] + B[a] > sc[b] + B[b]; });
            std::vector<float> wsel(TK);
            double sum = 0;
            for (uint32_t k = 0; k < TK; ++k) { wsel[k] = sc[ord[k]]; sum += wsel[k]; }
            for (uint32_t k = 0; k < TK; ++k) wsel[k] = float(wsel[k] / (sum + 1e-20) * c.route_scale);

            for (uint32_t k = 0; k < TK; ++k) {
                if (int32_t(ord[k]) != g_i[size_t(t) * TK + k]) ++idx_bad;
                w_err = std::max(w_err, std::fabs(double(wsel[k]) - double(g_w[size_t(t) * TK + k])));
            }
        }
        check(idx_bad == 0, "host recompute: top-" + std::to_string(TK) + " indices match the reference exactly",
              std::to_string(idx_bad) + " of " + std::to_string(T * TK) + " differ");
        check(w_err < 1e-5, "host recompute: routing weights within 1e-5",
              "max |diff| " + std::to_string(w_err) + " (bar 1e-5; this is not bit-equality)");
    }

    // ---- (1b) the ENGINE's device router, ds4_router_topk -----------------------------------
    // The Phase 5 gate pointed out that (1) is a host recompute and proves only that the gate
    // SEMANTICS are understood. The kernel the layer loop will actually run is ds4_router_topk
    // (sqrtsoftplus + noaux_tc, ties to the lower index). It has to reproduce the reference too.
    std::printf("\n=== device router (ds4_router_topk) vs the reference's routing ===\n");
    {
        sycl::device dev0;
        for (const auto& p : sycl::platform::get_platforms())
            for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
                if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev0 = d; goto found0; }
        std::fprintf(stderr, "no Arc GPU\n"); return 1;
    found0:
        sycl::queue q0(dev0, sycl::property::queue::in_order{});
        std::vector<float> gw(size_t(E) * H);
        const auto* W = reinterpret_cast<const uint16_t*>(L0.gate_w.w->data);
        for (size_t i = 0; i < gw.size(); ++i) gw[i] = bf16(W[i]);
        float* d_x = sycl::malloc_device<float>(x.size(), q0);
        float* d_w = sycl::malloc_device<float>(gw.size(), q0);
        float* d_b = sycl::malloc_device<float>(E, q0);
        float* d_logits = sycl::malloc_device<float>(size_t(T) * E, q0);
        float* d_wt = sycl::malloc_device<float>(size_t(T) * TK, q0);
        int32_t* d_ix = sycl::malloc_device<int32_t>(size_t(T) * TK, q0);
        q0.memcpy(d_x, x.data(), x.size() * 4); q0.memcpy(d_w, gw.data(), gw.size() * 4);
        q0.memcpy(d_b, L0.gate_bias.w->data, E * 4).wait();
        ie::ds4_router_topk(q0, d_x, d_w, d_b, d_logits, d_wt, d_ix, T, H, E, TK, c.route_scale).wait();
        std::vector<float> wt(size_t(T) * TK); std::vector<int32_t> ix(size_t(T) * TK);
        q0.memcpy(wt.data(), d_wt, wt.size() * 4); q0.memcpy(ix.data(), d_ix, ix.size() * 4).wait();
        // compare as per-token SETS (the engine orders by score, the reference by torch.topk)
        size_t set_bad = 0; double w_err2 = 0;
        for (uint32_t t = 0; t < T; ++t) {
            std::vector<std::pair<int32_t, float>> a, b;
            for (uint32_t k = 0; k < TK; ++k) {
                a.push_back({ix[size_t(t) * TK + k], wt[size_t(t) * TK + k]});
                b.push_back({g_i[size_t(t) * TK + k], g_w[size_t(t) * TK + k]});
            }
            std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
            for (uint32_t k = 0; k < TK; ++k) {
                if (a[k].first != b[k].first) ++set_bad;
                else w_err2 = std::max(w_err2, std::fabs(double(a[k].second) - double(b[k].second)));
            }
        }
        check(set_bad == 0, "device router: top-6 indices match the reference exactly (as sets)",
              std::to_string(set_bad) + " of " + std::to_string(T * TK) + " differ");
        check(w_err2 < 1e-4, "device router: routing weights within 1e-4 of the reference",
              "max |diff| " + std::to_string(w_err2));
        for (void* pp : {(void*)d_x,(void*)d_w,(void*)d_b,(void*)d_logits,(void*)d_wt,(void*)d_ix}) sycl::free(pp, q0);
    }

    // ---- (2) the routed-expert block on device -------------------------------------------
    std::printf("\n=== routed experts on device (reference's own routing) ===\n");
    sycl::device dev;
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; goto found; }
    std::fprintf(stderr, "no Arc GPU\n"); return 1;
found:
    sycl::queue q(dev, sycl::property::queue::in_order{});
    std::printf("       %s\n", dev.get_info<sycl::info::device::name>().c_str());

    ie::DS4ExpertBank b_gate, b_up, b_down;
    if (auto e = ie::ds41_expert_bank_upload(q, L0.exp_w1, H,  EF, E, b_gate); !e.empty()) { std::fprintf(stderr, "w1: %s\n", e.c_str()); return 1; }
    if (auto e = ie::ds41_expert_bank_upload(q, L0.exp_w3, H,  EF, E, b_up);   !e.empty()) { std::fprintf(stderr, "w3: %s\n", e.c_str()); return 1; }
    if (auto e = ie::ds41_expert_bank_upload(q, L0.exp_w2, EF, H,  E, b_down); !e.empty()) { std::fprintf(stderr, "w2: %s\n", e.c_str()); return 1; }
    std::printf("       uploaded 3 x %u experts, %.2f GiB\n", E,
                double(3ull * E * (b_gate.mx_qs_stride + b_gate.mx_e_stride)) / double(1u << 30));

    ie::DS4ExpertWorkspace ws;
    if (auto e = ie::ds4_expert_ws_alloc(q, H, EF, ws); !e.empty()) { std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1; }

    float*   d_x = sycl::malloc_device<float>(size_t(T) * H, q);
    float*   d_y = sycl::malloc_device<float>(size_t(T) * H, q);
    int32_t* d_i = sycl::malloc_device<int32_t>(size_t(T) * TK, q);
    float*   d_w = sycl::malloc_device<float>(size_t(T) * TK, q);
    q.memcpy(d_x, x.data(), x.size() * 4);
    q.memcpy(d_i, g_i.data(), g_i.size() * 4);
    q.memcpy(d_w, g_w.data(), g_w.size() * 4).wait();

    for (bool f16act : {true, false}) {
        if (auto e = ie::ds4_experts_forward(q, b_gate, b_up, b_down, d_x, g_i.data(), g_w.data(),
                                             d_y, T, H, EF, TK, c.swiglu_limit, ws, f16act);
            !e.empty()) { std::fprintf(stderr, "forward: %s\n", e.c_str()); return 1; }
        std::vector<float> got(size_t(T) * H);
        q.memcpy(got.data(), d_y, got.size() * 4).wait();
        const Err er = compare(got, g_out);
        const char* nm = f16act ? "fp16 activations" : "Q8_1 activations";
        std::printf("       %-18s max|diff| %.3e   rel %.3e   rms %.3e\n", nm, er.max_abs, er.rel, er.rms);
        // Tightened after the Phase 5 gate showed a w1/w3 swap passing at 1.90e-2 against the
        // old 2e-2 bar at the old std-0.02 input. At the realistic std-0.13 input the faithful
        // path measures 3.2e-4 (fp16) / 3.1e-3 (Q8_1) and a swap lands at 8.6e-2 / 8.6e-2
        // (Phase 6 gate) -- 43x and 11x outside these bars.
        check(er.rel < (f16act ? 2e-3 : 8e-3),
              std::string(nm) + ": routed output tracks the reference",
              "rel " + std::to_string(er.rel));
    }

    sycl::free(d_x, q); sycl::free(d_y, q); sycl::free(d_i, q); sycl::free(d_w, q);
    ie::ds4_expert_ws_free(q, ws);
    ie::ds4_expert_bank_free(q, b_gate);
    ie::ds4_expert_bank_free(q, b_up);
    ie::ds4_expert_bank_free(q, b_down);

    std::printf("\n%s\n", g_fail ? ("MoE TEST: " + std::to_string(g_fail) + " FAILURE(S)").c_str()
                                 : "MoE TEST: PASS");
    return g_fail ? 1 : 0;
}
