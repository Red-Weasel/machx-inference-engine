// tests/unit/moe_test.cpp — full MoE block forward (single token).
//
// Loads real Qwen3.6-35B-A3B layer-0 MoE tensors from the GGUF, runs the
// 8-step algorithm both on the GPU (orchestrated via the public ops) and on
// the CPU (dequant→fp32 forward, transcribed verbatim from research/06 §12),
// and diffs the two.

#include "ie/dequant_ref.hpp"
#include "ie/gguf.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Y = "\033[33m";
constexpr const char* Z = "\033[0m";

bool report(const char* name, bool ok, const char* extra = "") {
    std::printf("  %s%-48s%s %s%s\n", ok ? G : R, name, Z, ok ? "OK" : "FAIL", extra);
    return ok;
}

sycl::queue make_queue() {
    sycl::device dev;
    bool found = false;
    for (const auto& d : sycl::device::get_devices()) {
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("0xe223") != std::string::npos) {
            dev = d; found = true; break;
        }
    }
    if (!found) for (const auto& d : sycl::device::get_devices()) if (d.is_gpu()) { dev = d; found = true; break; }
    if (!found) { std::fputs("no GPU\n", stderr); std::exit(2); }
    return sycl::queue(dev, sycl::property::queue::enable_profiling{});
}

// ---- CPU reference (research/06 §12) ----
//
// Single token, single layer. Inputs (host fp32):
//   x[H], W_gate_fp32[E,H], W_shg[H], W_g_e[E,I,H] (dequanted gate per expert),
//   W_u_e[E,I,H] (dequanted up per expert), W_d_e[E,H,I] (dequanted down per expert),
//   W_g_sh[I,H], W_u_sh[I,H], W_d_sh[H,I]
//
// Output: y[H] fp32.
//
// Note: real GGUF tensors are 3D [H, I, E] stored axis-0 contiguous. The
// dequant calls write a flat 1D buffer; this function takes accessor lambdas
// to decouple from layout.
void moe_ref(uint32_t H, uint32_t E, uint32_t I_int, uint32_t k,
             const std::vector<float>& x,                // [H]
             const std::vector<float>& W_gate,           // [E, H], row-major: W_gate[e*H + h]
             const std::vector<float>& W_shg,            // [H]
             const std::vector<float>& W_gate_exps,      // [E, I_int, H]   per-expert flat row-major
             const std::vector<float>& W_up_exps,        // [E, I_int, H]
             const std::vector<float>& W_down_exps,      // [E, H, I_int]
             const std::vector<float>& W_gate_sh,        // [I_int, H]
             const std::vector<float>& W_up_sh,          // [I_int, H]
             const std::vector<float>& W_down_sh,        // [H, I_int]
             std::vector<float>& y_out,                  // [H]
             std::vector<int32_t>& idx_out,              // [k]
             std::vector<float>& w_out                   // [k]
            )
{
    // Step 1: router logits (fp32)
    std::vector<float> logits(E);
    for (uint32_t e = 0; e < E; ++e) {
        float s = 0.f;
        for (uint32_t h = 0; h < H; ++h) s += W_gate[e * H + h] * x[h];
        logits[e] = s;
    }
    // Step 2: softmax in fp32
    float m = -std::numeric_limits<float>::infinity();
    for (auto v : logits) m = std::fmax(m, v);
    float sum = 0.f;
    for (auto& v : logits) { v = std::exp(v - m); sum += v; }
    for (auto& v : logits) v /= sum;
    // Step 3: top-k + renorm
    std::vector<int32_t> idx(k);
    std::vector<float>   w(k);
    auto probs = logits;
    for (uint32_t kk = 0; kk < k; ++kk) {
        float best = -1.f; int32_t bi = -1;
        for (uint32_t e = 0; e < E; ++e) if (probs[e] > best) { best = probs[e]; bi = int32_t(e); }
        idx[kk] = bi;
        w[kk]   = best;
        probs[bi] = -2.f;
    }
    float ren = 0.f; for (auto v : w) ren += v;
    for (auto& v : w) v /= ren;
    // Sort ascending by expert id (bit-exact requirement).
    for (uint32_t i = 0; i + 1 < k; ++i)
        for (uint32_t j = 0; j + 1 + i < k; ++j)
            if (idx[j] > idx[j+1]) {
                std::swap(idx[j], idx[j+1]);
                std::swap(w  [j], w  [j+1]);
            }
    idx_out = idx; w_out = w;

    // Step 5: routed branch
    std::vector<float> y(H, 0.f);
    for (uint32_t kk = 0; kk < k; ++kk) {
        const uint32_t e = uint32_t(idx[kk]);
        // gate, up
        std::vector<float> gate(I_int), up(I_int);
        for (uint32_t i = 0; i < I_int; ++i) {
            float gs = 0.f, us = 0.f;
            for (uint32_t h = 0; h < H; ++h) {
                gs += W_gate_exps[(e * I_int + i) * H + h] * x[h];
                us += W_up_exps  [(e * I_int + i) * H + h] * x[h];
            }
            gate[i] = gs; up[i] = us;
        }
        // SwiGLU
        std::vector<float> hh(I_int);
        for (uint32_t i = 0; i < I_int; ++i) {
            const float silu = gate[i] / (1.f + std::exp(-gate[i]));
            hh[i] = silu * up[i];
        }
        // down
        for (uint32_t h = 0; h < H; ++h) {
            float ds = 0.f;
            for (uint32_t i = 0; i < I_int; ++i) ds += W_down_exps[(e * H + h) * I_int + i] * hh[i];
            y[h] += w[kk] * ds;
        }
    }

    // Steps 6-7: shared expert
    std::vector<float> gate_s(I_int), up_s(I_int);
    for (uint32_t i = 0; i < I_int; ++i) {
        float gs = 0.f, us = 0.f;
        for (uint32_t h = 0; h < H; ++h) {
            gs += W_gate_sh[i * H + h] * x[h];
            us += W_up_sh  [i * H + h] * x[h];
        }
        gate_s[i] = gs; up_s[i] = us;
    }
    std::vector<float> hs(I_int);
    for (uint32_t i = 0; i < I_int; ++i) {
        const float silu = gate_s[i] / (1.f + std::exp(-gate_s[i]));
        hs[i] = silu * up_s[i];
    }
    std::vector<float> sh_out(H, 0.f);
    for (uint32_t h = 0; h < H; ++h) {
        float ds = 0.f;
        for (uint32_t i = 0; i < I_int; ++i) ds += W_down_sh[h * I_int + i] * hs[i];
        sh_out[h] = ds;
    }
    float dot = 0.f;
    for (uint32_t h = 0; h < H; ++h) dot += W_shg[h] * x[h];
    const float gate_scalar = 1.0f / (1.0f + std::exp(-dot));

    // Step 8
    for (uint32_t h = 0; h < H; ++h) y[h] += gate_scalar * sh_out[h];

    y_out = std::move(y);
}

// Helper: dequant a 3D quantized tensor [d0, d1, n_experts] (axis-0 contiguous, block-aligned)
// into an [n_experts, d1, d0] flat fp32 buffer (per-expert row-major-ready slices).
template <typename DequantBuf>
std::vector<float> dequant_3d_to_per_expert_rowmajor(
    const ie::GgufTensorInfo* t, DequantBuf dequant_fn) {
    const uint64_t d0 = t->shape[0];
    const uint64_t d1 = t->shape[1];
    const uint64_t E  = t->shape[2];
    const uint64_t n_elem = d0 * d1 * E;
    std::vector<float> linear(n_elem);
    dequant_fn(t->data, n_elem, linear.data());
    // Reorder: GGUF flat order is (e * d1 + n) * d0 + k -> we want
    // per_expert[e * d1 * d0 + n * d0 + k] (which is the same!).
    return linear;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    int test_layer = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gguf"  && i+1 < argc) gguf_path = argv[++i];
        else if (a == "--layer" && i+1 < argc) test_layer = std::atoi(argv[++i]);
    }

    auto q = make_queue();
    std::printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "gguf open: %s\n", err.c_str());
        return 3;
    }

    constexpr uint32_t H = 2048;
    constexpr uint32_t E = 256;
    constexpr uint32_t I_int = 512;
    constexpr uint32_t K_top = 8;

    char buf[64];
    auto T = [&](const char* name) {
        std::snprintf(buf, sizeof(buf), "blk.%d.%s", test_layer, name);
        return g.find_tensor(buf);
    };

    auto* T_gate    = T("ffn_gate_inp.weight");           // F32 [H, E]   (GGUF leading=H contiguous)
    auto* T_shg     = T("ffn_gate_inp_shexp.weight");     // F32 [H]
    auto* T_g_exps  = T("ffn_gate_exps.weight");          // Q4_K [H, I_int, E]
    auto* T_u_exps  = T("ffn_up_exps.weight");            // Q4_K [H, I_int, E]
    auto* T_d_exps  = T("ffn_down_exps.weight");          // Q6_K [I_int, H, E]
    auto* T_g_sh    = T("ffn_gate_shexp.weight");         // Q4_K [H, I_int]
    auto* T_u_sh    = T("ffn_up_shexp.weight");           // Q4_K [H, I_int]
    auto* T_d_sh    = T("ffn_down_shexp.weight");         // Q6_K [I_int, H]
    if (!T_gate || !T_shg || !T_g_exps || !T_u_exps || !T_d_exps ||
        !T_g_sh || !T_u_sh || !T_d_sh) {
        std::printf("  %sFAIL%s required tensors not found in GGUF for layer %d\n", R, Z, test_layer);
        return 4;
    }

    // ---- Random input ----
    std::mt19937 rng(0xCAFE);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> hx_fp32(H);
    for (auto& v : hx_fp32) v = dist(rng);
    std::vector<sycl::half> hx(H);
    for (uint32_t i = 0; i < H; ++i) hx[i] = sycl::half(hx_fp32[i]);

    // ---- Dequant the *needed* expert tensors to fp32 host buffers (CPU ref) ----
    // GGUF flat order for [d0, d1, E] axis-0-contiguous quants:
    //   element (k=axis0, n=axis1, e=axis2) is at flat index e*d1*d0 + n*d0 + k.
    // For our reference, the natural shape is [E, n_per_expert_along_d1, K_axis0].
    //
    // ffn_gate_exps  [H=2048, I=512, E=256]:    expert e's gate weights at offset e*I*H, then n*H + k.
    //                Reference shape we want:  W_gate_exps[e*I*H + i*H + h]  (logical [E, I, H])
    //                Match: flat[e*I*H + i*H + h] = quant_dequant_in_linear_order? Let me think.
    //                GGUF flat: ((axis2=e) * d1 + axis1=i) * d0 + axis0=h  = e*I*H + i*H + h. YES.
    std::vector<float> hW_gate(uint64_t(E) * H);
    {
        // W_gate has shape [H, E], leading dim contiguous. Element (h, e) flat = h + e*H? No:
        // ffn_gate_inp.weight [H, E] non-quantized F32 row-major-with-leading-contiguous means
        // memory is [e][h] in 2D access? Actually for GGUF, the leading dim is the inner (contiguous) one:
        // for a 2D F32 tensor [d0, d1], element (a, b) is at offset a + b * d0.
        // So shape [H=2048, E=256]: element (h, e) at h + e * H.
        // Our reference wants W_gate[e * H + h] — same thing. Good.
        std::memcpy(hW_gate.data(), T_gate->data, hW_gate.size() * sizeof(float));
    }
    std::vector<float> hW_shg(H);
    std::memcpy(hW_shg.data(), T_shg->data, H * sizeof(float));

    std::vector<float> hW_g_exps(uint64_t(E) * I_int * H);
    std::vector<float> hW_u_exps(uint64_t(E) * I_int * H);
    std::vector<float> hW_d_exps(uint64_t(E) * H * I_int);
    ie::ref::dequant_q4_K_buffer(T_g_exps->data, hW_g_exps.size(), hW_g_exps.data());
    ie::ref::dequant_q4_K_buffer(T_u_exps->data, hW_u_exps.size(), hW_u_exps.data());
    ie::ref::dequant_q6_K_buffer(T_d_exps->data, hW_d_exps.size(), hW_d_exps.data());

    std::vector<float> hW_g_sh(uint64_t(I_int) * H);
    std::vector<float> hW_u_sh(uint64_t(I_int) * H);
    std::vector<float> hW_d_sh(uint64_t(H) * I_int);
    ie::ref::dequant_q4_K_buffer(T_g_sh->data, hW_g_sh.size(), hW_g_sh.data());
    ie::ref::dequant_q4_K_buffer(T_u_sh->data, hW_u_sh.size(), hW_u_sh.data());
    ie::ref::dequant_q6_K_buffer(T_d_sh->data, hW_d_sh.size(), hW_d_sh.data());

    std::printf("  Loaded MoE tensors for layer %d (E=%u, H=%u, I=%u, K=%u)\n",
                test_layer, E, H, I_int, K_top);

    // ---- CPU reference ----
    std::vector<float> y_ref;
    std::vector<int32_t> idx_ref;
    std::vector<float>   w_ref;
    auto t0 = std::chrono::steady_clock::now();
    moe_ref(H, E, I_int, K_top,
            hx_fp32, hW_gate, hW_shg,
            hW_g_exps, hW_u_exps, hW_d_exps,
            hW_g_sh, hW_u_sh, hW_d_sh,
            y_ref, idx_ref, w_ref);
    auto t1 = std::chrono::steady_clock::now();
    const double ref_ms = std::chrono::duration<double>(t1 - t0).count() * 1e3;
    std::printf("  CPU reference: top-8 = [");
    for (uint32_t kk = 0; kk < K_top; ++kk) std::printf("%d%s", idx_ref[kk], kk + 1 < K_top ? "," : "");
    std::printf("]  (%.1f ms)\n", ref_ms);

    // ---- GPU orchestration ----
    auto* d_x       = sycl::malloc_device<sycl::half>(H, q);
    auto* d_y       = sycl::malloc_device<sycl::half>(H, q);
    auto* d_W_gate  = sycl::malloc_device<float>(uint64_t(E) * H, q);
    auto* d_W_shg   = sycl::malloc_device<float>(H, q);
    auto* d_W_ge    = sycl::malloc_device(T_g_exps->nbytes, q);
    auto* d_W_ue    = sycl::malloc_device(T_u_exps->nbytes, q);
    auto* d_W_de    = sycl::malloc_device(T_d_exps->nbytes, q);
    auto* d_W_gsh   = sycl::malloc_device(T_g_sh->nbytes, q);
    auto* d_W_ush   = sycl::malloc_device(T_u_sh->nbytes, q);
    auto* d_W_dsh   = sycl::malloc_device(T_d_sh->nbytes, q);
    auto* d_idx     = sycl::malloc_device<int32_t>(K_top, q);
    auto* d_topw    = sycl::malloc_device<sycl::half>(K_top, q);
    auto* d_gate_o  = sycl::malloc_device<sycl::half>(I_int, q);
    auto* d_up_o    = sycl::malloc_device<sycl::half>(I_int, q);
    auto* d_h       = sycl::malloc_device<sycl::half>(I_int, q);
    auto* d_eo      = sycl::malloc_device<sycl::half>(H, q);
    auto* d_sh_out  = sycl::malloc_device<sycl::half>(H, q);
    auto* d_sh_g    = sycl::malloc_device<sycl::half>(1, q);

    q.memcpy(d_x,      hx.data(),     H * sizeof(sycl::half)).wait();
    q.memcpy(d_W_gate, T_gate->data,  uint64_t(E) * H * sizeof(float)).wait();
    q.memcpy(d_W_shg,  T_shg->data,   H * sizeof(float)).wait();
    q.memcpy(d_W_ge,   T_g_exps->data, T_g_exps->nbytes).wait();
    q.memcpy(d_W_ue,   T_u_exps->data, T_u_exps->nbytes).wait();
    q.memcpy(d_W_de,   T_d_exps->data, T_d_exps->nbytes).wait();
    q.memcpy(d_W_gsh,  T_g_sh->data,   T_g_sh->nbytes).wait();
    q.memcpy(d_W_ush,  T_u_sh->data,   T_u_sh->nbytes).wait();
    q.memcpy(d_W_dsh,  T_d_sh->data,   T_d_sh->nbytes).wait();
    q.memset(d_y, 0, H * sizeof(sycl::half)).wait();

    // Per-expert byte strides:
    //   Q4_K [H=2048, I=512, E=256]: bytes per expert = (H*I)/256 * 144 = 4096 * 144 = 589824 B
    //   Q6_K [I=512, H=2048, E=256]: bytes per expert = (I*H)/256 * 210 = 4096 * 210 = 860160 B
    const size_t q4k_bytes_per_expert = uint64_t(H) * I_int / 256 * 144;
    const size_t q6k_bytes_per_expert = uint64_t(I_int) * H / 256 * 210;

    auto t2 = std::chrono::steady_clock::now();

    // Router
    ie::moe_router(q, d_x, d_W_gate, d_idx, d_topw, /*n_tokens=*/1, H, E, K_top).wait();

    // Pull topk back to host so we can issue per-expert kernels in order.
    std::vector<int32_t> hidx(K_top);
    std::vector<sycl::half> htopw(K_top);
    q.memcpy(hidx.data(),  d_idx,  K_top * sizeof(int32_t)).wait();
    q.memcpy(htopw.data(), d_topw, K_top * sizeof(sycl::half)).wait();

    // Routed experts
    for (uint32_t kk = 0; kk < K_top; ++kk) {
        const uint32_t e = uint32_t(hidx[kk]);
        const sycl::half scale = htopw[kk];
        const uint8_t* W_g = static_cast<const uint8_t*>(d_W_ge) + e * q4k_bytes_per_expert;
        const uint8_t* W_u = static_cast<const uint8_t*>(d_W_ue) + e * q4k_bytes_per_expert;
        const uint8_t* W_d = static_cast<const uint8_t*>(d_W_de) + e * q6k_bytes_per_expert;
        ie::gemv_q4_K(q, d_x, W_g, d_gate_o, H, I_int).wait();
        ie::gemv_q4_K(q, d_x, W_u, d_up_o,   H, I_int).wait();
        ie::swiglu(q, d_gate_o, d_up_o, d_h, I_int).wait();
        ie::gemv_q6_K(q, d_h, W_d, d_eo, I_int, H).wait();
        ie::scaled_add(q, d_eo, scale, d_y, H).wait();
    }

    // Shared expert
    ie::gemv_q4_K(q, d_x, d_W_gsh, d_gate_o, H, I_int).wait();
    ie::gemv_q4_K(q, d_x, d_W_ush, d_up_o,   H, I_int).wait();
    ie::swiglu(q, d_gate_o, d_up_o, d_h, I_int).wait();
    ie::gemv_q6_K(q, d_h, d_W_dsh, d_sh_out, I_int, H).wait();
    ie::shared_expert_gate(q, d_x, d_W_shg, d_sh_g, /*n_tokens=*/1, H).wait();

    sycl::half hsh_g[1];
    q.memcpy(hsh_g, d_sh_g, sizeof(sycl::half)).wait();
    ie::scaled_add(q, d_sh_out, hsh_g[0], d_y, H).wait();

    auto t3 = std::chrono::steady_clock::now();
    const double gpu_ms = std::chrono::duration<double>(t3 - t2).count() * 1e3;

    std::vector<sycl::half> hy(H);
    q.memcpy(hy.data(), d_y, H * sizeof(sycl::half)).wait();
    sycl::free(d_x, q); sycl::free(d_y, q); sycl::free(d_W_gate, q); sycl::free(d_W_shg, q);
    sycl::free(d_W_ge, q); sycl::free(d_W_ue, q); sycl::free(d_W_de, q);
    sycl::free(d_W_gsh, q); sycl::free(d_W_ush, q); sycl::free(d_W_dsh, q);
    sycl::free(d_idx, q); sycl::free(d_topw, q);
    sycl::free(d_gate_o, q); sycl::free(d_up_o, q); sycl::free(d_h, q);
    sycl::free(d_eo, q); sycl::free(d_sh_out, q); sycl::free(d_sh_g, q);

    int fails = 0;

    // Compare top-k indices first
    {
        bool ok = true;
        for (uint32_t kk = 0; kk < K_top; ++kk) if (hidx[kk] != idx_ref[kk]) { ok = false; break; }
        char extra[160];
        std::snprintf(extra, sizeof(extra), "  GPU=[%d,%d,%d,%d,%d,%d,%d,%d] ref=[%d,%d,%d,%d,%d,%d,%d,%d]",
                      hidx[0], hidx[1], hidx[2], hidx[3], hidx[4], hidx[5], hidx[6], hidx[7],
                      idx_ref[0], idx_ref[1], idx_ref[2], idx_ref[3], idx_ref[4], idx_ref[5], idx_ref[6], idx_ref[7]);
        if (!report("router top-8 indices match", ok, extra)) ++fails;
    }
    // Compare routing weights
    {
        float max_abs = 0.f;
        for (uint32_t kk = 0; kk < K_top; ++kk) {
            const float d = std::fabs(float(htopw[kk]) - w_ref[kk]);
            if (d > max_abs) max_abs = d;
        }
        char extra[120];
        std::snprintf(extra, sizeof(extra), "  max_abs=%.5f", max_abs);
        if (!report("router top-8 weights match", max_abs <= 1e-3f, extra)) ++fails;
    }
    // Compare full block output
    {
        float max_abs = 0.f, max_rel = 0.f;
        size_t worst = 0;
        for (uint32_t i = 0; i < H; ++i) {
            const float a = y_ref[i];
            const float b = float(hy[i]);
            const float d = std::fabs(a - b);
            if (d > max_abs) { max_abs = d; worst = i; }
            const float scale = std::fmax(std::fabs(a), 1e-3f);
            const float r = d / scale;
            if (r > max_rel) max_rel = r;
        }
        char extra[160];
        std::snprintf(extra, sizeof(extra),
                      "  max_abs=%.5f max_rel=%.5f worst@%zu (ref=%g got=%g)  GPU=%.1f ms / CPU=%.1f ms",
                      max_abs, max_rel, worst, y_ref[worst], float(hy[worst]),
                      gpu_ms, ref_ms);
        // Spec gate is max_abs ≤ 1e-2.
        if (!report("MoE block forward y vs CPU ref (gate ≤ 1e-2 abs)", max_abs <= 1e-2f, extra)) ++fails;
    }

    std::printf("\n%s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails;
}
