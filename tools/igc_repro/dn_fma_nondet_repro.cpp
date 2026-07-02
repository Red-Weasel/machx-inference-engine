// dn_fma_nondet_repro.cpp — standalone reproducer for run-to-run
// non-deterministic FMA results in a chunked-scan SYCL kernel on
// Intel Arc Pro B70 (BMG-G31, Xe2-HPG).
//
// SELF-CONTAINED: no external headers beyond <sycl/sycl.hpp> + libc++.
// No ESIMD, no block loads, no extensions — plain SYCL 2020 only.
//
// Build (exactly — JIT path, same as production engine; AOT codegen differences are signal):
//   icpx -fsycl -O2 tools/igc_repro/dn_fma_nondet_repro.cpp -o /tmp/dn_repro
// Run:
//   /tmp/dn_repro                 # defaults: --iters 200 --T 512 --chain 4
//   /tmp/dn_repro --T 1024        # structured variation: longer chunk
//   /tmp/dn_repro --wide          # structured variation: wide exponent range
//
// What it does
// ------------
// The kernel below is a structural clone of a gated-DeltaNet chunked
// recurrence ("dn_recurrence") from a production LLM inference engine:
//   * one work-group of 128 work-items per (batch, head); 32 heads
//   * each lane owns one column of a 128x128 fp32 state matrix in
//     private memory (128 fp32 registers per lane)
//   * a sequential t-loop over the chunk (T=512 default); per step:
//     SLM broadcast of q_t/k_t/v_t + 2 scalars, then four unrolled
//     128-deep fp32 FMA chains (decay, dot, rank-1 update, dot)
//   * group barriers between SLM fill and use, exactly as production
//
// Inputs are synthetic (xorshift64-seeded, realistic distributions:
// L2-normalized q/k, alpha=exp(g) in ~(0.6,1], beta=sigmoid in (0,1)).
// A warmup pass from state=0 produces a realistic non-zero reference
// state.  Then each "iteration" performs:
//   state <- reference state (device memcpy)
//   CHAIN(=4) back-to-back kernel launches carrying state forward
//   (mimics chunked long prefill: 4 x 512 = 2048 tokens)
//   FNV-1a hash of out[] and state[] on the host
// All inputs are bit-identical for every iteration, so every iteration
// MUST produce bit-identical buffers.  On first hash mismatch prints:
//   DIVERGED at iter N (hash 0x... vs 0x...)
// otherwise:
//   <iters>/<iters> identical
//
// Context: in the full engine the same kernel produces run-to-run
// different results on identical inputs (first divergence between
// ~7 and ~500 launches), after a 28-step bisect ruled out barriers,
// SLM races, USM/cache coherency, atomics/fences, and compiler
// reordering.  This file is the minimal standalone candidate.

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ===== xorshift64 (deterministic synthetic inputs) =====================
struct Xorshift64 {
    uint64_t s;
    explicit Xorshift64(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    uint64_t next() {
        uint64_t x = s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return s = x;
    }
    // uniform in [0, 1)
    float uf() { return float((next() >> 40) * (1.0 / 16777216.0)); }
    // uniform in [-1, 1)
    float sf() { return 2.f * uf() - 1.f; }
};

// ===== FNV-1a 64 ========================================================
uint64_t fnv1a(const void* data, size_t bytes,
               uint64_t h = 0xcbf29ce484222325ULL) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        h ^= uint64_t(p[i]);
        h *= 0x100000001b3ULL;
    }
    return h;
}

constexpr int      K_DIM  = 128;  // k_head_dim = v_head_dim = 128
constexpr uint32_t N_HEAD = 32;   // production v-head count
constexpr uint32_t B      = 1;

// ===== the kernel under test ===========================================
// Structural clone of the production deltanet_recurrence (plain SYCL).
sycl::event run_recurrence(sycl::queue& q,
                           const float* q_in, const float* k_in,
                           const float* v_in, const float* g_in,
                           const float* beta_in,
                           float* state, float* out, uint32_t T) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> q_slm(K_DIM, h);
        sycl::local_accessor<float, 1> k_slm(K_DIM, h);
        sycl::local_accessor<float, 1> v_slm(K_DIM, h);
        sycl::local_accessor<float, 1> sc_slm(2, h);

        const uint32_t WG_ITEMS = K_DIM;
        // outer dim = head index (B*N_HEAD WGs of 1x128); inner dim is the 128-lane WG
        h.parallel_for(
            sycl::nd_range<2>({uint64_t(B) * N_HEAD, WG_ITEMS}, {1, WG_ITEMS}),
            [=](sycl::nd_item<2> it) {
            const uint32_t bh = uint32_t(it.get_group(0));
            const uint32_t b  = bh / N_HEAD;
            const uint32_t hh = bh % N_HEAD;
            const uint32_t vv = uint32_t(it.get_local_id(1));

            float S_col[K_DIM];
            const uint64_t state_base =
                (uint64_t(b) * N_HEAD + hh) * K_DIM * K_DIM;
            #pragma unroll
            for (int kk = 0; kk < K_DIM; ++kk)
                S_col[kk] = state[state_base + uint64_t(kk) * K_DIM + vv];

            for (uint32_t t = 0; t < T; ++t) {
                const uint64_t qkv_row = (uint64_t(b) * T + t) * N_HEAD + hh;
                q_slm[vv] = q_in[qkv_row * K_DIM + vv];
                k_slm[vv] = k_in[qkv_row * K_DIM + vv];
                v_slm[vv] = v_in[qkv_row * K_DIM + vv];
                if (vv == 0) {
                    sc_slm[0] = sycl::native::exp(g_in[qkv_row]);
                    sc_slm[1] = beta_in[qkv_row];
                }
                sycl::group_barrier(it.get_group());

                const float alpha  = sc_slm[0];
                const float beta_t = sc_slm[1];
                const float v_t    = v_slm[vv];

                // decay
                #pragma unroll
                for (int kk = 0; kk < K_DIM; ++kk) S_col[kk] *= alpha;

                // kv_mem = <S_col, k_t>
                float kv_mem = 0.f;
                #pragma unroll
                for (int kk = 0; kk < K_DIM; ++kk)
                    kv_mem += S_col[kk] * k_slm[kk];

                const float delta = (v_t - kv_mem) * beta_t;

                // rank-1 update
                #pragma unroll
                for (int kk = 0; kk < K_DIM; ++kk)
                    S_col[kk] += k_slm[kk] * delta;

                // out[vv] = <S_col, q_t>
                float out_v = 0.f;
                #pragma unroll
                for (int kk = 0; kk < K_DIM; ++kk)
                    out_v += S_col[kk] * q_slm[kk];

                out[(uint64_t(b) * T + t) * N_HEAD * K_DIM +
                    hh * K_DIM + vv] = out_v;

                sycl::group_barrier(it.get_group());
            }

            #pragma unroll
            for (int kk = 0; kk < K_DIM; ++kk)
                state[state_base + uint64_t(kk) * K_DIM + vv] = S_col[kk];
        });
    });
}

}  // namespace

int main(int argc, char** argv) {
    uint32_t iters = 200;
    uint32_t T     = 512;
    uint32_t chain = 4;     // back-to-back launches per iteration
    bool     wide  = false; // wide exponent range on v
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--T"     && i + 1 < argc) T     = std::atoi(argv[++i]);
        else if (a == "--chain" && i + 1 < argc) chain = std::atoi(argv[++i]);
        else if (a == "--wide")                  wide  = true;
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }

    // Pick the GPU with the most compute units (prefers the discrete
    // B70 over an iGPU on multi-GPU hosts).
    sycl::device dev;
    {
        uint32_t best_cu = 0;
        bool found = false;
        for (const auto& p : sycl::platform::get_platforms()) {
            for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) {
                const uint32_t cu =
                    d.get_info<sycl::info::device::max_compute_units>();
                if (!found || cu > best_cu) { dev = d; best_cu = cu; found = true; }
            }
        }
        if (!found) { std::fprintf(stderr, "no GPU device found\n"); return 2; }
    }
    sycl::queue q(dev, sycl::property::queue::in_order{});
    std::printf("device : %s\n",
                dev.get_info<sycl::info::device::name>().c_str());
    std::printf("driver : %s\n",
                dev.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("shape  : B=%u T=%u heads=%u head_dim=%d  iters=%u chain=%u%s\n",
                B, T, N_HEAD, K_DIM, iters, chain, wide ? "  [wide]" : "");

    // ===== synthetic inputs (host, deterministic) ======================
    const uint64_t rows    = uint64_t(B) * T * N_HEAD;       // per-(t,h)
    const uint64_t qkv_n   = rows * K_DIM;
    const uint64_t state_n = uint64_t(B) * N_HEAD * K_DIM * K_DIM;

    std::vector<float> h_q(qkv_n), h_k(qkv_n), h_v(qkv_n);
    std::vector<float> h_g(rows), h_beta(rows);
    Xorshift64 rng(0x00b70b70dead5eedULL);

    const float qscale = 1.0f / std::sqrt(float(K_DIM));
    for (uint64_t r = 0; r < rows; ++r) {
        // q, k: L2-normalized 128-vectors (production l2_norm_scale).
        for (int pass = 0; pass < 2; ++pass) {
            float* dst = (pass == 0 ? h_q.data() : h_k.data()) + r * K_DIM;
            double ss = 0.0;
            for (int i = 0; i < K_DIM; ++i) {
                // approx-gaussian: sum of 3 uniforms
                const float g3 = rng.sf() + rng.sf() + rng.sf();
                dst[i] = g3;
                ss += double(g3) * double(g3);
            }
            const float r2 = float(1.0 / std::sqrt(ss + 1e-6)) *
                             (pass == 0 ? qscale : 1.0f);
            for (int i = 0; i < K_DIM; ++i) dst[i] *= r2;
        }
        // v: fp16-roundtripped activations; optionally wide exponents.
        float vscale = 1.0f;
        if (wide) vscale = std::ldexp(1.0f, int(rng.uf() * 24.f) - 12);
        for (int i = 0; i < K_DIM; ++i) {
            const float g3 = (rng.sf() + rng.sf() + rng.sf()) * 1.5f * vscale;
            h_v[r * K_DIM + i] = g3;
        }
        // g: A_log * softplus(a) with A in [-0.12, -0.02] -> alpha=exp(g)
        // mostly in (0.6, 1] (production decay-gate distribution).
        const float a  = 4.f * rng.sf();
        const float sp = (a > 0.f ? a : 0.f) + std::log1p(std::exp(-std::fabs(a)));
        h_g[r] = -(0.02f + 0.10f * rng.uf()) * sp;
        // beta: sigmoid(b), b in (-3, 3)
        h_beta[r] = 1.f / (1.f + std::exp(-3.f * rng.sf()));
    }

    // ===== device buffers ==============================================
    float* d_q     = sycl::malloc_device<float>(qkv_n, q);
    float* d_k     = sycl::malloc_device<float>(qkv_n, q);
    float* d_v     = sycl::malloc_device<float>(qkv_n, q);
    float* d_g     = sycl::malloc_device<float>(rows, q);
    float* d_beta  = sycl::malloc_device<float>(rows, q);
    float* d_state = sycl::malloc_device<float>(state_n, q);
    float* d_sref  = sycl::malloc_device<float>(state_n, q);
    float* d_out   = sycl::malloc_device<float>(qkv_n, q);
    if (!d_q || !d_k || !d_v || !d_g || !d_beta || !d_state || !d_sref || !d_out) {
        std::fprintf(stderr, "device alloc failed\n"); return 2;
    }
    q.memcpy(d_q, h_q.data(), qkv_n * sizeof(float));
    q.memcpy(d_k, h_k.data(), qkv_n * sizeof(float));
    q.memcpy(d_v, h_v.data(), qkv_n * sizeof(float));
    q.memcpy(d_g, h_g.data(), rows * sizeof(float));
    q.memcpy(d_beta, h_beta.data(), rows * sizeof(float));
    q.memset(d_state, 0, state_n * sizeof(float));
    q.wait();

    // Warmup from state=0 -> realistic non-zero reference state.
    run_recurrence(q, d_q, d_k, d_v, d_g, d_beta, d_state, d_out, T);
    q.memcpy(d_sref, d_state, state_n * sizeof(float));
    q.wait();

    // ===== iteration loop ==============================================
    std::vector<float> h_out(qkv_n), h_state(state_n);
    uint64_t hash0 = 0;
    bool diverged = false;

    for (uint32_t it = 0; it < iters; ++it) {
        q.memcpy(d_state, d_sref, state_n * sizeof(float));
        for (uint32_t c = 0; c < chain; ++c)
            run_recurrence(q, d_q, d_k, d_v, d_g, d_beta, d_state, d_out, T);
        q.memcpy(h_out.data(), d_out, qkv_n * sizeof(float));
        q.memcpy(h_state.data(), d_state, state_n * sizeof(float));
        q.wait();

        uint64_t h = fnv1a(h_out.data(), qkv_n * sizeof(float));
        h = fnv1a(h_state.data(), state_n * sizeof(float), h);

        if (it == 0) {
            hash0 = h;
            std::printf("iter 0 reference hash: 0x%016llx\n",
                        (unsigned long long)h);
        } else if (h != hash0) {
            std::printf("DIVERGED at iter %u (hash 0x%016llx vs 0x%016llx)\n",
                        it, (unsigned long long)h,
                        (unsigned long long)hash0);
            diverged = true;
            break;
        }
    }

    if (!diverged)
        std::printf("%u/%u identical\n", iters, iters);

    sycl::free(d_q, q);     sycl::free(d_k, q);    sycl::free(d_v, q);
    sycl::free(d_g, q);     sycl::free(d_beta, q);
    sycl::free(d_state, q); sycl::free(d_sref, q); sycl::free(d_out, q);
    return diverged ? 1 : 0;
}
