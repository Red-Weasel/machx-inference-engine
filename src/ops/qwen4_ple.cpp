// src/ops/qwen4_ple.cpp — qwen4exp PLE n-gram embeddings: host hash/gather +
// device blk.1 PLE layer kernels. Spec: docs/qwen4/12_ple_ngram.md.

#include "ie/qwen4_ple.hpp"

#include "ie/dequant_ref.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

#include <cmath>

namespace ie {

// ---------------------------------------------------------------------------
// 1. Host n-gram hash (spec §2). Streaming form of the EOS segmentation:
//    hist.h1/h2 hold the EFFECTIVE predecessors (a predecessor cut by an EOS
//    is stored as eos). Advancing past an EOS token resets both to eos, which
//    reproduces "predecessors at/before the last EOS strictly before p read
//    as EOS" and the "once EOS, all further EOS" cascade; the token's own EOS
//    enters ctx[0] without cutting its own ctx[1]/ctx[2].
// ---------------------------------------------------------------------------
void qwen4_ple_hash(const PleHashConsts& c,
                    const int32_t* tokens, uint32_t T,
                    PleHistory& hist,
                    uint64_t* rows_out) {
    for (uint32_t p = 0; p < T; ++p) {
        const uint64_t c0 = uint64_t(uint32_t(tokens[p]));
        const uint64_t c1 = uint64_t(hist.h1);
        const uint64_t c2 = uint64_t(hist.h2);
        const uint64_t mixed_2 = (c0 * c.M[0]) ^ (c1 * c.M[1]);
        const uint64_t mixed_3 = mixed_2 ^ (c2 * c.M[2]);
        for (uint32_t h = 0; h < kPleHeads; ++h) {
            const uint64_t mix = (h < 8) ? mixed_2 : mixed_3;
            rows_out[size_t(p) * kPleHeads + h] = mix % c.V[h] + c.O[h];
        }
        if (uint32_t(tokens[p]) == c.eos) {
            hist.h1 = hist.h2 = c.eos;
        } else {
            hist.h2 = hist.h1;
            hist.h1 = uint32_t(tokens[p]);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Host row gather + IQ4_NL dequant (spec §3, E assembly). 160 elems =
//    5 blocks of 32 = 90 B per row; head index slowest in the 2560.
// ---------------------------------------------------------------------------
void qwen4_ple_gather(const void* table,
                      const uint64_t* rows, uint32_t T,
                      float* E_out) {
    const uint8_t* base = static_cast<const uint8_t*>(table);
    for (uint32_t p = 0; p < T; ++p) {
        for (uint32_t h = 0; h < kPleHeads; ++h) {
            const uint64_t r = rows[size_t(p) * kPleHeads + h];
            ref::dequant_iq4_nl_buffer(base + r * kPleRowBytes, kPleHeadDim,
                                       E_out + size_t(p) * kPleD + size_t(h) * kPleHeadDim);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Device blk.1 PLE layer (spec §3, after the two projections). fp32 math
//    throughout (group-RMS, gate dot, signed-sqrt sigmoid, conv accumulation);
//    accurate (non-native) exp/sqrt so the CPU fp64 reference tracks tightly.
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t D  = kPleD;         // 2560
constexpr uint32_t SI = kPleSI;        // 10240
constexpr uint32_t HC = kPleStreams;   // 4
constexpr uint32_t SR = kPleStateRows; // 9

inline float dev_sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

}  // namespace

sycl::event qwen4_ple_layer(sycl::queue& q,
                            const sycl::half* key, const sycl::half* v,
                            float* H,
                            const float* gamma_k, const float* gamma_q,
                            const float* gamma_c, const float* conv_w,
                            float* conv_state,
                            float* ws,
                            uint32_t T, float eps,
                            const std::vector<sycl::event>& deps) {
    // Workspace carve (qwen4_ple_workspace_floats).
    float* u        = ws;                              // [T, SI]
    float* gated    = u + size_t(T) * SI;              // [T, SI]
    float* gate     = gated + size_t(T) * SI;          // [T, HC]
    float* state_in = gate + size_t(T) * HC;           // [SR, SI]

    // Snapshot the incoming conv state: the conv kernel reads the pre-chunk
    // columns from state_in while the state-update kernel rewrites conv_state.
    sycl::event e_cp = q.memcpy(state_in, conv_state,
                                size_t(SR) * SI * sizeof(float), deps);

    constexpr int WG = 256;
    constexpr int SG = 16;
    const float inv_sqrt_d = 1.0f / std::sqrt(float(D));

    // Kernel 1 — per (t, stream j): the fp32 gate scalar.
    //   dot(k_hat_j, q_hat_j) = sum(key*gk * H*gq) / (rms_k * rms_q)
    // computed as three fused reductions over the 2560-group.
    sycl::event e1 = ie::ps(q, "ple_gate", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * HC * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t g   = uint32_t(it.get_group(0));
            const uint32_t t   = g / HC;
            const uint32_t j   = g % HC;
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const size_t base  = size_t(t) * SI + size_t(j) * D;
            float skk = 0.f, sqq = 0.f, sx = 0.f;
            for (uint32_t c = lid; c < D; c += WG) {
                const float kv = float(key[base + c]);
                const float qv = H[base + c];
                skk += kv * kv;
                sqq += qv * qv;
                sx  += (kv * gamma_k[j * D + c]) * (qv * gamma_q[j * D + c]);
            }
            skk = sycl::reduce_over_group(it.get_group(), skk, sycl::plus<float>());
            sqq = sycl::reduce_over_group(it.get_group(), sqq, sycl::plus<float>());
            sx  = sycl::reduce_over_group(it.get_group(), sx,  sycl::plus<float>());
            if (lid == 0) {
                const float rk = sycl::sqrt(skk / float(D) + eps);
                const float rq = sycl::sqrt(sqq / float(D) + eps);
                const float s  = sx / (rk * rq) * inv_sqrt_d;
                const float sg = (s > 0.f) ? 1.f : ((s < 0.f) ? -1.f : 0.f);
                const float a  = sycl::fmax(sycl::fabs(s), 1e-6f);
                gate[size_t(t) * HC + j] = dev_sigmoid(sg * sycl::sqrt(a));
            }
        });
    });

    // Kernel 2 — per (t, j): gated_j = g_j * v (v broadcast across streams),
    // then u = GN(gated, gamma_c) over the same 2560-group.
    sycl::event e2 = ie::ps(q, "ple_gn_c", [&](sycl::handler& h) {
        h.depends_on(e1);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * HC * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t g   = uint32_t(it.get_group(0));
            const uint32_t t   = g / HC;
            const uint32_t j   = g % HC;
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const float gs     = gate[size_t(t) * HC + j];
            const size_t base  = size_t(t) * SI + size_t(j) * D;
            float ss = 0.f;
            for (uint32_t c = lid; c < D; c += WG) {
                const float gv = gs * float(v[size_t(t) * D + c]);
                gated[base + c] = gv;
                ss += gv * gv;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float inv = 1.0f / sycl::sqrt(ss / float(D) + eps);
            for (uint32_t c = lid; c < D; c += WG) {
                const float gv = gs * float(v[size_t(t) * D + c]);
                u[base + c] = gv * gamma_c[j * D + c] * inv;
            }
        });
    });

    // Kernel 3 — depthwise causal conv over the token axis (kernel 4, dil 3;
    // pre-chunk columns from state_in), out = gated + SiLU(y), H += out.
    sycl::event e3 = ie::ps(q, "ple_conv", [&](sycl::handler& h) {
        h.depends_on({e2, e_cp});
        h.parallel_for(sycl::range<1>(size_t(T) * SI), [=](sycl::id<1> id) {
            const size_t idx  = id[0];
            const uint32_t t  = uint32_t(idx / SI);
            const uint32_t c  = uint32_t(idx % SI);
            float y = 0.f;
            for (uint32_t k = 0; k < kPleConvK; ++k) {
                const int p = int(t) - int((kPleConvK - 1 - k) * kPleConvDil);
                const float uv = (p >= 0)
                    ? u[size_t(p) * SI + c]
                    : state_in[size_t(p + int(SR)) * SI + c];
                y += conv_w[size_t(k) * SI + c] * uv;
            }
            const float sy = y * dev_sigmoid(y);
            H[idx] += gated[idx] + sy;
        });
    });

    // Kernel 4 — conv_state <- last SR token-columns of u (mixing in the old
    // state when T < SR), so the next chunk continues exactly.
    sycl::event e4 = ie::ps(q, "ple_state", [&](sycl::handler& h) {
        h.depends_on(e3);
        h.parallel_for(sycl::range<1>(size_t(SR) * SI), [=](sycl::id<1> id) {
            const size_t idx = id[0];
            const uint32_t s = uint32_t(idx / SI);
            const uint32_t c = uint32_t(idx % SI);
            const int p = int(T) - int(SR) + int(s);
            conv_state[idx] = (p >= 0) ? u[size_t(p) * SI + c]
                                       : state_in[size_t(p + int(SR)) * SI + c];
        });
    });

    return e4;
}

}  // namespace ie
