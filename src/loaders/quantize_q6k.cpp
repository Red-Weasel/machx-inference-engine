// src/loaders/quantize_q6k.cpp — fp32 → Q6_K encoder. Faithful port of
// ggml-quants.c quantize_row_q6_K_ref + make_qx_quants (rmse_type=1, qw=NULL).
#include "ie/quantize.hpp"

#include <cmath>
#include <cstring>

namespace ie {

namespace {

constexpr float GROUP_MAX_EPS = 1e-15f;

// ggml's fast round-to-nearest (matches the quantizer bit-for-bit).
inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int   i;
    std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
inline int imax(int a, int b) { return a > b ? a : b; }
inline int imin(int a, int b) { return a < b ? a : b; }

// make_qx_quants(n, nmax, x, L, rmse_type=1, qw=NULL): pick the per-group scale
// minimizing weighted L2, scanning ±9 perturbations of -nmax/max.
float make_qx_quants(int n, int nmax, const float* x, int8_t* L) {
    float max = 0, amax = 0;
    for (int i = 0; i < n; ++i) {
        const float ax = std::fabs(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { for (int i = 0; i < n; ++i) L[i] = 0; return 0.f; }

    float iscale = -float(nmax) / max;
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = imax(-nmax, imin(nmax - 1, l));
        L[i] = int8_t(l + nmax);
        const float w = x[i] * x[i];        // rmse_type == 1
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 ? sumlx / suml2 : 0.f;
    float best  = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = imax(-nmax, imin(nmax - 1, l));
            const float w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = int8_t(nmax + imax(-nmax, imin(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best  = scale * sumlx;
        }
    }
    return scale;
}

}  // namespace

void quantize_row_q6_K(const float* x, block_q6_K* y, int64_t k) {
    constexpr int QK_K = 256;
    const int64_t nb = k / QK_K;

    int8_t L[QK_K];
    float  scales[QK_K / 16];

    for (int64_t i = 0; i < nb; ++i) {
        float max_scale = 0, max_abs_scale = 0;
        for (int ib = 0; ib < QK_K / 16; ++ib) {
            const float scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib);
            scales[ib] = scale;
            const float a = std::fabs(scale);
            if (a > max_abs_scale) { max_abs_scale = a; max_scale = scale; }
        }

        if (max_abs_scale < GROUP_MAX_EPS) {
            std::memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = fp32_to_fp16(0.f);
            x += QK_K;
            continue;
        }

        const float iscale = -128.f / max_scale;
        y[i].d = fp32_to_fp16(1.f / iscale);
        for (int ib = 0; ib < QK_K / 16; ++ib)
            y[i].scales[ib] = int8_t(imin(127, nearest_int(iscale * scales[ib])));

        for (int j = 0; j < QK_K / 16; ++j) {
            const float d = fp16_to_fp32(y[i].d) * y[i].scales[j];
            if (!d) continue;
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16 * j + ii] / d);
                l = imax(-32, imin(31, l));
                L[16 * j + ii] = int8_t(l + 32);
            }
        }

        uint8_t* ql = y[i].ql;
        uint8_t* qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l +  0] = uint8_t(q1 | (q3 << 4));
                ql[l + 32] = uint8_t(q2 | (q4 << 4));
                qh[l] = uint8_t((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) |
                                ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6));
            }
            ql += 64;
            qh += 32;
        }
        x += QK_K;
    }
}

}  // namespace ie
