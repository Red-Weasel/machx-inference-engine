// src/loaders/quantize_q4k.cpp — fp32 → Q4_K encoder. Faithful port of
// ggml-quants.c quantize_row_q4_K_ref + make_qkx2_quants + get_scale_min_k4.
#include "ie/quantize.hpp"

#include <cmath>

namespace ie {

namespace {

inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int   i;
    __builtin_memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
inline int   imax(int a, int b)   { return a > b ? a : b; }
inline int   imin(int a, int b)   { return a < b ? a : b; }
inline float fmaxf_(float a, float b) { return a > b ? a : b; }

// Joint (scale,min) search — ggml make_qkx2_quants (use_mad=false here).
float make_qkx2_quants(int n, int nmax, const float* x, const float* weights,
                       uint8_t* L, float* the_min, uint8_t* Laux,
                       float rmin, float rdelta, int nstep) {
    float min = x[0], max = x[0];
    float sum_w = weights[0], sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        const float w = weights[i];
        sum_w += w; sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) { for (int i = 0; i < n; ++i) L[i] = 0; *the_min = -min; return 0.f; }

    float iscale = nmax / (max - min);
    float scale  = 1.f / iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        const int l = nearest_int(iscale * (x[i] - min));
        L[i] = uint8_t(imax(0, imin(nmax, l)));
        const float diff = scale * L[i] + min - x[i];
        best_error += weights[i] * diff * diff;
    }
    if (nstep < 1) { *the_min = -min; return scale; }

    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * (x[i] - min));
            l = imax(0, imin(nmax, l));
            Laux[i] = uint8_t(l);
            const float w = weights[i];
            sum_l += w * l; sum_l2 += w * l * l; sum_xl += w * l * x[i];
        }
        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) { this_min = 0; this_scale = sum_xl / sum_l2; }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                const float diff = this_scale * Laux[i] + this_min - x[i];
                cur_error += weights[i] * diff * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_error = cur_error; scale = this_scale; min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>   4) | ((q[j - 0] >> 6) << 4);
    }
}

}  // namespace

void quantize_row_q4_K(const float* x, block_q4_K* y, int64_t k) {
    constexpr int QK_K = 256;
    const int64_t nb = k / QK_K;

    uint8_t L[QK_K], Laux[32];
    float   weights[32], mins[QK_K / 32], scales[QK_K / 32];

    for (int64_t i = 0; i < nb; ++i) {
        float max_scale = 0, max_min = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
            const float av_x = std::sqrt(sum_x2 / 32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + std::fabs(x[32 * j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                         &mins[j], Laux, -1.f, 0.1f, 20);
            max_scale = fmaxf_(max_scale, scales[j]);
            max_min   = fmaxf_(max_min, mins[j]);
        }

        const float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
        const float inv_min   = max_min   > 0 ? 63.f / max_min   : 0.f;
        for (int j = 0; j < QK_K / 32; ++j) {
            uint8_t ls = uint8_t(imin(63, nearest_int(inv_scale * scales[j])));
            uint8_t lm = uint8_t(imin(63, nearest_int(inv_min * mins[j])));
            if (j < 4) {
                y[i].scales[j]     = ls;
                y[i].scales[j + 4] = lm;
            } else {
                y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j - 4] |= ((ls >> 4) << 6);
                y[i].scales[j - 0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d    = fp32_to_fp16(max_scale / 63.f);
        y[i].dmin = fp32_to_fp16(max_min / 63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K / 32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = fp16_to_fp32(y[i].d) * sc;
            if (!d) continue;
            const float dm = fp16_to_fp32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32 * j + ii] + dm) / d);
                l = imax(0, imin(15, l));
                L[32 * j + ii] = uint8_t(l);
            }
        }

        uint8_t* q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = uint8_t(L[j + l] | (L[j + l + 32] << 4));
            q += 32;
        }
        x += QK_K;
    }
}

}  // namespace ie
