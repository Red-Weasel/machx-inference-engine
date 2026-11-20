// src/model/hy4flash_host.cpp — Hy4-flash host-side arithmetic: rope tables, the n-gram rows' fp8 round trip, the router,
// the compressed indexer. Compiled with -fp-model=precise (src/CMakeLists.txt): rounding to the FP8 grid, the top-k
// comparisons and the divisions here must not be re-associated or turned into reciprocal multiplies.
#include "ie/hy4flash_forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <omp.h>

namespace ie {

void hy4f_rope_row(float theta, const uint32_t mrope[3], uint32_t rope_dim, const int32_t pos3[3], float* cos_out, float* sin_out) {
    // HYV4FlashRotaryEmbedding: inv_freq_i = 1 / theta^(2i / dim) in fp32; the time plane everywhere except
    // i = 1, 4, 7, .. < 3 * mrope[1] (height) and i = 2, 5, 8, .. < 3 * mrope[2] (width).
    const uint32_t n = rope_dim / 2;
    for (uint32_t i = 0; i < n; ++i) {
        const float inv = 1.0f / std::pow(theta, float(2 * i) / float(rope_dim));
        int plane = 0;
        if (i % 3 == 1 && i < 3 * mrope[1]) plane = 1;
        else if (i % 3 == 2 && i < 3 * mrope[2]) plane = 2;
        const float f = float(pos3[plane]) * inv;
        cos_out[i] = std::cos(f); sin_out[i] = std::sin(f);
    }
}

namespace {
// |v| <= 448 -> the nearest FP8 E4M3 magnitude, ties to even (subnormals step 2^-9 below 2^-6).
inline float e4m3_round(float a) {
    if (a < 0.015625f) return std::nearbyint(a * 512.0f) / 512.0f;
    int ex; (void)std::frexp(a, &ex);                 // a = fr * 2^ex, fr in [0.5, 1): the binade is 2^(ex - 1)
    const float quantum = std::ldexp(1.0f, ex - 4);   // 8 steps a binade
    const float r = std::nearbyint(a / quantum) * quantum;
    return r > 448.0f ? 448.0f : r;
}
}  // namespace

void hy4f_fp8_round_trip(float* x, uint32_t n) {
    float amax = 0.f;
    for (uint32_t i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float scale = std::max(amax, 1e-12f) / 448.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float v = x[i] / scale;
        v = std::min(448.0f, std::max(-448.0f, v));
        const float r = e4m3_round(std::fabs(v));
        x[i] = (v < 0.f ? -r : r) * scale;
    }
}

namespace {
// the E4M3 code of a magnitude e4m3_round produced: 0, k * 2^-9 (k < 8), or 2^e * (1 + m / 8) up to 448
inline uint8_t e4m3_code(float r) {
    if (r == 0.f) return 0;
    if (r < 0.015625f) return uint8_t(std::lrintf(r * 512.0f));
    int ex; const float fr = std::frexp(r, &ex);                   // r = fr * 2^ex, fr in [0.5, 1)
    const int m = int(std::lrintf((fr * 2.0f - 1.0f) * 8.0f));
    return uint8_t(((ex - 1 + 7) << 3) | m);
}
inline float e4m3_value(uint8_t c) {
    const int e = (c >> 3) & 15, m = c & 7;
    return e == 0 ? float(m) / 512.0f : std::ldexp(1.0f + float(m) / 8.0f, e - 7);
}
}  // namespace

void hy4f_fp8_encode_row(const float* x, uint32_t n, uint8_t* row) {
    float amax = 0.f;
    for (uint32_t i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float scale = std::max(amax, 1e-12f) / 448.0f;
    std::memcpy(row, &scale, 4);
    for (uint32_t i = 0; i < n; ++i) {
        float v = x[i] / scale;
        v = std::min(448.0f, std::max(-448.0f, v));
        row[4 + i] = uint8_t(e4m3_code(e4m3_round(std::fabs(v))) | (v < 0.f ? 0x80 : 0));
    }
}

void hy4f_fp8_decode_row(const uint8_t* row, uint32_t n, float* x) {
    float scale; std::memcpy(&scale, row, 4);
    for (uint32_t i = 0; i < n; ++i) { const uint8_t c = row[4 + i]; const float r = e4m3_value(uint8_t(c & 0x7F)); x[i] = ((c & 0x80) ? -r : r) * scale; }
}

void hy4f_route(const float* logits, const float* bias, uint32_t E, uint32_t top_k, bool norm, float scale, int32_t* idx, float* w) {
    float s[1024], c[1024]; uint32_t order[1024];
    for (uint32_t e = 0; e < E; ++e) { s[e] = 1.0f / (1.0f + std::exp(-logits[e])); c[e] = s[e] + bias[e]; order[e] = e; }
    std::partial_sort(order, order + top_k, order + E, [&](uint32_t a, uint32_t b) { return c[a] > c[b] || (c[a] == c[b] && a < b); });
    float sum = 0.f;
    for (uint32_t k = 0; k < top_k; ++k) sum += s[order[k]];
    for (uint32_t k = 0; k < top_k; ++k) {
        idx[k] = int32_t(order[k]);
        w[k] = (norm ? s[order[k]] / (sum + 1e-20f) : s[order[k]]) * scale;
    }
}

void Hy4fIndexer::rewind(uint32_t n) {
    if (n < n_tokens()) tok_keys.resize(size_t(n) * hd);
    const uint32_t g = n / ratio;
    if (g < n_groups()) grp_keys.resize(size_t(g) * hd);
}

void Hy4fIndexer::append(const float* raw, const int32_t* pos3_all) {
    // token_key_norm (the model's RMSNorm: x * rsqrt(mean(x^2) + eps) * w)
    double ms = 0;
    for (uint32_t d = 0; d < hd; ++d) ms += double(raw[d]) * raw[d];
    const float r = 1.0f / std::sqrt(float(ms / hd) + eps);
    const size_t at = tok_keys.size(); tok_keys.resize(at + hd);
    for (uint32_t d = 0; d < hd; ++d) tok_keys[at + d] = raw[d] * r * tok_norm[d];
    // a group whose last token just arrived: the gated pooling over its slots, key_norm, the rope of its first position
    while ((uint64_t(n_groups()) + 1) * ratio <= n_tokens()) {
        const uint32_t g = n_groups();
        std::vector<float> val(size_t(ratio) * hd), gat(size_t(ratio) * hd), pooled(hd);
        for (uint32_t s = 0; s < ratio; ++s) {
            const float* k = tok_keys.data() + (size_t(g) * ratio + s) * hd;
            for (uint32_t d = 0; d < hd; ++d) {
                float av = 0.f, ag = 0.f;
                const float* rv = kv.data() + size_t(d) * hd; const float* rg = kg.data() + size_t(d) * hd;
                for (uint32_t j = 0; j < hd; ++j) { av += rv[j] * k[j]; ag += rg[j] * k[j]; }
                val[size_t(s) * hd + d] = av; gat[size_t(s) * hd + d] = ag + ape[size_t(s) * hd + d];
            }
        }
        double pms = 0;
        for (uint32_t d = 0; d < hd; ++d) {
            float mx = gat[d];
            for (uint32_t s = 1; s < ratio; ++s) mx = std::max(mx, gat[size_t(s) * hd + d]);
            float den = 0.f, num = 0.f;
            for (uint32_t s = 0; s < ratio; ++s) { const float e = std::exp(gat[size_t(s) * hd + d] - mx); den += e; num += e * val[size_t(s) * hd + d]; }
            pooled[d] = num / den; pms += double(pooled[d]) * pooled[d];
        }
        const float pr = 1.0f / std::sqrt(float(pms / hd) + eps);
        for (uint32_t d = 0; d < hd; ++d) pooled[d] = pooled[d] * pr * key_norm[d];
        float cs[128], sn[128];
        const int32_t p = int32_t(g * ratio); const int32_t own3[3] = {p, p, p};
        hy4f_rope_row(theta, mrope, rope_dim, pos3_all ? pos3_all + size_t(g) * ratio * 3 : own3, cs, sn);
        const uint32_t pass = hd - rope_dim;
        for (uint32_t i = 0; i < rope_dim / 2; ++i) {
            const float a = pooled[pass + 2 * i], b = pooled[pass + 2 * i + 1];
            pooled[pass + 2 * i] = a * cs[i] - b * sn[i]; pooled[pass + 2 * i + 1] = b * cs[i] + a * sn[i];
        }
        grp_keys.insert(grp_keys.end(), pooled.begin(), pooled.end());
    }
}

bool Hy4fIndexer::recall(const float* q, const float* head_w, uint32_t pos, uint8_t* allow, std::vector<float>& scratch, std::vector<uint32_t>* picked, int threads, const int32_t* pos3_in) const {
    const uint32_t own = pos / ratio;
    if (own + 1 <= group_topk) return false;                // every earlier group fits the budget: dense
    // the query's heads, rotated at its own position; the head weights scaled as the reference scales them
    scratch.resize(size_t(heads) * hd + own);
    float* qr = scratch.data(); float* score = scratch.data() + size_t(heads) * hd;
    float cs[128], sn[128];
    const int32_t pos3[3] = {int32_t(pos), int32_t(pos), int32_t(pos)};
    hy4f_rope_row(theta, mrope, rope_dim, pos3_in ? pos3_in : pos3, cs, sn);
    const uint32_t pass = hd - rope_dim;
    for (uint32_t h = 0; h < heads; ++h) {
        const float* s = q + size_t(h) * hd; float* d = qr + size_t(h) * hd;
        for (uint32_t j = 0; j < pass; ++j) d[j] = s[j];
        for (uint32_t i = 0; i < rope_dim / 2; ++i) {
            const float a = s[pass + 2 * i], b = s[pass + 2 * i + 1];
            d[pass + 2 * i] = a * cs[i] - b * sn[i]; d[pass + 2 * i + 1] = b * cs[i] + a * sn[i];
        }
    }
    const float hs = 1.0f / std::sqrt(float(heads)) / std::sqrt(float(hd));
    // (each group's score is its own serial sum: the team changes who computes it, not a bit of it)
    #pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1 && own >= 256)
    for (uint32_t g = 0; g < own; ++g) {
        const float* k = grp_keys.data() + size_t(g) * hd;
        float acc = 0.f;
        for (uint32_t h = 0; h < heads; ++h) {
            const float* d = qr + size_t(h) * hd;
            float dot = 0.f;
            for (uint32_t j = 0; j < hd; ++j) dot += d[j] * k[j];
            if (dot > 0.f) acc += head_w[h] * hs * dot;
        }
        score[g] = acc;
    }
    std::vector<uint32_t> order(own);
    std::iota(order.begin(), order.end(), 0u);
    const uint32_t keep = group_topk - 1;
    std::partial_sort(order.begin(), order.begin() + keep, order.end(), [&](uint32_t a, uint32_t b) { return score[a] > score[b] || (score[a] == score[b] && a < b); });
    if (picked) { picked->assign(order.begin(), order.begin() + keep); std::sort(picked->begin(), picked->end()); }
    if (allow) {
        std::memset(allow, 0, size_t(pos) + 1);
        for (uint32_t i = 0; i < keep; ++i)
            for (uint32_t s = 0; s < ratio; ++s) allow[size_t(order[i]) * ratio + s] = 1;
        for (uint32_t p = own * ratio; p <= pos; ++p) allow[p] = 1;
    }
    return true;
}

void Hy4fIndexer::select_rows(const float* scores, size_t stride, uint32_t pos0, uint32_t T, std::vector<std::vector<uint32_t>>& picked, int threads) const {
    picked.assign(T, {});
    if (threads < 1) threads = 1;
    const int old_bt = kmp_get_blocktime(); kmp_set_blocktime(1);
    #pragma omp parallel for schedule(dynamic, 4) num_threads(threads) if (T > 1)
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t own = (pos0 + t) / ratio;
        if (own + 1 <= group_topk) continue;                   // within the budget: every key, nothing to pick
        const float* sc = scores + size_t(t) * stride;
        std::vector<uint32_t> order(own);
        std::iota(order.begin(), order.end(), 0u);
        const uint32_t keep = group_topk - 1;
        std::nth_element(order.begin(), order.begin() + keep, order.end(), [&](uint32_t a, uint32_t b) { return sc[a] > sc[b] || (sc[a] == sc[b] && a < b); });
        picked[t].assign(order.begin(), order.begin() + keep);
        std::sort(picked[t].begin(), picked[t].end());
    }
    kmp_set_blocktime(old_bt);
}

void Hy4fIndexer::recall_rows(const float* q, const float* head_w, uint32_t pos0, uint32_t T, uint8_t* allow, size_t stride,
                              std::vector<std::vector<uint32_t>>* picked, int threads, const int32_t* pos3_rows) const {
    if (picked) picked->assign(T, {});
    if (threads < 1) threads = 1;
    const int old_bt = kmp_get_blocktime(); kmp_set_blocktime(1);          // (the leg's rule: idle workers spin 1 ms, not 200)
    if (T == 1) {
        std::vector<float> scr;
        if (!recall(q, head_w, pos0, allow, scr, picked ? &(*picked)[0] : nullptr, threads, pos3_rows) && allow) std::memset(allow, 1, size_t(pos0) + 1);
    } else {
        #pragma omp parallel num_threads(threads)
        {
            std::vector<float> scr;
            #pragma omp for schedule(dynamic, 1)
            for (uint32_t t = 0; t < T; ++t) {
                uint8_t* row = allow ? allow + size_t(t) * stride : nullptr; const uint32_t p = pos0 + t;
                if (!recall(q + size_t(t) * heads * hd, head_w + size_t(t) * heads, p, row, scr, picked ? &(*picked)[t] : nullptr, 1, pos3_rows ? pos3_rows + size_t(t) * 3 : nullptr) && row) std::memset(row, 1, size_t(p) + 1);
            }
        }
    }
    kmp_set_blocktime(old_bt);
}

}  // namespace ie
