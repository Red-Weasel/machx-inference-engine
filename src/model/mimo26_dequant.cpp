// src/model/mimo26_dequant.cpp — MiMo-V2.6 host dequant references (P1, docs/mimo26/00_PORT_PLAN.md).
// Scalar, exact: every product here is a power of two times a small integer or an fp8 value times
// an fp32 scale, so the P1 gate compares these against llama.cpp's converter bit for bit.
#include "ie/mimo26.hpp"

#include "ie/fp8.hpp"

#include <cstring>

namespace ie {

namespace {

// Read one F32 scale value from the (possibly unaligned) safetensors data region.
inline float f32_at(const uint8_t* base, uint64_t i) {
    float f;
    std::memcpy(&f, base + i * 4, 4);
    return f;
}

std::string shape_str(const std::vector<int64_t>& s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size(); ++i) o += (i ? ", " : "") + std::to_string(s[i]);
    return o + "]";
}

}  // namespace

void mimo26_bf16_to_f32(const uint8_t* src, uint32_t n, float* out) {
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t bits = (uint32_t(src[2 * i]) | (uint32_t(src[2 * i + 1]) << 8)) << 16;
        std::memcpy(out + i, &bits, 4);
    }
}

std::string mimo26_fp8_rows(const Ds41Tensor& t, uint32_t block_n, uint32_t block_k, std::vector<uint8_t>& w8, std::vector<float>& srow) {
    if (!t.w || !t.s) return "fp8 rows: unbound tensor";
    if (t.w->shape.size() != 2 || t.w->dtype_str != "F8_E4M3") return "fp8 rows: '" + t.w->name + "' is not a 2-D F8_E4M3 tensor";
    const int64_t N = t.w->shape[0], K = t.w->shape[1];
    const int64_t SN = (N + block_n - 1) / block_n, SK = (K + block_k - 1) / block_k;
    if (t.s->dtype_str != "F32" || t.s->shape != std::vector<int64_t>{SN, SK})
        return "fp8 rows: scale '" + t.s->name + "' is " + t.s->dtype_str + " " + shape_str(t.s->shape) + ", expected F32 [" +
               std::to_string(SN) + ", " + std::to_string(SK) + "]";
    w8.assign(t.w->data, t.w->data + size_t(N) * size_t(K));
    srow.resize(size_t(N) * size_t(SK));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t kb = 0; kb < SK; ++kb) srow[size_t(n) * size_t(SK) + size_t(kb)] = f32_at(t.s->data, uint64_t(n / block_n) * uint64_t(SK) + uint64_t(kb));
    return {};
}

std::string mimo26_fp8_dequant_f32(const Ds41Tensor& t, uint32_t block_n, uint32_t block_k, std::vector<float>& out) {
    std::vector<uint8_t> w8; std::vector<float> srow;
    if (auto e = mimo26_fp8_rows(t, block_n, block_k, w8, srow); !e.empty()) return e;
    const int64_t N = t.w->shape[0], K = t.w->shape[1];
    mimo26_fp8_rows_to_f32(w8.data(), srow.data(), uint32_t(N), uint32_t(K), block_k, out);
    return {};
}

std::string mimo26_qkv_fp8_rows(const Ds41Tensor& t, uint32_t q_rows, uint32_t k_rows, uint32_t v_rows, uint32_t block,
                                std::vector<uint8_t>& w8, std::vector<float>& srow, uint32_t* tp_out, uint32_t force_tp) {
    if (!t.w || !t.s) return "qkv dequant: unbound tensor";
    const int64_t N = int64_t(q_rows) + k_rows + v_rows;
    if (t.w->shape.size() != 2 || t.w->shape[0] != N || t.w->dtype_str != "F8_E4M3")
        return "qkv dequant: '" + t.w->name + "' is " + t.w->dtype_str + " " + shape_str(t.w->shape) + ", expected F8_E4M3 [" + std::to_string(N) + ", K]";
    const int64_t K = t.w->shape[1];
    const int64_t SK = (K + block - 1) / block;
    if (t.s->dtype_str != "F32" || t.s->shape.size() != 2 || t.s->shape[1] != SK)
        return "qkv dequant: scale '" + t.s->name + "' is " + t.s->dtype_str + " " + shape_str(t.s->shape) + ", expected F32 [rows, " + std::to_string(SK) + "]";

    // Detect the tensor-parallel shard count from the scale plane's row count (candidates 8, 4, 1).
    uint32_t tp = 0;
    int64_t rpr = 0, bpr = 0;
    for (uint32_t cand : {8u, 4u, 1u}) {
        if (N % cand || (force_tp && cand != force_tp)) continue;
        const int64_t r = N / cand, b = (r + block - 1) / block;
        if (t.s->shape[0] == int64_t(cand) * b) { tp = cand; rpr = r; bpr = b; break; }
    }
    if (!tp) return "qkv dequant: scale '" + t.s->name + "' has " + std::to_string(t.s->shape[0]) + " block-rows: no TP in {8, 4, 1} fits";
    if (q_rows % tp || k_rows % tp || v_rows % tp) return "qkv dequant: q/k/v rows are not divisible by tp " + std::to_string(tp);
    if (tp_out) *tp_out = tp;
    const int64_t q_per = q_rows / tp, k_per = k_rows / tp, v_per = v_rows / tp;

    w8.assign(size_t(N) * size_t(K), 0);
    srow.assign(size_t(N) * size_t(SK), 0.f);
    const uint8_t* w = t.w->data;
    for (int64_t n = 0; n < N; ++n) {
        const int64_t rank = n / rpr, rr = n % rpr;
        // rank r's rows are [Q_r (q_per) | K_r (k_per) | V_r (v_per)]; the unsharded order is [Q | K | V]
        int64_t dst;
        if (rr < q_per)              dst = rank * q_per + rr;
        else if (rr < q_per + k_per) dst = int64_t(q_rows) + rank * k_per + (rr - q_per);
        else                         dst = int64_t(q_rows) + k_rows + rank * v_per + (rr - q_per - k_per);
        const uint64_t sbase = uint64_t(rank * bpr + rr / block) * uint64_t(SK);
        std::memcpy(w8.data() + size_t(dst) * size_t(K), w + size_t(n) * size_t(K), size_t(K));
        for (int64_t kb = 0; kb < SK; ++kb) srow[size_t(dst) * size_t(SK) + size_t(kb)] = f32_at(t.s->data, sbase + uint64_t(kb));
    }
    return {};
}

std::string mimo26_qkv_dequant_f32(const Ds41Tensor& t, uint32_t q_rows, uint32_t k_rows, uint32_t v_rows,
                                   uint32_t block, std::vector<float>& out, uint32_t* tp_out, uint32_t force_tp) {
    std::vector<uint8_t> w8; std::vector<float> srow;
    if (auto e = mimo26_qkv_fp8_rows(t, q_rows, k_rows, v_rows, block, w8, srow, tp_out, force_tp); !e.empty()) return e;
    mimo26_fp8_rows_to_f32(w8.data(), srow.data(), q_rows + k_rows + v_rows, uint32_t(t.w->shape[1]), block, out);
    return {};
}

void mimo26_fp8_rows_to_f32(const uint8_t* w8, const float* srow, uint32_t N, uint32_t K, uint32_t block_k, std::vector<float>& out) {
    const uint32_t SK = (K + block_k - 1) / block_k;
    out.assign(size_t(N) * size_t(K), 0.f);
    for (uint32_t n = 0; n < N; ++n) {
        float* o = out.data() + size_t(n) * K;
        const uint8_t* r = w8 + size_t(n) * K;
        const float* sr = srow + size_t(n) * SK;
        for (uint32_t k = 0; k < K; ++k) o[k] = e4m3_to_f32(r[k]) * sr[k / block_k];
    }
}

void mimo26_mxfp4_dequant_ref(const uint8_t* qs, const uint8_t* e, uint32_t N, uint32_t K, float* out) {
    // OCP FP4 E2M1 by code, sign in bit 3; code 8 (-0) decodes to +0 exactly as ggml's kvalues_mxfp4 does.
    static const float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                    0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    const uint32_t KB = K / 2, KS = K / 32;
    for (uint32_t n = 0; n < N; ++n) {
        const uint8_t* row = qs + size_t(n) * KB;
        const uint8_t* sc  = e + size_t(n) * KS;
        float* o = out + size_t(n) * K;
        for (uint32_t k = 0; k < K; ++k) {
            const uint8_t byte = row[k >> 1];
            const uint32_t nib = (k & 1u) ? (byte >> 4) : (byte & 0xFu);
            o[k] = kE2M1[nib] * e8m0_to_f32(sc[k >> 5]);
        }
    }
}

}  // namespace ie
