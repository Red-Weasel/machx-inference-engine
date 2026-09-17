// src/ops/deepseek41_upload.cpp — V4.1 expert bank upload + dense FP8 dequant.
#include "ie/deepseek41_upload.hpp"

#include <cstring>

namespace ie {

std::string ds41_expert_bank_upload(sycl::queue& q, const std::vector<Ds41Tensor>& experts,
                                    uint32_t K, uint32_t N, uint32_t E_take, DS4ExpertBank& out) {
    if (E_take == 0 || E_take > experts.size())
        return "ds41_expert_bank_upload: E_take " + std::to_string(E_take) + " out of range";
    if (K % 32 != 0) return "ds41_expert_bank_upload: K must be a multiple of 32";

    const uint64_t qs_stride = uint64_t(N) * (K / 2);
    const uint64_t e_stride  = uint64_t(N) * (K / 32);
    for (uint32_t i = 0; i < E_take; ++i) {
        const auto& t = experts[i];
        if (!t.w || !t.s) return "expert " + std::to_string(i) + ": weight or scale not bound";
        if (t.w->shape != std::vector<int64_t>{N, K / 2})
            return "expert " + std::to_string(i) + " ('" + t.w->name + "'): weight is not [N, K/2]";
        if (t.s->shape != std::vector<int64_t>{N, K / 32})
            return "expert " + std::to_string(i) + " ('" + t.s->name + "'): scale is not [N, K/32]";
        if (t.w->nbytes != qs_stride || t.s->nbytes != e_stride)
            return "expert " + std::to_string(i) + ": byte count disagrees with its own shape";
    }

    out = DS4ExpertBank{};
    out.dtype = DType::kMXFP4;
    out.K = K; out.N = N; out.E = E_take;
    out.mx_qs_stride = qs_stride;
    out.mx_e_stride  = e_stride;
    out.mx_qs = sycl::malloc_device<uint8_t>(size_t(E_take) * qs_stride, q);
    out.mx_e  = sycl::malloc_device<uint8_t>(size_t(E_take) * e_stride,  q);
    if (!out.mx_qs || !out.mx_e) { ds4_expert_bank_free(q, out); return "device alloc failed"; }

    // Scales go straight across: [N, K/32] E8M0 row-major is exactly mx_e.
    for (uint32_t i = 0; i < E_take; ++i)
        q.memcpy(out.mx_e + uint64_t(i) * e_stride, experts[i].s->data, e_stride);

    // The FP4 planes do NOT go straight across, which is the one thing this port scope got
    // wrong. Both layouts are [N, K/2] bytes with the same 16-code nibble table, but the
    // element order INSIDE each 32-element block differs:
    //
    //   engine (gpt-oss / gemv_mxfp4_soa_f16): byte j holds element j in its low nibble and
    //                                          element j+16 in its high nibble  — interleaved
    //   V4.1   (OCP MX packed-along-K):        byte b holds elements 2b and 2b+1 — sequential
    //
    // Only elements 0 and 31 of a block land in the same place, so reading V4.1 bytes with the
    // engine kernel silently returns a permutation of the right weights. Found by the one-hot
    // GEMV check in tools/ds41_device_test.cpp.
    //
    // The fix is a pure byte shuffle at upload, so every existing tuned MXFP4 kernel keeps
    // working unmodified:
    //     engine_byte[j] = nib(v41[j/2], j&1) | (nib(v41[j/2 + 8], j&1) << 4)
    // It is done in place, one work-item per 16-byte block holding all 16 bytes in registers,
    // so there is no second buffer and no read/write race.
    for (uint32_t i = 0; i < E_take; ++i)
        q.memcpy(out.mx_qs + uint64_t(i) * qs_stride, experts[i].w->data, qs_stride);
    q.wait_and_throw();

    {
        uint8_t* base = out.mx_qs;
        const uint64_t n_blocks = uint64_t(E_take) * qs_stride / 16;
        q.parallel_for(sycl::range<1>(n_blocks), [=](sycl::id<1> gid) {
            uint8_t* p = base + gid[0] * 16;
            uint8_t v[16];
            #pragma unroll
            for (int j = 0; j < 16; ++j) v[j] = p[j];
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int      s2 = j >> 1;
                const uint32_t sh = uint32_t(j & 1) * 4u;   // pick low or high nibble of both
                p[j] = uint8_t(((uint32_t(v[s2])     >> sh) & 0xFu) |
                               (((uint32_t(v[s2 + 8]) >> sh) & 0xFu) << 4));
            }
        }).wait_and_throw();
    }
    return {};
}

sycl::event ds41_dense_dequant_f16(sycl::queue& q, const uint8_t* w, const uint8_t* scale,
                                   uint32_t N, uint32_t K, uint32_t bn, uint32_t bk,
                                   sycl::half* out, const std::vector<sycl::event>& deps) {
    const uint32_t SK = (K + bk - 1) / bk;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(size_t(N) * size_t(K)), [=](sycl::id<1> gid) {
            const uint64_t i = gid[0];
            const uint32_t n = uint32_t(i / K), k = uint32_t(i % K);
            const float v = ds41_e4m3(w[i]) * ds41_e8m0(scale[uint64_t(n / bn) * SK + k / bk]);
            out[i] = sycl::half(v);
        });
    });
}

}  // namespace ie
