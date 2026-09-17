// tests/awq_dequant_test.cpp — P3e Task 1. Bit-exact host check of the AWQ
// nibble interleave + (q-z)*s dequant against hand-computed weights, plus
// quantization_config parsing. No GPU, no external model.
#undef NDEBUG
#include "ie/awq.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

// Pack 8 logical nibbles into one int32 using the AWQ gemm interleave.
int32_t pack(const int q[8]) {
    uint32_t p = 0;
    for (int j = 0; j < 8; ++j)
        p |= (uint32_t(q[j]) & 0xFu) << ie::kAwqShift[j];
    return int32_t(p);
}

// Pack 8 nibbles in natural order (GPTQ): nibble j at bit j*4.
int32_t pack_nat(const int q[8]) {
    uint32_t p = 0;
    for (int j = 0; j < 8; ++j) p |= (uint32_t(q[j]) & 0xFu) << (j * 4);
    return int32_t(p);
}

}  // namespace

int main() {
    // in=4, out=8, group_size=2 → 2 groups, packed_cols=1.
    const int64_t in = 4, out = 8, G = 2;

    int qw[4][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {7, 6, 5, 4, 3, 2, 1, 0},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {15, 14, 13, 12, 11, 10, 9, 8},
    };
    int qz[2][8] = {
        {8, 8, 8, 8, 8, 8, 8, 8},   // group 0 (rows 0,1)
        {0, 1, 2, 3, 4, 5, 6, 7},   // group 1 (rows 2,3)
    };
    const float sf[2] = {1.0f, 0.5f};  // per-group scale (exact in fp16)

    int32_t qweight[4], qzeros[2];
    uint16_t scales[2 * 8];
    for (int r = 0; r < 4; ++r) qweight[r] = pack(qw[r]);
    for (int g = 0; g < 2; ++g) {
        qzeros[g] = pack(qz[g]);
        for (int o = 0; o < 8; ++o) scales[g * 8 + o] = ie::fp32_to_fp16(sf[g]);
    }

    uint16_t w[4 * 8];
    const std::string err =
        ie::awq_dequant_to_fp16(qweight, qzeros, scales, in, out, G, w);
    assert(err.empty());

    // Expected: (q - z) * s, exact in fp16.
    for (int r = 0; r < 4; ++r) {
        const int g = r / int(G);
        for (int o = 0; o < 8; ++o) {
            const float expect = float(qw[r][o] - qz[g][o]) * sf[g];
            const float got    = ie::fp16_to_fp32(w[r * 8 + o]);
            if (got != expect) {
                std::fprintf(stderr, "mismatch r=%d o=%d: got %.4f expect %.4f\n",
                             r, o, got, expect);
                return 1;
            }
        }
    }

    // Interleave sanity: logical column 1 lives at bit 16, NOT bit 4.
    assert(ie::kAwqShift[1] == 16 && ie::kAwqShift[2] == 4);

    // config.json parsing.
    ie::AwqConfig cfg;
    assert(ie::parse_awq_config(
        R"({"quantization_config":{"quant_method":"awq","bits":4,"group_size":128,"version":"GEMM","zero_point":true}})",
        cfg).empty());
    assert(cfg.group_size == 128 && cfg.version == "gemm" && cfg.bits == 4);

    // Unsupported pack order / bits must error clearly, not silently mis-dequant.
    ie::AwqConfig bad;
    assert(!ie::parse_awq_config(
        R"({"quantization_config":{"quant_method":"awq","bits":4,"version":"gemv"}})", bad).empty());
    assert(!ie::parse_awq_config(
        R"({"quantization_config":{"quant_method":"awq","bits":8}})", bad).empty());
    assert(!ie::parse_awq_config(
        R"({"quantization_config":{"quant_method":"foo","bits":4}})", bad).empty());

    // GPTQ config parses (no AWQ 'version' requirement); reads desc_act/sym.
    ie::AwqConfig gptq;
    assert(ie::parse_awq_config(
        R"({"quantization_config":{"quant_method":"gptq","bits":4,"group_size":128,"desc_act":false,"sym":true}})",
        gptq).empty());
    assert(gptq.quant_method == "gptq" && gptq.group_size == 128 &&
           gptq.desc_act == false && gptq.bits == 4);

    // ---- GPTQ (classic AutoGPTQ): input-axis weight pack, output-axis zero
    // pack, natural order, w = s*(q-(z+1)), optional g_idx. ----
    {
        const int64_t gin = 8, gout = 8, gG = 4;   // 2 groups
        int gqw[8][8];
        for (int i = 0; i < 8; ++i)
            for (int o = 0; o < 8; ++o) gqw[i][o] = (i + 2 * o) & 0xF;
        const int gqz[2][8] = {
            {0, 0, 0, 0, 0, 0, 0, 0},   // group 0 → z+1 = 1
            {1, 1, 1, 1, 1, 1, 1, 1},   // group 1 → z+1 = 2
        };
        const float gsf[2] = {1.0f, 0.5f};

        // qweight [in/8=1][out=8]: column o packs the 8 input rows' nibbles.
        int32_t gqweight[8];
        for (int o = 0; o < 8; ++o) {
            int col[8];
            for (int i = 0; i < 8; ++i) col[i] = gqw[i][o];
            gqweight[o] = pack_nat(col);
        }
        // qzeros [groups=2][out/8=1]: row g packs the 8 output zeros.
        int32_t gqzeros[2];
        uint16_t gscales[2 * 8];
        for (int g = 0; g < 2; ++g) {
            gqzeros[g] = pack_nat(gqz[g]);
            for (int o = 0; o < 8; ++o) gscales[g * 8 + o] = ie::fp32_to_fp16(gsf[g]);
        }

        uint16_t gw[8 * 8];
        // g_idx == nullptr → contiguous i/group_size.
        assert(ie::gptq_dequant_to_fp16(gqweight, gqzeros, gscales, nullptr,
                                        gin, gout, gG, gw).empty());
        for (int i = 0; i < 8; ++i) {
            const int g = i / int(gG);
            for (int o = 0; o < 8; ++o) {
                const float expect = gsf[g] * float(gqw[i][o] - (gqz[g][o] + 1));
                const float got    = ie::fp16_to_fp32(gw[i * 8 + o]);
                if (got != expect) {
                    std::fprintf(stderr, "gptq mismatch i=%d o=%d: got %.4f expect %.4f\n",
                                 i, o, got, expect);
                    return 1;
                }
            }
        }
        // Explicit g_idx (act-order form) mapping rows 0..7 → same groups.
        int32_t g_idx[8] = {0, 0, 0, 0, 1, 1, 1, 1};
        uint16_t gw2[8 * 8];
        assert(ie::gptq_dequant_to_fp16(gqweight, gqzeros, gscales, g_idx,
                                        gin, gout, gG, gw2).empty());
        for (int k = 0; k < 64; ++k) assert(gw2[k] == gw[k]);
    }

    std::printf("awq_dequant_test: all OK (AWQ+GPTQ bit-exact, config parsed)\n");
    return 0;
}
