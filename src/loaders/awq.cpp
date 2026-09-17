// src/loaders/awq.cpp — AWQ 4-bit dequant (ingestion-time, host). See awq.hpp.
#include "ie/awq.hpp"

#include "../../third_party/nlohmann/json.hpp"

#include <cctype>
#include <string>

namespace ie {

using json = nlohmann::json;

std::string parse_awq_config(const std::string& config_json, AwqConfig& out) {
    json c;
    try {
        c = json::parse(config_json);
    } catch (const std::exception& e) {
        return std::string("config.json parse error: ") + e.what();
    }
    if (!c.contains("quantization_config"))
        return "config.json has no quantization_config";
    const json& q = c["quantization_config"];

    if (q.contains("quant_method")) out.quant_method = q["quant_method"].get<std::string>();
    if (q.contains("bits"))         out.bits        = q["bits"].get<int>();
    if (q.contains("w_bit"))        out.bits        = q["w_bit"].get<int>();      // AutoAWQ alias
    if (q.contains("group_size"))   out.group_size  = q["group_size"].get<int>();
    if (q.contains("q_group_size")) out.group_size  = q["q_group_size"].get<int>();
    if (q.contains("version"))      out.version     = q["version"].get<std::string>();
    if (q.contains("zero_point"))   out.zero_point  = q["zero_point"].get<bool>();
    if (q.contains("desc_act"))     out.desc_act    = q["desc_act"].get<bool>();
    if (q.contains("sym"))          out.sym         = q["sym"].get<bool>();

    for (auto& ch : out.version) ch = char(std::tolower((unsigned char)ch));
    if (out.bits != 4)
        return "4-bit only (got " + std::to_string(out.bits) + ")";

    if (out.quant_method == "awq") {
        if (out.version != "gemm")
            return "AWQ pack order '" + out.version + "' not supported yet (only 'gemm')";
    } else if (out.quant_method == "gptq") {
        // classic AutoGPTQ/gptqmodel: input-axis pack, natural order, (z+1).
        // desc_act=true (act-order) needs the g_idx permutation — handled in the
        // driver only when g_idx is present; here just record it.
    } else {
        return "unsupported quant_method '" + out.quant_method + "' (awq|gptq)";
    }
    return {};
}

std::string awq_dequant_to_fp16(const int32_t* qweight,
                                const int32_t* qzeros,
                                const uint16_t* scales,
                                int64_t in_features,
                                int64_t out_features,
                                int64_t group_size,
                                uint16_t* w_out) {
    if (out_features % 8 != 0)        return "out_features must be a multiple of 8";
    if (group_size <= 0)             return "group_size must be > 0";
    if (in_features % group_size != 0) return "in_features must be a multiple of group_size";

    const int64_t packed_cols = out_features / 8;   // int32 columns

    for (int64_t r = 0; r < in_features; ++r) {
        const int64_t g          = r / group_size;
        const int32_t* qw_row    = qweight + r * packed_cols;
        const int32_t* qz_row    = qzeros  + g * packed_cols;
        const uint16_t* sc_row   = scales  + g * out_features;
        uint16_t*       out_row  = w_out   + r * out_features;

        for (int64_t o = 0; o < out_features; ++o) {
            const int64_t  c  = o >> 3;          // o / 8 (packed int32 column)
            const int      j  = int(o & 7);      // o % 8 (logical lane)
            const int      sh = kAwqShift[j];
            const uint32_t q  = (uint32_t(qw_row[c]) >> sh) & 0xFu;
            const uint32_t z  = (uint32_t(qz_row[c]) >> sh) & 0xFu;
            const float    s  = fp16_to_fp32(sc_row[o]);
            const float    w  = (float(int(q) - int(z))) * s;
            out_row[o] = fp32_to_fp16(w);
        }
    }
    return {};
}

std::string gptq_dequant_to_fp16(const int32_t* qweight,
                                 const int32_t* qzeros,
                                 const uint16_t* scales,
                                 const int32_t* g_idx,
                                 int64_t in_features,
                                 int64_t out_features,
                                 int64_t group_size,
                                 uint16_t* w_out) {
    if (in_features % 8 != 0)  return "in_features must be a multiple of 8";
    if (out_features % 8 != 0) return "out_features must be a multiple of 8";
    if (group_size <= 0)       return "group_size must be > 0";

    const int64_t zpacked_cols = out_features / 8;   // qzeros int32 columns

    for (int64_t i = 0; i < in_features; ++i) {
        const int64_t  g       = g_idx ? int64_t(g_idx[i]) : (i / group_size);
        const int      ish     = int(i & 7) * 4;     // input-row nibble shift (natural)
        const int32_t* qw_row  = qweight + (i >> 3) * out_features;
        const int32_t* qz_row  = qzeros  + g * zpacked_cols;
        const uint16_t* sc_row = scales  + g * out_features;
        uint16_t*       out_row = w_out  + i * out_features;

        for (int64_t o = 0; o < out_features; ++o) {
            const uint32_t q  = (uint32_t(qw_row[o]) >> ish) & 0xFu;
            const int      osh = int(o & 7) * 4;       // output nibble shift (natural)
            const uint32_t z  = (uint32_t(qz_row[o >> 3]) >> osh) & 0xFu;
            const float    s  = fp16_to_fp32(sc_row[o]);
            const float    w  = s * float(int(q) - (int(z) + 1));
            out_row[o] = fp32_to_fp16(w);
        }
    }
    return {};
}

}  // namespace ie
