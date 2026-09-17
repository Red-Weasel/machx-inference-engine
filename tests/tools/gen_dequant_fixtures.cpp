// tests/tools/gen_dequant_fixtures.cpp — regenerate the dequant KAT fixtures.
//
// Emits tests/unit/dequant_fixtures.inc: for each quant type it quantizes
// deterministic fp32 data with ggml, captures the packed GGUF bytes + ggml's
// OWN dequantized golden floats (as bit-exact hex-float literals), and — as a
// build-time self-check — proves ie::ref::dequant_* reproduces ggml bit-for-bit.
//
// This is a DEV TOOL, not part of the engine build (it links libggml). The
// committed fixture is what tests/unit/dequant_ref_test.cpp consumes; only rerun
// this when the block structs / reference dequant change.
//
// Build + regenerate (paths are for this dev box; adjust the llama.cpp root):
//   GG=~/llama.cpp
//   g++ -std=c++20 -O2 -I include -I "$GG/ggml/include" \
//       tests/tools/gen_dequant_fixtures.cpp -o /tmp/gen_dequant_fixtures \
//       -L "$GG/build-cpu/bin" -lggml-base -Wl,-rpath,"$GG/build-cpu/bin"
//   /tmp/gen_dequant_fixtures --emit tests/unit/dequant_fixtures.inc
// A clean run prints "ALL BIT-EXACT" (ie::ref == ggml for every element).
#include "ggml.h"
#include "ie/quant_blocks.hpp"
#include "ie/dequant_ref.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>

struct Case {
    const char*  name;
    ggml_type    type;
    int          block_elems;
    size_t       block_bytes;
    void       (*ref)(const void*, size_t, float*);
};

static const Case cases[] = {
    {"Q4_1", GGML_TYPE_Q4_1, 32,  sizeof(ie::block_q4_1), ie::ref::dequant_q4_1_buffer},
    {"Q5_0", GGML_TYPE_Q5_0, 32,  sizeof(ie::block_q5_0), ie::ref::dequant_q5_0_buffer},
    {"Q5_1", GGML_TYPE_Q5_1, 32,  sizeof(ie::block_q5_1), ie::ref::dequant_q5_1_buffer},
    {"Q2_K", GGML_TYPE_Q2_K, 256, sizeof(ie::block_q2_K), ie::ref::dequant_q2_K_buffer},
    {"Q3_K", GGML_TYPE_Q3_K, 256, sizeof(ie::block_q3_K), ie::ref::dequant_q3_K_buffer},
    {"IQ4_NL", GGML_TYPE_IQ4_NL, 32,  sizeof(ie::block_iq4_nl), ie::ref::dequant_iq4_nl_buffer},
    {"IQ4_XS", GGML_TYPE_IQ4_XS, 256, sizeof(ie::block_iq4_xs), ie::ref::dequant_iq4_xs_buffer},
};

int main(int argc, char** argv) {
    const bool emit = (argc > 1 && std::string(argv[1]) == "--emit");
    const int NB = 4;  // blocks per type in the fixture

    std::string hdr;
    hdr += "// dequant_fixtures.inc — GENERATED, do not edit by hand.\n";
    hdr += "// ggml known-answer vectors: packed bytes + ggml's own dequantized golden\n";
    hdr += "// floats (hex-float = bit-exact). Regenerate via tests/tools/\n";
    hdr += "// gen_dequant_fixtures.cpp (see its header for the libggml build command).\n";
    hdr += "namespace ie::test {\nstruct Fixture { const char* name; const unsigned char* packed;\n"
           "  size_t packed_len; const float* golden; size_t n; };\n";

    int fails = 0;
    std::vector<std::string> reg;
    for (const auto& c : cases) {
        const int n = NB * c.block_elems;
        // Deterministic, varied fp32 input (range chosen so quant scales are normal fp16).
        std::vector<float> src(n);
        uint32_t s = 0x1234567u;
        for (int i = 0; i < n; ++i) {
            s = s * 1664525u + 1013904223u;                 // LCG
            src[i] = (float(s >> 8) / float(1u << 24) - 0.5f) * 3.0f;  // ~[-1.5,1.5]
        }
        std::vector<uint8_t> packed(NB * c.block_bytes);
        ggml_quantize_chunk(c.type, src.data(), packed.data(), 0, NB, c.block_elems, nullptr);

        std::vector<float> golden(n);
        ggml_get_type_traits(c.type)->to_float(packed.data(), golden.data(), n);

        std::vector<float> mine(n);
        c.ref(packed.data(), size_t(n), mine.data());

        int mismatch = 0; float worst = 0;
        for (int i = 0; i < n; ++i) {
            if (std::memcmp(&golden[i], &mine[i], 4) != 0) {   // bit-exact fp32
                ++mismatch;
                worst = std::fmax(worst, std::fabs(golden[i] - mine[i]));
            }
        }
        std::printf("%-5s  blocks=%d elems=%d  bit-exact-mismatches=%d  worst_abs=%.3g  %s\n",
                    c.name, NB, n, mismatch, worst, mismatch ? "FAIL" : "OK");
        if (mismatch) ++fails;

        if (emit) {
            const std::string pv = std::string("packed_") + c.name;
            const std::string gv = std::string("golden_") + c.name;
            char buf[64];
            hdr += "static const unsigned char " + pv + "[] = {";
            for (size_t i = 0; i < packed.size(); ++i) {
                std::snprintf(buf, sizeof(buf), "%s%u", (i ? "," : ""), packed[i]);
                hdr += buf;
            }
            hdr += "};\n";
            hdr += "static const float " + gv + "[] = {";
            for (int i = 0; i < n; ++i) {
                if (i) hdr += ",";
                // Hex-float (%a): the exact, sign-of-zero-preserving representation
                // of the float. float→double promotion is lossless and the double
                // hex literal re-narrows to the identical float, so the committed
                // golden is bit-identical to ggml's output (enables strict memcmp).
                std::snprintf(buf, sizeof(buf), "%a", golden[i]);
                hdr += buf;
            }
            hdr += "};\n";
            reg.push_back(std::string("{\"") + c.name + "\"," + pv + ",sizeof(" + pv + ")," +
                          gv + "," + std::to_string(n) + "}");
        }
    }

    if (emit) {
        hdr += "static const Fixture kFixtures[] = {\n  ";
        for (size_t i = 0; i < reg.size(); ++i) hdr += (i ? ",\n  " : "") + reg[i];
        hdr += "\n};\n} // namespace ie::test\n";
        FILE* f = std::fopen(argv[2], "w");
        std::fwrite(hdr.data(), 1, hdr.size(), f);
        std::fclose(f);
        std::printf("wrote %s (%zu bytes)\n", argv[2], hdr.size());
    }
    std::printf("\n%s\n", fails ? "FAILURES" : "ALL BIT-EXACT");
    return fails;
}
