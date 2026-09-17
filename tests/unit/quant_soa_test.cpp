// tests/unit/quant_soa_test.cpp — loader gate for the per-expert SoA repack
// (docs/prefill_crown_plan.md step 1).
//
// Host-only (no GPU):
//   1. Fabricate AoS expert banks (random bytes, valid struct framing) for
//      Q4_K and Q6_K at the production block counts.
//   2. repack_moe_*_soa_host → SoA buffer.
//   3. Field-by-field roundtrip: every block's qs/scales/d/dmin (resp.
//      ql/qh/scales/d) must be recoverable from the SoA streams at the
//      addresses the q4k_wview/q6k_wview<true> device views compute.
//   4. View equivalence: q4k_wview<false> over the AoS buffer and
//      q4k_wview<true> over the SoA buffer must yield byte-identical block
//      contents for every (expert, block) — this is exactly the contract the
//      templated kernels rely on for bit-identical math.

#include "ie/quant_blocks.hpp"
#include "ie/quant_soa.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace ie;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

template <typename View>
bool views_equal_q4k(const View& a, const q4k_wview<true>& b, uint64_t bi) {
    if (std::memcmp(a.qs_blk(bi), b.qs_blk(bi), 128) != 0) return false;
    if (std::memcmp(a.sc_blk(bi), b.sc_blk(bi), 12) != 0) return false;
    if (a.d_bits(bi) != b.d_bits(bi)) return false;
    if (a.dmin_bits(bi) != b.dmin_bits(bi)) return false;
    return true;
}

void test_q4k(uint32_t E, uint64_t nb, std::mt19937& rng) {
    const uint64_t stride = nb * sizeof(block_q4_K);
    std::vector<uint8_t> aos(E * stride), soa(E * stride);
    for (auto& v : aos) v = uint8_t(rng());

    repack_moe_q4k_soa_host(aos.data(), soa.data(), E, nb);

    for (uint32_t e = 0; e < E; ++e) {
        const auto* blocks =
            reinterpret_cast<const block_q4_K*>(aos.data() + uint64_t(e) * stride);
        const auto av = q4k_wview<false>::at(aos.data(), stride, e, nb);
        const auto sv = q4k_wview<true >::at(soa.data(), stride, e, nb);
        for (uint64_t b = 0; b < nb; ++b) {
            // Roundtrip vs the original struct fields.
            check(std::memcmp(sv.qs_blk(b), blocks[b].qs, 128) == 0, "q4k qs roundtrip");
            check(std::memcmp(sv.sc_blk(b), blocks[b].scales, 12) == 0, "q4k scales roundtrip");
            check(sv.d_bits(b) == blocks[b].d, "q4k d roundtrip");
            check(sv.dmin_bits(b) == blocks[b].dmin, "q4k dmin roundtrip");
            // AoS view ≡ SoA view (the kernel contract).
            check(views_equal_q4k(av, sv, b), "q4k view equivalence");
            if (g_fail) { std::fprintf(stderr, "  at e=%u b=%llu\n", e,
                                       (unsigned long long)b); return; }
        }
    }
}

void test_q6k(uint32_t E, uint64_t nb, std::mt19937& rng) {
    const uint64_t stride = nb * sizeof(block_q6_K);
    std::vector<uint8_t> aos(E * stride), soa(E * stride);
    for (auto& v : aos) v = uint8_t(rng());

    repack_moe_q6k_soa_host(aos.data(), soa.data(), E, nb);

    for (uint32_t e = 0; e < E; ++e) {
        const auto* blocks =
            reinterpret_cast<const block_q6_K*>(aos.data() + uint64_t(e) * stride);
        const auto av = q6k_wview<false>::at(aos.data(), stride, e, nb);
        const auto sv = q6k_wview<true >::at(soa.data(), stride, e, nb);
        for (uint64_t b = 0; b < nb; ++b) {
            check(std::memcmp(sv.ql_blk(b), blocks[b].ql, 128) == 0, "q6k ql roundtrip");
            check(std::memcmp(sv.qh_blk(b), blocks[b].qh, 64) == 0, "q6k qh roundtrip");
            check(std::memcmp(sv.sc_blk(b), blocks[b].scales, 16) == 0, "q6k scales roundtrip");
            check(sv.d_bits(b) == blocks[b].d, "q6k d roundtrip");
            check(std::memcmp(av.ql_blk(b), sv.ql_blk(b), 128) == 0, "q6k ql view eq");
            check(std::memcmp(av.qh_blk(b), sv.qh_blk(b), 64) == 0, "q6k qh view eq");
            check(std::memcmp(av.sc_blk(b), sv.sc_blk(b), 16) == 0, "q6k sc view eq");
            check(av.d_bits(b) == sv.d_bits(b), "q6k d view eq");
            if (g_fail) { std::fprintf(stderr, "  at e=%u b=%llu\n", e,
                                       (unsigned long long)b); return; }
        }
    }
}

}  // namespace

int main() {
    std::mt19937 rng(0x50A50A);

    // Small odd shapes first (catches stride math), then the production
    // shape: 4096 blocks/expert (gate/up: H=2048 × E_ffn=512 / 256;
    // down: E_ffn=512 × H=2048 / 256) at a reduced expert count to keep
    // runtime trivial.
    test_q4k(/*E=*/3, /*nb=*/5, rng);
    test_q6k(/*E=*/3, /*nb=*/5, rng);
    test_q4k(/*E=*/8, /*nb=*/4096, rng);
    test_q6k(/*E=*/8, /*nb=*/4096, rng);

    // SoA stream-start alignment at the production block count — the kernels
    // do 16 B vector loads from the qs/ql streams.
    {
        const q4k_soa_offsets o4(4096);
        const q6k_soa_offsets o6(4096);
        check(o4.scales % 16 == 0 && o4.dm % 16 == 0, "q4k stream alignment");
        check(o6.qh % 16 == 0 && o6.scales % 16 == 0 && o6.d % 16 == 0,
              "q6k stream alignment");
    }

    if (g_fail) { std::printf("quant_soa_test: FAIL (%d)\n", g_fail); return 1; }
    std::printf("quant_soa_test: all OK\n");
    return 0;
}
