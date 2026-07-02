// tools/q4_soa_test.cpp — host-only correctness gate for repack_q4_K_to_soa.
//
// NO GPU.  Generates random block_q4_K weights, runs the host SoA-Q4 repack
// (repack_q4_K_to_soa), reconstructs each element's value from the SoA streams
// using the documented natural-order formula, and compares — element by element
// — against the canonical AoS Q4_K dequant (value = d·s_raw·q4 − dmin·m_raw, the
// exact value gemv_q4_K produces).  The reconstructions must be BIT-IDENTICAL
// (same fp32 value math), proving the repack is a pure layout move and the
// maintainer's PPL gate (≤6.57) has the best chance of holding when the int-dot
// kernel is wired in.
//
// Build: ie-q4soa-test (links ie_core for repack_q4_K_to_soa).
// Run:   ./build/tools/ie-q4soa-test  → prints PASS/FAIL + max abs diff.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace ie;

namespace {

// get_scale_min_k4 — identical to gemv_q4k.cpp / ggml.  Reference for the AoS
// canonical dequant below.
void get_scale_min_k4_ref(int sub, const uint8_t* scales,
                          uint8_t& s_raw, uint8_t& m_raw) {
    if (sub < 4) {
        s_raw = scales[sub]     & 0x3F;
        m_raw = scales[sub + 4] & 0x3F;
    } else {
        s_raw = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4);
        m_raw = (scales[sub + 4] >>   4) | ((scales[sub    ] >> 6) << 4);
    }
}

// Canonical AoS Q4_K dequant of one super-block into natural element order
// (e = 0..255), matching gemv_q4_K's (g, hi_nib, half) lattice exactly:
//   out_off = g*64 + hi_nib*32 + half*16, q4 = qs[g*32 + half*16 + i] nibble
//   (low if hi_nib==0 else high), sub-block = g*2 + hi_nib.
// Inverting for natural e: g=e/64, r=e%64, hi_nib=r/32, half=(r%32)/16, i=e%16.
void aos_dequant_block(const block_q4_K& blk, float* out256) {
    const float d    = fp16_to_fp32(blk.d);
    const float dmin = fp16_to_fp32(blk.dmin);
    for (uint32_t e = 0; e < 256; ++e) {
        const uint32_t g      = e >> 6;
        const uint32_t r      = e & 63;
        const uint32_t hi_nib = r >> 5;
        const uint32_t half   = (r & 31) >> 4;
        const uint32_t i      = e & 15;
        const uint8_t  byte   = blk.qs[g * 32 + half * 16 + i];
        const int q4 = hi_nib ? int((byte >> 4) & 0x0F) : int(byte & 0x0F);
        const int sub = int(g) * 2 + int(hi_nib);
        uint8_t s_raw, m_raw;
        get_scale_min_k4_ref(sub, blk.scales, s_raw, m_raw);
        // value = d·s_raw·q4 − dmin·m_raw  (the exact gemv_q4_K fold).
        out256[e] = d * float(s_raw) * float(q4) - dmin * float(m_raw);
    }
}

}  // namespace

int main() {
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.05f);

    // A few shapes: K must be a multiple of 256; exercise multiple columns and
    // multiple super-blocks per column.
    struct Shape { uint32_t K, N; };
    const Shape shapes[] = {
        {256, 1}, {256, 5}, {512, 3}, {2048, 7}, {768, 4}, {256, 64},
    };

    double global_max_abs = 0.0;
    uint64_t total_elems = 0;
    bool all_pass = true;

    for (const Shape& sh : shapes) {
        const uint32_t K = sh.K, N = sh.N;
        const uint32_t bpc = K / 256;

        // Build N columns of random block_q4_K (each K/256 contiguous blocks).
        std::vector<block_q4_K> W(uint64_t(N) * bpc);
        for (auto& blk : W) {
            blk.d    = fp32_to_fp16(scale_dist(rng));
            blk.dmin = fp32_to_fp16(scale_dist(rng));
            for (int i = 0; i < kKScaleSize; ++i) blk.scales[i] = uint8_t(byte_dist(rng));
            for (int i = 0; i < kQK_K / 2;   ++i) blk.qs[i]     = uint8_t(byte_dist(rng));
        }

        // Allocate the SoA streams.
        std::vector<uint8_t>  q4_q (uint64_t(N) * (K / 2));
        std::vector<int8_t>   q4_sc(uint64_t(N) * (K / 32));
        std::vector<int8_t>   q4_mn(uint64_t(N) * (K / 32));
        std::vector<uint16_t> q4_d (uint64_t(N) * (K / 256));
        std::vector<uint16_t> q4_dmin(uint64_t(N) * (K / 256));

        repack_q4_K_to_soa(W.data(), K, N,
                           q4_q.data(), q4_sc.data(), q4_mn.data(),
                           q4_d.data(), q4_dmin.data());

        // Reconstruct from the SoA streams (the documented natural-order math)
        // and compare against the AoS canonical dequant element-by-element.
        double shape_max_abs = 0.0;
        uint64_t shape_mismatch = 0;
        std::vector<float> aos(256);

        for (uint64_t n = 0; n < N; ++n) {
            const uint8_t*  q_col  = q4_q.data()   + n * (uint64_t(K) / 2);
            const int8_t*   sc_col = q4_sc.data()  + n * (uint64_t(K) / 32);
            const int8_t*   mn_col = q4_mn.data()  + n * (uint64_t(K) / 32);
            const uint16_t* d_col  = q4_d.data()   + n * (uint64_t(K) / 256);
            const uint16_t* dm_col = q4_dmin.data()+ n * (uint64_t(K) / 256);

            for (uint32_t sb = 0; sb < bpc; ++sb) {
                aos_dequant_block(W[n * bpc + sb], aos.data());
                const float d    = fp16_to_fp32(d_col[sb]);
                const float dmin = fp16_to_fp32(dm_col[sb]);
                const uint32_t kb = sb * 256;

                for (uint32_t e = 0; e < 256; ++e) {
                    const uint32_t k    = kb + e;
                    const uint8_t  byte = q_col[k >> 1];
                    const int q4 = int((byte >> ((k & 1) * 4)) & 0x0F);
                    const int s  = int(e >> 5);               // sub-block 0..7
                    // Same fold the SoA int-dot kernel applies per element.
                    const float soa_val = d * float(sc_col[(kb >> 5) + s]) * float(q4)
                                        - dmin * float(mn_col[(kb >> 5) + s]);
                    const float ref = aos[e];
                    const float diff = soa_val - ref;
                    const double a = diff < 0 ? -double(diff) : double(diff);
                    if (a > shape_max_abs) shape_max_abs = a;
                    // BIT-IDENTICAL required (identical fp32 ops): demand exact.
                    if (soa_val != ref) {
                        if (shape_mismatch < 8) {
                            std::printf("  MISMATCH K=%u N=%u col=%llu sb=%u e=%u: "
                                        "soa=%.9g ref=%.9g (q4=%d s_raw=%d m_raw=%d)\n",
                                        K, N, (unsigned long long)n, sb, e,
                                        soa_val, ref, q4,
                                        int(sc_col[(kb >> 5) + s]),
                                        int(mn_col[(kb >> 5) + s]));
                        }
                        ++shape_mismatch;
                    }
                    ++total_elems;
                }
            }
        }

        const bool pass = (shape_mismatch == 0);
        all_pass = all_pass && pass;
        if (shape_max_abs > global_max_abs) global_max_abs = shape_max_abs;
        std::printf("[K=%5u N=%3u] %s  mismatches=%llu  max_abs_diff=%.3g\n",
                    K, N, pass ? "PASS" : "FAIL",
                    (unsigned long long)shape_mismatch, shape_max_abs);
    }

    std::printf("----------------------------------------\n");
    std::printf("%s  total_elems=%llu  global_max_abs_diff=%.3g\n",
                all_pass ? "PASS" : "FAIL",
                (unsigned long long)total_elems, global_max_abs);
    return all_pass ? 0 : 1;
}
