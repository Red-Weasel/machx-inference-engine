// tests/unit/deepseek4_expertgemm_gate_test.cpp — gate for the GROUPED,
// column-tiled routed-expert GEMM in src/ops/deepseek4_experts.cpp.
//
// WHAT CHANGED, AND THEREFORE WHAT HAS TO BE PROVEN
// --------------------------------------------------
// Three things moved in the batched expert GEMM, all of them for speed:
//
//   (a) the MXFP4 nibble decode became a 4-bit-packed magnitude table
//       (0xC8643210) instead of an exponent branch,
//   (b) a sub-group now owns kNCols=4 weight columns instead of 1, so one
//       activation block load feeds four dot products,
//   (c) every occupied expert of a chunk is issued as ONE 2-D launch
//       (grid dim 0 = job) instead of one launch per expert.
//
// None of the three is allowed to change a single output bit.  That is a
// stronger claim than "within tolerance", and it is the claim tested here —
// there is no tolerance anywhere in this file, because every comparison is
// between two computations that MUST agree exactly.  The error model is
// degenerate by construction: the permitted deviation is 0 ULP, and the
// justification is structural, not statistical —
//
//   * (a) is an identity over a 16-element integer domain, so it is checked
//     EXHAUSTIVELY (§1).  An identity that holds on all 16 inputs holds.
//   * (b) changes which columns share a sub-group.  Columns never interact:
//     each output element still accumulates over lane -> sub-block strides of
//     kSG in the same order and is still produced by a 16-lane sub-group
//     reduce.  §2/§3 check that against the UNTOUCHED reference GEMVs
//     (`gemv_mxfp4_soa_q8`, `gemv_iq3_xxs_soa_q8`, which live in files this
//     change does not own) rather than against another copy of the new code.
//   * (c) changes only the work-group -> (job, column-tile) mapping.  §4 checks
//     a grouped launch against per-job single-expert launches.
//
// BOTH TILINGS ARE COVERED, AND THAT IS ASSERTED.  The column tiling is chosen
// per launch from the resulting grid width, so a gate that only ever ran small
// groups would test the untiled kernel twice and the tiled one never.  §4
// therefore runs two group sizes straddling the threshold and asserts, via
// `ds4_expert_gemm_column_tiled`, that one landed on each side.  §2/§3 pin the
// untiled kernel to the untouched GEMVs; §4 chains the tiled kernel to it.
//
// NEGATIVE CONTROLS (§5).  A bit-exact comparison is worthless if the two sides
// are secretly the same buffer, or if the kernel writes nothing.  §5 injects
// five faults — one weight nibble, one weight scale byte, one grid index bit,
// one activation int8 bit, one job's row count — and asserts each is CAUGHT.
// It also asserts §1's exhaustive check rejects every single-bit corruption of
// the table constant.  Every fault is the smallest representable one, so
// passing §5 means the comparison discriminates at the finest granularity there
// is.  §2/§4 additionally assert the reference side is finite and mostly
// non-zero, which is what rules out the "both sides are zeros" pass.
//
// WHY SYNTHETIC WEIGHTS.  deepseek4_experts_test already gates this code
// against the real GGUF and the bit-exact host dequant; that is the VALUE
// contract and it is untouched.  This file gates the STRUCTURAL rewrite, for
// which random planes are strictly harder than trained ones: every nibble
// pattern, every sign nibble and every grid index occurs.  It also means this
// gate stands on its own with no 151 GB model present.
#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/deepseek4_experts.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using ie::block_q8_1x;
using ie::DS4ExpertBank;
using ie::DS4GemmGroupWs;
using ie::DS4GemmJob;
using ie::DType;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

// §1 reference: the MXFP4 nibble decode EXACTLY as written in the files this
// change does not own (gemv_mxfp4.cpp, ie::ref::dequant_mxfp4).  Transcribed
// rather than included, so a change to the fast path cannot drag the reference
// along with it.
int mxfp4_nibble_ref(uint32_t nb) {
    const int exp  = int((nb >> 1) & 3u);
    const int mant = int(nb & 1u);
    const int mag  = exp ? ((2 + mant) << (exp - 1)) : mant;
    return (nb & 8u) ? -mag : mag;
}
int mxfp4_nibble_lut(uint32_t nb, uint32_t table) {
    const int mag = int((table >> ((nb & 7u) * 4u)) & 0xFu);
    return (nb & 8u) ? -mag : mag;
}

// Synthetic banks.  Plane contents are uniform random bytes with two
// constraints, both present so the RESULT stays a meaningful fp16 number rather
// than a saturated inf that would make every comparison trivially pass:
//   * MXFP4 E8M0 scale bytes in [123,128]        -> 2^-5 .. 2^0
//   * IQ3_XXS fp16 super-scales, exponent 13..16 -> 2^-2 .. 2^1
// Everything else — every weight nibble, every grid index, every sign nibble,
// the 4-bit sub-scale in aux>>28 — is fully random, which is the point.
struct Bank {
    DS4ExpertBank b{};
    std::vector<uint8_t>  h_qs, h_e, h_gp;
    std::vector<uint32_t> h_ap;
    std::vector<uint16_t> h_dp;
};

Bank make_bank(sycl::queue& q, DType t, uint32_t K, uint32_t N, uint32_t E,
               std::mt19937& rng) {
    Bank r;
    r.b.dtype = t; r.b.K = K; r.b.N = N; r.b.E = E;
    if (t == DType::kMXFP4) {
        const uint64_t qs = uint64_t(N) * (uint64_t(K) / 2);
        const uint64_t ep = uint64_t(N) * (uint64_t(K) / 32);
        r.b.mx_qs_stride = qs; r.b.mx_e_stride = ep;
        r.h_qs.resize(uint64_t(E) * qs);
        r.h_e.resize(uint64_t(E) * ep);
        for (auto& v : r.h_qs) v = uint8_t(rng());
        for (auto& v : r.h_e)  v = uint8_t(123 + rng() % 6);
        r.b.mx_qs = sycl::malloc_device<uint8_t>(r.h_qs.size(), q);
        r.b.mx_e  = sycl::malloc_device<uint8_t>(r.h_e.size(), q);
        q.memcpy(r.b.mx_qs, r.h_qs.data(), r.h_qs.size());
        q.memcpy(r.b.mx_e,  r.h_e.data(),  r.h_e.size()).wait();
    } else {
        const uint64_t gp = uint64_t(N) * (uint64_t(K) / 4);
        const uint64_t ap = uint64_t(N) * (uint64_t(K) / 32);
        const uint64_t dp = uint64_t(N) * (uint64_t(K) / 256);
        r.b.gp_stride = gp; r.b.ap_stride = ap; r.b.dp_stride = dp;
        r.h_gp.resize(uint64_t(E) * gp);
        r.h_ap.resize(uint64_t(E) * ap);
        r.h_dp.resize(uint64_t(E) * dp);
        for (auto& v : r.h_gp) v = uint8_t(rng());
        for (auto& v : r.h_ap) v = uint32_t(rng());
        for (auto& v : r.h_dp) v = uint16_t(((13u + rng() % 4u) << 10) | (rng() & 0x3FFu));
        r.b.gp = sycl::malloc_device<uint8_t>(r.h_gp.size(), q);
        r.b.ap = sycl::malloc_device<uint32_t>(r.h_ap.size(), q);
        r.b.dp = sycl::malloc_device<uint16_t>(r.h_dp.size(), q);
        q.memcpy(r.b.gp, r.h_gp.data(), r.h_gp.size());
        q.memcpy(r.b.ap, r.h_ap.data(), r.h_ap.size() * sizeof(uint32_t));
        q.memcpy(r.b.dp, r.h_dp.data(), r.h_dp.size() * sizeof(uint16_t)).wait();
    }
    return r;
}

std::vector<uint16_t> read_h(sycl::queue& q, const sycl::half* p, uint64_t n) {
    std::vector<uint16_t> v(n);
    q.memcpy(v.data(), p, n * sizeof(uint16_t)).wait();
    return v;
}

// Differing fp16 BIT PATTERNS.  Not a norm — the contract is exact.
uint64_t bitdiff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    uint64_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += (a[i] != b[i]);
    return n;
}

// Rules out the degenerate "both sides are all zeros / all inf" pass.
bool nontrivial(const std::vector<uint16_t>& v) {
    uint64_t nz = 0, bad = 0;
    for (uint16_t h : v) {
        if ((h & 0x7FFFu) != 0) ++nz;
        if (((h >> 10) & 0x1Fu) == 0x1Fu) ++bad;   // inf or NaN
    }
    return nz > v.size() / 2 && bad == 0;
}

}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v,
                  {sycl::property::queue::in_order(),
                   sycl::property::queue::enable_profiling()}};
    std::printf("device: %s\n\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(0xD5E4);

    // -----------------------------------------------------------------------
    // §1  MXFP4 magnitude table — EXHAUSTIVE over the whole 16-value domain
    // -----------------------------------------------------------------------
    std::printf("§1 MXFP4 nibble table, exhaustive (domain = 16 values)\n");
    {
        bool all = true;
        for (uint32_t nb = 0; nb < 16; ++nb)
            all &= (mxfp4_nibble_lut(nb, 0xC8643210u) == mxfp4_nibble_ref(nb));
        check(all, "0xC8643210 reproduces the exponent-branch decode on all 16 nibbles");

        // NEGATIVE CONTROL: every single-bit corruption of the table must show up.
        int caught = 0;
        for (int bit = 0; bit < 32; ++bit) {
            const uint32_t bad = 0xC8643210u ^ (1u << bit);
            bool same = true;
            for (uint32_t nb = 0; nb < 16; ++nb)
                same &= (mxfp4_nibble_lut(nb, bad) == mxfp4_nibble_ref(nb));
            caught += !same;
        }
        check(caught == 32, "all 32 single-bit table corruptions rejected (" +
                            std::to_string(caught) + "/32)");
    }

    // N=96 and N=160 are deliberate: kColsPerWG is 128, so neither is a whole
    // number of column tiles and both exercise the tail mask the column tiling
    // introduced.  N=256/512 are whole tiles.  K=256 is the down-projection.
    struct Shape { uint32_t K, N; const char* tag; };
    const Shape shapes[] = {
        {512, 256, "K=512 N=256 (2 whole column tiles)"},
        {512,  96, "K=512 N=96  (partial tile, N < kColsPerWG)"},
        {512, 160, "K=512 N=160 (1 whole tile + 32-column tail)"},
        {256, 512, "K=256 N=512 (down-projection shape)"},
    };
    const DType dtypes[] = {DType::kMXFP4, DType::kIQ3_XXS};
    const char* dtag[]   = {"MXFP4  ", "IQ3_XXS"};

    const uint32_t E    = 6;
    const uint32_t Mmax = 19;   // > kMTile(8) and not a multiple of it: tiles 8,8,3

    // -----------------------------------------------------------------------
    // §2  M=1 vs the UNTOUCHED reference GEMVs — bit-exact
    // §3  M rows in one call vs M separate M=1 calls — bit-exact
    // -----------------------------------------------------------------------
    std::printf("\n§2/§3 kernel vs untouched reference GEMV, and M-batching\n");
    for (int d = 0; d < 2; ++d) {
        for (const Shape& s : shapes) {
            if (dtypes[d] == DType::kIQ3_XXS && s.K % 256 != 0) continue;
            Bank bk = make_bank(q, dtypes[d], s.K, s.N, E, rng);

            const uint32_t bpc = s.K / 32;
            std::vector<block_q8_1x> hx(uint64_t(Mmax) * bpc);
            for (auto& b : hx) {
                b.d = 0.002f + float(rng() % 64) * 1e-5f; b.s = 0.f;
                for (auto& c : b.qs) c = int8_t(rng());
            }
            auto* x = sycl::malloc_device<block_q8_1x>(hx.size(), q);
            q.memcpy(x, hx.data(), hx.size() * sizeof(block_q8_1x)).wait();

            auto* y_ref = sycl::malloc_device<sycl::half>(uint64_t(Mmax) * s.N, q);
            auto* y_new = sycl::malloc_device<sycl::half>(uint64_t(Mmax) * s.N, q);
            q.memset(y_ref, 0, uint64_t(Mmax) * s.N * 2);
            q.memset(y_new, 0, uint64_t(Mmax) * s.N * 2).wait();

            // reference: the M=1 decode GEMVs, one per row, rotating the expert
            for (uint32_t m = 0; m < Mmax; ++m) {
                const uint32_t e = m % E;
                const void* xr = x + uint64_t(m) * bpc;
                if (dtypes[d] == DType::kMXFP4)
                    ie::gemv_mxfp4_soa_q8(q, xr,
                                          bk.b.mx_qs + uint64_t(e) * bk.b.mx_qs_stride,
                                          bk.b.mx_e  + uint64_t(e) * bk.b.mx_e_stride,
                                          y_ref + uint64_t(m) * s.N, s.K, s.N);
                else
                    ie::gemv_iq3_xxs_soa_q8(q, xr,
                                            bk.b.gp + uint64_t(e) * bk.b.gp_stride,
                                            bk.b.ap + uint64_t(e) * bk.b.ap_stride,
                                            bk.b.dp + uint64_t(e) * bk.b.dp_stride,
                                            y_ref + uint64_t(m) * s.N, s.K, s.N);
            }
            for (uint32_t m = 0; m < Mmax; ++m)
                ie::ds4_expert_gemm_q8(q, bk.b, m % E, 1, x + uint64_t(m) * bpc,
                                       y_new + uint64_t(m) * s.N);
            q.wait();
            auto R = read_h(q, y_ref, uint64_t(Mmax) * s.N);
            auto A = read_h(q, y_new, uint64_t(Mmax) * s.N);
            check(nontrivial(R), std::string(dtag[d]) + " " + s.tag +
                                 " : reference is finite and mostly non-zero");
            check(bitdiff(R, A) == 0,
                  std::string("§2 ") + dtag[d] + " " + s.tag +
                  " : M=1 bit-exact vs GEMV (" + std::to_string(bitdiff(R, A)) +
                  " differing halves of " + std::to_string(R.size()) + ")");

            q.memset(y_ref, 0, uint64_t(Mmax) * s.N * 2);
            q.memset(y_new, 0, uint64_t(Mmax) * s.N * 2).wait();
            for (uint32_t m = 0; m < Mmax; ++m)
                ie::ds4_expert_gemm_q8(q, bk.b, 0, 1, x + uint64_t(m) * bpc,
                                       y_ref + uint64_t(m) * s.N);
            ie::ds4_expert_gemm_q8(q, bk.b, 0, Mmax, x, y_new);
            q.wait();
            R = read_h(q, y_ref, uint64_t(Mmax) * s.N);
            A = read_h(q, y_new, uint64_t(Mmax) * s.N);
            check(bitdiff(R, A) == 0,
                  std::string("§3 ") + dtag[d] + " " + s.tag + " : M=" +
                  std::to_string(Mmax) + " bit-exact vs " + std::to_string(Mmax) +
                  "x M=1 (" + std::to_string(bitdiff(R, A)) + " differing halves)");

            sycl::free(x, q); sycl::free(y_ref, q); sycl::free(y_new, q);
            ie::ds4_expert_bank_free(q, bk.b);
        }
    }

    // -----------------------------------------------------------------------
    // §4  grouped launch vs per-job launches — MIXED dtypes in ONE call, ragged
    //     M, and a job count larger than the ring slot capacity so the chunking
    //     branch (and a ring wrap) are exercised too.
    // -----------------------------------------------------------------------
    std::printf("\n§4 grouped launch vs per-job launches (mixed dtype, ragged M)\n");
    const uint32_t kNJobs[] = {40, 280};   // straddles the column-tiling threshold
    bool tiled_mx = false, tiled_iq3 = false, untiled_mx = false, untiled_iq3 = false;
    for (uint32_t NJ : kNJobs) {
        const uint32_t K = 512, N = 160, EJ = 8;
        Bank mx  = make_bank(q, DType::kMXFP4,   K, N, EJ, rng);
        Bank iq3 = make_bank(q, DType::kIQ3_XXS, K, N, EJ, rng);

        std::vector<uint32_t>    Ms;
        std::vector<const Bank*> src;
        uint32_t rows = 0;
        for (uint32_t j = 0; j < NJ; ++j) {
            const uint32_t m = 1 + (rng() % 17);
            Ms.push_back(m); rows += m;
            // Small group: uneven buckets, both below the tiling threshold.
            // Large group: even split, both ABOVE it — so each dtype is seen
            // through BOTH kernels, which is what the assertions below require.
            src.push_back(((NJ > 100) ? (j % 2 == 0) : (j % 3 == 0)) ? &iq3 : &mx);
        }
        const uint32_t bpc = K / 32;
        std::vector<block_q8_1x> hx(uint64_t(rows) * bpc);
        for (auto& b : hx) {
            b.d = 0.002f + float(rng() % 64) * 1e-5f; b.s = 0.f;
            for (auto& c : b.qs) c = int8_t(rng());
        }
        auto* x = sycl::malloc_device<block_q8_1x>(hx.size(), q);
        q.memcpy(x, hx.data(), hx.size() * sizeof(block_q8_1x)).wait();

        auto* y_ref = sycl::malloc_device<sycl::half>(uint64_t(rows) * N, q);
        auto* y_grp = sycl::malloc_device<sycl::half>(uint64_t(rows) * N, q);
        q.memset(y_ref, 0, uint64_t(rows) * N * 2);
        q.memset(y_grp, 0, uint64_t(rows) * N * 2).wait();

        std::vector<DS4GemmJob> jobs;
        uint32_t off = 0;
        for (size_t j = 0; j < Ms.size(); ++j) {
            const void* xr = x + uint64_t(off) * bpc;
            jobs.push_back({src[j]->b, uint32_t(j) % EJ, Ms[j], xr,
                            y_grp + uint64_t(off) * N});
            ie::ds4_expert_gemm_q8(q, src[j]->b, uint32_t(j) % EJ, Ms[j], xr,
                                   y_ref + uint64_t(off) * N);
            off += Ms[j];
        }
        // cap 12 << NJ and < either bucket: forces the split AND a ring wrap.
        // The split also means each launch sees ~12 jobs, so the tiling decision
        // is made on the SPLIT size — which is what makes NJ=40 vs NJ=280 land on
        // opposite sides only via the bucket sizes, so read the decision back
        // rather than predicting it.
        DS4GemmGroupWs gws;
        check(ie::ds4_gemm_group_ws_alloc(q, 1024, gws).empty(), "group workspace alloc");
        // Twice: the second pass is the one that can land on a ring slot whose
        // previous H2D has not completed — the hazard the per-slot event closes.
        for (int pass = 0; pass < 2; ++pass)
            check(ie::ds4_expert_gemm_q8_grouped(q, jobs.data(), uint32_t(jobs.size()),
                                                 gws).empty(),
                  "grouped launch pass " + std::to_string(pass));
        q.wait();
        auto R = read_h(q, y_ref, uint64_t(rows) * N);
        auto A = read_h(q, y_grp, uint64_t(rows) * N);
        check(nontrivial(R), "per-job output is finite and mostly non-zero");
        // Which kernel did the two buckets actually run?  ~2/3 of the jobs are
        // MXFP4 and ~1/3 IQ3_XXS, so both buckets are non-empty at either NJ.
        const uint32_t n_mx  = uint32_t(std::count(src.begin(), src.end(), &mx));
        const uint32_t n_iq3 = NJ - n_mx;
        const bool t_mx  = ie::ds4_expert_gemm_column_tiled(n_mx,  N);
        const bool t_iq3 = ie::ds4_expert_gemm_column_tiled(n_iq3, N);
        tiled_mx    |= t_mx;    untiled_mx  |= !t_mx;
        tiled_iq3   |= t_iq3;   untiled_iq3 |= !t_iq3;
        check(bitdiff(R, A) == 0,
              std::to_string(NJ) + " jobs (MXFP4 " + std::to_string(n_mx) +
              (t_mx ? " tiled" : " untiled") + ", IQ3_XXS " + std::to_string(n_iq3) +
              (t_iq3 ? " tiled" : " untiled") + ") : grouped bit-exact vs per-job (" +
              std::to_string(bitdiff(R, A)) + " differing halves of " +
              std::to_string(R.size()) + ")");

        // -------------------------------------------------------------------
        // §5  NEGATIVE CONTROLS — smallest representable fault, each must fail
        // -------------------------------------------------------------------
        if (NJ != kNJobs[1]) {
            ie::ds4_gemm_group_ws_free(q, gws);
            sycl::free(x, q); sycl::free(y_ref, q); sycl::free(y_grp, q);
            ie::ds4_expert_bank_free(q, mx.b);
            ie::ds4_expert_bank_free(q, iq3.b);
            continue;
        }
        std::printf("\n§5 negative controls (smallest representable faults)\n");
        // Inject into an expert slice this job list ACTUALLY reads.  Hardcoding
        // expert 0 is wrong: which experts a dtype touches depends on the
        // interleave, and a fault on an unread slice would silently "pass" the
        // bit-exact comparison for the wrong reason.
        uint32_t e_mx = UINT32_MAX, e_iq3 = UINT32_MAX;
        for (size_t j = 0; j < jobs.size(); ++j) {
            if (src[j] == &mx  && e_mx  == UINT32_MAX) e_mx  = jobs[j].e;
            if (src[j] == &iq3 && e_iq3 == UINT32_MAX) e_iq3 = jobs[j].e;
        }
        check(e_mx != UINT32_MAX && e_iq3 != UINT32_MAX,
              "both dtypes present in the job list (MXFP4 expert " +
              std::to_string(e_mx) + ", IQ3_XXS expert " + std::to_string(e_iq3) + ")");
        const uint64_t o_qs = uint64_t(e_mx)  * mx.b.mx_qs_stride + 7;
        const uint64_t o_ep = uint64_t(e_mx)  * mx.b.mx_e_stride  + 3;
        const uint64_t o_gp = uint64_t(e_iq3) * iq3.b.gp_stride   + 11;
        auto rerun = [&] {
            q.memset(y_grp, 0, uint64_t(rows) * N * 2).wait();
            ie::ds4_expert_gemm_q8_grouped(q, jobs.data(), uint32_t(jobs.size()), gws);
            q.wait();
            return read_h(q, y_grp, uint64_t(rows) * N);
        };
        {   // one MXFP4 weight NIBBLE (low 4 bits of one byte)
            const uint8_t v   = mx.h_qs[o_qs];
            const uint8_t bad = uint8_t((v & 0xF0u) | ((v + 1u) & 0x0Fu));
            q.memcpy(mx.b.mx_qs + o_qs, &bad, 1).wait();
            check(bitdiff(R, rerun()) != 0, "one MXFP4 weight nibble +1 -> caught");
            q.memcpy(mx.b.mx_qs + o_qs, &v, 1).wait();
        }
        {   // one MXFP4 E8M0 scale byte, one exponent step
            const uint8_t v   = mx.h_e[o_ep];
            const uint8_t bad = uint8_t(v + 1u);
            q.memcpy(mx.b.mx_e + o_ep, &bad, 1).wait();
            check(bitdiff(R, rerun()) != 0, "one MXFP4 scale byte +1 -> caught");
            q.memcpy(mx.b.mx_e + o_ep, &v, 1).wait();
        }
        {   // one IQ3_XXS grid index bit
            const uint8_t v   = iq3.h_gp[o_gp];
            const uint8_t bad = uint8_t(v ^ 1u);
            q.memcpy(iq3.b.gp + o_gp, &bad, 1).wait();
            check(bitdiff(R, rerun()) != 0, "one IQ3_XXS grid index bit -> caught");
            q.memcpy(iq3.b.gp + o_gp, &v, 1).wait();
        }
        {   // one activation int8 bit
            auto* qs0 = reinterpret_cast<int8_t*>(x) + offsetof(block_q8_1x, qs) + 5;
            const int8_t v   = hx[0].qs[5];
            const int8_t bad = int8_t(v ^ 1);
            q.memcpy(qs0, &bad, 1).wait();
            check(bitdiff(R, rerun()) != 0, "one activation int8 bit -> caught");
            q.memcpy(qs0, &v, 1).wait();
        }
        {   // one job's row count short by one -> its last row must be left unwritten
            const uint32_t keep = jobs[3].M;
            jobs[3].M = keep - 1;
            check(bitdiff(R, rerun()) != 0, "one job's M short by 1 -> caught");
            jobs[3].M = keep;
        }
        check(bitdiff(R, rerun()) == 0, "all faults reverted -> bit-exact again");

        // -------------------------------------------------------------------
        // Launch-count evidence.  Reported, not gated: the pass/fail contract of
        // this file is exactness.  The kernel-count reduction is what the change
        // exists for, so it is measured here rather than asserted from theory.
        // -------------------------------------------------------------------
        {
            ie::KernelProfiler prof;
            ie::g_profiler = &prof;
            prof.begin_step();
            for (const DS4GemmJob& j : jobs)
                ie::ds4_expert_gemm_q8(q, j.bank, j.e, j.M, j.x_q8, y_ref);
            q.wait();
            uint32_t per_job = 0;
            for (const auto& st : prof.harvest()) per_job += st.calls;

            prof.begin_step();
            ie::ds4_expert_gemm_q8_grouped(q, jobs.data(), uint32_t(jobs.size()), gws);
            q.wait();
            uint32_t grouped = 0;
            for (const auto& st : prof.harvest()) grouped += st.calls;
            ie::g_profiler = nullptr;

            std::printf("\n  %zu jobs (2 dtypes): per-job launches = %u,"
                        "  grouped launches = %u\n", jobs.size(), per_job, grouped);
            check(grouped < per_job, "grouped issues strictly fewer launches");
        }

        ie::ds4_gemm_group_ws_free(q, gws);
        sycl::free(x, q); sycl::free(y_ref, q); sycl::free(y_grp, q);
        ie::ds4_expert_bank_free(q, mx.b);
        ie::ds4_expert_bank_free(q, iq3.b);
    }

    check(tiled_mx   && untiled_mx,  "§4 exercised MXFP4 through BOTH grouped kernels");
    check(tiled_iq3  && untiled_iq3, "§4 exercised IQ3_XXS through BOTH grouped kernels");

    // -----------------------------------------------------------------------
    // §6  THE M == 1 (SINGLE-TOKEN DECODE) KERNELS
    //
    // A launch whose jobs ALL have M == 1 now takes a different pair of kernel
    // bodies (row loop removed) at a column width chosen from the grid rather
    // than from the two-way `tile_columns` boolean.  Two claims need proving,
    // and they are different in kind:
    //
    //   VALUE.  The M=1 bodies must be bit-identical to the decode GEMVs.  §6b
    //     checks that against `gemv_mxfp4_soa_q8` / `gemv_iq3_xxs_soa_q8` —
    //     the UNTOUCHED kernels in files this change does not own — not against
    //     `ds4_expert_gemm_q8`, which is also new code.  Structurally the claim
    //     holds because nothing an output element's value depends on moved: the
    //     lane -> sub-block assignment, the dp4a operand order, the
    //     `db * q8d * float(idot)` fold and the 16-lane reduce are unchanged.
    //
    //   DISPATCH SAFETY.  The M=1 body does NOT read the per-job row count, so
    //     it is only correct when every job in the launch has M == 1.  §6c puts
    //     one M>1 job into an otherwise all-M==1 group and checks the result is
    //     still exact — if the guard were wrong (checking only the first job,
    //     say) that job's extra rows would be dropped.  §6d then shortens that
    //     same job by one row and asserts the comparison CATCHES it, which is
    //     what makes §6c's pass meaningful rather than vacuous.
    //
    // ALL FOUR COLUMN WIDTHS ARE COVERED, AND THAT IS ASSERTED.  The width is
    // 1, 2, 4 or 8 depending on the launch's job count and N; a gate that ran
    // one group size would test one of them.  §6b runs four per-dtype bucket
    // sizes straddling all three thresholds and asserts, via
    // `ds4_expert_gemm_m1_ncols`, that four distinct widths were reached.
    // -----------------------------------------------------------------------
    std::printf("\n§6 M=1 decode kernels\n");
    {
        // §6a  the byte table the M=1 MXFP4 body indexes.  Its entries are
        // BUILT from the same `mxfp4_nibble_int` the M>1 body uses, so the only
        // thing that can be wrong is the packing — the two int8 results in the
        // two halves of a uint16.  Checked over the whole 256-byte domain
        // against the independently transcribed exponent-branch reference.
        auto lut_entry = [](uint32_t b) {
            return uint16_t((uint32_t(uint8_t(int8_t(mxfp4_nibble_lut(b >> 4, 0xC8643210u)))) << 8) |
                            uint8_t(int8_t(mxfp4_nibble_lut(b & 0x0Fu, 0xC8643210u))));
        };
        bool all = true;
        for (uint32_t b = 0; b < 256; ++b) {
            const uint16_t e = lut_entry(b);
            all &= (int8_t(e & 0xFFu) == mxfp4_nibble_ref(b & 0x0Fu));
            all &= (int8_t(e >> 8)    == mxfp4_nibble_ref(b >> 4));
        }
        check(all, "§6a byte table reproduces the exponent-branch decode on all 256 bytes");
        // NEGATIVE CONTROL: swapping the two halves must be rejected.  It is the
        // packing error the formula above could plausibly have.
        int caught = 0;
        for (uint32_t b = 0; b < 256; ++b) {
            const uint16_t e = lut_entry(b);
            const uint16_t sw = uint16_t((e << 8) | (e >> 8));
            if (int8_t(sw & 0xFFu) != mxfp4_nibble_ref(b & 0x0Fu)) ++caught;
        }
        check(caught > 200, "§6a half-swapped packing rejected on " +
                            std::to_string(caught) + "/256 bytes");
    }

    {
        const uint32_t K = 512, N = 160, EJ = 8;
        // Per-dtype bucket sizes chosen to straddle every width threshold at
        // N=160 (NC=2 needs 64 jobs, NC=4 needs 96, NC=8 needs 192).
        const uint32_t kPerDtype[] = {10, 70, 120, 200};
        std::vector<uint32_t> widths;

        for (uint32_t m : kPerDtype) {
            const uint32_t NJ = 2 * m;               // alternating dtypes
            Bank mx  = make_bank(q, DType::kMXFP4,   K, N, EJ, rng);
            Bank iq3 = make_bank(q, DType::kIQ3_XXS, K, N, EJ, rng);

            // Every LUT entry must actually be exercised, or §6b would be a
            // weaker check than it looks.  Random bytes make that true with
            // overwhelming probability; assert it instead of assuming it.
            if (m == kPerDtype[0]) {
                bool seen[256] = {};
                for (uint8_t v : mx.h_qs) seen[v] = true;
                uint32_t n_seen = 0;
                for (bool s : seen) n_seen += s;
                check(n_seen == 256,
                      "§6b MXFP4 weights contain all 256 byte values (" +
                      std::to_string(n_seen) + "/256) — every table entry is exercised");
            }

            const uint32_t bpc = K / 32;
            std::vector<block_q8_1x> hx(uint64_t(NJ) * bpc);
            for (auto& b : hx) {
                b.d = 0.002f + float(rng() % 64) * 1e-5f; b.s = 0.f;
                for (auto& c : b.qs) c = int8_t(rng());
            }
            auto* x = sycl::malloc_device<block_q8_1x>(hx.size(), q);
            q.memcpy(x, hx.data(), hx.size() * sizeof(block_q8_1x)).wait();

            auto* y_ref = sycl::malloc_device<sycl::half>(uint64_t(NJ) * N, q);
            auto* y_grp = sycl::malloc_device<sycl::half>(uint64_t(NJ) * N, q);
            q.memset(y_ref, 0, uint64_t(NJ) * N * 2);
            q.memset(y_grp, 0, uint64_t(NJ) * N * 2).wait();

            std::vector<DS4GemmJob> jobs;
            for (uint32_t j = 0; j < NJ; ++j) {
                const Bank& bk = (j % 2 == 0) ? mx : iq3;
                const uint32_t e = j % EJ;
                const void* xr = x + uint64_t(j) * bpc;
                jobs.push_back({bk.b, e, 1, xr, y_grp + uint64_t(j) * N});
                // REFERENCE: the untouched decode GEMVs, one row each.
                if (j % 2 == 0)
                    ie::gemv_mxfp4_soa_q8(q, xr, bk.b.mx_qs + uint64_t(e) * bk.b.mx_qs_stride,
                                          bk.b.mx_e + uint64_t(e) * bk.b.mx_e_stride,
                                          y_ref + uint64_t(j) * N, K, N);
                else
                    ie::gemv_iq3_xxs_soa_q8(q, xr, bk.b.gp + uint64_t(e) * bk.b.gp_stride,
                                            bk.b.ap + uint64_t(e) * bk.b.ap_stride,
                                            bk.b.dp + uint64_t(e) * bk.b.dp_stride,
                                            y_ref + uint64_t(j) * N, K, N);
            }
            DS4GemmGroupWs gws;
            check(ie::ds4_gemm_group_ws_alloc(q, 1024, gws).empty(), "group workspace alloc");
            check(ie::ds4_expert_gemm_q8_grouped(q, jobs.data(), uint32_t(jobs.size()), gws).empty(),
                  "grouped M=1 launch");
            q.wait();
            auto R = read_h(q, y_ref, uint64_t(NJ) * N);
            auto A = read_h(q, y_grp, uint64_t(NJ) * N);
            const uint32_t nc = ie::ds4_expert_gemm_m1_ncols(m, N);
            widths.push_back(nc);
            check(nontrivial(R), "§6b reference GEMV output is finite and mostly non-zero");
            check(bitdiff(R, A) == 0,
                  "§6b " + std::to_string(NJ) + " jobs all M=1 (" + std::to_string(m) +
                  " per dtype, column width " + std::to_string(nc) +
                  ") : bit-exact vs the untouched decode GEMVs (" +
                  std::to_string(bitdiff(R, A)) + " differing halves of " +
                  std::to_string(R.size()) + ")");

            // -------------------------------------------------------------
            // §6c/§6d  DISPATCH SAFETY.  One job in the group is given M=2,
            // which must force the whole bucket back onto the M>1 kernels.
            // Only done at the smallest size — it is a property of the guard,
            // not of the group width.
            // -------------------------------------------------------------
            if (m == kPerDtype[0]) {
                auto* x2 = sycl::malloc_device<block_q8_1x>(uint64_t(NJ + 1) * bpc, q);
                q.memcpy(x2, hx.data(), hx.size() * sizeof(block_q8_1x));
                q.memcpy(x2 + uint64_t(NJ) * bpc, hx.data(), bpc * sizeof(block_q8_1x)).wait();
                auto* y2r = sycl::malloc_device<sycl::half>(uint64_t(NJ + 1) * N, q);
                auto* y2g = sycl::malloc_device<sycl::half>(uint64_t(NJ + 1) * N, q);
                q.memset(y2r, 0, uint64_t(NJ + 1) * N * 2);
                q.memset(y2g, 0, uint64_t(NJ + 1) * N * 2).wait();

                // job 4 (MXFP4, mid-bucket — not the first, which is exactly
                // the job a guard that only inspected jobs[0] would miss)
                std::vector<DS4GemmJob> mj;
                uint32_t row = 0;
                for (uint32_t j = 0; j < NJ; ++j) {
                    const Bank& bk = (j % 2 == 0) ? mx : iq3;
                    const uint32_t M = (j == 4) ? 2u : 1u;
                    const void* xr = x2 + uint64_t(row) * bpc;
                    mj.push_back({bk.b, j % EJ, M, xr, y2g + uint64_t(row) * N});
                    ie::ds4_expert_gemm_q8(q, bk.b, j % EJ, M, xr, y2r + uint64_t(row) * N);
                    row += M;
                }
                check(ie::ds4_expert_gemm_q8_grouped(q, mj.data(), uint32_t(mj.size()), gws).empty(),
                      "grouped mixed-M launch");
                q.wait();
                auto R2 = read_h(q, y2r, uint64_t(row) * N);
                auto A2 = read_h(q, y2g, uint64_t(row) * N);
                check(nontrivial(R2), "§6c mixed-M reference is finite and mostly non-zero");
                check(bitdiff(R2, A2) == 0,
                      "§6c one M=2 job among " + std::to_string(NJ - 1) +
                      " M=1 jobs : still bit-exact — the M=1 kernel was NOT used (" +
                      std::to_string(bitdiff(R2, A2)) + " differing halves)");

                // NEGATIVE CONTROL: the M=2 job's second row must be observed.
                // Without this, §6c would pass even if the row were dropped on
                // BOTH sides of the comparison.
                mj[4].M = 1;
                q.memset(y2g, 0, uint64_t(row) * N * 2).wait();
                ie::ds4_expert_gemm_q8_grouped(q, mj.data(), uint32_t(mj.size()), gws);
                q.wait();
                auto A3 = read_h(q, y2g, uint64_t(row) * N);
                check(bitdiff(R2, A3) != 0,
                      "§6d the M=2 job's second row is actually compared "
                      "(dropping it is caught)");
                sycl::free(x2, q); sycl::free(y2r, q); sycl::free(y2g, q);

                // §6e  NEGATIVE CONTROLS on the M=1 path itself.  §6b's
                // comparison is against a different kernel in a different file,
                // so it cannot be vacuous by construction — but it can still be
                // blunt.  Injecting the smallest representable fault into each
                // dtype's weights proves it discriminates at nibble/bit level
                // here too, exactly as §5 does for the M>1 grouped path.
                auto rerun_m1 = [&] {
                    q.memset(y_grp, 0, uint64_t(NJ) * N * 2).wait();
                    ie::ds4_expert_gemm_q8_grouped(q, jobs.data(), uint32_t(jobs.size()), gws);
                    q.wait();
                    return read_h(q, y_grp, uint64_t(NJ) * N);
                };
                {   // one MXFP4 weight nibble, in expert 0 (job 0 reads it)
                    const uint64_t o = 7;
                    const uint8_t v = mx.h_qs[o];
                    const uint8_t bad = uint8_t((v & 0xF0u) | ((v + 1u) & 0x0Fu));
                    q.memcpy(mx.b.mx_qs + o, &bad, 1).wait();
                    check(bitdiff(R, rerun_m1()) != 0,
                          "§6e M=1: one MXFP4 weight nibble +1 -> caught");
                    q.memcpy(mx.b.mx_qs + o, &v, 1).wait();
                }
                {   // one IQ3_XXS grid index bit, in expert 1 (job 1 reads it)
                    const uint64_t o = iq3.b.gp_stride + 11;
                    const uint8_t v = iq3.h_gp[o];
                    const uint8_t bad = uint8_t(v ^ 1u);
                    q.memcpy(iq3.b.gp + o, &bad, 1).wait();
                    check(bitdiff(R, rerun_m1()) != 0,
                          "§6e M=1: one IQ3_XXS grid index bit -> caught");
                    q.memcpy(iq3.b.gp + o, &v, 1).wait();
                }
                check(bitdiff(R, rerun_m1()) == 0,
                      "§6e M=1: all faults reverted -> bit-exact again");
            }

            ie::ds4_gemm_group_ws_free(q, gws);
            sycl::free(x, q); sycl::free(y_ref, q); sycl::free(y_grp, q);
            ie::ds4_expert_bank_free(q, mx.b);
            ie::ds4_expert_bank_free(q, iq3.b);
        }
        std::sort(widths.begin(), widths.end());
        widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
        std::string w;
        for (uint32_t v : widths) w += std::to_string(v) + " ";
        check(widths.size() == 4,
              "§6 exercised all four M=1 column widths (saw: " + w + ")");
    }

    // Token-tiled grouped MXFP4: skewed jobs, eight-row boundaries, empty
    // jobs, partial column tiles, and descriptor splitting/ring reuse.
    for (const uint32_t N : {131u, 4096u}) {
        const uint32_t K = N == 131 ? 512u : 1024u, NJ = 32, stride = 97;
        Bank mx = make_bank(q, DType::kMXFP4, K, N, NJ, rng);
        const size_t xn = size_t(NJ) * stride * (K / 32);
        const size_t yn = size_t(NJ) * stride * N;
        std::vector<block_q8_1x> hx(xn);
        for (auto& b : hx) {
            b.d = 0.002f + float(rng() % 64) * 1e-5f; b.s = 0.f;
            for (auto& c : b.qs) c = int8_t(rng());
        }
        auto* x = sycl::malloc_device<block_q8_1x>(xn, q);
        auto* yr = sycl::malloc_device<sycl::half>(yn, q);
        auto* yg = sycl::malloc_device<sycl::half>(yn, q);
        check(x && yr && yg, "token tile allocations");
        if (!x || !yr || !yg) return 1;
        q.memcpy(x, hx.data(), xn * sizeof(block_q8_1x));
        q.fill(yr, sycl::half(123.f), yn);
        q.fill(yg, sycl::half(123.f), yn).wait();
        const uint32_t sizes[] = {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 48, 95};
        std::vector<DS4GemmJob> jobs;
        for (uint32_t j = 0; j < NJ; ++j) {
            const uint32_t M = sizes[j % std::size(sizes)];
            const auto* xj = x + size_t(j) * stride * (K / 32);
            jobs.push_back({mx.b, j, M, xj, yg + size_t(j) * stride * N});
            if (M) ie::ds4_expert_gemm_q8(q, mx.b, j, M, xj,
                                         yr + size_t(j) * stride * N);
        }
        q.wait();
        const auto expected = read_h(q, yr, yn);
        for (const uint32_t cap : {7u, 64u}) {
            q.fill(yg, sycl::half(123.f), yn).wait();
            DS4GemmGroupWs ws;
            check(ie::ds4_gemm_group_ws_alloc(q, cap, ws).empty(), "token tile workspace");
            for (uint32_t repeat = 0; repeat < 10; ++repeat)
                check(ie::ds4_expert_gemm_q8_grouped(q, jobs.data(), NJ, ws).empty(),
                      "token tile grouped submission");
            q.wait();
            const auto actual = read_h(q, yg, yn);
            check(bitdiff(expected, actual) == 0,
                  "token tiles N=" + std::to_string(N) + " cap=" + std::to_string(cap) +
                  ": exact outputs and untouched row guards");
            ie::ds4_gemm_group_ws_free(q, ws);
        }
        sycl::free(x, q); sycl::free(yr, q); sycl::free(yg, q);
        ie::ds4_expert_bank_free(q, mx.b);
    }

    std::printf("\n%s (%d failure%s)\n", g_fail ? "GATE FAILED" : "GATE PASSED",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
