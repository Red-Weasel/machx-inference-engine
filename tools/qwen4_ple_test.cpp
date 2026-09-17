// tools/qwen4_ple_test.cpp — qwen4exp PLE n-gram machinery gate (no model file).
//
// 1. Host hash vs hand-computed uint64 expectations (EOS-segmentation boundary
//    cases) + 3 random sequences vs an independent scan-based reimplementation
//    + chunked-hash / spec-decode rollback equivalence through PleHistory.
// 2. IQ4_NL row gather vs direct reference dequant — byte-exact fp32.
// 3. Device blk.1 PLE layer math vs a CPU fp64 reference, T in {1, 5, 64}.
// 4. Chunked-vs-monolithic T=64 with conv-state carry (same precision path).
//
// usage: ie-qwen4-ple-test
#include "ie/dequant_ref.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/qwen4_ple.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace ie;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what);
    if (!ok) ++g_fail;
}

// -- real constants from docs/qwen4/12_ple_ngram.md ---------------------------
constexpr int32_t kEos = 248044;

const PleHashConsts kConsts = {
    {23703573157769ull, 20109073645365ull, 8052911324071ull},
    {20000003ull, 20000023ull, 20000033ull, 20000047ull, 20000059ull, 20000063ull,
     20000069ull, 20000077ull, 20000081ull, 20000093ull, 20000107ull, 20000147ull,
     20000153ull, 20000159ull, 20000161ull, 20000171ull},
    {0ull, 20000003ull, 40000026ull, 60000059ull, 80000106ull, 100000165ull,
     120000228ull, 140000297ull, 160000374ull, 180000455ull, 200000548ull,
     220000655ull, 240000802ull, 260000955ull, 280001114ull, 300001275ull},
    uint32_t(kEos)};

// Independent scan-based reimplementation of the hash (per-position last-EOS
// scan; the production code uses a streaming history recurrence instead).
void scan_hash(const std::vector<int32_t>& toks, std::vector<uint64_t>& rows) {
    const size_t T = toks.size();
    rows.resize(T * 16);
    for (size_t p = 0; p < T; ++p) {
        long last_eos = -1;
        for (size_t q = 0; q < p; ++q)
            if (toks[q] == kEos) last_eos = long(q);
        uint64_t ctx[3] = {uint64_t(uint32_t(toks[p])), uint64_t(kEos), uint64_t(kEos)};
        for (int s = 1; s <= 2; ++s) {
            const long q = long(p) - s;
            if (q < 0 || q <= last_eos) break;  // this + all further stay EOS
            ctx[s] = uint64_t(uint32_t(toks[q]));
        }
        const uint64_t m2 = ctx[0] * kConsts.M[0] ^ ctx[1] * kConsts.M[1];
        const uint64_t m3 = m2 ^ ctx[2] * kConsts.M[2];
        for (int h = 0; h < 16; ++h)
            rows[p * 16 + h] = ((h < 8) ? m2 : m3) % kConsts.V[h] + kConsts.O[h];
    }
}

// -- CPU fp64 reference of the device layer (spec §3, zero conv history) -----
void ref_ple_layer(const std::vector<sycl::half>& key, const std::vector<sycl::half>& v,
                   std::vector<double>& H,
                   const std::vector<float>& gk, const std::vector<float>& gq,
                   const std::vector<float>& gc, const std::vector<float>& w,
                   uint32_t T, double eps) {
    constexpr uint32_t D = 2560, SI = 10240, HC = 4;
    std::vector<double> u(size_t(T) * SI), gated(size_t(T) * SI);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t j = 0; j < HC; ++j) {
            const size_t base = size_t(t) * SI + size_t(j) * D;
            double skk = 0, sqq = 0, sx = 0;
            for (uint32_t c = 0; c < D; ++c) {
                const double kv = double(float(key[base + c]));
                const double qv = H[base + c];
                skk += kv * kv;
                sqq += qv * qv;
                sx  += (kv * double(gk[j * D + c])) * (qv * double(gq[j * D + c]));
            }
            const double rk = std::sqrt(skk / D + eps);
            const double rq = std::sqrt(sqq / D + eps);
            const double s  = sx / (rk * rq) / std::sqrt(double(D));
            const double sg = (s > 0) ? 1.0 : ((s < 0) ? -1.0 : 0.0);
            const double a  = std::max(std::fabs(s), 1e-6);
            const double g  = 1.0 / (1.0 + std::exp(-sg * std::sqrt(a)));
            double ss = 0;
            for (uint32_t c = 0; c < D; ++c) {
                const double gv = g * double(float(v[size_t(t) * D + c]));
                gated[base + c] = gv;
                ss += gv * gv;
            }
            const double inv = 1.0 / std::sqrt(ss / D + eps);
            for (uint32_t c = 0; c < D; ++c)
                u[base + c] = gated[base + c] * double(gc[j * D + c]) * inv;
        }
    }
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t c = 0; c < SI; ++c) {
            double y = 0;
            for (uint32_t k = 0; k < 4; ++k) {
                const int p = int(t) - int((3 - k) * 3);
                if (p >= 0) y += double(w[size_t(k) * SI + c]) * u[size_t(p) * SI + c];
            }
            const double sy = y / (1.0 + std::exp(-y));
            H[size_t(t) * SI + c] += gated[size_t(t) * SI + c] + sy;
        }
    }
}

// Run the device layer over `chunks` (must sum to T) with conv-state carry;
// returns the updated H.
std::vector<float> run_device(sycl::queue& q,
                              const std::vector<sycl::half>& key,
                              const std::vector<sycl::half>& v,
                              const std::vector<float>& H0,
                              const std::vector<float>& gk, const std::vector<float>& gq,
                              const std::vector<float>& gc, const std::vector<float>& w,
                              uint32_t T, const std::vector<uint32_t>& chunks) {
    constexpr uint32_t D = 2560, SI = 10240, SR = 9;
    uint32_t max_chunk = 0;
    for (uint32_t c : chunks) max_chunk = std::max(max_chunk, c);

    auto* d_key = sycl::malloc_device<sycl::half>(size_t(T) * SI, q);
    auto* d_v   = sycl::malloc_device<sycl::half>(size_t(T) * D, q);
    auto* d_H   = sycl::malloc_device<float>(size_t(T) * SI, q);
    auto* d_gk  = sycl::malloc_device<float>(SI, q);
    auto* d_gq  = sycl::malloc_device<float>(SI, q);
    auto* d_gc  = sycl::malloc_device<float>(SI, q);
    auto* d_w   = sycl::malloc_device<float>(4 * SI, q);
    auto* d_st  = sycl::malloc_device<float>(size_t(SR) * SI, q);
    auto* d_ws  = sycl::malloc_device<float>(qwen4_ple_workspace_floats(max_chunk), q);

    q.memcpy(d_key, key.data(), key.size() * sizeof(sycl::half));
    q.memcpy(d_v, v.data(), v.size() * sizeof(sycl::half));
    q.memcpy(d_H, H0.data(), H0.size() * sizeof(float));
    q.memcpy(d_gk, gk.data(), SI * sizeof(float));
    q.memcpy(d_gq, gq.data(), SI * sizeof(float));
    q.memcpy(d_gc, gc.data(), SI * sizeof(float));
    q.memcpy(d_w, w.data(), 4 * SI * sizeof(float));
    q.memset(d_st, 0, size_t(SR) * SI * sizeof(float));  // sequence start
    q.wait();

    uint32_t off = 0;
    for (uint32_t ct : chunks) {
        qwen4_ple_layer(q, d_key + size_t(off) * SI, d_v + size_t(off) * D,
                        d_H + size_t(off) * SI, d_gk, d_gq, d_gc, d_w,
                        d_st, d_ws, ct, 1e-6f).wait();
        off += ct;
    }

    std::vector<float> H_out(size_t(T) * SI);
    q.memcpy(H_out.data(), d_H, H_out.size() * sizeof(float)).wait();
    for (void* p : {(void*)d_key, (void*)d_v, (void*)d_H, (void*)d_gk, (void*)d_gq,
                    (void*)d_gc, (void*)d_w, (void*)d_st, (void*)d_ws})
        sycl::free(p, q);
    return H_out;
}

}  // namespace

int main() {
    std::printf("\n\033[1mqwen4exp PLE n-gram gate\033[0m\n");

    // ---- 1. hash: hand-computed boundary cases ------------------------------
    std::printf("\n\033[1m1. n-gram hash (EOS segmentation)\033[0m\n");
    {
        struct Case {
            const char* name;
            std::vector<int32_t> toks;
            std::vector<std::vector<uint64_t>> want;  // [T][16]
        };
        const std::vector<Case> cases = {
            {"[12345] (p=0: no predecessors -> EOS,EOS)", {12345},
             {{11270970ull, 23825822ull, 53627731ull, 75297762ull, 86108900ull, 117131242ull, 124369521ull, 148669731ull, 173053968ull, 186938961ull, 219076364ull, 237563026ull, 246838602ull, 277288915ull, 294366759ull, 301712965ull}}},
            {"[12345,678] (p=1: one predecessor, ctx[2]=EOS)", {12345, 678},
             {{11270970ull, 23825822ull, 53627731ull, 75297762ull, 86108900ull, 117131242ull, 124369521ull, 148669731ull, 173053968ull, 186938961ull, 219076364ull, 237563026ull, 246838602ull, 277288915ull, 294366759ull, 301712965ull},
              {15118845ull, 38743173ull, 50732867ull, 63717260ull, 86459970ull, 107412122ull, 138875864ull, 140893662ull, 166401433ull, 190941423ull, 207873384ull, 237383154ull, 253049689ull, 269039702ull, 281108310ull, 301990158ull}}},
            {"[111,EOS,222] (EOS at p-1 cuts all context)", {111, kEos, 222},
             {{18029874ull, 38263580ull, 52119953ull, 67707116ull, 87813783ull, 108647142ull, 130645055ull, 154705004ull, 160120739ull, 188327707ull, 214101459ull, 235954048ull, 242932251ull, 271136779ull, 287477492ull, 311224194ull},
              {1486346ull, 26399009ull, 53266562ull, 67821634ull, 84885046ull, 111513965ull, 122339489ull, 158420438ull, 167883139ull, 194446559ull, 203632947ull, 221591114ull, 244971916ull, 269314536ull, 290975920ull, 300885055ull},
              {16628311ull, 21205561ull, 47232059ull, 79855549ull, 94563007ull, 100262897ull, 129560365ull, 143352361ull, 169565113ull, 183569863ull, 206110010ull, 227308642ull, 247190569ull, 268299118ull, 295608044ull, 314196078ull}}},
            {"[EOS,333,444] (EOS at p-2 only)", {kEos, 333, 444},
             {{9663979ull, 26558231ull, 56120240ull, 74755659ull, 80459717ull, 109265651ull, 132697467ull, 151022725ull, 170054832ull, 192967038ull, 200687722ull, 225763581ull, 259275737ull, 272983544ull, 297596484ull, 300986548ull},
              {5648768ull, 35699019ull, 44460873ull, 76912641ull, 82900296ull, 105693408ull, 120630361ull, 148608036ull, 179147405ull, 188262866ull, 215101535ull, 220025757ull, 247468600ull, 276138593ull, 292634677ull, 317159854ull},
              {8083160ull, 27867742ull, 57770766ull, 75646958ull, 99551993ull, 107522622ull, 129480739ull, 145428849ull, 161442580ull, 185045576ull, 207721735ull, 228941756ull, 244494232ull, 260403942ull, 299120165ull, 313295231ull}}},
            {"[55,66,EOS,77] (own EOS keeps own context; then reset)", {55, 66, kEos, 77},
             {{5116423ull, 25785398ull, 49861567ull, 63758826ull, 88133535ull, 117056721ull, 121189706ull, 141430612ull, 176661835ull, 185732010ull, 219178299ull, 237230079ull, 244637154ull, 273270290ull, 296420534ull, 314214129ull},
              {10152290ull, 32183063ull, 53199821ull, 74624795ull, 95847624ull, 116255565ull, 136867721ull, 157684418ull, 168488685ull, 183827053ull, 216874357ull, 230661005ull, 242108395ull, 273915770ull, 291264912ull, 318609859ull},
              {6122837ull, 37611049ull, 57765805ull, 74922351ull, 99929321ull, 115872600ull, 130669613ull, 145378906ull, 169055256ull, 193607785ull, 210937312ull, 226492529ull, 259883840ull, 274333566ull, 286052097ull, 306408700ull},
              {16611476ull, 29512305ull, 49704971ull, 66165983ull, 89881645ull, 105251896ull, 139055756ull, 152191225ull, 178497739ull, 192880870ull, 205859405ull, 228292887ull, 258357841ull, 269648670ull, 287018169ull, 315908561ull}}},
        };
        for (const auto& cs : cases) {
            PleHistory hist;
            hist.reset(kConsts.eos);
            std::vector<uint64_t> rows(cs.toks.size() * 16);
            qwen4_ple_hash(kConsts, cs.toks.data(), uint32_t(cs.toks.size()), hist, rows.data());
            bool ok = true;
            for (size_t p = 0; p < cs.toks.size(); ++p)
                for (int h = 0; h < 16; ++h)
                    if (rows[p * 16 + h] != cs.want[p][size_t(h)]) ok = false;
            check(ok, cs.name);
        }

        // 3 random sequences vs the independent scan-based reimplementation.
        std::mt19937 rng(20260826);
        std::uniform_int_distribution<int32_t> tok(0, 248056);
        std::uniform_real_distribution<float> uf(0.f, 1.f);
        for (int cse = 0; cse < 3; ++cse) {
            std::vector<int32_t> toks(40);
            for (auto& t : toks) t = (uf(rng) < 0.15f) ? kEos : tok(rng);
            std::vector<uint64_t> want;
            scan_hash(toks, want);
            PleHistory hist;
            hist.reset(kConsts.eos);
            std::vector<uint64_t> got(toks.size() * 16);
            qwen4_ple_hash(kConsts, toks.data(), uint32_t(toks.size()), hist, got.data());
            char what[96];
            std::snprintf(what, sizeof what, "random seq %d (T=40) == scan-based reimpl", cse);
            check(got == want, what);
        }

        // Chunked hash through PleHistory == monolithic; save/rollback rewinds.
        {
            std::vector<int32_t> toks(40);
            for (auto& t : toks) t = (uf(rng) < 0.15f) ? kEos : tok(rng);
            std::vector<uint64_t> mono(toks.size() * 16);
            PleHistory h0;
            h0.reset(kConsts.eos);
            qwen4_ple_hash(kConsts, toks.data(), 40, h0, mono.data());

            std::vector<uint64_t> chunked(toks.size() * 16);
            PleHistory h1;
            h1.reset(kConsts.eos);
            qwen4_ple_hash(kConsts, toks.data(), 17, h1, chunked.data());
            qwen4_ple_hash(kConsts, toks.data() + 17, 23, h1, chunked.data() + 17 * 16);
            check(chunked == mono, "chunked hash (17+23) == monolithic (T=40)");

            // spec-decode rewind: save before a draft, hash the draft, rollback,
            // re-hash the true continuation -> identical to never drafting.
            PleHistory h2;
            h2.reset(kConsts.eos);
            std::vector<uint64_t> scratch(20 * 16);
            qwen4_ple_hash(kConsts, toks.data(), 20, h2, scratch.data());  // fills history
            h2.save();
            const int32_t draft[4] = {9, kEos, 8, 7};                      // rejected draft
            qwen4_ple_hash(kConsts, draft, 4, h2, scratch.data());
            h2.rollback();
            std::vector<uint64_t> resumed(8 * 16);
            qwen4_ple_hash(kConsts, toks.data() + 20, 8, h2, resumed.data());
            check(std::memcmp(resumed.data(), mono.data() + 20 * 16,
                              8 * 16 * sizeof(uint64_t)) == 0,
                  "PleHistory save/rollback rewinds a rejected draft exactly");
        }
    }

    // ---- 2. IQ4_NL row gather + dequant ------------------------------------
    std::printf("\n\033[1m2. IQ4_NL row gather (byte-exact)\033[0m\n");
    {
        constexpr uint32_t R = 64;  // synthetic table rows
        std::mt19937 rng(777);
        std::uniform_real_distribution<float> df(-0.1f, 0.1f);
        std::uniform_int_distribution<int> byte(0, 255);
        std::vector<uint8_t> table(size_t(R) * kPleRowBytes);
        for (uint32_t r = 0; r < R; ++r) {
            for (int b = 0; b < 5; ++b) {  // 5 blocks of 18 B per row
                uint8_t* blk = table.data() + size_t(r) * kPleRowBytes + size_t(b) * 18;
                const uint16_t d = fp32_to_fp16(df(rng));
                std::memcpy(blk, &d, 2);
                for (int i = 0; i < 16; ++i) blk[2 + i] = uint8_t(byte(rng));
            }
        }
        constexpr uint32_t T = 9;
        std::uniform_int_distribution<uint64_t> rowd(0, R - 1);
        std::vector<uint64_t> rows(size_t(T) * 16);
        for (auto& r : rows) r = rowd(rng);

        std::vector<float> E(size_t(T) * kPleD);
        qwen4_ple_gather(table.data(), rows.data(), T, E.data());

        bool ok = true;
        float direct[kPleHeadDim];
        for (uint32_t p = 0; p < T && ok; ++p) {
            for (uint32_t h = 0; h < 16 && ok; ++h) {
                ref::dequant_iq4_nl_buffer(
                    table.data() + rows[size_t(p) * 16 + h] * kPleRowBytes,
                    kPleHeadDim, direct);
                if (std::memcmp(direct, E.data() + size_t(p) * kPleD + size_t(h) * kPleHeadDim,
                                sizeof direct) != 0)
                    ok = false;
            }
        }
        check(ok, "E[T,2560] == per-row ref dequant, all 9x16 rows byte-exact");
    }

    // ---- 3 + 4. device layer ------------------------------------------------
    std::printf("\n\033[1m3. device blk.1 PLE layer vs CPU fp64\033[0m\n");
    sycl::queue q;
    try {
        q = sycl::queue{sycl::gpu_selector_v};
        std::printf("  device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception& e) {
        std::printf("  \033[31mno GPU: %s\033[0m\n", e.what());
        ++g_fail;
        std::printf("\n\033[1m\033[31mGATE FAILED\033[0m  (%d failures)\n\n", g_fail);
        return 1;
    }

    constexpr uint32_t D = 2560, SI = 10240;
    std::vector<float> H_mono_ref;   // reused by section 4
    std::vector<sycl::half> key64, v64;
    std::vector<float> H064, gk, gq, gc, w;
    for (uint32_t T : {1u, 5u, 64u}) {
        std::mt19937 rng(1234 + T);
        std::normal_distribution<float> nf(0.f, 1.f);
        std::normal_distribution<float> wf(0.f, 0.5f);
        std::uniform_real_distribution<float> gf(0.5f, 1.5f);
        std::vector<sycl::half> key(size_t(T) * SI), v(size_t(T) * D);
        std::vector<float> H0(size_t(T) * SI);
        gk.resize(SI); gq.resize(SI); gc.resize(SI); w.resize(4 * SI);
        for (auto& x : key) x = sycl::half(nf(rng));
        for (auto& x : v)   x = sycl::half(nf(rng));
        for (auto& x : H0)  x = nf(rng);
        for (auto& x : gk)  x = gf(rng);
        for (auto& x : gq)  x = gf(rng);
        for (auto& x : gc)  x = gf(rng);
        for (auto& x : w)   x = wf(rng);

        std::vector<double> H_ref(H0.begin(), H0.end());
        ref_ple_layer(key, v, H_ref, gk, gq, gc, w, T, 1e-6);

        const std::vector<float> H_gpu = run_device(q, key, v, H0, gk, gq, gc, w, T, {T});

        double num = 0, den = 0, max_rel = 0;
        for (size_t i = 0; i < H_gpu.size(); ++i) {
            const double d2 = double(H_gpu[i]) - H_ref[i];
            num += d2 * d2;
            den += H_ref[i] * H_ref[i];
            max_rel = std::max(max_rel, std::fabs(d2) / (1.0 + std::fabs(H_ref[i])));
        }
        const double l2 = std::sqrt(num / den);
        char what[96];
        std::snprintf(what, sizeof what,
                      "T=%-2u  l2-rel %.3e, max elem-rel %.3e  (<= 1e-4)", T, l2, max_rel);
        check(l2 <= 1e-4 && max_rel <= 1e-4, what);

        if (T == 64) {  // keep the T=64 inputs + monolithic GPU result for §4
            key64 = key; v64 = v; H064 = H0;
            H_mono_ref = H_gpu;
        }
    }

    std::printf("\n\033[1m4. chunked vs monolithic (conv-state carry)\033[0m\n");
    {
        const std::vector<std::vector<uint32_t>> splits = {
            {13, 13, 13, 13, 12}, {51, 13}, {4, 60}};
        for (const auto& sp : splits) {
            const std::vector<float> H_chunk =
                run_device(q, key64, v64, H064, gk, gq, gc, w, 64, sp);
            double max_abs = 0;
            for (size_t i = 0; i < H_chunk.size(); ++i)
                max_abs = std::max(max_abs,
                                   std::fabs(double(H_chunk[i]) - double(H_mono_ref[i])));
            char what[96], spstr[48] = {0};
            for (size_t i = 0; i < sp.size(); ++i)
                std::snprintf(spstr + std::strlen(spstr), sizeof spstr - std::strlen(spstr),
                              "%s%u", i ? "+" : "", sp[i]);
            std::snprintf(what, sizeof what,
                          "T=64 chunks {%s}: max |delta| %.3e  (<= 1e-6)", spstr, max_abs);
            check(max_abs <= 1e-6, what);
        }
    }

    std::printf("\n\033[1m%s\033[0m  (%d failure%s)\n\n",
                g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
