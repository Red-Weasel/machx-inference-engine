// tools/ie_ds4_dense_prefill_bench.cpp — what does the DeepSeek-V4 T > 1 DENSE
// route actually cost, per route, at the REAL per-card shapes, with no model.
//
// WHY THIS TOOL EXISTS.  Turning `IE_DS4_DENSE_Q8` on halves prefill
// (pp512 115.88 -> 52.91 tok/s, i.e. +5,259 ms of wall on one pp512 run) and the
// only stated explanation was "at T > 1 the weight is materialised back to fp16
// for the oneDNN route".  That explanation does not survive arithmetic:
//
//   requantised elements per card per chunk        3.411e9   (derived below)
//   materialisation traffic  1.0625 B read + 2 B write        10.45 GB
//   at this card's measured streaming rate (~450-590 GB/s)   18 - 23 ms
//
// 23 ms cannot be 5,259 ms.  The gap is 229x, so the materialisation is NOT what
// runs — some MULTI-PASS route is.  There are exactly two multi-pass routes
// reachable from `dense_w`'s requantised branch, and both are entered SILENTLY
// on a null scratch pointer:
//
//   dense_f16<16>          32 passes over the weight AND an N-fold re-read of
//                          the activation: T*4*sum(N*K) = 6,986 GB from L2
//                          -> ~3,500 ms at 2 TB/s.
//   per-token GEMV loop    T=512 passes over the weight: 1,857 GB
//                          -> ~4,100 ms plus 308k launches.
//
// Both land within 1.5x of the observed 5,259 ms; the materialisation model is
// off by 229x.  Reading the code cannot separate them.  This tool measures them.
//
// It also measures the FIX: oneDNN's weight-decompression path (s8 weights, f16
// source, per-Q8_0-block f16 scales), which materialises nothing.  The brief's
// warning is taken literally — a wrong operand combination is ACCEPTED by oneDNN
// and dispatched to a reference kernel — so this tool prints the implementation
// string oneDNN actually chose (`impl_info_str`) beside every timing.  "ref:any"
// is visible, not inferred from a stopwatch.
//
// NO MODEL IS LOADED.  Weights are synthetic; only the SHAPES are real.
//
// L2 HYGIENE (an L2 artifact already produced one false 11-43x claim in this
// project).  B70 L2 is 25.17 MB.  Every route is measured over a ROTATING pool
// of distinct weight buffers totalling at least --rotate-mb (default 256 MB), so
// no repetition can read a weight its predecessor left in cache, and the first
// --warmup repetitions are discarded because an idle card sits at 400 MHz.
//
// CORRECTNESS IS GATED HERE TOO.  --check runs every route against a host
// DOUBLE-precision reference of the EXACT Q8_0-SoA weight values, under a bound
// derived from the error model of each route, and runs three negative controls
// that prove the bound discriminates.  A scale plane read with the wrong stride
// is the single most likely way the decompression route can be fast and wrong;
// the controls target exactly that.

#include "ie/deepseek4.hpp"
#include "ie/ds4_decode_gemv.hpp"
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using ie::DType;

double now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// THE REAL PER-CARD DENSE CALL SET AT T > 1
// ---------------------------------------------------------------------------
// Shapes from docs/deepseek4/01_gguf_manifest.md §3 (verified against the file);
// the per-card split from src/model/deepseek4.cpp `upload_layer` — attn_q_b,
// attn_output_a and ffn_{gate,up}_shexp split by ROWS, attn_output_b and
// ffn_down_shexp by COLUMNS, everything else mirrored.  Layer multiplicities
// from the manifest's presence map: 43 layers, compressors on 41 of them at two
// different ranks (ratio 128 on the 20 odd layers, ratio 4 on the 21 even ones
// from layer 2), indexer on those same 21.
//
// THE CHECK THAT THIS TABLE IS THE REAL ONE, and it is exact rather than
// approximate.  The rows below sum to 3,412,066,304 elements per card.  They are
// the REQUANTISED set only, so they deliberately exclude the three always-
// resident families that never become Q8_0: `indexer_compressor_kv` and
// `indexer_compressor_gate` ([4096, 256] each, mirrored, on 21 layers, held wide
// by call-site policy) and `indexer.proj` ([4096, 64], F32).  Those add
// 2*1,048,576*21 + 262,144*21 = 49,545,216 elements, for a total of
// 3,461,611,520 = 6.923 GB of fp16.
//
// src/model/deepseek4.cpp records the MEASURED per-card dense weight stream as
// 6.92 GB.  The table reproduces it to 0.05% from shapes and split rules alone,
// which is what makes it a derivation rather than a guess.  Two independent
// spot-checks agree as well: q_b, o_a and o_b all come out at 16.777 M elements
// per card, which is the "16.78 M elements = 33.6 MB" the fp16 staging buffer in
// deepseek4.cpp is sized by.
//
// The LM head is deliberately ABSENT: `forward_epilogue` passes TL = 1, so it is
// never a T > 1 call.  (It is, however, still counted by the load-time scratch
// reservation — see the note this tool's output prints.)
struct Shape {
    const char* name;
    uint32_t    K, N;
    uint32_t    count;      // calls per chunk per card
};

const Shape kShapes[] = {
    // name                    K       N   count
    {"attn_q_a",            4096,   1024,    43},
    {"attn_q_b (R)",        1024,  16384,    43},
    {"attn_kv",             4096,    512,    43},
    {"attn_output_a (R)",   4096,   4096,    43},
    {"attn_output_b (C)",   4096,   4096,    43},
    {"ffn_gate_shexp (R)",  4096,   1024,    43},
    {"ffn_up_shexp (R)",    4096,   1024,    43},
    {"ffn_down_shexp (C)",  1024,   4096,    43},
    {"attn_compressor r128",4096,    512,    40},   // gate+kv on 20 odd layers
    {"attn_compressor r4",  4096,   1024,    42},   // gate+kv on 21 even layers
    {"indexer.attn_q_b",    1024,   8192,    21},
};

struct Opts {
    uint32_t device    = 0;
    uint32_t T         = 512;
    uint32_t reps      = 8;
    uint32_t warmup    = 3;
    uint32_t rotate_mb = 256;
    bool     check     = false;
    bool     pertoken  = false;    // the 512-launch fallback; slow on purpose
};

// ---------------------------------------------------------------------------
// One shape's device state.  Allocated, measured and freed per shape so peak
// VRAM stays near half a gigabyte rather than the 7 GB the whole set would cost.
// ---------------------------------------------------------------------------
struct Bufs {
    uint32_t              K = 0, N = 0, R = 0;
    std::vector<sycl::half*> w16;      // R rotating fp16 weights   [N, K]
    std::vector<int8_t*>     qs;       // R rotating int8 planes    [N][K]
    std::vector<sycl::half*> qd;       // R rotating scale planes   [N][K/32]
    float*      x   = nullptr;         // [T, K] fp32 activation
    sycl::half* a16 = nullptr;         // [T, K] fp16 activation
    float*      y   = nullptr;         // [T, N] fp32 output
    sycl::half* mat = nullptr;         // [N, K] materialisation target
};

void free_bufs(sycl::queue& q, Bufs& b) {
    q.wait();
    for (auto* p : b.w16) if (p) sycl::free(p, q);
    for (auto* p : b.qs)  if (p) sycl::free(p, q);
    for (auto* p : b.qd)  if (p) sycl::free(p, q);
    if (b.x)   sycl::free(b.x, q);
    if (b.a16) sycl::free(b.a16, q);
    if (b.y)   sycl::free(b.y, q);
    if (b.mat) sycl::free(b.mat, q);
    b = Bufs{};
}

// Fills R DISTINCT weights so that a rotation cannot be served from L2, and so
// that a route which silently reads the wrong buffer produces a wrong answer
// rather than the right one by luck.
bool make_bufs(sycl::queue& q, uint32_t K, uint32_t N, uint32_t T, uint32_t rotate_mb,
               Bufs& b, std::string& err) {
    const uint64_t elems = uint64_t(K) * N;
    const uint64_t f16_b = elems * 2;
    // Enough copies that the rotating fp16 working set clears --rotate-mb, which
    // must clear L2 (25.17 MB) by a wide margin for the timing to mean anything.
    uint32_t R = uint32_t((uint64_t(rotate_mb) << 20) / std::max<uint64_t>(f16_b, 1)) + 1;
    R = std::min<uint32_t>(std::max<uint32_t>(R, 2), 16);
    b.K = K; b.N = N; b.R = R;

    std::mt19937 rng(0x5eed'1234u ^ (K * 2654435761u) ^ N);
    std::normal_distribution<float> nd(0.f, 0.02f);

    std::vector<float>       hw(elems);
    std::vector<sycl::half>  hf(elems);
    std::vector<int8_t>      hq(elems);
    std::vector<sycl::half>  hd(uint64_t(N) * (K / 32));

    for (uint32_t r = 0; r < R; ++r) {
        for (uint64_t i = 0; i < elems; ++i) hw[i] = nd(rng);
        // The Q8_0-SoA form is built by THE ENGINE'S OWN quantiser, so this tool
        // cannot disagree with the engine about what the stored weight means.
        ie::ds4_requantise_q8_soa(hw.data(), K, N, hq.data(), hd.data());
        // ...and the fp16 arm is given the SAME weight, decoded from those very
        // bytes.  Otherwise "fp16 vs Q8" would also be "weight A vs weight B" and
        // no timing comparison between them would be clean.
        for (uint64_t i = 0; i < elems; ++i) {
            const uint32_t n = uint32_t(i / K), k = uint32_t(i - uint64_t(n) * K);
            hf[i] = sycl::half(float(hd[uint64_t(n) * (K / 32) + (k >> 5)]) * float(hq[i]));
        }
        sycl::half* pw = sycl::malloc_device<sycl::half>(elems, q);
        int8_t*     pq = sycl::malloc_device<int8_t>(elems, q);
        sycl::half* pd = sycl::malloc_device<sycl::half>(uint64_t(N) * (K / 32), q);
        if (!pw || !pq || !pd) { err = "malloc_device failed for a rotating weight"; return false; }
        q.memcpy(pw, hf.data(), elems * sizeof(sycl::half)).wait();
        q.memcpy(pq, hq.data(), elems).wait();
        q.memcpy(pd, hd.data(), hd.size() * sizeof(sycl::half)).wait();
        b.w16.push_back(pw); b.qs.push_back(pq); b.qd.push_back(pd);
    }

    std::vector<float> hx(uint64_t(T) * K);
    std::normal_distribution<float> xd(0.f, 1.0f);
    for (auto& v : hx) v = xd(rng);
    b.x   = sycl::malloc_device<float>(uint64_t(T) * K, q);
    b.a16 = sycl::malloc_device<sycl::half>(uint64_t(T) * K, q);
    b.y   = sycl::malloc_device<float>(uint64_t(T) * N, q);
    b.mat = sycl::malloc_device<sycl::half>(elems, q);
    if (!b.x || !b.a16 || !b.y || !b.mat) { err = "malloc_device failed for a workspace"; return false; }
    q.memcpy(b.x, hx.data(), hx.size() * sizeof(float)).wait();
    ie::cast_fp32_to_fp16(q, b.x, b.a16, uint64_t(T) * K).wait();
    return true;
}

struct Timing {
    double   ms_per_call = 0;      // median of the timed repetitions
    uint64_t launches    = 0;      // kernel submissions per call
    uint64_t bytes       = 0;      // weight bytes read per call
    bool     ran         = false;
};

double median(std::vector<double>& v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

// Every route is timed the same way: warmup repetitions discarded, each timed
// repetition wrapped in a queue drain so nothing leaks into the next, and the
// weight rotated every repetition.
template <typename Fn>
Timing time_route(sycl::queue& q, const Opts& o, uint64_t launches, uint64_t bytes, Fn&& fn) {
    Timing t;
    t.launches = launches;
    t.bytes    = bytes;
    for (uint32_t i = 0; i < o.warmup; ++i) { fn(i); q.wait(); }
    std::vector<double> s;
    for (uint32_t i = 0; i < o.reps; ++i) {
        const double t0 = now_s();
        fn(o.warmup + i);
        q.wait();
        s.push_back((now_s() - t0) * 1e3);
    }
    t.ms_per_call = median(s);
    t.ran = true;
    return t;
}

// ---------------------------------------------------------------------------
// CORRECTNESS — every route against a host double reference of the EXACT stored
// Q8_0-SoA weight, with a bound derived per route and printed beside it.
// ---------------------------------------------------------------------------
//
// The reference is deliberately built from the int8 plane and the fp16 scale
// plane rather than from the pre-quantisation floats: the question a route must
// answer is "did you compute the matmul of the weight that is STORED", not "did
// you undo the quantisation".  Comparing against the pre-quantisation weight
// would fold the quantisation error into every route equally and hide a real
// indexing bug underneath it.
struct CheckResult {
    double max_abs = 0;
    double bound   = 0;
    bool   pass    = false;
};

CheckResult check_route(const std::vector<float>& y, const std::vector<double>& ref,
                        const std::vector<double>& absum, double eps_mul, uint32_t K) {
    CheckResult r;
    // Error model.  Each output is sum_k a_k * w_nk accumulated in fp32.
    //   eps_mul   the per-product relative error the route admits: 2^-11 for a
    //             route that rounds the activation to fp16 (both oneDNN routes),
    //             plus another 2^-11 for a route that ALSO rounds the decoded
    //             weight to fp16 (the materialisation route), 0 for neither.
    //   K*2^-24   the fp32 accumulation, worst case one rounding per term.
    // The bound is against |a_k * w_nk| summed, which is what a rounding can
    // actually act on; using |y| instead would be unbounded near cancellation.
    const double acc = double(K) * std::ldexp(1.0, -24);
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = std::fabs(double(y[i]) - ref[i]);
        r.max_abs = std::max(r.max_abs, e);
        r.bound   = std::max(r.bound, absum[i] * (eps_mul + acc));
    }
    r.pass = r.max_abs <= r.bound;
    return r;
}

int run_check(sycl::queue& q) {
    // Small enough to reference on the host in double, large enough that K/32
    // (the scale-plane row length) differs from N — so a scale plane read with a
    // transposed stride cannot accidentally be shape-compatible.
    const uint32_t K = 256, N = 40, T = 6;
    const uint64_t elems = uint64_t(K) * N, nb = K / 32;
    std::mt19937 rng(20260803u);
    std::normal_distribution<float> nd(0.f, 0.05f), xd(0.f, 1.f);

    std::vector<float>      hw(elems), hx(uint64_t(T) * K);
    std::vector<int8_t>     hq(elems);
    std::vector<sycl::half> hd(uint64_t(N) * nb);
    // Per-ROW and per-BLOCK scale variation is the point: with a flat scale a
    // transposed or off-by-a-block scale index would still give the right answer.
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t k = 0; k < K; ++k)
            hw[uint64_t(n) * K + k] = nd(rng) * (1.0f + 0.35f * float(n)) *
                                      (1.0f + 0.20f * float(k / 32));
    for (auto& v : hx) v = xd(rng);
    ie::ds4_requantise_q8_soa(hw.data(), K, N, hq.data(), hd.data());

    // The exact stored weight, in double.
    auto stored = [&](uint32_t n, uint32_t k) -> double {
        return double(float(hd[uint64_t(n) * nb + (k >> 5)])) * double(hq[uint64_t(n) * K + k]);
    };
    std::vector<double> ref(uint64_t(T) * N), absum(uint64_t(T) * N);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t n = 0; n < N; ++n) {
            double s = 0, a = 0;
            for (uint32_t k = 0; k < K; ++k) {
                const double p = double(hx[uint64_t(t) * K + k]) * stored(n, k);
                s += p; a += std::fabs(p);
            }
            ref[uint64_t(t) * N + n]   = s;
            absum[uint64_t(t) * N + n] = a;
        }

    float*      dx  = sycl::malloc_device<float>(uint64_t(T) * K, q);
    sycl::half* da  = sycl::malloc_device<sycl::half>(uint64_t(T) * K, q);
    int8_t*     dq  = sycl::malloc_device<int8_t>(elems, q);
    sycl::half* dd  = sycl::malloc_device<sycl::half>(hd.size(), q);
    sycl::half* dw  = sycl::malloc_device<sycl::half>(elems, q);
    float*      dy  = sycl::malloc_device<float>(uint64_t(T) * N, q);
    if (!dx || !da || !dq || !dd || !dw || !dy) {
        std::printf("[dpb] check: FAIL malloc_device\n");
        return 1;
    }
    q.memcpy(dx, hx.data(), hx.size() * sizeof(float)).wait();
    q.memcpy(dq, hq.data(), elems).wait();
    q.memcpy(dd, hd.data(), hd.size() * sizeof(sycl::half)).wait();
    ie::cast_fp32_to_fp16(q, dx, da, uint64_t(T) * K).wait();

    std::vector<float> hy(uint64_t(T) * N);
    int fails = 0;
    auto report = [&](const char* what, const CheckResult& r) {
        std::printf("[dpb] check.%-26s max_abs=%.3e bound=%.3e ratio=%.3f %s\n",
                    what, r.max_abs, r.bound,
                    r.bound > 0 ? r.max_abs / r.bound : 0.0, r.pass ? "PASS" : "FAIL");
        if (!r.pass) ++fails;
    };

    // ---- route B: materialise to fp16, then the fp16 NT matmul --------------
    // Admits TWO fp16 roundings per product: the activation and the decoded
    // weight (`half(d * qs)` is rounded before oneDNN ever sees it).
    ie::ds4_q8_soa_to_f16(q, dq, dd, dw, K, N).wait();
    ie::gemm_nt_f16_onednn(q, da, dw, dy, T, N, K).wait();
    q.memcpy(hy.data(), dy, hy.size() * sizeof(float)).wait();
    report("q8_materialise_onednn", check_route(hy, ref, absum, 2.0 * std::ldexp(1.0, -11), K));

    // ---- route E: oneDNN weight decompression, nothing materialised --------
    // Admits ONE fp16 rounding for the activation.  oneDNN's f16 fpmath also
    // rounds the up-converted weight/product, so the same 2^-11 is allowed twice
    // here as well — this bound is not tighter than route B's, it is the same
    // bound, which is the honest statement: the two routes are numerically
    // equivalent to within the model, and the check discriminates INDEXING, not
    // a fourth-decimal accuracy claim.
    const std::string impl = ie::onednn_nt_s8_impl(q, T, N, K, 32);
    std::printf("[dpb] check.s8_impl = \"%s\"\n", impl.c_str());
    // oneDNN consumes attr scales in PLAIN kb-major [K/32][N] layout and
    // ignores custom strides on the scales md — this arm passes the kb-major
    // plane, exactly as `dense_w` now passes `Ds4Dense::q8dt`.  (Until
    // 2026-08-08 this arm passed the n-major plane and FAILed at 125x the
    // bound; the "transposed" control passed.  That inversion WAS the bug.)
    std::vector<sycl::half> htr(hd.size());
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t b = 0; b < nb; ++b)
            htr[uint64_t(b) * N + n] = hd[uint64_t(n) * nb + b];
    sycl::half* ddt = sycl::malloc_device<sycl::half>(htr.size(), q);
    q.memcpy(ddt, htr.data(), htr.size() * sizeof(sycl::half)).wait();
    sycl::event ev;
    if (ie::gemm_nt_s8_onednn(q, da, dq, ddt, dy, T, N, K, 32, {}, &ev)) {
        ev.wait();
        q.memcpy(hy.data(), dy, hy.size() * sizeof(float)).wait();
        report("q8_s8_decompress_onednn", check_route(hy, ref, absum, 2.0 * std::ldexp(1.0, -11), K));

        // ---- NEGATIVE CONTROL 1: the n-major (GEMV-layout) scale plane -----
        // The single most likely silent failure of this route, and the one it
        // actually shipped with: feeding the n-major [N][K/32] plane the decode
        // GEMV reads must FAIL the same bound the kb-major layout passes; if it
        // does not, the bound is measuring nothing and neither is the PASS
        // above.
        {
            sycl::event e2;
            ie::gemm_nt_s8_onednn(q, da, dq, dd, dy, T, N, K, 32, {}, &e2);
            e2.wait();
            q.memcpy(hy.data(), dy, hy.size() * sizeof(float)).wait();
            const CheckResult r = check_route(hy, ref, absum, 2.0 * std::ldexp(1.0, -11), K);
            std::printf("[dpb] control.nmajor_scales          max_abs=%.3e bound=%.3e %s\n",
                        r.max_abs, r.bound, r.pass ? "FAIL (bound does not discriminate)"
                                                   : "PASS (rejected, as it must be)");
            if (r.pass) ++fails;
        }

        // ---- NEGATIVE CONTROL 2: one scale block perturbed -----------------
        // Proves the bound is tight enough to see a SINGLE wrong block, not just
        // a wholesale layout error.  Perturbs the kb-major plane the main arm
        // passes, so a rejection here is the block, not the layout.
        {
            std::vector<sycl::half> pd2 = htr;
            pd2[uint64_t(nb / 2) * N + (N / 2)] = sycl::half(float(pd2[uint64_t(nb / 2) * N + (N / 2)]) * 1.5f);
            sycl::half* dp = sycl::malloc_device<sycl::half>(pd2.size(), q);
            q.memcpy(dp, pd2.data(), pd2.size() * sizeof(sycl::half)).wait();
            sycl::event e3;
            ie::gemm_nt_s8_onednn(q, da, dq, dp, dy, T, N, K, 32, {}, &e3);
            e3.wait();
            q.memcpy(hy.data(), dy, hy.size() * sizeof(float)).wait();
            const CheckResult r = check_route(hy, ref, absum, 2.0 * std::ldexp(1.0, -11), K);
            std::printf("[dpb] control.one_block_scaled_1.5x  max_abs=%.3e bound=%.3e %s\n",
                        r.max_abs, r.bound, r.pass ? "FAIL (bound does not discriminate)"
                                                   : "PASS (rejected, as it must be)");
            if (r.pass) ++fails;
            sycl::free(dp, q);
        }

        // ---- NEGATIVE CONTROL 3: the reference itself --------------------
        // A bound that rejects everything would pass the two controls above for
        // the wrong reason.  Feeding the EXACT stored weight through the fp16
        // route must PASS, and it is reported here so the two rejections cannot
        // be read as "the bound is simply tiny".
        {
            ie::gemm_nt_f16_onednn(q, da, dw, dy, T, N, K).wait();
            q.memcpy(hy.data(), dy, hy.size() * sizeof(float)).wait();
            const CheckResult r = check_route(hy, ref, absum, 2.0 * std::ldexp(1.0, -11), K);
            std::printf("[dpb] control.exact_weight_must_pass max_abs=%.3e bound=%.3e %s\n",
                        r.max_abs, r.bound, r.pass ? "PASS" : "FAIL (bound is too tight)");
            if (!r.pass) ++fails;
        }
    } else {
        std::printf("[dpb] check.q8_s8_decompress_onednn  NOT RUN — oneDNN built no primitive\n");
    }

    // ---- route C: the per-token GEMV fallback, for completeness ------------
    // It is the route `dense_w` silently takes when the fp16 staging buffer is
    // null, so its ACCURACY matters even though its cost is the point.
    {
        sycl::event e;
        for (uint32_t t = 0; t < T; ++t)
            e = ie::ds4_decode_gemv_q8(q, "dpb_gemv", dx + uint64_t(t) * K, dq, dd,
                                       dy + uint64_t(t) * N, K, N,
                                       t == 0 ? std::vector<sycl::event>{}
                                              : std::vector<sycl::event>{e});
        e.wait();
        q.memcpy(hy.data(), dy, hy.size() * sizeof(float)).wait();
        // fp32 activation, fp32 accumulate: no fp16 product rounding at all.
        report("q8_pertoken_gemv", check_route(hy, ref, absum, 0.0, K));
    }

    for (void* p : {(void*)dx, (void*)da, (void*)dq, (void*)dd, (void*)ddt, (void*)dw, (void*)dy})
        sycl::free(p, q);
    std::printf("[dpb] check.failures = %d\n", fails);
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if      (s == "--device")    o.device    = uint32_t(std::atoi(next().c_str()));
        else if (s == "--T")         o.T         = uint32_t(std::atoi(next().c_str()));
        else if (s == "--reps")      o.reps      = uint32_t(std::atoi(next().c_str()));
        else if (s == "--warmup")    o.warmup    = uint32_t(std::atoi(next().c_str()));
        else if (s == "--rotate-mb") o.rotate_mb = uint32_t(std::atoi(next().c_str()));
        else if (s == "--check")     o.check     = true;
        else if (s == "--pertoken")  o.pertoken  = true;
        else if (s == "-h" || s == "--help") {
            std::printf(
                "ie-ds4-dense-prefill-bench — per-route cost of the DeepSeek-V4 T>1 dense\n"
                "projections at the real per-card shapes.  NO MODEL IS LOADED.\n\n"
                "  --device <n>     GPU ordinal (default 0)\n"
                "  --T <n>          token tile (default 512 = the pp512 chunk)\n"
                "  --reps <n>       timed repetitions per route (default 8)\n"
                "  --warmup <n>     discarded repetitions (default 3; an idle B70\n"
                "                   sits at 400 MHz and the first 1-2 read 3-6x slow)\n"
                "  --rotate-mb <n>  rotating weight pool per shape (default 256; must\n"
                "                   clear the 25.17 MB L2 by a wide margin)\n"
                "  --check          numerics + negative controls, then exit\n"
                "  --pertoken       ALSO time the T-launch per-token GEMV fallback\n"
                "                   (slow on purpose: it is what `dense_w` silently\n"
                "                    runs when the fp16 staging buffer is null)\n");
            return 0;
        }
    }

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            devs.push_back(d);
    if (o.device >= devs.size()) {
        std::fprintf(stderr, "[dpb] device %u requested, %zu GPUs visible\n", o.device, devs.size());
        return 2;
    }
    sycl::queue q(devs[o.device], sycl::property_list{sycl::property::queue::in_order{}});
    std::printf("[dpb] device = %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("[dpb] onednn_available = %d\n", int(ie::onednn_available()));
    std::printf("[dpb] global_mem_MB = %llu  L2_KB = %llu\n",
                (unsigned long long)(q.get_device().get_info<sycl::info::device::global_mem_size>() >> 20),
                (unsigned long long)(q.get_device().get_info<sycl::info::device::global_mem_cache_size>() >> 10));

    if (o.check) return run_check(q);

    std::printf("[dpb] T=%u reps=%u warmup=%u rotate_mb=%u\n", o.T, o.reps, o.warmup, o.rotate_mb);
    std::printf("[dpb] %-22s %8s %8s %10s %10s %10s %10s\n",
                "shape", "K", "N", "f16+dnn", "q8mat+dnn", "q8s8+dnn", "pertoken");
    std::printf("[dpb] %-22s %8s %8s %10s %10s %10s %10s\n",
                "", "", "", "ms/call", "ms/call", "ms/call", "ms/call");

    // Per-chunk-per-card totals, which is the number that has to be compared
    // against the 5,259 ms the real model actually lost.
    double tot_f16 = 0, tot_mat = 0, tot_s8 = 0, tot_pt = 0;
    uint64_t tot_calls = 0, tot_elems = 0;
    uint64_t tot_launch_f16 = 0, tot_launch_mat = 0, tot_launch_pt = 0;
    bool any_s8 = false;
    std::string s8_impl_seen;

    for (const Shape& sh : kShapes) {
        Bufs b;
        std::string err;
        if (!make_bufs(q, sh.K, sh.N, o.T, o.rotate_mb, b, err)) {
            std::printf("[dpb] %-22s SKIPPED: %s\n", sh.name, err.c_str());
            free_bufs(q, b);
            continue;
        }
        const uint64_t elems = uint64_t(sh.K) * sh.N;

        // Route A — fp16 weight straight into the NT matmul.  This is what
        // IE_DS4_DENSE_Q8=0 runs and is the bar every other route is judged by.
        Timing tA = time_route(q, o, 1, elems * 2, [&](uint32_t i) {
            ie::gemm_nt_f16_onednn(q, b.a16, b.w16[i % b.R], b.y, o.T, sh.N, sh.K);
        });

        // Route B — the CURRENT IE_DS4_DENSE_Q8=1 route: expand the Q8_0-SoA
        // planes into fp16, then the same matmul.
        Timing tB = time_route(q, o, 2, elems * 2 + elems * 17 / 16, [&](uint32_t i) {
            auto e = ie::ds4_q8_soa_to_f16(q, b.qs[i % b.R], b.qd[i % b.R], b.mat, sh.K, sh.N);
            ie::gemm_nt_f16_onednn(q, b.a16, b.mat, b.y, o.T, sh.N, sh.K, {e});
        });

        // Route E — oneDNN weight decompression.  Nothing is materialised.
        const std::string impl = ie::onednn_nt_s8_impl(q, o.T, sh.N, sh.K, 32);
        Timing tE;
        {
            sycl::event probe;
            if (ie::gemm_nt_s8_onednn(q, b.a16, b.qs[0], b.qd[0], b.y, o.T, sh.N, sh.K, 32, {}, &probe)) {
                probe.wait();
                any_s8 = true;
                if (s8_impl_seen.empty()) s8_impl_seen = impl;
                tE = time_route(q, o, 1, elems * 17 / 16, [&](uint32_t i) {
                    sycl::event e;
                    ie::gemm_nt_s8_onednn(q, b.a16, b.qs[i % b.R], b.qd[i % b.R], b.y,
                                          o.T, sh.N, sh.K, 32, {}, &e);
                });
            }
        }

        // Route C — the per-token GEMV fallback.  One launch per token, each
        // re-reading the whole weight; opt-in because it is minutes, not ms.
        Timing tC;
        if (o.pertoken) {
            tC = time_route(q, o, o.T, elems * 17 / 16 * o.T, [&](uint32_t i) {
                sycl::event e;
                for (uint32_t t = 0; t < o.T; ++t)
                    e = ie::ds4_decode_gemv_q8(q, "dpb_gemv", b.x + uint64_t(t) * sh.K,
                                               b.qs[i % b.R], b.qd[i % b.R],
                                               b.y + uint64_t(t) * sh.N, sh.K, sh.N,
                                               t == 0 ? std::vector<sycl::event>{}
                                                      : std::vector<sycl::event>{e});
            });
        }

        // Weight bytes / ms, i.e. the rate each route sustains against the bytes
        // it is OBLIGED to read.  It is the number that says whether a route is
        // at the streaming roofline or an order of magnitude under it, which
        // ms/call alone cannot.
        auto gbs = [](const Timing& t) {
            return t.ran && t.ms_per_call > 0 ? double(t.bytes) / (t.ms_per_call * 1e6) : 0.0;
        };
        const std::string se = tE.ran ? std::to_string(tE.ms_per_call).substr(0, 8) : "n/a";
        const std::string sc = tC.ran ? std::to_string(tC.ms_per_call).substr(0, 8) : "-";
        std::printf("[dpb] %-22s %8u %8u %10.3f %10.3f %10s %10s   R=%u impl=%s\n",
                    sh.name, sh.K, sh.N, tA.ms_per_call, tB.ms_per_call,
                    se.c_str(), sc.c_str(), b.R, impl.c_str());
        std::printf("[dpb]   %-20s launches/call f16=%llu q8mat=%llu q8s8=%llu pertoken=%llu | "
                    "GB/s f16=%.0f q8mat=%.0f q8s8=%.0f\n",
                    "", (unsigned long long)tA.launches, (unsigned long long)tB.launches,
                    (unsigned long long)(tE.ran ? tE.launches : 0),
                    (unsigned long long)(tC.ran ? tC.launches : 0),
                    gbs(tA), gbs(tB), gbs(tE));

        tot_f16 += tA.ms_per_call * sh.count;
        tot_mat += tB.ms_per_call * sh.count;
        tot_s8  += (tE.ran ? tE.ms_per_call : tA.ms_per_call) * sh.count;
        tot_pt  += (tC.ran ? tC.ms_per_call : 0.0) * sh.count;
        tot_calls   += sh.count;
        tot_elems   += elems * sh.count;
        tot_launch_f16 += uint64_t(tA.launches) * sh.count;
        tot_launch_mat += uint64_t(tB.launches) * sh.count;
        tot_launch_pt  += uint64_t(tC.ran ? tC.launches : 0) * sh.count;
        free_bufs(q, b);
    }

    std::printf("\n[dpb] ---- projected PER CHUNK PER CARD at T=%u ----\n", o.T);
    std::printf("[dpb] dense calls              = %llu\n", (unsigned long long)tot_calls);
    std::printf("[dpb] dense elements           = %.4fe9  (fp16 %.3f GB, Q8_0-SoA %.3f GB)\n",
                double(tot_elems) / 1e9, double(tot_elems) * 2 / 1e9,
                double(tot_elems) * 17.0 / 16.0 / 1e9);
    std::printf("[dpb] weight bytes read        = fp16 %.3f GB, Q8_0-SoA %.3f GB, "
                "materialise moves %.3f GB (read+write)\n",
                double(tot_elems) * 2 / 1e9, double(tot_elems) * 17.0 / 16.0 / 1e9,
                double(tot_elems) * (17.0 / 16.0 + 2.0) / 1e9);
    std::printf("[dpb] kernel launches          = f16 %llu, materialise %llu, per-token %llu\n",
                (unsigned long long)tot_launch_f16, (unsigned long long)tot_launch_mat,
                (unsigned long long)tot_launch_pt);
    std::printf("[dpb] route f16 + oneDNN       = %10.1f ms\n", tot_f16);
    std::printf("[dpb] route Q8 materialise+dnn = %10.1f ms   (delta vs f16 %+.1f ms)\n",
                tot_mat, tot_mat - tot_f16);
    if (any_s8)
        std::printf("[dpb] route Q8 s8-decompress   = %10.1f ms   (delta vs f16 %+.1f ms)  impl=%s\n",
                    tot_s8, tot_s8 - tot_f16, s8_impl_seen.c_str());
    else
        std::printf("[dpb] route Q8 s8-decompress   =        n/a   oneDNN built no primitive\n");
    if (o.pertoken)
        std::printf("[dpb] route Q8 per-token GEMV  = %10.1f ms   (delta vs f16 %+.1f ms)\n",
                    tot_pt, tot_pt - tot_f16);
    std::printf("[dpb] the real-model gap to explain is +5259 ms of pp512 wall "
                "(115.88 -> 52.91 tok/s, one 512-token chunk).\n");
    return 0;
}
