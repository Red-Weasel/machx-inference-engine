// tests/unit/moe_q8_decode_v2_test.cpp -- P4 B32 (2): moe_gate_up_silu_q8_v2 / moe_down_q8_v2 (src/ops/moe_q8_decode_v2.cpp)
// must write the SAME bytes as the v1 kernels, moe_gate_up_silu_q8 / moe_down_q8 under IE_Q8_MOE_DECODE_V2=0 (main()
// sets it before the first call: the switch is read once per process). These per-slot kernels run T = 1 decode and the
// request lanes' rows step (T = G rows of 8 slots, G = 2..16), so every T = 1..16 is checked at the crown's shape.
// Random Q8_0 planes (int8 qs over the full range; fp16 scales with subnormal, zero and negative ones), block_q8_1x
// activations (2 % all-zero blocks), top-8 distinct experts per row from a skewed histogram (rows share experts), fp16
// routing weights. At H = 2048 about 8 % of the gate sums sit past v1's silu clamps, so both clamp branches are compared.
//   moe_q8_decode_v2_test           -- every case: h_out (gate_up) and y_packed (down) v2 == v1 bit for bit, every element
//                                      written by both, v1 and v2 each within a host fp64 bound on sampled outputs
//   moe_q8_decode_v2_test --small   -- the small models only (a CPU device: build the kernels with IE_DP4A_PORTABLE=1)
//   moe_q8_decode_v2_test --bench   -- device time per launch (event profiling, median of 50 after 5 warm-ups) of v1 and
//                                      v2, gate_up and down, at the crown's shape for T = 1, 2, 4, 8, 16
// Models x T: crown  E=256 H=2048 E_ffn=512   T = 1..16 (gate_up: 4 K-blocks per lane, down: 1)
//             small  E=16  H=256  E_ffn=128   T = 1, 5, 16 (gate_up: lanes 8..15 idle; down: lanes 4..15 idle)
//             edge   E=12  H=160  E_ffn=96    T = 1, 3, 16 (half-empty last column chunk in both kernels)
//             wide   E=8   H=1024 E_ffn=1536  T = 1, 9 (down: 3 K-blocks per lane)
#include "ie/moe_q8.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t K_TOP = 8;

uint16_t f16_bits(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
double   f16_val(uint16_t h) { return double(float(sycl::bit_cast<sycl::half>(h))); }

void fill_q(std::vector<int8_t>& q, uint64_t n, std::mt19937& rng) {   // the full int8 range, 4 bytes per draw
    q.resize(n);
    for (uint64_t i = 0; i < n; i += 4) {
        const uint32_t r = uint32_t(rng());
        std::memcpy(q.data() + i, &r, std::min<uint64_t>(4, n - i));
    }
}
void fill_scales(std::vector<uint16_t>& d, uint64_t n, std::mt19937& rng) {
    std::uniform_int_distribution<int> pct(0, 99), sub(1, 0x3ff);
    std::uniform_real_distribution<float> dw(2e-4f, 2e-2f);
    d.resize(n);
    for (auto& v : d) {
        const int p = pct(rng);
        v = p < 2 ? uint16_t(sub(rng)) : p < 4 ? uint16_t(0) : p < 6 ? f16_bits(-dw(rng)) : f16_bits(dw(rng));
    }
}
void fill_act(std::vector<ie::block_q8_1x>& x, uint64_t n, std::mt19937& rng) {
    std::uniform_int_distribution<int> q7(-127, 127), pct(0, 99);
    std::uniform_real_distribution<float> dx(1e-3f, 5e-2f);
    x.resize(n);
    for (auto& b : x) {
        if (pct(rng) < 2) { b.d = 0.f; b.s = 0.f; std::memset(b.qs, 0, sizeof(b.qs)); continue; }
        b.d = dx(rng);
        int sum = 0;
        for (auto& q : b.qs) { q = int8_t(q7(rng)); sum += q; }
        b.s = b.d * float(sum);
    }
}

// One model's expert banks, host and device: gate / up [E][EFF][H], down [E][H][EFF], SoA qs + per-32 fp16 scales.
struct Model {
    const char* name;
    uint32_t E, H, EFF;
    std::vector<uint32_t> Ts;
    bool small;
    std::vector<int8_t> gq, uq, dq;
    std::vector<uint16_t> gd, ud, dd;
    int8_t *dgq = nullptr, *duq = nullptr, *ddq = nullptr;
    uint16_t *dgd = nullptr, *dud = nullptr, *ddd = nullptr;
    void make(sycl::queue& q, uint32_t seed) {
        std::mt19937 rng(seed);
        const uint64_t nqs = uint64_t(E) * EFF * H;
        fill_q(gq, nqs, rng); fill_q(uq, nqs, rng); fill_q(dq, nqs, rng);
        fill_scales(gd, nqs / 32, rng); fill_scales(ud, nqs / 32, rng); fill_scales(dd, nqs / 32, rng);
        dgq = sycl::malloc_device<int8_t>(nqs, q); duq = sycl::malloc_device<int8_t>(nqs, q);
        ddq = sycl::malloc_device<int8_t>(nqs, q);
        dgd = sycl::malloc_device<uint16_t>(nqs / 32, q); dud = sycl::malloc_device<uint16_t>(nqs / 32, q);
        ddd = sycl::malloc_device<uint16_t>(nqs / 32, q);
        q.memcpy(dgq, gq.data(), nqs); q.memcpy(duq, uq.data(), nqs); q.memcpy(ddq, dq.data(), nqs);
        q.memcpy(dgd, gd.data(), nqs / 32 * 2); q.memcpy(dud, ud.data(), nqs / 32 * 2);
        q.memcpy(ddd, dd.data(), nqs / 32 * 2).wait();
    }
    void release(sycl::queue& q) {
        sycl::free(dgq, q); sycl::free(duq, q); sycl::free(ddq, q);
        sycl::free(dgd, q); sycl::free(dud, q); sycl::free(ddd, q);
    }
    uint64_t gu_qs_stride() const { return uint64_t(EFF) * H; }
    uint64_t gu_d_stride() const { return uint64_t(EFF) * (H / 32); }
    uint64_t dn_qs_stride() const { return uint64_t(H) * EFF; }
    uint64_t dn_d_stride() const { return uint64_t(H) * (EFF / 32); }
};

// One step's inputs: T token rows (x), T*8 slot rows (h), the router's ids and weights
struct Step {
    uint32_t T = 0;
    std::vector<ie::block_q8_1x> x, hq;
    std::vector<int32_t> idx;
    std::vector<sycl::half> w;
    ie::block_q8_1x *dx = nullptr, *dh = nullptr;
    int32_t* didx = nullptr;
    sycl::half *dw = nullptr, *g1 = nullptr, *g2 = nullptr, *y1 = nullptr, *y2 = nullptr;
    void make(sycl::queue& q, const Model& m, uint32_t T_, uint32_t seed) {
        T = T_;
        std::mt19937 rng(seed);
        fill_act(x, uint64_t(T) * (m.H / 32), rng);
        fill_act(hq, uint64_t(T) * K_TOP * (m.EFF / 32), rng);
        // top-8 distinct experts per row, drawn with weight 1/(e+3): rows share the hot experts, as routing does
        std::vector<double> pw(m.E);
        for (uint32_t e = 0; e < m.E; ++e) pw[e] = 1.0 / (double(e) + 3.0);
        std::discrete_distribution<int> pick(pw.begin(), pw.end());
        std::uniform_real_distribution<float> rw(0.01f, 1.f);
        idx.clear(); w.clear();
        for (uint32_t t = 0; t < T; ++t) {
            std::vector<int32_t> row;
            while (row.size() < std::min<uint32_t>(K_TOP, m.E)) {
                const int32_t e = pick(rng);
                if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e);
            }
            for (int32_t e : row) { idx.push_back(e); w.push_back(sycl::half(rw(rng))); }
        }
        dx = sycl::malloc_device<ie::block_q8_1x>(x.size(), q);
        dh = sycl::malloc_device<ie::block_q8_1x>(hq.size(), q);
        didx = sycl::malloc_device<int32_t>(idx.size(), q);
        dw = sycl::malloc_device<sycl::half>(w.size(), q);
        g1 = sycl::malloc_device<sycl::half>(uint64_t(T) * K_TOP * m.EFF, q);
        g2 = sycl::malloc_device<sycl::half>(uint64_t(T) * K_TOP * m.EFF, q);
        y1 = sycl::malloc_device<sycl::half>(uint64_t(T) * K_TOP * m.H, q);
        y2 = sycl::malloc_device<sycl::half>(uint64_t(T) * K_TOP * m.H, q);
        q.memcpy(dx, x.data(), x.size() * sizeof(ie::block_q8_1x));
        q.memcpy(dh, hq.data(), hq.size() * sizeof(ie::block_q8_1x));
        q.memcpy(didx, idx.data(), idx.size() * sizeof(int32_t));
        q.memcpy(dw, w.data(), w.size() * sizeof(sycl::half)).wait();
    }
    void release(sycl::queue& q) {
        sycl::free(dx, q); sycl::free(dh, q); sycl::free(didx, q); sycl::free(dw, q);
        sycl::free(g1, q); sycl::free(g2, q); sycl::free(y1, q); sycl::free(y2, q);
    }
};

sycl::event gu(sycl::queue& q, const Model& m, const Step& s, int v) {
    auto f = v == 1 ? &ie::moe_gate_up_silu_q8 : &ie::moe_gate_up_silu_q8_v2;
    return f(q, s.dx, m.dgq, m.dgd, m.duq, m.dud, m.gu_qs_stride(), m.gu_d_stride(), s.didx, v == 1 ? s.g1 : s.g2,
             s.T, K_TOP, m.H, m.EFF);
}
sycl::event dn(sycl::queue& q, const Model& m, const Step& s, int v) {
    auto f = v == 1 ? &ie::moe_down_q8 : &ie::moe_down_q8_v2;
    return f(q, s.dh, m.ddq, m.ddd, m.dn_qs_stride(), m.dn_d_stride(), s.didx, s.dw, v == 1 ? s.y1 : s.y2, s.T, K_TOP,
             m.EFF, m.H);
}

// fp64 references: gate_up slot tk column n -> silu(g) * u (+ its fp32 error scale); down slot tk column n -> w * sum
void ref_gu(const Model& m, const Step& s, uint32_t tk, uint32_t n, double& ref, double& es) {
    const uint32_t e = uint32_t(s.idx[tk]), t = tk / K_TOP, nb = m.H / 32;
    double gs = 0, ga = 0, us = 0, ua = 0;
    for (uint32_t b = 0; b < nb; ++b) {
        const ie::block_q8_1x& xb = s.x[uint64_t(t) * nb + b];
        const uint64_t w0 = (uint64_t(e) * m.EFF + n) * m.H + uint64_t(b) * 32;
        long gi = 0, ui = 0;
        for (int k = 0; k < 32; ++k) { gi += long(m.gq[w0 + k]) * xb.qs[k]; ui += long(m.uq[w0 + k]) * xb.qs[k]; }
        const uint64_t s0 = (uint64_t(e) * m.EFF + n) * nb + b;
        const double pg = f16_val(m.gd[s0]) * double(xb.d) * double(gi);
        const double pu = f16_val(m.ud[s0]) * double(xb.d) * double(ui);
        gs += pg; ga += std::fabs(pg); us += pu; ua += std::fabs(pu);
    }
    const double sig = 1.0 / (1.0 + std::exp(-gs));
    ref = gs * sig * us;
    es = (ga * 1.1 * std::fabs(us) + ua * std::fabs(gs * sig)) * 1e-6;
}
void ref_dn(const Model& m, const Step& s, uint32_t tk, uint32_t n, double& ref, double& es) {
    const uint32_t e = uint32_t(s.idx[tk]), nb = m.EFF / 32;
    const double rw = double(float(s.w[tk]));
    double sum = 0, a = 0;
    for (uint32_t b = 0; b < nb; ++b) {
        const ie::block_q8_1x& hb = s.hq[uint64_t(tk) * nb + b];
        const uint64_t w0 = (uint64_t(e) * m.H + n) * m.EFF + uint64_t(b) * 32;
        long id = 0;
        for (int k = 0; k < 32; ++k) id += long(m.dq[w0 + k]) * hb.qs[k];
        const double p = f16_val(m.dd[(uint64_t(e) * m.H + n) * nb + b]) * double(hb.d) * double(id);
        sum += p; a += std::fabs(p);
    }
    ref = rw * sum;
    es = std::fabs(rw) * a * 1e-6;
}

struct Tally { uint64_t n = 0, diff = 0, unw1 = 0, unw2 = 0, bad1 = 0, bad2 = 0, ref_n = 0, clamped = 0; };

Tally compare(sycl::queue& q, const Model& m, const Step& s, bool gate_up, uint32_t seed) {
    const uint32_t N = gate_up ? m.EFF : m.H;
    const uint64_t n_out = uint64_t(s.T) * K_TOP * N;
    std::vector<uint16_t> a(n_out), b(n_out);
    q.memcpy(a.data(), gate_up ? s.g1 : s.y1, n_out * 2);
    q.memcpy(b.data(), gate_up ? s.g2 : s.y2, n_out * 2).wait();
    Tally r;
    r.n = n_out;
    for (uint64_t i = 0; i < n_out; ++i) {
        r.unw1 += a[i] == 0xFFFFu;
        r.unw2 += b[i] == 0xFFFFu;
        if (a[i] != b[i]) {
            if (r.diff < 3)
                std::printf("    %s differ at slot %llu col %llu: v1 0x%04x v2 0x%04x\n", gate_up ? "gate_up" : "down",
                            (unsigned long long)(i / N), (unsigned long long)(i % N), a[i], b[i]);
            ++r.diff;
        }
    }
    std::mt19937 rng(seed);
    const uint64_t samples = std::min<uint64_t>(n_out, 2048);
    for (uint64_t k = 0; k < samples; ++k) {
        const uint64_t i = n_out <= 2048 ? k : std::uniform_int_distribution<uint64_t>(0, n_out - 1)(rng);
        double ref, es;
        if (gate_up) ref_gu(m, s, uint32_t(i / N), uint32_t(i % N), ref, es);
        else         ref_dn(m, s, uint32_t(i / N), uint32_t(i % N), ref, es);
        const double tol = std::fabs(ref) / 1024.0 + es + 1e-6;
        // past fp16's largest finite value (round-to-nearest overflows at 65520) the right fp16 store is inf of the
        // reference's sign: the random data reaches -81598 at crown T=3 (B31/B32 gate), where v1 and v2 both store -inf
        const auto near = [&](uint16_t h) {
            const double v = f16_val(h);
            if (std::fabs(ref) + tol >= 65520.0 && std::isinf(v) && std::signbit(v) == std::signbit(ref)) return true;
            return std::fabs(v - ref) <= tol;
        };
        const bool ok1 = near(a[i]), ok2 = near(b[i]);
        if (!ok1) ++r.bad1;
        if (!ok2) {
            if (r.bad2 < 3) std::printf("    v2 %s off the reference at %llu: got %.6g ref %.6g tol %.3g\n",
                                        gate_up ? "gate_up" : "down", (unsigned long long)i, f16_val(b[i]), ref, tol);
            ++r.bad2;
        }
        if (gate_up) {   // the gate sum's magnitude: is this sample past the silu clamps?
            const uint32_t tk = uint32_t(i / N), n = uint32_t(i % N), e = uint32_t(s.idx[tk]), t = tk / K_TOP;
            double gs = 0;
            for (uint32_t bb = 0; bb < m.H / 32; ++bb) {
                const ie::block_q8_1x& xb = s.x[uint64_t(t) * (m.H / 32) + bb];
                const uint64_t w0 = (uint64_t(e) * m.EFF + n) * m.H + uint64_t(bb) * 32;
                long gi = 0;
                for (int kk = 0; kk < 32; ++kk) gi += long(m.gq[w0 + kk]) * xb.qs[kk];
                gs += f16_val(m.gd[(uint64_t(e) * m.EFF + n) * (m.H / 32) + bb]) * double(xb.d) * double(gi);
            }
            r.clamped += std::fabs(gs) > 105.0;
        }
        ++r.ref_n;
    }
    return r;
}

std::vector<Model> all_models() {
    std::vector<Model> v(4);
    v[0].name = "crown"; v[0].E = 256; v[0].H = 2048; v[0].EFF = 512; v[0].small = false;
    for (uint32_t t = 1; t <= 16; ++t) v[0].Ts.push_back(t);
    v[1].name = "small"; v[1].E = 16; v[1].H = 256; v[1].EFF = 128; v[1].Ts = {1, 5, 16}; v[1].small = true;
    v[2].name = "edge";  v[2].E = 12; v[2].H = 160; v[2].EFF = 96;  v[2].Ts = {1, 3, 16}; v[2].small = true;
    v[3].name = "wide";  v[3].E = 8;  v[3].H = 1024; v[3].EFF = 1536; v[3].Ts = {1, 9}; v[3].small = true;
    return v;
}

int run_check(bool small_only) {
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int fails = 0, n = 0;
    uint32_t seed = 777;
    for (Model& m : all_models()) {
        if (small_only && !m.small) continue;
        m.make(q, seed++);
        for (uint32_t T : m.Ts) {
            Step s;
            s.make(q, m, T, seed++);
            q.memset(s.g1, 0xFF, uint64_t(T) * K_TOP * m.EFF * 2);
            q.memset(s.g2, 0xFF, uint64_t(T) * K_TOP * m.EFF * 2);
            q.memset(s.y1, 0xFF, uint64_t(T) * K_TOP * m.H * 2);
            q.memset(s.y2, 0xFF, uint64_t(T) * K_TOP * m.H * 2);
            gu(q, m, s, 1); gu(q, m, s, 2); dn(q, m, s, 1); dn(q, m, s, 2);
            q.wait();
            const Tally g = compare(q, m, s, true, seed), d = compare(q, m, s, false, seed + 1);
            const bool ok = g.diff == 0 && d.diff == 0 && g.unw1 + g.unw2 + d.unw1 + d.unw2 == 0 &&
                            g.bad1 + g.bad2 + d.bad1 + d.bad2 == 0;
            std::printf("  %-6s T=%-2u  gate_up: %llu of %llu differ, unwritten %llu/%llu, ref bad %llu/%llu of %llu "
                        "(|gate|>105: %llu)   down: %llu of %llu differ, unwritten %llu/%llu, ref bad %llu/%llu of %llu  %s\n",
                        m.name, T, (unsigned long long)g.diff, (unsigned long long)g.n, (unsigned long long)g.unw1,
                        (unsigned long long)g.unw2, (unsigned long long)g.bad1, (unsigned long long)g.bad2,
                        (unsigned long long)g.ref_n, (unsigned long long)g.clamped, (unsigned long long)d.diff,
                        (unsigned long long)d.n, (unsigned long long)d.unw1, (unsigned long long)d.unw2,
                        (unsigned long long)d.bad1, (unsigned long long)d.bad2, (unsigned long long)d.ref_n,
                        ok ? "PASS" : "FAIL");
            fails += ok ? 0 : 1;
            ++n;
            s.release(q);
        }
        m.release(q);
    }
    if (fails) { std::printf("moe_q8_decode_v2_test: FAIL (%d of %d cases)\n", fails, n); return 1; }
    std::printf("moe_q8_decode_v2_test: ALL PASS (%d cases)\n", n);
    return 0;
}

double median_us(sycl::queue& q, const std::function<sycl::event()>& launch) {
    for (int i = 0; i < 5; ++i) launch();
    q.wait();
    std::vector<double> t;
    for (int i = 0; i < 50; ++i) {
        sycl::event ev = launch();
        ev.wait();
        t.push_back(double(ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                           ev.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-3);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int run_bench() {
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{},
                                                                sycl::property::queue::enable_profiling{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    Model m = all_models()[0];
    m.make(q, 99);
    for (uint32_t T : {1u, 2u, 4u, 8u, 16u}) {
        Step s;
        s.make(q, m, T, 1000 + T);
        const double g1 = median_us(q, [&] { return gu(q, m, s, 1); });
        const double g2 = median_us(q, [&] { return gu(q, m, s, 2); });
        const double d1 = median_us(q, [&] { return dn(q, m, s, 1); });
        const double d2 = median_us(q, [&] { return dn(q, m, s, 2); });
        // weight bytes each launch streams (every slot reads its expert's matrices: qs + fp16 scales)
        const double gbytes = double(T) * K_TOP * 2.0 * m.EFF * m.H * (1.0 + 2.0 / 32);
        const double dbytes = double(T) * K_TOP * 1.0 * m.EFF * m.H * (1.0 + 2.0 / 32);
        std::printf("  T=%-2u  gate_up v1 %8.1f us v2 %8.1f us (%.2fx, v2 %6.1f GB/s slot-weights)   "
                    "down v1 %8.1f us v2 %8.1f us (%.2fx, v2 %6.1f GB/s)\n", T, g1, g2, g1 / g2, gbytes / (g2 * 1e3),
                    d1, d2, d1 / d2, dbytes / (d2 * 1e3));
        s.release(q);
    }
    m.release(q);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    setenv("IE_Q8_MOE_DECODE_V2", "0", 1);   // moe_gate_up_silu_q8 / moe_down_q8 = the v1 kernels in this process
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--bench") return run_bench();
    return run_check(mode == "--small");
}
