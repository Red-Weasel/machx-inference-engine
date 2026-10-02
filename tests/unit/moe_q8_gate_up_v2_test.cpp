// tests/unit/moe_q8_gate_up_v2_test.cpp -- P4 B32 (1): moe_prefill_gate_up_silu_q8_v2 (src/ops/moe_q8_gate_up_v2.cpp) must
// write the SAME bytes as the v1 kernel, moe_prefill_gate_up_silu_q8 under IE_Q8_MOE_GATEUP_V2=0 (main() sets it before the
// first call: the switch is read once per process). Random Q8_0 gate / up planes [E][E_ffn][H] (int8 qs over the full
// range; fp16 scales with subnormal, zero and negative ones), block_q8_1x activations (2 % all-zero blocks), expert-sorted
// rows partitioned by expert_offsets as the crown split's prefill builds them. At H = 2048 about 8 % of the gate sums
// fall past v1's silu clamps (|gate| > 105), so both clamp branches are compared too.
//   moe_q8_gate_up_v2_test           -- every case: h_packed v2 == v1 bit for bit, every element written by both, and v1
//                                       and v2 each within a host fp64 bound on sampled (row, column) pairs. PASS = all 8.
//   moe_q8_gate_up_v2_test --small   -- the small cases only (a CPU device: build the kernels with IE_DP4A_PORTABLE=1)
//   moe_q8_gate_up_v2_test --bench   -- device time per launch (event profiling, median of 20 after 3 warm-ups) of v1
//                                       and v2 on the crown cases, with TOPS (4 * rows * H * E_ffn per launch)
// Cases: crown-skew  E=256 H=2048 E_ffn=512  rows per expert cycling 0/1/7/64/300/600 (41,472 rows)
//        crown-t2048 E=256 H=2048 E_ffn=512  16,384 rows (T=2048 x top-8) over a Zipf-like expert histogram
//        small-skew  E=12  H=512  E_ffn=512  rows cycling 0/1/7/64/300/600 (one K-block per lane)
//        h256        E=6   H=256  E_ffn=192  lanes 8..15 hold no K-block
//        h1024       E=6   H=1024 E_ffn=128  two K-blocks per lane
//        h2048       E=6   H=2048 E_ffn=256  four K-blocks per lane, the crown's K
//        ffn96       E=6   H=512  E_ffn=96   v2's last 64-column chunk is half empty
//        cols99      E=6   H=512  E_ffn=99   v2 only: a subgroup's second column is past E_ffn (v2's column guard);
//                    E_ffn % 32 != 0 makes v1 run a group_barrier in subgroup-divergent control flow (moe_q8.cpp:
//                    `if (n >= E_ffn) { group_barrier; continue; }`), UB by SYCL. v1 vs v2 reported, not required.
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
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

struct Case {
    const char* name;
    uint32_t E, H, EFF;
    std::vector<uint32_t> rows;   // routed rows per expert
    bool small;
    bool v2_only = false;         // v1 is outside its contract here (cols99): v1 vs v2 reported, not required
};

std::vector<uint32_t> cycle_rows(uint32_t E) {
    static const uint32_t pat[6] = {0, 1, 7, 64, 300, 600};
    std::vector<uint32_t> r(E);
    for (uint32_t e = 0; e < E; ++e) r[e] = pat[e % 6];
    return r;
}

// total slots over E experts, weight 1 / (rank + 4)^1.1 for a shuffled rank: a few hot experts, a long cold tail
std::vector<uint32_t> zipf_rows(uint32_t E, uint32_t total, uint32_t seed) {
    std::vector<uint32_t> rank(E);
    std::iota(rank.begin(), rank.end(), 0u);
    std::mt19937 rng(seed);
    std::shuffle(rank.begin(), rank.end(), rng);
    std::vector<double> w(E);
    for (uint32_t e = 0; e < E; ++e) w[e] = 1.0 / std::pow(double(rank[e]) + 4.0, 1.1);
    const double sw = std::accumulate(w.begin(), w.end(), 0.0);
    std::vector<uint32_t> r(E);
    uint32_t used = 0;
    for (uint32_t e = 0; e < E; ++e) { r[e] = uint32_t(double(total) * w[e] / sw); used += r[e]; }
    for (uint32_t e = 0; used < total; e = (e + 1) % E, ++used) ++r[e];
    return r;
}

std::vector<Case> all_cases() {
    std::vector<Case> c;
    c.push_back({"crown-skew", 256, 2048, 512, cycle_rows(256), false});
    c.push_back({"crown-t2048", 256, 2048, 512, zipf_rows(256, 2048 * 8, 31), false});
    c.push_back({"small-skew", 12, 512, 512, cycle_rows(12), true});
    c.push_back({"h256", 6, 256, 192, {0, 9, 16, 17, 40, 5}, true});
    c.push_back({"h1024", 6, 1024, 128, {3, 0, 33, 8, 1, 70}, true});
    c.push_back({"h2048", 6, 2048, 256, {1, 19, 0, 64, 7, 12}, true});
    c.push_back({"ffn96", 6, 512, 96, {2, 0, 25, 9, 64, 1}, true});
    c.push_back({"cols99", 6, 512, 99, {8, 1, 0, 17, 30, 3}, true, true});
    return c;
}

struct Host {
    std::vector<int8_t>          gq, uq;    // [E][EFF][H]
    std::vector<uint16_t>        gd, ud;    // [E][EFF][H/32] fp16 bits
    std::vector<ie::block_q8_1x> x;         // [rows][H/32]
    std::vector<uint32_t>        off;       // [E + 1]
    uint32_t                     n_rows = 0;
};

uint16_t f16_bits(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
double   f16_val(uint16_t h) { return double(float(sycl::bit_cast<sycl::half>(h))); }

Host make_host(const Case& c, uint32_t seed) {
    Host h;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> q7(-127, 127), pct(0, 99), sub(1, 0x3ff);
    std::uniform_real_distribution<float> dw(2e-4f, 2e-2f), dx(1e-3f, 5e-2f);
    const uint32_t nb = c.H / 32;
    for (auto* q : {&h.gq, &h.uq}) {                            // the full int8 range, 4 bytes per draw
        q->resize(uint64_t(c.E) * c.EFF * c.H);
        for (uint64_t i = 0; i < q->size(); i += 4) {
            const uint32_t r = uint32_t(rng());
            std::memcpy(q->data() + i, &r, 4);
        }
    }
    for (auto* d : {&h.gd, &h.ud}) {
        d->resize(uint64_t(c.E) * c.EFF * nb);
        for (auto& v : *d) {
            const int p = pct(rng);
            v = p < 2 ? uint16_t(sub(rng))                     // fp16 subnormal scale
              : p < 4 ? uint16_t(0)                            // zero scale
              : p < 6 ? f16_bits(-dw(rng))                     // negative scale
              :         f16_bits(dw(rng));
        }
    }
    h.off.assign(c.E + 1, 0);
    for (uint32_t e = 0; e < c.E; ++e) h.off[e + 1] = h.off[e] + c.rows[e];
    h.n_rows = h.off[c.E];
    h.x.resize(uint64_t(h.n_rows) * nb);
    for (auto& b : h.x) {
        if (pct(rng) < 2) { b.d = 0.f; b.s = 0.f; std::memset(b.qs, 0, sizeof(b.qs)); continue; }
        b.d = dx(rng);
        int sum = 0;
        for (auto& q : b.qs) { q = int8_t(q7(rng)); sum += q; }
        b.s = b.d * float(sum);
    }
    return h;
}

struct Dev {
    int8_t* gq = nullptr; int8_t* uq = nullptr; uint16_t* gd = nullptr; uint16_t* ud = nullptr;
    ie::block_q8_1x* x = nullptr; uint32_t* off = nullptr; sycl::half* y1 = nullptr; sycl::half* y2 = nullptr;
    Dev(sycl::queue& q, const Host& h, uint32_t EFF) {
        gq  = sycl::malloc_device<int8_t>(h.gq.size(), q);
        uq  = sycl::malloc_device<int8_t>(h.uq.size(), q);
        gd  = sycl::malloc_device<uint16_t>(h.gd.size(), q);
        ud  = sycl::malloc_device<uint16_t>(h.ud.size(), q);
        x   = sycl::malloc_device<ie::block_q8_1x>(std::max<size_t>(h.x.size(), 1), q);
        off = sycl::malloc_device<uint32_t>(h.off.size(), q);
        y1  = sycl::malloc_device<sycl::half>(std::max<uint64_t>(uint64_t(h.n_rows) * EFF, 1), q);
        y2  = sycl::malloc_device<sycl::half>(std::max<uint64_t>(uint64_t(h.n_rows) * EFF, 1), q);
        q.memcpy(gq, h.gq.data(), h.gq.size());
        q.memcpy(uq, h.uq.data(), h.uq.size());
        q.memcpy(gd, h.gd.data(), h.gd.size() * sizeof(uint16_t));
        q.memcpy(ud, h.ud.data(), h.ud.size() * sizeof(uint16_t));
        if (!h.x.empty()) q.memcpy(x, h.x.data(), h.x.size() * sizeof(ie::block_q8_1x));
        q.memcpy(off, h.off.data(), h.off.size() * sizeof(uint32_t));
        q.wait();
    }
    void free(sycl::queue& q) {
        sycl::free(gq, q); sycl::free(uq, q); sycl::free(gd, q); sycl::free(ud, q);
        sycl::free(x, q); sycl::free(off, q); sycl::free(y1, q); sycl::free(y2, q);
    }
};

sycl::event run_v1(sycl::queue& q, const Case& c, const Dev& dv) {
    return ie::moe_prefill_gate_up_silu_q8(q, dv.x, dv.gq, dv.gd, dv.uq, dv.ud, uint64_t(c.EFF) * c.H,
                                           uint64_t(c.EFF) * (c.H / 32), dv.off, dv.y1, c.E, c.H, c.EFF);
}
sycl::event run_v2(sycl::queue& q, const Case& c, const Dev& dv) {
    return ie::moe_prefill_gate_up_silu_q8_v2(q, dv.x, dv.gq, dv.gd, dv.uq, dv.ud, uint64_t(c.EFF) * c.H,
                                              uint64_t(c.EFF) * (c.H / 32), dv.off, dv.y2, c.E, c.H, c.EFF);
}

// fp64 reference of h_packed[row, n] = silu(gate) * up, and the scale of the fp32 accumulation error it may carry
void reference(const Case& c, const Host& h, uint32_t row, uint32_t n, double& ref, double& err_scale) {
    const uint32_t e = uint32_t(std::upper_bound(h.off.begin(), h.off.end(), row) - h.off.begin()) - 1;
    const uint32_t nb = c.H / 32;
    double gs = 0, ga = 0, us = 0, ua = 0;
    for (uint32_t b = 0; b < nb; ++b) {
        const ie::block_q8_1x& xb = h.x[uint64_t(row) * nb + b];
        const uint64_t w0 = (uint64_t(e) * c.EFF + n) * c.H + uint64_t(b) * 32;
        long gi = 0, ui = 0;
        for (int k = 0; k < 32; ++k) {
            gi += long(h.gq[w0 + k]) * long(xb.qs[k]);
            ui += long(h.uq[w0 + k]) * long(xb.qs[k]);
        }
        const uint64_t s0 = (uint64_t(e) * c.EFF + n) * nb + b;
        const double pg = f16_val(h.gd[s0]) * double(xb.d) * double(gi);
        const double pu = f16_val(h.ud[s0]) * double(xb.d) * double(ui);
        gs += pg; ga += std::fabs(pg); us += pu; ua += std::fabs(pu);
    }
    const double sig = 1.0 / (1.0 + std::exp(-gs));
    ref = gs * sig * us;
    // d(silu(g) u) / dg <= 1.1 |u|, d / du = |silu(g)|: fp32 sums carry ~1e-6 of their magnitude sums
    err_scale = (ga * 1.1 * std::fabs(us) + ua * std::fabs(gs * sig)) * 1e-6;
}

bool check_case(sycl::queue& q, const Case& c, uint32_t seed) {
    const Host h = make_host(c, seed);
    Dev dv(q, h, c.EFF);
    const uint64_t n_out = uint64_t(h.n_rows) * c.EFF;
    if (n_out) {
        q.memset(dv.y1, 0xFF, n_out * sizeof(sycl::half));        // 0xFFFF (NaN): "not written"
        q.memset(dv.y2, 0xFF, n_out * sizeof(sycl::half));
    }
    run_v1(q, c, dv);
    run_v2(q, c, dv);
    q.wait();
    std::vector<uint16_t> y1(n_out), y2(n_out);
    if (n_out) {
        q.memcpy(y1.data(), dv.y1, n_out * sizeof(uint16_t));
        q.memcpy(y2.data(), dv.y2, n_out * sizeof(uint16_t)).wait();
    }
    dv.free(q);

    uint64_t diff = 0, unwritten1 = 0, unwritten2 = 0;
    for (uint64_t i = 0; i < n_out; ++i) {
        unwritten1 += y1[i] == 0xFFFFu;
        unwritten2 += y2[i] == 0xFFFFu;
        if (y1[i] != y2[i]) {
            if (diff < 5)
                std::printf("    differ at row %llu col %llu: v1 0x%04x v2 0x%04x\n", (unsigned long long)(i / c.EFF),
                            (unsigned long long)(i % c.EFF), y1[i], y2[i]);
            ++diff;
        }
    }
    uint64_t ref_bad[2] = {0, 0}, ref_n = 0, clamped = 0;
    double max_rel = 0;
    std::mt19937 rng(seed ^ 0x9e3779b9u);
    const uint64_t samples = std::min<uint64_t>(n_out, 4096);
    for (uint64_t k = 0; k < samples; ++k) {
        const uint64_t i = n_out <= 4096 ? k : std::uniform_int_distribution<uint64_t>(0, n_out - 1)(rng);
        const uint32_t row = uint32_t(i / c.EFF), n = uint32_t(i % c.EFF);
        double ref, err_scale;
        reference(c, h, row, n, ref, err_scale);
        const double tol = std::fabs(ref) / 1024.0 + err_scale + 1e-6;
        for (int v = 0; v < 2; ++v) {
            const double got = f16_val(v ? y2[i] : y1[i]);
            const double err = std::fabs(got - ref);
            if (!(err <= tol)) {
                if (ref_bad[v] < 3)
                    std::printf("    v%d off the reference at row %u col %u: got %.6g ref %.6g tol %.3g\n", v + 1, row, n,
                                got, ref, tol);
                ++ref_bad[v];
            }
            if (v == 1 && std::fabs(ref) > 1e-2) max_rel = std::max(max_rel, err / std::fabs(ref));
        }
        ++ref_n;
    }
    // how many outputs sit past the silu clamps (|gate| > 105): both branches must be among the compared bytes
    {
        const uint32_t nb = c.H / 32;
        for (uint64_t i = 0; i < std::min<uint64_t>(n_out, 20000); ++i) {
            const uint32_t row = uint32_t(i / c.EFF), n = uint32_t(i % c.EFF);
            const uint32_t e = uint32_t(std::upper_bound(h.off.begin(), h.off.end(), row) - h.off.begin()) - 1;
            double gs = 0;
            for (uint32_t b = 0; b < nb; ++b) {
                const ie::block_q8_1x& xb = h.x[uint64_t(row) * nb + b];
                const uint64_t w0 = (uint64_t(e) * c.EFF + n) * c.H + uint64_t(b) * 32;
                long gi = 0;
                for (int kk = 0; kk < 32; ++kk) gi += long(h.gq[w0 + kk]) * long(xb.qs[kk]);
                gs += f16_val(h.gd[(uint64_t(e) * c.EFF + n) * nb + b]) * double(xb.d) * double(gi);
            }
            clamped += std::fabs(gs) > 105.0;
        }
    }
    const bool v1_ok = diff == 0 && unwritten1 == 0 && ref_bad[0] == 0;
    const bool ok = unwritten2 == 0 && ref_bad[1] == 0 && (v1_ok || c.v2_only);
    std::printf("  %-12s E=%-3u H=%-4u E_ffn=%-3u rows=%-6u  v2 vs v1: %llu of %llu differ; unwritten v1 %llu v2 %llu; "
                "host ref outside bound: v1 %llu v2 %llu of %llu (v2 max rel %.2e); |gate|>105: %llu of the first %llu  %s%s\n",
                c.name, c.E, c.H, c.EFF, h.n_rows, (unsigned long long)diff, (unsigned long long)n_out,
                (unsigned long long)unwritten1, (unsigned long long)unwritten2, (unsigned long long)ref_bad[0],
                (unsigned long long)ref_bad[1], (unsigned long long)ref_n, max_rel, (unsigned long long)clamped,
                (unsigned long long)std::min<uint64_t>(n_out, 20000), ok ? "PASS" : "FAIL",
                c.v2_only ? " (v2 only: v1 vs v2 not required)" : "");
    return ok;
}

double median_ms(sycl::queue& q, const std::function<sycl::event()>& launch) {
    for (int i = 0; i < 3; ++i) launch();
    q.wait();
    std::vector<double> t;
    for (int i = 0; i < 20; ++i) {
        sycl::event ev = launch();
        ev.wait();
        t.push_back(double(ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                           ev.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int run_bench() {
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{},
                                                                sycl::property::queue::enable_profiling{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    for (const Case& c : all_cases()) {
        if (c.small) continue;
        const Host h = make_host(c, 7);
        Dev dv(q, h, c.EFF);
        const double t1 = median_ms(q, [&] { return run_v1(q, c, dv); });
        const double t2 = median_ms(q, [&] { return run_v2(q, c, dv); });
        const double ops = 4.0 * double(h.n_rows) * c.H * c.EFF;   // gate + up, 2 ops per MAC
        std::printf("  %-12s rows=%-6u  v1 %8.3f ms (%5.2f TOPS)   v2 %8.3f ms (%5.2f TOPS)   %.2fx\n", c.name, h.n_rows,
                    t1, ops / (t1 * 1e9), t2, ops / (t2 * 1e9), t1 / t2);
        dv.free(q);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    setenv("IE_Q8_MOE_GATEUP_V2", "0", 1);   // moe_prefill_gate_up_silu_q8 = the v1 kernel in this process
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--bench") return run_bench();
    const bool small_only = mode == "--small";
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int fails = 0, n = 0;
    for (const Case& c : all_cases()) {
        if (small_only && !c.small) continue;
        fails += check_case(q, c, 4321 + uint32_t(n)) ? 0 : 1;
        ++n;
    }
    if (fails) { std::printf("moe_q8_gate_up_v2_test: FAIL (%d of %d cases)\n", fails, n); return 1; }
    std::printf("moe_q8_gate_up_v2_test: ALL PASS (%d cases)\n", n);
    return 0;
}
