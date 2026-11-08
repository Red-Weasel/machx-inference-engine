// tools/qwen4_bank_test.cpp -- qwen4exp expert-bank kernel gate for Q8_0, Q5_K, Q6_K, Q5_0 and Q4_0 banks
// (include/ie/qwen4_bank.hpp). No model file needed. For every kind, on random blocks (fixed seed; scales kept small and
// finite), at the down shape (K=640, the 32-block kinds), the gate/up shape (K=2560) and an N=7 edge:
//   1. the solo GEMV vs a CPU fp64 reference built on dequant_ref.hpp (max rel err <= 2e-3, the Q5_1 gate's rule), and
//      unit-vector activations reproducing the dequantised columns (<= 1e-3 rel: the half store);
//   2. the grouped down kernel, the grouped dual gate/up kernel, the grouped tiles (rows 1 / 5 / 16, both mats) and the
//      batched rows (M = 33) BIT-IDENTICAL to the solo kernel per output, over a synthetic [K, N, E] bank addressed by
//      expert slot (slot_base + slot * slot_bytes + off);
//   3. qwen4_bank_kind's table: the dtypes and the K alignment each kind takes;
//   4. (device routing phase 2) every grouped pick kernel -- these kinds' dual / dual+swiglu / down / int-dot, the Q4_K
//      int-dot dual / dual+swiglu, the Q5_1 down -- through the slot address table, its rows a VRAM copy of the slots and
//      three pinned host banks: bit-identical to the slot form.
// Rel err denominator: max(|ref|, 0.01 * max|ref| over the output) for the random-x GEMVs; max(|ref|, 1e-6) otherwise.
// usage: ie-qwen4-bank-test          exit 0 = GATE PASSED, 1 = FAILED
#include "ie/qwen4_bank.hpp"
#include "ie/qwen4_quant.hpp"
#include "ie/dequant_ref.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

using namespace ie;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what.c_str());
    if (!ok) ++g_fail;
}
void check_rel(double max_rel, double tol, const std::string& what) {
    const bool ok = (max_rel <= tol) && std::isfinite(max_rel);
    std::printf("  [%s] %-60s max rel %.3e (tol %.1e)\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what.c_str(), max_rel, tol);
    if (!ok) ++g_fail;
}

uint16_t half_bits(float f) { const sycl::half h(f); uint16_t b; std::memcpy(&b, &h, 2); return b; }
float half_val(uint16_t b) { sycl::half h; std::memcpy(&h, &b, 2); return float(h); }

// Random bank bytes for one kind: quant bytes uniform, scales small and finite.
void fill_blocks(BankQ kind, uint8_t* dst, uint64_t n_blocks, std::mt19937& rng) {
    std::uniform_real_distribution<float> dsc(0.01f, 1.0f);
    std::uniform_int_distribution<int> byte(0, 255), s8(-20, 20);
    for (uint64_t b = 0; b < n_blocks; ++b) {
        switch (kind) {
            case BankQ::kQ8_0: { auto* k = reinterpret_cast<block_q8_0*>(dst) + b; k->d = half_bits(dsc(rng)); for (auto& v : k->qs) v = int8_t(byte(rng)); break; }
            case BankQ::kQ4_0: { auto* k = reinterpret_cast<block_q4_0*>(dst) + b; k->d = half_bits(dsc(rng)); for (auto& v : k->qs) v = uint8_t(byte(rng)); break; }
            case BankQ::kQ5_0: { auto* k = reinterpret_cast<block_q5_0*>(dst) + b; k->d = half_bits(dsc(rng)); for (auto& v : k->qh) v = uint8_t(byte(rng)); for (auto& v : k->qs) v = uint8_t(byte(rng)); break; }
            case BankQ::kQ5_K: { auto* k = reinterpret_cast<block_q5_K*>(dst) + b; k->d = half_bits(dsc(rng) * 0.05f); k->dmin = half_bits(dsc(rng) * 0.05f);
                                 for (auto& v : k->scales) v = uint8_t(byte(rng)); for (auto& v : k->qh) v = uint8_t(byte(rng)); for (auto& v : k->qs) v = uint8_t(byte(rng)); break; }
            case BankQ::kQ6_K: { auto* k = reinterpret_cast<block_q6_K*>(dst) + b; k->d = half_bits(dsc(rng) * 0.05f);
                                 for (auto& v : k->ql) v = uint8_t(byte(rng)); for (auto& v : k->qh) v = uint8_t(byte(rng)); for (auto& v : k->scales) v = int8_t(s8(rng)); break; }
        }
    }
}

void ref_dequant(BankQ kind, const uint8_t* col, uint32_t K, float* out) {
    switch (kind) {
        case BankQ::kQ8_0: ref::dequant_q8_0_buffer(col, K, out); break;
        case BankQ::kQ4_0: ref::dequant_q4_0_buffer(col, K, out); break;
        case BankQ::kQ5_0: ref::dequant_q5_0_buffer(col, K, out); break;
        case BankQ::kQ5_K: ref::dequant_q5_K_buffer(col, K, out); break;
        case BankQ::kQ6_K: ref::dequant_q6_K_buffer(col, K, out); break;
    }
}

uint32_t block_elems(BankQ kind) { return (kind == BankQ::kQ5_K || kind == BankQ::kQ6_K) ? 256u : 32u; }

struct Dev {
    sycl::queue q{sycl::default_selector_v};
    template <class T> T* alloc(size_t n) { return static_cast<T*>(sycl::malloc_shared(n * sizeof(T), q)); }
    void free(void* p) { sycl::free(p, q); }
};

// Device routing phase 2 (2026-11-08): the grouped pick kernels read through a slot address table [row][3] {gate, up, down}.
// The fixture: rows [0, E) = a device (VRAM) copy of the test's slots, rows [E, 2E) = three PINNED HOST banks laid out as
// the model keeps its experts (one expert-major array per matrix; a kernel reads them over PCIe -- the read-through miss).
// The table-form jobs send even picks to the VRAM row of their slot and odd picks to the host row (E + slot): every kernel
// must return the slot form's bits. A matrix the slot layout lacks (off = UINT64_MAX) gets a null entry.
struct TabFixture {
    uint8_t* dslots = nullptr;
    uint8_t* hbank[3] = {nullptr, nullptr, nullptr};
    const uint8_t** tab = nullptr;   // device memory, [2E][3]
    int32_t* jobs = nullptr;         // [P][3]
    void release(Dev& dev) {
        dev.free(dslots); for (auto* h : hbank) if (h) dev.free(h); dev.free(tab); dev.free(jobs);
    }
};
TabFixture make_tab(Dev& dev, const uint8_t* slots, uint64_t slot_bytes, const uint64_t off[3], uint64_t slice, uint32_t E,
                    const int32_t* jobs_h, uint32_t P) {
    TabFixture f;
    f.dslots = sycl::malloc_device<uint8_t>(slot_bytes * E, dev.q);
    dev.q.memcpy(f.dslots, slots, slot_bytes * E).wait();
    std::vector<const uint8_t*> t(uint64_t(2) * E * 3, nullptr);
    for (int m = 0; m < 3; ++m) {
        if (off[m] == UINT64_MAX) continue;
        f.hbank[m] = sycl::malloc_host<uint8_t>(slice * E, dev.q);
        for (uint32_t e = 0; e < E; ++e) {
            std::memcpy(f.hbank[m] + uint64_t(e) * slice, slots + uint64_t(e) * slot_bytes + off[m], slice);
            t[uint64_t(e) * 3 + m]       = f.dslots + uint64_t(e) * slot_bytes + off[m];
            t[uint64_t(E + e) * 3 + m]   = f.hbank[m] + uint64_t(e) * slice;
        }
    }
    f.tab = sycl::malloc_device<const uint8_t*>(t.size(), dev.q);
    dev.q.memcpy(f.tab, t.data(), t.size() * sizeof(t[0])).wait();
    f.jobs = dev.alloc<int32_t>(uint64_t(P) * 3);
    for (uint32_t p = 0; p < P; ++p) {
        f.jobs[p * 3 + 0] = jobs_h[p * 3 + 0];
        f.jobs[p * 3 + 1] = jobs_h[p * 3 + 1] + ((p & 1) ? int32_t(E) : 0);
        f.jobs[p * 3 + 2] = jobs_h[p * 3 + 2];
    }
    return f;
}
bool same_bits(const sycl::half* a, const sycl::half* b, uint64_t n) { return std::memcmp(a, b, n * 2) == 0; }

// One kind at one (K, N): the solo vs fp64 reference, then every variant vs the solo.
void run_kind(Dev& dev, BankQ kind, uint32_t K, uint32_t N, std::mt19937& rng) {
    const std::string tag = std::string(qwen4_bank_name(kind)) + " K=" + std::to_string(K) + " N=" + std::to_string(N) + ": ";
    const uint64_t cb = qwen4_bank_col_bytes(kind, K);
    const uint32_t E = 6;                                     // experts in the synthetic bank
    const uint64_t slice = cb * N;
    uint8_t* bank = dev.alloc<uint8_t>(slice * E);
    fill_blocks(kind, bank, (slice * E) / (cb / (K / block_elems(kind))), rng);
    // --- 1. solo vs fp64 reference on expert 0 ---
    const uint8_t* W0 = bank;
    std::vector<float> wcol(K);
    std::vector<double> ref(N);
    sycl::half* x = dev.alloc<sycl::half>(K);
    sycl::half* y = dev.alloc<sycl::half>(N);
    std::uniform_real_distribution<float> ux(-1.f, 1.f);
    for (uint32_t k = 0; k < K; ++k) x[k] = sycl::half(ux(rng));
    double maxabs = 0.0;
    for (uint32_t n = 0; n < N; ++n) {
        ref_dequant(kind, W0 + uint64_t(n) * cb, K, wcol.data());
        double s = 0.0;
        for (uint32_t k = 0; k < K; ++k) s += double(float(x[k])) * double(wcol[k]);
        ref[n] = s; maxabs = std::max(maxabs, std::fabs(s));
    }
    qwen4_bank_gemv(dev.q, kind, x, W0, y, K, N).wait();
    double max_rel = 0.0;
    for (uint32_t n = 0; n < N; ++n) {
        const double den = std::max(std::fabs(ref[n]), 0.01 * maxabs);
        max_rel = std::max(max_rel, std::fabs(double(float(y[n])) - ref[n]) / std::max(den, 1e-12));
    }
    check_rel(max_rel, 2e-3, tag + "solo GEMV vs fp64 reference");
    // unit vectors: columns come back as the dequantised weights (the half store's rounding only)
    {
        double mr = 0.0;
        for (uint32_t k : {0u, 17u, K / 2 + 3, K - 1}) {
            for (uint32_t i = 0; i < K; ++i) x[i] = sycl::half(i == k ? 1.f : 0.f);
            qwen4_bank_gemv(dev.q, kind, x, W0, y, K, N).wait();
            for (uint32_t n = 0; n < N; ++n) {
                ref_dequant(kind, W0 + uint64_t(n) * cb, K, wcol.data());
                const double w = wcol[k];
                mr = std::max(mr, std::fabs(double(float(y[n])) - w) / std::max(std::fabs(w), 1e-6));
            }
        }
        check_rel(mr, 1e-3, tag + "unit vectors reproduce the dequantised columns");
    }
    // --- 2. the variants vs the solo kernel, bit for bit, over the expert slots ---
    const uint32_t P = 9;                                     // picks
    std::vector<int32_t> jobs_h(P * 3);
    sycl::half* rows = dev.alloc<sycl::half>(uint64_t(P) * K);           // one activation row per pick
    for (uint64_t i = 0; i < uint64_t(P) * K; ++i) rows[i] = sycl::half(ux(rng));
    for (uint32_t p = 0; p < P; ++p) { jobs_h[p * 3 + 0] = int32_t(p); jobs_h[p * 3 + 1] = int32_t((p * 5 + 1) % E); jobs_h[p * 3 + 2] = int32_t(P - 1 - p); }
    int32_t* jobs = dev.alloc<int32_t>(P * 3);
    std::memcpy(jobs, jobs_h.data(), P * 3 * sizeof(int32_t));
    // the slot layout: [gate | up | down] = three slices of this bank per slot (gate from expert e, up from e+1, down from e+2 mod E)
    const uint64_t slot_bytes = 3 * slice, gate_off = 0, up_off = slice, down_off = 2 * slice;
    uint8_t* slots = dev.alloc<uint8_t>(slot_bytes * E);
    for (uint32_t s = 0; s < E; ++s) {
        std::memcpy(slots + s * slot_bytes + gate_off, bank + uint64_t(s) * slice, slice);
        std::memcpy(slots + s * slot_bytes + up_off,   bank + uint64_t((s + 1) % E) * slice, slice);
        std::memcpy(slots + s * slot_bytes + down_off, bank + uint64_t((s + 2) % E) * slice, slice);
    }
    // solo answers per pick and per mat
    std::vector<std::vector<uint16_t>> solo_g(P, std::vector<uint16_t>(N)), solo_u = solo_g, solo_d = solo_g;
    for (uint32_t p = 0; p < P; ++p) {
        const uint32_t slot = uint32_t(jobs_h[p * 3 + 1]);
        qwen4_bank_gemv(dev.q, kind, rows + uint64_t(p) * K, slots + slot * slot_bytes + gate_off, y, K, N).wait();
        for (uint32_t n = 0; n < N; ++n) std::memcpy(&solo_g[p][n], &y[n], 2);
        qwen4_bank_gemv(dev.q, kind, rows + uint64_t(p) * K, slots + slot * slot_bytes + up_off, y, K, N).wait();
        for (uint32_t n = 0; n < N; ++n) std::memcpy(&solo_u[p][n], &y[n], 2);
        qwen4_bank_gemv(dev.q, kind, rows + uint64_t(p) * K, slots + slot * slot_bytes + down_off, y, K, N).wait();
        for (uint32_t n = 0; n < N; ++n) std::memcpy(&solo_d[p][n], &y[n], 2);
    }
    const auto same = [&](const sycl::half* got, const std::vector<uint16_t>& want) {
        for (uint32_t n = 0; n < N; ++n) { uint16_t b; std::memcpy(&b, &got[n], 2); if (b != want[n]) return false; }
        return true;
    };
    {   // grouped dual (gate/up): x rows indexed by vt (= p here), outputs pick-indexed
        sycl::half* g = dev.alloc<sycl::half>(uint64_t(P) * N); sycl::half* u = dev.alloc<sycl::half>(uint64_t(P) * N);
        qwen4_bank_gemv_grouped_dual(dev.q, kind, rows, slots, slot_bytes, gate_off, up_off, jobs, g, u, K, N, P).wait();
        bool ok = true;
        for (uint32_t p = 0; p < P; ++p) ok = ok && same(g + uint64_t(p) * N, solo_g[p]) && same(u + uint64_t(p) * N, solo_u[p]);
        check(ok, tag + "grouped dual gate/up == solo, bit for bit, every pick");
        dev.free(g); dev.free(u);
    }
    {   // the fused swiglu epilogue == grouped dual + swiglu(), bit for bit (the decode host gap, option 1, 2026-11-08)
        sycl::half* g = dev.alloc<sycl::half>(uint64_t(P) * N); sycl::half* u = dev.alloc<sycl::half>(uint64_t(P) * N);
        sycl::half* hs = dev.alloc<sycl::half>(uint64_t(P) * N); sycl::half* hf = dev.alloc<sycl::half>(uint64_t(P) * N);
        qwen4_bank_gemv_grouped_dual(dev.q, kind, rows, slots, slot_bytes, gate_off, up_off, jobs, g, u, K, N, P).wait();
        swiglu(dev.q, g, u, hs, uint64_t(P) * N).wait();
        qwen4_bank_gemv_grouped_dual_swiglu(dev.q, kind, rows, slots, slot_bytes, gate_off, up_off, jobs, hf, K, N, P).wait();
        bool okf = true; uint64_t nz = 0;
        for (uint64_t i = 0; i < uint64_t(P) * N; ++i) {
            uint16_t a, b; std::memcpy(&a, &hs[i], 2); std::memcpy(&b, &hf[i], 2); okf = okf && a == b; nz += (a != 0);
        }
        check(okf && nz > uint64_t(P) * N / 2, tag + "fused swiglu epilogue == grouped dual + swiglu, bit for bit, every pick");
        dev.free(g); dev.free(u); dev.free(hs); dev.free(hf);
    }
    if (qwen4_bank_has_q8dot(kind)) {   // the int-dot (A8) variants: dual == solo per pick; fp64 reference on the dequantised Q8_1s rows
        const uint32_t nqb = K / 32;
        block_q8_1s* xq = dev.alloc<block_q8_1s>(uint64_t(P) * nqb);
        quantize_q8_1s(dev.q, rows, xq, uint64_t(P) * K).wait();
        sycl::half* g = dev.alloc<sycl::half>(uint64_t(P) * N); sycl::half* u = dev.alloc<sycl::half>(uint64_t(P) * N);
        qwen4_bank_gemv_grouped_dual_q8s(dev.q, kind, xq, slots, slot_bytes, gate_off, up_off, jobs, g, u, K, N, P).wait();
        bool ok = true; double mr = 0.0, mr16 = 0.0;
        std::vector<double> xd(K), rf(N);
        std::vector<uint16_t> yb(N);
        for (uint32_t p = 0; p < P; ++p) {
            const uint32_t slot = uint32_t(jobs_h[p * 3 + 1]);
            for (uint32_t k = 0; k < K; ++k) {
                const block_q8_1s& xb = xq[uint64_t(p) * nqb + k / 32];
                xd[k] = double(xb.d) * double(xb.qs[k % 32]);
            }
            for (int mat = 0; mat < 2; ++mat) {
                const uint8_t* Wm = slots + slot * slot_bytes + (mat ? up_off : gate_off);
                const sycl::half* got = (mat ? u : g) + uint64_t(p) * N;
                qwen4_bank_gemv_q8s(dev.q, kind, xq + uint64_t(p) * nqb, Wm, y, K, N).wait();
                for (uint32_t n = 0; n < N; ++n) std::memcpy(&yb[n], &y[n], 2);
                ok = ok && same(got, yb);
                double mx = 0.0;
                for (uint32_t n = 0; n < N; ++n) {
                    ref_dequant(kind, Wm + uint64_t(n) * cb, K, wcol.data());
                    double sacc = 0.0;
                    for (uint32_t k = 0; k < K; ++k) sacc += xd[k] * double(wcol[k]);
                    rf[n] = sacc; mx = std::max(mx, std::fabs(sacc));
                }
                const auto& a16 = mat ? solo_u[p] : solo_g[p];
                for (uint32_t n = 0; n < N; ++n) {
                    const double den = std::max(std::max(std::fabs(rf[n]), 0.01 * mx), 1e-12);
                    mr   = std::max(mr,   std::fabs(double(float(got[n])) - rf[n]) / den);
                    mr16 = std::max(mr16, std::fabs(double(float(got[n])) - double(half_val(a16[n]))) / den);
                }
            }
        }
        check(ok, tag + "int-dot dual gate/up == int-dot solo, bit for bit, every pick");
        {   // the int-dot fused swiglu epilogue == int-dot dual + swiglu(), bit for bit
            sycl::half* hs = dev.alloc<sycl::half>(uint64_t(P) * N); sycl::half* hf = dev.alloc<sycl::half>(uint64_t(P) * N);
            swiglu(dev.q, g, u, hs, uint64_t(P) * N).wait();
            qwen4_bank_gemv_grouped_dual_q8s_swiglu(dev.q, kind, xq, slots, slot_bytes, gate_off, up_off, jobs, hf, K, N, P).wait();
            bool okf = true; uint64_t nz = 0;
            for (uint64_t i = 0; i < uint64_t(P) * N; ++i) {
                uint16_t a, b; std::memcpy(&a, &hs[i], 2); std::memcpy(&b, &hf[i], 2); okf = okf && a == b; nz += (a != 0);
            }
            check(okf && nz > uint64_t(P) * N / 2, tag + "int-dot fused swiglu epilogue == int-dot dual + swiglu, bit for bit, every pick");
            dev.free(hs); dev.free(hf);
        }
        {   // the int-dot tiles: jobs of 1 / 5 / 16 / 2 rows over a 24-row Q8_1s mega buffer, both mats, per row == the int-dot solo
            const uint32_t M = 24;
            sycl::half* mega = dev.alloc<sycl::half>(uint64_t(M) * K);
            for (uint64_t i = 0; i < uint64_t(M) * K; ++i) mega[i] = sycl::half(ux(rng));
            block_q8_1s* mq = dev.alloc<block_q8_1s>(uint64_t(M) * nqb);
            quantize_q8_1s(dev.q, mega, mq, uint64_t(M) * K).wait();
            std::vector<int32_t> tj = {0, 1, 2, 0,   1, 5, 4, 1,   6, 16, 3, 6,   22, 2, 0, 22};   // {x0, rows, slot, y0}
            int32_t* tjobs = dev.alloc<int32_t>(tj.size());
            std::memcpy(tjobs, tj.data(), tj.size() * sizeof(int32_t));
            sycl::half* o0 = dev.alloc<sycl::half>(uint64_t(M) * N); sycl::half* o1 = dev.alloc<sycl::half>(uint64_t(M) * N);
            qwen4_bank_gemv_grouped_tiles_q8s(dev.q, kind, mq, slots, slot_bytes, gate_off, up_off, tjobs, o0, o1, K, N, 4).wait();
            bool okt = true;
            for (size_t jj = 0; jj < tj.size(); jj += 4) {
                const uint32_t x0 = uint32_t(tj[jj]), rows_j = uint32_t(tj[jj + 1]), slot = uint32_t(tj[jj + 2]), y0 = uint32_t(tj[jj + 3]);
                for (uint32_t t = 0; t < rows_j; ++t)
                    for (int mat = 0; mat < 2; ++mat) {
                        qwen4_bank_gemv_q8s(dev.q, kind, mq + uint64_t(x0 + t) * nqb, slots + slot * slot_bytes + (mat ? up_off : gate_off), y, K, N).wait();
                        for (uint32_t n = 0; n < N; ++n) std::memcpy(&yb[n], &y[n], 2);
                        okt = okt && same((mat ? o1 : o0) + uint64_t(y0 + t) * N, yb);
                    }
            }
            check(okt, tag + "int-dot tiles (rows 1 / 5 / 16 / 2, both mats) == int-dot solo per row, bit for bit");
            dev.free(mega); dev.free(mq); dev.free(tjobs); dev.free(o0); dev.free(o1);
        }
        check_rel(mr, 2e-3, tag + "int-dot vs fp64 reference on the Q8_1s rows");
        std::printf("  [info] %sint-dot vs the A16 kernels (the int8 activation rounding): max rel %.3e\n", tag.c_str(), mr16);
        dev.free(xq); dev.free(g); dev.free(u);
    }
    if (qwen4_bank_has_q8dpas(kind) && N % 16 == 0 && (kind == BankQ::kQ8_0 || K % 256 == 0)) {
        // the int8-DPAS tiles: jobs of 1 / 5 / 16 / 2 rows over a 24-row Q8_1s mega buffer, both mats, vs an fp64 reference on
        // the dequantised Q8_1s rows (the dpas-form fp order is not the s-form's: no bit-identity claim)
        const uint32_t M = 24, nqb = K / 32;
        sycl::half* mega = dev.alloc<sycl::half>(uint64_t(M) * K);
        for (uint64_t i = 0; i < uint64_t(M) * K; ++i) mega[i] = sycl::half(ux(rng));
        block_q8_1s* mq = dev.alloc<block_q8_1s>(uint64_t(M) * nqb);
        quantize_q8_1s(dev.q, mega, mq, uint64_t(M) * K).wait();
        std::vector<int32_t> tj = {0, 1, 2, 0,   1, 5, 4, 1,   6, 16, 3, 6,   22, 2, 0, 22};
        int32_t* tjobs = dev.alloc<int32_t>(tj.size());
        std::memcpy(tjobs, tj.data(), tj.size() * sizeof(int32_t));
        sycl::half* o0 = dev.alloc<sycl::half>(uint64_t(M) * N); sycl::half* o1 = dev.alloc<sycl::half>(uint64_t(M) * N);
        qwen4_bank_gemv_grouped_tiles_q8d(dev.q, kind, mq, slots, slot_bytes, gate_off, up_off, tjobs, o0, o1, K, N, 4).wait();
        double mr = 0.0;
        std::vector<double> xd(K), rf(N);
        for (size_t jj = 0; jj < tj.size(); jj += 4) {
            const uint32_t x0 = uint32_t(tj[jj]), rows_j = uint32_t(tj[jj + 1]), slot = uint32_t(tj[jj + 2]), y0 = uint32_t(tj[jj + 3]);
            for (uint32_t t = 0; t < rows_j; ++t) {
                for (uint32_t k = 0; k < K; ++k) { const block_q8_1s& xb = mq[uint64_t(x0 + t) * nqb + k / 32]; xd[k] = double(xb.d) * double(xb.qs[k % 32]); }
                for (int mat = 0; mat < 2; ++mat) {
                    const uint8_t* Wm = slots + slot * slot_bytes + (mat ? up_off : gate_off);
                    const sycl::half* got = (mat ? o1 : o0) + uint64_t(y0 + t) * N;
                    double mx = 0.0;
                    for (uint32_t nn = 0; nn < N; ++nn) {
                        ref_dequant(kind, Wm + uint64_t(nn) * cb, K, wcol.data());
                        double sacc = 0.0;
                        for (uint32_t k = 0; k < K; ++k) sacc += xd[k] * double(wcol[k]);
                        rf[nn] = sacc; mx = std::max(mx, std::fabs(sacc));
                    }
                    for (uint32_t nn = 0; nn < N; ++nn) {
                        const double den = std::max(std::max(std::fabs(rf[nn]), 0.01 * mx), 1e-12);
                        mr = std::max(mr, std::fabs(double(float(got[nn])) - rf[nn]) / den);
                    }
                }
            }
        }
        check_rel(mr, 2e-3, tag + "int8-DPAS tiles (rows 1 / 5 / 16 / 2, both mats) vs fp64 on the Q8_1s rows");
        dev.free(mega); dev.free(mq); dev.free(tjobs); dev.free(o0); dev.free(o1);
    }
    if (qwen4_bank_has_down_f16d(kind) && N % 16 == 0) {   // the f16-DPAS down tiles vs fp64 on the f16 rows (down slice)
        const uint32_t M = 24;
        sycl::half* mega = dev.alloc<sycl::half>(uint64_t(M + 16) * K);          // + the 8-row group over-read
        for (uint64_t i = 0; i < uint64_t(M + 16) * K; ++i) mega[i] = sycl::half(ux(rng));
        std::vector<int32_t> tj = {0, 1, 2, 0,   1, 5, 4, 1,   6, 16, 3, 6,   22, 2, 0, 22};
        int32_t* tjobs = dev.alloc<int32_t>(tj.size());
        std::memcpy(tjobs, tj.data(), tj.size() * sizeof(int32_t));
        sycl::half* od = dev.alloc<sycl::half>(uint64_t(M) * N);
        qwen4_bank_gemv_grouped_tiles_down_f16d(dev.q, kind, mega, slots, slot_bytes, down_off, tjobs, od, K, N, 4).wait();
        double mr = 0.0;
        std::vector<double> rf(N);
        for (size_t jj = 0; jj < tj.size(); jj += 4) {
            const uint32_t x0 = uint32_t(tj[jj]), rows_j = uint32_t(tj[jj + 1]), slot = uint32_t(tj[jj + 2]), y0 = uint32_t(tj[jj + 3]);
            for (uint32_t t = 0; t < rows_j; ++t) {
                const sycl::half* xr = mega + uint64_t(x0 + t) * K;
                const uint8_t* Wm = slots + slot * slot_bytes + down_off;
                double mx = 0.0;
                for (uint32_t nn = 0; nn < N; ++nn) {
                    ref_dequant(kind, Wm + uint64_t(nn) * cb, K, wcol.data());
                    double sacc = 0.0;
                    for (uint32_t k = 0; k < K; ++k) sacc += double(float(xr[k])) * double(wcol[k]);
                    rf[nn] = sacc; mx = std::max(mx, std::fabs(sacc));
                }
                const sycl::half* got = od + uint64_t(y0 + t) * N;
                for (uint32_t nn = 0; nn < N; ++nn) {
                    const double den = std::max(std::max(std::fabs(rf[nn]), 0.01 * mx), 1e-12);
                    mr = std::max(mr, std::fabs(double(float(got[nn])) - rf[nn]) / den);
                }
            }
        }
        check_rel(mr, 2e-3, tag + "f16-DPAS down tiles (rows 1 / 5 / 16 / 2) vs fp64 on the f16 rows");
        dev.free(mega); dev.free(tjobs); dev.free(od);
    }
    {   // grouped down: writes ystage + ycell * y_stride
        const uint32_t y_stride = N + 8;
        sycl::half* ys = dev.alloc<sycl::half>(uint64_t(P) * y_stride);
        qwen4_bank_gemv_grouped(dev.q, kind, rows, slots, slot_bytes, down_off, jobs, ys, y_stride, K, N, P).wait();
        bool ok = true;
        for (uint32_t p = 0; p < P; ++p) ok = ok && same(ys + uint64_t(jobs_h[p * 3 + 2]) * y_stride, solo_d[p]);
        check(ok, tag + "grouped down == solo, bit for bit, every pick (ycell-indexed)");
        dev.free(ys);
    }
    {   // the slot address table (device routing phase 2): VRAM rows and pinned host rows == the slot form, bit for bit
        const uint64_t offs[3] = {gate_off, up_off, down_off};
        TabFixture tf = make_tab(dev, slots, slot_bytes, offs, slice, E, jobs_h.data(), P);
        const uint64_t PN = uint64_t(P) * N;
        sycl::half* g = dev.alloc<sycl::half>(PN); sycl::half* u = dev.alloc<sycl::half>(PN);
        sycl::half* h0 = dev.alloc<sycl::half>(PN); sycl::half* h1 = dev.alloc<sycl::half>(PN);
        qwen4_bank_gemv_grouped_dual(dev.q, kind, rows, nullptr, 0, 0, 0, tf.jobs, g, u, K, N, P, {}, tf.tab).wait();
        bool ok = true;
        for (uint32_t p = 0; p < P; ++p) ok = ok && same(g + uint64_t(p) * N, solo_g[p]) && same(u + uint64_t(p) * N, solo_u[p]);
        qwen4_bank_gemv_grouped_dual_swiglu(dev.q, kind, rows, slots, slot_bytes, gate_off, up_off, jobs, h0, K, N, P).wait();
        qwen4_bank_gemv_grouped_dual_swiglu(dev.q, kind, rows, nullptr, 0, 0, 0, tf.jobs, h1, K, N, P, {}, tf.tab).wait();
        ok = ok && same_bits(h0, h1, PN);
        const uint32_t y_stride = N + 8;
        sycl::half* ys = dev.alloc<sycl::half>(uint64_t(P) * y_stride);
        qwen4_bank_gemv_grouped(dev.q, kind, rows, nullptr, 0, 0, tf.jobs, ys, y_stride, K, N, P, {}, tf.tab + 2).wait();
        for (uint32_t p = 0; p < P; ++p) ok = ok && same(ys + uint64_t(jobs_h[p * 3 + 2]) * y_stride, solo_d[p]);
        if (qwen4_bank_has_q8dot(kind)) {
            block_q8_1s* xq = dev.alloc<block_q8_1s>(uint64_t(P) * (K / 32));
            quantize_q8_1s(dev.q, rows, xq, uint64_t(P) * K).wait();
            sycl::half* g2 = dev.alloc<sycl::half>(PN); sycl::half* u2 = dev.alloc<sycl::half>(PN);
            qwen4_bank_gemv_grouped_dual_q8s(dev.q, kind, xq, slots, slot_bytes, gate_off, up_off, jobs, g, u, K, N, P).wait();
            qwen4_bank_gemv_grouped_dual_q8s(dev.q, kind, xq, nullptr, 0, 0, 0, tf.jobs, g2, u2, K, N, P, {}, tf.tab).wait();
            ok = ok && same_bits(g, g2, PN) && same_bits(u, u2, PN);
            qwen4_bank_gemv_grouped_dual_q8s_swiglu(dev.q, kind, xq, slots, slot_bytes, gate_off, up_off, jobs, h0, K, N, P).wait();
            qwen4_bank_gemv_grouped_dual_q8s_swiglu(dev.q, kind, xq, nullptr, 0, 0, 0, tf.jobs, h1, K, N, P, {}, tf.tab).wait();
            ok = ok && same_bits(h0, h1, PN);
            dev.free(xq); dev.free(g2); dev.free(u2);
        }
        check(ok, tag + "slot address table (VRAM + pinned host rows): dual, dual+swiglu, down" +
                  (qwen4_bank_has_q8dot(kind) ? ", int-dot dual + swiglu" : "") + " == the slot form, bit for bit");
        dev.free(g); dev.free(u); dev.free(h0); dev.free(h1); dev.free(ys);
        tf.release(dev);
    }
    {   // grouped tiles: jobs of 1, 5 and 16 rows over a mega row buffer, both mats; then the batched rows (M = 33)
        const uint32_t M = 33;
        sycl::half* mega = dev.alloc<sycl::half>(uint64_t(M) * K);
        for (uint64_t i = 0; i < uint64_t(M) * K; ++i) mega[i] = sycl::half(ux(rng));
        std::vector<int32_t> tj = {0, 1, 2, 0,   1, 5, 4, 1,   6, 16, 3, 6,   22, 11, 0, 22};   // {x0, rows, slot, y0}
        int32_t* tjobs = dev.alloc<int32_t>(tj.size());
        std::memcpy(tjobs, tj.data(), tj.size() * sizeof(int32_t));
        sycl::half* o0 = dev.alloc<sycl::half>(uint64_t(M) * N); sycl::half* o1 = dev.alloc<sycl::half>(uint64_t(M) * N);
        qwen4_bank_gemv_grouped_tiles(dev.q, kind, mega, slots, slot_bytes, gate_off, up_off, tjobs, o0, o1, K, N, 4, 2).wait();
        bool ok = true;
        for (size_t j = 0; j < tj.size() / 4; ++j)
            for (int32_t t = 0; t < tj[j * 4 + 1]; ++t) {
                const uint32_t slot = uint32_t(tj[j * 4 + 2]);
                std::vector<uint16_t> wg(N), wu(N);
                qwen4_bank_gemv(dev.q, kind, mega + uint64_t(tj[j * 4] + t) * K, slots + slot * slot_bytes + gate_off, y, K, N).wait();
                for (uint32_t n = 0; n < N; ++n) std::memcpy(&wg[n], &y[n], 2);
                qwen4_bank_gemv(dev.q, kind, mega + uint64_t(tj[j * 4] + t) * K, slots + slot * slot_bytes + up_off, y, K, N).wait();
                for (uint32_t n = 0; n < N; ++n) std::memcpy(&wu[n], &y[n], 2);
                ok = ok && same(o0 + uint64_t(tj[j * 4 + 3] + t) * N, wg) && same(o1 + uint64_t(tj[j * 4 + 3] + t) * N, wu);
            }
        check(ok, tag + "grouped tiles (rows 1 / 5 / 16 / 11, both mats) == solo per row, bit for bit");
        sycl::half* ym = dev.alloc<sycl::half>(uint64_t(M) * N);
        qwen4_bank_gemm(dev.q, kind, mega, W0, ym, M, K, N).wait();
        bool okm = true;
        for (uint32_t m = 0; m < M; ++m) {
            std::vector<uint16_t> w(N);
            qwen4_bank_gemv(dev.q, kind, mega + uint64_t(m) * K, W0, y, K, N).wait();
            for (uint32_t n = 0; n < N; ++n) std::memcpy(&w[n], &y[n], 2);
            okm = okm && same(ym + uint64_t(m) * N, w);
        }
        check(okm, tag + "batched rows (M = 33) == solo per row, bit for bit");
        dev.free(mega); dev.free(tjobs); dev.free(o0); dev.free(o1); dev.free(ym);
    }
    dev.free(bank); dev.free(x); dev.free(y); dev.free(rows); dev.free(jobs); dev.free(slots);
}

// IE_BANK_BENCH=1: the grouped-dual gate/up kernels' speed at the real shape (K = 2560, N = 640, 10 picks a launch, 48
// launches a step as one decode step has) over a 64-expert bank, the A16 kernel and (Q5_K / Q6_K) the int-dot one.
void bench_dual(Dev& dev, BankQ kind, std::mt19937& rng) {
    const uint32_t K = 2560, N = 640, P = 10, E = 64, L = 48;
    const uint64_t cb = qwen4_bank_col_bytes(kind, K), slice = cb * N, slot_bytes = 2 * slice;
    uint8_t* slots = dev.alloc<uint8_t>(slot_bytes * E);
    fill_blocks(kind, slots, (slot_bytes * E) / (cb / (K / block_elems(kind))), rng);
    sycl::half* rows = dev.alloc<sycl::half>(uint64_t(P) * K);
    std::uniform_real_distribution<float> ux(-1.f, 1.f);
    for (uint64_t i = 0; i < uint64_t(P) * K; ++i) rows[i] = sycl::half(ux(rng));
    block_q8_1s* xq = dev.alloc<block_q8_1s>(uint64_t(P) * (K / 32));
    quantize_q8_1s(dev.q, rows, xq, uint64_t(P) * K).wait();
    std::vector<int32_t*> jobs(L);
    for (uint32_t l = 0; l < L; ++l) {   // each launch picks 10 distinct experts; the 48 launches walk the whole bank
        jobs[l] = dev.alloc<int32_t>(P * 3);
        for (uint32_t p = 0; p < P; ++p) { jobs[l][p * 3 + 0] = int32_t(p); jobs[l][p * 3 + 1] = int32_t((l * 10 + p * 7) % E); jobs[l][p * 3 + 2] = int32_t(p); }
    }
    sycl::half* g = dev.alloc<sycl::half>(uint64_t(P) * N); sycl::half* u = dev.alloc<sycl::half>(uint64_t(P) * N);
    sycl::half* hh = dev.alloc<sycl::half>(uint64_t(P) * N);
    const double bytes_step = double(L) * P * 2 * slice;
    // dual + swiglu (two launches a layer) against the fused-epilogue kernel (one): the same bank bytes either way
    auto run = [&](bool intdot, bool fused) {
        std::vector<double> ms;
        for (int rep = 0; rep < 6; ++rep) {
            dev.q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (uint32_t l = 0; l < L; ++l) {
                if (intdot && fused) qwen4_bank_gemv_grouped_dual_q8s_swiglu(dev.q, kind, xq, slots, slot_bytes, 0, slice, jobs[l], hh, K, N, P);
                else if (intdot)     qwen4_bank_gemv_grouped_dual_q8s(dev.q, kind, xq, slots, slot_bytes, 0, slice, jobs[l], g, u, K, N, P);
                else if (fused)      qwen4_bank_gemv_grouped_dual_swiglu(dev.q, kind, rows, slots, slot_bytes, 0, slice, jobs[l], hh, K, N, P);
                else                 qwen4_bank_gemv_grouped_dual(dev.q, kind, rows, slots, slot_bytes, 0, slice, jobs[l], g, u, K, N, P);
                if (!fused) swiglu(dev.q, g, u, hh, uint64_t(P) * N);
            }
            dev.q.wait();
            const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (rep) ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        std::printf("  [bench] %-5s %-8s %-12s gate/up: %6.2f ms / 48 layers = %6.1f us/layer, %5.0f GB/s of bank bytes\n",
                    qwen4_bank_name(kind), intdot ? "int-dot" : "A16", fused ? "fused" : "dual+swiglu",
                    med, med * 1000.0 / L, bytes_step / (med * 1e-3) / 1e9);
    };
    run(false, false); run(false, true);
    if (qwen4_bank_has_q8dot(kind)) { run(true, false); run(true, true); }
    for (auto* j : jobs) dev.free(j);
    dev.free(slots); dev.free(rows); dev.free(xq); dev.free(g); dev.free(u); dev.free(hh);
}

// IE_BANK_BENCH=1: the prefill tiles at a chunk's shape -- 64 jobs of 16 rows (1024 rows), N = 640, over a 64-expert bank
// (each job its own expert): the A16 tiles, the s-form int-dot tiles (Q5_K / Q6_K) and the int8-DPAS tiles.
void bench_tiles(Dev& dev, BankQ kind, std::mt19937& rng) {
    const uint32_t K = 2560, N = 640, J = 64, R = 16, E = 64, M = J * R;
    const uint64_t cb = qwen4_bank_col_bytes(kind, K), slice = cb * N, slot_bytes = 2 * slice;
    uint8_t* slots = dev.alloc<uint8_t>(slot_bytes * E);
    fill_blocks(kind, slots, (slot_bytes * E) / (cb / (K / block_elems(kind))), rng);
    sycl::half* mega = dev.alloc<sycl::half>(uint64_t(M) * K);
    std::uniform_real_distribution<float> ux(-1.f, 1.f);
    for (uint64_t i = 0; i < uint64_t(M) * K; ++i) mega[i] = sycl::half(ux(rng));
    block_q8_1s* mq = dev.alloc<block_q8_1s>(uint64_t(M) * (K / 32));
    quantize_q8_1s(dev.q, mega, mq, uint64_t(M) * K).wait();
    std::vector<int32_t> tj(J * 4);
    for (uint32_t j = 0; j < J; ++j) { tj[4 * j] = int32_t(j * R); tj[4 * j + 1] = int32_t(R); tj[4 * j + 2] = int32_t(j % E); tj[4 * j + 3] = int32_t(j * R); }
    int32_t* tjobs = dev.alloc<int32_t>(tj.size());
    std::memcpy(tjobs, tj.data(), tj.size() * sizeof(int32_t));
    sycl::half* o0 = dev.alloc<sycl::half>(uint64_t(M) * N); sycl::half* o1 = dev.alloc<sycl::half>(uint64_t(M) * N);
    const double rows_x_bytes = double(M) * 2 * slice / R;   // bank bytes touched once per job (both mats)
    auto run = [&](const char* name, auto&& launch) {
        std::vector<double> ms;
        for (int rep = 0; rep < 6; ++rep) {
            dev.q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            launch();
            dev.q.wait();
            const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (rep) ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        std::printf("  [bench] %-5s %-9s tiles (64 jobs x 16 rows, N 640): %7.2f ms, %6.0f rows/ms, %5.0f GB/s of bank bytes\n",
                    qwen4_bank_name(kind), name, med, double(M) / med, rows_x_bytes / (med * 1e-3) / 1e9);
    };
    run("A16", [&] { qwen4_bank_gemv_grouped_tiles(dev.q, kind, mega, slots, slot_bytes, 0, slice, tjobs, o0, o1, K, N, J, 2); });
    if (qwen4_bank_has_q8dot(kind)) run("int-dot", [&] { qwen4_bank_gemv_grouped_tiles_q8s(dev.q, kind, mq, slots, slot_bytes, 0, slice, tjobs, o0, o1, K, N, J); });
    if (qwen4_bank_has_q8dpas(kind)) run("DPAS", [&] { qwen4_bank_gemv_grouped_tiles_q8d(dev.q, kind, mq, slots, slot_bytes, 0, slice, tjobs, o0, o1, K, N, J); });
    dev.free(slots); dev.free(mega); dev.free(mq); dev.free(tjobs); dev.free(o0); dev.free(o1);
}

// IE_BANK_BENCH=1: the Q8_0 DOWN tiles at a chunk's shape -- 64 jobs of 16 rows, K = 640, N = 2560, 64-expert bank: the A16
// tiles (mats = 1) vs the f16-DPAS down tiles.
void bench_down_tiles(Dev& dev, std::mt19937& rng) {
    const BankQ kind = BankQ::kQ8_0;
    const uint32_t K = 640, N = 2560, J = 64, R = 16, E = 64, M = J * R;
    const uint64_t cb = qwen4_bank_col_bytes(kind, K), slice = cb * N, slot_bytes = slice;
    uint8_t* slots = dev.alloc<uint8_t>(slot_bytes * E);
    fill_blocks(kind, slots, (slot_bytes * E) / (cb / (K / block_elems(kind))), rng);
    sycl::half* mega = dev.alloc<sycl::half>(uint64_t(M + 16) * K);
    std::uniform_real_distribution<float> ux(-1.f, 1.f);
    for (uint64_t i = 0; i < uint64_t(M + 16) * K; ++i) mega[i] = sycl::half(ux(rng));
    std::vector<int32_t> tj(J * 4);
    for (uint32_t j = 0; j < J; ++j) { tj[4 * j] = int32_t(j * R); tj[4 * j + 1] = int32_t(R); tj[4 * j + 2] = int32_t(j % E); tj[4 * j + 3] = int32_t(j * R); }
    int32_t* tjobs = dev.alloc<int32_t>(tj.size());
    std::memcpy(tjobs, tj.data(), tj.size() * sizeof(int32_t));
    sycl::half* o0 = dev.alloc<sycl::half>(uint64_t(M) * N);
    const double rows_x_bytes = double(M) * slice / R;
    auto run = [&](const char* name, auto&& launch) {
        std::vector<double> ms;
        for (int rep = 0; rep < 6; ++rep) {
            dev.q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            launch();
            dev.q.wait();
            const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (rep) ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        std::printf("  [bench] Q8_0  %-9s DOWN tiles (64 jobs x 16 rows, K 640, N 2560): %7.2f ms, %6.0f rows/ms, %5.0f GB/s of bank bytes\n",
                    name, med, double(M) / med, rows_x_bytes / (med * 1e-3) / 1e9);
    };
    run("A16", [&] { qwen4_bank_gemv_grouped_tiles(dev.q, kind, mega, slots, slot_bytes, 0, 0, tjobs, o0, nullptr, K, N, J, 1); });
    run("f16-DPAS", [&] { qwen4_bank_gemv_grouped_tiles_down_f16d(dev.q, kind, mega, slots, slot_bytes, 0, tjobs, o0, K, N, J); });
    dev.free(slots); dev.free(mega); dev.free(tjobs); dev.free(o0);
}

}  // namespace

// Random Q4_K blocks: small finite super-scales, random packed scale / mins and quant bytes (any byte pattern is a block).
void fill_q4k(uint8_t* dst, uint64_t n_blocks, std::mt19937& rng) {
    std::uniform_real_distribution<float> dsc(0.01f, 1.0f);
    std::uniform_int_distribution<int> byte(0, 255);
    for (uint64_t b = 0; b < n_blocks; ++b) {
        auto* k = reinterpret_cast<block_q4_K*>(dst) + b;
        k->d = half_bits(dsc(rng) * 0.01f); k->dmin = half_bits(dsc(rng) * 0.01f);
        for (auto& v : k->scales) v = uint8_t(byte(rng));
        for (auto& v : k->qs) v = uint8_t(byte(rng));
    }
}

// The Q4_K int-dot grouped dual (gemv_q8dot.cpp, the daily UD-Q4_K_XL's decode gate/up kernel) with the fused swiglu
// epilogue == dual + swiglu(), bit for bit: 9 picks over 12 slots, K = 2560 (the gate/up bank's K), N = 64.
void run_q4k_fused(Dev& dev, std::mt19937& rng) {
    const uint32_t K = 2560, N = 64, P = 9, E = 12, nblk = K / 256;
    const uint64_t slice = uint64_t(N) * nblk * sizeof(block_q4_K), slot_bytes = 2 * slice;
    uint8_t* slots = dev.alloc<uint8_t>(slot_bytes * E);
    fill_q4k(slots, (slot_bytes * E) / sizeof(block_q4_K), rng);
    std::uniform_real_distribution<float> ux(-1.f, 1.f);
    sycl::half* rows = dev.alloc<sycl::half>(uint64_t(P) * K);
    for (uint64_t i = 0; i < uint64_t(P) * K; ++i) rows[i] = sycl::half(ux(rng));
    block_q8_1s* xq = dev.alloc<block_q8_1s>(uint64_t(P) * (K / 32));
    quantize_q8_1s(dev.q, rows, xq, uint64_t(P) * K).wait();
    int32_t* jobs = dev.alloc<int32_t>(P * 3);
    for (uint32_t p = 0; p < P; ++p) { jobs[p * 3 + 0] = int32_t(p); jobs[p * 3 + 1] = int32_t((p * 5 + 1) % E); jobs[p * 3 + 2] = int32_t(p); }
    sycl::half* g = dev.alloc<sycl::half>(uint64_t(P) * N); sycl::half* u = dev.alloc<sycl::half>(uint64_t(P) * N);
    sycl::half* hs = dev.alloc<sycl::half>(uint64_t(P) * N); sycl::half* hf = dev.alloc<sycl::half>(uint64_t(P) * N);
    gemv_q4_K_q8s_grouped_dual(dev.q, xq, slots, slot_bytes, 0, slice, jobs, g, u, K, N, P).wait();
    swiglu(dev.q, g, u, hs, uint64_t(P) * N).wait();
    gemv_q4_K_q8s_grouped_dual_swiglu(dev.q, xq, slots, slot_bytes, 0, slice, jobs, hf, K, N, P).wait();
    bool ok = true; uint64_t nz = 0;
    for (uint64_t i = 0; i < uint64_t(P) * N; ++i) {
        uint16_t a, b; std::memcpy(&a, &hs[i], 2); std::memcpy(&b, &hf[i], 2); ok = ok && a == b; nz += (a != 0);
    }
    std::printf("\nQ4_K int-dot grouped dual (the daily file's decode kernel), K=%u N=%u P=%u\n", K, N, P);
    check(ok && nz > uint64_t(P) * N / 2, "Q4_K fused swiglu epilogue == int-dot dual + swiglu, bit for bit, every pick");
    {   // the slot address table (device routing phase 2)
        std::vector<int32_t> jh(jobs, jobs + P * 3);
        const uint64_t offs[3] = {0, slice, UINT64_MAX};
        TabFixture tf = make_tab(dev, slots, slot_bytes, offs, slice, E, jh.data(), P);
        const uint64_t PN = uint64_t(P) * N;
        sycl::half* g2 = dev.alloc<sycl::half>(PN); sycl::half* u2 = dev.alloc<sycl::half>(PN); sycl::half* h2 = dev.alloc<sycl::half>(PN);
        gemv_q4_K_q8s_grouped_dual(dev.q, xq, nullptr, 0, 0, 0, tf.jobs, g2, u2, K, N, P, {}, tf.tab).wait();
        gemv_q4_K_q8s_grouped_dual_swiglu(dev.q, xq, nullptr, 0, 0, 0, tf.jobs, h2, K, N, P, {}, tf.tab).wait();
        check(same_bits(g, g2, PN) && same_bits(u, u2, PN) && same_bits(hf, h2, PN),
              "Q4_K slot address table (VRAM + pinned host rows): int-dot dual and dual+swiglu == the slot form, bit for bit");
        dev.free(g2); dev.free(u2); dev.free(h2);
        tf.release(dev);
    }
    dev.free(slots); dev.free(rows); dev.free(xq); dev.free(jobs); dev.free(g); dev.free(u); dev.free(hs); dev.free(hf);
}

// The Q5_1 grouped down (gemv_q5_1.cpp, the daily UD-Q4_K_XL's decode down kernel) through the slot address table == its
// slot form, bit for bit: 9 picks over 12 slots, K = 640 (the down bank's K), N = 64; slots [gate | up | down] with the down
// slice at 2*slice (the gate/up bytes are filler -- only the down entry is read).
void run_q51_tab(Dev& dev, std::mt19937& rng) {
    const uint32_t K = 640, N = 64, P = 9, E = 12, nblk = K / 32;
    const uint64_t slice = uint64_t(N) * nblk * sizeof(block_q5_1), slot_bytes = 3 * slice;
    uint8_t* slots = dev.alloc<uint8_t>(slot_bytes * E);
    std::uniform_real_distribution<float> dsc(0.01f, 1.0f), ux(-1.f, 1.f);
    std::uniform_int_distribution<int> byte(0, 255);
    for (uint64_t b = 0; b < (slot_bytes * E) / sizeof(block_q5_1); ++b) {
        auto* k = reinterpret_cast<block_q5_1*>(slots) + b;
        k->d = half_bits(dsc(rng) * 0.05f); k->m = half_bits(-dsc(rng) * 0.05f);
        for (auto& v : k->qh) v = uint8_t(byte(rng));
        for (auto& v : k->qs) v = uint8_t(byte(rng));
    }
    sycl::half* rows = dev.alloc<sycl::half>(uint64_t(P) * K);
    for (uint64_t i = 0; i < uint64_t(P) * K; ++i) rows[i] = sycl::half(ux(rng));
    std::vector<int32_t> jh(P * 3);
    for (uint32_t p = 0; p < P; ++p) { jh[p * 3 + 0] = int32_t(p); jh[p * 3 + 1] = int32_t((p * 5 + 1) % E); jh[p * 3 + 2] = int32_t(P - 1 - p); }
    int32_t* jobs = dev.alloc<int32_t>(P * 3);
    std::memcpy(jobs, jh.data(), P * 3 * sizeof(int32_t));
    const uint64_t offs[3] = {0, slice, 2 * slice};
    TabFixture tf = make_tab(dev, slots, slot_bytes, offs, slice, E, jh.data(), P);
    const uint32_t y_stride = N + 8;
    sycl::half* y0 = dev.alloc<sycl::half>(uint64_t(P) * y_stride); sycl::half* y1 = dev.alloc<sycl::half>(uint64_t(P) * y_stride);
    std::memset(y0, 0, uint64_t(P) * y_stride * 2); std::memset(y1, 0, uint64_t(P) * y_stride * 2);
    gemv_q5_1_grouped(dev.q, rows, slots, slot_bytes, 2 * slice, jobs, y0, y_stride, K, N, P).wait();
    gemv_q5_1_grouped(dev.q, rows, nullptr, 0, 0, tf.jobs, y1, y_stride, K, N, P, {}, tf.tab + 2).wait();
    uint64_t nz = 0;
    for (uint64_t i = 0; i < uint64_t(P) * y_stride; ++i) { uint16_t a; std::memcpy(&a, &y0[i], 2); nz += (a != 0); }
    std::printf("\nQ5_1 grouped down (the daily file's decode kernel), K=%u N=%u P=%u\n", K, N, P);
    check(same_bits(y0, y1, uint64_t(P) * y_stride) && nz > uint64_t(P) * N / 2,
          "Q5_1 slot address table (VRAM + pinned host rows): grouped down == the slot form, bit for bit");
    tf.release(dev);
    dev.free(slots); dev.free(rows); dev.free(jobs); dev.free(y0); dev.free(y1);
}

// IE_BANK_BENCH=1: the Q4_K decode gate/up at the daily shape (10 picks, K 2560, N 640, 48 layers walking a 64-expert bank):
// dual + swiglu (two launches a layer) against the fused kernel (one).
void bench_q4k_dual(Dev& dev, std::mt19937& rng) {
    const uint32_t K = 2560, N = 640, P = 10, E = 64, L = 48, nblk = K / 256;
    const uint64_t slice = uint64_t(N) * nblk * sizeof(block_q4_K), slot_bytes = 2 * slice;
    uint8_t* slots = dev.alloc<uint8_t>(slot_bytes * E);
    fill_q4k(slots, (slot_bytes * E) / sizeof(block_q4_K), rng);
    std::uniform_real_distribution<float> ux(-1.f, 1.f);
    sycl::half* rows = dev.alloc<sycl::half>(uint64_t(P) * K);
    for (uint64_t i = 0; i < uint64_t(P) * K; ++i) rows[i] = sycl::half(ux(rng));
    block_q8_1s* xq = dev.alloc<block_q8_1s>(uint64_t(P) * (K / 32));
    quantize_q8_1s(dev.q, rows, xq, uint64_t(P) * K).wait();
    std::vector<int32_t*> jobs(L);
    for (uint32_t l = 0; l < L; ++l) {
        jobs[l] = dev.alloc<int32_t>(P * 3);
        for (uint32_t p = 0; p < P; ++p) { jobs[l][p * 3 + 0] = int32_t(p); jobs[l][p * 3 + 1] = int32_t((l * 10 + p * 7) % E); jobs[l][p * 3 + 2] = int32_t(p); }
    }
    sycl::half* g = dev.alloc<sycl::half>(uint64_t(P) * N); sycl::half* u = dev.alloc<sycl::half>(uint64_t(P) * N);
    sycl::half* hh = dev.alloc<sycl::half>(uint64_t(P) * N);
    const double bytes_step = double(L) * P * 2 * slice;
    auto run = [&](bool fused) {
        std::vector<double> ms;
        for (int rep = 0; rep < 6; ++rep) {
            dev.q.wait();
            const auto t0 = std::chrono::steady_clock::now();
            for (uint32_t l = 0; l < L; ++l) {
                if (fused) gemv_q4_K_q8s_grouped_dual_swiglu(dev.q, xq, slots, slot_bytes, 0, slice, jobs[l], hh, K, N, P);
                else { gemv_q4_K_q8s_grouped_dual(dev.q, xq, slots, slot_bytes, 0, slice, jobs[l], g, u, K, N, P);
                       swiglu(dev.q, g, u, hh, uint64_t(P) * N); }
            }
            dev.q.wait();
            const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (rep) ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        std::printf("  [bench] Q4_K  int-dot  %-12s gate/up: %6.2f ms / 48 layers = %6.1f us/layer, %5.0f GB/s of bank bytes\n",
                    fused ? "fused" : "dual+swiglu", med, med * 1000.0 / L, bytes_step / (med * 1e-3) / 1e9);
    };
    run(false); run(true);
    for (auto* j : jobs) dev.free(j);
    dev.free(slots); dev.free(rows); dev.free(xq); dev.free(g); dev.free(u); dev.free(hh);
}

int main() {
    Dev dev;
    std::printf("qwen4exp bank kernels on %s\n", dev.q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(20261107u);
    {   // 3. the kind table
        BankQ k;
        check(qwen4_bank_kind(DType::kQ8_0, 640, k) && k == BankQ::kQ8_0 && qwen4_bank_kind(DType::kQ8_0, 2560, k), "kind: Q8_0 at K=640 and 2560");
        check(qwen4_bank_kind(DType::kQ5_0, 640, k) && k == BankQ::kQ5_0 && qwen4_bank_kind(DType::kQ4_0, 640, k) && k == BankQ::kQ4_0, "kind: Q5_0 and Q4_0 at K=640");
        check(qwen4_bank_kind(DType::kQ5_K, 2560, k) && k == BankQ::kQ5_K && qwen4_bank_kind(DType::kQ6_K, 2560, k) && k == BankQ::kQ6_K, "kind: Q5_K and Q6_K at K=2560");
        check(!qwen4_bank_kind(DType::kQ5_K, 640, k) && !qwen4_bank_kind(DType::kQ6_K, 640, k), "kind: a K-quant needs K % 256 == 0 (640 refused)");
        check(!qwen4_bank_kind(DType::kQ8_0, 650, k) && !qwen4_bank_kind(DType::kQ4_K, 2560, k) && !qwen4_bank_kind(DType::kQ5_1, 640, k) && !qwen4_bank_kind(DType::kF16, 640, k),
              "kind: K % 32, and Q4_K / Q5_1 / F16 are not these kernels' (they have their own paths)");
        check(qwen4_bank_col_bytes(BankQ::kQ8_0, 640) == 20 * 34 && qwen4_bank_col_bytes(BankQ::kQ5_K, 2560) == 10 * 176 &&
              qwen4_bank_col_bytes(BankQ::kQ6_K, 2560) == 10 * 210 && qwen4_bank_col_bytes(BankQ::kQ5_0, 640) == 20 * 22 &&
              qwen4_bank_col_bytes(BankQ::kQ4_0, 640) == 20 * 18, "column bytes per kind");
    }
    for (BankQ kind : {BankQ::kQ8_0, BankQ::kQ5_0, BankQ::kQ4_0}) {
        run_kind(dev, kind, 640, 64, rng);     // the down bank's K
        run_kind(dev, kind, 2560, 48, rng);    // the gate/up bank's K (the Q8_0 build)
        run_kind(dev, kind, 64, 7, rng);       // the N tail
    }
    for (BankQ kind : {BankQ::kQ5_K, BankQ::kQ6_K}) {
        run_kind(dev, kind, 2560, 64, rng);
        run_kind(dev, kind, 512, 7, rng);
    }
    run_q4k_fused(dev, rng);
    run_q51_tab(dev, rng);
    if (const char* b = std::getenv("IE_BANK_BENCH"); b && *b == '1') {   // timing, not a gate: the dual kernels at the gate/up shape
        for (BankQ kind : {BankQ::kQ5_K, BankQ::kQ6_K, BankQ::kQ8_0}) bench_dual(dev, kind, rng);
        for (BankQ kind : {BankQ::kQ5_K, BankQ::kQ6_K, BankQ::kQ8_0}) bench_tiles(dev, kind, rng);
        bench_down_tiles(dev, rng);
        bench_q4k_dual(dev, rng);
    }
    std::printf("\n%s  (%d failures)\n", g_fail ? "\033[1m\033[31mGATE FAILED\033[0m" : "\033[1m\033[32mGATE PASSED\033[0m", g_fail);
    return g_fail ? 1 : 0;
}
