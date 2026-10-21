// tests/unit/ds41_tp_slice_test.cpp -- tensor parallel's expert slices (2026-10-20): a part of an expert is exactly the
// whole expert's gate / up rows and down columns over the part's intermediate units, byte for byte after the pack's shuffle,
// and the parts tile the expert. Host only: synthetic MXFP4 planes of the checkpoint's shapes (H 5120, EF 2304).
#include "ie/deepseek41_experts.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace ie;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

struct Mat { SafeTensorInfo w, s; std::vector<uint8_t> wb, sb; Ds41Tensor t; };
static void make(Mat& m, uint32_t N, uint32_t K, std::mt19937& rng) {   // [N, K] MXFP4: nibbles [N, K/2], E8M0 [N, K/32]
    m.wb.resize(size_t(N) * K / 2); m.sb.resize(size_t(N) * K / 32);
    for (auto& b : m.wb) b = uint8_t(rng()); for (auto& b : m.sb) b = uint8_t(rng());
    m.w.shape = {int64_t(N), int64_t(K / 2)}; m.w.data = m.wb.data(); m.w.nbytes = m.wb.size();
    m.s.shape = {int64_t(N), int64_t(K / 32)}; m.s.data = m.sb.data(); m.s.nbytes = m.sb.size();
    m.t.w = &m.w; m.t.s = &m.s;
}

int main() {
    const uint32_t H = 5120, EF = 2304;
    Ds4SlotLayout whole;
    CHECK(ds41_slot_layout(H, EF, whole).empty(), "whole layout");
    Ds4SlotLayout one;
    CHECK(ds41_slot_layout_tp(H, EF, 0, 1, one).empty() && one.bytes == whole.bytes && one.down.K == EF, "1 part == the whole layout");
    Ds4SlotLayout bad;
    CHECK(!ds41_slot_layout_tp(H, EF, 0, 5, bad).empty(), "5 parts must be refused (72 blocks)");
    CHECK(!ds41_slot_layout_tp(H, EF, 2, 2, bad).empty(), "part 2 of 2 must be refused");

    std::mt19937 rng(20261020u);
    Mat g, u, d; make(g, EF, H, rng); make(u, EF, H, rng); make(d, H, EF, rng);   // gate / up [EF, H], down [H, EF]
    std::vector<uint8_t> A(whole.bytes);
    CHECK(ds41_slot_pack(whole, g.t, u.t, d.t, A.data()).empty(), "pack whole");

    for (uint32_t n : {2u, 3u, 4u, 8u}) {
        const uint32_t efc = EF / n;
        uint64_t sum_bytes = 0;
        for (uint32_t p = 0; p < n; ++p) {
            Ds4SlotLayout lay;
            if (!ds41_slot_layout_tp(H, EF, p, n, lay).empty()) { CHECK(false, "layout %u of %u", p, n); continue; }
            CHECK(lay.gate.N == efc && lay.gate.K == H && lay.up.N == efc && lay.down.K == efc && lay.down.N == H, "shapes %u/%u", p, n);
            sum_bytes += lay.gate.len0 + lay.gate.len1 + lay.up.len0 + lay.up.len1 + lay.down.len0 + lay.down.len1;
            std::vector<uint8_t> B(lay.bytes);
            if (auto e = ds41_slot_pack(lay, g.t, u.t, d.t, B.data()); !e.empty()) { CHECK(false, "pack %u/%u: %s", p, n, e.c_str()); continue; }
            // gate / up: rows [p*efc, (p+1)*efc) of the whole slot's planes, nibbles and scales
            for (const auto* pr : {&lay.gate, &lay.up}) {
                const Ds4MatPlanes& w = pr == &lay.gate ? whole.gate : whole.up;
                CHECK(std::memcmp(B.data() + pr->off0, A.data() + w.off0 + uint64_t(p) * efc * (H / 2), pr->len0) == 0, "gate/up nibbles %u/%u", p, n);
                CHECK(std::memcmp(B.data() + pr->off1, A.data() + w.off1 + uint64_t(p) * efc * (H / 32), pr->len1) == 0, "gate/up scales %u/%u", p, n);
            }
            // down: each row's columns [p*efc, (p+1)*efc)
            uint32_t bad_rows = 0;
            for (uint32_t r = 0; r < H; ++r) {
                if (std::memcmp(B.data() + lay.down.off0 + uint64_t(r) * (efc / 2), A.data() + whole.down.off0 + uint64_t(r) * (EF / 2) + uint64_t(p) * efc / 2, efc / 2)) ++bad_rows;
                if (std::memcmp(B.data() + lay.down.off1 + uint64_t(r) * (efc / 32), A.data() + whole.down.off1 + uint64_t(r) * (EF / 32) + uint64_t(p) * efc / 32, efc / 32)) ++bad_rows;
            }
            CHECK(bad_rows == 0, "down: %u row runs differ (%u/%u)", bad_rows, p, n);
        }
        // the parts tile the expert: their planes hold exactly the whole slot's bytes
        const uint64_t whole_planes = whole.gate.len0 + whole.gate.len1 + whole.up.len0 + whole.up.len1 + whole.down.len0 + whole.down.len1;
        CHECK(sum_bytes == whole_planes, "%u parts hold %llu plane bytes, the whole %llu", n, (unsigned long long)sum_bytes, (unsigned long long)whole_planes);
        Ds4SlotLayout l0; ds41_slot_layout_tp(H, EF, 0, n, l0);
        CHECK(ds41_pack_bounce_bytes(l0) >= uint64_t(H) * (EF / 2), "bounce covers down's whole source rows at %u parts", n);
    }
    if (g_fail) { std::printf("ds41_tp_slice_test: %d FAILED\n", g_fail); return 1; }
    std::printf("ds41_tp_slice_test: PASS (layouts for 1-8 parts, packs byte-exact against the whole expert, parts tile it)\n");
    return 0;
}
