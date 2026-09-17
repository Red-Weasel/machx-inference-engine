// tools/ds41_tier_test.cpp — V4.1 experts through V4's residency machinery, all three tiers,
// vs the reference MoE golden. Tiering moves bytes, not arithmetic: two different tier splits
// must give the same output, and both must track the reference's routed output (Q8 path).
#include "ie/deepseek41_experts.hpp"

#include <cmath>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <chrono>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& w, const std::string& d = "") {
    std::printf("%s%s%s\n", ok ? "[ ok ] " : "[FAIL] ", w.c_str(), d.empty() ? "" : ("  (" + d + ")").c_str()); if (!ok) ++g_fail; }
template <class T> std::vector<T> rd(const std::string& p, size_t n) {
    std::vector<T> v(n); std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(1); }
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T))); return v; }
double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0, s = 0; for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, std::fabs(double(a[i]) - double(b[i]))); s = std::max(s, std::fabs(double(b[i]))); }
    return s > 0 ? m / s : 0; }
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : "/tmp/ds41_golden";
    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    const auto& c = m.config();
    const uint32_t T = 8, H = c.dim, E = c.n_routed_experts, TK = c.n_activated_experts;
    auto x = rd<float>(gd + "/moe_in.f32", size_t(T) * H);
    auto gi = rd<int32_t>(gd + "/gate_indices.i32", size_t(T) * TK);
    auto gw = rd<float>(gd + "/gate_weights.f32", size_t(T) * TK);
    auto gout = rd<float>(gd + "/moe_routed_out.f32", size_t(T) * H);

    sycl::device dev;
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; goto found; }
    std::fprintf(stderr, "no Arc GPU\n"); return 1;
found:
    // explicit single-device context (gate 8b finding 7): sycl::queue(dev) binds the two-card platform context
    sycl::queue q(sycl::context(dev), dev, sycl::property_list{sycl::property::queue::in_order{}});
    const bool hf = dev.has(sycl::aspect::ext_intel_free_memory);
    auto fv = [&]() -> uint64_t { return hf ? dev.get_info<sycl::ext::intel::info::device::free_memory>() : 0; };
    const uint64_t f0 = fv();
    float* dx = sycl::malloc_device<float>(x.size(), q); float* dy = sycl::malloc_device<float>(size_t(T) * H, q);
    q.memcpy(dx, x.data(), x.size() * 4).wait();

    // The two pack paths -- through the mapping, and pread + in-place permute -- must produce the
    // same bytes: every tier is filled by the pread path now, and this is what ties it to the
    // mapping path the reference comparison below was first proved on. Expert 0 of layer 0.
    {
        ie::Ds4SlotLayout lay;
        if (auto e = ie::ds41_slot_layout(H, c.moe_inter_dim, lay); !e.empty()) { std::fprintf(stderr, "layout: %s\n", e.c_str()); return 1; }
        std::vector<uint8_t> a(lay.bytes, 0xA5), b(lay.bytes, 0x5A);
        void* bounce = nullptr;
        if (posix_memalign(&bounce, 4096, size_t(ie::ds41_pack_bounce_bytes(lay))) != 0) { std::fprintf(stderr, "bounce alloc\n"); return 1; }
        const auto& Lw = m.layers()[0];
        if (auto e = ie::ds41_slot_pack(lay, Lw.exp_w1[0], Lw.exp_w3[0], Lw.exp_w2[0], a.data()); !e.empty()) { std::fprintf(stderr, "pack: %s\n", e.c_str()); return 1; }
        if (auto e = ie::ds41_slot_pack_pread(lay, m.store(), Lw.exp_w1[0], Lw.exp_w3[0], Lw.exp_w2[0], b.data(), bounce); !e.empty()) { std::fprintf(stderr, "pack_pread: %s\n", e.c_str()); return 1; }
        size_t first = 0; while (first < a.size() && a[first] == b[first]) ++first;
        check(a == b, "pread pack == mapping pack, byte for byte (one expert, all three planes + scales)",
              first == a.size() ? std::to_string(a.size()) + " bytes" : "first difference at byte " + std::to_string(first));
        // the decode shape (Phase 11): the same expert packed as 8 slices, and as 3 (uneven), must be the same bytes
        for (uint32_t ns : {8u, 3u}) {
            std::vector<uint8_t> d(lay.bytes, 0x3C);
            for (uint32_t sl = 0; sl < ns; ++sl)
                if (auto e = ie::ds41_slot_pack_pread(lay, m.store(), Lw.exp_w1[0], Lw.exp_w3[0], Lw.exp_w2[0], d.data(), bounce, nullptr, nullptr, sl, ns); !e.empty()) { std::fprintf(stderr, "pack_pread slice: %s\n", e.c_str()); return 1; }
            size_t f2 = 0; while (f2 < a.size() && a[f2] == d[f2]) ++f2;
            check(a == d, "pread pack in " + std::to_string(ns) + " slices == mapping pack, byte for byte", f2 == a.size() ? "" : "first difference at byte " + std::to_string(f2));
        }
        std::free(bounce);
    }

    // Phase 15 C2 (gate 15 finding 6): with IE_DS41_EXPERT_FILE, whole slots read from the file equal the pack
    // path's bytes, byte for byte, over several experts -- the first covered of layer 0, the first covered of
    // layer 20 (the card boundary), the last covered of the last layer
    if (const char* fp = std::getenv("IE_DS41_EXPERT_FILE")) {
        ie::Ds4SlotLayout fl; if (auto e = ie::ds41_slot_layout(H, c.moe_inter_dim, fl); !e.empty()) { std::fprintf(stderr, "layout: %s\n", e.c_str()); return 1; }
        const int fd = open(fp, O_RDONLY), dfd = open(fp, O_RDONLY | O_DIRECT);
        if (fd < 0 || dfd < 0) { std::printf("[FAIL] expert file %s: %s\n", fp, std::strerror(errno)); ++g_fail; }
        else {
            ie::Ds41ExpertFileHeader hd{}; bool ok = pread(fd, &hd, sizeof hd, 0) == ssize_t(sizeof hd) && std::memcmp(hd.magic, ie::kDs41ExpertFileMagic, 8) == 0 && hd.slot_bytes == fl.bytes;
            std::vector<uint64_t> table(ok ? size_t(hd.n_layers) * hd.n_experts : 0);
            if (ok) ok = pread(fd, table.data(), table.size() * 8, off_t(hd.table_off)) == ssize_t(table.size() * 8);
            check(ok, "the expert file's header and table read (IESLOT01, the slot layout of this model)");
            if (ok) {
                auto first_of = [&](uint32_t l, bool last) -> int32_t { if (last) { for (int32_t e = int32_t(hd.n_experts) - 1; e >= 0; --e) if (table[size_t(l) * hd.n_experts + e]) return e; } else { for (uint32_t e = 0; e < hd.n_experts; ++e) if (table[size_t(l) * hd.n_experts + e]) return int32_t(e); } return -1; };
                void* a = nullptr; void* b = nullptr; void* bounce = nullptr;
                posix_memalign(&a, 4096, size_t(fl.bytes)); posix_memalign(&b, 4096, size_t(fl.bytes)); posix_memalign(&bounce, 4096, size_t(ie::ds41_pack_bounce_bytes(fl)));
                for (auto [l, last] : {std::pair<uint32_t, bool>{0, false}, {hd.n_layers / 2, false}, {hd.n_layers - 1, true}}) {
                    const int32_t e = first_of(l, last);
                    if (e < 0) { check(false, "layer " + std::to_string(l) + ": the file holds no expert of it"); continue; }
                    const auto& Lw = m.layers()[l];
                    const std::string e1 = ie::ds41_slot_pack_pread(fl, m.store(), Lw.exp_w1[e], Lw.exp_w3[e], Lw.exp_w2[e], a, bounce);
                    const std::string e2 = ie::ds41_slot_pread_file(dfd, table[size_t(l) * hd.n_experts + e], b, fl.bytes, 0, 1, nullptr);
                    check(e1.empty() && e2.empty() && std::memcmp(a, b, size_t(fl.bytes)) == 0, "expert file: layer " + std::to_string(l) + " expert " + std::to_string(e) + " read whole from the file == the pack path, byte for byte", e1 + e2);
                }
                std::free(a); std::free(b); std::free(bounce);
            }
            close(fd); close(dfd);
        }
    }
    // Phase 15 C0: the SSE2 permute equals the scalar reference on random planes of the model's two sizes and an odd one
    {
        ie::Ds4SlotLayout pl; if (auto e = ie::ds41_slot_layout(H, c.moe_inter_dim, pl); !e.empty()) { std::fprintf(stderr, "layout: %s\n", e.c_str()); return 1; }
        bool all = true; for (uint64_t n : {uint64_t(pl.gate.len0), uint64_t(pl.down.len0), uint64_t(16 * 1237)}) for (uint32_t sd = 1; sd <= 3; ++sd) all &= ie::ds41_permute_plane_selftest(n, sd);
        check(all, "the vectorised nibble permute equals the scalar reference on 9 random planes (byte for byte)");
    }
    // index-order ranking (no profile yet); layer 0 only
    std::vector<std::vector<uint32_t>> rank(c.n_layers, std::vector<uint32_t>(E));
    for (auto& r : rank) std::iota(r.begin(), r.end(), 0u);
    // which tiers does the golden's routing touch? report it so "all three exercised" is a fact
    auto tiers_hit = [&](uint32_t ns, uint32_t np) {
        uint32_t a = 0, b = 0, cc = 0;
        for (int32_t e : gi) { if (uint32_t(e) < ns) ++a; else if (uint32_t(e) < ns + np) ++b; else ++cc; }
        return std::to_string(a) + " static / " + std::to_string(b) + " pinned / " + std::to_string(cc) + " mmap selections"; };

    std::vector<float> out_a, out_b; uint32_t full_n = 0;
    for (int cfg = 0; cfg < 2; ++cfg) {
        const uint32_t ns = cfg == 0 ? 96 : 40, np = cfg == 0 ? 192 : 100, ss = cfg == 0 ? 8 : 4;
        std::printf("=== config %d: static %u, pinned %u, stream %u, mmap %u -> %s ===\n", cfg, ns, np, ss, E - ns - np, tiers_hit(ns, np).c_str());
        ie::Ds41ExpertTier tier;
        const auto t0 = std::chrono::steady_clock::now();
        if (auto e = tier.init(q, m, 0, 1, rank, ns, np, ss, 64); !e.empty()) { std::fprintf(stderr, "init: %s\n", e.c_str()); return 1; }
        std::printf("       init %.1f s: VRAM %.2f GiB, pinned %.2f GiB\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                    double(tier.vram_bytes()) / (1u << 30), double(tier.pinned_bytes()) / (1u << 30));
        if (auto e = tier.moe(q, 0, dx, gi.data(), gw.data(), T, dy, c.swiglu_limit); !e.empty()) { std::fprintf(stderr, "moe: %s\n", e.c_str()); return 1; }
        std::vector<float> got(size_t(T) * H); q.memcpy(got.data(), dy, got.size() * 4).wait();
        const auto& s = tier.last();
        std::printf("       moe: %u static, %u pinned, %u mmap experts; %.1f MiB mmap->VRAM; %.1f ms\n",
                    s.experts_static, s.experts_pinned, s.experts_mmap, double(s.bytes_mmap_to_vram) / (1 << 20), s.ms);
        const double r = rel(got, gout);
        check(r < 8e-3, "tiered routed output tracks the reference (Q8 int-dot path)", "rel " + std::to_string(r));
        (cfg == 0 ? out_a : out_b) = got; if (cfg == 0) full_n = s.experts_static + s.experts_pinned + s.experts_mmap;
        // a second call must hit the stream slots it just filled (no mmap change, fewer pinned fetches)
        if (auto e = tier.moe(q, 0, dx, gi.data(), gw.data(), T, dy, c.swiglu_limit); !e.empty()) { std::fprintf(stderr, "moe2: %s\n", e.c_str()); return 1; }
        std::vector<float> got2(size_t(T) * H); q.memcpy(got2.data(), dy, got2.size() * 4).wait();
        check(got2 == got, "a second identical call is bit-identical (residency is not arithmetic)");
        tier.free_storage(q);
    }
    check(out_a == out_b, "two different tier splits give BIT-IDENTICAL output");
    // expert parallel (docs/deepseek41/38 step 1): two complementary half tiers, each with half of
    // config 0's counts over its half of the ranking, sum to the full tier's output
    {
        std::vector<float> sum(size_t(T) * H, 0.f); uint32_t n_sel = 0; double pinned_gib = 0;
        for (uint32_t part = 0; part < 2; ++part) {
            ie::Ds41ExpertTier tier;
            if (auto e = tier.init(q, m, 0, 1, rank, 48, 96, 4, 64, part, 2); !e.empty()) { std::fprintf(stderr, "init part %u: %s\n", part, e.c_str()); return 1; }
            if (auto e = tier.moe(q, 0, dx, gi.data(), gw.data(), T, dy, c.swiglu_limit); !e.empty()) { std::fprintf(stderr, "moe part %u: %s\n", part, e.c_str()); return 1; }
            std::vector<float> got(size_t(T) * H); q.memcpy(got.data(), dy, got.size() * 4).wait();
            const auto& s = tier.last(); n_sel += s.experts_static + s.experts_pinned + s.experts_mmap; pinned_gib += double(tier.pinned_bytes()) / (1u << 30);
            std::printf("       part %u of 2: %u static, %u pinned, %u mmap experts of its half; %.1f ms\n", part, s.experts_static, s.experts_pinned, s.experts_mmap, s.ms);
            if (auto e = tier.moe(q, 0, dx, gi.data(), gw.data(), T, dy, c.swiglu_limit); !e.empty()) { std::fprintf(stderr, "moe2 part %u: %s\n", part, e.c_str()); return 1; }
            std::vector<float> got2(size_t(T) * H); q.memcpy(got2.data(), dy, got2.size() * 4).wait();
            check(got2 == got, "part " + std::to_string(part) + ": a second identical call is bit-identical");
            for (size_t i = 0; i < sum.size(); ++i) sum[i] += got[i];
            tier.free_storage(q);
        }
        const double r = rel(sum, out_a);
        char rb[32]; std::snprintf(rb, sizeof rb, "rel %.2e", r);
        check(r < 1e-6, "the two half tiers' partials sum to the full tier's output within 1e-6 (fp32 reassociation of ~6 terms is ~1e-7)", rb);
        check(n_sel == full_n, "the two halves' expert counts add up to the full tier's", std::to_string(n_sel) + " of " + std::to_string(full_n));
        std::printf("       the two halves pin %.2f GiB together\n", pinned_gib);
    }
    // the same at layers 1 and 14 (the engram layers, where the forward's EP path first diverged) with random
    // routing at T = 12: the full tier vs the two halves
    for (uint32_t L : {1u, 14u}) {
        const uint32_t T2 = 12; std::mt19937 rng(100 + L);
        std::vector<int32_t> ri(size_t(T2) * c.n_activated_experts); std::vector<float> rw(ri.size());
        for (uint32_t t = 0; t < T2; ++t) { std::vector<uint32_t> perm(E); std::iota(perm.begin(), perm.end(), 0u); std::shuffle(perm.begin(), perm.end(), rng);
            for (uint32_t k = 0; k < c.n_activated_experts; ++k) { ri[size_t(t) * c.n_activated_experts + k] = int32_t(perm[k]); rw[size_t(t) * c.n_activated_experts + k] = 0.1f + float(rng() % 1000) / 2000.f; } }
        std::vector<float> xr(size_t(T2) * H); for (auto& v2 : xr) v2 = float(int(rng() % 2001) - 1000) / 1000.f;
        float* dx2 = sycl::malloc_device<float>(xr.size(), q); float* dy2 = sycl::malloc_device<float>(xr.size(), q); q.memcpy(dx2, xr.data(), xr.size() * 4).wait();
        std::vector<float> full, sum(xr.size(), 0.f);
        for (int arm = 0; arm < 3; ++arm) {   // 0 = full, 1-2 = the halves
            ie::Ds41ExpertTier tier;
            const auto e = arm == 0 ? tier.init(q, m, L, 1, rank, 96, 192, 8, 64) : tier.init(q, m, L, 1, rank, 48, 96, 4, 64, uint32_t(arm - 1), 2);
            if (!e.empty()) { std::fprintf(stderr, "init L%u arm %d: %s\n", L, arm, e.c_str()); return 1; }
            if (auto e2 = tier.moe(q, L, dx2, ri.data(), rw.data(), T2, dy2, c.swiglu_limit); !e2.empty()) { std::fprintf(stderr, "moe L%u arm %d: %s\n", L, arm, e2.c_str()); return 1; }
            std::vector<float> got(xr.size()); q.memcpy(got.data(), dy2, got.size() * 4).wait();
            if (arm == 0) full = got; else for (size_t i = 0; i < sum.size(); ++i) sum[i] += got[i];
            tier.free_storage(q);
        }
        const double r = rel(sum, full); char rb[32]; std::snprintf(rb, sizeof rb, "rel %.2e", r);
        check(r < 1e-6, "layer " + std::to_string(L) + ", T = 12, random routing: the two halves sum to the full tier within 1e-6", rb);
        sycl::free(dx2, q); sycl::free(dy2, q);
    }
    sycl::free(dx, q); sycl::free(dy, q); q.wait_and_throw();
    check(hf && int64_t(f0) - int64_t(fv()) < int64_t(256ull << 20), "VRAM returned after both configs",
          std::to_string(double(int64_t(f0) - int64_t(fv())) / (1 << 20)).substr(0, 7) + " MiB unreturned");
    std::printf("\n%s\n", g_fail ? "TIER TEST: FAILURE(S)" : "TIER TEST: PASS");
    return g_fail ? 1 : 0;
}
