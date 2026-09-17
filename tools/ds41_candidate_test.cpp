// tools/ds41_candidate_test.cpp — V4.1 Phase 24 step 3, criterion B(b): the two-level candidate
// top-k's level one against the reference's semantics, on the cases that actually carry the logic.
// The CPU reference below is a line-by-line transcription of `select_candidate_blocks`
// (inference/model.py:583-610); the GPU kernel must agree with it EXACTLY -- this is a discrete
// selection, so a near-miss is a failure, not a tolerance.
#include "ie/deepseek41_candidate.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& w, const std::string& d = "") {
    std::printf("%s%s%s\n", ok ? "[ ok ] " : "[FAIL] ", w.c_str(), d.empty() ? "" : ("  (" + d + ")").c_str());
    if (!ok) ++g_fail; }
constexpr float NEG = -std::numeric_limits<float>::infinity();

// the reference, transcribed. `scores` [T, n_keys] raw; returns keep [T, n_blocks].
std::vector<uint8_t> ref_select(const std::vector<float>& scores, const std::vector<int32_t>& pos,
                                uint32_t T, uint32_t n_keys, uint32_t bs, uint32_t topk_blocks, uint32_t rate) {
    const uint32_t NB = (n_keys + bs - 1) / bs;
    std::vector<uint8_t> keep(size_t(T) * NB, 0);
    for (uint32_t t = 0; t < T; ++t) {
        const long thr = (long(pos[t]) + 1) / long(rate);            // compress_lens
        std::vector<float> b(NB, NEG);
        for (uint32_t k = 0; k < NB; ++k) {                          // amax over the block, -inf past thr
            float m = NEG;
            for (uint32_t e = k * bs; e < std::min((k + 1) * bs, n_keys); ++e)
                if (long(e) < thr) m = std::max(m, scores[size_t(t) * n_keys + e]);
            b[k] = m;
        }
        if (thr > 0) b[size_t((thr - 1) / long(bs))] = -NEG;         // pin the partly-filled newest block
        // topk(min(topk_blocks, NB)) then keep only picks with value > -inf; ties by smaller index,
        // matching the engine's strict total order (the reference's own tie-break is arbitrary)
        std::vector<uint32_t> ord(NB); std::iota(ord.begin(), ord.end(), 0u);
        std::stable_sort(ord.begin(), ord.end(), [&](uint32_t x, uint32_t y) { return b[x] > b[y]; });
        const uint32_t K = std::min(topk_blocks, NB);
        for (uint32_t i = 0; i < K; ++i) if (b[ord[i]] > NEG) keep[size_t(t) * NB + ord[i]] = 1;
    }
    return keep;
}

struct Gpu {
    sycl::queue q;
    explicit Gpu(const sycl::device& d) : q(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}}) {}
};
}  // namespace

int main() {
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) { if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue; for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d); }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    Gpu g(devs[0]);
    std::mt19937 rng(20260915);

    struct Case { const char* what; uint32_t T, n_keys, bs, tkb, rate; bool lowest_newest; };
    const std::vector<Case> cases = {
        // the stage is a NO-OP here: NB <= topk_blocks, so every reachable block must be kept
        {"a no-op length (fewer blocks than topk_blocks): every reachable block kept, no -inf pick admitted", 8, 4096, 8, 2048, 1, false},
        // the real selection: more blocks than topk_blocks
        {"the real selection (NB 8192 > topk_blocks 2048), rate 1", 8, 65536, 8, 2048, 1, false},
        // rate 2, so the causal threshold is half the position -- the ratio-2 layers' shape
        {"rate 2 (the ratio-2 layers' causal threshold)", 8, 65536, 8, 2048, 2, false},
        // the newest block given the LOWEST score in its row: only the +inf pin can keep it
        {"the partly-filled newest block pinned in even when its score is the lowest", 8, 65536, 8, 2048, 1, true},
        // a reachable length that is NOT a multiple of block_size, and a ragged last block
        {"a reachable length that is not a multiple of block_size, and a ragged last block", 8, 65533, 8, 2048, 1, false},
        // a small topk_blocks so the K-th largest sits deep in the histogram
        {"topk_blocks 37 (the K-th largest deep in the radix histogram)", 4, 8192, 8, 37, 1, false},
    };

    for (const Case& cs : cases) {
        const uint32_t NB = (cs.n_keys + cs.bs - 1) / cs.bs;
        std::vector<float> sc(size_t(cs.T) * cs.n_keys);
        std::uniform_real_distribution<float> U(-4.f, 4.f);
        for (auto& v : sc) v = U(rng);
        // positions chosen so the reachable length varies per row and is sometimes ragged
        std::vector<int32_t> pos(cs.T);
        for (uint32_t t = 0; t < cs.T; ++t) pos[t] = int32_t(cs.n_keys * cs.rate - 1 - t * 37u * cs.rate);
        if (cs.lowest_newest)
            for (uint32_t t = 0; t < cs.T; ++t) {
                const long thr = (long(pos[t]) + 1) / long(cs.rate);
                const uint32_t nb = uint32_t((thr - 1) / long(cs.bs));
                for (uint32_t e = nb * cs.bs; e < std::min((nb + 1) * cs.bs, cs.n_keys); ++e) sc[size_t(t) * cs.n_keys + e] = -1e30f;
            }

        float* d_sc = sycl::malloc_device<float>(sc.size(), g.q);
        int32_t* d_pos = sycl::malloc_device<int32_t>(pos.size(), g.q);
        float* d_bs = sycl::malloc_device<float>(size_t(cs.T) * NB, g.q);
        uint8_t* d_keep = sycl::malloc_device<uint8_t>(size_t(cs.T) * NB, g.q);
        g.q.memcpy(d_sc, sc.data(), sc.size() * 4).wait();
        g.q.memcpy(d_pos, pos.data(), pos.size() * 4).wait();
        ie::ds41_candidate_block_scores(g.q, d_sc, d_pos, d_bs, cs.T, cs.n_keys, cs.bs, cs.rate).wait();
        ie::ds41_candidate_select(g.q, d_bs, d_keep, cs.T, cs.n_keys, cs.bs, cs.tkb).wait();
        std::vector<uint8_t> got(size_t(cs.T) * NB);
        g.q.memcpy(got.data(), d_keep, got.size()).wait();

        const auto want = ref_select(sc, pos, cs.T, cs.n_keys, cs.bs, cs.tkb, cs.rate);
        size_t bad = 0, nkeep = 0; std::string first;
        for (size_t i = 0; i < want.size(); ++i) {
            nkeep += want[i];
            if (want[i] != got[i]) { if (first.empty()) first = "row " + std::to_string(i / NB) + " block " + std::to_string(i % NB) + " want " + std::to_string(want[i]) + " got " + std::to_string(got[i]); ++bad; }
        }
        // the count the reference implies, checked separately so a systematic off-by-one shows
        const uint32_t K = std::min(cs.tkb, NB);
        size_t row0 = 0; for (uint32_t b = 0; b < NB; ++b) row0 += got[b];
        check(bad == 0, std::string(cs.what), bad ? (std::to_string(bad) + " of " + std::to_string(want.size()) + " differ, first " + first)
                                                  : ("identical over " + std::to_string(want.size()) + " blocks, " + std::to_string(nkeep) + " kept"));
        check(row0 <= K, std::string("  .. row 0 keeps at most min(topk_blocks, n_blocks) = ") + std::to_string(K), std::to_string(row0) + " kept");
        // A POSITIVE assertion for the pin, because agreeing with my own transcription would not catch a
        // shared misreading: the newest block's scores were forced to -1e30 (the lowest in the row), so it is
        // kept ONLY if the +inf pin fired. Its score must also NOT have survived as the raw max.
        if (cs.lowest_newest) {
            uint32_t pinned_kept = 0;
            for (uint32_t t = 0; t < cs.T; ++t) {
                const long thr = (long(pos[t]) + 1) / long(cs.rate);
                pinned_kept += got[size_t(t) * NB + size_t((thr - 1) / long(cs.bs))];
            }
            check(pinned_kept == cs.T, "  .. and the pin FIRED: every row's newest block is kept despite the lowest score",
                  std::to_string(pinned_kept) + " of " + std::to_string(cs.T) + " rows");
        }
        sycl::free(d_sc, g.q); sycl::free(d_pos, g.q); sycl::free(d_bs, g.q); sycl::free(d_keep, g.q);
    }
    std::printf("\nCANDIDATE TEST: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
