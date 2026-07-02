// tools/ie_det_probe.cpp — kernel-level run-to-run determinism probe.
//
// 2026-06-10 dense nondeterminism bisect: ie-perplexity on qwen3-8b shows
// run-to-run PPL spread (19.0-19.4) starting at the SECOND decode step
// (ctx_len=2); ctx_len=1 is bit-stable.  The only op whose computation
// structurally changes between ctx 1 and 2 is the FA-2 split-K decode
// inner online-softmax loop.  This tool freezes pseudo-random inputs and
// re-runs one kernel N times, hashing the output bytes each iteration —
// any hash flicker is intrinsic kernel nondeterminism (same methodology
// as ie-bug-monitor from the DeltaNet investigation).
//
// Usage:
//   ie-det-probe [--iters N] [--test fa2_dense|fa2_crown|fa2_int8|naive|gemv_fp16|all]
//                [--start-pos P]
//
// Exit code: 0 if all selected tests bit-stable, 1 otherwise.

#include "ie/allocator.hpp"
#include "ie/dense_transformer.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Deterministic xorshift fill in [-1, 1).
void fill_half(std::vector<sycl::half>& v, uint64_t seed) {
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
    for (auto& x : v) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        x = sycl::half(float(int64_t(s % 20001) - 10000) / 10000.0f);
    }
}

uint64_t fnv1a(const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

struct Result { int distinct = 0; int first_div = -1; int n_div = 0; };

// Runs `submit(iter)` n_iters times; after each, copies `n_out` halfs from
// d_out to host and hashes.  Reports flicker vs iter-0 hash.
template <typename F>
Result run_loop(sycl::queue& q, sycl::half* d_out, size_t n_out,
                int n_iters, F&& submit) {
    std::vector<sycl::half> h_out(n_out);
    std::vector<uint64_t> seen;
    Result r;
    uint64_t h0 = 0;
    for (int it = 0; it < n_iters; ++it) {
        submit(it);
        q.wait();
        q.memcpy(h_out.data(), d_out, n_out * sizeof(sycl::half)).wait();
        const uint64_t h = fnv1a(h_out.data(), n_out * sizeof(sycl::half));
        if (it == 0) h0 = h;
        else if (h != h0) {
            if (r.first_div < 0) r.first_div = it;
            r.n_div++;
        }
        bool found = false;
        for (uint64_t s : seen) if (s == h) { found = true; break; }
        if (!found) seen.push_back(h);
    }
    r.distinct = int(seen.size());
    return r;
}

void report(const char* name, const Result& r, int iters) {
    std::printf("  %-28s : %s  distinct_hashes=%d  first_div_iter=%d  n_div=%d/%d\n",
                name, r.distinct == 1 ? "BIT-STABLE" : "NON-DETERMINISTIC",
                r.distinct, r.first_div, r.n_div, iters - 1);
}

}  // namespace

int main(int argc, char** argv) {
    int n_iters = 200;
    std::string test = "all";
    uint32_t start_pos = 1;          // ctx_len = start_pos + 1 = 2
    int dump_step = -1;              // model test: per-layer dump at this step
    bool perturb = false;            // fa2 tests: vary out-of-range memory
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--iters" && i + 1 < argc) n_iters = std::atoi(argv[++i]);
        else if (a == "--test"  && i + 1 < argc) test    = argv[++i];
        else if (a == "--start-pos" && i + 1 < argc) start_pos = std::atoi(argv[++i]);
        else if (a == "--dump-step" && i + 1 < argc) dump_step = std::atoi(argv[++i]);
        else if (a == "--perturb") perturb = true;
    }

    ie::DeviceAllocator alloc;
    if (auto err = alloc.init(); !err.empty()) {
        std::fprintf(stderr, "alloc: %s\n", err.c_str()); return 1;
    }
    auto& q = alloc.queue();
    std::printf("ie-det-probe — device: %s, iters=%d, start_pos=%u (ctx=%u)\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(),
                n_iters, start_pos, start_pos + 1);

    bool any_nondet = false;
    const uint32_t max_ctx = 1024;

    // ---- FA-2 fp16 split-K decode at a given head shape ------------------
    auto fa2_case = [&](const char* name, uint32_t n_q, uint32_t n_kv,
                        uint32_t hd) {
        const uint32_t ctx = start_pos + 1;
        const size_t q_sz = size_t(n_q) * hd, kv_sz = size_t(n_kv) * hd;
        const size_t cache_sz = size_t(n_kv) * max_ctx * hd;
        const uint32_t n_chunks_max = (max_ctx + 63) / 64;
        const size_t part_sz = size_t(n_chunks_max) * n_q * (hd + 2);

        std::vector<sycl::half> h_q(q_sz), h_k(kv_sz), h_v(kv_sz), h_cache_k(cache_sz), h_cache_v(cache_sz);
        fill_half(h_q, 11); fill_half(h_k, 22); fill_half(h_v, 33);
        fill_half(h_cache_k, 44); fill_half(h_cache_v, 55);

        auto* d_q  = sycl::malloc_device<sycl::half>(q_sz, q);
        auto* d_k  = sycl::malloc_device<sycl::half>(kv_sz, q);
        auto* d_v  = sycl::malloc_device<sycl::half>(kv_sz, q);
        auto* d_kc = sycl::malloc_device<sycl::half>(cache_sz, q);
        auto* d_vc = sycl::malloc_device<sycl::half>(cache_sz, q);
        auto* d_y  = sycl::malloc_device<sycl::half>(q_sz, q);
        auto* d_p  = sycl::malloc_device<float>(part_sz, q);
        q.memcpy(d_q, h_q.data(), q_sz * 2).wait();
        q.memcpy(d_k, h_k.data(), kv_sz * 2).wait();
        q.memcpy(d_v, h_v.data(), kv_sz * 2).wait();
        q.memcpy(d_kc, h_cache_k.data(), cache_sz * 2).wait();
        q.memcpy(d_vc, h_cache_v.data(), cache_sz * 2).wait();
        q.memset(d_y, 0, q_sz * 2).wait();
        (void)ctx;

        // --perturb: refill cache rows >= ctx and ALL partials scratch with
        // iteration-dependent garbage before each launch.  Valid inputs
        // (q/k/v, cache rows < ctx) stay frozen.  Output change => kernel
        // reads beyond the valid region.
        std::vector<sycl::half> h_garb(cache_sz);
        std::vector<float>      h_garbf(part_sz);
        auto r = run_loop(q, d_y, q_sz, n_iters, [&](int it) {
            if (perturb) {
                fill_half(h_garb, 1000 + it);
                // overwrite only rows >= ctx for each kv head
                for (uint32_t kvh = 0; kvh < n_kv; ++kvh) {
                    const size_t row0 = (size_t(kvh) * max_ctx + ctx) * hd;
                    const size_t nel  = size_t(max_ctx - ctx) * hd;
                    q.memcpy(d_kc + row0, h_garb.data() + row0, nel * 2);
                    q.memcpy(d_vc + row0, h_garb.data() + row0, nel * 2);
                }
                uint64_t s = 777 + it;
                for (auto& x : h_garbf) {
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                    x = float(int64_t(s % 20001) - 10000) / 100.0f;
                }
                q.memcpy(d_p, h_garbf.data(), part_sz * 4);
                q.wait();
            }
            ie::full_attention_fa2_decode(q, d_q, d_k, d_v, d_kc, d_vc, d_y,
                                          d_p, start_pos, n_q, n_kv, hd,
                                          max_ctx);
        });
        report(name, r, n_iters);
        if (r.distinct != 1) any_nondet = true;
        sycl::free(d_q, q); sycl::free(d_k, q); sycl::free(d_v, q);
        sycl::free(d_kc, q); sycl::free(d_vc, q); sycl::free(d_y, q);
        sycl::free(d_p, q);
    };

    // ---- naive full_attention at T=1, dense shape -------------------------
    auto naive_case = [&](const char* name, uint32_t n_q, uint32_t n_kv,
                          uint32_t hd) {
        const size_t q_sz = size_t(n_q) * hd, kv_sz = size_t(n_kv) * hd;
        const size_t cache_sz = size_t(n_kv) * max_ctx * hd;
        std::vector<sycl::half> h_q(q_sz), h_k(kv_sz), h_v(kv_sz), h_ck(cache_sz), h_cv(cache_sz);
        fill_half(h_q, 11); fill_half(h_k, 22); fill_half(h_v, 33);
        fill_half(h_ck, 44); fill_half(h_cv, 55);
        auto* d_q  = sycl::malloc_device<sycl::half>(q_sz, q);
        auto* d_k  = sycl::malloc_device<sycl::half>(kv_sz, q);
        auto* d_v  = sycl::malloc_device<sycl::half>(kv_sz, q);
        auto* d_kc = sycl::malloc_device<sycl::half>(cache_sz, q);
        auto* d_vc = sycl::malloc_device<sycl::half>(cache_sz, q);
        auto* d_y  = sycl::malloc_device<sycl::half>(q_sz, q);
        q.memcpy(d_q, h_q.data(), q_sz * 2).wait();
        q.memcpy(d_k, h_k.data(), kv_sz * 2).wait();
        q.memcpy(d_v, h_v.data(), kv_sz * 2).wait();
        q.memcpy(d_kc, h_ck.data(), cache_sz * 2).wait();
        q.memcpy(d_vc, h_cv.data(), cache_sz * 2).wait();
        q.memset(d_y, 0, q_sz * 2).wait();

        auto r = run_loop(q, d_y, q_sz, n_iters, [&](int) {
            ie::full_attention(q, d_q, d_k, d_v, d_kc, d_vc, d_y,
                               /*T=*/1, start_pos, n_q, n_kv, hd, max_ctx);
        });
        report(name, r, n_iters);
        if (r.distinct != 1) any_nondet = true;
        sycl::free(d_q, q); sycl::free(d_k, q); sycl::free(d_v, q);
        sycl::free(d_kc, q); sycl::free(d_vc, q); sycl::free(d_y, q);
    };

    // ---- gemv_fp16 (qwen3 attn_v shape) -----------------------------------
    auto gemv_case = [&]() {
        const uint32_t K = 4096, N = 1024;
        std::vector<sycl::half> h_a(K), h_w(size_t(K) * N);
        fill_half(h_a, 7); fill_half(h_w, 8);
        auto* d_a = sycl::malloc_device<sycl::half>(K, q);
        auto* d_w = sycl::malloc_device<sycl::half>(size_t(K) * N, q);
        auto* d_y = sycl::malloc_device<sycl::half>(N, q);
        q.memcpy(d_a, h_a.data(), K * 2).wait();
        q.memcpy(d_w, h_w.data(), size_t(K) * N * 2).wait();
        q.memset(d_y, 0, N * 2).wait();
        auto r = run_loop(q, d_y, N, n_iters, [&](int) {
            ie::gemv_fp16(q, d_a, d_w, d_y, K, N);
        });
        report("gemv_fp16 [4096x1024]", r, n_iters);
        if (r.distinct != 1) any_nondet = true;
        sycl::free(d_a, q); sycl::free(d_w, q); sycl::free(d_y, q);
    };

    // ---- full-model in-process repeat: {kv.reset; T=1 steps} x reps -------
    // Distinguishes true in-process nondeterminism from across-process-only
    // variation (address/init-dependent numerics).  Hashes the full fp16
    // logits after every step of every repetition.
    auto model_case = [&](const std::string& gguf_path, int n_steps) {
        ie::GgufReader g;
        if (auto err = g.open(gguf_path); !err.empty()) {
            std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return;
        }
        ie::DenseConfig dcfg;
        if (auto err = ie::read_dense_config_auto(g, dcfg); !err.empty()) {
            std::fprintf(stderr, "read_dense_config: %s\n", err.c_str()); return;
        }
        ie::DenseModel dm;
        if (auto err = dm.load(alloc, g, dcfg); !err.empty()) {
            std::fprintf(stderr, "model.load: %s\n", err.c_str()); return;
        }
        const uint32_t mc = 1024;
        ie::KvCache kv;
        ie::KvCacheConfig kvcfg{};
        kvcfg.n_layers_full = dcfg.n_layers;
        kvcfg.n_kv_heads    = dcfg.n_kv_heads;
        kvcfg.head_dim      = dcfg.head_dim;
        kvcfg.max_ctx       = mc;
        if (auto err = kv.init(alloc, kvcfg); !err.empty()) {
            std::fprintf(stderr, "kv: %s\n", err.c_str()); return;
        }
        if (auto err = dm.ensure_workspace(1); !err.empty()) {
            std::fprintf(stderr, "ws: %s\n", err.c_str()); return;
        }
        if (auto err = dm.ensure_attn_partials(mc); !err.empty()) {
            std::fprintf(stderr, "partials: %s\n", err.c_str()); return;
        }
        auto* d_logits = sycl::malloc_device<sycl::half>(dcfg.vocab, q);
        auto* d_id     = sycl::malloc_device<int32_t>(1, q);
        std::vector<sycl::half> h_logits(dcfg.vocab);
        // Fixed token ids — same corpus opening as ie-perplexity's builtin
        // ("I am by birth a Genevese..."): plain ids, clipped to vocab.
        std::vector<int32_t> ids = {40, 1079, 553, 7194, 264, 9279, 586, 288};
        if (n_steps > int(ids.size())) n_steps = int(ids.size());

        std::vector<std::vector<uint64_t>> hashes;   // [rep][step]
        for (int rep = 0; rep < n_iters; ++rep) {
            kv.reset();
            std::vector<uint64_t> hs;
            for (int s = 0; s < n_steps; ++s) {
                if (dump_step >= 0) {
                    if (s == dump_step) {
                        char pfx[128];
                        std::snprintf(pfx, sizeof(pfx), "/tmp/detp_rep%02d", rep);
                        dm.set_dump_prefix(pfx);
                    } else {
                        dm.set_dump_prefix("");
                    }
                }
                q.memcpy(d_id, &ids[s], sizeof(int32_t)).wait();
                dm.forward(q, d_id, 1, uint32_t(s), kv, d_logits).wait();
                q.memcpy(h_logits.data(), d_logits,
                         size_t(dcfg.vocab) * 2).wait();
                hs.push_back(fnv1a(h_logits.data(), size_t(dcfg.vocab) * 2));
            }
            hashes.push_back(std::move(hs));
        }
        // Per-step flicker analysis vs rep 0.
        for (int s = 0; s < n_steps; ++s) {
            int n_div = 0, first = -1;
            std::vector<uint64_t> seen;
            for (int rep = 0; rep < n_iters; ++rep) {
                const uint64_t h = hashes[rep][s];
                if (rep && h != hashes[0][s]) { if (first < 0) first = rep; n_div++; }
                bool f = false;
                for (uint64_t x : seen) if (x == h) { f = true; break; }
                if (!f) seen.push_back(h);
            }
            std::printf("  model step %d (ctx=%d)        : %s  distinct=%zu  first_div_rep=%d  n_div=%d/%d\n",
                        s, s + 1,
                        seen.size() == 1 ? "BIT-STABLE" : "NON-DETERMINISTIC",
                        seen.size(), first, n_div, n_iters - 1);
            if (seen.size() != 1) any_nondet = true;
        }
        sycl::free(d_logits, q); sycl::free(d_id, q);
    };

    if (test == "model") {
        std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
        model_case(home + "/.seal/models/Qwen3-8B-Q4_K_M.gguf", 8);
        std::printf("%s\n", any_nondet ? "RESULT: NON-DETERMINISM DETECTED"
                                       : "RESULT: all bit-stable");
        return any_nondet ? 1 : 0;
    }

    if (test == "all" || test == "fa2_dense")
        fa2_case("fa2_decode dense 32/8/128", 32, 8, 128);
    if (test == "all" || test == "fa2_crown")
        fa2_case("fa2_decode crown 16/2/256", 16, 2, 256);
    if (test == "all" || test == "naive")
        naive_case("naive T=1 dense 32/8/128", 32, 8, 128);
    if (test == "all" || test == "gemv_fp16")
        gemv_case();

    std::printf("%s\n", any_nondet ? "RESULT: NON-DETERMINISM DETECTED"
                                   : "RESULT: all bit-stable");
    return any_nondet ? 1 : 0;
}
