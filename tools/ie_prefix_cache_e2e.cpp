// tools/ie_prefix_cache_e2e.cpp — PR #3 stage 2 end-to-end test.
//
// Exercises the multi-entry PrefixCache class:
//
//   Test 1: empty-cache miss returns match_len=0.
//   Test 2: single-entry insert + lookup hit, restore, decode equivalence.
//   Test 3: two endpoints along same path → longest-match wins.
//   Test 4: divergent paths share a prefix node → no false match.
//   Test 5: LRU eviction at capacity.
//   Test 6: end-to-end speedup measurement (cached vs fresh prefill).
//
// Stage-1 primitives (KvCache::copy_prefix_from, DeltaNetState::copy_from)
// already verified in ie-prefix-cache-test (PR #3 stage 1).  This tool
// validates that the multi-entry layer correctly orchestrates them and
// does not introduce additional state-divergence.

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/ops.hpp"
#include "ie/prefix_cache.hpp"
#include "ie/qwen36.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr const char* B = "\033[1m";
constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Z = "\033[0m";

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char* kSampleText =
    "I am by birth a Genevese; and my family is one of the most distinguished "
    "of that republic. My ancestors had been for many years counsellors and "
    "syndics; and my father had filled several public situations with honour "
    "and reputation. He was respected by all who knew him for his integrity "
    "and indefatigable attention to public business. He passed his younger "
    "days perpetually occupied by the affairs of his country; and it was not "
    "until the decline of life that he thought of marrying, and bestowing on "
    "the state sons who might carry his virtues and his name down to "
    "posterity. As the circumstances of his marriage illustrate his "
    "character, I cannot refrain from relating them.";

int32_t argmax_h16(const std::vector<sycl::half>& v) {
    int32_t best = 0;
    float bv = float(v[0]);
    for (size_t i = 1; i < v.size(); ++i) {
        const float f = float(v[i]);
        if (f > bv) { bv = f; best = int32_t(i); }
    }
    return best;
}

struct TestCounter {
    uint32_t passed = 0;
    uint32_t failed = 0;
    void check(bool cond, const char* name) {
        if (cond) {
            std::printf("    %s✓%s %s\n", G, Z, name);
            ++passed;
        } else {
            std::printf("    %s✗%s %s\n", R, Z, name);
            ++failed;
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    uint32_t T_prefix = 128;
    uint32_t T_decode = 16;
    uint32_t max_ctx  = 4096;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"   && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--prefix" && i + 1 < argc) T_prefix  = std::atoi(argv[++i]);
        else if (a == "--decode" && i + 1 < argc) T_decode  = std::atoi(argv[++i]);
        else if (a == "--ctx"    && i + 1 < argc) max_ctx   = std::atoi(argv[++i]);
    }

    std::printf("%sie-prefix-cache-e2e%s — PR #3 stage 2 end-to-end\n", B, Z);
    std::printf("  T_prefix=%u  T_decode=%u  ctx=%u\n\n", T_prefix, T_decode, max_ctx);

    // Load model.
    ie::GgufReader g;
    if (auto e = g.open(gguf_path); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init();      !e.empty()) { std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1; }
    auto& q = alloc.queue();

    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }

    const double t_load0 = now_ms();
    std::printf("  loading model...\n");
    ie::QwenConfig cfg;
    ie::QwenModel  model;
    if (auto e = model.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    std::printf("  loaded in %.1f s\n\n", (now_ms() - t_load0) / 1000.0);

    // Tokenize.
    auto ids = tok.encode(kSampleText, /*allow_special=*/false);
    if (tok.add_bos_token() && tok.bos_token_id() >= 0 &&
        (ids.empty() || ids.front() != tok.bos_token_id())) {
        ids.insert(ids.begin(), tok.bos_token_id());
    }
    if (ids.size() < T_prefix) { std::fprintf(stderr, "corpus too short\n"); return 1; }
    ids.resize(T_prefix);

    const uint32_t L_full = cfg.n_layers / cfg.full_attn_interval;
    const uint32_t L_lin  = cfg.n_layers - L_full;

    auto make_kv = [&](uint32_t mctx, ie::KvCache& kv) -> std::string {
        ie::KvCacheConfig c{}; c.n_layers_full = L_full; c.n_kv_heads = cfg.n_kv_heads;
        c.max_ctx = mctx; c.head_dim = cfg.head_dim; c.use_int8 = false;
        return kv.init(alloc, c);
    };
    auto make_dn = [&](ie::DeltaNetState& dn) -> std::string {
        return dn.init(alloc, ie::DeltaNetStateConfig{
            L_lin, cfg.ssm_n_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim,
            cfg.ssm_inner * 2, cfg.ssm_conv_kernel});
    };

    if (auto e = model.ensure_workspace(T_prefix); !e.empty()) { std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1; }
    if (auto e = model.ensure_attn_partials(max_ctx); !e.empty()) { std::fprintf(stderr, "ap: %s\n", e.c_str()); return 1; }

    auto* d_pf     = sycl::malloc_device<int32_t>(T_prefix, q);
    auto* d_id     = sycl::malloc_device<int32_t>(1, q);
    auto* d_logits = sycl::malloc_device<sycl::half>(cfg.vocab, q);
    std::vector<sycl::half> h_logits(cfg.vocab);
    q.memcpy(d_pf, ids.data(), T_prefix * sizeof(int32_t)).wait();

    TestCounter T;

    // -----------------------------------------------------------------
    // Test 1: empty-cache miss.
    std::printf("%sTest 1 — empty cache miss%s\n", B, Z);
    {
        ie::PrefixCache pc;
        if (auto e = pc.init(alloc, cfg, ie::PrefixCacheConfig{4, max_ctx}); !e.empty()) {
            std::fprintf(stderr, "pc.init: %s\n", e.c_str()); return 1;
        }
        auto r = pc.find_longest_match(ids);
        T.check(r.match_len == 0 && r.kv == nullptr && r.dn == nullptr,
                "empty cache returns match_len=0");
    }

    // -----------------------------------------------------------------
    // Test 2: insert + lookup hit + restore + decode equivalence.
    std::printf("\n%sTest 2 — insert/lookup/restore + decode equivalence%s\n", B, Z);
    {
        // Path A (control): fresh prefill + decode.
        ie::KvCache       kv_a;
        ie::DeltaNetState dn_a;
        if (auto e = make_kv(max_ctx, kv_a); !e.empty()) return 1;
        if (auto e = make_dn(dn_a);          !e.empty()) return 1;
        kv_a.reset(); dn_a.reset(q);
        model.forward(q, d_pf, T_prefix, 0, kv_a, dn_a, d_logits).wait();
        q.memcpy(h_logits.data(), d_logits, uint64_t(cfg.vocab) * sizeof(sycl::half)).wait();
        std::vector<int32_t> recs_a;
        recs_a.reserve(T_decode);
        int32_t next_a = argmax_h16(h_logits);
        for (uint32_t s = 0; s < T_decode; ++s) {
            const uint32_t pos = T_prefix + s;
            q.memcpy(d_id, &next_a, sizeof(int32_t)).wait();
            model.forward(q, d_id, 1, pos, kv_a, dn_a, d_logits).wait();
            q.memcpy(h_logits.data(), d_logits, uint64_t(cfg.vocab) * sizeof(sycl::half)).wait();
            const int32_t am = argmax_h16(h_logits);
            recs_a.push_back(am);
            next_a = am;
        }

        // Path B (cached): prefill → insert → fresh-state restore-from-cache → decode.
        ie::PrefixCache pc;
        pc.init(alloc, cfg, ie::PrefixCacheConfig{4, max_ctx});

        ie::KvCache       kv_pre;
        ie::DeltaNetState dn_pre;
        if (auto e = make_kv(max_ctx, kv_pre); !e.empty()) return 1;
        if (auto e = make_dn(dn_pre);          !e.empty()) return 1;
        kv_pre.reset(); dn_pre.reset(q);

        const double t_pf = now_ms();
        model.forward(q, d_pf, T_prefix, 0, kv_pre, dn_pre, d_logits).wait();
        const double pf_ms = now_ms() - t_pf;
        std::printf("    fresh prefill T=%u: %.1f ms\n", T_prefix, pf_ms);

        if (auto e = pc.insert(q, ids, kv_pre, dn_pre); !e.empty()) {
            std::fprintf(stderr, "insert: %s\n", e.c_str()); return 1;
        }
        T.check(pc.n_entries() == 1, "after insert, n_entries==1");

        auto r = pc.find_longest_match(ids);
        T.check(r.match_len == T_prefix, "lookup returns match_len==T_prefix");
        T.check(r.kv != nullptr && r.dn != nullptr, "lookup returns valid kv/dn pointers");

        // Restore into fresh main caches.
        ie::KvCache       kv_b;
        ie::DeltaNetState dn_b;
        if (auto e = make_kv(max_ctx, kv_b); !e.empty()) return 1;
        if (auto e = make_dn(dn_b);          !e.empty()) return 1;
        kv_b.reset(); dn_b.reset(q);

        const double t_r = now_ms();
        if (auto e = kv_b.copy_prefix_from(q, *r.kv, r.match_len); !e.empty()) {
            std::fprintf(stderr, "restore kv: %s\n", e.c_str()); return 1;
        }
        if (auto e = dn_b.copy_from(q, *r.dn); !e.empty()) {
            std::fprintf(stderr, "restore dn: %s\n", e.c_str()); return 1;
        }
        const double r_ms = now_ms() - t_r;
        std::printf("    cache restore: %.2f ms (%.0f× speedup vs fresh prefill)\n",
                    r_ms, pf_ms / std::max(0.01, r_ms));

        // To get the same `next` as Path A, redo the last token of prefill on a
        // throwaway state to populate d_logits.  (We freed the original.)
        {
            ie::KvCache       kv_seed;
            ie::DeltaNetState dn_seed;
            if (auto e = make_kv(T_prefix, kv_seed); !e.empty()) return 1;
            if (auto e = make_dn(dn_seed);           !e.empty()) return 1;
            kv_seed.reset(); dn_seed.reset(q);
            model.forward(q, d_pf, T_prefix, 0, kv_seed, dn_seed, d_logits).wait();
        }
        q.memcpy(h_logits.data(), d_logits, uint64_t(cfg.vocab) * sizeof(sycl::half)).wait();
        int32_t next_b = argmax_h16(h_logits);

        std::vector<int32_t> recs_b;
        recs_b.reserve(T_decode);
        for (uint32_t s = 0; s < T_decode; ++s) {
            const uint32_t pos = T_prefix + s;
            q.memcpy(d_id, &next_b, sizeof(int32_t)).wait();
            model.forward(q, d_id, 1, pos, kv_b, dn_b, d_logits).wait();
            q.memcpy(h_logits.data(), d_logits, uint64_t(cfg.vocab) * sizeof(sycl::half)).wait();
            const int32_t am = argmax_h16(h_logits);
            recs_b.push_back(am);
            next_b = am;
        }

        uint32_t matches = 0;
        for (uint32_t s = 0; s < T_decode; ++s) if (recs_a[s] == recs_b[s]) ++matches;
        T.check(matches >= T_decode - 1,
                "≥T_decode-1 argmax matches between fresh and cached paths");
        std::printf("    argmax match rate: %u/%u (%.0f%%)\n",
                    matches, T_decode, 100.0 * matches / T_decode);
    }

    // -----------------------------------------------------------------
    // Test 3: longest-match selection across two endpoints on same path.
    std::printf("\n%sTest 3 — longest-match selection%s\n", B, Z);
    {
        ie::PrefixCache pc;
        pc.init(alloc, cfg, ie::PrefixCacheConfig{4, max_ctx});

        // Need real state for two depths.  Use the prefill-once reuse trick:
        // run prefill T_prefix once with the FULL ids, then snapshot at depth N1
        // (slice the ids list) and depth N2.  But the KV/DN state captured in
        // kv_pre at the end is at depth T_prefix, not at N1 or N2.  copy_prefix_from
        // takes only the first N1 KV slots which are valid for depth=N1 (the tokens
        // [0..N1) only saw their causal context).  DN state is end-of-sequence,
        // however — it DOES NOT correspond to depth-N1 state for N1 < T_prefix.
        //
        // For Test 3 we therefore only check the TRIE selection, not state
        // semantics: we use the same (kv_pre, dn_pre) for both inserts because
        // the test's purpose is "which endpoint does find_longest_match return".
        ie::KvCache       kv_pre;
        ie::DeltaNetState dn_pre;
        if (auto e = make_kv(max_ctx, kv_pre); !e.empty()) return 1;
        if (auto e = make_dn(dn_pre);          !e.empty()) return 1;
        kv_pre.reset(); dn_pre.reset(q);
        model.forward(q, d_pf, T_prefix, 0, kv_pre, dn_pre, d_logits).wait();

        const uint32_t N1 = T_prefix / 2;
        const uint32_t N2 = T_prefix;
        std::vector<int32_t> ids_n1(ids.begin(), ids.begin() + N1);
        std::vector<int32_t> ids_n2(ids.begin(), ids.begin() + N2);

        // Insert order matters for LRU but not for correctness here.
        if (auto e = pc.insert(q, ids_n1, kv_pre, dn_pre); !e.empty()) {
            std::fprintf(stderr, "insert N1: %s\n", e.c_str()); return 1;
        }
        if (auto e = pc.insert(q, ids_n2, kv_pre, dn_pre); !e.empty()) {
            std::fprintf(stderr, "insert N2: %s\n", e.c_str()); return 1;
        }
        T.check(pc.n_entries() == 2, "two endpoints inserted");

        // Lookup with full prefix → should hit N2 (deeper).
        auto r2 = pc.find_longest_match(ids_n2);
        T.check(r2.match_len == N2, "lookup full prefix → match_len=N2 (deeper wins)");

        // Lookup with truncated prefix at N1 → should hit N1.
        auto r1 = pc.find_longest_match(ids_n1);
        T.check(r1.match_len == N1, "lookup at N1 → match_len=N1");

        // Lookup with N1 + 1 different token → should hit N1 (since N1 endpoint
        // was crossed before divergence).
        std::vector<int32_t> ids_n1_diff = ids_n1;
        ids_n1_diff.push_back(ids[N1] == 999 ? 1000 : 999);  // some token that differs from ids[N1]
        auto rd = pc.find_longest_match(ids_n1_diff);
        T.check(rd.match_len == N1, "lookup with divergence past N1 → match_len=N1");
    }

    // -----------------------------------------------------------------
    // Test 4: divergent paths share prefix, no false match.
    std::printf("\n%sTest 4 — divergent path no false match%s\n", B, Z);
    {
        ie::PrefixCache pc;
        pc.init(alloc, cfg, ie::PrefixCacheConfig{4, max_ctx});

        ie::KvCache       kv_pre;
        ie::DeltaNetState dn_pre;
        if (auto e = make_kv(max_ctx, kv_pre); !e.empty()) return 1;
        if (auto e = make_dn(dn_pre);          !e.empty()) return 1;
        kv_pre.reset(); dn_pre.reset(q);
        model.forward(q, d_pf, T_prefix, 0, kv_pre, dn_pre, d_logits).wait();

        // Insert full prefix.
        if (auto e = pc.insert(q, ids, kv_pre, dn_pre); !e.empty()) return 1;

        // Lookup with diverging tokens AT depth 0 (totally different prompt).
        std::vector<int32_t> diff = ids;
        diff[0] = (ids[0] == 1) ? 2 : 1;  // change the first token
        auto r = pc.find_longest_match(diff);
        T.check(r.match_len == 0, "divergence at depth 0 → match_len=0");

        // Lookup with diverging tokens at depth T_prefix/2 — but no endpoint
        // exists at any depth ≤ T_prefix/2, only at T_prefix → match_len=0.
        std::vector<int32_t> diff_mid = ids;
        diff_mid[T_prefix / 2] = (ids[T_prefix / 2] == 1) ? 2 : 1;
        diff_mid.resize(T_prefix);
        auto r_mid = pc.find_longest_match(diff_mid);
        T.check(r_mid.match_len == 0,
                "divergence past depth-T_prefix/2 with no endpoint there → match_len=0");
    }

    // -----------------------------------------------------------------
    // Test 5: LRU eviction at capacity.
    std::printf("\n%sTest 5 — LRU eviction%s\n", B, Z);
    {
        ie::PrefixCache pc;
        pc.init(alloc, cfg, ie::PrefixCacheConfig{2, max_ctx});  // capacity = 2

        ie::KvCache       kv_pre;
        ie::DeltaNetState dn_pre;
        if (auto e = make_kv(max_ctx, kv_pre); !e.empty()) return 1;
        if (auto e = make_dn(dn_pre);          !e.empty()) return 1;
        kv_pre.reset(); dn_pre.reset(q);
        model.forward(q, d_pf, T_prefix, 0, kv_pre, dn_pre, d_logits).wait();

        // Build three distinct token sequences.
        std::vector<int32_t> a(ids.begin(), ids.begin() + T_prefix / 4);
        std::vector<int32_t> b = a; b.push_back(ids[a.size()] + 1);  // diverge after a
        std::vector<int32_t> c = a; c.push_back(ids[a.size()] + 2);  // diverge after a

        if (auto e = pc.insert(q, a, kv_pre, dn_pre); !e.empty()) return 1;
        if (auto e = pc.insert(q, b, kv_pre, dn_pre); !e.empty()) return 1;
        T.check(pc.n_entries() == 2, "at capacity (2 entries)");

        // Touch `a` so `b` becomes the LRU.
        pc.find_longest_match(a);

        // Insert `c` → should evict `b` (LRU, since `a` was most-recent).
        if (auto e = pc.insert(q, c, kv_pre, dn_pre); !e.empty()) return 1;
        T.check(pc.n_entries() == 2, "still 2 entries after insert at capacity");

        auto rc = pc.find_longest_match(c);
        T.check(rc.match_len == c.size(), "newly-inserted c is present");

        auto rb = pc.find_longest_match(b);
        T.check(rb.match_len < b.size(), "b was evicted (not full match)");

        auto ra = pc.find_longest_match(a);
        T.check(ra.match_len == a.size(), "a (touched) survived");
    }

    // -----------------------------------------------------------------
    std::printf("\n%sSummary%s — %u passed, %u failed\n",
                B, Z, T.passed, T.failed);

    sycl::free(d_pf, q);
    sycl::free(d_id, q);
    sycl::free(d_logits, q);
    return T.failed > 0 ? 2 : 0;
}
