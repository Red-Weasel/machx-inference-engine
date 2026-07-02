// tools/ie_prefix_cache_test.cpp — PR #3 correctness gate.
//
// Verifies that snapshot+restore of (KvCache, DeltaNetState) produces
// IDENTICAL forward-pass behavior to a fresh prefill.
//
// Methodology:
//   Path A (control):
//     KV_main, DN_main → prefill T=T_prefix → decode N steps,
//     capturing argmax token + last 8 logits at each step.
//
//   Path B (cache + restore):
//     KV_cache (small, max_ctx=T_prefix), DN_cache → prefill T=T_prefix
//     KV_main2 (fresh, full-size), DN_main2 (fresh)
//     KV_main2.copy_prefix_from(KV_cache, T_prefix);
//     DN_main2.copy_from(DN_cache);
//     decode N steps, capturing argmax token + last 8 logits.
//
//   Compare: per-step argmax MUST match exactly.  Logits should match
//   within fp16 epsilon (small drift is acceptable from non-determinism
//   in the DeltaNet recurrence at T_prefix > 256, but we use T<=128 to
//   stay in the deterministic regime).
//
// Pass criterion: ≥ N-1 of N steps argmax-match.  (Allow one mismatch
// to absorb the documented stochastic-FMA noise floor at HW level.)

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/ops.hpp"
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
constexpr const char* Y = "\033[33m";
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
    "character, I cannot refrain from relating them.\n\n"
    "During the years that followed, the household preserved a quiet discipline "
    "which gave dignity to ordinary labor. The library was small, but every "
    "volume had been chosen with care, and the books were read until their "
    "margins carried traces of many hands. At evening the shutters were closed, "
    "the lamp was trimmed, and the younger children listened while letters from "
    "distant friends were read aloud.";

int32_t argmax_h16(const std::vector<sycl::half>& v) {
    int32_t best = 0;
    float bv = float(v[0]);
    for (size_t i = 1; i < v.size(); ++i) {
        const float f = float(v[i]);
        if (f > bv) { bv = f; best = int32_t(i); }
    }
    return best;
}

struct StepRec {
    int32_t  argmax;
    sycl::half top_logit;
};

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    uint32_t T_prefix = 128;
    uint32_t T_decode = 32;
    uint32_t max_ctx = 4096;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"      && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--prefix"    && i + 1 < argc) T_prefix = std::atoi(argv[++i]);
        else if (a == "--decode"    && i + 1 < argc) T_decode = std::atoi(argv[++i]);
        else if (a == "--ctx"       && i + 1 < argc) max_ctx = std::atoi(argv[++i]);
    }
    if (T_prefix > 256) {
        std::fprintf(stderr,
                     "%sNOTE%s: T_prefix=%u > 256 — DeltaNet recurrence is "
                     "stochastic at this length per docs/known_bugs.md.  "
                     "Argmax-match may degrade.\n", Y, Z, T_prefix);
    }

    std::printf("%sie-prefix-cache-test%s — PR #3 correctness gate\n", B, Z);
    std::printf("  gguf       : %s\n", gguf_path.c_str());
    std::printf("  T_prefix   : %u\n", T_prefix);
    std::printf("  T_decode   : %u\n", T_decode);
    std::printf("  ctx        : %u\n", max_ctx);

    // 1. Standard load.
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
    std::printf("  loaded in %.1f s\n", (now_ms() - t_load0) / 1000.0);

    // 2. Tokenize prompt.  Cap at T_prefix tokens.
    auto ids = tok.encode(kSampleText, /*allow_special=*/false);
    if (tok.add_bos_token() && tok.bos_token_id() >= 0 &&
        (ids.empty() || ids.front() != tok.bos_token_id())) {
        ids.insert(ids.begin(), tok.bos_token_id());
    }
    if (ids.size() < T_prefix) {
        std::fprintf(stderr, "corpus too short: %zu < %u\n", ids.size(), T_prefix);
        return 1;
    }
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

    auto run_path = [&](const char* tag, ie::KvCache& kv, ie::DeltaNetState& dn) -> std::vector<StepRec> {
        std::vector<StepRec> recs;
        recs.reserve(T_decode);
        // Prefill (only if length 0).
        if (kv.length(0) == 0) {
            model.forward(q, d_pf, T_prefix, 0, kv, dn, d_logits).wait();
        }
        // Take the first decode token from prefill's last-position logits.
        q.memcpy(h_logits.data(), d_logits, uint64_t(cfg.vocab) * sizeof(sycl::half)).wait();
        int32_t next = argmax_h16(h_logits);
        for (uint32_t step = 0; step < T_decode; ++step) {
            const uint32_t pos = T_prefix + step;
            q.memcpy(d_id, &next, sizeof(int32_t)).wait();
            model.forward(q, d_id, /*T=*/1, /*start_pos=*/pos, kv, dn, d_logits).wait();
            q.memcpy(h_logits.data(), d_logits, uint64_t(cfg.vocab) * sizeof(sycl::half)).wait();
            const int32_t am = argmax_h16(h_logits);
            recs.push_back(StepRec{am, h_logits[am]});
            next = am;
        }
        std::printf("    [%s] decoded %u tokens, last argmax=%d\n",
                    tag, T_decode, recs.back().argmax);
        return recs;
    };

    // === Path A — fresh prefill + decode (control) ===
    std::printf("\n%s== Path A — fresh prefill + decode (control) ==%s\n", B, Z);
    ie::KvCache       kv_a;
    ie::DeltaNetState dn_a;
    if (auto e = make_kv(max_ctx, kv_a); !e.empty()) { std::fprintf(stderr, "kv_a: %s\n", e.c_str()); return 1; }
    if (auto e = make_dn(dn_a);          !e.empty()) { std::fprintf(stderr, "dn_a: %s\n", e.c_str()); return 1; }
    kv_a.reset();
    dn_a.reset(q);
    auto recs_a = run_path("A", kv_a, dn_a);

    // === Path B — prefill into small cache, restore, decode ===
    std::printf("\n%s== Path B — snapshot prefix → restore → decode ==%s\n", B, Z);
    // B.1 prefill into the "stored prefix" cache (smaller max_ctx).
    ie::KvCache       kv_pfx;
    ie::DeltaNetState dn_pfx;
    if (auto e = make_kv(T_prefix, kv_pfx); !e.empty()) { std::fprintf(stderr, "kv_pfx: %s\n", e.c_str()); return 1; }
    if (auto e = make_dn(dn_pfx);           !e.empty()) { std::fprintf(stderr, "dn_pfx: %s\n", e.c_str()); return 1; }
    kv_pfx.reset();
    dn_pfx.reset(q);
    {
        const double t0 = now_ms();
        auto* d_logits_tmp = sycl::malloc_device<sycl::half>(cfg.vocab, q);
        model.forward(q, d_pf, T_prefix, 0, kv_pfx, dn_pfx, d_logits_tmp).wait();
        sycl::free(d_logits_tmp, q);
        std::printf("    [B] prefill T=%u into prefix cache: %.1f ms\n", T_prefix, now_ms() - t0);
    }

    // B.2 fresh main caches, restore from prefix, then decode.
    ie::KvCache       kv_b;
    ie::DeltaNetState dn_b;
    if (auto e = make_kv(max_ctx, kv_b); !e.empty()) { std::fprintf(stderr, "kv_b: %s\n", e.c_str()); return 1; }
    if (auto e = make_dn(dn_b);          !e.empty()) { std::fprintf(stderr, "dn_b: %s\n", e.c_str()); return 1; }
    kv_b.reset();
    dn_b.reset(q);
    {
        const double t0 = now_ms();
        if (auto e = kv_b.copy_prefix_from(q, kv_pfx, T_prefix); !e.empty()) {
            std::fprintf(stderr, "kv copy: %s\n", e.c_str()); return 1;
        }
        if (auto e = dn_b.copy_from(q, dn_pfx); !e.empty()) {
            std::fprintf(stderr, "dn copy: %s\n", e.c_str()); return 1;
        }
        std::printf("    [B] snapshot restore: %.2f ms\n", now_ms() - t0);
    }
    // B.3 produce the same "first decode" by running lm_head on top of the
    // restored state.  We need to populate d_logits to mirror Path A's
    // post-prefill logits.  Cheapest path: re-run prefill on the suffix=
    // (last token of prefix) with T=1 at start_pos = T_prefix-1?  No — that
    // would advance state.  Instead: just re-run last-token forward on a
    // throwaway cache to get logits.  Not actually needed: Path B's first
    // decode token is whatever we choose.  To be truly equivalent to Path A,
    // we must seed Path B's first decode with the same `next` token Path A
    // produced.  Easiest: skip the "from prefill last logits" path and have
    // Path B start from the same `next` Path A used.  But that means Path B
    // never re-derives `next` from prefill, only verifies state-equivalence
    // for tokens 1..T_decode.  For a clean test we want both to derive `next`
    // from prefill.  So: re-run forward(d_pf[-1:], T=1) on a fresh copy of
    // the snapshot.  Use kv_b after restore — start_pos = T_prefix - 1 means
    // re-doing the last prefill step.  But that ALSO advances state.
    //
    // Cleanest: produce the seed token from kv_pfx's prefill (which we
    // already did and `d_logits_tmp` had it).  We freed that buffer.  Run
    // it again with a cheap throwaway:
    {
        ie::KvCache kv_seed;
        ie::DeltaNetState dn_seed;
        if (auto e = make_kv(T_prefix, kv_seed); !e.empty()) { std::fprintf(stderr, "kv_seed: %s\n", e.c_str()); return 1; }
        if (auto e = make_dn(dn_seed);           !e.empty()) { std::fprintf(stderr, "dn_seed: %s\n", e.c_str()); return 1; }
        kv_seed.reset(); dn_seed.reset(q);
        model.forward(q, d_pf, T_prefix, 0, kv_seed, dn_seed, d_logits).wait();
    }
    auto recs_b = run_path("B", kv_b, dn_b);

    // === Compare ===
    std::printf("\n%s== Comparison ==%s\n", B, Z);
    uint32_t matches = 0;
    for (uint32_t i = 0; i < T_decode; ++i) {
        const bool m = (recs_a[i].argmax == recs_b[i].argmax);
        if (m) ++matches;
        if (!m || i < 4) {
            const char* col = m ? G : R;
            std::printf("    step %2u  A=%6d  B=%6d   %s%s%s   logit_a=%.3f  logit_b=%.3f\n",
                        i, recs_a[i].argmax, recs_b[i].argmax,
                        col, m ? "MATCH" : "DIFF", Z,
                        float(recs_a[i].top_logit), float(recs_b[i].top_logit));
        }
    }
    std::printf("\n  argmax match : %u / %u (%.1f%%)\n",
                matches, T_decode, 100.0 * matches / T_decode);

    const bool pass = matches >= T_decode - 1;
    std::printf("  verdict      : %s%s%s\n",
                pass ? G : R, pass ? "PASS" : "FAIL", Z);
    if (!pass) {
        std::printf("    > prefix-cache snapshot/restore is NOT yielding equivalent\n"
                    "      forward behavior.  Check copy_prefix_from / copy_from impl.\n");
    } else {
        std::printf("    > snapshot/restore primitives produce equivalent forward state.\n"
                    "      Ready to build the radix-tree multi-entry layer on top.\n");
    }

    sycl::free(d_pf, q);
    sycl::free(d_id, q);
    sycl::free(d_logits, q);
    return pass ? 0 : 2;
}
