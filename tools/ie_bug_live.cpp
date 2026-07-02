// tools/ie_bug_live.cpp — terminal live monitor of the DeltaNet recurrence
// stochastic non-determinism on Xe2 BMG-G31.
//
// Runs two parallel forward(T=1) chains in one process (separate KvCache
// and DeltaNetState per chain, shared model + workspace, single in-order
// queue).  After every iter, hashes both DN state slabs on the host and
// compares.  As soon as the hashes diverge, prints a colored alert with
// the first divergent DN layer index and continues to show the cascade.
//
// Usage:
//   ie-bug-live [--gguf <path>] [--max-iters N] [--text <file>]
//   default:  --max-iters 1024  --gguf /home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf
//
// Exit code: 1 if divergence observed, 0 if all matched.

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/qwen36.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ANSI control sequences.  All printf calls flush stdout so the live feed
// renders immediately even when stdout is line-buffered against a pipe.
constexpr const char* C_RESET   = "\033[0m";
constexpr const char* C_GREEN   = "\033[32m";
constexpr const char* C_RED     = "\033[31;1m";
constexpr const char* C_YELLOW  = "\033[33;1m";
constexpr const char* C_BLUE    = "\033[34;1m";
constexpr const char* C_DIM     = "\033[2m";
constexpr const char* C_BOLD    = "\033[1m";
constexpr const char* CLR_LINE  = "\r\033[2K";

uint64_t hash64(const void* data, size_t bytes,
                 uint64_t seed = 0xcbf29ce484222325ULL) {
    uint64_t h = seed;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    const size_t  nw = bytes / 8;
    const uint64_t* w = reinterpret_cast<const uint64_t*>(p);
    for (size_t i = 0; i < nw; ++i) { h ^= w[i]; h *= 0x100000001b3ULL; }
    for (size_t i = nw * 8; i < bytes; ++i) {
        h ^= uint64_t(p[i]); h *= 0x100000001b3ULL;
    }
    return h;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Built-in clean English prose, ample for ≥ 1024 tokens.
constexpr const char* kSampleText =
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
    "distant friends were read aloud. These letters spoke of voyages, harvests, "
    "public debates, and the patient work by which families keep faith with one "
    "another across time and weather.\n\n"
    "I learned early that knowledge is not gathered by haste alone. A page "
    "understood clearly was worth more than a chapter passed over in restless "
    "curiosity. My teachers encouraged questions, but they also required proof, "
    "comparison, and a willingness to revise an opinion when the evidence did "
    "not support it. In that habit I found a kind of freedom: the mind became "
    "less anxious when it could distinguish a bright guess from a settled fact.\n\n"
    "When I was older, I travelled beyond the familiar streets of my childhood "
    "and saw how much of human life depends on arrangements too common to be "
    "praised. Roads, bridges, ledgers, workshops, schools, and markets seemed "
    "plain enough at first glance, yet each required memory, trust, and daily "
    "attention. A careless hand could waste what many careful hands had built. "
    "This observation made me cautious in judgment and more grateful for the "
    "uncelebrated skill that supports a peaceful city.\n\n"
    "The strongest impression of those years was not a single event, but a "
    "gradual conviction that character is measured in repeated choices. A "
    "person may speak generously in public and still fail in private duties; "
    "another may say little and yet become indispensable by doing necessary "
    "work at the proper hour. I admired the latter kind of excellence. It did "
    "not glitter, but it endured, and it left the world more orderly than it "
    "found it.\n\n"
    "Thus my education joined affection with inquiry. I loved the people who "
    "had formed me, but I also learned to examine my own certainties. Whenever "
    "a new subject drew my attention, I tried to ask what could be tested, what "
    "must be inferred, and what ought to remain undecided. This discipline did "
    "not diminish wonder. On the contrary, it made wonder steadier, because it "
    "rested on patient attention rather than surprise alone.";

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    std::string text_path;
    uint32_t    max_iters = 1024;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"      && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--max-iters" && i + 1 < argc) max_iters = std::atoi(argv[++i]);
        else if (a == "--text"      && i + 1 < argc) text_path = argv[++i];
    }

    std::printf("\n%s╔══════════════════════════════════════════════════════════════════════╗%s\n",
                C_BLUE, C_RESET);
    std::printf(  "%s║  ie-bug-live — DeltaNet recurrence non-determinism live monitor      ║%s\n",
                C_BLUE, C_RESET);
    std::printf(  "%s║  Two parallel forward(T=1) chains, per-iter DN-state hash compare    ║%s\n",
                C_BLUE, C_RESET);
    std::printf(  "%s╚══════════════════════════════════════════════════════════════════════╝%s\n",
                C_BLUE, C_RESET);
    std::printf("  GGUF      : %s\n", gguf_path.c_str());
    std::printf("  max iters : %u\n", max_iters);
    std::fflush(stdout);

    // ---- Allocator + queue ----
    ie::DeviceAllocator alloc;
    if (auto err = alloc.init(); !err.empty()) {
        std::fprintf(stderr, "alloc: %s\n", err.c_str()); return 1;
    }
    auto& q = alloc.queue();
    std::printf("  device    : %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::fflush(stdout);

    // ---- GGUF + tokenizer ----
    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return 1;
    }
    ie::Tokenizer tok;
    if (auto err = tok.load_from_gguf(g); !err.empty()) {
        std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1;
    }

    // ---- Model ----
    const double t0 = now_ms();
    std::printf("  loading model (~22 GB)...\n");
    std::fflush(stdout);
    ie::QwenConfig cfg;
    ie::QwenModel  model;
    if (auto err = model.load(alloc, g, cfg); !err.empty()) {
        std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
    }
    std::printf("  loaded in %.1f s\n", (now_ms() - t0) / 1000.0);

    // ---- Tokenize corpus ----
    std::string corpus = text_path.empty() ? kSampleText : read_text_file(text_path);
    if (corpus.empty()) {
        std::fprintf(stderr, "corpus empty (path '%s')\n", text_path.c_str()); return 1;
    }
    auto ids = tok.encode(corpus, /*allow_special=*/false);
    if (tok.add_bos_token() && tok.bos_token_id() >= 0 &&
        (ids.empty() || ids.front() != tok.bos_token_id())) {
        ids.insert(ids.begin(), tok.bos_token_id());
    }
    if (ids.size() < 2) {
        std::fprintf(stderr, "corpus tokenizes to %zu tokens — need ≥ 2.\n", ids.size());
        return 1;
    }
    if (uint32_t(ids.size()) < max_iters) {
        std::fprintf(stderr, "corpus only %zu tokens, capping max_iters to that.\n", ids.size());
        max_iters = uint32_t(ids.size());
    }
    std::printf("  corpus    : %s, %zu bytes, %u tokens (using %u)\n",
                text_path.empty() ? "<builtin>" : text_path.c_str(),
                corpus.size(), uint32_t(ids.size()), max_iters);

    // ---- Workspace + caches (TWO of each) ----
    if (auto err = model.ensure_workspace(1u); !err.empty()) {
        std::fprintf(stderr, "ws: %s\n", err.c_str()); return 1;
    }
    const uint32_t max_ctx = std::max<uint32_t>(max_iters + 4, 32);
    if (auto err = model.ensure_attn_partials(max_ctx); !err.empty()) {
        std::fprintf(stderr, "attn_partials: %s\n", err.c_str()); return 1;
    }
    const uint32_t L_full = cfg.n_layers / cfg.full_attn_interval;
    const uint32_t L_lin  = cfg.n_layers - L_full;

    auto init_kv = [&](ie::KvCache& kv) {
        ie::KvCacheConfig kvcfg{};
        kvcfg.n_layers_full = L_full;
        kvcfg.n_kv_heads    = cfg.n_kv_heads;
        kvcfg.max_ctx       = max_ctx;
        kvcfg.head_dim      = cfg.head_dim;
        kvcfg.use_int8      = false;
        return kv.init(alloc, kvcfg);
    };
    auto init_dn = [&](ie::DeltaNetState& dn) {
        return dn.init(alloc, ie::DeltaNetStateConfig{
            L_lin, cfg.ssm_n_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim,
            cfg.ssm_inner * 2, cfg.ssm_conv_kernel
        });
    };

    ie::KvCache kv_a, kv_b;
    ie::DeltaNetState dn_a, dn_b;
    if (auto e = init_kv(kv_a); !e.empty()) { std::fprintf(stderr, "kv_a: %s\n", e.c_str()); return 1; }
    if (auto e = init_kv(kv_b); !e.empty()) { std::fprintf(stderr, "kv_b: %s\n", e.c_str()); return 1; }
    if (auto e = init_dn(dn_a); !e.empty()) { std::fprintf(stderr, "dn_a: %s\n", e.c_str()); return 1; }
    if (auto e = init_dn(dn_b); !e.empty()) { std::fprintf(stderr, "dn_b: %s\n", e.c_str()); return 1; }
    kv_a.reset(); kv_b.reset();
    dn_a.reset(q); dn_b.reset(q);

    // ---- Device buffers ----
    auto* d_logits_a = sycl::malloc_device<sycl::half>(cfg.vocab, q);
    auto* d_logits_b = sycl::malloc_device<sycl::half>(cfg.vocab, q);
    auto* d_ids      = sycl::malloc_device<int32_t>(max_iters, q);
    q.memcpy(d_ids, ids.data(), max_iters * sizeof(int32_t)).wait();

    // ---- Hash scratch (host-side) ----
    const auto& dnc = dn_a.config();
    const size_t per_layer = size_t(dnc.n_v_heads) * dnc.v_head_dim *
                              dnc.k_head_dim * sizeof(float);
    const size_t total     = per_layer * dnc.n_layers_linear;
    std::vector<uint8_t> hb_a(total), hb_b(total);

    std::printf("\n  %sStarting live monitor — running %u iters...%s\n\n",
                C_YELLOW, max_iters, C_RESET);
    std::printf("  %s%s%s\n",
                C_DIM,
                "  iter   A_hash             B_hash             status     note",
                C_RESET);
    std::printf("  %s%s%s\n", C_DIM,
                "  ────── ────────────────── ────────────────── ────────── "
                "─────────────────────",
                C_RESET);
    std::fflush(stdout);

    int  first_diff_iter = -1;
    int  first_diff_dn   = -1;
    int  match_count     = 0;
    int  diff_count      = 0;
    bool diff_seen       = false;

    const auto t_start = std::chrono::steady_clock::now();

    for (uint32_t k = 0; k < max_iters; ++k) {
        // Forward chain A then chain B with the SAME token at this position.
        model.forward(q, d_ids + k, /*T=*/1, /*start_pos=*/k,
                       kv_a, dn_a, d_logits_a).wait();
        model.forward(q, d_ids + k, /*T=*/1, /*start_pos=*/k,
                       kv_b, dn_b, d_logits_b).wait();

        // D2H both DN state slabs and hash on host (fast: 60 MB per copy).
        q.memcpy(hb_a.data(), dn_a.state_ptr(), total).wait();
        q.memcpy(hb_b.data(), dn_b.state_ptr(), total).wait();
        const uint64_t ha = hash64(hb_a.data(), total);
        const uint64_t hb = hash64(hb_b.data(), total);
        const bool match = (ha == hb);

        if (match) {
            ++match_count;
            if (!diff_seen) {
                // Overwrite live status line while still matching.
                std::printf("%s  %6u  %s%016lx%s  %s%016lx%s  %s✓ match%s   %d/%u",
                            CLR_LINE,
                            k,
                            C_DIM, (unsigned long)ha, C_RESET,
                            C_DIM, (unsigned long)hb, C_RESET,
                            C_GREEN, C_RESET,
                            match_count, max_iters);
                std::fflush(stdout);
            } else {
                std::printf("\n  %s%6u%s  %s%016lx%s  %s%016lx%s  %s✓ match%s   reconverged",
                            C_BOLD, k, C_RESET,
                            C_GREEN, (unsigned long)ha, C_RESET,
                            C_GREEN, (unsigned long)hb, C_RESET,
                            C_GREEN, C_RESET);
                std::fflush(stdout);
            }
        } else {
            ++diff_count;
            if (!diff_seen) {
                diff_seen = true;
                first_diff_iter = int(k);
                // Per-DN-layer scan to find the first divergent layer.
                for (uint32_t L = 0; L < dnc.n_layers_linear; ++L) {
                    if (std::memcmp(hb_a.data() + L * per_layer,
                                     hb_b.data() + L * per_layer,
                                     per_layer) != 0) {
                        first_diff_dn = int(L);
                        break;
                    }
                }
                std::printf("%s\n", CLR_LINE);
                std::printf("  %s⚡ FIRST DIVERGENCE FIRED ⚡%s\n", C_RED, C_RESET);
                std::printf("    iter           : %s%d%s\n", C_BOLD, first_diff_iter, C_RESET);
                std::printf("    first DN layer : %s%d%s  "
                            "(model layer = %d)\n",
                            C_BOLD, first_diff_dn, C_RESET,
                            first_diff_dn + first_diff_dn / 3);
                std::printf("    A hash         : %s%016lx%s\n",
                            C_RED, (unsigned long)ha, C_RESET);
                std::printf("    B hash         : %s%016lx%s\n",
                            C_RED, (unsigned long)hb, C_RESET);
                std::printf("    %d prior iters were bytewise IDENTICAL between A and B.\n",
                            first_diff_iter);
                std::printf("    Continuing past divergence to show cascade...\n\n");
                std::printf("  %s%s%s\n", C_DIM,
                            "  iter   A_hash             B_hash             status     note",
                            C_RESET);
                std::printf("  %s%s%s",
                            C_DIM,
                            "  ────── ────────────────── ────────────────── "
                            "────────── ─────────────────────",
                            C_RESET);
                std::fflush(stdout);
            }
            std::printf("\n  %s%6u%s  %s%016lx%s  %s%016lx%s  %s✗ DIFF%s    %s",
                        C_BOLD, k, C_RESET,
                        C_RED, (unsigned long)ha, C_RESET,
                        C_RED, (unsigned long)hb, C_RESET,
                        C_RED, C_RESET,
                        first_diff_iter == int(k) ? "← FIRST" : "cascade");
            std::fflush(stdout);
        }
    }

    const auto   t_end = std::chrono::steady_clock::now();
    const double tot_s = std::chrono::duration<double>(t_end - t_start).count();

    std::printf("\n\n%s╔══════════════════════════════════════════════════════════════════════╗%s\n",
                C_BLUE, C_RESET);
    std::printf(  "%s║                              SUMMARY                                 ║%s\n",
                C_BLUE, C_RESET);
    std::printf(  "%s╚══════════════════════════════════════════════════════════════════════╝%s\n",
                C_BLUE, C_RESET);
    std::printf("  iters run         : %u\n", max_iters);
    std::printf("  match (A == B)    : %s%d%s\n",
                match_count == int(max_iters) ? C_GREEN : C_DIM, match_count, C_RESET);
    std::printf("  diff  (A != B)    : %s%d%s\n",
                diff_count > 0 ? C_RED : C_DIM, diff_count, C_RESET);
    if (first_diff_iter >= 0) {
        std::printf("  first divergence  : iter %s%d%s, DN layer %s%d%s "
                    "(model layer %d)\n",
                    C_BOLD, first_diff_iter, C_RESET,
                    C_BOLD, first_diff_dn, C_RESET,
                    first_diff_dn + first_diff_dn / 3);
        std::printf("  cascade duration  : %d iters\n",
                    int(max_iters) - first_diff_iter);
        std::printf("\n  %s>>> BUG REPRODUCED — DeltaNet recurrence is stochastically non-deterministic <<<%s\n",
                    C_RED, C_RESET);
    } else {
        std::printf("  no divergence in this run.\n");
        std::printf("\n  %s>>> BUG NOT OBSERVED in this run (re-run to confirm — bug is stochastic) <<<%s\n",
                    C_GREEN, C_RESET);
    }
    std::printf("  wall time         : %.1f s (%.1f ms / iter, both chains)\n",
                tot_s, 1000.0 * tot_s / std::max<uint32_t>(max_iters, 1));
    std::fflush(stdout);

    sycl::free(d_ids,      q);
    sycl::free(d_logits_a, q);
    sycl::free(d_logits_b, q);
    return first_diff_iter < 0 ? 0 : 1;
}
