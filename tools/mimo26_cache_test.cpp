// tools/mimo26_cache_test.cpp -- ie-mimo26-cache-test: the gate of MiMo-V2.6's host slots (P7, fix-list #70) and the drafter's
// fault containment (#64), docs/mimo26/P7_FIX64_FIX70.md. Every slot arm compares a run that went THROUGH a host slot (a
// conversation kept in a slot, the cards dirtied by an unrelated prompt, the slot restored) against a control that runs the
// same chunks with no slot involved: the last logits (fp32 bytes, FNV), 16 greedy tokens and -- with the DFlash drafter --
// the drafter's bookkeeping and its drafts and their probabilities before each of those steps must be IDENTICAL. Requests go
// through the engine's own prefix step (Mimo26PrefixCache::prepare, then the cut mimo26_run_ids makes). The CPU expert leg
// is off (IE_DS41_CPU_MISS=0: its split follows the stream-slot history, so its rows are not bit-reproducible run to run).
//   usage: ie-mimo26-cache-test <model_dir> <text_file> [--ctx N] [--no-dflash] [--big N] [--only87] [--ttft87 N]
// <text_file>: plain text of at least 32,000 tokens (e.g. wikitext-2 test). --big N also times keeping and restoring an
// N-token state (needs N + 2,200 tokens of text and --ctx above N + 100). Arm 7 (#87): a reply longer than the SWA ring
// is discarded and the next prompt diverges at (or just before) the old prompt's end. --only87 runs arm 7 alone;
// --ttft87 N adds its time to first token at an N-token prompt, against a fresh prefill of the same prompt (needs
// N + 3,300 tokens of text and --ctx above N + 3,300).
#include "ie/expert_stream.hpp"      // ds4_expert_priority_read_layers
#include "ie/mimo26.hpp"
#include "ie/mimo26_dflash.hpp"
#include "ie/mimo26_forward.hpp"
#include "ie/mimo26_prefix_cache.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? " ok " : "FAIL", what.c_str(), detail.empty() ? "" : "  ", detail.c_str());
    std::fflush(stdout);
    if (!ok) ++g_fail;
}
uint64_t fnv(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    for (float x : v) { uint32_t u; std::memcpy(&u, &x, 4); h = (h ^ u) * 1099511628211ull; }
    return h;
}
std::vector<int32_t> slice(const std::vector<int32_t>& v, size_t a, size_t b) {
    return std::vector<int32_t>(v.begin() + std::ptrdiff_t(a), v.begin() + std::ptrdiff_t(b));
}
std::vector<int32_t> cat(std::vector<int32_t> a, const std::vector<int32_t>& b) { a.insert(a.end(), b.begin(), b.end()); return a; }
float from_bits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// A greedy continuation: the logits it started from, its tokens, and (drafter on) the drafter's bookkeeping at its start and
// its drafts + probabilities before every step (draft() writes no state, so recording them perturbs nothing).
struct Trace {
    uint64_t logits = 0;
    std::vector<int32_t> tokens, drafts;
    std::vector<float> probs;
    uint32_t ctx_lo = 0, ctx_end = 0, ctx_hi = 0;
};
bool same(const Trace& a, const Trace& b) {
    return a.logits == b.logits && a.tokens == b.tokens && a.drafts == b.drafts && a.probs.size() == b.probs.size() &&
           (a.probs.empty() || std::memcmp(a.probs.data(), b.probs.data(), a.probs.size() * sizeof(float)) == 0) &&
           a.ctx_lo == b.ctx_lo && a.ctx_end == b.ctx_end && a.ctx_hi == b.ctx_hi;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: ie-mimo26-cache-test <model_dir> <text_file> [--ctx N] [--no-dflash] [--big N]\n"); return 2; }
    setenv("IE_DS41_CPU_MISS", "0", 1);
    const std::string dir = argv[1], text_file = argv[2];
    uint32_t ctx = 40960, big = 0, ttft87 = 0; bool use_df = true, only87 = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--ctx") ctx = uint32_t(std::atol(val().c_str()));
        else if (a == "--big") big = uint32_t(std::atol(val().c_str()));
        else if (a == "--no-dflash") use_df = false;
        else if (a == "--only87") only87 = true;
        else if (a == "--ttft87") ttft87 = uint32_t(std::atol(val().c_str()));
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    ie::Mimo26Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    std::ifstream tf(text_file, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
    const std::vector<int32_t> all = tok.encode(text, false);
    if (all.size() < 32000 || (big && all.size() < size_t(big) + 2200)) { std::fprintf(stderr, "need >= %u tokens of text, got %zu\n", std::max(32000u, big + 2200), all.size()); return 2; }
    if (ttft87 && (all.size() < size_t(ttft87) + 3300 || ctx < ttft87 + 3300)) { std::fprintf(stderr, "--ttft87 %u needs %u tokens of text and --ctx above that\n", ttft87, ttft87 + 3300); return 2; }
    if (ctx < 12000 || (big && ctx < big + 100)) { std::fprintf(stderr, "--ctx %u is too small\n", ctx); return 2; }
    size_t z0 = 12000;
    while (z0 < 14000 && all[z0] == all[0]) ++z0;                    // Z must not share even a first token with S
    const std::vector<int32_t> S = slice(all, 0, 12000), Z = slice(all, z0, z0 + 18000);   // two unrelated sequences
    const std::vector<int32_t> S7000 = slice(S, 0, 7000), S8192 = slice(S, 0, 8192), S9000 = slice(S, 0, 9000), Z6000 = slice(Z, 0, 6000);

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) { queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); qs.push_back(queues.back().get()); }
    // the engine's configuration (mimo26_load): 2,048-row chunks, the auto static tier, the chat ranking when installed
    ie::Mimo26Options opt; opt.max_ctx = ctx; opt.max_tokens = std::min<uint32_t>(2048, ctx); opt.n_static = 0;
    if (const std::string rp = dir + "/ie_ranking_mimo26_chat.txt"; std::ifstream(rp).good())
        if (auto e = ie::ds4_expert_priority_read_layers(rp, m.config().n_routed_experts, m.config().n_layers, opt.ranking); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }
    ie::Mimo26DFlash df_obj;                     // on the last card, before the forward (as mimo26_load)
    ie::Mimo26DFlash* df = nullptr;
    if (use_df && std::ifstream(dir + "/dflash/config.json").good()) {
        if (auto e = df_obj.load(dir); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        if (auto e = df_obj.init(*qs.back(), m.embed.w->data, m.config().vocab_size, df_obj.config().window); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        df = &df_obj;
    }
    ie::Mimo26Forward fwd;
    if (auto e = fwd.init(qs, m, opt); !e.empty()) { std::fprintf(stderr, "init: %s\n", e.c_str()); return 1; }
    if (df) {
        df->set_head(fwd.head_weights());
        if (auto e = fwd.set_feature_layers(df->config().target_layers, df->config().window); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    }
    std::printf("mimo26: %zu card(s), ctx %u, SWA ring %u, drafter %s\n", qs.size(), ctx, fwd.ring(), df ? "ON" : "off");
    const uint32_t C = opt.max_tokens;
    ie::Mimo26PrefixCache cache;
    std::vector<int32_t> live;   // the ids the cards hold, as Mimo26Bundle::live

    const ie::Mimo26PrefixCache::Options def;   // the serving defaults
    auto fresh = [&](const ie::Mimo26PrefixCache::Options& o) {
        if (auto e = cache.init(fwd, df, o); !e.empty()) { std::printf("cache init: %s\n", e.c_str()); std::exit(1); }
        fwd.reset(); if (df) df->reset(); live.clear();
    };
    auto ctx_add = [&](uint32_t n, uint32_t p0, uint32_t stride) -> bool {   // mimo26_run_ids' add_ctx
        if (!df) return true;
        if (auto e = df->add_context(fwd.features(), n, p0, stride); !e.empty()) { std::printf("dflash context at %u: %s\n", p0, e.c_str()); return false; }
        return true;
    };
    // positions [from, to) of ids in chunks of C (mimo26_run_ids' prefill loop), no cache involved
    auto prefill = [&](const std::vector<int32_t>& ids, uint32_t from, uint32_t to, std::vector<float>& lg) -> bool {
        for (uint32_t off = from; off < to; off += C) {
            const uint32_t n = std::min(C, to - off);
            if (auto e = fwd.forward(ids.data() + off, n, off, lg, false); !e.empty()) { std::printf("forward at %u: %s\n", off, e.c_str()); return false; }
            live.insert(live.end(), ids.begin() + off, ids.begin() + off + n);
            if (!ctx_add(fwd.feat_rows(), off + n - fwd.feat_rows(), fwd.feat_rows())) return false;
        }
        return true;
    };
    // the prefix step of mimo26_run_ids: prepare, then the cut to the served prefix
    auto prefix = [&](const std::vector<int32_t>& ids, uint32_t& L, std::string& src) -> bool {
        if (auto e = cache.prepare(ids, live, L, src); !e.empty()) { std::printf("prepare: %s\n", e.c_str()); return false; }
        fwd.rewind(L); live.resize(L);
        if (L == 0) fwd.reset();
        if (df) { if (L == 0) df->reset(); else df->rewind(L); }
        return true;
    };
    // one whole request's prompt, as mimo26_run_ids runs it
    auto request = [&](const std::vector<int32_t>& ids, std::vector<float>& lg, uint32_t& L, std::string& src) -> bool {
        if (!prefix(ids, L, src) || !prefill(ids, L, uint32_t(ids.size()), lg)) return false;
        if (auto e = cache.prompt_done(uint32_t(ids.size())); !e.empty()) { std::printf("prompt_done: %s\n", e.c_str()); return false; }
        return true;
    };
    // n greedy one-row steps from lg (left holding the last step's logits)
    auto greedy = [&](std::vector<float>& lg, uint32_t n, Trace& t) -> bool {
        t = Trace{}; t.logits = fnv(lg);
        if (df) { t.ctx_lo = df->ctx_lo(); t.ctx_end = df->ctx_end(); t.ctx_hi = df->ctx_hi(); }
        for (uint32_t i = 0; i < n; ++i) {
            const int32_t id = int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin());
            t.tokens.push_back(id);
            if (df) {
                std::vector<int32_t> d;
                if (auto e = df->draft(id, df->config().block - 1, d, 0.f); !e.empty()) { std::printf("draft at %u: %s\n", fwd.n_pos(), e.c_str()); return false; }
                t.drafts.insert(t.drafts.end(), d.begin(), d.end());
                t.probs.insert(t.probs.end(), df->last_probs().begin(), df->last_probs().end());
            }
            const uint32_t pos = fwd.n_pos();
            if (auto e = fwd.forward(&id, 1, pos, lg, false); !e.empty()) { std::printf("decode at %u: %s\n", pos, e.c_str()); return false; }
            live.push_back(id);
            if (!ctx_add(1, pos, 1)) return false;
        }
        return true;
    };
    // a verify pass that keeps only its first row (mimo26_run_ids on a fully rejected draft): rows [id, 7 x junk] written at
    // n_pos, the cut back to n_pos + 1 -- written_end() is then 7 past n_pos(), the case of P3b gate finding 3
    auto verify_reject = [&](int32_t id) -> bool {
        std::vector<int32_t> rows = {id};
        rows.insert(rows.end(), Z.end() - 7, Z.end());
        const uint32_t pos = fwd.n_pos(); std::vector<float> vlg;
        if (auto e = fwd.forward(rows.data(), 8, pos, vlg, true); !e.empty()) { std::printf("verify at %u: %s\n", pos, e.c_str()); return false; }
        fwd.rewind(pos + 1); live.push_back(id);
        return ctx_add(1, pos, 8);
    };
    auto argmax = [](const std::vector<float>& lg) { return int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin()); };
    auto mib = [](uint64_t b) { return std::to_string(b >> 20) + " MiB"; };
    auto check_trace = [&](const Trace& t, const Trace& c, const std::string& arm) {
        check(t.logits == c.logits, arm + ": last logits identical to its control");
        check(t.tokens == c.tokens, arm + ": 16 greedy tokens identical to its control");
        if (df) check(same(t, c), arm + ": the drafter's bookkeeping, drafts and probabilities at those 16 steps identical",
                      "context [" + std::to_string(t.ctx_lo) + ", " + std::to_string(t.ctx_end) + ") written to " + std::to_string(t.ctx_hi));
    };
    std::vector<float> lg;
    uint32_t L = 0; std::string src;

    // ---- control 1: S[0, 8192) then S[8192, 9000), straight through ----------------------------------------------------------
    Trace c1, tmp;
    if (!only87) {
    fresh(def);
    if (!prefill(S9000, 0, 8192, lg) || !prefill(S9000, 8192, 9000, lg) || !greedy(lg, 16, c1)) return 1;
    std::printf("control 1: logits fnv %016llx, greedy %d %d %d %d ...\n", (unsigned long long)c1.logits, c1.tokens[0], c1.tokens[1], c1.tokens[2], c1.tokens[3]);

    // ---- arm 1: S kept in a host slot when an unrelated prompt Z arrives, then restored for its continuation --------------------
    {
        fresh(def);
        if (!request(S8192, lg, L, src)) return 1;
        const uint32_t lo = df ? df->ctx_lo() : 0, end = df ? df->ctx_end() : 0, hi = df ? df->ctx_hi() : 0;
        if (!request(Z6000, lg, L, src)) return 1;
        const auto s1 = cache.stats();
        check(L == 0 && s1.slots == 1, "arm 1: an unrelated prompt serves nothing and keeps S (8,192 tokens) in a host slot",
              mib(s1.bytes) + " in " + std::to_string(int(s1.last_save_ms)) + " ms");
        if (!prefix(S9000, L, src)) return 1;
        const auto s2 = cache.stats();
        check(L == 8192 && src.rfind("slot ", 0) == 0, "arm 1: S comes back from its slot at 8,192",
              "from " + src + " in " + std::to_string(int(s2.last_load_ms)) + " ms; Z kept too: " + std::to_string(s2.slots) + " slot(s)");
        if (df) check(df->ctx_lo() == lo && df->ctx_end() == end && df->ctx_hi() == hi, "arm 1: the drafter's context bookkeeping comes back with it",
                      "[" + std::to_string(lo) + ", " + std::to_string(end) + ") written to " + std::to_string(hi));
        if (!prefill(S9000, L, 9000, lg)) return 1;
        if (auto e = cache.prompt_done(9000); !e.empty()) { std::printf("prompt_done: %s\n", e.c_str()); return 1; }
        Trace t;
        if (!greedy(lg, 16, t)) return 1;
        check_trace(t, c1, "arm 1");
        check(cache.stats().slots == 1, "arm 1: S's slot is dropped once its continuation ran (Z's stays)");
    }

    // ---- control 2 / 2b: rejected verify rows past the end, then a full and a partial continuation --------------------------
    // the state: S[0, 8192), 40 greedy steps, a verify that keeps only its first row -> 8,233 positions written to 8,240
    Trace c2, c2b;
    std::vector<int32_t> p2, p2b, held;
    auto build2 = [&](bool through_cache) -> bool {
        if (through_cache) { if (!request(S8192, lg, L, src)) return false; }
        else if (!prefill(S8192, 0, 8192, lg)) return false;
        if (!greedy(lg, 40, tmp)) return false;
        return verify_reject(argmax(lg));
    };
    fresh(def);
    if (!build2(false)) return 1;
    held = live;
    p2 = cat(held, slice(S, 9000, 9500));                          // continues the whole state
    p2b = cat(slice(held, 0, 8212), slice(S, 9000, 9500));         // diverges inside the 40 generated tokens
    const uint32_t n2 = uint32_t(held.size());
    uint32_t L2b = 8212;                                           // (a generated token may equal the text's next one)
    while (L2b < n2 && L2b + 1 < p2b.size() && held[L2b] == p2b[L2b]) ++L2b;
    std::printf("state 2: %u positions written to %u (ring %u); the partial prompt shares %u\n", fwd.n_pos(), fwd.written_end(), fwd.ring(), L2b);
    if (!prefill(p2, n2, uint32_t(p2.size()), lg) || !greedy(lg, 16, c2)) return 1;
    fresh(def);
    if (!build2(false) || live != held) { check(false, "control 2b reproduces state 2"); return 1; }
    fwd.rewind(L2b); live.resize(L2b); if (df) df->rewind(L2b);
    if (!prefill(p2b, L2b, uint32_t(p2b.size()), lg) || !greedy(lg, 16, c2b)) return 1;

    // ---- arm 2 / 2b: the same state through a host slot ---------------------------------------------------------------------
    for (const bool partial : {false, true}) {
        const std::string arm = partial ? "arm 2b" : "arm 2";
        fresh(def);
        if (!build2(true) || live != held) { check(false, arm + ": reproduces state 2"); return 1; }
        if (!request(Z6000, lg, L, src)) return 1;
        const auto& p = partial ? p2b : p2;
        const uint32_t want = partial ? L2b : n2;
        if (!request(p, lg, L, src)) return 1;
        check(L == want && src.rfind("slot ", 0) == 0, arm + (partial ? ": a prompt diverging inside the slot's generated tail is served to the divergence from it"
                                                                      : ": the slot (rejected rows past its end) serves the whole state"),
              "served " + std::to_string(L) + " from " + src);
        Trace t;
        if (!greedy(lg, 16, t)) return 1;
        check_trace(t, partial ? c2b : c2, arm);
    }

    // ---- control 4 / arm 4: the #70 pattern -- a continued conversation, six one-shot side requests, the conversation again ----
    // a 3 GiB budget holds the conversation and about four of the side states: one-shot states are evicted, the continued one
    // never is (pure LRU would have evicted it at the first overflow)
    Trace c4;
    fresh(def);
    if (!prefill(S9000, 0, 7000, lg) || !prefill(S9000, 7000, 8192, lg) || !prefill(S9000, 8192, 9000, lg) || !greedy(lg, 16, c4)) return 1;
    {
        ie::Mimo26PrefixCache::Options o; o.budget = 3ull << 30; o.keep_free = 0;
        fresh(o);
        if (!request(S7000, lg, L, src) || !request(S8192, lg, L, src)) return 1;
        check(L == 7000 && src == "live", "arm 4: the second turn continues the live conversation", "served " + std::to_string(L) + " from " + src);
        std::string counts;
        for (uint32_t k = 0; k < 6; ++k) {
            if (!request(slice(Z, 2000 * k, 2000 * k + 4000), lg, L, src)) return 1;
            counts += std::to_string(cache.stats().slots) + (k < 5 ? " " : "");
        }
        if (!request(S9000, lg, L, src)) return 1;
        const auto st = cache.stats();
        check(L == 8192 && src.rfind("slot ", 0) == 0, "arm 4: after six one-shot requests the conversation comes back from its slot",
              "served " + std::to_string(L) + " from " + src + "; slots after each side request: " + counts);
        check(st.slots < 6, "arm 4: the budget forced evictions (7 states kept in all)", std::to_string(st.slots) + " slot(s), " + mib(st.bytes));
        Trace t;
        if (!greedy(lg, 16, t)) return 1;
        check_trace(t, c4, "arm 4");
    }

    // ---- arm 5: the MemAvailable floor refuses every slot: nothing is kept, the prompt is re-read, the result is unchanged -----
    {
        ie::Mimo26PrefixCache::Options o; o.keep_free = 1ull << 60;
        fresh(o);
        if (!request(S8192, lg, L, src) || !request(Z6000, lg, L, src)) return 1;
        check(cache.stats().slots == 0, "arm 5: a floor MemAvailable cannot meet keeps nothing (logged, not an error)");
        if (!request(S9000, lg, L, src)) return 1;
        check(L == 0 && src == "none", "arm 5: the conversation is re-read from the start", "served " + std::to_string(L) + " from " + src);
        Trace t;
        if (!greedy(lg, 16, t)) return 1;
        check(t.logits == c1.logits && t.tokens == c1.tokens, "arm 5: last logits and 16 greedy tokens identical to control 1 (the same chunks)");
    }

    // ---- arm 6 (#64): a drafter reset mid-conversation re-syncs on the next prompt; a feature fp16 cannot hold is refused -------
    if (df) {
        fresh(def);
        if (!request(S8192, lg, L, src)) return 1;
        df->reset();                                                    // what mimo26_run_ids does on a drafter fault
        if (!request(S9000, lg, L, src)) return 1;
        check(L == 8192 && df->ctx_lo() == 8192 && df->ctx_end() == fwd.n_pos(),
              "arm 6: after a drafter reset the next prompt's prefill restarts its context (the gap rule) at the target's position",
              "context [" + std::to_string(df->ctx_lo()) + ", " + std::to_string(df->ctx_end()) + "), target at " + std::to_string(fwd.n_pos()));
        Trace t;
        if (!greedy(lg, 16, t)) return 1;
        check(t.logits == c1.logits && t.tokens == c1.tokens, "arm 6: the target is untouched by the drafter reset (control 1's logits and tokens)");
        const size_t F = df->config().target_layers.size() * size_t(df->config().hidden);
        for (const auto& [bits, name] : {std::pair<uint32_t, const char*>{0x7F800000u, "+inf"}, {0x7FC00000u, "NaN"}, {0x47888800u /* 69,904 */, "69,904"}}) {
            std::vector<float> feat(F, 0.5f);
            feat[3 * size_t(df->config().hidden) + 77] = from_bits(bits);
            const uint32_t lo = df->ctx_lo(), end = df->ctx_end(), hi = df->ctx_hi();
            const std::string e = df->add_context(feat.data(), 1, end, 1);
            check(e.find("outside fp16's range") != std::string::npos && df->ctx_lo() == lo && df->ctx_end() == end && df->ctx_hi() == hi,
                  std::string("arm 6: a context feature of ") + name + " is refused before the drafter changes", e);
        }
    }
    }   // !only87

    // ---- control 7 / arm 7 (#87): a reply longer than ring - window is discarded ------------------------------------------------
    // The prompt P = S[0, 8192) runs, then a "reply" of 8 greedy steps, 2,600 more rows (chunked forwards: the rings see what a
    // long decode leaves) and a rejected verify, so the rings no longer hold the window before P. The next prompt keeps P up to
    // a divergence at its end (a discarded reply: 8,192), just before it (the generation prompt dropped: 8,188), or repeats it
    // (a regenerate: served to 8,191), then another continuation. Before #87 the live state served none of it and the whole
    // prompt was re-read; the control is the state at the end of P (the same chunks), cut to the divergence, then the same
    // continuation.
    {
        const std::vector<int32_t> cont = slice(S, 9000, 9500);
        for (const uint32_t variant : {0u, 1u, 2u}) {
            const std::string arm = variant == 0 ? "arm 7" : variant == 1 ? "arm 7b" : "arm 7c";
            const std::vector<int32_t> p7 = variant == 2 ? S8192 : cat(slice(S8192, 0, variant == 0 ? 8192 : 8188), cont);
            uint32_t want = variant == 2 ? 8191 : variant == 0 ? 8192 : 8188;
            while (variant != 2 && want < 8192 && want + 1 < p7.size() && S8192[want] == p7[want]) ++want;   // (a token may repeat)
            Trace c7;
            fresh(def);
            if (!prefill(S8192, 0, 8192, lg)) return 1;
            fwd.rewind(want); live.resize(want); if (df) df->rewind(want);
            if (!prefill(p7, want, uint32_t(p7.size()), lg) || !greedy(lg, 16, c7)) return 1;
            fresh(def);
            if (!request(S8192, lg, L, src)) return 1;
            const auto sn = cache.stats();
            std::printf("%s: the prompt-end snapshot: %s (%llu bytes) in %.1f ms\n", arm.c_str(), mib(sn.snap_bytes).c_str(), (unsigned long long)sn.snap_bytes, sn.last_snap_ms);
            if (!greedy(lg, 8, tmp)) return 1;
            const std::vector<int32_t> reply = cat(live, slice(Z, 0, 2600));
            if (!prefill(reply, uint32_t(live.size()), uint32_t(reply.size()), lg) || !verify_reject(argmax(lg))) return 1;
            std::printf("%s: the reply left %u positions written to %u (ring %u): the live state %s the prompt's end\n", arm.c_str(), fwd.n_pos(),
                        fwd.written_end(), fwd.ring(), fwd.written_end() - 8192 + 128 > fwd.ring() ? "no longer serves" : "still serves");
            if (!request(p7, lg, L, src)) return 1;
            check(L == want && src != "none", arm + ": the discarded reply's prompt is served to the divergence, not re-read",
                  "served " + std::to_string(L) + " of " + std::to_string(p7.size()) + " from " + src + " (want " + std::to_string(want) + ")");
            check(cache.stats().slots == 0, arm + ": the discarded reply's state is not kept as a host slot",
                  std::to_string(cache.stats().slots) + " slot(s), " + mib(cache.stats().bytes) + "; snapshot restored in " +
                  std::to_string(cache.stats().last_snap_load_ms) + " ms");
            Trace t;
            if (!greedy(lg, 16, t)) return 1;
            check_trace(t, c7, arm);
        }
    }
    // --ttft87 N: the #87 request's time to its first logits at an N-token prompt (the prefix step, then the continuation's
    // prefill), against a fresh prefill of the same request (what it cost before #87, less keeping the reply's state)
    if (ttft87) {
        const std::vector<int32_t> PN = slice(all, 0, ttft87), cont = slice(all, all.size() - 600, all.size() - 100);
        fresh(def);
        if (!request(PN, lg, L, src) || !greedy(lg, 8, tmp)) return 1;
        const std::vector<int32_t> reply = cat(live, slice(all, size_t(ttft87) + 8, size_t(ttft87) + 2708));
        if (!prefill(reply, uint32_t(live.size()), uint32_t(reply.size()), lg) || !verify_reject(argmax(lg))) return 1;
        const std::vector<int32_t> req = cat(PN, cont);
        auto t0 = std::chrono::steady_clock::now();
        if (!request(req, lg, L, src)) return 1;
        const double ms_87 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const uint32_t L87 = L; const std::string src87 = src;
        fresh(def);
        t0 = std::chrono::steady_clock::now();
        if (!request(req, lg, L, src)) return 1;
        const double ms_fresh = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("ttft87: %u-token prompt + 500 after a 2,708-token reply: %.0f ms (served %u from %s); a fresh prefill of the same %zu tokens: %.0f ms\n",
                    ttft87, ms_87, L87, src87.c_str(), req.size(), ms_fresh);
        const auto sn = cache.stats();
        std::printf("ttft87: the fresh prompt's snapshot: %s in %.1f ms\n", mib(sn.snap_bytes).c_str(), sn.last_snap_ms);
    }

    // ---- --big N: the time and size of keeping and restoring an N-token conversation ------------------------------------------
    if (big) {
        ie::Mimo26PrefixCache::Options o; o.budget = 64ull << 30;
        fresh(o);
        const std::vector<int32_t> B = slice(all, 0, big), B2 = slice(all, 0, size_t(big) + 100), side = slice(all, all.size() - 2000, all.size());
        if (!request(B, lg, L, src) || !request(side, lg, L, src)) return 1;
        const auto s1 = cache.stats();
        if (!request(B2, lg, L, src)) return 1;
        const auto s2 = cache.stats();
        check(L == big && src.rfind("slot ", 0) == 0, "big: an " + std::to_string(big) + "-token conversation kept and restored",
              "kept in " + std::to_string(int(s1.last_save_ms)) + " ms, " + mib(s1.bytes) + "; restored in " + std::to_string(int(s2.last_load_ms)) + " ms");
    }

    cache.free_all();
    if (df) df->free_all();
    fwd.free_all();
    std::printf("\nMIMO26 CACHE TEST: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
