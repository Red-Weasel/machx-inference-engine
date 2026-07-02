// tools/ie_partial_stack_eval.cpp — PR #1 decision-gate experiment.
//
// Measures the per-token agreement rate between (a) the full 40-layer
// Qwen3.6 forward and (b) a "partial stack" that exits early after
// layer N and applies the same final-norm + lm_head.  This is the
// viability test for self-speculative decoding using our hybrid
// architecture's 30 cheap DeltaNet layers as a built-in draft path.
//
// Decision criterion (per the inference-engine-expert recommendation):
//   if α_top1(N=10) >= 0.50, the reused-lm_head draft path is viable
//   and PR #2 (EAGLE-2 with DeltaNet-stack drafting) is structural;
//   otherwise we fall back to training a small head with LoRA.
//
// Methodology:
//   * stream-decode a calibration corpus at T=1 (matches the operating
//     point we'd actually accelerate);
//   * at each token step, the model writes full logits AND partial
//     logits at each cut layer (single forward pass, no extra state);
//   * compute α_top1 (cut argmax == full argmax) and α_top5 (cut top-1
//     in full top-5);
//   * also compute mean rank of cut top-1 in full distribution, and
//     mean KL(full || cut) — soft-quality measures.
//
// Read-only w.r.t. KV cache + DeltaNet state.  PPL preserved by
// construction (the full out_logits returned by forward() is bit-
// identical to the no-capture baseline; the new path only reads ws_x_
// and writes side buffers).

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

constexpr const char* B  = "\033[1m";
constexpr const char* G  = "\033[32m";
constexpr const char* Y  = "\033[33m";
constexpr const char* R  = "\033[31m";
constexpr const char* Z  = "\033[0m";

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Same calibration text as ie-perplexity (Frankenstein-style prose, in-
// distribution for any general LM).  Stable across runs.
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

std::string read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// argmax over a vocab-sized fp32 logits row.
int32_t argmax_f32(const std::vector<float>& v) {
    int32_t best = 0;
    float bv = v[0];
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > bv) { bv = v[i]; best = int32_t(i); }
    }
    return best;
}

// top-K indices (descending), partial sort.
std::vector<int32_t> topk_f32(const std::vector<float>& v, int K) {
    std::vector<int32_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int32_t a, int32_t b){ return v[a] > v[b]; });
    idx.resize(K);
    return idx;
}

// Rank of a target id in a full distribution (0 = top-1).
int32_t rank_in(const std::vector<float>& v, int32_t target_id) {
    const float vt = v[target_id];
    int32_t rk = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] > vt) ++rk;
    }
    return rk;
}

// Stable log-softmax of an fp32 row.
std::vector<float> log_softmax_f32(const std::vector<float>& v) {
    float m = v[0];
    for (float x : v) if (x > m) m = x;
    double s = 0.0;
    for (float x : v) s += std::exp(double(x) - double(m));
    const double lse = double(m) + std::log(s);
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = float(double(v[i]) - lse);
    return out;
}

// KL(full || cut) using stable log-softmax on both sides.
//   sum_v exp(log_p_full[v]) * (log_p_full[v] - log_p_cut[v])
double kl_full_vs_cut(const std::vector<float>& full_logits,
                      const std::vector<float>& cut_logits) {
    auto lpf = log_softmax_f32(full_logits);
    auto lpc = log_softmax_f32(cut_logits);
    double kl = 0.0;
    for (size_t i = 0; i < lpf.size(); ++i) {
        const double pf = std::exp(double(lpf[i]));
        kl += pf * (double(lpf[i]) - double(lpc[i]));
    }
    return kl;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    std::string text_path;
    uint32_t    max_tokens = 256;
    uint32_t    max_ctx    = 4096;
    std::vector<uint32_t> cuts = {5, 10, 15, 20, 25, 30, 35};
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"       && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--text"       && i + 1 < argc) text_path = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--ctx"        && i + 1 < argc) max_ctx = std::atoi(argv[++i]);
        else if (a == "--cuts"       && i + 1 < argc) {
            cuts.clear();
            std::stringstream ss(argv[++i]);
            std::string item;
            while (std::getline(ss, item, ',')) cuts.push_back(uint32_t(std::atoi(item.c_str())));
        }
        else if (a == "--verbose" || a == "-v") verbose = true;
    }

    std::printf("%sie-partial-stack-eval%s — PR #1 self-speculative viability\n", B, Z);
    std::printf("  gguf       : %s\n", gguf_path.c_str());
    std::printf("  ctx        : %u\n", max_ctx);
    std::printf("  max_tokens : %u\n", max_tokens);
    std::printf("  cuts       : ");
    for (size_t i = 0; i < cuts.size(); ++i) std::printf("%s%u", i ? "," : "", cuts[i]);
    std::printf("\n");

    // 1. Load GGUF + allocator + tokenizer + model (same as ie-perplexity).
    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return 1;
    }
    ie::DeviceAllocator alloc;
    if (auto err = alloc.init(); !err.empty()) {
        std::fprintf(stderr, "alloc: %s\n", err.c_str()); return 1;
    }
    auto& q = alloc.queue();
    std::printf("  device     : %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    ie::Tokenizer tok;
    if (auto err = tok.load_from_gguf(g); !err.empty()) {
        std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1;
    }
    std::printf("  vocab      : %u\n", tok.vocab_size());

    const double t0 = now_ms();
    std::printf("  loading model (~22 GB)...\n");
    ie::QwenConfig cfg;
    ie::QwenModel  model;
    if (auto err = model.load(alloc, g, cfg); !err.empty()) {
        std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
    }
    std::printf("  loaded in %.1f s\n", (now_ms() - t0) / 1000.0);

    // Sanity-check requested cuts are within range.
    for (uint32_t c : cuts) {
        if (c == 0 || c > cfg.n_layers) {
            std::fprintf(stderr, "cut %u out of range [1..%u]\n", c, cfg.n_layers);
            return 1;
        }
    }

    // 2. Tokenize corpus.
    std::string corpus = text_path.empty() ? kSampleText : read_text_file(text_path);
    if (corpus.empty()) { std::fprintf(stderr, "empty corpus\n"); return 1; }

    auto ids = tok.encode(corpus, /*allow_special=*/false);
    if (tok.add_bos_token() && tok.bos_token_id() >= 0 &&
        (ids.empty() || ids.front() != tok.bos_token_id())) {
        ids.insert(ids.begin(), tok.bos_token_id());
    }
    if (ids.size() > max_tokens) ids.resize(max_tokens);
    if (ids.size() < 2) { std::fprintf(stderr, "need >= 2 tokens, got %zu\n", ids.size()); return 1; }
    if (uint32_t(ids.size()) + 4 > max_ctx) max_ctx = uint32_t(ids.size()) + 4;
    std::printf("  tokens     : %zu\n", ids.size());

    // 3. KV / DN caches.
    const uint32_t L_full = cfg.n_layers / cfg.full_attn_interval;
    const uint32_t L_lin  = cfg.n_layers - L_full;
    ie::KvCache kv;
    {
        ie::KvCacheConfig kvcfg{};
        kvcfg.n_layers_full = L_full;
        kvcfg.n_kv_heads    = cfg.n_kv_heads;
        kvcfg.max_ctx       = max_ctx;
        kvcfg.head_dim      = cfg.head_dim;
        kvcfg.use_int8      = false;
        if (auto err = kv.init(alloc, kvcfg); !err.empty()) {
            std::fprintf(stderr, "kv: %s\n", err.c_str()); return 1;
        }
    }
    ie::DeltaNetState dn;
    if (auto err = dn.init(alloc, ie::DeltaNetStateConfig{
            L_lin, cfg.ssm_n_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim,
            cfg.ssm_inner * 2, cfg.ssm_conv_kernel}); !err.empty()) {
        std::fprintf(stderr, "dn: %s\n", err.c_str()); return 1;
    }
    if (auto err = model.ensure_workspace(1u); !err.empty()) {
        std::fprintf(stderr, "ws: %s\n", err.c_str()); return 1;
    }
    if (auto err = model.ensure_attn_partials(max_ctx); !err.empty()) {
        std::fprintf(stderr, "attn_partials: %s\n", err.c_str()); return 1;
    }

    // 4. Allocate per-cut device logits buffers and register.
    std::vector<sycl::half*> cut_logits_dev(cuts.size(), nullptr);
    for (size_t i = 0; i < cuts.size(); ++i) {
        cut_logits_dev[i] = sycl::malloc_device<sycl::half>(cfg.vocab, q);
    }
    auto* full_logits_dev = sycl::malloc_device<sycl::half>(cfg.vocab, q);
    auto* d_id            = sycl::malloc_device<int32_t>(1, q);

    model.set_partial_logits_capture(cuts, cut_logits_dev);

    std::vector<sycl::half> h_full_h16(cfg.vocab);
    std::vector<float>      h_full_f32(cfg.vocab);
    std::vector<sycl::half> h_cut_h16(cfg.vocab);
    std::vector<float>      h_cut_f32(cfg.vocab);

    // Per-cut accumulators.
    std::vector<uint64_t> n_top1_match(cuts.size(), 0);
    std::vector<uint64_t> n_top5_match(cuts.size(), 0);
    std::vector<uint64_t> n_top10_match(cuts.size(), 0);
    std::vector<double>   sum_rank(cuts.size(), 0.0);   // rank of cut argmax in full distribution
    std::vector<double>   sum_kl(cuts.size(), 0.0);     // KL(full || cut)
    uint64_t n_steps = 0;

    // 5. Stream-decode the corpus T=1.
    kv.reset();
    dn.reset(q);
    const double t_loop0 = now_ms();

    for (uint32_t i = 0; i < ids.size(); ++i) {
        q.memcpy(d_id, &ids[i], sizeof(int32_t)).wait();
        model.forward(q, d_id, /*T=*/1, /*start_pos=*/i, kv, dn, full_logits_dev).wait();

        // We need the next token to score -- skip last position.
        if (i + 1 >= ids.size()) break;

        // Pull full logits.
        q.memcpy(h_full_h16.data(), full_logits_dev,
                 uint64_t(cfg.vocab) * sizeof(sycl::half)).wait();
        for (uint32_t v = 0; v < cfg.vocab; ++v) h_full_f32[v] = float(h_full_h16[v]);
        const int32_t target_full_top1 = argmax_f32(h_full_f32);
        const auto    target_full_top5 = topk_f32(h_full_f32, 5);
        const auto    target_full_top10 = topk_f32(h_full_f32, 10);

        // For each cut, pull and score.
        for (size_t ci = 0; ci < cuts.size(); ++ci) {
            q.memcpy(h_cut_h16.data(), cut_logits_dev[ci],
                     uint64_t(cfg.vocab) * sizeof(sycl::half)).wait();
            for (uint32_t v = 0; v < cfg.vocab; ++v) h_cut_f32[v] = float(h_cut_h16[v]);
            const int32_t cut_top1 = argmax_f32(h_cut_f32);
            const int32_t rk = rank_in(h_full_f32, cut_top1);
            sum_rank[ci] += double(rk);
            sum_kl[ci]   += kl_full_vs_cut(h_full_f32, h_cut_f32);
            if (cut_top1 == target_full_top1) ++n_top1_match[ci];
            if (std::find(target_full_top5.begin(),  target_full_top5.end(),  cut_top1) != target_full_top5.end())  ++n_top5_match[ci];
            if (std::find(target_full_top10.begin(), target_full_top10.end(), cut_top1) != target_full_top10.end()) ++n_top10_match[ci];
        }
        ++n_steps;

        if (verbose && (i % 32 == 0)) {
            std::printf("    i=%4u  full_top1=%6d", i, target_full_top1);
            for (size_t ci = 0; ci < cuts.size(); ++ci) {
                std::printf("  L%u_a1=%.2f", cuts[ci],
                            double(n_top1_match[ci]) / std::max<uint64_t>(1, n_steps));
            }
            std::printf("\n");
        }
    }

    const double loop_ms = now_ms() - t_loop0;

    // 6. Report.
    std::printf("\n%sresults%s — %llu scored steps, %.1f s wall (%.1f ms/step)\n",
                B, Z, (unsigned long long)n_steps,
                loop_ms / 1000.0,
                loop_ms / std::max<uint64_t>(1, n_steps));
    std::printf("\n  cut N  |  α_top1   α_top5   α_top10  |  mean_rank   mean_KL\n");
    std::printf("  -------+--------------------------------+----------------------\n");
    for (size_t ci = 0; ci < cuts.size(); ++ci) {
        const double a1  = double(n_top1_match[ci])  / double(std::max<uint64_t>(1, n_steps));
        const double a5  = double(n_top5_match[ci])  / double(std::max<uint64_t>(1, n_steps));
        const double a10 = double(n_top10_match[ci]) / double(std::max<uint64_t>(1, n_steps));
        const double mr  = sum_rank[ci]  / double(std::max<uint64_t>(1, n_steps));
        const double mkl = sum_kl[ci]    / double(std::max<uint64_t>(1, n_steps));
        const char*  c   = a1 >= 0.50 ? G : (a1 >= 0.35 ? Y : R);
        std::printf("   L%-4u | %s%6.3f%s   %6.3f   %6.3f   |  %8.1f   %7.4f\n",
                    cuts[ci], c, a1, Z, a5, a10, mr, mkl);
    }

    // 7. Decision call.
    std::printf("\n%sdecision%s\n", B, Z);
    bool any_viable = false;
    for (size_t ci = 0; ci < cuts.size(); ++ci) {
        const double a1 = double(n_top1_match[ci]) / double(std::max<uint64_t>(1, n_steps));
        if (a1 >= 0.50) {
            std::printf("  %s✓%s  cut L%u: α_top1=%.3f ≥ 0.50 — reused-lm_head draft path is VIABLE\n",
                        G, Z, cuts[ci], a1);
            any_viable = true;
            break;  // report the smallest-N viable cut
        }
    }
    if (!any_viable) {
        std::printf("  %s✗%s  no cut reaches α_top1 ≥ 0.50 with reused lm_head\n", R, Z);
        std::printf("        → fall back: train a small EAGLE head with LoRA (Tier-S1 option b)\n");
    }

    // Cleanup.
    model.clear_partial_logits_capture();
    for (auto* p : cut_logits_dev) sycl::free(p, q);
    sycl::free(full_logits_dev, q);
    sycl::free(d_id, q);
    return 0;
}
