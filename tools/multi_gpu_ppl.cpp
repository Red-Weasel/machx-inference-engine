// tools/multi_gpu_ppl.cpp — perplexity over the multi-GPU LAYER-SPLIT path.
//
// The companion to ie-multi-gpu-run for models too big for ONE GPU (e.g.
// Qwen2.5-72B across 2x B70): it computes the SAME streaming-T=1 perplexity
// ie-perplexity does, but drives DenseModelSplit::forward instead of the
// single-GPU model. ADDITIVE — a new tool only; touches no validated path.
//
// Methodology is byte-for-byte the ie-perplexity streaming-decode mode so the
// number is directly comparable to every other engine PPL:
//   1. reset KV (reset_kv=true on the first forward),
//   2. for each input token i: forward(ids[i:i+1], T=1, start_pos=i),
//      host-copy logits, accumulate NLL of ids[i+1],
//   3. PPL = exp(mean NLL).
// Uses the same built-in clean-prose corpus and the same add_bos logic.
//
// usage: ie-multi-gpu-ppl --gguf <dense.gguf> [--gpus N=2] [--text <file>]
//                         [--max-tokens 512] [-v]

#include "ie/allocator.hpp"          // DeviceFleet
#include "ie/dense_split.hpp"
#include "ie/dense_tp.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr const char* B = "\033[1m";
constexpr const char* G = "\033[32m";
constexpr const char* Z = "\033[0m";

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Identical to ie-perplexity's built-in corpus so the PPL is comparable.
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

// NLL = -log P[target], numerically stable. Identical to ie-perplexity.
double nll_of_target(const std::vector<float>& logits, int32_t target_id) {
    float m = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) m = std::max(m, logits[i]);
    double sum_exp = 0.0;
    for (float v : logits) sum_exp += std::exp(double(v) - double(m));
    const double lse   = double(m) + std::log(sum_exp);
    const double log_p = double(logits[target_id]) - lse;
    return -log_p;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf, text_path;
    uint32_t n_gpus = 2, max_tokens = 512;
    bool verbose = false, tp = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"       && i + 1 < argc) gguf       = argv[++i];
        else if (a == "--gpus"       && i + 1 < argc) n_gpus     = uint32_t(std::atoi(argv[++i]));
        else if (a == "--text"       && i + 1 < argc) text_path  = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = uint32_t(std::atoi(argv[++i]));
        else if (a == "--verbose" || a == "-v")       verbose    = true;
        else if (a == "--tp")                         tp         = true;   // tensor-parallel
    }
    if (gguf.empty()) {
        std::fprintf(stderr, "usage: %s --gguf <dense.gguf> [--gpus N] [--text <file>] [--max-tokens N] [-v]\n", argv[0]);
        return 2;
    }

    std::printf("%sie-multi-gpu-ppl%s — layer-split perplexity\n", B, Z);
    std::printf("  gguf       : %s\n", gguf.c_str());
    std::printf("  max_tokens : %u\n", max_tokens);

    ie::GgufReader g;
    if (auto e = g.open(gguf); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    if (!ie::is_dense_arch(ie::detect_arch(g))) {
        std::fprintf(stderr, "not a dense GGUF (qwen3/qwen2/llama)\n"); return 1;
    }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    ie::DenseConfig cfg;
    if (auto e = ie::read_dense_config_auto(g, cfg); !e.empty()) { std::fprintf(stderr, "cfg: %s\n", e.c_str()); return 1; }
    std::printf("  model      : %u layers, hidden %u, vocab %u\n", cfg.n_layers, cfg.hidden, cfg.vocab);

    ie::DeviceFleet fleet;
    if (auto e = fleet.init(n_gpus); !e.empty()) { std::fprintf(stderr, "fleet: %s\n", e.c_str()); return 1; }
    std::printf("  fleet      : %u GPU(s)\n", fleet.size());

    ie::LayerPlan plan = ie::LayerPlan::contiguous(cfg.n_layers, fleet.size());
    ie::DenseModelSplit ms;
    ie::DenseModelTP    mt;
    std::function<std::string(const int32_t*, uint32_t, uint32_t, bool, sycl::half*)> fwd;
    const double t0 = now_ms();
    std::printf("  loading %s across %u device(s)...\n", tp ? "tensor-parallel" : "layer-split", fleet.size());
    std::vector<uint64_t> db;
    if (tp) {
        if (auto e = mt.load(fleet, g, cfg); !e.empty()) { std::fprintf(stderr, "tp load: %s\n", e.c_str()); return 1; }
        db  = mt.device_bytes();
        fwd = [&](const int32_t* in, uint32_t TT, uint32_t sp, bool rk, sycl::half* lo) { return mt.forward(in, TT, sp, rk, lo); };
    } else {
        if (auto e = ms.load(fleet, plan, g, cfg); !e.empty()) { std::fprintf(stderr, "split load: %s\n", e.c_str()); return 1; }
        db  = ms.device_bytes();
        fwd = [&](const int32_t* in, uint32_t TT, uint32_t sp, bool rk, sycl::half* lo) { return ms.forward(in, TT, sp, rk, lo); };
    }
    for (uint32_t d = 0; d < fleet.size(); ++d) std::printf("    dev %u: %.1f GB weights\n", d, db[d] / 1e9);
    std::printf("  loaded in %.1f s\n", (now_ms() - t0) / 1000.0);

    // Tokenize the corpus exactly as ie-perplexity does.
    std::string corpus = text_path.empty() ? kSampleText : read_text_file(text_path);
    if (corpus.empty()) { std::fprintf(stderr, "corpus empty (path '%s')\n", text_path.c_str()); return 1; }
    auto ids = tok.encode(corpus, /*allow_special=*/false);
    if (tok.add_bos_token() && tok.bos_token_id() >= 0 &&
        (ids.empty() || ids.front() != tok.bos_token_id())) {
        ids.insert(ids.begin(), tok.bos_token_id());
    }
    if (ids.size() > max_tokens) ids.resize(max_tokens);
    if (ids.size() < 2) { std::fprintf(stderr, "corpus tokenized to %zu tokens — need >= 2.\n", ids.size()); return 1; }
    std::printf("  corpus     : %s, tokens %zu\n",
                text_path.empty() ? "<builtin>" : text_path.c_str(), ids.size());

    std::vector<sycl::half> logits(cfg.vocab);
    std::vector<float>      logits_f32(cfg.vocab);

    double   nll_sum = 0.0;
    uint32_t n_pred  = 0;
    const double t_loop0 = now_ms();
    for (uint32_t i = 0; i < ids.size(); ++i) {
        if (auto e = fwd(&ids[i], /*T=*/1, /*start_pos=*/i,
                         /*reset_kv=*/(i == 0), logits.data()); !e.empty()) {
            std::fprintf(stderr, "forward i=%u: %s\n", i, e.c_str()); return 1;
        }
        if (i + 1 < ids.size()) {
            for (uint32_t v = 0; v < cfg.vocab; ++v) logits_f32[v] = float(logits[v]);
            const double nll = nll_of_target(logits_f32, ids[i + 1]);
            nll_sum += nll;
            n_pred  += 1;
            if (verbose) {
                auto si = tok.token_str(ids[i]);
                auto st = tok.token_str(ids[i + 1]);
                std::printf("    i=%4u  in='%.*s' -> tgt='%.*s'  nll=%.4f\n",
                            i, int(si.size()), si.data(), int(st.size()), st.data(), nll);
            }
        }
    }
    const double loop_ms = now_ms() - t_loop0;
    if (n_pred == 0) { std::fprintf(stderr, "no predictions scored\n"); return 1; }

    const double avg_nll = nll_sum / n_pred;
    const double ppl     = std::exp(avg_nll);
    std::printf("\n%sresults%s\n", B, Z);
    std::printf("  predicted  : %u tokens\n", n_pred);
    std::printf("  total time : %.1f s (%.1f ms/token incl. transfer)\n",
                loop_ms / 1000.0, loop_ms / std::max<uint32_t>(1u, n_pred));
    std::printf("  avg NLL    : %.4f nats / token\n", avg_nll);
    std::printf("  perplexity : %s%.2f%s\n", G, ppl, Z);
    std::printf("\n# TSV: gpus\ttokens\tavg_nll\tperplexity\n");
    std::printf("# %u\t%u\t%.6f\t%.4f\n", fleet.size(), n_pred, avg_nll, ppl);
    return 0;
}
