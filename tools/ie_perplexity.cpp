// tools/ie_perplexity.cpp — Phase 12 perplexity validation harness.
//
// Computes per-token negative log-likelihood (NLL) and perplexity for a
// text corpus.  Used to validate that quantization / kernel changes don't
// silently degrade quality:
//   * fp16 vs INT8 KV cache (item 8 acceptance gate)
//   * Phase 9 kernel changes (regression check)
//   * Cross-engine sanity (compare PPL number against llama.cpp on the
//     same corpus + GGUF for ship-readiness)
//
// Streaming-decode mode (one token per forward call):
//   1. Reset KV / DeltaNet state.
//   2. For each input token i in 0..N-1:
//        a. forward(ids[i:i+1], T=1, start_pos=i)
//        b. host-copy logits, compute log_softmax[ids[i+1]]
//        c. accumulate NLL
//   3. PPL = exp(mean NLL).
//
// O(N) forward calls — slow at long ctx (each step pays full FA-2 over
// all prior KV).  ~512 tokens is enough for a stable PPL bound, and
// finishes in well under a minute even with the 35B model.
//
// Usage:
//   ie-perplexity --gguf <path> [--text <file>] [--max-tokens 512]
//                 [--ctx 4096] [--int8-kv] [--prefill-chunk N] [-v]
//
// If --text is omitted, uses a built-in clean-prose sample for smoke.

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/dense_transformer.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"
#include "ie/qwen35_dense.hpp"
#include "ie/qwen36.hpp"
#include "ie/qwen3moe.hpp"
#include "ie/gptoss.hpp"
#include "ie/gptoss_tp.hpp"
#include "ie/qwen35moe_split.hpp"   // crown all-Q8_0 2-card layer-split (pulls LayerPlan)
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

// Built-in clean English prose sample. Stable register, in-distribution for
// any general LM, used so `ie-perplexity` produces a meaningful number without
// requiring a corpus. Keep this comfortably above the default max_tokens so
// the INT8 KV drift gate is not dominated by a handful of high-variance tokens.
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

// log P[target_id] for one fp16 logits row, numerically stable.
//   log P[t] = logits[t] - logsumexp(logits)
//   Returns NLL = -log P (in nats).
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
    std::string gguf_path  = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    std::string text_path;                 // empty → built-in sample
    uint32_t    max_tokens = 512;
    uint32_t    max_ctx    = 4096;
    bool        int8_kv    = false;
    bool        verbose    = false;
    uint32_t    prefill_chunk = 0;      // 0 = streaming T=1
    uint32_t    n_gpus     = 1;         // >1 → gpt-oss tensor-parallel (the TP PPL gate)

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"       && i + 1 < argc) gguf_path  = argv[++i];
        else if (a == "--text"       && i + 1 < argc) text_path  = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--ctx"        && i + 1 < argc) max_ctx    = std::atoi(argv[++i]);
        else if (a == "--int8-kv")                    int8_kv    = true;
        else if (a == "--verbose" || a == "-v")       verbose    = true;
        else if (a == "--prefill-chunk" && i + 1 < argc) prefill_chunk = std::atoi(argv[++i]);
        else if (a == "--gpus"       && i + 1 < argc) n_gpus     = std::atoi(argv[++i]);
    }

    std::printf("%sie-perplexity%s — Phase 12 PPL validation\n", B, Z);
    std::printf("  gguf       : %s\n", gguf_path.c_str());
    std::printf("  ctx        : %u\n", max_ctx);
    std::printf("  max_tokens : %u\n", max_tokens);
    std::printf("  kv mode    : %s\n", int8_kv ? "INT8 + fp16 shadow" : "fp16");
    std::printf("  pf_chunk   : %u (%s)\n",
                prefill_chunk,
                prefill_chunk > 0 ? "prefill->decode mode" : "streaming T=1");

    // 1. GGUF + allocator + tokenizer.
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

    // 2. Model — arch-tagged dispatch (P2 Task 5): qwen35moe → QwenModel
    //    (crown path, unchanged), qwen3 → DenseModel (no DeltaNet).
    const ie::ModelArch arch = ie::detect_arch(g);
    if (arch == ie::ModelArch::kUnknown) {
        const auto* a = g.find_kv("general.architecture");
        std::fprintf(stderr, "unsupported architecture: %.*s\n",
                     a ? int(a->as_string().size()) : 9,
                     a ? a->as_string().data() : "<missing>");
        return 1;
    }
    const bool is_dense = ie::is_dense_arch(arch);
    const bool is_q35   = (arch == ie::ModelArch::kQwen35Dense);
    const bool is_q3moe = (arch == ie::ModelArch::kQwen3Moe);
    const bool is_gptoss = (arch == ie::ModelArch::kGptOss);
    const bool is_gptoss_tp = is_gptoss && n_gpus > 1;   // tensor-parallel PPL gate
    // crown all-Q8_0 layer-split PPL gate: kQwen35Moe (the final-else single-GPU arch)
    // across >1 cards uses Qwen35MoeSplitModel (host-in/host-out, owns its own KV+DN).
    const bool is_q35moe_split = (arch == ie::ModelArch::kQwen35Moe) && n_gpus > 1;
    std::printf("  arch       : %s\n", arch == ie::ModelArch::kLlama3 ? "llama (dense)"
                                     : is_dense ? "qwen3 (dense)"
                                     : is_q35   ? "qwen35 (dense-hybrid)"
                                     : is_q3moe ? "qwen3moe (standard MoE)"
                                     : is_gptoss ? "gpt-oss (OpenAI MoE)" : "qwen35moe");

    const double t0 = now_ms();
    std::printf("  loading model...\n");
    ie::QwenConfig  cfg;
    ie::QwenModel   model;
    ie::DenseConfig dcfg;
    ie::DenseModel  dmodel;
    ie::Qwen35Config     qcfg;
    ie::Qwen35DenseModel q35model;
    ie::Qwen3MoeConfig   q3mcfg;
    ie::Qwen3MoeModel    q3mmodel;
    ie::GptOssConfig     gcfg;
    ie::GptOssModel      gmodel;
    ie::DeviceFleet      fleet;          // fleet for is_gptoss_tp OR is_q35moe_split
    ie::GptOssTpModel    gtp;
    ie::Qwen35MoeSplitModel q35ms;       // crown all-Q8_0 layer-split (only when is_q35moe_split)
    std::vector<sycl::half> tp_logits;   // host logits from the TP / split forward
    uint32_t vocab = 0;
    if (is_dense) {
        if (auto err = ie::read_dense_config_auto(g, dcfg); !err.empty()) {
            std::fprintf(stderr, "read_dense_config: %s\n", err.c_str()); return 1;
        }
        if (auto err = dmodel.load(alloc, g, dcfg); !err.empty()) {
            std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
        }
        vocab = dcfg.vocab;
    } else if (is_q35) {
        if (auto err = ie::read_qwen35_config(g, qcfg); !err.empty()) {
            std::fprintf(stderr, "read_qwen35_config: %s\n", err.c_str()); return 1;
        }
        if (auto err = q35model.load(alloc, g, qcfg); !err.empty()) {
            std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
        }
        vocab = qcfg.dense.vocab;
    } else if (is_q3moe) {
        if (auto err = ie::read_qwen3moe_config(g, q3mcfg); !err.empty()) {
            std::fprintf(stderr, "read_qwen3moe_config: %s\n", err.c_str()); return 1;
        }
        if (auto err = q3mmodel.load(alloc, g, q3mcfg); !err.empty()) {
            std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
        }
        vocab = q3mcfg.dense.vocab;
    } else if (is_gptoss) {
        if (auto err = ie::read_gptoss_config(g, gcfg); !err.empty()) {
            std::fprintf(stderr, "read_gptoss_config: %s\n", err.c_str()); return 1;
        }
        if (is_gptoss_tp) {
            if (auto err = fleet.init(n_gpus); !err.empty()) {
                std::fprintf(stderr, "fleet.init: %s\n", err.c_str()); return 1;
            }
            if (fleet.size() < 2) { std::fprintf(stderr, "TP needs >=2 GPUs (have %u)\n", fleet.size()); return 1; }
            if (auto err = gtp.load(fleet, g, gcfg, max_ctx); !err.empty()) {
                std::fprintf(stderr, "gptoss-tp load: %s\n", err.c_str()); return 1;
            }
            tp_logits.resize(gcfg.dense.vocab);
            std::printf("  gptoss TP  : %u cards\n", fleet.size());
        } else {
            if (auto err = gmodel.load(alloc, g, gcfg); !err.empty()) {
                std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
            }
        }
        vocab = gcfg.dense.vocab;
    } else if (is_q35moe_split) {
        if (auto err = fleet.init(n_gpus); !err.empty()) {
            std::fprintf(stderr, "fleet.init: %s\n", err.c_str()); return 1;
        }
        if (fleet.size() < 2) { std::fprintf(stderr, "split needs >=2 GPUs (have %u)\n", fleet.size()); return 1; }
        ie::LayerPlan plan = ie::LayerPlan::contiguous(cfg.n_layers, n_gpus);
        if (auto err = q35ms.load(fleet, plan, g, cfg, max_ctx); !err.empty()) {
            std::fprintf(stderr, "qwen35moe_split load: %s\n", err.c_str()); return 1;
        }
        tp_logits.resize(cfg.vocab);
        std::printf("  crown split: %u cards (all-Q8_0)\n", fleet.size());
        vocab = cfg.vocab;
    } else {
        if (auto err = model.load(alloc, g, cfg); !err.empty()) {
            std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
        }
        vocab = cfg.vocab;
    }
    std::printf("  loaded in %.1f s\n", (now_ms() - t0) / 1000.0);

    // 3. Tokenize corpus.
    std::string corpus = text_path.empty() ? kSampleText : read_text_file(text_path);
    if (corpus.empty()) {
        std::fprintf(stderr, "corpus is empty (path '%s')\n", text_path.c_str());
        return 1;
    }
    std::printf("  corpus     : %s, %zu bytes\n",
                text_path.empty() ? "<builtin>" : text_path.c_str(), corpus.size());

    auto ids = tok.encode(corpus, /*allow_special=*/false);
    if (tok.add_bos_token() && tok.bos_token_id() >= 0 &&
        (ids.empty() || ids.front() != tok.bos_token_id())) {
        ids.insert(ids.begin(), tok.bos_token_id());
    }
    if (ids.size() > max_tokens) ids.resize(max_tokens);
    if (ids.size() < 2) {
        std::fprintf(stderr, "corpus tokenized to %zu tokens — need ≥ 2.\n", ids.size());
        return 1;
    }
    if (prefill_chunk > 0) {
        const size_t min_prefill_tokens = size_t(prefill_chunk) + 2;
        if (ids.size() < min_prefill_tokens) {
            std::fprintf(stderr,
                         "prefill-chunk %u requires at least %zu tokens, got %zu "
                         "(need prefill_chunk + 2 so at least one prediction is scored).\n",
                         prefill_chunk, min_prefill_tokens, ids.size());
            return 1;
        }
    }
    std::printf("  tokens     : %zu (capped at max_tokens=%u)\n",
                ids.size(), max_tokens);
    if (uint32_t(ids.size()) + 4 > max_ctx) {
        const uint32_t old_ctx = max_ctx;
        max_ctx = uint32_t(ids.size()) + 4;
        std::printf("  ctx grew %u → %u (need slack for ids.size())\n", old_ctx, max_ctx);
    }

    // 4. KV / DeltaNet caches sized for max_ctx. Dense: every layer is
    //    full-attention (KV for all n_layers) and there is no DeltaNetState
    //    (dn stays uninit; dn.reset() is a no-op).
    ie::KvCache kv;
    {
        ie::KvCacheConfig kvcfg{};
        if (is_dense) {
            kvcfg.n_layers_full = dcfg.n_layers;
            kvcfg.n_kv_heads    = dcfg.n_kv_heads;
            kvcfg.head_dim      = dcfg.head_dim;
        } else if (is_q35) {
            kvcfg.n_layers_full = qcfg.n_transformer_layers() / qcfg.full_attn_interval;
            kvcfg.n_kv_heads    = qcfg.dense.n_kv_heads;
            kvcfg.head_dim      = qcfg.dense.head_dim;
        } else if (is_q3moe) {
            kvcfg.n_layers_full = q3mcfg.dense.n_layers;   // every layer full-attn
            kvcfg.n_kv_heads    = q3mcfg.dense.n_kv_heads;
            kvcfg.head_dim      = q3mcfg.dense.head_dim;
        } else if (is_gptoss) {
            kvcfg.n_layers_full = gcfg.dense.n_layers;     // every layer full KV (SWA is a mask)
            kvcfg.n_kv_heads    = gcfg.dense.n_kv_heads;
            kvcfg.head_dim      = gcfg.dense.head_dim;
        } else {
            kvcfg.n_layers_full = cfg.n_layers / cfg.full_attn_interval;
            kvcfg.n_kv_heads    = cfg.n_kv_heads;
            kvcfg.head_dim      = cfg.head_dim;
        }
        kvcfg.max_ctx = max_ctx;
        kvcfg.use_int8 = int8_kv;
        // TP / crown-split models own their per-card KV; the standalone kv is unused.
        if (!is_gptoss_tp && !is_q35moe_split) {
            if (auto err = kv.init(alloc, kvcfg); !err.empty()) {
                std::fprintf(stderr, "kv: %s\n", err.c_str()); return 1;
            }
        }
    }
    ie::DeltaNetState dn;
    if (is_q35) {
        const uint32_t n_tx   = qcfg.n_transformer_layers();
        const uint32_t L_full = n_tx / qcfg.full_attn_interval;
        const uint32_t L_lin  = n_tx - L_full;
        const uint32_t conv_ch = qcfg.ssm_inner + 2u * qcfg.ssm_n_k_heads * qcfg.ssm_state;
        if (auto err = dn.init(alloc, ie::DeltaNetStateConfig{
                L_lin, qcfg.ssm_n_v_heads, qcfg.ssm_state, qcfg.ssm_state,
                conv_ch, qcfg.ssm_conv_kernel}); !err.empty()) {
            std::fprintf(stderr, "dn: %s\n", err.c_str()); return 1;
        }
    } else if (!is_dense && !is_q3moe && !is_gptoss && !is_q35moe_split) {   // single-GPU crown; split owns its own per-card DN
        const uint32_t L_full = cfg.n_layers / cfg.full_attn_interval;
        const uint32_t L_lin  = cfg.n_layers - L_full;
        if (auto err = dn.init(alloc, ie::DeltaNetStateConfig{
                L_lin, cfg.ssm_n_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim,
                cfg.ssm_inner * 2, cfg.ssm_conv_kernel}); !err.empty()) {
            std::fprintf(stderr, "dn: %s\n", err.c_str()); return 1;
        }
    }
    const uint32_t ws_T = prefill_chunk > 0 ? prefill_chunk : 1u;
    if (!is_gptoss_tp && !is_q35moe_split) {   // TP / split lazily size their own per-card workspace + partials
        auto wserr = is_dense ? dmodel.ensure_workspace(ws_T)
                   : is_q35   ? q35model.ensure_workspace(ws_T)
                   : is_q3moe ? q3mmodel.ensure_workspace(ws_T)
                   : is_gptoss ? gmodel.ensure_workspace(ws_T)
                              : model.ensure_workspace(ws_T);
        if (!wserr.empty()) { std::fprintf(stderr, "ws: %s\n", wserr.c_str()); return 1; }
        auto aperr = is_dense ? dmodel.ensure_attn_partials(max_ctx)
                   : is_q35   ? q35model.ensure_attn_partials(max_ctx)
                   : is_q3moe ? q3mmodel.ensure_attn_partials(max_ctx)
                   : is_gptoss ? gmodel.ensure_attn_partials(max_ctx)
                              : model.ensure_attn_partials(max_ctx);
        if (!aperr.empty()) { std::fprintf(stderr, "attn_partials: %s\n", aperr.c_str()); return 1; }
    }

    // 5. Device buffers.
    auto* d_logits = sycl::malloc_device<sycl::half>(vocab, q);
    auto* d_id     = sycl::malloc_device<int32_t>(1, q);
    std::vector<sycl::half> h_logits_h16(vocab);
    std::vector<float>      h_logits_f32(vocab);

    // Arch-agnostic forward shim (same shape as Engine::forward_step). The TP path is
    // host-in/host-out (like Engine::forward_step's tp branches): copy device ids →
    // host, run the fleet forward, bounce host logits back into d_logits so the PPL
    // loop below is byte-for-byte unchanged.
    std::vector<int32_t> tp_hids;
    auto forward = [&](const int32_t* d_in, uint32_t T, uint32_t pos) -> sycl::event {
        if (is_gptoss_tp) {
            if (tp_hids.size() < T) tp_hids.resize(T);
            q.memcpy(tp_hids.data(), d_in, T * sizeof(int32_t)).wait();
            if (auto m = gtp.forward(tp_hids.data(), T, pos, /*reset_kv=*/(pos == 0),
                                     tp_logits.data()); !m.empty())
                std::fprintf(stderr, "gptoss-tp forward: %s\n", m.c_str());
            return q.memcpy(d_logits, tp_logits.data(), uint64_t(vocab) * sizeof(sycl::half));
        }
        if (is_q35moe_split) {
            if (tp_hids.size() < T) tp_hids.resize(T);
            q.memcpy(tp_hids.data(), d_in, T * sizeof(int32_t)).wait();
            if (auto m = q35ms.forward(tp_hids.data(), T, pos, /*reset_kv=*/(pos == 0),
                                       tp_logits.data()); !m.empty())
                std::fprintf(stderr, "qwen35moe_split forward: %s\n", m.c_str());
            return q.memcpy(d_logits, tp_logits.data(), uint64_t(vocab) * sizeof(sycl::half));
        }
        return is_dense ? dmodel.forward(q, d_in, T, pos, kv, d_logits)
             : is_q35   ? q35model.forward(q, d_in, T, pos, kv, dn, d_logits)
             : is_q3moe ? q3mmodel.forward(q, d_in, T, pos, kv, d_logits)
             : is_gptoss ? gmodel.forward(q, d_in, T, pos, kv, d_logits)
                        : model.forward(q, d_in, T, pos, kv, dn, d_logits);
    };

    // 6. PPL loop: streaming T=1, or T=prefill_chunk prefill + T=1 decode.
    if (!is_gptoss_tp && !is_q35moe_split) { kv.reset(); dn.reset(q); }   // TP/split reset their own KV via reset_kv=(pos==0)

    double  nll_sum     = 0.0;
    uint32_t n_pred     = 0;
    bool     prefill_branch_ran = false;
    const double t_loop0 = now_ms();

    if (prefill_chunk > 0 && ids.size() > prefill_chunk) {
        auto* d_pf = sycl::malloc_device<int32_t>(prefill_chunk, q);
        q.memcpy(d_pf, ids.data(), prefill_chunk * sizeof(int32_t)).wait();
        forward(d_pf, prefill_chunk, 0).wait();
        prefill_branch_ran = true;
        sycl::free(d_pf, q);

        for (uint32_t i = prefill_chunk; i < ids.size(); ++i) {
            q.memcpy(d_id, &ids[i], sizeof(int32_t)).wait();
            forward(d_id, /*T=*/1, /*start_pos=*/i).wait();

            if (i + 1 < ids.size()) {
                q.memcpy(h_logits_h16.data(), d_logits,
                         uint64_t(vocab) * sizeof(sycl::half)).wait();
                for (uint32_t v = 0; v < vocab; ++v) {
                    h_logits_f32[v] = float(h_logits_h16[v]);
                }
                const int32_t target = ids[i + 1];
                const double  nll    = nll_of_target(h_logits_f32, target);
                nll_sum += nll;
                n_pred  += 1;

                if (verbose) {
                    auto si = tok.token_str(ids[i]);
                    auto st = tok.token_str(target);
                    std::printf("    i=%4u  in='%.*s' -> tgt='%.*s'  nll=%.4f\n",
                                i,
                                int(si.size()), si.data(),
                                int(st.size()), st.data(),
                                nll);
                }
            }
        }
    } else {
        for (uint32_t i = 0; i < ids.size(); ++i) {
            q.memcpy(d_id, &ids[i], sizeof(int32_t)).wait();
            forward(d_id, /*T=*/1, /*start_pos=*/i).wait();

            if (i + 1 < ids.size()) {
                q.memcpy(h_logits_h16.data(), d_logits,
                         uint64_t(vocab) * sizeof(sycl::half)).wait();
                for (uint32_t v = 0; v < vocab; ++v) {
                    h_logits_f32[v] = float(h_logits_h16[v]);
                }
                const int32_t target = ids[i + 1];
                const double  nll    = nll_of_target(h_logits_f32, target);
                nll_sum += nll;
                n_pred  += 1;

                if (verbose) {
                    auto si = tok.token_str(ids[i]);
                    auto st = tok.token_str(target);
                    std::printf("    i=%4u  in='%.*s' -> tgt='%.*s'  nll=%.4f\n",
                                i,
                                int(si.size()), si.data(),
                                int(st.size()), st.data(),
                                nll);
                }
            }
        }
    }
    const double loop_ms = now_ms() - t_loop0;
    if (n_pred == 0) {
        std::fprintf(stderr, "no predictions scored\n");
        sycl::free(d_logits, q);
        sycl::free(d_id, q);
        return 1;
    }
    const double avg_nll = nll_sum / n_pred;
    const double ppl     = std::exp(avg_nll);

    std::printf("\n%sresults%s\n", B, Z);
    std::printf("  predicted  : %u tokens\n", n_pred);
    std::printf("  total time : %.1f s (%.1f ms/token incl. transfer)\n",
                loop_ms / 1000.0,
                loop_ms / std::max<uint32_t>(1u, n_pred));
    std::printf("  avg NLL    : %.4f nats / token\n", avg_nll);
    std::printf("  perplexity : %s%.2f%s\n", G, ppl, Z);
    const char* kv_tag = int8_kv ? "int8" : "fp16";
    const char* pfc_tag = prefill_branch_ran ? "-pfc" : "";
    std::printf("\n# TSV: kv_mode\ttokens\tavg_nll\tperplexity\n");
    std::printf("# %s%s\t%u\t%.6f\t%.4f\n",
                kv_tag, pfc_tag, n_pred, avg_nll, ppl);

    sycl::free(d_logits, q);
    sycl::free(d_id, q);
    return 0;
}
