// tools/ds41_load_test.cpp — DeepSeek-V4.1-Flash Phase 3 load test (host-only, no GPU).
// Parses config.json, binds all 96,085 tensors, and checks the derived per-layer schedule and
// the byte accounting against what the checkpoint actually ships.
// Criteria: docs/deepseek41/04_PHASE3_CRITERIA_2026-09-12.md
#include "ie/deepseek41.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    const std::string l = std::string(ok ? "[ ok ] " : "[FAIL] ") + what +
                          (detail.empty() ? "" : "  (" + detail + ")");
    std::printf("%s\n", l.c_str());
    if (!ok) ++g_fail;
}
std::string set_str(const std::set<uint32_t>& s) {
    std::string o = "{";
    for (auto v : s) o += (o.size() > 1 ? "," : "") + std::to_string(v);
    return o + "}";
}
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";

    ie::DeepSeek41Model m;
    if (const auto e = m.load(dir); !e.empty()) {
        std::fprintf(stderr, "load(%s): %s\n", dir.c_str(), e.c_str());
        return 1;
    }
    const auto& c = m.config();

    std::printf("=== C1 config ===\n");
    std::printf("       %u layers + %u MTP, dim %u, %u heads x head_dim %u (rope %u / nope %u)\n",
                c.n_layers, c.n_mtp_layers, c.dim, c.n_heads, c.head_dim, c.rope_head_dim,
                c.nope_head_dim());
    std::printf("       MoE %u routed / %u shared / %u active, inter %u, score '%s', scale %.2f\n",
                c.n_routed_experts, c.n_shared_experts, c.n_activated_experts, c.moe_inter_dim,
                c.score_func.c_str(), double(c.route_scale));
    std::printf("       o_groups %u x o_lora %u, q_lora %u, window %u, eps %g, hc_mult %u\n",
                c.o_groups, c.o_lora_rank, c.q_lora_rank, c.window_size, double(c.norm_eps), c.hc_mult);
    std::printf("       indexer %u heads x %u, topk %u; yarn factor %.0f over %u\n",
                c.index_n_heads, c.index_head_dim, c.index_topk, double(c.rope_factor),
                c.original_seq_len);
    check(c.n_layers == 40 && c.n_mtp_layers == 3 && c.dim == 5120 && c.head_dim == 512 &&
          c.n_routed_experts == 384 && c.n_activated_experts == 6 && c.hc_mult == 4,
          "config matches the shipped shapes");

    std::printf("\n=== C2 derived per-layer schedule ===\n");
    std::set<uint32_t> kv_src, idx_src, gates, engram;
    for (uint32_t L = 0; L < m.layers().size(); ++L) {
        const auto& k = m.layers()[L].kind;
        if (k.is_kv_source)         kv_src.insert(L);
        if (k.is_index_source)      idx_src.insert(L);
        if (k.has_compressor_gate)  gates.insert(L);
        if (k.has_engram)           engram.insert(L);
    }
    check(kv_src == std::set<uint32_t>{2, 8, 14, 20},  "compressors on the KV-source layers", set_str(kv_src));
    check(gates  == std::set<uint32_t>{2, 8, 14},      "gates only where compress_ratio > 1", set_str(gates));
    check(idx_src == std::set<uint32_t>{2, 8, 14, 20, 24, 28, 32, 36}, "indexers on the index-source layers", set_str(idx_src));
    check(engram == std::set<uint32_t>{1, 14},         "engram on its two layers", set_str(engram));
    check(m.layers()[40].kind.n_routed == 128 && m.layers()[40].kind.n_activated == 3,
          "MTP layers use the DSpark expert pool",
          std::to_string(m.layers()[40].kind.n_routed) + "/" +
          std::to_string(m.layers()[40].kind.n_activated));

    std::printf("\n=== C3 binding is total ===\n");
    check(m.unclaimed().empty(), "no unclaimed tensors",
          m.unclaimed().empty() ? "all " + std::to_string(m.store().all().size()) + " accounted for"
                                : std::to_string(m.unclaimed().size()) + " unknown, first: " + m.unclaimed()[0]);
    for (size_t i = 0; i < m.unclaimed().size() && i < 10; ++i)
        std::printf("       UNCLAIMED: %s\n", m.unclaimed()[i].c_str());

    size_t n_exp = 0, no_scale = 0;
    for (const auto& w : m.layers()) {
        n_exp += w.exp_w1.size() + w.exp_w2.size() + w.exp_w3.size();
        for (const auto* v : {&w.exp_w1, &w.exp_w2, &w.exp_w3})
            for (const auto& t : *v) if (!t.w || !t.s) ++no_scale;
        for (const auto* t : {&w.wq_a, &w.wq_b, &w.wkv, &w.wo_a, &w.wo_b, &w.sh_w1, &w.sh_w2, &w.sh_w3})
            if (!t->w || !t->s) ++no_scale;
    }
    check(no_scale == 0, "every quantised weight is paired with its scale plane",
          std::to_string(n_exp) + " expert planes bound");

    std::printf("\n=== C6 byte accounting ===\n");
    const auto& b = m.budget();
    const double G = double(1u << 30);
    struct Row { const char* n; uint64_t v; };
    for (const Row& r : {Row{"routed experts", b.routed_experts}, Row{"engram", b.engram},
                         Row{"MTP / DSpark", b.mtp}, Row{"attention", b.attention},
                         Row{"shared experts", b.shared_experts}, Row{"embed", b.embed},
                         Row{"lm_head", b.lm_head}, Row{"vision", b.vision},
                         Row{"norms/gates/hc", b.norms_gates_hc}, Row{"indexer", b.indexer},
                         Row{"compressor", b.compressor}})
        std::printf("       %-18s %9.2f GiB\n", r.n, double(r.v) / G);
    std::printf("       %-18s %9.2f GiB\n", "TOTAL", double(b.total()) / G);

    const uint64_t text_core = b.total() - b.engram - b.vision - b.mtp;
    std::printf("       text-only core (no engram/vision/MTP): %.2f GiB, of which experts %.2f\n",
                double(text_core) / G, double(b.routed_experts) / G);
    check(b.total() == 510286023000ull, "byte total == index.json metadata.total_size",
          std::to_string(b.total()) + " vs 510286023000");

    // ---- C1/C4 negative tests --------------------------------------------------------
    // A checkpoint the binder cannot trust must fail loudly and name what is wrong. Both
    // cases run against the REAL shards (symlinked, never copied) with only config.json
    // mutated, so they exercise the same code path a real load takes.
    std::printf("\n=== C1/C4 negative tests ===\n");
    {
        namespace fs = std::filesystem;
        const char* tmp = std::getenv("TMPDIR");
        const fs::path box = fs::path(tmp ? tmp : "/tmp") / "ds41_negative";
        std::error_code ec;
        fs::remove_all(box, ec);
        fs::create_directories(box, ec);
        for (const auto& e : fs::directory_iterator(dir, ec))
            if (e.path().filename() != "config.json")
                fs::create_symlink(e.path(), box / e.path().filename(), ec);

        std::string body;
        { std::ifstream f(dir + "/config.json", std::ios::binary);
          body.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()); }

        auto write_cfg = [&](const std::string& text) {
            std::ofstream o(box / "config.json", std::ios::binary | std::ios::trunc);
            o << text;
        };
        auto load_err = [&]() {
            ie::DeepSeek41Model bad;
            return bad.load(box.string());
        };

        // C1: a required key removed must fail NAMING that key, never silently default.
        std::string missing = body;
        const auto k = missing.find("\"n_routed_experts\"");
        if (k != std::string::npos) missing.replace(k, 18, "\"n_routed_expertsX\"");
        write_cfg(missing);
        const std::string e1 = load_err();
        check(e1.find("n_routed_experts") != std::string::npos && e1.find("missing") != std::string::npos,
              "a missing required config key fails the load and names the key",
              e1.empty() ? "load SUCCEEDED — it silently defaulted" : e1);

        // C4: a shape the config predicts but the file does not carry must fail naming both.
        std::string wrong = body;
        const auto q = wrong.find("\"q_lora_rank\": 1280");
        if (q != std::string::npos) wrong.replace(q, 19, "\"q_lora_rank\": 1281");
        write_cfg(wrong);
        const std::string e2 = load_err();
        check(e2.find("wq_a") != std::string::npos && e2.find("1281") != std::string::npos &&
              e2.find("1280") != std::string::npos,
              "a wrong predicted shape fails the load and names expected AND found",
              e2.empty() ? "load SUCCEEDED — the shape check did not run" : e2);

        fs::remove_all(box, ec);
    }

    std::printf("\n%s\n", g_fail ? ("LOAD TEST: " + std::to_string(g_fail) + " FAILURE(S)").c_str()
                                 : "LOAD TEST: PASS");
    return g_fail ? 1 : 0;
}
