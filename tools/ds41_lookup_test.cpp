// tools/ds41_lookup_test.cpp — V4.1 Phase 58 gate: prompt-lookup speculation (docs/deepseek41/97).
//   1. lossless: greedy with lookup ON emits exactly the plain loop's tokens, on prompts built to make it copy
//      (a list to rewrite with one change, a paragraph to repeat, a code block to return renamed) and on free text
//   2. it engaged: the copy prompts ran verify passes and accepted drafts (a gate that never drafts proves nothing)
//   3. speed: decode ms/token, plain vs lookup, per prompt (the same process, the same state)
//   4. seeded sampling at temperature 0.7 runs and ends normally with lookup on (the one-hot-draft acceptance rule)
//   Run with IE_DS41_CPU_MISS=0: the CPU miss leg serves one-row steps only, so with it on a one-row step and a
//   verify row compute some experts in different places and a near-tie may resolve differently (docs/deepseek41/59).
//   usage: ie-ds41-lookup-test <model> <tables_dir> [ranking]
#include "ie/deepseek41.hpp"
#include "ie/deepseek41_engram.hpp"
#include "ie/deepseek41_forward.hpp"
#include "ie/deepseek41_generate.hpp"
#include "ie/deepseek41_prompt.hpp"
#include "ie/expert_stream.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {
int g_fail = 0, g_n = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    ++g_n; if (!ok) ++g_fail;
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(), detail.empty() ? "" : "  -- ", detail.c_str());
}
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string tdir = argc > 2 ? argv[2] : dir + "/ie_golden";
    const std::string rank_in = argc > 3 ? argv[3] : dir + "/ie_ranking_decode_chat.txt";

    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(tdir); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) { queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); qs.push_back(queues.back().get()); }
    std::vector<std::vector<uint32_t>> ranking;
    if (auto e = ie::ds4_expert_priority_read_layers(rank_in, m.config().n_routed_experts, m.config().n_layers, ranking); !e.empty())
        std::fprintf(stderr, "ranking %s: %s (index order)\n", rank_in.c_str(), e.c_str());
    ie::Ds41Forward fwd; ie::Ds41Forward::ResidentOptions opt; opt.max_tokens = 4096;
    if (auto e = fwd.init_resident(qs, m, tb, ranking, opt); !e.empty()) { std::fprintf(stderr, "init_resident: %s\n", e.c_str()); return 1; }
    // every exit from here frees the runtime first (the tier's threads and the pinned arena), as the other tools do
    struct FreeAtExit { ie::Ds41Forward& f; ~FreeAtExit() { f.free_resident(); } } free_at_exit{fwd};
    ie::Ds41Generator gen(fwd, tok);

    std::string list;
    for (int i = 0; i < 20; ++i) list += "- task " + std::to_string(100 + i) + ": owner team-" + std::to_string(i % 4) + ", status " + (i % 3 ? "open" : "blocked") + ", due 2026-10-" + std::to_string(10 + i) + "\n";
    const std::string para = "The engine keeps the two cards busy by splitting the forward into pipeline stages. Card zero runs the first "
        "twenty layers while card one runs the last twenty on the previous chunk, and the hidden stream crosses the PCIe link once "
        "per chunk. The prefix cache stores checkpoints at chunk boundaries so a returning conversation resumes where it left off.";
    const std::string code = "def score(rows, weights):\n    total = 0.0\n    for row in rows:\n        value = row.value * weights.get(row.name, 1.0)\n"
        "        if row.flagged:\n            value *= 0.5\n        total += value\n    return total / max(1, len(rows))\n\n"
        "def best(rows, weights):\n    return max(rows, key=lambda row: row.value * weights.get(row.name, 1.0))\n";
    struct Case { const char* name; std::string user; uint32_t n; bool copy; };
    const std::vector<Case> cases = {
        {"list rewrite", "Here is a task list:\n\n" + list + "\nRewrite the COMPLETE list with task 107's status changed to done. Output only the list.", 360, true},
        {"paragraph repeat", "Repeat the following paragraph exactly, then add one sentence about why it helps.\n\n" + para, 160, true},
        {"code rename", "Return this code with the parameter `rows` renamed to `items` everywhere. Output only the code.\n\n```python\n" + code + "```", 170, true},
        {"free text", "In about 120 words, explain what a mixture-of-experts layer does.", 150, false},
    };
    const char* cm = std::getenv("IE_DS41_CPU_MISS"); const bool split_off = cm && std::string(cm) == "0";
    std::printf("CPU miss split %s: %s\n", split_off ? "OFF" : "ON", split_off ? "lookup must be bit-identical to the plain loop" : "divergence is reported, not failed");
    ie::Ds41SampleParams greedy; greedy.temperature = 0.f;
    for (const auto& c : cases) {
        std::vector<ie::Ds41ChatMessage> msgs = {{"user", c.user, "", false, false, "", "", ""}};
        ie::Ds41PromptOptions po; po.thinking = false; std::string err;
        const std::string prompt = ie::ds41_encode_messages(msgs, po, err);
        if (!err.empty()) { std::fprintf(stderr, "encode: %s\n", err.c_str()); return 1; }
        const auto ids = tok.encode(prompt, true);
        std::vector<int32_t> a, b; ie::Ds41GenStats sa, sb;
        gen.set_lookup(0);
        if (auto e = gen.run(ids, c.n, greedy, {}, [](std::string_view) { return true; }, a, sa); !e.empty()) { std::fprintf(stderr, "plain: %s\n", e.c_str()); return 1; }
        gen.set_lookup(1);
        if (auto e = gen.run(ids, c.n, greedy, {}, [](std::string_view) { return true; }, b, sb); !e.empty()) { std::fprintf(stderr, "lookup: %s\n", e.c_str()); return 1; }
        size_t first_diff = 0; while (first_diff < std::min(a.size(), b.size()) && a[first_diff] == b[first_diff]) ++first_diff;
        const double ma = sa.n_gen ? sa.decode_s * 1e3 / sa.n_gen : 0, mb = sb.n_gen ? sb.decode_s * 1e3 / sb.n_gen : 0;
        std::printf("%s: prompt %zu, plain %zu tokens %.1f ms/token, lookup %zu tokens %.1f ms/token (x%.2f); %u passes, %u rows, %u drafts accepted, %u plain steps\n",
                    c.name, ids.size(), a.size(), ma, b.size(), mb, mb > 0 ? ma / mb : 0.0, sb.lookup_passes, sb.lookup_rows, sb.lookup_accepted, sb.lookup_plain);
        // Bit-identity holds with the CPU miss split off (verify rows and one-row steps then run every expert the same
        // way); with it on, a one-row step and a verify row put different experts on the CPU (fp32) vs the GPU (int8
        // activations), so a near-tie may resolve differently -- reported, not failed (distribution-exact either way).
        if (split_off) check(a == b, std::string(c.name) + ": lookup emits exactly the plain loop's tokens",
                             a == b ? "" : "first difference at token " + std::to_string(first_diff) + " of " + std::to_string(a.size()));
        else std::printf("  [info] %s: split ON -- %s\n", c.name, a == b ? "identical tokens" : ("first difference at token " + std::to_string(first_diff) + " of " + std::to_string(a.size())).c_str());
        if (c.copy) check(sb.lookup_passes > 0 && sb.lookup_accepted > sb.lookup_passes, std::string(c.name) + ": verify passes ran and accepted more than one draft per pass on average");
    }
    {   // seeded sampling with lookup on: runs, ends normally, and repeats under the same seed
        std::vector<ie::Ds41ChatMessage> msgs = {{"user", cases[0].user, "", false, false, "", "", ""}};
        ie::Ds41PromptOptions po; po.thinking = false; std::string err;
        const auto ids = tok.encode(ie::ds41_encode_messages(msgs, po, err), true);
        ie::Ds41SampleParams sp; sp.temperature = 0.7f; sp.top_p = 0.95f; sp.seed = 11;
        std::vector<int32_t> x, y; ie::Ds41GenStats s1, s2;
        gen.set_lookup(1);
        const std::string e1 = gen.run(ids, 200, sp, {}, [](std::string_view) { return true; }, x, s1);
        const std::string e2 = gen.run(ids, 200, sp, {}, [](std::string_view) { return true; }, y, s2);
        check(e1.empty() && e2.empty() && !x.empty() && x == y && s1.lookup_passes > 0,
              "sampling at temperature 0.7 with lookup: runs, repeats under its seed, and verified " + std::to_string(s1.lookup_passes) + " passes");
    }
    std::printf("\nLOOKUP TEST: %s (%d/%d)\n", g_fail ? "FAIL" : "PASS", g_n - g_fail, g_n);
    return g_fail ? 1 : 0;
}
