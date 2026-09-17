// tools/ds41_cache_test.cpp -- V4.1 Phase 46 (docs/deepseek41/86): the prefix cache's gate. Every arm compares a run
// that went THROUGH the cache (a checkpoint restored after the state was dirtied, the live state kept, a host slot
// swapped out and back in, a divergence below the head) against a control that computes the same chunks with no cache
// involvement: the last logits (fp32 bytes, FNV) and 16 greedy tokens must be IDENTICAL. The CPU miss split is off
// (its rows are not bit-identical against the GPU's, and which experts miss depends on the residency history).
//   usage: ie-ds41-cache-test <model> <tables_dir> <text_file> [ranking]
#include "ie/deepseek41_forward.hpp"
#include "ie/expert_stream.hpp"
#include "ie/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? " ok " : "FAIL", what.c_str(), detail.empty() ? "" : "  ", detail.c_str());
    std::fflush(stdout); if (!ok) ++g_fail;
}
uint64_t fnv(const std::vector<float>& v) { uint64_t h = 1469598103934665603ull; for (float x : v) { uint32_t u; std::memcpy(&u, &x, 4); h = (h ^ u) * 1099511628211ull; } return h; }
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: ie-ds41-cache-test <model> <tables_dir> <text_file> [ranking]\n"); return 2; }
    setenv("IE_DS41_CPU_MISS", "0", 1);
    const std::string dir = argv[1], tables = argv[2], text_file = argv[3], rank_in = argc > 4 ? argv[4] : dir + "/ie_ranking_heldout.txt";
    ie::DeepSeek41Model m; if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb; if (auto e = tb.load(tables); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok; if (auto e = tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    std::ifstream tf(text_file, std::ios::binary); const std::string text((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
    const auto all = tok.encode(text, false);
    if (all.size() < 24000) { std::fprintf(stderr, "need >= 24,000 tokens of text, got %zu\n", all.size()); return 2; }
    const std::vector<int32_t> S(all.begin(), all.begin() + 12000), Z(all.begin() + 12000, all.begin() + 24000);   // two unrelated sequences

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) { if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d); }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) { queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); qs.push_back(queues.back().get()); }
    std::vector<std::vector<uint32_t>> ranking;
    if (std::ifstream(rank_in).good()) if (auto e = ie::ds4_expert_priority_read_layers(rank_in, m.config().n_routed_experts, m.config().n_layers, ranking); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }
    ie::Ds41Forward fwd; ie::Ds41Forward::ResidentOptions opt; opt.max_tokens = 16384; opt.max_forward_tokens = 2048;
    if (auto e = fwd.init_resident(qs, m, tb, ranking, opt); !e.empty()) { std::fprintf(stderr, "init_resident: %s\n", e.c_str()); return 1; }
    fwd.set_logits_last_only(true);
    const uint32_t cap = 2048;
    std::string err;
    // the generator's own continuation rule: chunks of cap from `from` (a pos0 = 0 first chunk is a prefill), pipelined when
    // there are several, never leaving 2..8 rows, a remainder of 1..8 fed one token at a time
    auto run_from = [&](const std::vector<int32_t>& ids, uint32_t from, uint32_t to, std::vector<float>& lg) -> bool {
        std::vector<std::pair<uint32_t, uint32_t>> chunks; uint32_t off = from;
        while (to - off > ie::kDs41MaxDecodeRows) {
            uint32_t t = std::min(cap, to - off); const uint32_t rest = to - off - t;
            if (rest >= 1 && rest <= ie::kDs41MaxDecodeRows) t -= ie::kDs41MaxDecodeRows + 1 - rest;
            chunks.push_back({off, t}); off += t;
        }
        if (chunks.size() > 1) err = fwd.forward_pipelined(ids.data(), chunks, lg);
        else for (const auto& [p0, t] : chunks) { err = fwd.forward(ids.data() + p0, t, p0, lg); if (!err.empty()) break; }
        for (; err.empty() && off < to; ++off) err = fwd.forward(ids.data() + off, 1, off, lg);
        if (!err.empty()) { std::printf("forward: %s\n", err.c_str()); return false; }
        return true;
    };
    auto greedy = [&](std::vector<float> lg, uint32_t n, std::vector<int32_t>& out) -> bool {
        out.clear();
        for (uint32_t i = 0; i < n; ++i) {
            const int32_t id = int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin()); out.push_back(id);
            err = fwd.forward(&id, 1, fwd.n_pos(), lg); if (!err.empty()) { std::printf("decode: %s\n", err.c_str()); return false; }
        }
        return true;
    };
    auto fresh_cache = [&] { fwd.set_prefix_cache(false); if (auto e = fwd.set_prefix_cache(true); !e.empty()) { std::printf("set_prefix_cache: %s\n", e.c_str()); std::exit(1); } };
    auto prefix = [](const std::vector<int32_t>& v, size_t n) { return std::vector<int32_t>(v.begin(), v.begin() + std::ptrdiff_t(n)); };
    auto cat = [](std::vector<int32_t> a, const std::vector<int32_t>& b) { a.insert(a.end(), b.begin(), b.end()); return a; };
    std::vector<float> lg, lg_c; std::vector<int32_t> g_c, g, dirt;
    const std::vector<int32_t> S9000 = prefix(S, 9000);

    // ---- control 1: S[0, 8192) then S[8192, 9000) straight through -------------------------------------------------------
    fresh_cache(); fwd.reset_state();
    if (!run_from(S, 0, 8192, lg) || !run_from(S, 8192, 9000, lg_c) || !greedy(lg_c, 16, g_c)) return 1;
    const uint64_t h_c1 = fnv(lg_c); const std::vector<int32_t> g_c1 = g_c;
    std::printf("control 1: logits fnv %016llx, greedy %d %d %d %d ...\n", (unsigned long long)h_c1, g_c1[0], g_c1[1], g_c1[2], g_c1[3]);

    // ---- arm 1: a CHECKPOINT restored after the state was dirtied by 40 decode steps -------------------------------------
    { fresh_cache(); fwd.reset_state();
      if (!run_from(S, 0, 8192, lg) || !greedy(lg, 40, dirt)) return 1;
      uint32_t reused = 0; std::string src; double t0 = now_ms();
      if (auto e = fwd.prefix_prepare(S9000, reused, &src); !e.empty()) { std::printf("prefix_prepare: %s\n", e.c_str()); return 1; }
      const double ms = now_ms() - t0;
      check(reused == 8192 && src == "checkpoint", "arm 1: the prompt resumes from the chunk checkpoint at 8,192 after 40 dirty steps", "reused " + std::to_string(reused) + " from " + src + " in " + std::to_string(int(ms)) + " ms");
      if (!run_from(S9000, reused, 9000, lg) || !greedy(lg, 16, g)) return 1;
      check(fnv(lg) == h_c1, "arm 1: last logits identical to the control");
      check(g == g_c1, "arm 1: 16 greedy tokens identical to the control"); }

    // ---- arm 2: the LIVE state kept (the prompt extends everything it holds), then an ODD checkpoint restored ------------
    { fresh_cache(); fwd.reset_state();
      // greedy() runs every token it samples, so 21 steps leave the state at 8,213 (odd: a ratio-2 half held)
      if (!run_from(S, 0, 8192, lg) || !greedy(lg, 21, g)) return 1;
      std::vector<int32_t> p2 = cat(cat(prefix(S, 8192), g), std::vector<int32_t>(S.begin() + 9000, S.begin() + 9500));
      // control 2: the same state built with no prefix_prepare, then the same continuation
      fresh_cache(); fwd.reset_state();
      if (!run_from(S, 0, 8192, lg) || !greedy(lg, 21, dirt) || dirt != g) { check(false, "arm 2: control reproduces the 21 greedy tokens"); return 1; }
      const uint32_t P = 8192 + 21;
      if (!run_from(p2, P, uint32_t(p2.size()), lg_c) || !greedy(lg_c, 16, g_c)) return 1;
      const uint64_t h_c2 = fnv(lg_c); const std::vector<int32_t> g_c2 = g_c; const uint32_t odd_end = uint32_t(p2.size());
      // the arm
      fresh_cache(); fwd.reset_state();
      if (!run_from(S, 0, 8192, lg) || !greedy(lg, 21, dirt)) return 1;
      uint32_t reused = 0; std::string src;
      if (auto e = fwd.prefix_prepare(p2, reused, &src); !e.empty()) { std::printf("prefix_prepare: %s\n", e.c_str()); return 1; }
      check(reused == P && src == "live", "arm 2: an extending prompt keeps the live state (odd position 8,213)", "reused " + std::to_string(reused) + " from " + src);
      if (!run_from(p2, reused, odd_end, lg) || !greedy(lg, 16, g)) return 1;
      check(fnv(lg) == h_c2, "arm 2: last logits identical to the control");
      check(g == g_c2, "arm 2: 16 greedy tokens identical to the control");
      // now the continuation's end (8,711, odd, a ratio-2 half held) is a checkpoint: dirty the state and come back to it
      std::vector<int32_t> p3 = cat(p2, std::vector<int32_t>(S.begin() + 9500, S.begin() + 9800));
      uint32_t r3 = 0; std::string s3;
      if (auto e = fwd.prefix_prepare(p3, r3, &s3); !e.empty()) { std::printf("prefix_prepare: %s\n", e.c_str()); return 1; }
      check(r3 == odd_end && s3 == "checkpoint", "arm 2b: back to the ODD checkpoint at 8,713 (the continuation chunk's end) after 16 dirty steps", "reused " + std::to_string(r3) + " from " + s3);
      if (!run_from(p3, r3, uint32_t(p3.size()), lg) || !greedy(lg, 16, g)) return 1;
      const uint64_t h_arm = fnv(lg); const std::vector<int32_t> g_arm = g;
      // control 2b: the same prompt p3 built without the dirty steps
      fresh_cache(); fwd.reset_state();
      if (!run_from(S, 0, 8192, lg) || !greedy(lg, 21, dirt) || !run_from(p2, P, odd_end, lg) || !run_from(p3, odd_end, uint32_t(p3.size()), lg_c) || !greedy(lg_c, 16, g_c)) return 1;
      check(h_arm == fnv(lg_c), "arm 2b: last logits identical to the control");
      check(g_arm == g_c, "arm 2b: 16 greedy tokens identical to the control"); }

    // ---- arm 3: a HOST SLOT -- S swapped out by an unrelated prompt Z, then swapped back in ------------------------------
    { fresh_cache(); fwd.reset_state();
      if (!run_from(S, 0, 8192, lg)) return 1;
      uint32_t reused = 0; std::string src;
      if (auto e = fwd.prefix_prepare(prefix(Z, 6000), reused, &src); !e.empty()) { std::printf("prefix_prepare: %s\n", e.c_str()); return 1; }
      auto st1 = fwd.prefix_cache_stats();
      check(reused == 0 && st1.host_slots == 1, "arm 3: an unrelated prompt reuses nothing and keeps S in a host slot",
            "slots " + std::to_string(st1.host_slots) + ", " + std::to_string(st1.host_bytes >> 20) + " MiB, saved in " + std::to_string(int(st1.last_save_ms)) + " ms");
      fwd.reset_state();
      if (!run_from(Z, 0, 6000, lg)) return 1;
      if (auto e = fwd.prefix_prepare(S9000, reused, &src); !e.empty()) { std::printf("prefix_prepare: %s\n", e.c_str()); return 1; }
      auto st2 = fwd.prefix_cache_stats();
      check(reused == 8192 && src == "host slot", "arm 3: S comes back from its host slot at 8,192",
            "reused " + std::to_string(reused) + " from " + src + " in " + std::to_string(int(st2.last_restore_ms)) + " ms; Z kept too: " + std::to_string(st2.host_slots) + " slot(s)");
      if (!run_from(S9000, reused, 9000, lg) || !greedy(lg, 16, g)) return 1;
      check(fnv(lg) == h_c1, "arm 3: last logits identical to control 1");
      check(g == g_c1, "arm 3: 16 greedy tokens identical to control 1");
      // ---- arm 4: a divergence BELOW the head (the first 5,000 tokens shared): back to the checkpoint at 4,096 ----------
      std::vector<int32_t> p4 = cat(prefix(S, 5000), std::vector<int32_t>(Z.begin() + 7000, Z.begin() + 7600));
      if (auto e = fwd.prefix_prepare(p4, reused, &src); !e.empty()) { std::printf("prefix_prepare: %s\n", e.c_str()); return 1; }
      check(reused == 4096 && src == "checkpoint", "arm 4: a prompt diverging at 5,000 resumes from the checkpoint at 4,096", "reused " + std::to_string(reused) + " from " + src);
      if (!run_from(p4, reused, uint32_t(p4.size()), lg) || !greedy(lg, 16, g)) return 1;
      const uint64_t h4 = fnv(lg); const std::vector<int32_t> g4 = g;
      fresh_cache(); fwd.reset_state();
      if (!run_from(p4, 0, 4096, lg) || !run_from(p4, 4096, uint32_t(p4.size()), lg_c) || !greedy(lg_c, 16, g_c)) return 1;
      check(h4 == fnv(lg_c), "arm 4: last logits identical to the control");
      check(g4 == g_c, "arm 4: 16 greedy tokens identical to the control"); }

    // ---- arm 5 (Phase 47): a DISK entry -- S's state at 8,192 written, the cache torn down (a new process's view: nothing
    // in memory), an unrelated prompt run, then S resumed from the file ------------------------------------------------------
    { const std::string dir = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/ds41_cache_test_disk";
      std::error_code ec; std::filesystem::remove_all(dir, ec);
      ie::Ds41Forward::PrefixCacheOptions o; o.disk_dir = dir;
      fwd.set_prefix_cache(false); if (auto e = fwd.set_prefix_cache(true, o); !e.empty()) { std::printf("set_prefix_cache: %s\n", e.c_str()); return 1; }
      fwd.reset_state();
      if (!run_from(S, 0, 8192, lg)) return 1;
      const double tw = now_ms();
      if (auto e = fwd.prefix_persist(8192); !e.empty()) { std::printf("prefix_persist: %s\n", e.c_str()); return 1; }
      fwd.set_prefix_cache(false);                                   // joins the writer; drops every checkpoint and slot
      const double write_ms = now_ms() - tw;
      if (auto e = fwd.set_prefix_cache(true, o); !e.empty()) { std::printf("set_prefix_cache: %s\n", e.c_str()); return 1; }
      size_t files = 0; uint64_t fbytes = 0;
      for (const auto& de : std::filesystem::directory_iterator(dir, ec)) if (de.path().extension() == ".ds41pfx") { ++files; fbytes += de.file_size(ec); }
      check(files == 1, "arm 5: one disk entry written", std::to_string(fbytes >> 20) + " MiB in " + std::to_string(int(write_ms)) + " ms (gather + write)");
      fwd.reset_state();
      if (!run_from(Z, 0, 4096, lg)) return 1;
      uint32_t reused = 0; std::string src;
      if (auto e = fwd.prefix_prepare(S9000, reused, &src); !e.empty()) { std::printf("prefix_prepare: %s\n", e.c_str()); return 1; }
      check(reused == 8192 && src == "disk", "arm 5: S resumes from the disk entry at 8,192", "reused " + std::to_string(reused) + " from " + src + " in " + std::to_string(int(fwd.prefix_cache_stats().last_restore_ms)) + " ms");
      if (!run_from(S9000, reused, 9000, lg) || !greedy(lg, 16, g)) return 1;
      check(fnv(lg) == h_c1, "arm 5: last logits identical to control 1");
      check(g == g_c1, "arm 5: 16 greedy tokens identical to control 1");
      // a prompt that holds only PART of the entry's prefix must not use it
      std::vector<int32_t> p5 = cat(prefix(S, 7000), prefix(Z, 3000));
      fwd.reset_state();
      if (auto e = fwd.prefix_prepare(p5, reused, &src); !e.empty()) { std::printf("prefix_prepare: %s\n", e.c_str()); return 1; }
      check(src != "disk", "arm 5: a prompt diverging inside the entry's prefix does not load it", "source " + src);
      std::filesystem::remove_all(dir, ec); }

    fwd.set_prefix_cache(false);
    fwd.free_resident();
    std::printf("\nCACHE TEST: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
