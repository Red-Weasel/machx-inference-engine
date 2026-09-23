// tools/mimo26_reuse.cpp -- ie-mimo26-reuse: P3b gate (b) (docs/mimo26/00_PORT_PLAN.md) -- a follow-up turn prefilled on
// top of the live, DECODE-written caches vs a cold prefill of the whole conversation, compared at the suffix positions.
//   ie-mimo26-reuse <model_dir> <turn1.txt> <suffix.txt> <out_prefix> [--n N] [--lookup] [--chunk C] [--ctx MAX]
// 1. turn1 (raw text, special tokens parsed) is prefilled in chunks of C, then N tokens are decoded greedily -- one-row
//    steps, or with --lookup the engine's prompt-lookup loop (multi-row verify passes and rewinds, the case where a
//    rewind leaves rejected rows in the SWA rings);
// 2. REUSE: the suffix is prefilled on top of those caches, every suffix row's logits kept, exactly as `ie serve` does
//    for a follow-up turn -- after the engine's ring-validity rule (Mimo26Forward::written_end, gate finding 3) says
//    the rings still hold the window before the suffix; if not, the tool says so and exits 3;
// 3. COLD: the caches are reset, [turn1 + generated] is prefilled in chunks of C and the suffix follows as its own chunk
//    (the same boundary as REUSE), the suffix rows' logits kept.
// Writes <out_prefix>_reuse.bin and <out_prefix>_cold.bin in tools/mimo26/compare_logits.py's MLOG format (one sequence:
// the suffix ids and their logits).
#include "ie/mimo26.hpp"
#include "ie/mimo26_forward.hpp"
#include "ie/ngram_draft.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::string& f) {
    std::ifstream in(f, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool write_mlog(const std::string& path, const std::vector<int32_t>& ids, const std::vector<float>& logits, uint32_t V) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t n_seq = 1, n = uint32_t(ids.size());
    std::fwrite("MLOG", 1, 4, f); std::fwrite(&n_seq, 4, 1, f); std::fwrite(&V, 4, 1, f);
    std::fwrite(&n, 4, 1, f); std::fwrite(ids.data(), 4, n, f); std::fwrite(logits.data(), 4, size_t(n) * V, f);
    return std::fclose(f) == 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: ie-mimo26-reuse <model_dir> <turn1.txt> <suffix.txt> <out_prefix> [--n N] [--lookup] [--chunk C] [--ctx MAX]\n"); return 2; }
    const std::string model = argv[1], t1_path = argv[2], sfx_path = argv[3], out = argv[4];
    uint32_t N = 128, C = 2048, ctx = 16384; bool lookup = false;
    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i]; auto val = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--n") N = uint32_t(std::atol(val().c_str()));
        else if (a == "--chunk") C = uint32_t(std::atol(val().c_str()));
        else if (a == "--ctx") ctx = uint32_t(std::atol(val().c_str()));
        else if (a == "--lookup") lookup = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    ie::Mimo26Model m;
    if (auto e = m.load(model); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(model + "/tokenizer.json", model + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    const std::vector<int32_t> t1 = tok.encode(read_file(t1_path), true), sfx = tok.encode(read_file(sfx_path), true);
    if (t1.empty() || sfx.empty()) { std::fprintf(stderr, "empty turn1 or suffix\n"); return 2; }
    if (t1.size() + N + sfx.size() > ctx) { std::fprintf(stderr, "turn1 + n + suffix exceeds --ctx %u\n", ctx); return 2; }

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) { queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); qs.push_back(queues.back().get()); }
    ie::Mimo26Options opt; opt.max_ctx = ctx; opt.max_tokens = std::max<uint32_t>(C, 8);
    ie::Mimo26Forward fwd;
    if (auto e = fwd.init(qs, m, opt); !e.empty()) { std::fprintf(stderr, "init: %s\n", e.c_str()); return 1; }
    const uint32_t V = m.config().vocab_size;
    auto argmax = [](const float* lg, size_t n) { return int32_t(std::max_element(lg, lg + n) - lg); };
    auto prefill = [&](const std::vector<int32_t>& ids, uint32_t from, std::vector<float>& logits, bool all_rows_last) -> std::string {
        for (uint32_t off = from; off < ids.size(); off += C) {
            const uint32_t n = std::min<uint32_t>(C, uint32_t(ids.size()) - off);
            const bool last = off + n == ids.size();
            if (auto e = fwd.forward(ids.data() + off, n, off, logits, last && all_rows_last); !e.empty()) return "prefill at " + std::to_string(off) + ": " + e;
        }
        return {};
    };

    // 1. turn 1: prefill, then N greedy tokens (plain one-row steps or the engine's lookup loop)
    std::vector<float> logits, vlg;
    if (auto e = prefill(t1, 0, logits, false); !e.empty()) { std::fprintf(stderr, "turn1 %s\n", e.c_str()); return 1; }
    std::vector<int32_t> live = t1;
    uint32_t n_pass = 0, n_acc = 0;
    ie::Ds41NgramIndex idx; if (lookup) idx.reset(live);
    int32_t id = argmax(logits.data(), V);
    for (uint32_t k = 0; k < N; ++k) {
        std::vector<int32_t> d;
        if (lookup) { idx.push(id); d = idx.draft(ie::Mimo26Forward::kDecodeRows - 1, 12); if (d.size() > N - 1 - k) d.resize(N - 1 - k); }
        const uint32_t pos = fwd.n_pos();
        if (d.empty()) {
            if (auto e = fwd.forward(&id, 1, pos, logits, false); !e.empty()) { std::fprintf(stderr, "decode: %s\n", e.c_str()); return 1; }
            live.push_back(id); id = argmax(logits.data(), V); continue;
        }
        std::vector<int32_t> rows(1, id); rows.insert(rows.end(), d.begin(), d.end());
        if (auto e = fwd.forward(rows.data(), uint32_t(rows.size()), pos, vlg, true); !e.empty()) { std::fprintf(stderr, "verify: %s\n", e.c_str()); return 1; }
        uint32_t acc = 0; int32_t next = -1;
        for (uint32_t r = 0; r <= d.size(); ++r) {
            const int32_t a = argmax(vlg.data() + size_t(r) * V, V);
            if (r == d.size() || a != d[r]) { next = a; break; }
            ++acc; idx.push(a);
        }
        fwd.rewind(pos + 1 + acc);
        live.insert(live.end(), rows.begin(), rows.begin() + 1 + acc);
        ++n_pass; n_acc += acc; k += acc;
        id = next;
    }
    std::printf("turn1: %zu prompt + %zu generated tokens (%s; %u verify passes, %u drafts accepted); caches hold %u, written to %u\n",
                t1.size(), live.size() - t1.size(), lookup ? "lookup loop" : "one-row steps", n_pass, n_acc, fwd.n_pos(), fwd.written_end());

    // 2. REUSE: the suffix on top of the live caches, under the engine's ring-validity rule
    const uint32_t L = uint32_t(live.size());
    if (fwd.ring() && std::max<uint32_t>(L, fwd.written_end()) - L + fwd.window() > fwd.ring()) {
        std::printf("the SWA rings no longer hold the window before the suffix (the engine would re-prefill cold)\n"); return 3;
    }
    std::vector<int32_t> full = live; full.insert(full.end(), sfx.begin(), sfx.end());
    std::vector<float> reuse;
    if (auto e = prefill(full, L, reuse, true); !e.empty()) { std::fprintf(stderr, "reuse %s\n", e.c_str()); return 1; }
    // 3. COLD: the whole conversation from empty caches -- [turn1 + generated] prefilled in chunks, then the suffix as its
    //    own chunk at the same position: the two runs differ only in how the generated tokens' K/V were written (decode
    //    steps / verify rows vs a prefill), which is the question (b) asks
    fwd.reset();
    std::vector<float> cold, tmp;
    if (auto e = prefill(live, 0, tmp, false); !e.empty()) { std::fprintf(stderr, "cold %s\n", e.c_str()); return 1; }
    if (auto e = prefill(full, L, cold, true); !e.empty()) { std::fprintf(stderr, "cold suffix %s\n", e.c_str()); return 1; }
    const size_t S = sfx.size();
    if (S > C) { std::fprintf(stderr, "the suffix (%zu tokens) must fit one chunk (%u)\n", S, C); return 2; }
    const size_t reuse_rows = reuse.size() / V, cold_rows = cold.size() / V;
    if (reuse_rows < S || cold_rows < S) { std::fprintf(stderr, "internal: %zu / %zu logits rows for a %zu-token suffix\n", reuse_rows, cold_rows, S); return 1; }
    std::vector<float> r_s(reuse.end() - std::ptrdiff_t(S * V), reuse.end()), c_s(cold.end() - std::ptrdiff_t(S * V), cold.end());
    if (!write_mlog(out + "_reuse.bin", sfx, r_s, V) || !write_mlog(out + "_cold.bin", sfx, c_s, V)) { std::fprintf(stderr, "cannot write %s_*.bin\n", out.c_str()); return 1; }
    uint32_t agree = 0;
    for (size_t r = 0; r < S; ++r) agree += argmax(r_s.data() + r * V, V) == argmax(c_s.data() + r * V, V);
    std::printf("suffix: %zu tokens at positions [%u, %zu): top-1 agreement reuse vs cold %u/%zu; wrote %s_reuse.bin, %s_cold.bin\n",
                S, L, full.size(), agree, S, out.c_str(), out.c_str());
    return 0;
}
