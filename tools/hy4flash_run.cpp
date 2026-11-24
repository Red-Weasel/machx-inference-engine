// tools/hy4flash_run.cpp — ie-hy4flash-run: Hy4-flash on the real weights, bring-up form
// (docs/hy4flash/00_PORT_AND_ABLITERATION_PLAN.md, phase 3).
//
//   ie-hy4flash-run <ie_dir> --ids-of <file>
//       the token ids of the file's text, one line (the tokenizer check against the `tokenizers` library)
//   ie-hy4flash-run <ie_dir> (--prompt "<text>" | --prompt-file <file>) [-n N] [--no-bos]
//       reads the prompt, prints the five likeliest next tokens, then N greedy tokens
//   ie-hy4flash-run <ie_dir> --nll-file <file> [--tokens N]
//       teacher-forced: mean negative log-likelihood, perplexity and top-1 accuracy over the text's first N tokens
//       (--nll-out F: every prediction's NLL as fp32, for paired comparisons between runs)
//   ie-hy4flash-run <ie_dir> --batch <file> [--acts-out F] [--logits-out F] [--gen-out F.jsonl -n N [--stop-ids a,b,..]]
//       <file>: one prompt a line, "name<TAB>id id id ..." (token ids as they are fed: no BOS is added). Per prompt:
//       --acts-out    the residual streams entering every layer at the prompt's LAST position, [prompts, layers, hc, dim] fp32
//       --logits-out  the next-token logits, [prompts, vocab] fp32
//       --gen-out     N greedy tokens (stopping at the end token or a --stop-ids id), one JSON object a line
//   common: [--cards 1|2] [--split L] [--ctx N] [--chunk C] [--static N] [--pinned N] [--exact-experts] [--emulate FMT]
//
// --emulate FMT (mxfp4, mxfp6, fp8, int8, q6s, q5a, q4a): the full-precision path with every expert matrix rounded to that
// format first -- a format's quality without building its pack (Hy4fOptions::emulate).
//
// --exact-experts computes the routed experts from Tencent's BF16 weights (slow; the yardstick for the 4-bit pack).
#include "ie/hy4flash.hpp"
#include "ie/hy4flash_forward.hpp"
#include "ie/kernel_profiler.hpp"
#include <unistd.h>
#include <sys/mman.h>
#include "ie/hy4flash_q8.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <numeric>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <thread>
#include <vector>

namespace {
bool read_text(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::stringstream ss; ss << f.rdbuf(); out = ss.str(); return true;
}
double secs_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
std::string shown(const std::string& s) {     // a token's text on one line
    std::string o;
    for (char ch : s) { if (ch == '\n') o += "\\n"; else if (ch == '\t') o += "\\t"; else if (ch == '\r') o += "\\r"; else o += ch; }
    return o;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: ie-hy4flash-run <ie_dir> --ids-of F | --prompt T | --prompt-file F [-n N] [--no-bos] | --nll-file F [--tokens N]  [--cards N] [--split L] [--ctx N] [--chunk C] [--static N] [--pinned N] [--exact-experts]\n"); return 2; }
    const std::string dir = argv[1];
    std::string ids_of, prompt, prompt_file, nll_file, batch, acts_out, logits_out, gen_out, stop_ids_s, emulate, nll_out, route_hist, emulate_hot, hot_file; double hot_frac = 0.25; std::string ranking_file; bool q8 = false, q8_dense = false; uint32_t q8_slots = 0; uint32_t qb_gu = 8, qb_dn = 8; std::vector<uint32_t> q8_slots_card; std::string ablate_list, routes_out, cov_out; bool cov_x_only = false, pack_tables = false; uint32_t kprofile = 0;
    uint32_t rewind_test = 0;
    bool draft_mode = false, draft_eval = false, draft_check = false;
    uint32_t n_cards = 1, split = 0, ctx = 4096, chunk = 256, big_chunk = 0, n_static = 0, n_pinned = 0, n_gen = 0, n_tokens = 0; bool exact = false, bos = true, have_prompt = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--ids-of") ids_of = val();
        else if (a == "--prompt") { prompt = val(); have_prompt = true; }
        else if (a == "--prompt-file") prompt_file = val();
        else if (a == "--nll-file") nll_file = val();
        else if (a == "--batch") batch = val();
        else if (a == "--ablate-list") ablate_list = val();
        else if (a == "--routes-out") routes_out = val();
        else if (a == "--cov-out") cov_out = val();
        else if (a == "--pack-tables") pack_tables = true;
        else if (a == "--kprofile") kprofile = uint32_t(std::atoi(val()));
        else if (a == "--cov-inputs-only") cov_x_only = true;
        else if (a == "--nll-out") nll_out = val();
        else if (a == "--route-hist") route_hist = val();
        else if (a == "--emulate-hot") { emulate_hot = val(); exact = true; }
        else if (a == "--hot-file") hot_file = val();
        else if (a == "--hot-frac") { hot_frac = std::atof(val()); if (!(hot_frac >= 0.0 && hot_frac <= 1.0)) { std::fprintf(stderr, "--hot-frac must be between 0 and 1\n"); return 2; } }
        else if (a == "--acts-out") acts_out = val();
        else if (a == "--logits-out") logits_out = val();
        else if (a == "--gen-out") gen_out = val();
        else if (a == "--stop-ids") stop_ids_s = val();
        else if (a == "-n") n_gen = uint32_t(std::atoi(val()));
        else if (a == "--tokens") n_tokens = uint32_t(std::atoi(val()));
        else if (a == "--no-bos") bos = false;
        else if (a == "--cards") n_cards = uint32_t(std::atoi(val()));
        else if (a == "--split") split = uint32_t(std::atoi(val()));
        else if (a == "--ctx") ctx = uint32_t(std::atoi(val()));
        else if (a == "--chunk") chunk = uint32_t(std::atoi(val()));
        else if (a == "--big-chunk") big_chunk = uint32_t(std::atoi(val()));
        else if (a == "--static") n_static = uint32_t(std::atoi(val()));
        else if (a == "--pinned") n_pinned = uint32_t(std::atoi(val()));
        else if (a == "--exact-experts") exact = true;
        else if (a == "--q8") q8 = true;
        else if (a == "--q8-dense") q8_dense = true;
        else if (a == "--qbits") { const std::string v = val(); const size_t cm = v.find(','); qb_gu = uint32_t(std::atoi(v.c_str())); qb_dn = cm == std::string::npos ? qb_gu : uint32_t(std::atoi(v.c_str() + cm + 1)); }
        else if (a == "--q8-slots") {            // N, or N0,N1 (one count per card)
            const std::string v = val(); std::stringstream ss(v); std::string part;
            while (std::getline(ss, part, ',')) q8_slots_card.push_back(uint32_t(std::atoi(part.c_str())));
            q8_slots = q8_slots_card.empty() ? 0 : q8_slots_card[0]; if (q8_slots_card.size() < 2) q8_slots_card.clear();
        }
        else if (a == "--ranking") ranking_file = val();
        else if (a == "--rewind-test") rewind_test = uint32_t(std::atoi(val()));
        else if (a == "--draft") draft_mode = true;
        else if (a == "--draft-eval") draft_eval = true;
        else if (a == "--draft-check") draft_check = true;
        else if (a == "--emulate") { emulate = val(); exact = true; }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    if (!ids_of.empty()) {
        std::string text;
        if (!read_text(ids_of, text)) { std::fprintf(stderr, "cannot read %s\n", ids_of.c_str()); return 1; }
        const auto ids = tok.encode(text, true);
        for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? " " : "", ids[i]);
        std::printf("\n");
        return 0;
    }
    std::string text;
    const bool nll = !nll_file.empty();
    if (nll) { if (!read_text(nll_file, text)) { std::fprintf(stderr, "cannot read %s\n", nll_file.c_str()); return 1; } }
    else if (!prompt_file.empty()) { if (!read_text(prompt_file, text)) { std::fprintf(stderr, "cannot read %s\n", prompt_file.c_str()); return 1; } }
    else if (have_prompt) text = prompt;
    else if (batch.empty() && !pack_tables) { std::fprintf(stderr, "nothing to do\n"); return 2; }

    ie::Hy4FlashModel m;
    auto t0 = std::chrono::steady_clock::now();
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    if (pack_tables) {          // the n-gram tables as the exact FP8 values the forward rounds them to, beside the pack
        const uint32_t HD = m.config().engram_head_dim();
        if (!m.config().engram_fp8 || HD > 256) { std::fprintf(stderr, "--pack-tables: this config does not round its n-gram rows to FP8\n"); return 2; }
        const long page = ::sysconf(_SC_PAGESIZE);
        for (const ie::Hy4fEngram& eg : m.engrams()) {
            const std::string out = dir + "/engram-" + std::to_string(eg.id) + ".fp8";
            if (eg.fp8) { std::printf("n-gram table %u: already FP8 (%s)\n", eg.id, out.c_str()); continue; }
            if (!eg.table.w || eg.table.w->dtype_str != "BF16") { std::fprintf(stderr, "n-gram table %u: no BF16 source\n", eg.id); return 1; }
            const uint64_t rows = eg.rows(), step = uint64_t(2) << 20, rb = 4 + uint64_t(HD);
            std::FILE* f = std::fopen((out + ".tmp").c_str(), "wb");
            if (!f) { std::fprintf(stderr, "cannot write %s.tmp\n", out.c_str()); return 1; }
            std::vector<uint8_t> buf(step * rb); uint64_t bad = 0; const auto tp = std::chrono::steady_clock::now();
            for (uint64_t r0 = 0; r0 < rows; r0 += step) {
                const uint64_t n = std::min(step, rows - r0);
                const uint8_t* src = eg.table.w->data + r0 * HD * 2;
                const uintptr_t a0 = reinterpret_cast<uintptr_t>(src) & ~uintptr_t(page - 1);
                ::madvise(reinterpret_cast<void*>(a0), size_t(reinterpret_cast<uintptr_t>(src) + n * HD * 2 - a0), MADV_WILLNEED);     // (the mapping is advised random)
                bad += ie::hy4f_engram_pack_fp8(src, n, HD, buf.data());
                if (std::fwrite(buf.data(), 1, size_t(n * rb), f) != size_t(n * rb)) { std::fprintf(stderr, "write error on %s.tmp (drive full?)\n", out.c_str()); std::fclose(f); std::remove((out + ".tmp").c_str()); return 1; }
                ::madvise(reinterpret_cast<void*>(a0), size_t(reinterpret_cast<uintptr_t>(src) + n * HD * 2 - a0), MADV_DONTNEED);
            }
            if (std::fclose(f) != 0 || bad) { std::fprintf(stderr, "n-gram table %u: %llu rows differ from the reference rounding -- nothing kept\n", eg.id, static_cast<unsigned long long>(bad)); std::remove((out + ".tmp").c_str()); return 1; }
            if (std::rename((out + ".tmp").c_str(), out.c_str()) != 0) { std::fprintf(stderr, "cannot rename %s.tmp\n", out.c_str()); return 1; }
            std::printf("n-gram table %u: %llu rows -> %s (%.1f GiB), every row equal to the reference rounding, %.0f s\n", eg.id, static_cast<unsigned long long>(rows), out.c_str(),
                        double(rows * rb) / double(1ull << 30), std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
            std::fflush(stdout);
        }
        return 0;
    }
    const auto& c = m.config();
    std::vector<int32_t> ids;
    std::vector<std::pair<std::string, std::vector<int32_t>>> prompts;      // --batch
    if (!batch.empty()) {
        std::ifstream f(batch);
        if (!f) { std::fprintf(stderr, "cannot read %s\n", batch.c_str()); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            const size_t tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::vector<int32_t> v; std::stringstream ss(line.substr(tab + 1)); long x;
            while (ss >> x) v.push_back(int32_t(x));
            if (v.empty() || v.size() + n_gen > ctx) { std::fprintf(stderr, "prompt '%s': %zu tokens (+ %u) does not fit --ctx %u\n", line.substr(0, tab).c_str(), v.size(), n_gen, ctx); return 2; }
            prompts.emplace_back(line.substr(0, tab), std::move(v));
        }
        if (prompts.empty()) { std::fprintf(stderr, "no prompts in %s\n", batch.c_str()); return 2; }
    } else {
        if (bos) ids.push_back(c.bos_id);
        { const auto t = tok.encode(text, true); ids.insert(ids.end(), t.begin(), t.end()); }
        if (nll && n_tokens && ids.size() > n_tokens) ids.resize(n_tokens);
        if (ids.size() < 2) { std::fprintf(stderr, "the text tokenizes to fewer than 2 tokens\n"); return 2; }
        if (ids.size() + n_gen > ctx) { std::fprintf(stderr, "%zu tokens + %u exceed --ctx %u\n", ids.size(), n_gen, ctx); return 2; }
    }

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.size() < n_cards || n_cards == 0) { std::fprintf(stderr, "%zu Arc GPUs found, %u asked\n", devs.size(), n_cards); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (uint32_t i = 0; i < n_cards; ++i) {
        queues.push_back(kprofile ? std::make_unique<sycl::queue>(sycl::context(devs[i]), devs[i], sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}})
                                 : std::make_unique<sycl::queue>(sycl::context(devs[i]), devs[i], sycl::property_list{sycl::property::queue::in_order{}}));
        qs.push_back(queues.back().get());
    }
    ie::Hy4fOptions o; o.max_ctx = ctx; o.max_tokens = chunk; o.big_tokens = big_chunk; o.split_layer = split; o.n_static = n_static; o.n_pinned = n_pinned; o.exact_experts = exact; o.emulate = emulate; o.q8_experts = q8; o.drafter = draft_mode || draft_eval || draft_check; if (q8_slots) o.q8_slots = q8_slots; o.q8_dense = q8_dense; o.q_bits_gu = qb_gu; o.q_bits_dn = qb_dn; o.q8_slots_card = q8_slots_card;
    if (!ranking_file.empty()) {      // [MoE layers][experts] u32, most-used first -> the forward's per-layer ranking
        std::ifstream rf(ranking_file, std::ios::binary); uint32_t n_moe_r = 0; for (uint32_t L = 0; L < c.n_layers; ++L) n_moe_r += c.moe[L];
        std::vector<uint32_t> flat(size_t(n_moe_r) * c.n_experts);
        if (!rf || !rf.read(reinterpret_cast<char*>(flat.data()), std::streamsize(flat.size() * 4))) { std::fprintf(stderr, "--ranking: cannot read %u x %u u32 from %s\n", n_moe_r, c.n_experts, ranking_file.c_str()); return 2; }
        o.ranking.assign(c.n_layers, std::vector<uint32_t>(c.n_experts)); uint32_t ord = 0;
        for (uint32_t L = 0; L < c.n_layers; ++L) {
            if (c.moe[L]) { std::copy(flat.begin() + size_t(ord) * c.n_experts, flat.begin() + size_t(ord + 1) * c.n_experts, o.ranking[L].begin()); ++ord; }
            else for (uint32_t e = 0; e < c.n_experts; ++e) o.ranking[L][e] = e;
        }
    }
    uint32_t n_moe_layers = 0; for (uint32_t L = 0; L < c.n_layers; ++L) n_moe_layers += c.moe[L];
    if (!emulate_hot.empty()) {
        // the hot set: per MoE layer, the experts that take the top `hot_frac` of the picks counted in a --route-hist file
        std::ifstream f(hot_file, std::ios::binary);
        std::vector<uint64_t> cnt(size_t(n_moe_layers) * c.n_experts);
        if (!f || !f.read(reinterpret_cast<char*>(cnt.data()), std::streamsize(cnt.size() * 8))) { std::fprintf(stderr, "--emulate-hot needs --hot-file (a --route-hist file of this model)\n"); return 2; }
        o.emulate_hot = emulate_hot; o.emulate_hot_set.assign(n_moe_layers, std::vector<uint8_t>(c.n_experts, 0));
        uint64_t n_hot = 0; double share = 0;
        for (uint32_t l = 0; l < n_moe_layers; ++l) {
            std::vector<uint32_t> order(c.n_experts); for (uint32_t e = 0; e < c.n_experts; ++e) order[e] = e;
            const uint64_t* row = cnt.data() + size_t(l) * c.n_experts;
            std::sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return row[x] > row[y] || (row[x] == row[y] && x < y); });
            const uint32_t k = uint32_t(hot_frac * c.n_experts + 0.5);
            uint64_t tot = 0, top = 0; for (uint32_t e = 0; e < c.n_experts; ++e) tot += row[e];
            for (uint32_t i = 0; i < k; ++i) { o.emulate_hot_set[l][order[i]] = 1; top += row[order[i]]; }
            n_hot += k; share += tot ? double(top) / double(tot) : 0;
        }
        std::printf("hot set: %.1f %% of the experts (%llu) as %s, the rest as %s; they took %.1f %% of the picks in %s\n", 100.0 * hot_frac, static_cast<unsigned long long>(n_hot),
                    emulate_hot.c_str(), emulate.empty() ? "exact" : emulate.c_str(), 100.0 * share / n_moe_layers, hot_file.c_str());
    }
    ie::Hy4fForward fw;
    if (auto e = fw.init(qs, m, o); !e.empty()) { std::fprintf(stderr, "init: %s\n", e.c_str()); return 1; }
    if (!cov_out.empty()) fw.record_cov(true, cov_x_only);
    if (q8_dense) std::printf("non-expert matrices as 8-bit: %u\n", fw.dense_q8_mats());
    if (q8) { std::printf("expert slots a layer:"); for (size_t i = 0; i < size_t(n_cards); ++i) std::printf(" card %zu %u", i, fw.q8_slots_of(i)); std::printf("\n"); }
    std::printf("loaded in %.1f s; experts: %s;", secs_since(t0), q8 ? "8-bit integers (the pack on disk where present)" : exact ? (emulate.empty() ? "full precision from the source (slow)" : ("the source's, rounded to " + emulate + " on the card (slow)").c_str()) : "4-bit pack");
    for (size_t i = 0; i < fw.n_cards(); ++i) std::printf(" card %zu %.2f GiB", i, double(fw.vram_bytes(i)) / double(1ull << 30));
    if (m.overlay_tensors()) std::printf("\nOVERLAY: %zu source tensors replaced by overlay.safetensors", m.overlay_tensors());
    std::printf("\n%zu %s\n", batch.empty() ? ids.size() : prompts.size(), batch.empty() ? "tokens" : "prompts");

    const uint32_t V = c.vocab_size;
    std::vector<float> logits;
    if (!batch.empty()) {
        std::vector<int32_t> stops{c.eos_id};
        { std::stringstream ss(stop_ids_s); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) stops.push_back(std::atoi(t.c_str())); }
        std::ofstream fa, fl, fg;
        if (!acts_out.empty()) fa.open(acts_out, std::ios::binary);
        if (!logits_out.empty()) fl.open(logits_out, std::ios::binary);
        if (!gen_out.empty()) fg.open(gen_out);
        std::ofstream frh, frl;
        if (!routes_out.empty()) { frh.open(routes_out + ".hist.u16", std::ios::binary); frl.open(routes_out + ".last.i32", std::ios::binary); }
        std::vector<uint16_t> rhist(size_t(c.n_layers) * c.n_experts); std::vector<int32_t> rlast(size_t(c.n_layers) * c.top_k);
        // IE_HY4F_GATE=<file of "layer expert" lines> [IE_HY4F_GATE_THR=n, default 6]: a candidate's write maps are applied
        // only to a prompt whose last row routes to at least n of the listed experts (the experts that fire on the kind of
        // request the maps are for): its last 4 rows are read again with the maps on and the reply is generated with them;
        // every other prompt runs the unmodified model.
        std::vector<std::vector<uint8_t>> gate; uint32_t gate_thr = 6; std::vector<float> candA, candB; uint32_t candR = 0;
        if (const char* v = std::getenv("IE_HY4F_GATE"); v && *v) {
            gate.assign(c.n_layers, std::vector<uint8_t>(c.n_experts, 0)); std::ifstream gf(v); uint32_t gl, ge2;
            while (gf >> gl >> ge2) if (gl < c.n_layers && ge2 < c.n_experts) gate[gl][ge2] = 1;
        }
        if (const char* v = std::getenv("IE_HY4F_GATE_THR"); v && *v) gate_thr = uint32_t(std::atoi(v));
        std::vector<float> dump, wdump;
        // IE_HY4F_WRITES_OUT=<file> [IE_HY4F_WRITES_ROWS=n]: every prompt's sublayer writes, [2 * n_layers][n][dim] fp32 a prompt
        std::ofstream fw_out; uint32_t writes_rows = 1;
        if (const char* v = std::getenv("IE_HY4F_WRITES_OUT"); v && *v) fw_out.open(v, std::ios::binary);
        if (const char* v = std::getenv("IE_HY4F_WRITES_ROWS"); v && *v) writes_rows = uint32_t(std::max(1, std::atoi(v)));
        auto esc = [](const std::string& x) {
            std::string o; char b[8];
            for (unsigned char ch : x) {
                if (ch == '"') o += "\\\"";
                else if (ch == '\\') o += "\\\\";
                else if (ch == '\n') o += "\\n";
                else if (ch == '\r') o += "\\r";
                else if (ch == '\t') o += "\\t";
                else if (ch < 0x20) { std::snprintf(b, sizeof b, "\\u%04x", ch); o += b; }
                else o += char(ch);
            }
            return o;
        };
        // --ablate-list: the whole batch once per candidate (name, a raw fp32 [k, dim] orthonormal basis file, k, strength);
        // "none" as the file = the unmodified model. --gen-out lines then carry the candidate's name.
        struct Cand { std::string name, file; uint32_t k = 0; float strength = 1.f; std::string mask, steer; float alpha = 0.f; std::string maps; uint32_t rank = 0; };   // maps: <prefix>.A.f32 / .B.f32, [2 * n_layers][rank][dim]
   // steer: raw fp32 [n_layers, hc, dim]      // mask: a text file of "layer expert" lines, "" = none
        std::vector<Cand> cands;
        if (!ablate_list.empty()) {
            std::ifstream f(ablate_list); std::string line;
            while (std::getline(f, line)) {
                std::stringstream ss(line); Cand cd; std::string ks, st;
                if (!std::getline(ss, cd.name, '\t') || !std::getline(ss, cd.file, '\t')) continue;
                if (std::getline(ss, ks, '\t')) cd.k = uint32_t(std::atoi(ks.c_str()));
                if (std::getline(ss, st, '\t')) cd.strength = float(std::atof(st.c_str()));
                std::getline(ss, cd.mask, '\t');
                { std::string al; if (std::getline(ss, cd.steer, '\t') && std::getline(ss, al, '\t')) cd.alpha = float(std::atof(al.c_str())); }
                { std::string rk; if (std::getline(ss, cd.maps, '\t') && std::getline(ss, rk, '\t')) cd.rank = uint32_t(std::atoi(rk.c_str())); }
                cands.push_back(cd);
            }
            if (cands.empty()) { std::fprintf(stderr, "no candidates in %s\n", ablate_list.c_str()); return 2; }
        } else cands.push_back({"", "none", 0, 1.f, "", "", 0.f, "", 0});
        t0 = std::chrono::steady_clock::now(); uint64_t fed = 0, made = 0;
        for (const Cand& cd : cands) {
        if (cd.file == "none" || cd.k == 0) fw.set_ablation(nullptr, 0);
        else {
            std::vector<float> basis(size_t(cd.k) * c.dim);
            std::ifstream bf(cd.file, std::ios::binary);
            if (!bf || !bf.read(reinterpret_cast<char*>(basis.data()), std::streamsize(basis.size() * 4))) { std::fprintf(stderr, "candidate %s: cannot read %u x %u floats from %s\n", cd.name.c_str(), cd.k, c.dim, cd.file.c_str()); return 2; }
            if (auto e = fw.set_ablation(basis.data(), cd.k, cd.strength); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 2; }
        }
        {
            std::vector<std::pair<uint32_t, uint32_t>> mk;
            if (!cd.mask.empty() && cd.mask != "none") {
                std::ifstream mf(cd.mask); uint32_t ml, me;
                if (!mf) { std::fprintf(stderr, "candidate %s: cannot open the expert mask %s\n", cd.name.c_str(), cd.mask.c_str()); return 2; }
                while (mf >> ml >> me) mk.emplace_back(ml, me);
            }
            if (auto e = fw.set_expert_mask(mk); !e.empty()) { std::fprintf(stderr, "candidate %s: %s\n", cd.name.c_str(), e.c_str()); return 2; }
            if (!cd.steer.empty() && cd.steer != "none" && cd.alpha != 0.f) {
                std::vector<float> sv(size_t(c.n_layers) * c.hc_mult * c.dim);
                std::ifstream sf(cd.steer, std::ios::binary);
                if (!sf || !sf.read(reinterpret_cast<char*>(sv.data()), std::streamsize(sv.size() * 4))) { std::fprintf(stderr, "candidate %s: cannot read the steering vectors %s\n", cd.name.c_str(), cd.steer.c_str()); return 2; }
                fw.set_steering(sv.data(), cd.alpha);
                std::printf("  steering: %s, alpha %.2f\n", cd.steer.c_str(), double(cd.alpha));
            } else fw.set_steering(nullptr, 0.f);
            if (!cd.maps.empty() && cd.maps != "none" && cd.rank) {
                const size_t n = size_t(2) * c.n_layers * cd.rank * c.dim; std::vector<float> A(n), B(n);
                std::ifstream fA(cd.maps + ".A.f32", std::ios::binary), fB(cd.maps + ".B.f32", std::ios::binary);
                if (!fA || !fB || !fA.read(reinterpret_cast<char*>(A.data()), std::streamsize(n * 4)) || !fB.read(reinterpret_cast<char*>(B.data()), std::streamsize(n * 4))) { std::fprintf(stderr, "candidate %s: cannot read the write maps %s.{A,B}.f32 at rank %u\n", cd.name.c_str(), cd.maps.c_str(), cd.rank); return 2; }
                candA = A; candB = B; candR = cd.rank;
                if (gate.empty()) { if (auto e = fw.set_write_maps(A.data(), B.data(), cd.rank); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 2; } }
                else fw.set_write_maps(nullptr, nullptr, 0);
                std::printf("  write maps: %s, rank %u%s\n", cd.maps.c_str(), cd.rank, gate.empty() ? "" : " (gated)");
            } else { fw.set_write_maps(nullptr, nullptr, 0); candR = 0; }
            if (!cd.name.empty()) { std::printf("candidate %s (%u directions, strength %.2f, %zu experts masked)\n", cd.name.c_str(), cd.k, double(cd.strength), mk.size()); std::fflush(stdout); }
        }
        for (size_t pi = 0; pi < prompts.size(); ++pi) {
            const auto& [name, p] = prompts[pi];
            fw.reset();
            std::fill(rhist.begin(), rhist.end(), uint16_t(0)); std::fill(rlast.begin(), rlast.end(), -1);
            for (size_t pos = 0; pos < p.size(); pos += chunk) {
                const uint32_t T = uint32_t(std::min<size_t>(chunk, p.size() - pos));
                const bool last = pos + T == p.size();
                dump.clear(); fw.set_stream_dump(last && fa.is_open() ? &dump : nullptr, true);
                wdump.clear(); fw.record_writes(last && fw_out.is_open() ? &wdump : nullptr, writes_rows);
                if (auto e = fw.forward(p.data() + pos, T, logits, false); !e.empty()) { std::fprintf(stderr, "prompt '%s' at %zu: %s\n", name.c_str(), pos, e.c_str()); return 1; }
                if (frh.is_open()) {
                    const auto& rt = fw.last_routes();
                    for (uint32_t L = 0; L < c.n_layers && L < rt.size(); ++L) {
                        if (rt[L].size() != size_t(T) * c.top_k) continue;
                        for (int32_t e : rt[L]) ++rhist[size_t(L) * c.n_experts + size_t(e)];
                        if (last) std::copy(rt[L].end() - c.top_k, rt[L].end(), rlast.begin() + size_t(L) * c.top_k);
                    }
                }
            }
            if (frh.is_open()) {
                frh.write(reinterpret_cast<const char*>(rhist.data()), std::streamsize(rhist.size() * 2));
                frl.write(reinterpret_cast<const char*>(rlast.data()), std::streamsize(rlast.size() * 4));
            }
            uint32_t flags = 0;
            if (!gate.empty()) {
                const auto& rt = fw.last_routes();
                for (uint32_t L = 0; L < c.n_layers && L < rt.size(); ++L) {
                    if (rt[L].size() < c.top_k) continue;
                    for (size_t k = rt[L].size() - c.top_k; k < rt[L].size(); ++k) if (rt[L][k] >= 0 && gate[L][size_t(rt[L][k])]) ++flags;
                }
                if (candR && flags >= gate_thr && p.size() > 4) {
                    fw.rewind(uint32_t(p.size() - 4));
                    fw.set_write_maps(candA.data(), candB.data(), candR);
                    if (auto e = fw.forward(p.data() + p.size() - 4, 4, logits, false); !e.empty()) { std::fprintf(stderr, "prompt '%s' (gated re-read): %s\n", name.c_str(), e.c_str()); return 1; }
                }
            }
            fw.set_stream_dump(nullptr); fw.record_writes(nullptr);
            if (fw_out.is_open()) fw_out.write(reinterpret_cast<const char*>(wdump.data()), std::streamsize(wdump.size() * 4));
            fed += p.size();
            if (fa.is_open()) fa.write(reinterpret_cast<const char*>(dump.data()), std::streamsize(dump.size() * 4));
            if (fl.is_open()) fl.write(reinterpret_cast<const char*>(logits.data()), std::streamsize(size_t(V) * 4));
            if (fg.is_open()) {
                std::vector<int32_t> gen;
                for (uint32_t i = 0; i < n_gen; ++i) {
                    const int32_t next = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
                    gen.push_back(next);
                    if (std::find(stops.begin(), stops.end(), next) != stops.end() || i + 1 == n_gen) break;
                    if (auto e = fw.forward(&next, 1, logits, false); !e.empty()) { std::fprintf(stderr, "prompt '%s' step %u: %s\n", name.c_str(), i, e.c_str()); return 1; }
                }
                made += gen.size();
                if (!gate.empty()) fw.set_write_maps(nullptr, nullptr, 0);
                fg << "{\"candidate\": \"" << esc(cd.name) << "\", \"name\": \"" << esc(name) << "\", \"flags\": " << flags << ", \"ids\": [";
                for (size_t i = 0; i < gen.size(); ++i) fg << (i ? ", " : "") << gen[i];
                fg << "], \"text\": \"" << esc(tok.decode(std::span<const int32_t>(gen.data(), gen.size()), false)) << "\"}\n";
                fg.flush();
            }
            if ((pi + 1) % 8 == 0 || pi + 1 == prompts.size()) { std::printf("  %zu / %zu prompts, %.0f s\n", pi + 1, prompts.size(), secs_since(t0)); std::fflush(stdout); }
        }
        }
        std::printf("batch: %zu prompts, %llu tokens read, %llu generated, %.1f s\n", prompts.size(), static_cast<unsigned long long>(fed), static_cast<unsigned long long>(made), secs_since(t0));
        if (!cov_out.empty()) {             // (the batch mode returns here: the recorder's files are written here too)
            if (auto e = fw.write_cov(cov_out); !e.empty()) { std::fprintf(stderr, "calibration: %s\n", e.c_str()); return 1; }
            std::printf("calibration statistics written to %s.{xtx,hth}.f64 and .json\n", cov_out.c_str());
        }
        return 0;
    }
    if (nll) {
        double sum = 0; uint64_t n = 0, hit = 0; t0 = std::chrono::steady_clock::now(); std::vector<float> per_token; std::vector<uint64_t> hist;
        for (size_t pos = 0; pos < ids.size(); pos += chunk) {
            const uint32_t T = uint32_t(std::min<size_t>(chunk, ids.size() - pos));
            if (auto e = fw.forward(ids.data() + pos, T, logits, true); !e.empty()) { std::fprintf(stderr, "forward at %zu: %s\n", pos, e.c_str()); return 1; }
            if (!route_hist.empty()) {
                if (hist.empty()) hist.assign(size_t(n_moe_layers) * c.n_experts, 0);
                uint32_t ord = 0;
                for (uint32_t L = 0; L < c.n_layers; ++L) { if (!c.moe[L]) continue; for (int32_t e : fw.last_routes()[L]) ++hist[size_t(ord) * c.n_experts + size_t(e)]; ++ord; }
            }
            for (uint32_t t = 0; t < T && pos + t + 1 < ids.size(); ++t) {
                const float* lg = logits.data() + size_t(t) * V; const int32_t want = ids[pos + t + 1];
                float mx = lg[0]; uint32_t am = 0;
                for (uint32_t v = 1; v < V; ++v) if (lg[v] > mx) { mx = lg[v]; am = v; }
                double se = 0;
                for (uint32_t v = 0; v < V; ++v) se += std::exp(double(lg[v]) - mx);
                const double nl = -(double(lg[want]) - mx - std::log(se));
                sum += nl; ++n; hit += am == uint32_t(want); per_token.push_back(float(nl));
            }
            std::printf("  %zu / %zu tokens: mean NLL %.4f, perplexity %.3f, top-1 %.1f %%\n", std::min(ids.size(), pos + T), ids.size(), sum / double(std::max<uint64_t>(n, 1)), std::exp(sum / double(std::max<uint64_t>(n, 1))), 100.0 * double(hit) / double(std::max<uint64_t>(n, 1)));
            std::fflush(stdout);
        }
        const double s = secs_since(t0);
        if (!route_hist.empty()) {
            std::ofstream f(route_hist, std::ios::binary); f.write(reinterpret_cast<const char*>(hist.data()), std::streamsize(hist.size() * 8));
            // how unequal the picks are: the share of the picks the most-used quarter / half of a layer's experts take, and experts never picked
            double q25 = 0, q50 = 0; uint64_t never = 0;
            for (uint32_t l = 0; l < n_moe_layers; ++l) {
                std::vector<uint64_t> row(hist.begin() + size_t(l) * c.n_experts, hist.begin() + size_t(l + 1) * c.n_experts);
                std::sort(row.begin(), row.end(), std::greater<uint64_t>());
                uint64_t tot = 0, a25 = 0, a50 = 0;
                for (uint32_t e = 0; e < c.n_experts; ++e) { tot += row[e]; if (e < c.n_experts / 4) a25 += row[e]; if (e < c.n_experts / 2) a50 += row[e]; never += row[e] == 0; }
                q25 += double(a25) / double(tot); q50 += double(a50) / double(tot);
            }
            std::printf("expert usage over %llu predictions: the most-used 25 %% of a layer's experts take %.1f %% of the picks, the most-used 50 %% take %.1f %%; %llu of %u (layer, expert) pairs never picked\n",
                        static_cast<unsigned long long>(n), 100.0 * q25 / n_moe_layers, 100.0 * q50 / n_moe_layers, static_cast<unsigned long long>(never), n_moe_layers * c.n_experts);
        }
        if (!nll_out.empty()) { std::ofstream f(nll_out, std::ios::binary); f.write(reinterpret_cast<const char*>(per_token.data()), std::streamsize(per_token.size() * 4)); }
        std::printf("NLL %.5f  PPL %.4f  top-1 %.2f %%  over %llu predictions  (%.1f s, %.1f tok/s)\n", sum / double(n), std::exp(sum / double(n)), 100.0 * double(hit) / double(n), static_cast<unsigned long long>(n), s, double(ids.size()) / s);
        return 0;
    }
    // --kprofile N with IE_HY4F_KPROF_PROMPT=1: the kernel profile of the prompt read instead of decode tokens
    const bool kprof_prompt = kprofile && std::getenv("IE_HY4F_KPROF_PROMPT") && *std::getenv("IE_HY4F_KPROF_PROMPT") == '1';
    ie::KernelProfiler pprof; if (kprof_prompt) { ie::g_profiler = &pprof; pprof.begin_step(); pprof.mark_token(); }
    t0 = std::chrono::steady_clock::now();
    uint32_t last_rows = 1;                 // rows of the latest forward() (the drafter is told which row was sampled from)
    for (size_t pos = 0; pos < ids.size();) {
        // --big-chunk B: while more than --chunk rows remain, chunks of up to B rows (in VRAM the expert slots lend)
        const size_t left = ids.size() - pos;
        const uint32_t T = uint32_t(big_chunk > chunk && left > chunk ? std::min<size_t>(big_chunk, left) : std::min<size_t>(chunk, left));
        if (auto e = fw.forward(ids.data() + pos, T, logits, false); !e.empty()) { std::fprintf(stderr, "forward at %zu: %s\n", pos, e.c_str()); return 1; }
        last_rows = T; pos += T;
    }
    const double s_pre = secs_since(t0);
    if (kprof_prompt) {
        for (auto& qq : queues) qq->wait();
        auto stats = pprof.harvest(); ie::g_profiler = nullptr;
        std::sort(stats.begin(), stats.end(), [](const auto& x, const auto& y) { return x.total_ns > y.total_ns; });
        double tot_ms = 0; for (const auto& st : stats) tot_ms += st.total_ms();
        std::printf("kernel profile of the prompt read (%zu tokens): %.0f ms on the cards in all (wall %.0f ms, profiled: not a speed)\n", ids.size(), tot_ms, 1000.0 * s_pre);
        for (size_t i = 0; i < stats.size() && i < 30; ++i) std::printf("  %-28s %8llu calls %10.1f ms %6.1f %%\n", stats[i].name.c_str(), static_cast<unsigned long long>(stats[i].calls), stats[i].total_ms(), 100.0 * stats[i].total_ms() / tot_ms);
        for (size_t d = 0; d < pprof.dev_spans.size(); ++d) std::printf("  card %zu: executing %.0f ms of a %.0f ms window\n", d, 1e-6 * double(pprof.dev_spans[d].busy_ns), 1e-6 * double(pprof.dev_spans[d].span_ns));
        return 0;
    }
    std::printf("prompt read: %.2f s (%.1f tok/s)\n", s_pre, double(ids.size()) / s_pre);
    if (rewind_test && rewind_test < ids.size()) {
        // --rewind-test K: back K tokens, read them again, and the last row's logits must be the ones just computed (the
        // sliding layers' key windows come back from host RAM, the indexers and the residency carry on)
        const std::vector<float> first(logits.begin(), logits.end());
        const size_t at = ids.size() - rewind_test;
        fw.rewind(uint32_t(at));
        for (size_t pos = at; pos < ids.size(); pos += chunk) {
            const uint32_t T = uint32_t(std::min<size_t>(chunk, ids.size() - pos));
            if (auto e = fw.forward(ids.data() + pos, T, logits, false); !e.empty()) { std::fprintf(stderr, "rewind test, forward at %zu: %s\n", pos, e.c_str()); return 1; }
        }
        auto lsm = [&](const std::vector<float>& x, std::vector<double>& o) { double mx = *std::max_element(x.begin(), x.end()), se = 0; for (float v : x) se += std::exp(double(v) - mx); o.resize(x.size()); for (size_t i = 0; i < x.size(); ++i) o[i] = double(x[i]) - mx - std::log(se); };
        std::vector<double> a, b; lsm(first, a); lsm(logits, b);
        double kl = 0, md = 0; for (size_t i = 0; i < a.size(); ++i) { kl += std::exp(a[i]) * (a[i] - b[i]); md = std::max(md, std::fabs(double(first[i]) - double(logits[i]))); }
        const bool same_top = std::max_element(first.begin(), first.end()) - first.begin() == std::max_element(logits.begin(), logits.end()) - logits.begin();
        std::printf("rewind test: back %u tokens to %zu and read again: KL %.2e, largest logit difference %.4f, top token %s\n", rewind_test, at, kl, md, same_top ? "the same" : "DIFFERENT");
    }
    {
        std::vector<uint32_t> order(V); for (uint32_t v = 0; v < V; ++v) order[v] = v;
        std::partial_sort(order.begin(), order.begin() + 5, order.end(), [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
        double mx = logits[order[0]], se = 0;
        for (uint32_t v = 0; v < V; ++v) se += std::exp(double(logits[v]) - mx);
        std::printf("next token:");
        for (int k = 0; k < 5; ++k) { const int32_t id = int32_t(order[size_t(k)]); std::printf("  [%s] %.1f %%", shown(tok.decode(std::span<const int32_t>(&id, 1), false)).c_str(), 100.0 * std::exp(double(logits[order[size_t(k)]]) - mx) / se); }
        std::printf("\n");
    }
    if (n_gen) {
        // IE_HY4F_WAIT_PACK=1 (a measurement): generate only after the background thread has mapped the whole expert pack
        // into this process -- the steady state of a server, where no missed expert takes page faults any more
        if (const char* wp = std::getenv("IE_HY4F_WAIT_PACK"); wp && *wp == '1') {
            const auto tw = std::chrono::steady_clock::now();
            while (!fw.pack_in_ram()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::printf("expert pack mapped after %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count());
        }
        // --kprofile N: the engine's kernel profiler over N decode tokens (after 8 untimed ones): every launch by name with
        // its time ON THE CARD, the launches a token, and how much of a card's own window it was executing. The wall time
        // of this mode means nothing (profiling queues make every submit slow); the card times and the counts are the data.
        if (kprofile) {
            int32_t next = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
            for (uint32_t i = 0; i < 8; ++i) { if (auto e = fw.forward(&next, 1, logits, false); !e.empty()) { std::fprintf(stderr, "decode: %s\n", e.c_str()); return 1; } next = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin()); }
            ie::KernelProfiler prof; ie::g_profiler = &prof; prof.begin_step();
            const auto tk = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < kprofile; ++i) {
                prof.mark_token();
                if (auto e = fw.forward(&next, 1, logits, false); !e.empty()) { std::fprintf(stderr, "decode: %s\n", e.c_str()); return 1; }
                next = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
            }
            for (auto& qq : queues) qq->wait();
            const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tk).count() / kprofile;
            auto stats = prof.harvest(); ie::g_profiler = nullptr;
            std::sort(stats.begin(), stats.end(), [](const auto& x, const auto& y) { return x.total_ns > y.total_ns; });
            double tot_ms = 0; uint64_t calls = 0;
            for (const auto& st : stats) { tot_ms += st.total_ms(); calls += st.calls; }
            std::printf("kernel profile over %u decode tokens: %.0f launches a token, %.2f ms a token on the cards (profiled wall %.0f ms a token: not a speed)\n", kprofile, double(calls) / kprofile, tot_ms / kprofile, wall_ms);
            std::printf("  %-28s %10s %12s %10s %7s\n", "launch", "per token", "ms / token", "us each", "share");
            for (const auto& st : stats) std::printf("  %-28s %10.1f %12.3f %10.1f %6.1f %%\n", st.name.c_str(), double(st.calls) / kprofile, st.total_ms() / kprofile, 1000.0 * st.avg_ms(), 100.0 * st.total_ms() / tot_ms);
            for (size_t d = 0; d < prof.dev_spans.size(); ++d) if (prof.dev_spans[d].tokens)
                std::printf("  card %zu: executing %.2f ms of a %.2f ms window a token (%.0f %%)\n", d, 1e-6 * double(prof.dev_spans[d].busy_ns) / prof.dev_spans[d].tokens, 1e-6 * double(prof.dev_spans[d].span_ns) / prof.dev_spans[d].tokens,
                            prof.dev_spans[d].span_ns ? 100.0 * double(prof.dev_spans[d].busy_ns) / double(prof.dev_spans[d].span_ns) : 0.0);
            std::printf("  (the CPU leg for missed experts: %.2f ms a token on the host over this run)\n", fw.stats().q8_ms_cpu / double(fw.stats().calls ? fw.stats().calls : 1));
            return 0;
        }
        // --draft: the same greedy tokens through the checkpoint's drafter -- a round drafts 5 tokens after the one just
        // sampled, one pass of the model verifies the block, the drafts that match the model's own choice are kept. Run
        // AFTER the plain greedy decode from the same state: the tokens must be the same; the counts and the speed are printed.
        auto gap_of = [](const float* lg, size_t V) { float a = -3e38f, b = -3e38f; for (size_t x = 0; x < V; ++x) { const float v = lg[x]; if (v > a) { b = a; a = v; } else if (v > b) b = v; } return a - b; };
        std::vector<float> ref_gap; std::vector<std::vector<float>> plain_lg;      // plain_lg[i]: the row gen[i] was chosen from
        auto drafted = [&](const std::vector<int32_t>& ref, double s_plain, float conf_arg) -> int {
            if (!fw.has_drafter()) { std::fprintf(stderr, "--draft: this load has no drafter\n"); return 1; }
            const size_t V = c.vocab_size;
            fw.rewind(uint32_t(ids.size() - 1));
            if (auto e = fw.forward(&ids[ids.size() - 1], 1, logits, false); !e.empty()) { std::fprintf(stderr, "--draft: %s\n", e.c_str()); return 1; }
            std::vector<float> lg; std::vector<int32_t> out, d, block; std::vector<uint64_t> hist(fw.draft_width() + 1, 0);
            size_t n_seq = ids.size(); uint32_t row = 0; uint64_t rounds = 0; double ms_draft = 0, ms_verify = 0;
            // conf_arg: a draft is verified while the confidence head gives the drafts so far this chance together
            // (0 = no drafting: plain greedy again, the baseline once the residency has settled on this text)
            const size_t draft_n = conf_arg <= 0.f ? size_t(0) : [] { const char* v = std::getenv("IE_HY4F_DRAFT_N"); return v ? size_t(std::atoi(v)) : size_t(64); }();
            const float draft_conf = conf_arg;
            std::vector<uint64_t> widths(fw.draft_width() + 1, 0);
            const double cpu0 = fw.stats().q8_ms_cpu; const uint64_t miss0 = fw.stats().q8_cpu, hit0 = fw.stats().q8_hits;
            int32_t next = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
            std::vector<float> out_gap; float next_gap = gap_of(logits.data(), V);
            const auto td = std::chrono::steady_clock::now();
            while (out.size() < n_gen) {
                out.push_back(next); out_gap.push_back(next_gap);
                if (next == c.eos_id || out.size() >= n_gen) break;
                auto t1 = std::chrono::steady_clock::now();
                if (draft_n == 0) d.clear();                               // IE_HY4F_DRAFT_N=0: plain greedy again (the run-to-run baseline)
                else if (auto e = fw.draft(row, next, d, {}, nullptr, draft_conf); !e.empty()) { std::fprintf(stderr, "--draft: %s\n", e.c_str()); return 1; }
                ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
                if (draft_n < d.size()) d.resize(draft_n);                 // IE_HY4F_DRAFT_N: verify only the first N drafts
                // a block is the token just chosen and its drafts: never past the context, never more than is still asked for
                { const size_t room = ctx > n_seq + 1 ? size_t(ctx) - n_seq - 1 : 0, left = n_gen - out.size(); if (d.size() > std::min(room, left)) d.resize(std::min(room, left)); }
                block.assign(1, next); block.insert(block.end(), d.begin(), d.end());
                const uint32_t W = uint32_t(block.size());
                ++widths[std::min<size_t>(d.size(), widths.size() - 1)];
                lg.resize(size_t(W) * V);
                t1 = std::chrono::steady_clock::now();
                if (auto e = fw.forward(block.data(), W, lg, true); !e.empty()) { std::fprintf(stderr, "--draft verify: %s\n", e.c_str()); return 1; }
                ms_verify += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
                uint32_t a = 0; bool stop = false;
                for (;; ++a) {
                    next = int32_t(std::max_element(lg.begin() + long(size_t(a) * V), lg.begin() + long(size_t(a + 1) * V)) - (lg.begin() + long(size_t(a) * V)));
                    next_gap = gap_of(lg.data() + size_t(a) * V, V);
                    if (a == d.size() || next != d[a]) break;
                    out.push_back(next); out_gap.push_back(next_gap);
                    if (next == c.eos_id || out.size() >= n_gen) { stop = true; ++a; break; }
                }
                ++rounds; ++hist[std::min<size_t>(a, hist.size() - 1)];
                if (stop) break;
                n_seq += size_t(a) + 1; row = a;
                if (a + 1 < W) fw.rewind(uint32_t(n_seq));
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count();
            size_t same = 0; while (same < out.size() && same < ref.size() && out[same] == ref[same]) ++same;
            uint64_t acc = 0; for (size_t i = 0; i < hist.size(); ++i) acc += hist[i] * i;
            std::printf("%s %.2f: %zu tokens in %.2f s (%.2f tok/s; first plain pass %.2f), %llu rounds, %.2f drafts kept a round (", conf_arg > 0.f ? "drafted, confidence" : "plain again", double(conf_arg), out.size(), s, double(out.size() > 1 ? out.size() - 1 : 1) / s,
                        double(ref.size() > 1 ? ref.size() - 1 : 1) / s_plain, static_cast<unsigned long long>(rounds), rounds ? double(acc) / double(rounds) : 0.0);
            for (size_t i = 0; i < hist.size(); ++i) std::printf("%s%zu: %llu", i ? ", " : "", i, static_cast<unsigned long long>(hist[i]));
            std::printf("; verified");
            for (size_t i = 0; i < widths.size(); ++i) std::printf(" %zu: %llu", i, static_cast<unsigned long long>(widths[i]));
            std::printf("); a round = draft %.1f ms + verify %.1f ms (experts on the CPU %.1f ms: %.0f of %.0f picks)\n", rounds ? ms_draft / double(rounds) : 0.0, rounds ? ms_verify / double(rounds) : 0.0,
                        rounds ? (fw.stats().q8_ms_cpu - cpu0) / double(rounds) : 0.0, rounds ? double(fw.stats().q8_cpu - miss0) / double(rounds) : 0.0,
                        rounds ? double(fw.stats().q8_cpu - miss0 + fw.stats().q8_hits - hit0) / double(rounds) : 0.0);
            std::printf("  tokens %s the first plain pass (%zu of %zu the same)\n", same == out.size() && out.size() == ref.size() ? "==" : "differ from", same, ref.size());
            if (same < out.size() && same < ref.size()) {
                std::vector<float> g = out_gap; std::sort(g.begin(), g.end());
                std::printf("  first difference at %zu: plain [%s], drafted [%s]; top-1 minus top-2 logit there: plain %.4f, drafted %.4f (median over the drafted tokens %.3f)\n", same,
                            tok.decode(std::span<const int32_t>(&ref[same], 1), false).c_str(), tok.decode(std::span<const int32_t>(&out[same], 1), false).c_str(),
                            same < ref_gap.size() ? ref_gap[same] : -1.f, out_gap[same], g.empty() ? 0.f : g[g.size() / 2]);
            }
            return 0;
        };
        // --draft-eval: along the plain greedy decode, at every token the drafter's 5 drafts against the tokens the model
        // then really produced (drafts kept per position), and its confidence head against the measured acceptance
        static const char* kVar[] = {"drafter"};
        const std::vector<int> ev_vars = {0};
        std::vector<std::vector<std::vector<int32_t>>> ev(ev_vars.size());
        std::vector<float> q1; double conf1 = 0, ov_sum = 0, conf_sum = 0, conf_k[8] = {0, 0, 0, 0, 0, 0, 0, 0}; uint64_t ov_n = 0, conf_n = 0;
        std::vector<int32_t> gen; t0 = std::chrono::steady_clock::now();
        // IE_HY4F_FORCE_IDS=<file of ids>: the reply is replayed from the file (every variant then decodes the same tokens, a
        // like-for-like timing); IE_HY4F_GEN_IDS_OUT=<file>: the reply's ids written there
        std::vector<int32_t> force;
        if (const char* v = std::getenv("IE_HY4F_FORCE_IDS"); v && *v) { std::ifstream ff(v); int32_t x; while (ff >> x) force.push_back(x); }
        std::vector<double> dec_ms; std::vector<std::pair<uint64_t, double>> dec_leg;   // per decode token: experts computed on the CPU, the leg's ms
        for (uint32_t i = 0; i < n_gen; ++i) {
            const int32_t next = i < force.size() ? force[i] : int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
            gen.push_back(next); ref_gap.push_back(gap_of(logits.data(), logits.size()));
            if (draft_check) plain_lg.push_back(logits);
            if (next == c.eos_id) break;
            // (the first draft against the model's own next distribution: sum of min(p, q), what the confidence head was
            // trained to predict -- its mean beside the measured mean says whether the drafter sees what it was trained on)
            if (draft_eval && !q1.empty()) {
                auto sm = [&](const std::vector<float>& lg, std::vector<double>& pr) { const double mx = double(*std::max_element(lg.begin(), lg.end())); double se = 0; pr.resize(lg.size()); for (size_t x = 0; x < lg.size(); ++x) { pr[x] = std::exp(double(lg[x]) - mx); se += pr[x]; } for (auto& x : pr) x /= se; };
                std::vector<double> pp, qq; sm(logits, pp); sm(q1, qq);
                double ov = 0; for (size_t x = 0; x < pp.size(); ++x) ov += std::min(pp[x], qq[x]);
                ov_sum += ov; conf_sum += conf1; ++ov_n; q1.clear();
            }
            if (draft_eval && fw.has_drafter()) for (size_t v = 0; v < ev_vars.size(); ++v) {
                std::vector<int32_t> d; std::vector<float> cf;
                std::function<int32_t(uint32_t, const float*)> pk;
                if (v == 0) pk = [&](uint32_t k, const float* lg) { if (k == 0) q1.assign(lg, lg + c.vocab_size); return int32_t(std::max_element(lg, lg + c.vocab_size) - lg); };
                if (auto e = fw.draft(last_rows - 1, next, d, pk, v == 0 ? &cf : nullptr); !e.empty()) { std::fprintf(stderr, "--draft-eval: %s\n", e.c_str()); return 1; }
                if (v == 0) { conf1 = cf.empty() ? 0.0 : double(cf[0]); for (size_t k = 0; k < cf.size() && k < 8; ++k) { conf_k[k] += double(cf[k]); } ++conf_n; }
                ev[v].push_back(d);
            }
            last_rows = 1;
            if (i + 1 == n_gen) break;
            const uint64_t m0 = fw.stats().q8_cpu; const double c0 = fw.stats().q8_ms_cpu; const auto tt = std::chrono::steady_clock::now();
            if (auto e = fw.forward(&next, 1, logits, false); !e.empty()) { std::fprintf(stderr, "decode step %u: %s\n", i, e.c_str()); return 1; }
            dec_leg.emplace_back(fw.stats().q8_cpu - m0, fw.stats().q8_ms_cpu - c0); dec_ms.push_back(secs_since(tt) * 1e3);
        }
        const double s = secs_since(t0);
        if (const char* v = std::getenv("IE_HY4F_GEN_IDS_OUT"); v && *v) { std::ofstream fo(v); for (int32_t x : gen) fo << x << ' '; }
        if (q8 && !dec_leg.empty()) {   // the decode alone, in windows of 50 tokens: the experts that missed the cards, the leg's ms, the token's ms
            std::printf("decode by 50 tokens (CPU experts a token / leg ms / token ms):");
            for (size_t w = 0; w < dec_leg.size(); w += 50) {
                double m = 0, c = 0, t = 0; const size_t n = std::min<size_t>(50, dec_leg.size() - w);
                for (size_t j = w; j < w + n; ++j) { m += double(dec_leg[j].first); c += dec_leg[j].second; t += dec_ms[j]; }
                std::printf("  %.0f / %.1f / %.1f", m / double(n), c / double(n), t / double(n));
            }
            std::printf("\n");
        }
        std::printf("generated %zu tokens in %.2f s (%.2f tok/s):\n%s\n", gen.size(), s, double(gen.size() > 1 ? gen.size() - 1 : 1) / s, tok.decode(std::span<const int32_t>(gen.data(), gen.size()), false).c_str());
        if (draft_eval) for (size_t v = 0; v < ev_vars.size(); ++v) {
            const size_t NW = fw.draft_width(); std::vector<uint64_t> slot(NW, 0); uint64_t kept = 0, n = 0;
            for (size_t i = 0; i < ev[v].size() && i + NW < gen.size(); ++i, ++n) {
                bool run = true;
                for (size_t k = 0; k < NW; ++k) { const bool hit = ev[v][i][k] == gen[i + 1 + k]; if (hit) ++slot[k]; run = run && hit; if (run) ++kept; }
            }
            std::printf("draft-eval %-38s: %.2f drafts kept a position over %llu; slot accuracy", kVar[ev_vars[v]], n ? double(kept) / double(n) : 0.0, static_cast<unsigned long long>(n));
            for (size_t k = 0; k < NW; ++k) std::printf(" %.0f%%", n ? 100.0 * double(slot[k]) / double(n) : 0.0);
            std::printf("\n");
        }
        if (draft_eval && ov_n) {
            std::printf("draft-eval first draft: confidence head predicts %.3f, measured sum min(p, q) %.3f over %llu positions; confidence per draft", conf_sum / double(ov_n), ov_sum / double(ov_n), static_cast<unsigned long long>(ov_n));
            for (size_t k = 0; k < fw.draft_width() && k < 8; ++k) std::printf(" %.2f", conf_k[k] / double(conf_n));
            std::printf("\n");
        }
        // --draft-check: the rows of small multi-row passes against the plain single-row decode of the SAME tokens --
        // (A) the tokens fed again B rows a pass; (B) the same with rejected rows: each pass carries wrong tokens after
        // its first row and is rewound to that row, as a round whose drafts all fail. Per row: the largest logit
        // difference over the plain row's 5 best tokens, and whether the best token is the same.
        if (draft_check && gen.size() > 8) {
            const size_t V = c.vocab_size;
            std::vector<double> means, worsts;
            auto run = [&](const char* name, uint32_t B, bool junk) -> int {
                fw.rewind(uint32_t(ids.size() - 1));
                std::vector<float> lg;
                if (auto e = fw.forward(&ids[ids.size() - 1], 1, lg, false); !e.empty()) { std::fprintf(stderr, "--draft-check: %s\n", e.c_str()); return 1; }
                double worst = 0, sum = 0; uint64_t n = 0, flips = 0; size_t worst_at = 0; size_t n_seq = ids.size();
                auto cmp = [&](const float* row, size_t i) {            // row = the logits after feeding gen[i]; plain_lg[i + 1] is the same row
                    if (i + 1 >= plain_lg.size()) return;
                    const std::vector<float>& pl = plain_lg[i + 1];
                    std::vector<uint32_t> ord(V); std::iota(ord.begin(), ord.end(), 0u);
                    std::partial_sort(ord.begin(), ord.begin() + 5, ord.end(), [&](uint32_t x, uint32_t y) { return pl[x] > pl[y]; });
                    double d = 0; for (int k = 0; k < 5; ++k) d = std::max(d, std::fabs(double(row[ord[size_t(k)]]) - double(pl[ord[size_t(k)]])));
                    sum += d; ++n; if (d > worst) { worst = d; worst_at = i; }
                    if (size_t(std::max_element(row, row + V) - row) != size_t(ord[0])) ++flips;
                };
                for (size_t i = 0; i + 1 < gen.size();) {
                    std::vector<int32_t> blk;
                    const uint32_t Bc = uint32_t(std::min<size_t>(B, ctx > n_seq ? size_t(ctx) - n_seq : 0));       // (rows the context still holds)
                    if (!Bc) break;
                    if (junk) { blk.push_back(gen[i]); for (uint32_t k = 1; k < Bc; ++k) blk.push_back(gen[(i * 7 + k * 13 + 5) % gen.size()]); }
                    else for (uint32_t k = 0; k < Bc && i + k + 1 < gen.size(); ++k) blk.push_back(gen[i + k]);
                    if (auto e = fw.forward(blk.data(), uint32_t(blk.size()), lg, true); !e.empty()) { std::fprintf(stderr, "--draft-check: %s\n", e.c_str()); return 1; }
                    if (junk) { cmp(lg.data(), i); n_seq += 1; if (blk.size() > 1) fw.rewind(uint32_t(n_seq)); i += 1; }
                    else { for (size_t k = 0; k < blk.size(); ++k) cmp(lg.data() + k * V, i + k); n_seq += blk.size(); i += blk.size(); }
                }
                std::printf("draft-check %-44s: %llu rows, logit difference mean %.4f, worst %.4f (token %zu), best token differs on %llu\n", name, static_cast<unsigned long long>(n), n ? sum / double(n) : 0.0, worst, worst_at,
                            static_cast<unsigned long long>(flips));
                means.push_back(n ? sum / double(n) : 0.0); worsts.push_back(worst);
                return 0;
            };
            if (int rc = run("1 row a pass (plain again)", 1, false); rc) return rc;
            if (int rc = run("3 rows a pass", 3, false); rc) return rc;
            if (int rc = run("6 rows a pass", 6, false); rc) return rc;
            if (int rc = run("3 rows a pass, 2 rejected and rewound", 3, true); rc) return rc;
            if (int rc = run("6 rows a pass, 5 rejected and rewound", 6, true); rc) return rc;
            if (int rc = run("1 row a pass (plain a third time)", 1, false); rc) return rc;
            // The verdict. Plain decode is not reproducible to the bit here: which experts are in VRAM changes with every
            // token read, and a missed expert is computed on the CPU in other arithmetic than the card's. The plain passes
            // (first and last line) measure that; a block pass may not differ from plain by more than plain does from itself.
            const double noise_mean = std::max(means.front(), means.back()), noise_worst = std::max(worsts.front(), worsts.back());
            double blk_mean = 0, blk_worst = 0;
            for (size_t i = 1; i + 1 < means.size(); ++i) { blk_mean = std::max(blk_mean, means[i]); blk_worst = std::max(blk_worst, worsts[i]); }
            const bool pass = blk_mean <= 2.0 * noise_mean + 0.02 && blk_worst <= 2.0 * noise_worst + 0.05;
            std::printf("DRAFT CHECK: %s (block passes: mean %.4f, worst %.4f; plain against plain: mean %.4f, worst %.4f; limit = twice plain's own)\n", pass ? "PASS" : "FAIL", blk_mean, blk_worst, noise_mean, noise_worst);
            if (!pass) return 3;
        }
        if (draft_mode) {
            // IE_HY4F_DRAFT_SEQ: the passes to run after the plain one, e.g. "0,0.7,0,0.7" (0 = plain again)
            const char* sq = std::getenv("IE_HY4F_DRAFT_SEQ"); std::string seq = sq && *sq ? sq : "0,0.7";
            for (size_t a0 = 0; a0 < seq.size();) { const size_t a1 = std::min(seq.find(',', a0), seq.size()); if (int rc = drafted(gen, s, float(std::atof(seq.substr(a0, a1 - a0).c_str()))); rc) return rc; a0 = a1 + 1; }
        }
    }
    const auto& st = fw.stats();
    if (!cov_out.empty()) {
        if (auto e = fw.write_cov(cov_out); !e.empty()) { std::fprintf(stderr, "calibration: %s\n", e.c_str()); return 1; }
        std::printf("calibration statistics written to %s.{xtx,hth}.f64 and .json\n", cov_out.c_str());
    }
    if (q8) std::printf("expert pack in RAM: %s (%.0f GiB paged in so far)\n", fw.pack_in_ram() ? "all of it" : "still loading in the background", fw.pack_ram_gib());
    if (!fw.profile().empty()) {
        auto pr = fw.profile(); double tot = 0; for (auto& p : pr) tot += p.second;
        std::sort(pr.begin(), pr.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
        std::printf("where the time goes (every step waited for; prompt read + decode together, %.0f ms):\n", tot);
        for (auto& p : pr) std::printf("  %-30s %8.0f ms  %5.1f %%\n", p.first.c_str(), p.second, 100.0 * p.second / tot);
    }
    if (fw.stats().q8_ms_upload > 0 || fw.stats().q8_ms_kernel > 0) std::printf("decode experts, timed apart: uploads from RAM %.0f ms, the two kernels %.0f ms; router (incl. its wait for everything before it) %.0f ms\n", fw.stats().q8_ms_upload, fw.stats().q8_ms_kernel, fw.stats().ms_router);
    if (q8) std::printf("8-bit experts: load %.0f s; %llu served from VRAM, %llu computed on the CPU (%.0f ms in all), %llu promoted to VRAM (%llu of them after reading the prompt), %llu uploaded on demand\n", fw.stats().q8_build_s, static_cast<unsigned long long>(fw.stats().q8_hits), static_cast<unsigned long long>(fw.stats().q8_cpu), fw.stats().q8_ms_cpu, static_cast<unsigned long long>(fw.stats().q8_promotions), static_cast<unsigned long long>(fw.stats().q8_rebalanced), static_cast<unsigned long long>(fw.stats().q8_uploads));
    if (q8 && fw.stats().q8_streamed) std::printf("8-bit experts: %llu prompt picks streamed to the cards\n", static_cast<unsigned long long>(fw.stats().q8_streamed));
    std::printf("forward time: %.0f ms total (attention %.0f, FFN %.0f, n-gram %.0f; indexer %.0f inside attention)\n", st.ms_total, st.ms_attn, st.ms_ffn, st.ms_engram, st.ms_indexer);
    return 0;
}
