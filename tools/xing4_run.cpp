// tools/xing4_run.cpp -- run Xing4.0-29B-A4B from a Q8_0 GGUF on the cards (src/model/xing4.cpp).
//
//   ie-xing4-run <file.gguf> [--prompt TEXT | --prompt-file F | --ids-file ids.i32] [--chat] [--no-think]
//                [--gen N] [--ctx N] [--chunk N] [--cards 1|2] [--split L] [--logits-out F] [--repeat N]
//
// --logits-out: the prompt's logits for every row, [T, vocab] fp32 (for the KL against the reference oracle,
// scripts/xing4_oracle/kl.py); --logits-last N: only the last N rows. --gen: greedy tokens after the prompt, printed with the prompt read and decode speeds.
#include "ie/tokenizer.hpp"
#include "ie/xing4.hpp"
#include "ie/allocator.hpp"
#include "ie/xing4_engine.hpp"
#include "nlohmann/json.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "--tok-time") {   // <file.gguf> <text files...>: ids hash, count, time (IE_TOK_SPM_REF=1: the old loop)
        ie::GgufReader g;
        if (auto e = g.open(argv[2]); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        ie::Tokenizer tok;
        if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
        for (int i = 3; i < argc; ++i) {
            std::ifstream f(argv[i], std::ios::binary);
            const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            const auto t0 = std::chrono::steady_clock::now();
            const auto ids = tok.encode(text, true);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            uint64_t h = 1469598103934665603ull;
            for (int32_t id : ids) { h ^= uint32_t(id); h *= 1099511628211ull; }
            std::printf("%-40s %7zu chars -> %6zu tokens  hash %016llx  %9.1f ms\n", argv[i], text.size(), ids.size(), (unsigned long long)h, ms);
        }
        return 0;
    }
    if (argc >= 4 && std::string(argv[1]) == "--template-check") {   // <cases.json> <file.gguf>: the renderer and the tokenizer against transformers
        std::ifstream f(argv[2]);
        const auto J = nlohmann::ordered_json::parse(f);
        ie::GgufReader g;
        if (auto e = g.open(argv[3]); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        ie::Tokenizer tok;
        if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
        int bad = 0;
        for (auto it = J.begin(); it != J.end(); ++it) {
            const auto& c = it.value();
            std::vector<ie::ChatTurn> turns;
            for (const auto& m : c["messages"]) {
                ie::ChatTurn t; t.role = m["role"]; t.content = m.value("content", std::string());
                if (m.contains("reasoning_content")) t.reasoning_content = m["reasoning_content"].get<std::string>();
                if (m.contains("tool_calls")) t.tool_calls_json = m["tool_calls"].dump();
                turns.push_back(t);
            }
            std::string tools;
            if (c["kw"].contains("tools")) tools = c["kw"]["tools"].dump();
            const bool think = c["kw"].value("enable_thinking", true);
            std::string err;
            std::vector<std::pair<size_t, size_t>> content;
            const std::string r = ie::xing4_render_prompt(turns, tools, think, err, &content);
            const bool same = r == c["text"].get<std::string>();
            const std::vector<int32_t> want = c["ids"].get<std::vector<int32_t>>();
            const std::vector<int32_t> got = ie::xing4_encode(tok, r, content);
            const bool idsame = got == want;
            std::printf("%-26s render %s, ids %s (%zu vs %zu)\n", it.key().c_str(), same ? "same" : "DIFFERS", idsame ? "same" : "DIFFER", got.size(), want.size());
            if (!same) std::printf("  engine: %s\n  ref:    %s\n", nlohmann::json(r).dump().c_str(), c["text"].dump().c_str());
            bad += !same || !idsame;
        }
        return bad ? 1 : 0;
    }
    if (argc < 2) { std::fprintf(stderr, "usage: ie-xing4-run <file.gguf> [--prompt T|--prompt-file F|--ids-file F] [--chat] [--no-think] [--gen N] [--ctx N] [--chunk N] [--cards N] [--split L] [--logits-out F] [--repeat N]\n"); return 2; }
    std::string path = argv[1], prompt, ids_file, logits_out;
    uint32_t logits_last = 0;   // --logits-last N: only the prompt's last N rows go to --logits-out (0 = every row)
    bool chat = false, think = true;
    uint32_t n_gen = 64, ctx = 32768, chunk = 512, n_cards = 2, split = 0, repeat = 1, lanes = 1, kv_tokens = 0;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--prompt") prompt = next();
        else if (a == "--prompt-file") { std::ifstream f(next()); std::stringstream ss; ss << f.rdbuf(); prompt = ss.str(); }
        else if (a == "--ids-file") ids_file = next();
        else if (a == "--chat") chat = true;
        else if (a == "--no-think") think = false;
        else if (a == "--gen") n_gen = uint32_t(std::atoi(next().c_str()));
        else if (a == "--ctx") ctx = uint32_t(std::atoi(next().c_str()));
        else if (a == "--chunk") chunk = uint32_t(std::atoi(next().c_str()));
        else if (a == "--cards") n_cards = uint32_t(std::atoi(next().c_str()));
        else if (a == "--split") split = uint32_t(std::atoi(next().c_str()));
        else if (a == "--logits-out") logits_out = next();
        else if (a == "--logits-last") logits_last = uint32_t(std::stoul(next()));
        else if (a == "--repeat") repeat = uint32_t(std::atoi(next().c_str()));
        else if (a == "--lanes") lanes = uint32_t(std::atoi(next().c_str()));
        else if (a == "--kv-tokens") kv_tokens = uint32_t(std::atoi(next().c_str()));
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.size() < n_cards || n_cards == 0) { std::fprintf(stderr, "%zu Arc GPUs found, %u asked\n", devs.size(), n_cards); return 1; }
    // two cards in ONE context with peer access (IE_XG4_P2P=0: a context each): the streams move card to card as a push
    std::unique_ptr<sycl::context> p2p_ctx;
    if (n_cards == 2 && !(std::getenv("IE_XG4_P2P") && *std::getenv("IE_XG4_P2P") == '0')) {
        std::vector<sycl::device> pg;
        if (ie::gpu_p2p_shared_context("Arc", pg, p2p_ctx) && pg.size() >= 2) { devs = {pg[0], pg[1]}; std::fprintf(stderr, "[xing4] the two cards share a context: card-to-card pushes\n"); }
        else p2p_ctx.reset();
    }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (uint32_t i = 0; i < n_cards; ++i) {
        const char* pv = std::getenv("IE_XG4_PROFILE");
        const sycl::context cx = p2p_ctx ? *p2p_ctx : sycl::context(devs[i]);
        if (pv && *pv == '1') queues.push_back(std::make_unique<sycl::queue>(cx, devs[i], sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}));
        else queues.push_back(std::make_unique<sycl::queue>(cx, devs[i], sycl::property_list{sycl::property::queue::in_order{}}));
        qs.push_back(queues.back().get());
    }
    ie::Xing4Model m;
    ie::Xing4Options o; o.max_ctx = ctx; o.max_tokens = chunk; o.split = split; o.max_lanes = std::max<uint32_t>(lanes, 16); o.kv_tokens = kv_tokens ? kv_tokens : std::max<uint32_t>(ctx, 262144);
    const auto t_load = std::chrono::steady_clock::now();
    if (auto e = m.load(path, qs, o); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    std::fprintf(stderr, "loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count());
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(m.gguf()); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    std::vector<int32_t> ids;
    if (!ids_file.empty()) {
        std::ifstream f(ids_file, std::ios::binary); int32_t v;
        while (f.read(reinterpret_cast<char*>(&v), 4)) ids.push_back(v);
    } else {
        std::string text = prompt.empty() ? std::string("Write a short poem about the sea.") : prompt;
        if (chat) text = "<_system><_user>" + text + "<_bot>" + (think ? "<think>\n" : "</think>");
        ids = tok.encode(text, true);
    }
    if (ids.empty()) { std::fprintf(stderr, "empty prompt\n"); return 2; }
    const uint32_t V = m.config().vocab, T = uint32_t(ids.size());
    std::printf("prompt: %u tokens\n", T);
    if (lanes > 1) {                       // N lanes read the prompt one after the other, then decode together
        std::vector<int32_t> next(lanes);
        std::vector<float> lg(V);
        const auto tp = std::chrono::steady_clock::now();
        for (uint32_t l = 0; l < lanes; ++l) {
            std::vector<ie::Xing4Row> rows(T); for (uint32_t t = 0; t < T; ++t) rows[t] = {ids[t], l};
            const uint32_t lr = T - 1;
            if (auto e = m.forward_rows(rows.data(), T, lg.data(), &lr, 1); !e.empty()) { std::fprintf(stderr, "lane %u prompt: %s\n", l, e.c_str()); return 1; }
            next[l] = int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin());
        }
        std::printf("%u lanes: prompts read in %.2f s\n", lanes, std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
        std::vector<std::vector<int32_t>> out(lanes);
        std::vector<float> all(size_t(lanes) * V);
        std::vector<uint32_t> lrs(lanes); for (uint32_t l = 0; l < lanes; ++l) lrs[l] = l;
        const auto t1 = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < n_gen; ++i) {
            std::vector<ie::Xing4Row> rows(lanes);
            for (uint32_t l = 0; l < lanes; ++l) { out[l].push_back(next[l]); rows[l] = {next[l], l}; }
            if (auto e = m.forward_rows(rows.data(), lanes, all.data(), lrs.data(), lanes); !e.empty()) { std::fprintf(stderr, "batch decode: %s\n", e.c_str()); return 1; }
            for (uint32_t l = 0; l < lanes; ++l) { const float* r = all.data() + size_t(l) * V; next[l] = int32_t(std::max_element(r, r + V) - r); }
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        uint32_t same = 0; for (uint32_t l = 1; l < lanes; ++l) same += out[l] == out[0];
        std::printf("%u lanes x %u tokens in %.3f s = %.1f tok/s together, %.2f tok/s a lane; lanes identical to lane 0: %u / %u\n",
                    lanes, n_gen, dt, lanes * n_gen / dt, n_gen / dt, same, lanes - 1);
        std::fputs(m.profile_report().c_str(), stderr);
        std::printf("---\n%s\n---\n", tok.decode(out[0], false).c_str());
        return 0;
    }
    for (uint32_t rep = 0; rep < repeat; ++rep) {
        m.rewind(0);
        const uint32_t R = logits_out.empty() ? 1 : (logits_last && logits_last < T ? logits_last : T);
        std::vector<float> lg(size_t(R) * V);
        const auto t0 = std::chrono::steady_clock::now();
        if (auto e = m.forward(ids.data(), T, lg.data(), R); !e.empty()) { std::fprintf(stderr, "forward: %s\n", e.c_str()); return 1; }
        const double pp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("prompt read: %u tokens in %.3f s = %.1f tok/s\n", T, pp, T / pp);
        if (!logits_out.empty() && rep == 0) {
            std::ofstream f(logits_out, std::ios::binary); f.write(reinterpret_cast<const char*>(lg.data()), std::streamsize(lg.size() * 4));
        }
        const float* last = lg.data() + size_t(R - 1) * V;
        int32_t next = int32_t(std::max_element(last, last + V) - last);
        std::vector<int32_t> out;
        std::vector<float> one(V); uint32_t n_fw = 0;
        std::fputs(m.profile_report().c_str(), stderr);
        const auto t1 = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < n_gen; ++i) {
            out.push_back(next);
            if (next == tok.eos_token_id()) break;
            if (auto e = m.forward(&next, 1, one.data(), 1); !e.empty()) { std::fprintf(stderr, "decode: %s\n", e.c_str()); return 1; }
            ++n_fw;
            next = int32_t(std::max_element(one.begin(), one.end()) - one.begin());
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        if (n_gen) std::fprintf(stderr, "decode %s", m.profile_report().c_str());
        if (n_gen) std::printf("decode: %zu tokens in %.3f s = %.2f tok/s\n", out.size(), dt, n_fw / dt);
        if (rep == 0 && n_gen) std::printf("---\n%s\n---\n", tok.decode(out, false).c_str());
    }
    return 0;
}
