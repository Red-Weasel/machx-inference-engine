// tools/mimo26_ppl.cpp -- ie-mimo26-ppl: MiMo-V2.6 perplexity in llama-perplexity's protocol (P2 gate, docs/mimo26/00_PORT_PLAN.md).
//   ie-mimo26-ppl <model_dir> <text_file> [--chunk C] [--chunks K] [--ubatch U] [--out FILE] [--logprobs FILE] [--static S] [--pinned P] [--stream Q] [--ranking FILE]
// --ubatch U (default C, capped at 2048): each chunk runs as forward calls of at most U rows -- the chunked prefill a
// long context takes (P3a); the context of a chunk is still C.
// The text is tokenized raw (no special-token parsing, no BOS: the model has none), cut into K non-overlapping chunks
// of C tokens; every chunk starts from an empty KV cache and the logits at positions [C/2, C-1) score the tokens
// [C/2+1, C) -- exactly what `llama-perplexity -c C` does, so the two PPLs compare directly.
// --out writes (nll float32, argmax int32) per scored position; --logprobs writes a header (C, vocab, n_chunks as
// uint32), the n_chunks * C token ids (int32), then, per scored position, the fp16 log-softmax over the vocab -- the KL / top-1 comparison against
// llama-perplexity --kl-divergence-base is tools/mimo26/compare_logprobs.py.
#include "ie/expert_stream.hpp"
#include "ie/mimo26.hpp"
#include "ie/mimo26_forward.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: ie-mimo26-ppl <model_dir> <text_file> [--chunk C] [--chunks K] [--out FILE] [--logprobs FILE] [--static S] [--pinned P] [--stream Q] [--ranking FILE]\n"); return 2; }
    const std::string model = argv[1], text_file = argv[2];
    uint32_t C = 2048, K = 0, U = 0; std::string out_path, lp_path;
    ie::Mimo26Options opt;
    std::string ranking_path;   // --ranking: a per-layer expert priority file (ie-mimo26-profile)
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i]; auto val = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--chunk") C = uint32_t(std::atol(val().c_str()));
        else if (a == "--chunks") K = uint32_t(std::atol(val().c_str()));
        else if (a == "--ubatch") U = uint32_t(std::atol(val().c_str()));
        else if (a == "--out") out_path = val();
        else if (a == "--logprobs") lp_path = val();
        else if (a == "--static") opt.n_static = uint32_t(std::atol(val().c_str()));
        else if (a == "--pinned") opt.n_pinned = uint32_t(std::atol(val().c_str()));
        else if (a == "--stream") opt.stream_slots = uint32_t(std::atol(val().c_str()));
        else if (a == "--ranking") ranking_path = val();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (C < 16 || C % 2) { std::fprintf(stderr, "--chunk must be even and >= 16\n"); return 2; }

    ie::Mimo26Model m;
    if (auto e = m.load(model); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(model + "/tokenizer.json", model + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    std::ifstream tf(text_file, std::ios::binary);
    if (!tf) { std::fprintf(stderr, "cannot read %s\n", text_file.c_str()); return 2; }
    const std::string text((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
    const std::vector<int32_t> ids = tok.encode(text, /*allow_special=*/false);
    const uint32_t n_chunks_all = uint32_t(ids.size() / C);
    const uint32_t n_chunks = K ? std::min(K, n_chunks_all) : n_chunks_all;
    if (n_chunks == 0) { std::fprintf(stderr, "the text has %zu tokens, fewer than one chunk of %u\n", ids.size(), C); return 2; }

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) { queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); qs.push_back(queues.back().get()); }

    if (!U) U = std::min<uint32_t>(C, 2048);
    U = std::min(U, C);
    opt.max_ctx = C; opt.max_tokens = U;
    ie::Mimo26Forward fwd;
    if (!ranking_path.empty())
        if (auto e = ie::ds4_expert_priority_read_layers(ranking_path, m.config().n_routed_experts, m.config().n_layers, opt.ranking); !e.empty()) {
            std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }
    if (auto e = fwd.init(qs, m, opt); !e.empty()) { std::fprintf(stderr, "init: %s\n", e.c_str()); return 1; }
    const uint32_t V = m.config().vocab_size, first = C / 2, scored = C - 1 - first;
    std::printf("mimo26 ppl: %s, %zu tokens, %u chunks of %u (%u scored per chunk) in forward calls of %u rows, %zu card(s)\n", text_file.c_str(), ids.size(), n_chunks, C, scored, U, qs.size());

    std::FILE* fo = out_path.empty() ? nullptr : std::fopen(out_path.c_str(), "wb");
    std::FILE* fl = lp_path.empty() ? nullptr : std::fopen(lp_path.c_str(), "wb");
    if (fl) { const uint32_t hdr[3] = {C, V, n_chunks}; std::fwrite(hdr, 4, 3, fl);
              std::fwrite(ids.data(), 4, size_t(n_chunks) * C, fl); }   // the tokens, so the comparison can check both sides scored the same text
    std::vector<float> logits, lse_row(V);
    std::vector<uint16_t> lp16(V);
    double nll_sum = 0; uint64_t n_scored = 0, top1 = 0;
    for (uint32_t k = 0; k < n_chunks; ++k) {
        const int32_t* chunk = ids.data() + size_t(k) * C;
        fwd.reset();
        const auto t0 = std::chrono::steady_clock::now();
        double chunk_nll = 0;
        for (uint32_t off = 0; off < C; off += U) {
        const uint32_t n = std::min(U, C - off);
        if (auto e = fwd.forward(chunk + off, n, off, logits, true); !e.empty()) { std::fprintf(stderr, "chunk %u at %u: %s\n", k, off, e.c_str()); return 1; }
        for (uint32_t j = std::max(first, off); j < off + n && j + 1 < C; ++j) {
            const float* row = logits.data() + size_t(j - off) * V;
            float mx = row[0]; uint32_t am = 0;
            for (uint32_t v = 1; v < V; ++v) if (row[v] > mx) { mx = row[v]; am = v; }
            double sum = 0; for (uint32_t v = 0; v < V; ++v) sum += std::exp(double(row[v] - mx));
            const double lse = double(mx) + std::log(sum);
            const int32_t target = chunk[j + 1];
            const double nll = lse - double(row[target]);
            chunk_nll += nll; ++n_scored; top1 += (int32_t(am) == target);
            if (fo) { const float f = float(nll); const int32_t a = int32_t(am); std::fwrite(&f, 4, 1, fo); std::fwrite(&a, 4, 1, fo); }
            if (fl) {
                for (uint32_t v = 0; v < V; ++v) { const float lp = float(double(row[v]) - lse); lp16[v] = sycl::bit_cast<uint16_t>(sycl::half(lp)); }
                std::fwrite(lp16.data(), 2, V, fl);
            }
        }
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        nll_sum += chunk_nll;
        std::printf("[%u] chunk ppl %.4f | running ppl %.4f | top-1 %.2f%% | %.1f tok/s\n", k + 1, std::exp(chunk_nll / scored), std::exp(nll_sum / double(n_scored)),
                    100.0 * double(top1) / double(n_scored), C / s);
        std::fflush(stdout);
    }
    if (fo) std::fclose(fo);
    if (fl) std::fclose(fl);
    std::printf("Final estimate: PPL = %.4f over %llu scored tokens (%u chunks of %u), top-1 %.2f%%\n", std::exp(nll_sum / double(n_scored)), (unsigned long long)n_scored, n_chunks, C,
                100.0 * double(top1) / double(n_scored));
    return 0;
}
