// tools/ds41_ppl.cpp -- ie-ds41-ppl: DeepSeek-V4.1-Flash perplexity on real text, the measurement handoff decision 1
// (IE_DS41_PREFILL_XMX) asked for (docs/deepseek41/83).
//   ie-ds41-ppl <model> <text_file> [--tokens N] [--chunk C] [--ranking FILE] [--tables DIR] [--out FILE] [--compare FILE]
// The text is tokenized raw (no chat template), cut to N tokens (a multiple of C), and fed in chunks of C through the
// EXACT path (bounded replay off, full [C, vocab] logits), sequentially. For every position i < N - 1:
//     nll_i = logsumexp(logits_i) - logits_i[ids[i+1]]        PPL = exp(mean nll)
// --out writes (nll float32, argmax int32) per position, so two runs on the same tokens compare PAIRWISE: --compare
// reads another run's file and prints the mean NLL difference with its standard error over chunk-level blocks (the
// positions are not independent) and the top-1 agreement -- a route change's quality delta, not two noisy PPLs.
#include "ie/deepseek41_forward.hpp"
#include "ie/expert_stream.hpp"
#include "ie/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {
bool exists(const std::string& p) { return std::ifstream(p).good(); }
struct Rec { float nll; int32_t top; };
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: ie-ds41-ppl <model> <text_file> [--tokens N] [--chunk C] [--ranking FILE] [--tables DIR] [--out FILE] [--compare FILE]\n"); return 2; }
    const std::string model = argv[1], text_file = argv[2];
    std::string ranking = model + "/ie_ranking_heldout.txt", tables = model, out_path, cmp_path;
    uint32_t N = 16384, C = 1024;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i]; auto val = [&]() -> std::string { if (i + 1 >= argc) std::exit(2); return argv[++i]; };
        if (a == "--tokens") N = uint32_t(std::atol(val().c_str())); else if (a == "--chunk") C = uint32_t(std::atol(val().c_str()));
        else if (a == "--ranking") ranking = val(); else if (a == "--tables") tables = val();
        else if (a == "--out") out_path = val(); else if (a == "--compare") cmp_path = val();
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (C < 16 || C % 2) { std::fprintf(stderr, "--chunk must be even and >= 16 (a continuation chunk)\n"); return 2; }
    setenv("IE_DS41_CONT", "1", 0);                     // every chunk after the first is a continuation (the forward reads it once, at first use)

    ie::DeepSeek41Model m;
    if (auto e = m.load(model); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(tables); !e.empty()) { std::fprintf(stderr, "engram tables: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(model + "/tokenizer.json", model + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    std::ifstream tf(text_file, std::ios::binary);
    if (!tf) { std::fprintf(stderr, "cannot read %s\n", text_file.c_str()); return 2; }
    const std::string text((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
    auto ids = tok.encode(text, /*allow_special=*/false);
    N = std::min<uint32_t>(N, uint32_t(ids.size())) / C * C;
    if (N < C) { std::fprintf(stderr, "the text has %zu tokens, fewer than one chunk of %u\n", ids.size(), C); return 2; }
    ids.resize(N);

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) { queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); qs.push_back(queues.back().get()); }
    std::vector<std::vector<uint32_t>> rank;
    if (exists(ranking)) if (auto e = ie::ds4_expert_priority_read_layers(ranking, m.config().n_routed_experts, m.config().n_layers, rank); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }
    ie::Ds41Forward fwd; ie::Ds41Forward::ResidentOptions opt; opt.max_tokens = N + 64; opt.max_forward_tokens = C;
    if (auto e = fwd.init_resident(qs, m, tb, rank, opt); !e.empty()) { std::fprintf(stderr, "init_resident: %s\n", e.c_str()); return 1; }
    fwd.set_logits_last_only(false);
    fwd.set_bounded_replay(false);                      // the exact path: every position's logits are the full model's
    const uint32_t V = m.config().vocab_size;
    const char* xmx = std::getenv("IE_DS41_PREFILL_XMX");
    std::printf("ds41 ppl: %u tokens of %s in chunks of %u, exact path, IE_DS41_PREFILL_XMX=%s\n", N, text_file.c_str(), C, xmx ? xmx : "(unset)");

    std::vector<Rec> rec; rec.reserve(N);
    std::vector<float> lg;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t off = 0; off < N; off += C) {
        if (auto e = fwd.forward(ids.data() + off, C, off, lg); !e.empty()) { std::fprintf(stderr, "forward at %u: %s\n", off, e.c_str()); return 1; }
        if (lg.size() != size_t(C) * V) { std::fprintf(stderr, "forward at %u returned %zu logits, expected %zu\n", off, lg.size(), size_t(C) * V); return 1; }
        double chunk_nll = 0; uint32_t n_chunk = 0;
        for (uint32_t r = 0; r < C && off + r + 1 < N; ++r) {
            const float* row = lg.data() + size_t(r) * V;
            float mx = row[0]; int32_t top = 0;
            for (uint32_t v = 1; v < V; ++v) if (row[v] > mx) { mx = row[v]; top = int32_t(v); }
            double z = 0; for (uint32_t v = 0; v < V; ++v) z += std::exp(double(row[v]) - double(mx));   // stable logsumexp
            const double nll = double(mx) + std::log(z) - double(row[ids[off + r + 1]]);
            rec.push_back({float(nll), top}); chunk_nll += nll; ++n_chunk;
        }
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  chunk at %6u: mean nll %.4f over %u | running ppl %.4f | %.0f tok/s\n", off, chunk_nll / std::max<uint32_t>(1, n_chunk), n_chunk,
                    [&] { double s = 0; for (const auto& x : rec) s += x.nll; return std::exp(s / double(rec.size())); }(), double(off + C) / el);
        std::fflush(stdout);
    }
    double s = 0; for (const auto& x : rec) s += x.nll;
    std::printf("PPL %.5f  mean nll %.6f over %zu predictions\n", std::exp(s / double(rec.size())), s / double(rec.size()), rec.size());
    if (!out_path.empty()) {
        std::ofstream of(out_path, std::ios::binary); of.write(reinterpret_cast<const char*>(rec.data()), std::streamsize(rec.size() * sizeof(Rec)));
        std::printf("wrote %zu (nll, argmax) records -> %s\n", rec.size(), out_path.c_str());
    }
    if (!cmp_path.empty()) {
        std::ifstream cf(cmp_path, std::ios::binary);
        std::vector<Rec> other(rec.size());
        if (!cf || !cf.read(reinterpret_cast<char*>(other.data()), std::streamsize(other.size() * sizeof(Rec)))) { std::fprintf(stderr, "compare: cannot read %zu records from %s\n", rec.size(), cmp_path.c_str()); return 1; }
        // blocks of one chunk: the per-block mean difference is the unit whose spread gives the standard error
        std::vector<double> blk; double tot = 0; uint32_t agree = 0;
        for (size_t b0 = 0; b0 < rec.size(); b0 += C) {
            double d = 0; size_t n = 0;
            for (size_t i = b0; i < std::min(rec.size(), b0 + C); ++i, ++n) { d += double(rec[i].nll) - double(other[i].nll); agree += rec[i].top == other[i].top; }
            blk.push_back(d / double(n)); tot += d;
        }
        const double mean = tot / double(rec.size());
        double bm = 0; for (double v : blk) bm += v; bm /= double(blk.size());
        double var = 0; for (double v : blk) var += (v - bm) * (v - bm); var /= double(std::max<size_t>(1, blk.size() - 1));
        std::printf("vs %s: mean nll difference (this - that) %+.6f, block SE %.6f over %zu blocks (t = %.2f); top-1 agreement %.2f%%\n",
                    cmp_path.c_str(), mean, std::sqrt(var / double(blk.size())), blk.size(), bm / std::max(1e-12, std::sqrt(var / double(blk.size()))),
                    100.0 * double(agree) / double(rec.size()));
    }
    fwd.free_resident();
    return 0;
}
