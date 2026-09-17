// tools/qwen4exp_bvs_test.cpp — batched-vs-serial forward equivalence probe.
//
// The MTP spec gate is lossless ONLY if forward(T=K+1) reproduces the exact
// logits of K+1 sequential forward(T=1) calls over the same tokens (the
// verify batch IS the reference chain). This probe prefixes a prompt, runs a
// serial greedy lane, rewinds via snapshot/restore, replays the same tokens
// as ONE batch, and diffs logits row-by-row: first divergent row + max|d|
// localize the guilty kernel family (row 0 diverging = batched kernels,
// later rows = state advance).
//
// usage: ie-qwen4exp-bvs <model.gguf> [-n N] [gpu]
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"
#include "ie/tokenizer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ie;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what);
    if (!ok) ++g_fail;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [-n N] [gpu]\n", argv[0]); return 2; }
    uint32_t N = 8, ordinal = 0, depth = 0;
    bool verify_mode = false;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) N = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-c") && i + 1 < argc) depth = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-v")) verify_mode = true;
        else ordinal = uint32_t(std::atoi(argv[i]));
    }

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    DeviceAllocator alloc;
    if (auto e = alloc.init("B70", ordinal); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    Qwen4ExpModel m;
    if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    if (auto e = m.init_runtime(8192, 1024); !e.empty()) { std::fprintf(stderr, "rt: %s\n", e.c_str()); return 1; }
    sycl::queue& q = m.queue();
    const uint32_t V = cfg.vocab;

    const std::string prompt = "<|im_start|>user\nExplain why the sky is blue in one short paragraph.<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int32_t> ids = tok.encode(prompt, true);
    if (depth > uint32_t(ids.size())) {
        // -c: pad the prompt with synthetic ids to the requested depth so the
        // probe runs at QSA-active positions (past indexer_top_k + 3).
        const uint32_t base = uint32_t(ids.size());
        ids.resize(depth);
        for (uint32_t i = base; i < depth; ++i)
            ids[i] = int32_t(2000 + (uint64_t(i) * 97) % 200000);
    }
    const uint32_t P0 = uint32_t(ids.size());
    std::printf("\n\033[1mFlash-Next batched-vs-serial probe\033[0m  N=%u, prompt %u tok\n\n", N, P0);

    m.reset_state();
    {
        uint32_t pos = 0;
        while (pos < P0) {
            const uint32_t n = std::min(1024u, P0 - pos);
            if (auto e = m.forward(ids.data() + pos, n, pos, nullptr); !e.empty()) {
                std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1;
            }
            pos += n;
        }
    }
    std::vector<sycl::half> ltmp(V);
    auto argmax_cur = [&]() {
        q.memcpy(ltmp.data(), m.logits(), uint64_t(V) * 2).wait();
        float mx = -1e30f; int32_t am = 0;
        for (uint32_t v = 0; v < V; ++v)
            if (float(ltmp[v]) > mx) { mx = float(ltmp[v]); am = int32_t(v); }
        return am;
    };
    int32_t t0 = argmax_cur();
    if (auto e = m.snapshot(P0); !e.empty()) { std::fprintf(stderr, "snap: %s\n", e.c_str()); return 1; }

    // ---- serial lane: N x forward(T=1), keep every logits row -------------
    std::vector<int32_t> chain;                     // tokens consumed
    std::vector<std::vector<sycl::half>> ser(N);    // logits after each step
    int32_t t = t0;
    for (uint32_t s = 0; s < N; ++s) {
        chain.push_back(t);
        if (auto e = m.forward(&t, 1, P0 + s, nullptr); !e.empty()) {
            std::fprintf(stderr, "serial fwd: %s\n", e.c_str()); return 1;
        }
        ser[s].resize(V);
        q.memcpy(ser[s].data(), m.logits(), uint64_t(V) * 2).wait();
        float mx = -1e30f; int32_t am = 0;
        for (uint32_t v = 0; v < V; ++v)
            if (float(ser[s][v]) > mx) { mx = float(ser[s][v]); am = int32_t(v); }
        t = am;
    }

    // ---- batched lane: rewind, replay chain as ONE forward(T=N) -----------
    if (auto e = m.restore(); !e.empty()) { std::fprintf(stderr, "restore: %s\n", e.c_str()); return 1; }
    if (verify_mode) { m.set_spec_verify(true); std::printf("  (spec-verify mode ON for batched lane)\n"); }
    if (auto e = m.forward(chain.data(), N, P0, nullptr); !e.empty()) {
        std::fprintf(stderr, "batched fwd: %s\n", e.c_str()); return 1;
    }
    auto* d_rows = static_cast<sycl::half*>(alloc.malloc(uint64_t(N) * V * 2));
    if (auto e = m.logits_rows(N, d_rows); !e.empty()) { std::fprintf(stderr, "rows: %s\n", e.c_str()); return 1; }
    std::vector<sycl::half> rows(uint64_t(N) * V);
    q.memcpy(rows.data(), d_rows, rows.size() * 2).wait();

    // ---- diff ------------------------------------------------------------
    int first_bad = -1;
    for (uint32_t s = 0; s < N; ++s) {
        const sycl::half* br = rows.data() + uint64_t(s) * V;
        uint32_t nd = 0; float md = 0;
        float mx = -1e30f; int32_t am = 0;
        for (uint32_t v = 0; v < V; ++v) {
            const float b = float(br[v]), r = float(ser[s][v]);
            if (b != r) ++nd;
            md = std::max(md, std::abs(b - r));
            if (b > mx) { mx = b; am = int32_t(v); }
        }
        float rmx = -1e30f; int32_t ram = 0;
        for (uint32_t v = 0; v < V; ++v)
            if (float(ser[s][v]) > rmx) { rmx = float(ser[s][v]); ram = int32_t(v); }
        std::printf("  row %2u: ndiff %6u/%u  max|d| %-10g argmax %s (b=%d s=%d)\n",
                    s, nd, V, md, am == ram ? "match" : "\033[31mFLIP\033[0m", am, ram);
        if (nd && first_bad < 0) first_bad = int(s);
    }
    if (first_bad >= 0)
        std::printf("\n  first divergent row: %d  (row 0 = batched kernels themselves;"
                    " row >0 = recurrent-state advance)\n", first_bad);
    check(first_bad < 0, "batched forward(T=N) bit-matches N x forward(T=1)");
    alloc.free(d_rows);
    std::printf("\n\033[1m%s\033[0m\n\n", g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m");
    return g_fail ? 1 : 0;
}
