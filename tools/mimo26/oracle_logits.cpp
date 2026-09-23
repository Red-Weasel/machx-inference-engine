// tools/mimo26/oracle_logits.cpp -- the P2 gate's per-position oracle (docs/mimo26/00_PORT_PLAN.md): llama.cpp's own
// forward on a GGUF, logits at EVERY position of each given token sequence, teacher forced. Not part of the engine
// build: compiled against the upstream checkout by tools/mimo26/build_oracle_logits.sh.
//   oracle_logits <model.gguf> <ids.txt> <out.bin> [threads] [--greedy N greedy.txt]
// --greedy: after scoring, continue each sequence N tokens greedily (argmax, one llama_decode per token) and write the
// generated ids, one line per sequence, to greedy.txt (P2 clause (a)).
// ids.txt: one sequence per line, token ids separated by spaces or commas (brackets ignored).
// out.bin: "MLOG", n_seq (u32), vocab (u32), then per sequence: n (u32), ids (n x i32), logits (n x vocab f32).
// ie-mimo26-score writes the same format; tools/mimo26/compare_logits.py compares two such files.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <model.gguf> <ids.txt> <out.bin> [threads]\n", argv[0]); return 2; }
    const int threads = argc > 4 && argv[4][0] != '-' ? std::atoi(argv[4]) : 20;
    int greedy_n = 0; std::string greedy_path;
    for (int i = 4; i < argc; ++i) if (std::string(argv[i]) == "--greedy" && i + 2 < argc) { greedy_n = std::atoi(argv[i + 1]); greedy_path = argv[i + 2]; }
    std::vector<std::vector<llama_token>> seqs;
    {
        std::ifstream in(argv[2]);
        std::string line;
        while (std::getline(in, line)) {
            for (char& c : line) if (c == ',' || c == '[' || c == ']') c = ' ';
            std::istringstream ss(line);
            std::vector<llama_token> s; long v;
            while (ss >> v) s.push_back(llama_token(v));
            if (!s.empty()) seqs.push_back(std::move(s));
        }
    }
    if (seqs.empty()) { std::fprintf(stderr, "no sequences in %s\n", argv[2]); return 2; }
    size_t max_n = 0; for (const auto& s : seqs) max_n = std::max(max_n, s.size());

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = uint32_t(std::max<size_t>(max_n + 16 + size_t(greedy_n), 512));
    cp.n_batch = cp.n_ubatch = uint32_t(std::max<size_t>(max_n, 512));
    cp.n_threads = cp.n_threads_batch = threads;
    cp.no_perf = true;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "cannot create a context\n"); return 1; }
    const int32_t V = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::FILE* g = greedy_n > 0 ? std::fopen(greedy_path.c_str(), "w") : nullptr;
    std::FILE* f = std::fopen(argv[3], "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    const uint32_t hdr[2] = {uint32_t(seqs.size()), uint32_t(V)};
    std::fwrite("MLOG", 1, 4, f); std::fwrite(hdr, 4, 2, f);
    for (size_t si = 0; si < seqs.size(); ++si) {
        const auto& s = seqs[si];
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch b = llama_batch_init(int32_t(s.size()), 0, 1);
        for (size_t i = 0; i < s.size(); ++i) {
            b.token[i] = s[i]; b.pos[i] = llama_pos(i); b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        b.n_tokens = int32_t(s.size());
        if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed on sequence %zu\n", si); return 1; }
        const uint32_t n = uint32_t(s.size());
        std::fwrite(&n, 4, 1, f);
        std::fwrite(s.data(), 4, n, f);
        for (uint32_t i = 0; i < n; ++i) std::fwrite(llama_get_logits_ith(ctx, int32_t(i)), 4, size_t(V), f);
        llama_batch_free(b);
        if (g) {
            const float* lg = llama_get_logits_ith(ctx, int32_t(n - 1));
            llama_pos pos = llama_pos(n);
            for (int k = 0; k < greedy_n; ++k) {
                int32_t best = 0;
                for (int32_t v = 1; v < V; ++v) if (lg[v] > lg[best]) best = v;
                std::fprintf(g, "%s%d", k ? " " : "", best);
                if (k + 1 == greedy_n) break;
                llama_batch one = llama_batch_init(1, 0, 1);
                one.token[0] = best; one.pos[0] = pos++; one.n_seq_id[0] = 1; one.seq_id[0][0] = 0; one.logits[0] = 1; one.n_tokens = 1;
                if (llama_decode(ctx, one) != 0) { std::fprintf(stderr, "greedy decode failed\n"); return 1; }
                lg = llama_get_logits_ith(ctx, 0);
                llama_batch_free(one);
            }
            std::fprintf(g, "\n"); std::fflush(g);
        }
        std::fprintf(stderr, "sequence %zu: %u tokens\n", si, n);
    }
    std::fclose(f);
    if (g) std::fclose(g);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
