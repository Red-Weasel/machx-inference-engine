// tools/llama_dump.cpp — runs llama.cpp on a prompt and dumps per-layer
// residual stream activations as fp32 binary files. Used as the reference
// for diffing against our engine's `ie-debug` dump.
//
// Build: see tools/CMakeLists.txt — links against /home/weezy/llama.cpp/build-vk/
//
// File naming matches our engine (slots scale with --n-layers N, default 40):
//   <prefix>_L00.bin  = post-embedding residual            (T x H fp32)
//   <prefix>_L01..L<N>.bin = residual after layer 0..N-1
//   <prefix>_L<N+1>.bin  = final-norm output
// Each file has a sidecar `.meta` with text "T H".
//
// The naming inside llama.cpp's graph (per qwen35moe.cpp + llama-context.cpp):
//   "model.input_embed" (il=-1)            → our L00
//      (NOTE: the fork's qwen3.cpp graph does NOT emit this — for qwen3
//       models L00 is absent and diff_layers.sh skips it with a warning)
//   "l_out-<il>" (il=0..N-1)               → our L01..L<N>
//   "result_norm" (il=-1)                  → our L<N+1>

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct DumpCtx {
    std::string prefix;
    int         saved    = 0;
    int         n_layers = 40;   // --n-layers; result_norm → slot n_layers+1
};

// Write a tensor as raw fp32 binary plus sidecar meta.
// Tensor data is on the device; for simple cases it can be read directly.
void write_tensor(const std::string& path_prefix, int slot, ggml_tensor* t) {
    if (!t) return;
    // Get tensor data on host. Use ggml_backend_tensor_get to copy from device.
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);

    // Convert to fp32 if needed (most layer outputs are fp32 already).
    std::vector<float> ff;
    const enum ggml_type tt = t->type;
    const size_t n_elem = ggml_nelements(t);
    if (tt == GGML_TYPE_F32) {
        ff.assign(reinterpret_cast<float*>(raw.data()),
                  reinterpret_cast<float*>(raw.data()) + n_elem);
    } else if (tt == GGML_TYPE_F16) {
        ff.resize(n_elem);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < n_elem; ++i) {
            // half->float via the system fp16 helper
            ff[i] = ggml_fp16_to_fp32(h[i]);
        }
    } else if (tt == GGML_TYPE_BF16) {
        ff.resize(n_elem);
        const uint16_t* b = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < n_elem; ++i) {
            ff[i] = ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(&b[i]));
        }
    } else {
        std::fprintf(stderr, "  [dump] skipping tensor type %d (slot L%02d)\n", int(tt), slot);
        return;
    }

    char path[1024];
    std::snprintf(path, sizeof(path), "%s_L%02d.bin", path_prefix.c_str(), slot);
    if (FILE* fp = std::fopen(path, "wb")) {
        std::fwrite(ff.data(), sizeof(float), ff.size(), fp);
        std::fclose(fp);
    }
    std::snprintf(path, sizeof(path), "%s_L%02d.meta", path_prefix.c_str(), slot);
    if (FILE* fp = std::fopen(path, "w")) {
        // Tensor shape: ne[0] = innermost. For our [T, H] tensors usually
        // ne[0] = H, ne[1] = T. Write "T H" (T x H).
        const int64_t T_dim = t->ne[1] > 0 ? t->ne[1] : 1;
        const int64_t H_dim = t->ne[0];
        std::fprintf(fp, "%lld %lld\n",
                     (long long)T_dim, (long long)H_dim);
        std::fclose(fp);
    }
}

bool eval_cb(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* ctx = static_cast<DumpCtx*>(user_data);
    const char* name = t->name;
    if (!name || !*name) return ask ? false : false;

    int slot = -1;
    char tag = 'L';   // 'L' = residual stream, 'A' = attn block output
    if (std::strcmp(name, "model.input_embed") == 0) {
        slot = 0;
    } else if (std::strncmp(name, "l_out-", 6) == 0) {
        const int il = std::atoi(name + 6);
        if (il >= 0 && il < ctx->n_layers) slot = il + 1;
    } else if (std::strncmp(name, "linear_attn_out-", 16) == 0) {
        // DeltaNet block output (post-out_proj, before residual)
        const int il = std::atoi(name + 16);
        if (il >= 0 && il < ctx->n_layers) { slot = il; tag = 'A'; }
    } else if (std::strncmp(name, "attn_output-", 12) == 0) {
        // Full-attn block output (post-out_proj, before residual)
        const int il = std::atoi(name + 12);
        if (il >= 0 && il < ctx->n_layers) { slot = il; tag = 'A'; }
    } else if (std::strcmp(name, "result_norm") == 0) {
        slot = ctx->n_layers + 1;
    }
    if (slot < 0) return ask ? false : false;

    if (ask) return true;            // yes, please give me the data afterwards
    char p[1024];
    if (tag == 'A') std::snprintf(p, sizeof(p), "%s_A", ctx->prefix.c_str());
    else            std::snprintf(p, sizeof(p), "%s_L", ctx->prefix.c_str());

    // write_tensor builds path "<prefix>_L<NN>.bin"; we pre-bake the prefix so it
    // becomes "<base>_A_L<NN>.bin" when tag='A'. Easier: add a small dispatch.
    // Reuse write_tensor by passing the pre-tagged prefix and slot.
    {
        // local reimplementation to keep filename pattern <base>_<tag><NN>.bin:
        const size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        std::vector<float> ff;
        const enum ggml_type tt = t->type;
        const size_t n_elem = ggml_nelements(t);
        if (tt == GGML_TYPE_F32) {
            ff.assign(reinterpret_cast<float*>(raw.data()),
                      reinterpret_cast<float*>(raw.data()) + n_elem);
        } else if (tt == GGML_TYPE_F16) {
            ff.resize(n_elem);
            const uint16_t* h = reinterpret_cast<const uint16_t*>(raw.data());
            for (size_t i = 0; i < n_elem; ++i) ff[i] = ggml_fp16_to_fp32(h[i]);
        } else {
            return true;
        }
        char path[1100];
        std::snprintf(path, sizeof(path), "%s_%c%02d.bin", ctx->prefix.c_str(), tag, slot);
        if (FILE* fp = std::fopen(path, "wb")) {
            std::fwrite(ff.data(), sizeof(float), ff.size(), fp);
            std::fclose(fp);
        }
        std::snprintf(path, sizeof(path), "%s_%c%02d.meta", ctx->prefix.c_str(), tag, slot);
        if (FILE* fp = std::fopen(path, "w")) {
            const int64_t T_dim = t->ne[1] > 0 ? t->ne[1] : 1;
            const int64_t H_dim = t->ne[0];
            std::fprintf(fp, "%lld %lld\n", (long long)T_dim, (long long)H_dim);
            std::fclose(fp);
        }
    }
    ++ctx->saved;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt = "Hello, world";
    std::string dump_prefix;
    int ngl = 99;       // all layers on GPU
    int n_layers = 40;  // --n-layers: result_norm maps to slot n_layers+1
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "-m"      && i + 1 < argc) model_path  = argv[++i];
        else if (a == "-p"      && i + 1 < argc) prompt      = argv[++i];
        else if (a == "--dump"  && i + 1 < argc) dump_prefix = argv[++i];
        else if (a == "-ngl"    && i + 1 < argc) ngl         = std::atoi(argv[++i]);
        else if (a == "--n-layers" && i + 1 < argc) n_layers = std::atoi(argv[++i]);
    }
    if (model_path.empty() || dump_prefix.empty() || n_layers <= 0) {
        std::fprintf(stderr,
                     "usage: %s -m <gguf> --dump <prefix> [-p \"prompt\"] [-ngl N] [--n-layers N]\n"
                     "  writes <prefix>_L00..L<n_layers+1>.bin (one per layer's residual;\n"
                     "  default --n-layers 40 → L00..L41)\n",
                     argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 2; }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true) < 0) {
        std::fprintf(stderr, "tokenize failed\n");
        return 3;
    }

    std::printf("prompt='%s' tokens=", prompt.c_str());
    for (auto t : tokens) std::printf("%d ", int(t));
    std::printf("\n");

    DumpCtx dctx{dump_prefix, 0, n_layers};

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 256;
    cparams.n_batch   = 256;
    cparams.cb_eval   = eval_cb;
    cparams.cb_eval_user_data = &dctx;
    cparams.no_perf   = true;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "context init failed\n"); return 4; }

    // Prefill the prompt — the eval callback runs during decode().
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "decode failed\n");
        return 5;
    }

    // Last-token logits → argmax + top-5 for sanity / greedy-parity evidence.
    const float* logits = llama_get_logits_ith(ctx, -1);
    if (logits) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<int> idx(n_vocab);
        for (int i = 0; i < n_vocab; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        char buf[128];
        int n = llama_token_to_piece(vocab, idx[0], buf, sizeof(buf), 0, true);
        std::printf("argmax: id=%d logit=%.2f piece='%.*s'\n",
                    idx[0], double(logits[idx[0]]), n > 0 ? n : 0, buf);
        for (int r = 0; r < 5; ++r) {
            n = llama_token_to_piece(vocab, idx[r], buf, sizeof(buf), 0, true);
            std::printf("  top%d id=%-7d logit=%8.3f '%.*s'\n", r + 1, idx[r],
                        double(logits[idx[r]]), n > 0 ? n : 0, buf);
        }
    }

    std::printf("dumped %d slots to %s_LNN.bin\n", dctx.saved, dump_prefix.c_str());

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
