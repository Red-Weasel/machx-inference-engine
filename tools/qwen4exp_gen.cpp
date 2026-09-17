// tools/qwen4exp_gen.cpp — Qwen3.8-Flash-Next greedy text generation smoke.
//
// The decisive qualitative check before the oracle parity gate: tokenize a
// real prompt (engine tokenizer, pre=qwen35), greedy-generate, decode to
// text. A structurally-broken forward (HC/PLE/MoE misbind) produces salad;
// coherent text is strong evidence the chain is substantially right.
//
// usage: ie-qwen4exp-gen <model.gguf> [-n N] [-p "prompt"] [gpu]
//        [--image FILE --mmproj mmproj.gguf]   (vision: docs/qwen4/16_vision_port.md §4)
//        [--gpus 2] [--split L]                (2-stage pipeline, ViT on GPU1;
//                                               IE_P2P=1 for the peer handoff)
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"
#include "ie/qwen4_vision.hpp"
#include "ie/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <string>
#include <vector>

using namespace ie;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [-n N] [-p prompt] [--image FILE --mmproj FILE] [--gpus N] [--split L] [gpu]\n", argv[0]); return 2; }
    const char* path = argv[1];
    uint32_t n_gen = 48, ordinal = 0, n_gpus = 1, split = 24;
    std::string prompt = "The capital of France is";
    std::string image, mmproj;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n_gen = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!std::strcmp(argv[i], "--image") && i + 1 < argc) image = argv[++i];
        else if (!std::strcmp(argv[i], "--mmproj") && i + 1 < argc) mmproj = argv[++i];
        else if (!std::strcmp(argv[i], "--gpus") && i + 1 < argc) n_gpus = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--split") && i + 1 < argc) split = uint32_t(std::atoi(argv[++i]));
        else ordinal = uint32_t(std::atoi(argv[i]));
    }
    if (n_gpus > 2) { std::fprintf(stderr, "--gpus must be 1 or 2\n"); return 1; }

    GgufReader g;
    if (auto e = g.open(path); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }

    // ---- allocators: 1-GPU, or 2 (optionally one shared P2P context) --------
    DeviceAllocator a0, a1;
    const bool two = n_gpus == 2;
    std::vector<sycl::device> p2p_gpus;
    std::unique_ptr<sycl::context> p2p_ctx;
    bool p2p = false;
    if (two) {
        p2p = std::getenv("IE_P2P") != nullptr &&
              gpu_p2p_shared_context("B70", p2p_gpus, p2p_ctx);
        if (p2p) {
            if (auto e = a0.init_with(*p2p_ctx, p2p_gpus[0]); !e.empty()) { std::fprintf(stderr, "gpu0: %s\n", e.c_str()); return 1; }
            if (auto e = a1.init_with(*p2p_ctx, p2p_gpus[1]); !e.empty()) { std::fprintf(stderr, "gpu1: %s\n", e.c_str()); return 1; }
        } else {
            if (auto e = a0.init("B70", 0); !e.empty()) { std::fprintf(stderr, "gpu0: %s\n", e.c_str()); return 1; }
            if (auto e = a1.init("B70", 1); !e.empty()) { std::fprintf(stderr, "gpu1: %s\n", e.c_str()); return 1; }
        }
    } else {
        if (auto e = a0.init("B70", ordinal); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    }
    DeviceAllocator& vis_alloc = two ? a1 : a0;   // ViT rides GPU1 in pipeline mode

    // ---- model load (stage split in pipeline mode) --------------------------
    Qwen4ExpModel A, B;
    if (two) {
        if (auto e = A.load(a0, g, cfg, 0, 0, split); !e.empty()) { std::fprintf(stderr, "A: %s\n", e.c_str()); return 1; }
        if (auto e = B.load(a1, g, cfg, 0, split, cfg.n_layers); !e.empty()) { std::fprintf(stderr, "B: %s\n", e.c_str()); return 1; }
    } else {
        if (auto e = A.load(a0, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    }

    // ---- vision: decode + encode; in pipeline mode the GPU1 ViT encode runs
    // CONCURRENTLY with stage A's runtime init (disjoint allocators/queues) ---
    std::vector<float> vis_emb;
    uint32_t gh2 = 0, gw2 = 0;             // merged grid (h/2, w/2)
    Qwen4Vision vis;
    std::future<std::string> enc_fut;
    double enc_secs = 0;
    if (!image.empty()) {
        if (mmproj.empty()) { std::fprintf(stderr, "--image needs --mmproj\n"); return 1; }
        if (auto e = vis.load(mmproj); !e.empty()) { std::fprintf(stderr, "mmproj: %s\n", e.c_str()); return 1; }
        auto img = std::make_shared<std::vector<float>>();
        uint32_t H = 0, W = 0;
        if (auto e = qwen4_load_image(image, *img, H, W); !e.empty()) { std::fprintf(stderr, "image: %s\n", e.c_str()); return 1; }
        gh2 = H / 32; gw2 = W / 32;
        enc_fut = std::async(std::launch::async, [&vis, &vis_alloc, img, H, W,
                                                  &vis_emb, &enc_secs] {
            const auto t = std::chrono::steady_clock::now();
            auto e = vis.encode_gpu(vis_alloc, img->data(), H, W, vis_emb);
            enc_secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t).count();
            return e;
        });
        std::string pads;
        for (uint32_t i = 0; i < gh2 * gw2; ++i) pads += "<|image_pad|>";
        prompt = "<|im_start|>user\n<|vision_start|>" + pads + "<|vision_end|>" +
                 prompt + "<|im_end|>\n<|im_start|>assistant\n";
        if (!two) {   // single GPU: no safe overlap target — join before init
            if (auto e = enc_fut.get(); !e.empty()) { std::fprintf(stderr, "encode: %s\n", e.c_str()); return 1; }
        }
    }

    // Stage A init overlaps the GPU1 encode in pipeline mode.
    const uint32_t max_ctx = two ? 8192 : 2051;
    if (auto e = A.init_runtime(max_ctx, 1024); !e.empty()) { std::fprintf(stderr, "A rt: %s\n", e.c_str()); return 1; }
    if (!image.empty() && two) {   // join before B's ecache sizing sees GPU1
        if (auto e = enc_fut.get(); !e.empty()) { std::fprintf(stderr, "encode: %s\n", e.c_str()); return 1; }
    }
    if (two) {
        if (auto e = B.init_runtime(max_ctx, 1024); !e.empty()) { std::fprintf(stderr, "B rt: %s\n", e.c_str()); return 1; }
    }
    if (!image.empty())
        std::fprintf(stderr, "[vision] %s -> %u tokens, GPU%u encode %.2fs%s\n",
                     image.c_str(), gh2 * gw2, two ? 1u : 0u, enc_secs,
                     two ? " (overlapped with stage-A init)" : "");

    // ---- pipeline handoff ---------------------------------------------------
    std::vector<float> wide;
    bool p2p_hand = p2p;
    if (two) {
        wide.resize(uint64_t(1024) * cfg.hc_count * cfg.hidden);
        if (p2p_hand) A.set_wide_peer(B.wide_device());
    }
    auto fwd = [&](const int32_t* toks, uint32_t T, uint32_t pos) -> std::string {
        if (!two) return A.forward(toks, T, pos, nullptr);
        if (auto e = A.forward_range(toks, T, pos, nullptr, wide.data(), nullptr); !e.empty()) return e;
        return B.forward_range(toks, T, pos, p2p_hand ? nullptr : wide.data(),
                               nullptr, nullptr, /*wide_in_device=*/p2p_hand);
    };
    Qwen4ExpModel& tail = two ? B : A;
    DeviceAllocator& tail_a = two ? a1 : a0;

    std::vector<int32_t> ids = tok.encode(prompt, true);
    std::printf("prompt (%zu tokens)%s\n---\n", ids.size(),
                image.empty() ? (": " + prompt).c_str() : " [image prompt]");

    if (!image.empty()) {
        // Splice span = the <|image_pad|> run (id 248056 per the GGUF vocab).
        const int32_t kImagePad = 248056;
        uint32_t t0 = 0;
        while (t0 < ids.size() && ids[t0] != kImagePad) ++t0;
        const uint32_t n_img = gh2 * gw2;
        if (t0 + n_img > ids.size() || ids[t0 + n_img - 1] != kImagePad) {
            std::fprintf(stderr, "image_pad run mismatch in tokenized prompt\n"); return 1;
        }
        if (auto e = A.set_vision(vis_emb.data(), t0, n_img); !e.empty()) {
            std::fprintf(stderr, "set_vision: %s\n", e.c_str()); return 1;
        }
        // M-RoPE table (get_rope_index :2098-2121): text runs linear from
        // current_pos on all streams; the image block gets T=current_pos,
        // H=current_pos+row, W=current_pos+col over the merged grid; after it
        // current_pos advances by max(gh2, gw2).
        const uint32_t n = uint32_t(ids.size());
        std::vector<int32_t> pos3(size_t(3) * n);
        int32_t cur = 0;
        for (uint32_t t = 0; t < n; ) {
            if (t == t0) {
                for (uint32_t r = 0; r < gh2; ++r)
                    for (uint32_t c = 0; c < gw2; ++c) {
                        const uint32_t i2 = t0 + r * gw2 + c;
                        pos3[i2] = cur;                      // T
                        pos3[size_t(n) + i2] = cur + int32_t(r);      // H
                        pos3[size_t(2) * n + i2] = cur + int32_t(c);  // W
                    }
                cur += int32_t(std::max(gh2, gw2));
                t += n_img;
            } else {
                pos3[t] = pos3[size_t(n) + t] = pos3[size_t(2) * n + t] = cur;
                ++cur;
                ++t;
            }
        }
        int32_t mx = 0;
        for (int32_t v : pos3) mx = std::max(mx, v);
        A.set_mrope(pos3.data(), n, mx + 1 - int32_t(n));
        if (two) B.set_mrope(pos3.data(), n, mx + 1 - int32_t(n));
        std::fprintf(stderr, "[vision] splice at [%u, %u), rope_delta %d\n",
                     t0, t0 + n_img, mx + 1 - int32_t(n));
    }

    // Chunked prefill (image prompts exceed the old 64-token guard).
    for (uint32_t off = 0; off < ids.size(); ) {
        const uint32_t n2 = std::min<uint32_t>(1024, uint32_t(ids.size()) - off);
        if (auto e = fwd(ids.data() + off, n2, off); !e.empty()) {
            std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1;
        }
        off += n2;
    }
    std::vector<sycl::half> l16(cfg.vocab);
    auto pick = [&]() -> int32_t {
        tail_a.queue().memcpy(l16.data(), tail.logits(), cfg.vocab * 2).wait();
        float mx = -1e30f; uint32_t am = 0;
        for (uint32_t i = 0; i < cfg.vocab; ++i)
            if (float(l16[i]) > mx) { mx = float(l16[i]); am = i; }
        return int32_t(am);
    };

    std::vector<int32_t> gen;
    int32_t t = pick();
    uint32_t pos = uint32_t(ids.size());
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t s = 0; s < n_gen; ++s) {
        if (t == tok.eos_token_id()) break;
        gen.push_back(t);
        std::string piece = tok.decode(std::span<const int32_t>(&t, 1), true, {});
        std::printf("%s", piece.c_str());
        std::fflush(stdout);
        if (auto e = fwd(&t, 1, pos); !e.empty()) {
            std::fprintf(stderr, "\ndecode: %s\n", e.c_str()); return 1;
        }
        t = pick();
        ++pos;
    }
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("\n---\n%zu tokens in %.1f s (%.2f tok/s, %u gpu%s%s)\n",
                gen.size(), secs, gen.size() / secs, n_gpus,
                n_gpus > 1 ? "s" : "", p2p_hand ? ", p2p" : "");
    return 0;
}
