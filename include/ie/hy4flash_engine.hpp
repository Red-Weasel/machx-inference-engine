// include/ie/hy4flash_engine.hpp — Tencent Hy4-flash (kHy4Flash) in the engine: the bundle Engine holds and the template
// renderer. src/engine/hy4flash_engine.cpp. The runtime is Hy4fForward (two cards, 8-bit experts and 8-bit big
// non-expert matrices: scripts/hy4flash_oracle/QUANTIZATION_NOTES.md); the model is a DIRECTORY (hy4flash_ie.json).
#pragma once
#include "ie/hy4flash.hpp"
#include "ie/hy4flash_forward.hpp"
#include "ie/hy4flash_vision.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ie {

struct Hy4FlashBundle {
    std::vector<std::unique_ptr<sycl::queue>> queues;      // declared first: the forward below frees its card memory on them
    Hy4FlashModel model;
    Hy4fForward   fw;
    Hy4fVision    vis;                                     // the image tower, resident on the first card (allocated before the forward sizes its expert slots)
    bool          vis_lazy = false;   // the tower is loaded at the first image (its VRAM holds expert slots until then)
    bool          vis_ready = false; std::string vis_error; uint32_t image_tokens = 0;   // image_tokens: the most rows one image takes
    Tokenizer     tok;
    std::vector<int32_t> live;                             // the tokens the caches hold (the next request reuses their common prefix)
    struct ImgSpan { uint32_t pos0 = 0, n = 0; uint64_t hash = 0; };
    std::vector<ImgSpan> live_images;                      // the images inside `live` (their rows share one token id: the hash tells them apart)
    std::vector<int32_t> eos;
    // token id -> the text the STREAM carries for it: the model's own tags as the server's (GLM's) spelling, see the .cpp
    std::vector<std::pair<int32_t, std::string>> stream_text;
    uint32_t ctx = 0, chunk = 512, big = 0, big_min = 2048;   // big: a long prompt's chunks in VRAM the expert slots lend (more than big_min rows left)
    bool reuse = true;
    // The checkpoint's drafter (DFly2): each decode step first drafts the tokens after the one just sampled, one pass of
    // the model verifies the block, and the drafts the sampler agrees with are kept. draft_conf: a draft is verified
    // while the drafter's own confidence head gives the drafts so far this chance of surviving together.
    bool draft = false; float draft_conf = 0.6f;
};

// The checkpoint's chat template (chat_template.jinja), rendered natively with add_generation_prompt on.
// effort: "no_think" | "high" | "max". content (optional): the byte ranges of the request's own text (known_bugs 45).
// A turn's images stand where the server's kChatImageMarker sits in its content when the markers match the images, else
// all in front of its text; each is the template's <start><pad><end> triple (one pad: the engine expands it to the image's rows).
std::string hy4f_render_prompt(std::span<const ChatTurn> turns, std::string_view tools_json, std::string_view effort, std::string& err,
                               std::vector<std::pair<size_t, size_t>>* content = nullptr);
// The model's reply tags (<...:opensource>) as GLM's, which the engine's GLM parser and the server's stream rules read.
std::string hy4f_translate_reply(std::string_view raw);

}  // namespace ie
