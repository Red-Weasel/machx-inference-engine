// include/ie/xing4_engine.hpp -- Xing4.0-29B-A4B (kXing4) in the engine: the bundle Engine holds and the template renderer.
// src/engine/xing4_engine.cpp. The runtime is Xing4Model (src/model/xing4.cpp: two cards, every weight resident, a paged
// latent cache shared by the request lanes); requests run on the shared lanes server (LanesServe) with every decoding
// lane's row in one forward per step.
#pragma once
#include "ie/tokenizer.hpp"
#include "ie/xing4.hpp"

#include <sycl/sycl.hpp>

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ie {

class LanesServe;
class Xing4LanesModel;

struct Xing4Bundle {
    std::vector<std::unique_ptr<sycl::queue>> queues;      // declared first: the model frees its card memory on them
    Xing4Model model;
    Tokenizer  tok;
    std::vector<int32_t> eos;
    std::unique_ptr<Xing4LanesModel> lanes;
    std::unique_ptr<LanesServe> serve;
    int32_t think_close = -1;
    ~Xing4Bundle();
};

// The release's chat_template.jinja with add_generation_prompt on (byte-exact against transformers' rendering).
// enable_thinking false = the prompt ends in "</think>" (no reasoning). content (optional): the byte ranges of the
// request's own text (known_bugs 45: a tag spelled in it stays text).
std::string xing4_render_prompt(std::span<const ChatTurn> turns, std::string_view tools_json, bool enable_thinking, std::string& err,
                                std::vector<std::pair<size_t, size_t>>* content = nullptr);
// The prompt's ids as transformers' tokenizer gives them (no BOS; "<_observation>" as its four text pieces).
std::vector<int32_t> xing4_encode(const Tokenizer& tok, std::string_view prompt, std::span<const std::pair<size_t, size_t>> content);
// The reply's tool-call tags in GLM's spelling (<param_key> -> <arg_key>, <param_value> -> <arg_value>), which the engine's
// GLM parser and the server's stream rules read.
std::string xing4_translate_reply(std::string_view raw);

}  // namespace ie
