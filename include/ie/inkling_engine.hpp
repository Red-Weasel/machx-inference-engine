// include/ie/inkling_engine.hpp — the Inkling-Small bundle behind the Engine (docs/inkling/00_PORT_PLAN.md P5): the runtime
// (one card or two), its queues, the GGUF's tokenizer, the chat format's marker ids and the live conversation (the ids whose
// state the runtime holds). Built and used in src/engine/inkling_engine.cpp; complete here so Engine's destructor can destroy it.
#pragma once
#include "ie/inkling.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ie {

struct InklingBundle {
    InklingConfig                cfg;
    std::unique_ptr<sycl::queue> q;       // declared before the model: destroyed after it
    std::unique_ptr<sycl::queue> q2;      // the second card (two cards), likewise
    InklingModel                 model;
    Tokenizer                    tok;
    std::vector<int32_t>         live;    // the tokens the model's state holds (prefix reuse)
    std::vector<int32_t>         stop;    // <|content_model_end_sampling|> (the EOS) and the non-model message openers
    // the chat format's markers (token ids)
    int32_t msg_model = -1, c_text = -1, c_thinking = -1, c_tool = -1, end_msg = -1;
    uint32_t chunk = 512;
    bool     reuse = true;                // --no-prompt-cache: every request reads its whole prompt
};

// The GGUF's chat template rendered natively, add_generation_prompt on; thinking off = effort level 0, thinking on with no
// effort = the template's 0.9. tests/unit/inkling_template_test.cpp checks it against the Jinja template's own output.
std::string inkling_render_prompt(std::span<const ChatTurn> turns, std::string_view tools_json, bool thinking,
                                  std::string_view effort, std::string& err);
// A raw completion (markers included) -> its text blocks, thinking blocks and tool calls (OpenAI JSON, "[]" for none).
struct InklingReply { std::string content, reasoning, tool_calls_json = "[]"; };
InklingReply inkling_parse_reply(std::string_view raw);

}  // namespace ie
