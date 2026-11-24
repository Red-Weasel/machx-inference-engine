// include/ie/inkling2_engine.hpp — Inkling 2 Mini (HF `inkling2_loop`, kInkling2) in the engine: the bundle Engine holds,
// the native chat-template renderer, the reply parser and the stream translator. src/engine/inkling2_engine.cpp.
// The runtime is Inkling2Model (include/ie/inkling2.hpp, one card); the model is a DIRECTORY (config.json model_type
// "inkling2_loop", tokenizer.json, chat_template.jinja). The chat format is docs/inkling2/SPEC.md section 12.
#pragma once
#include "ie/inkling2.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ie {

// The chat format's markers a reply can hold (the token ids are looked up at load).
enum class Ink2Mark : uint8_t {
    kNone,        // ordinary text
    kModel,       // <|message_model|>        opens a model block (a tool name may follow before the content kind)
    kThinking,    // <|content_thinking|>     reasoning
    kText,        // <|content_text|>         answer text
    kToolError,   // <|content_tool_error|>   answer text (response_template: content)
    kToolText,    // <|content_invoke_tool_text|> answer text (response_template: content)
    kToolJson,    // <|content_invoke_tool_json|> {"name":..,"args":{..}}
    kEnd,         // <|end_message|>          closes a block
    kEos,         // <|content_model_end_sampling|> the end of the turn (EOS 200006)
    kOther,       // any other special token (kept out of every field)
};

struct Ink2Serve;                         // --parallel N > 1: the per-card batch loops (src/engine/inkling2_engine.cpp)
class Inkling2Vision;                     // include/ie/inkling2_vision.hpp
inline constexpr int32_t kInk2ImageToken = 200054;   // <|unused_200054|>: one per image token

// A request's images: rows [n, hidden] (the vision tower's output) for the prompt positions `at` (placeholder tokens),
// and per position a stand-in id for the prompt cache (negative: never a vocabulary id; from the image's bytes, so two
// requests share a prefix through an image only when it is the same image).
struct Ink2Images {
    std::vector<float>    rows;
    std::vector<uint32_t> at;
    std::vector<int32_t>  key;
};

struct Inkling2Bundle {
    std::unique_ptr<sycl::queue> q;       // declared before the model: destroyed after it
    std::unique_ptr<sycl::queue> q2;      // --gpus 2 --parallel 1: the second card (two cards on one conversation)
    std::unique_ptr<sycl::queue> vq;      // the vision tower's queue (card 0, the model's context)
    std::shared_ptr<Inkling2Vision> vis;  // the vision tower (null: text only)
    std::mutex vis_mu;                    // one image encode at a time
    uint32_t vis_max_tokens = 0;          // the largest image (in tokens) the tower takes; larger ones are scaled down
    Inkling2Model                model;
    Tokenizer                    tok;
    std::vector<int32_t>         live;    // the tokens the model's state holds, in order (prefix reuse)
    std::vector<int32_t>         stop;    // EOS 200006, <|endoftext|> and the non-model message openers
    std::vector<std::pair<int32_t, Ink2Mark>> marks;   // special token id -> marker
    uint32_t ctx = 0, chunk = 256, rollback = 0;   // rollback: how far truncate() can rewind behind the furthest position written
    bool     reuse = true;                // --no-prompt-cache: every request reads its whole prompt
    bool     open_turn = false;           // append <|message_model|> to the prompt (IE_INKLING2_OPEN_TURN=1; off: the reference)
    Ink2Mark mark_of(int32_t id) const;
    // --parallel N > 1 (declared last: its batch loops stop before the model and the queue above are freed). Null: the
    // one-request path (inkling2_run under the engine's mutex), exactly as before.
    std::shared_ptr<Ink2Serve> serve;
};
// /health fields and the slot count of the many-conversation server ("" / 0 without it).
std::string inkling2_serve_status(const Inkling2Bundle& b);
uint32_t    inkling2_serve_slots(const Inkling2Bundle& b);
bool        inkling2_vision_ready(const Inkling2Bundle& b);   // the vision tower is loaded (images accepted)

// The checkpoint's chat_template.jinja rendered natively, byte for byte (tests/unit/inkling2_serve_test.cpp holds the
// reference's renders). thinking off = effort level 0 (the template's only switch); thinking on with no effort = the
// template's default 0.9; effort names none / minimal / low / medium / high / xhigh / max (trimmed, case-sensitive).
// Where a request can say more than the template takes: tool-call arguments given as a JSON string are parsed (the
// template wants an object: "canonicalize upstream"); a "developer" turn renders as a system turn. open_turn appends
// <|message_model|> (the template itself ends at the last block). `content` (known_bugs 45): the request-text ranges.
// image_tokens: per image of the request, in turn order, its token count (rows x cols); a user / tool turn's content is
// split at its kChatImageMarker positions and each image becomes its own block <role><|content_image|> + that many
// <|unused_200054|> + <|end_message|> (the processor's expansion of the template's one placeholder). nullptr: a request
// with images is refused.
std::string inkling2_render_prompt(std::span<const ChatTurn> turns, std::string_view tools_json, bool thinking,
                                   std::string_view effort, std::string& err,
                                   std::vector<std::pair<size_t, size_t>>* content = nullptr, bool open_turn = false,
                                   const std::vector<uint32_t>* image_tokens = nullptr);
// "0", "0.1", ... "0.99" for (thinking, effort); false + err for an unknown name.
bool inkling2_effort_text(std::string_view effort, bool thinking, std::string& out, std::string& err);

// A reply as the model writes it, one event at a time (a marker or a piece of text), turned into the OpenAI fields and
// into the text the server's stream rules read (structured arch: the reasoning, then "</think>", then the content; a tool
// call as "<tool_call>{json}</tool_call>", sent whole when its block closes so a call cut off by max_tokens never reaches
// the stream).
class Inkling2ReplyParser {
public:
    explicit Inkling2ReplyParser(bool thinking) : thinking_(thinking) {}
    // Feed one event; returns the stream text it produces ("" for most markers).
    std::string feed(Ink2Mark m, std::string_view text = {});
    // The reply ended (EOS, a stop, max_tokens or an abort). complete = the model ended its turn itself.
    void finish(bool complete);
    const std::string& content() const { return content_; }
    const std::string& reasoning() const { return reasoning_; }
    std::string tool_calls_json() const;            // OpenAI tool_calls, "[]" for none
    size_t n_calls() const { return calls_.size(); }
    const std::string& truncated_call() const { return truncated_; }   // finish(false) inside a call: its name
    bool in_block() const { return kind_ != kNone; }
private:
    enum Kind { kNone, kHeader, kReason, kAnswer, kCall };
    void close_block();
    std::string open_answer();    // "</think>" the first time a non-reasoning block opens with thinking on
    bool thinking_, closed_ = false;
    Kind kind_ = kNone;
    std::string header_, body_, content_, reasoning_, truncated_;
    std::vector<std::pair<std::string, std::string>> calls_;   // (name, arguments JSON)
};

// Split a raw reply text at the marker spellings into parser events (tests; and a raw text the engine did not decode).
std::vector<std::pair<Ink2Mark, std::string>> inkling2_events_from_text(std::string_view raw);

}  // namespace ie
