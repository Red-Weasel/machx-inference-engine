// include/ie/tool_guard.hpp -- every tool call the model writes is a valid call of the request's tools.
//
// The request carries its tools (OpenAI "tools": name + JSON-schema parameters). While the model writes a tool call, the
// sampler checks each candidate token against them (ToolGuard::feed on a copy of the lane's state) and drops the ones that
// would break the call: the function name must be one of the tools; each argument key one of that tool's parameters, not
// repeated; a value must fit its type (integer, number, boolean, enum, JSON object / array); the call closes only once every
// required parameter is there; and the model cannot stop in the middle of a call. Text outside a call is untouched.
//
// The grammar is the tagged family (GLM-5 / Laguna / Hy4-flash: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>
// </tool_call>; Xing4 spells the key / value tags <param_key> / <param_value>). It reads the reply's TEXT, so a tag the
// tokenizer splits into several tokens is checked the same as a single-token tag.
#pragma once
#include <cctype>
#include <cstdint>
#include <initializer_list>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ie {

class ToolGuard {
public:
    struct Tags { std::string open = "<tool_call>", close = "</tool_call>", key_open = "<arg_key>", key_close = "</arg_key>",
                  val_open = "<arg_value>", val_close = "</arg_value>", think_close = "</think>"; };
    static Tags glm_tags() { return {}; }
    static Tags xing4_tags() { Tags t; t.key_open = "<param_key>"; t.key_close = "</param_key>"; t.val_open = "<param_value>";
                               t.val_close = "</param_value>"; return t; }

    // tools_json: the request's "tools" array. Null with err set when it cannot be read; null with err empty when there are no
    // tools (nothing to guard). make: the tagged family; make_dsml: DeepSeek-V4.1's DSML block (deepseek41_prompt.cpp):
    //   "\n\n<｜DSML｜ calls>\n" (<｜DSML｜ invoke name="N">\n (<｜DSML｜ parameter name="K" string="true|false">V</｜DSML｜ parameter>\n)*
    //   </｜DSML｜ invoke>\n)+ </｜DSML｜ calls> then the end of the reply; string="true" for a string / enum parameter, "false"
    //   (a JSON value) for the others.
    static std::shared_ptr<const ToolGuard> make(std::string_view tools_json, const Tags& tags, std::string& err);
    static std::shared_ptr<const ToolGuard> make_dsml(std::string_view tools_json, std::string& err);
    // make_xml: Qwen's XML form (Qwen3.6 / 3.8 / Flash-Next, MiMo): <tool_call>\n<function=N>\n(<parameter=K>\nV\n</parameter>\n)*
    //   </function>\n</tool_call> -- a value is the text between the tags less one newline each side (tokenizer.cpp
    //   xml_call_from_body), typed by its schema.
    static std::shared_ptr<const ToolGuard> make_xml(std::string_view tools_json, std::string& err);
    // tool_choice (SamplingParams::tool_choice): "" auto, "none" (no call may open), "required" (the reply is a call: it opens
    // one at once -- after the reasoning -- and cannot end without one), "fn:NAME" (required, that tool only). A copy.
    std::shared_ptr<const ToolGuard> with_choice(std::string_view choice, std::string& err) const;

    enum class Mode : uint8_t { kThink, kFree, kName, kKey, kValue, kAfter,
                                kDOpen, kDInvoke, kDName, kDParams, kDKey, kDFlag, kDValue, kDAfterInvoke, kDDone,
                                kXFn, kXName, kXParams, kXKey, kXValue, kXAfterFn };
    struct State {
        Mode mode = Mode::kFree;
        std::string buf;          // the part being written (name / key / value / next tag); in think / free: the text's tail
        int32_t tool = -1, param = -1;
        uint64_t used = 0;        // the parameters already given (bit per parameter)
        uint32_t calls = 0;       // completed calls
        uint8_t ws = 0;           // leading whitespace taken before a name / a required call (at most 2: no newline loops)
        std::string keys;         // an open-named call's keys so far ("\x1f"-joined: no key twice)
    };
    // The state a reply starts in: inside the reasoning when the prompt opened it (thinking on)
    State start(bool in_think) const { State s; s.mode = in_think ? Mode::kThink : Mode::kFree; return s; }
    // Advance s over the text; false = the text breaks the call (s is then undefined: feed a copy to test a candidate)
    bool feed(State& s, std::string_view text) const;
    // May the reply end here? (not in the middle of a call; after a DSML block: only the end; tool_choice required: not before a call)
    bool may_stop(const State& s) const {
        if (need_call_ && s.calls == 0) return false;
        return s.mode == Mode::kThink || s.mode == Mode::kFree || s.mode == Mode::kDDone;
    }
    // Inside a call: every candidate must be checked (outside, only the likely ones)
    static bool in_call(const State& s) { return s.mode != Mode::kThink && s.mode != Mode::kFree; }
    size_t n_tools() const { return tools_.size(); }

private:
    enum class Kind : uint8_t { kStr, kInt, kNum, kBool, kEnum, kJson, kAny };   // kAny: any JSON value
    struct Param { std::string name; Kind kind = Kind::kStr; std::vector<std::string> enums; bool required = false; };
    struct Tool { std::string name; std::vector<Param> params; uint64_t required = 0; };
    bool step(State& s, char c) const;
    bool step_dsml(State& s, char c) const;
    bool step_xml(State& s, char c) const;
    bool dsml_ = false, xml_ = false;
    bool no_calls_ = false, need_call_ = false;   // tool_choice none / required
    int32_t only_tool_ = -1;                     // tool_choice naming a function
    bool name_ok(size_t i) const { return only_tool_ < 0 || int32_t(i) == only_tool_; }
    // Dream's lazy tools (2026-11-20): a request that offers Dream's dispatcher (tool_call / tool_schema) lists the other tools in
    // its prompt and the model calls them by name, which Dream runs -- such a request lets a call name any tool (structure still
    // checked: the tags, an identifier name, no key twice, JSON where the format says JSON). s.tool = kOpen for such a call.
    bool open_names_ = false;
    // Any harness (2026-11-20): without the dispatcher, a call may still name a tool that is not in `tools` when that name
    // is a word of the request's conversation (ToolGuardPrompt) -- a harness that lists its tools in the prompt keeps working,
    // a made-up name is still refused. prompt_words_: the conversation's identifier words, sorted.
    bool prompt_names_ = false;
    std::vector<std::string> prompt_words_;
    bool name_allowed(std::string_view name, bool complete) const;
    static constexpr int32_t kOpen = -2;
    static bool ident(char c) { return std::isalnum(uint8_t(c)) || c == '_' || c == '-' || c == '.' || c == ':'; }
    // buf = identifier + a prefix of one of `ends`: 0 no, 1 partial, 2 + index of the end completed
    static int open_match(std::string_view buf, std::initializer_list<std::string_view> ends);
    Param open_param_ = {};   // the param an open-named call's value is checked against (string)
    Param open_json_ = {"", Kind::kAny};
    bool value_prefix_ok(const Param& p, std::string_view v) const;
    bool value_ok(const Param& p, std::string_view v) const;
    std::vector<Tool> tools_;
    Tags t_;
};

// The request's conversation text for the guard's prompt-listed names, on this thread for this scope (Engine::chat sets it;
// ToolGuard::make* on the same thread reads it).
struct ToolGuardPrompt {
    explicit ToolGuardPrompt(std::string text);
    ~ToolGuardPrompt();
    ToolGuardPrompt(const ToolGuardPrompt&) = delete;
    ToolGuardPrompt& operator=(const ToolGuardPrompt&) = delete;
    std::string text;
};

// A request's guard while it decodes (any arch with host logits): the state over the reply so far, the filter its next sample
// uses. pieces = each token's text inside a reply; stop = the arch's stop ids (ruled out mid-call).
struct LanesFilter { virtual void apply(std::vector<float>& logits) const = 0; virtual ~LanesFilter() = default; };
struct LaneGuard final : LanesFilter {
    const ToolGuard* g = nullptr;
    std::shared_ptr<const ToolGuard> keep;   // (optional ownership)
    ToolGuard::State st;
    const std::vector<std::string>* pieces = nullptr;
    std::function<bool(int32_t)> stop;
    mutable uint64_t masked = 0, scans = 0;   // tokens ruled out; full-vocabulary scans (no likely token was valid)
    void apply(std::vector<float>& logits) const override { apply_raw(logits.data(), logits.size()); }
    void apply_raw(float* logits, size_t V) const;
    void advance(int32_t id);                 // the sampled id into the state
};
// Each token's text as it appears inside a reply (decode([id, id]) minus decode([id]): no word-start trimming)
std::vector<std::string> tool_guard_pieces(uint32_t vocab, const std::function<std::string(const int32_t*, size_t)>& detok);

}  // namespace ie
