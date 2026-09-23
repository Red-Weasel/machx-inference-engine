// include/ie/mimo26_engine.hpp — the MiMo-V2.6 bundle behind the Engine (P3b, docs/mimo26/00_PORT_PLAN.md): the
// model, the tokenizer.json tokenizer, one in-order queue per Arc card, the two-card forward, and the live
// conversation (the ids whose K/V the caches hold). Built and used in src/engine/mimo26_engine.cpp; complete here so
// Engine's destructor can destroy the unique_ptr. One request at a time (EngineOptions::parallel is 1 for this arch).
#pragma once
#include "ie/mimo26.hpp"
#include "ie/mimo26_dflash.hpp"
#include "ie/mimo26_forward.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace ie {

struct Mimo26Bundle {
    Mimo26Model                               model;
    Tokenizer                                 tok;
    std::vector<std::unique_ptr<sycl::queue>> queues;
    std::vector<sycl::queue*>                 qs;
    Mimo26Forward                             fwd;
    std::vector<int32_t>                      live;        // the ids at positions [0, fwd.n_pos()), in order
    std::vector<int32_t>                      eos;         // stop ids (generation_config.json eos_token_id)
    bool                                      prefix_reuse = true;
    bool                                      lookup = false;   // prompt-lookup speculation (ie serve: on; IE_MIMO26_LOOKUP=0)
    std::unique_ptr<Mimo26DFlash>             dflash;           // the checkpoint's DFlash drafter (IE_MIMO26_DFLASH=K), null = off
    uint32_t                                  dflash_k = 0;     // drafts per pass
    float                                     dflash_minp = 0.7f;   // drafts cut at the first one below this drafter probability
    // Drain before the frees: an aborted generation can leave kernels in flight.
    ~Mimo26Bundle() {
        for (auto& q : queues) if (q) { try { q->wait_and_throw(); } catch (const sycl::exception& e) { std::fprintf(stderr, "[mimo26] teardown drain: %s\n", e.what()); } }
        if (dflash) dflash->free_all();
        fwd.free_all();
    }
};

// The checkpoint's chat_template.jinja in C++ (P3b): a tools block as the first system turn, ChatML turns with no
// separator between them, assistant turns as <think>{reasoning}</think>{content}{tool calls} (calls as
// <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>), and the generation prompt
// "<|im_start|>assistant\n" + "<think>" (thinking: the model's own first token, primed) or "<think></think>".
struct Mimo26Message { std::string role, content, reasoning, tool_calls_json; };
std::string mimo26_render_chat(const std::vector<Mimo26Message>& msgs, const std::string& tools_json, bool add_generation_prompt,
                               bool thinking, std::string& err);
// Python json.dumps(x, ensure_ascii=False) (the template's tojson): ", " and ": " separators, keys in order.
std::string mimo26_tojson(const std::string& json_text, std::string& err);

// The completion after a mimo26_render_chat prompt: reasoning up to </think> (thinking), the content, and the XML tool
// calls as an OpenAI tool_calls JSON array ("" when none); values typed by the tool's JSON schema.
// A call whose <parameter=K> has no </parameter> is REPAIRED -- the value runs to the next parameter or the function's end,
// a stray `">` dropped (MiMo at temperature 1 wrote `<parameter=action>status"></function>`); a call that cannot be
// parsed stays in `content` as text and the calls around it are still returned.
struct Mimo26Parsed { std::string reasoning, content, tool_calls_json; bool malformed_call = false; uint32_t repaired = 0; };
Mimo26Parsed mimo26_parse_completion(const std::string& text, bool thinking, const std::string& tools_json);

}  // namespace ie
