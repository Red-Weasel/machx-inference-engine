// include/ie/recommended_sampling.hpp — P4 B20: the publishers' recommended sampling per architecture and mode
// (sources and quotes: ~/ds41_work/p60/recsettings/PLAN.md section 1). `ie serve` fills a request's omitted sampling
// fields from the row of the request's ACTUAL mode (resolution order: request > CLI/env > this table > library
// default; openai_proto.hpp apply_recommended), and `ie capabilities` reports the table as "recommended".
// Fields a card leaves unspecified disable the knob (top_k 0, min_p 0, presence 0, repeat 1.0) rather than guess.
#pragma once
#include "ie/engine.hpp"
#include "ie/model_config.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ie {

struct RecommendedMode {
    double   temperature = 0, top_p = 1;
    uint32_t top_k = 0;
    double   min_p = 0, presence_penalty = 0, repeat_penalty = 1;
    const char* when = "";   // one line: when to use this mode
};
struct RecommendedEffort { const char* level; const char* when; };
struct Recommendation {
    const char* model = "";
    const char* source = "";                  // the model card
    RecommendedMode thinking;
    std::optional<RecommendedMode> instruct;  // nullopt: no official non-thinking mode (GLM)
    std::vector<RecommendedEffort> effort;    // empty: the model has no effort control
    const char* instruct_label = "Instruct";  // DeepSeek-V4.1 calls its non-thinking mode "chat"
    uint32_t max_output = 0;                  // the answer cap; 0 = not specified by the card
    uint32_t max_reasoning = 0;               // a separate reasoning cap (Qwen3.8); 0 = none stated
    bool max_output_instruct = true;          // false: max_output is the thinking mode's only (the 35B distill card)
    const char* max_output_note = "";
    uint32_t context = 0;                     // native context (tokens)
    const char* note = "";
};

// general.name as Dream compares it (model_defaults.py carded_model): lower case, each run of ' ', '_', '-' -> '-'.
inline std::string recommended_name_key(std::string_view name) {
    std::string out;
    for (char c : name) {
        if (c == ' ' || c == '_' || c == '-') { if (out.empty() || out.back() != '-') out += '-'; }
        else out += char(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    }
    return out;
}

// The row set for exactly these weights; nullopt keeps today's library defaults. The match mirrors Dream's
// carded_model (DREAM-179): Qwen3.8 (qwen35 / qwen4exp) by the template's xhigh sentence; the 35B-A3B distill by
// name (its template equals the plain Qwen3.6-35B-A3B's, which gets nothing); GLM-5.3-Flash by name; the
// directory-loaded DeepSeek-V4.1 and MiMo-V2.6 by architecture. `tmpl`, `name`, `base_repo` are the GGUF's
// tokenizer.chat_template, general.name and general.base_model.0.repo_url ("" for a directory model).
inline std::optional<Recommendation> recommended_sampling(ModelArch arch, std::string_view tmpl = {},
                                                          std::string_view name = {}, std::string_view base_repo = {}) {
    const bool think = tmpl.find("<think>") != std::string_view::npos;
    const std::string nm = recommended_name_key(name);
    const bool distill = nm.find("qwen3.8-35b-a3b-distill") != std::string::npos ||
                         (nm == "ours" && base_repo == "https://huggingface.co/Qwen/Qwen3.6-35B-A3B");
    if ((arch == ModelArch::kQwen35Dense || arch == ModelArch::kQwen4Exp) &&
        tmpl.find("Reasoning effort is set to xhigh.") != std::string_view::npos) {   // the Qwen3.8 template
        Recommendation r;
        const bool next = arch == ModelArch::kQwen4Exp;
        r.model  = next ? "Qwen3.8-Flash-Next" : "Qwen3.8-27B";
        r.source = next ? "https://huggingface.co/Qwen/Qwen3.8-Flash-Next" : "https://huggingface.co/Qwen/Qwen3.8-27B";
        r.thinking = {1.0, 0.95, 20, 0.0, 0.0, 1.0,
                      "math, coding, multi-step planning and agent work; slower, longer output"};
        r.instruct = RecommendedMode{0.7, 0.80, 20, 0.0, 1.5, 1.0,
                      "chat, quick answers, simple edits, high-volume subagents; faster, shorter output (general guidance, not from the card)"};
        r.effort = {{"xhigh", "complex tasks demanding thorough analysis (card; the default)"},
                    {"medium", "balancing accuracy and speed (card)"},
                    {"low", "efficient reasoning optimizing for speed and cost (card)"}};
        r.max_output = 131072;
        r.max_reasoning = 262144;
        r.max_output_note = "card: final response 131,072 tokens, reasoning 262,144";
        r.context = 262144;
        r.note = "card: presence_penalty 0-2 reduces endless repetition; higher values may mix languages";
        return r;
    }
    if (arch == ModelArch::kQwen35Moe && think && distill) {
        Recommendation r;
        r.model = "Qwen3.8-35B-A3B-Distill";
        r.source = "https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill";
        r.thinking = {0.6, 0.95, 20, 0.0, 0.0, 1.0,
                      "math, code and multi-step reasoning (the distill's traces are weighted toward hard math and competitive programming)"};
        r.instruct = RecommendedMode{0.7, 0.80, 20, 0.0, 1.5, 1.0,
                      "quick chat and simple edits (general guidance; the distill's publisher did not evaluate this mode)"};
        r.max_output = 16384;
        r.max_output_instruct = false;
        r.max_output_note = "distill card: 16,384 recommended (thinking); instruct not specified by the distill";
        r.context = 262144;
        r.note = "thinking temperature 0.6 = the distill card's best practice (its generation_config says 1.0; owner decision "
                 "2026-09-28); unspecified thinking fields and the instruct set come from the base Qwen3.6-35B-A3B card";
        return r;
    }
    if (arch == ModelArch::kMimo26) {
        Recommendation r;
        r.model = "MiMo-V2.6-Flash-RL";
        r.source = "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL";
        r.thinking = {1.0, 0.95, 0, 0.0, 0.0, 1.0, "agentic coding, planning and hard reasoning (general guidance; the card gives none)"};
        r.instruct = RecommendedMode{1.0, 0.95, 0, 0.0, 0.0, 1.0,
                      "quick replies and routine edits (general guidance; the card gives one sampling set for both modes)"};
        r.max_output_note = "not specified by the card";
        r.context = 1048576;
        r.note = "card: temperature 1.0, top_p 0.95; the other fields are not specified (off)";
        return r;
    }
    if (arch == ModelArch::kGlm5Next && nm.find("glm-5.3-flash") != std::string::npos) {
        Recommendation r;
        r.model = "GLM-5.3-Flash";
        r.source = "https://huggingface.co/zai-org/GLM-5.3-Flash";
        r.thinking = {1.0, 0.95, 0, 0.0, 0.0, 1.0, "GLM always reasons (the vendor template has no thinking-off switch); pick the effort"};
        r.effort = {{"max", "benchmarks, hard coding and agent work (card: keep the default max)"},
                    {"high", "faster, cheaper turns (general guidance)"},
                    {"low", "the fastest, cheapest turns (general guidance)"}};
        r.max_output_note = "not specified by the card (its evaluations used 64K-160K)";
        r.context = 1048576;
        r.note = "top_p 0.95 = generation_config (owner decision 2026-09-28); the card's coding and agent evaluations used top_p 1.0. "
                 "--thinking off is a local convention, not an official mode, and uses the same sampling";
        return r;
    }
    if (arch == ModelArch::kDeepSeek41) {
        Recommendation r;
        r.model = "DeepSeek-V4.1-Flash";
        r.source = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash";
        r.thinking = {1.0, 0.95, 0, 0.0, 0.0, 1.0, "reasoning, coding and agent work; pick the effort"};
        r.instruct = RecommendedMode{1.0, 0.95, 0, 0.0, 0.0, 1.0,
                      "chat mode: quick answers (general guidance; the card gives one sampling set for both modes)"};
        r.effort = {{"max", "the card's agent and coding benchmark setting (budget 100)"},
                    {"high", "the default (budget 75)"},
                    {"low", "faster turns (budget 50; general guidance)"}};
        r.instruct_label = "Chat";
        r.max_output = 262144;
        r.max_output_note = "card: max_tokens >= 256K";
        r.context = 1048576;
        r.note = "card: top_p 0.95 or 1.0; 0.95 is what every card evaluation used (owner decision 2026-09-28)";
        return r;
    }
    return std::nullopt;
}

// The row of a request's actual mode: instruct when thinking is off and the model has one, else the thinking row.
inline const RecommendedMode& recommended_mode(const Recommendation& r, bool thinking) {
    return !thinking && r.instruct ? *r.instruct : r.thinking;
}

// DeepSeek-V4.1 and MiMo-V2.6 sample on the host (deepseek41_generate.cpp), which has no presence or frequency penalty:
// such a setting is served WITHOUT it (owner decision 2026-09-28: warn, do not refuse). "" = nothing is dropped.
inline std::string dropped_penalty_warning(ModelArch arch, const SamplingParams& sp) {
    if (arch != ModelArch::kDeepSeek41 && arch != ModelArch::kMimo26) return {};
    if (sp.presence_penalty == 0.f && sp.frequency_penalty == 0.f) return {};
    return std::string("presence_penalty ") + std::to_string(sp.presence_penalty) + " / frequency_penalty " +
           std::to_string(sp.frequency_penalty) + " IGNORED: the " +
           (arch == ModelArch::kMimo26 ? "MiMo-V2.6" : "DeepSeek-V4.1") +
           " sampler has no presence/frequency penalty (repeat_penalty works)";
}

}  // namespace ie
