#pragma once
#include "ie/model_config.hpp"
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace ie {
struct ReasoningCapabilities {
    bool thinking = false;
    std::vector<std::string> effort_levels;
    std::string default_effort;
    std::string thinking_description;
};
// Match implemented prompt families, including the selected GGUF template.
// A model name or a <think> vocabulary token alone is not evidence of support.
inline ReasoningCapabilities reasoning_capabilities(ModelArch arch, std::string_view tmpl = {}) {
    if (arch == ModelArch::kGlm5Next)
        return {true,{"low","high","max"},"max",
                "on = GLM reasoning; off = local empty-think prompt convention (vendor template always opens thinking)"};
    if (arch == ModelArch::kDeepSeek4)
        return {true,{"low","high","max"},"low","on/off selects the DeepSeek thinking template"};
    if (arch == ModelArch::kDeepSeek41)
        return {true,{"low","high","max"},"high","on/off selects the DeepSeek-V4.1 thinking template; the effort is its numeric budget (low 50, high 75, max 100)"};
    if (arch == ModelArch::kMimo26)
        return {true,{}, {},"on/off selects MiMo-V2.6's thinking prompt (off renders <think></think>)"};
    if (arch == ModelArch::kHy4Flash)
        return {true,{"high","max"},"high","on/off selects Hy4-flash's reasoning mode (off = no_think); the effort is the template's high or max"};
    if (arch == ModelArch::kLaguna)
        return {true,{}, {},"on/off selects Laguna's thinking prompt (off renders </think>)"};
    if (arch == ModelArch::kXing4)
        return {true,{}, {},"on/off selects Xing4's thinking prompt (off renders </think>)"};
    if (arch == ModelArch::kInkling)
        return {true,{"minimal","low","medium","high","xhigh","max"},"high",
                "Inkling's 'Thinking effort level' line: off = 0; the effort picks the level (minimal 0.1 ... high 0.9, max 0.99)"};
    if (arch == ModelArch::kInkling2)
        return {true,{"none","minimal","low","medium","high","xhigh","max"},"high",
                "Inkling 2's 'Thinking effort level' line (chat_template.jinja): off = 0; none 0, minimal 0.1, low 0.2, medium 0.7, "
                "high 0.9 (the template's default), xhigh / max 0.99"};
    if (arch == ModelArch::kGptOss)
        return {false,{"low","medium","high"},"high","Harmony controls effort, with no reasoning-off switch"};
    if ((arch==ModelArch::kLlama3 || arch==ModelArch::kQwen3Dense) &&
        tmpl.find("<｜Assistant｜>")!=std::string_view::npos && tmpl.find("<think>")!=std::string_view::npos)
        return {true,{}, {},"on/off selects the DeepSeek-distill thinking prompt"};
    const bool qwen = arch == ModelArch::kQwen35Moe || arch == ModelArch::kQwen35Dense ||
        arch == ModelArch::kQwen3Dense || arch == ModelArch::kQwen3Moe ||
        arch == ModelArch::kQwen3Next || arch == ModelArch::kQwen4Exp;
    // 2026-10-28: a llama-arch GGUF whose own template is Qwen's ChatML with <think> (Nanbeige4.3-33B) renders through
    // the same ChatML builder and writes the same "reasoning\n</think>\n\nanswer", so it gets the same switch and split.
    const bool chatml_llama = arch == ModelArch::kLlama3 && tmpl.find("<|im_start|>") != std::string_view::npos;
    if (!(qwen || chatml_llama) || tmpl.find("<think>") == std::string_view::npos) return {};
    ReasoningCapabilities r{true,{}, {},"on/off selects this GGUF's thinking prompt"};
    if (tmpl.find("reasoning_effort") != std::string_view::npos &&
        tmpl.find("Reasoning effort is set to xhigh.") != std::string_view::npos) {
        r.effort_levels={"low","medium","high","xhigh"}; r.default_effort="xhigh";
    }
    return r;
}
// P4 B35: Engine::chat renders this model through the ChatML builder (tokenizer.hpp) with a template that opens <think>
// in the generation prompt -- the Qwen row above, as chat() dispatches it (a dense GGUF reaches the ChatML builder only
// with a ChatML or unrecognised template: R1-Distill's sentinels take their own path). The model writes
// "reasoning\n</think>\n\nanswer"; the engine shows the tag (Tokenizer::show_special) and splits the reasoning off
// there, streamed and not (split_chatml_think).
inline bool chatml_think_split(ModelArch arch, std::string_view tmpl) {
    const bool qwen = arch == ModelArch::kQwen35Moe || arch == ModelArch::kQwen35Dense ||
        arch == ModelArch::kQwen3Dense || arch == ModelArch::kQwen3Moe ||
        arch == ModelArch::kQwen3Next || arch == ModelArch::kQwen4Exp;
    const bool chatml_llama = arch == ModelArch::kLlama3 && tmpl.find("<|im_start|>") != std::string_view::npos;   // Nanbeige4.3
    if (!(qwen || chatml_llama) || tmpl.find("<think>") == std::string_view::npos) return false;
    if (!is_dense_arch(arch)) return true;
    const DenseConfig::TemplateFamily tf = classify_template_family(tmpl);
    return tf == DenseConfig::TemplateFamily::kChatML || tf == DenseConfig::TemplateFamily::kAuto;
}
inline bool known_reasoning_effort(std::string_view v) {
    return v=="low" || v=="medium" || v=="high" || v=="xhigh" || v=="max" ||
           v=="none" || v=="minimal";   // Inkling 2's template names: only kInkling2 takes them (arch_reasoning_effort_error)
}
// "none" and "minimal" parse for every request (known_reasoning_effort) because Inkling 2's template names them; every
// other arch refuses them exactly as it did before they were added: the request parser's message (a 400), the CLI's.
inline bool inkling2_only_effort(std::string_view v) { return v=="none" || v=="minimal"; }
inline constexpr const char* kUnknownEffortRequest = "invalid request setting: unknown reasoning_effort level";
inline constexpr const char* kUnknownEffortCli = "--reasoning-effort requires a supported model effort level";
inline std::string reasoning_effort_error(const ReasoningCapabilities& r, std::string_view effort);
// A request's effort for this arch (and its GGUF template): "" = accepted.
inline std::string arch_reasoning_effort_error(ModelArch arch, std::string_view tmpl, std::string_view effort) {
    if (arch != ModelArch::kInkling2 && inkling2_only_effort(effort)) return kUnknownEffortRequest;
    return reasoning_effort_error(reasoning_capabilities(arch, tmpl), effort);
}
inline std::string reasoning_effort_error(const ReasoningCapabilities& r, std::string_view effort) {
    if (effort.empty()) return {}; // keep the architecture's default
    if (std::find(r.effort_levels.begin(),r.effort_levels.end(),effort)!=r.effort_levels.end()) return {};
    if (r.effort_levels.empty()) return "reasoning_effort is not supported by this model's prompt backend";
    std::string msg="reasoning_effort must be one of: ";
    for (const auto& v:r.effort_levels) { if(msg.back()!=' ')msg+=", ";msg+=v; }
    return msg;
}
}
