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
    if (arch == ModelArch::kGptOss)
        return {false,{"low","medium","high"},"high","Harmony controls effort, with no reasoning-off switch"};
    if ((arch==ModelArch::kLlama3 || arch==ModelArch::kQwen3Dense) &&
        tmpl.find("<｜Assistant｜>")!=std::string_view::npos && tmpl.find("<think>")!=std::string_view::npos)
        return {true,{}, {},"on/off selects the DeepSeek-distill thinking prompt"};
    const bool qwen = arch == ModelArch::kQwen35Moe || arch == ModelArch::kQwen35Dense ||
        arch == ModelArch::kQwen3Dense || arch == ModelArch::kQwen3Moe ||
        arch == ModelArch::kQwen3Next || arch == ModelArch::kQwen4Exp;
    if (!qwen || tmpl.find("<think>") == std::string_view::npos) return {};
    ReasoningCapabilities r{true,{}, {},"on/off selects this GGUF's thinking prompt"};
    if (tmpl.find("reasoning_effort") != std::string_view::npos &&
        tmpl.find("Reasoning effort is set to xhigh.") != std::string_view::npos) {
        r.effort_levels={"low","medium","high","xhigh"}; r.default_effort="xhigh";
    }
    return r;
}
inline bool known_reasoning_effort(std::string_view v) {
    return v=="low" || v=="medium" || v=="high" || v=="xhigh" || v=="max";
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
