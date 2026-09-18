#include "ie/server_capabilities.hpp"
#include "ie/engine.hpp"
#include "ie/gguf.hpp"
#include "ie/reasoning.hpp"
#include "ie/glm5_memory_policy.hpp"
#include "nlohmann/json.hpp"
#include <cstdlib>
#include <stdexcept>
namespace ie {
std::string server_reasoning_error(const std::string& path,std::string_view effort) {
    if(effort.empty())return {};
    if(Engine::ds41_dir(path))return reasoning_effort_error(reasoning_capabilities(ModelArch::kDeepSeek41),effort);
    GgufReader g;
    if(auto e=g.open(path);!e.empty())return "gguf: "+e;
    const auto* ct=g.find_kv("tokenizer.chat_template");
    return reasoning_effort_error(reasoning_capabilities(detect_arch(g),
        ct && ct->type==GgufValueType::kString?ct->as_string():std::string_view{}),effort);
}
bool server_supports_arch(ModelArch arch) noexcept {
    switch (arch) {
        case ModelArch::kQwen35Moe: case ModelArch::kQwen3Dense:
        case ModelArch::kQwen35Dense: case ModelArch::kLlama3:
        case ModelArch::kQwen3Moe: case ModelArch::kQwen3Next:
        case ModelArch::kGemma4: case ModelArch::kGptOss:
        case ModelArch::kDeepSeek4: case ModelArch::kQwen4Exp:
        case ModelArch::kGlm5Next: case ModelArch::kDeepSeek41: return true;
        default: return false;
    }
}
bool server_streams_experts(ModelArch arch) noexcept {
    return arch == ModelArch::kDeepSeek4 || arch == ModelArch::kQwen4Exp ||
           arch == ModelArch::kGlm5Next || arch == ModelArch::kDeepSeek41;
}
std::string server_capabilities_json(const std::string& model_path) {
    using nlohmann::json;
    ModelArch arch = ModelArch::kUnknown;
    std::string name;
    std::string chat_template;
    if (!model_path.empty() && Engine::ds41_dir(model_path)) { arch = ModelArch::kDeepSeek41; name = "deepseek_v41"; }
    else if (!model_path.empty()) {
        GgufReader g;
        if (auto e = g.open(model_path); !e.empty()) throw std::runtime_error("gguf: " + e);
        arch = detect_arch(g);
        if (auto a = g.find_kv("general.architecture")) name = a->as_string();
        if (auto t=g.find_kv("tokenizer.chat_template");t && t->type==GgufValueType::kString)chat_template=t->as_string();
    }
    const bool generic = model_path.empty();
    const bool supported = generic || server_supports_arch(arch);
    const bool glm = arch == ModelArch::kGlm5Next;
    auto reasoning=reasoning_capabilities(arch,chat_template);
    const bool default_thinking=std::getenv("IE_SERVE_NO_THINK")==nullptr;
    if(!reasoning.effort_levels.empty()) {
        if(arch==ModelArch::kGptOss && !default_thinking)reasoning.default_effort="low";
        if(const char* v=std::getenv("IE_SERVE_REASONING_EFFORT"))reasoning.default_effort=v;
        else if(reasoning.default_effort=="xhigh") {
            if(const char* v=std::getenv("IE_REASONING_EFFORT"))reasoning.default_effort=v;
        }
        if(auto error=reasoning_effort_error(reasoning,reasoning.default_effort);!error.empty())
            throw std::runtime_error("invalid reasoning default: "+error);
    }
    const bool cache = generic || arch == ModelArch::kQwen35Moe ||
        arch == ModelArch::kQwen35Dense || arch == ModelArch::kQwen3Next ||
        arch == ModelArch::kGemma4 || arch == ModelArch::kQwen4Exp ||
        arch == ModelArch::kDeepSeek4 || arch == ModelArch::kDeepSeek41;
    const bool spec = generic || arch == ModelArch::kQwen35Dense || arch == ModelArch::kGemma4;
    // int8 KV is only offered where the Engine uses ordinary KvCache; multi-GPU
    // and concurrency combinations remain subject to Engine's load validation.
    const bool kv8 = generic || is_dense_arch(arch) || arch == ModelArch::kQwen35Moe ||
        arch == ModelArch::kQwen35Dense || arch == ModelArch::kQwen3Moe;
    json j = {
        {"schema_version",1}, {"architecture",name}, {"supported",supported},
        {"memory_planner",server_streams_experts(arch)?"streaming":"resident"},
        {"sampling",{"temperature","top_k","top_p","min_p","repeat_penalty",
                     "repeat_last_n","presence_penalty","frequency_penalty",
                     "seed","max_tokens","stop"}},
        {"load",{"gpus","ctx","prefill_chunk","parallel"}},
        {"reasoning",{{"effort_levels",reasoning.effort_levels},{"default_effort",reasoning.default_effort},
                      {"thinking_description",reasoning.thinking_description}}},
        {"features",{{"prompt_cache",cache},{"speculative",spec},{"int8_kv",kv8},
                     {"context_shift",false},
                     {"vision",arch==ModelArch::kDeepSeek4 || arch==ModelArch::kQwen4Exp || arch==ModelArch::kDeepSeek41}}},
        {"defaults",{{"temperature",0.7},{"top_k",40},{"top_p",0.95},{"min_p",0.0},
                     {"repeat_penalty",1.0},{"repeat_last_n",64},{"presence_penalty",0.0},
                     {"frequency_penalty",0.0},{"seed",0},{"max_tokens",16384},
                     {"stop",json::array()},{"thinking",default_thinking},{"threads",0},
                     {"prefill_chunk",256},{"parallel",1},{"slot_ctx",0},{"prompt_cache",cache}}},
        {"notes",json::array({"Architecture support is not a memory-fit or tensor-format guarantee.",
                             "Context overflow: reject at the server; client-side compaction is separate.",
                             "Sampling considers at most 1024 candidates; top_k=0 selects this ceiling, not the entire vocabulary.",
                             "All penalties use the last repeat_last_n prompt/output tokens, up to 512; zero disables penalties.",
                             "CPU threads configure OpenMP expert work; GPU kernels have separate scheduling."})}
    };
    if(reasoning.thinking)j["load"].push_back("thinking");
    if(!reasoning.effort_levels.empty()) {
        j["load"].push_back("reasoning_effort");
        j["defaults"]["reasoning_effort"]=reasoning.default_effort;
    }
    if (glm || arch == ModelArch::kQwen4Exp) j["max_gpus"] = 2;
    if (arch == ModelArch::kGemma4) j["max_gpus"] = 1;
    if (generic || glm) j["load"].push_back("threads");
    if (generic || arch == ModelArch::kQwen35Dense) j["load"].push_back("slot_ctx");
    if(cache) j["load"].push_back("prompt_cache");
    if(spec) {
        j["load"].push_back("spec");
        j["load"].push_back("spec_k");
    }
    if(kv8) j["load"].push_back("int8_kv");
    if(glm) {
        Glm5MemoryPolicy p;
        if(auto error=glm5_memory_policy(std::getenv("IE_G5_PIN_BANKS"),std::getenv("IE_G5_ECACHE_MB"),
            std::getenv("IE_G5_PIN_FLOOR_GIB"),std::getenv("IE_G5_PIN_MAX_GIB"),std::getenv("IE_G5_PIN_OVERHEAD"),p);!error.empty())
            throw std::runtime_error(error);
        j["memory_policy"]={{"host_banks",p.pin_banks?"pinned_auto":"mmap"},
                            {"expert_cache_bytes",p.cache_bytes},{"host_floor_gib",p.floor_gib},
                            {"pin_max_bytes",p.pin_max_bytes==UINT64_MAX?json(nullptr):json(p.pin_max_bytes)},
                            {"pin_overhead",p.pin_overhead}};
    }
    if(glm) j["notes"].push_back("GLM server uses one or two layer stages; each request resets state. Prefix reuse and MTP are unavailable in this server backend.");
    if(!supported) j["notes"].push_back("No server backend for this architecture. A standalone runner, if present, does not imply server support.");
    return j.dump();
}
}
