#include "ie/server_capabilities.hpp"
#include "ie/engine.hpp"
#include "ie/gguf.hpp"
#include "ie/reasoning.hpp"
#include "ie/glm5_memory_policy.hpp"
#include "ie/openai_proto.hpp"
#include "ie/recommended_sampling.hpp"
#include "nlohmann/json.hpp"
#include <cmath>
#include <cstdlib>
#include <stdexcept>
namespace ie {
std::string server_reasoning_error(const std::string& path,std::string_view effort) {
    if(effort.empty())return {};
    if(Engine::ds41_dir(path))return reasoning_effort_error(reasoning_capabilities(ModelArch::kDeepSeek41),effort);
    if(Engine::mimo26_dir(path))return reasoning_effort_error(reasoning_capabilities(ModelArch::kMimo26),effort);
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
        case ModelArch::kGlm5Next: case ModelArch::kDeepSeek41: case ModelArch::kMimo26: return true;
        default: return false;
    }
}
bool server_streams_experts(ModelArch arch) noexcept {
    return arch == ModelArch::kDeepSeek4 || arch == ModelArch::kQwen4Exp ||
           arch == ModelArch::kGlm5Next || arch == ModelArch::kDeepSeek41 || arch == ModelArch::kMimo26;
}
std::string server_capabilities_json(const std::string& model_path) {
    using nlohmann::json;
    ModelArch arch = ModelArch::kUnknown;
    std::string name;
    std::string chat_template, model_name, base_repo;
    if (!model_path.empty() && Engine::ds41_dir(model_path)) { arch = ModelArch::kDeepSeek41; name = "deepseek_v41"; }
    else if (!model_path.empty() && Engine::mimo26_dir(model_path)) { arch = ModelArch::kMimo26; name = "mimo_v2"; }
    else if (!model_path.empty()) {
        GgufReader g;
        if (auto e = g.open(model_path); !e.empty()) throw std::runtime_error("gguf: " + e);
        arch = detect_arch(g);
        if (auto a = g.find_kv("general.architecture")) name = a->as_string();
        if (auto t=g.find_kv("tokenizer.chat_template");t && t->type==GgufValueType::kString)chat_template=t->as_string();
        // P4 B20: what recommended_sampling matches the exact weights on
        if (auto t=g.find_kv("general.name");t && t->type==GgufValueType::kString)model_name=t->as_string();
        if (auto t=g.find_kv("general.base_model.0.repo_url");t && t->type==GgufValueType::kString)base_repo=t->as_string();
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
        arch == ModelArch::kDeepSeek4 || arch == ModelArch::kDeepSeek41 || arch == ModelArch::kMimo26;
    // v0.2.21: the 35B-A3B class (lookup drafts on the request lanes, P4 B55) and Flash-Next (lookup drafts at one lane, B56; its
    // MTP head only with a --spec-head path, B63) take --spec too, so a client's load screen can offer it.
    const bool spec = generic || arch == ModelArch::kQwen35Dense || arch == ModelArch::kGemma4 ||
        arch == ModelArch::kQwen35Moe || arch == ModelArch::kQwen4Exp;
    // int8 KV is only offered where the Engine uses ordinary KvCache; multi-GPU
    // and concurrency combinations remain subject to Engine's load validation.
    const bool kv8 = generic || is_dense_arch(arch) || arch == ModelArch::kQwen35Moe ||
        arch == ModelArch::kQwen35Dense || arch == ModelArch::kQwen3Moe;
    // P4 B20: "defaults" are what a request omitting a field gets in the default mode: IE_SERVE_* env, else the
    // model's recommended sampling (recommended_sampling.hpp), else the library default. CLI flags are not seen here.
    const std::optional<Recommendation> rec = generic ? std::nullopt : recommended_sampling(arch, chat_template, model_name, base_repo);
    oai::ChatRequest resolved = oai::server_defaults_from_environment();
    oai::apply_recommended(resolved, rec ? &*rec : nullptr);
    const SamplingParams& sp = resolved.sampling;
    // 0.95f -> 0.95, not 0.949999988 (through text: icpx's fast math turns a /1e6 into *1e-6, which is not exact)
    auto r6 = [](double v) { char b[32]; std::snprintf(b, sizeof b, "%.6g", v); return std::strtod(b, nullptr); };
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
                     // (P4 B45: the Qwen3.8 splits -- the 27B takes images on its two-card split with an mmproj at --parallel 1,
                     // the 35B-A3B is the next step; the load's /props "vision" says whether THIS load is ready)
                     {"vision",arch==ModelArch::kDeepSeek4 || arch==ModelArch::kQwen4Exp || arch==ModelArch::kDeepSeek41 || arch==ModelArch::kMimo26 ||
                               arch==ModelArch::kQwen35Dense || arch==ModelArch::kQwen35Moe}}},
        {"defaults",{{"temperature",r6(sp.temperature)},{"top_k",sp.top_k},{"top_p",r6(sp.top_p)},{"min_p",r6(sp.min_p)},
                     {"repeat_penalty",r6(sp.repeat_penalty)},{"repeat_last_n",sp.repeat_window},{"presence_penalty",r6(sp.presence_penalty)},
                     {"frequency_penalty",r6(sp.frequency_penalty)},{"seed",sp.seed},
                     {"max_tokens",sp.max_tokens==kMaxTokensUnlimited?0u:sp.max_tokens},
                     {"stop",json::array()},{"thinking",default_thinking},{"threads",0},
                     {"prefill_chunk",256},{"parallel",1},{"slot_ctx",0},{"prompt_cache",cache}}},
        {"notes",json::array({"Architecture support is not a memory-fit or tensor-format guarantee.",
                             "Context overflow: reject at the server; client-side compaction is separate.",
                             "Sampling considers at most 1024 candidates; top_k=0 selects this ceiling, not the entire vocabulary.",
                             "All penalties use the last repeat_last_n prompt/output tokens, up to 512; zero disables penalties.",
                             "CPU threads configure OpenMP expert work; GPU kernels have separate scheduling."})}
    };
    j["recommended"] = nullptr;
    if (rec) {
        // The shape Dream's reader takes (DREAM-179, dream/local/model_defaults.py recommended_modes): the modes as
        // top-level "thinking" / "instruct" objects with label, when, temp, top_p, top_k, min_p, presence, repeat and
        // max_output; "effort" as {level: note}; "note"; "source_url". Unstated fields carry the neutral value.
        auto mode = [&](const RecommendedMode& m, const char* label, bool thinking) {
            json o{{"label",label},{"when",m.when},{"temp",m.temperature},{"top_p",m.top_p},{"top_k",m.top_k},
                   {"min_p",m.min_p},{"presence",m.presence_penalty},{"repeat",m.repeat_penalty}};
            if (thinking && rec->max_reasoning) o["max_output"] = {{"reasoning",rec->max_reasoning},{"answer",rec->max_output}};
            else if (rec->max_output && (thinking || rec->max_output_instruct)) o["max_output"] = rec->max_output;
            return o;
        };
        json effort = json::object();
        for (const auto& e : rec->effort) effort[e.level] = e.when;
        j["recommended"] = {
            {"model",rec->model},{"source_url",rec->source},
            {"thinking",mode(rec->thinking,"Thinking",true)},
            {"instruct",rec->instruct ? mode(*rec->instruct,rec->instruct_label,false) : json(nullptr)},
            {"default_mode",!resolved.enable_thinking && rec->instruct ? "instruct" : "thinking"},
            {"effort",effort},{"max_output_note",rec->max_output_note},
            {"context",rec->context},{"note",rec->note},
            {"resolution","request > CLI/env > the recommendation for the request's mode > library default; "
                          "a request or server default of temperature 0 (greedy) takes nothing from it"}};
    }
    if(reasoning.thinking)j["load"].push_back("thinking");
    if (arch == ModelArch::kMimo26 || arch == ModelArch::kDeepSeek41) {   // the auto expert tier honours a VRAM headroom knob;
        j["load"].push_back("vram_reserve_gib");                              // Dream sends the advertised default EXPLICITLY, so it
        j["defaults"]["vram_reserve_gib"] = arch == ModelArch::kMimo26 ? 1.5 : 6.0;   // must equal each engine's own (mimo26_forward.cpp
    }                                                                         // 1.5; Ds41Forward::ResidentOptions::vram_reserve 6 GiB)
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
