#include "ie/server_capabilities.hpp"
#include "ie/glm5_server.hpp"   // kGlm5PrefillChunk
#include "ie/engine.hpp"
#include "ie/gguf.hpp"
#include "ie/dense_fleet_rules.hpp"   // dense_fleet_loads: which dense files keep a prompt cache (2026-10-28, serving campaign P1)
#include "ie/model_config.hpp"        // read_dense_config_auto
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
    if(inkling2_only_effort(effort) && !Engine::inkling2_dir(path))return kUnknownEffortCli;   // refused as before these names parsed
    if(Engine::ds41_dir(path))return reasoning_effort_error(reasoning_capabilities(ModelArch::kDeepSeek41),effort);
    if(Engine::mimo26_dir(path))return reasoning_effort_error(reasoning_capabilities(ModelArch::kMimo26),effort);
    if(Engine::hy4flash_dir(path))return reasoning_effort_error(reasoning_capabilities(ModelArch::kHy4Flash),effort);
    if(Engine::inkling2_dir(path))return reasoning_effort_error(reasoning_capabilities(ModelArch::kInkling2),effort);
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
        case ModelArch::kGlm5Next: case ModelArch::kDeepSeek41: case ModelArch::kMimo26:
        case ModelArch::kLaguna: case ModelArch::kInkling: case ModelArch::kHy4Flash: case ModelArch::kXing4: case ModelArch::kInkling2: return true;
        default: return false;
    }
}
bool server_streams_experts(ModelArch arch) noexcept {
    return arch == ModelArch::kDeepSeek4 || arch == ModelArch::kQwen4Exp ||
           arch == ModelArch::kGlm5Next || arch == ModelArch::kDeepSeek41 || arch == ModelArch::kMimo26 ||
           arch == ModelArch::kLaguna || arch == ModelArch::kInkling || arch == ModelArch::kHy4Flash;
}
// 2026-11-21 (owner request): the launch that is used and tested on this machine (2 x Arc Pro B70, 256 GB RAM), every
// load field a client's model screen shows, each as {value, why, source}, so Dream's "Use recommendations" can fill the
// whole form instead of only the sampler. The values are docs/MODEL_SETTINGS.md's "Launch (tested)" rows; a model
// without a tested launch gets null rather than a guess. file_type: the GGUF's general.file_type (7 Q8_0, 15 Q4_K_M,
// 16 Q5_K_S, 17 Q5_K_M, 18 Q6_K); -1 when not a GGUF.
static nlohmann::json recommended_launch(ModelArch arch, int64_t file_type, const std::string& model_name) {
    using nlohmann::json;
    auto s = [](json v, std::string why, std::string src) { return json{{"value", std::move(v)}, {"why", std::move(why)}, {"source", std::move(src)}}; };
    const std::string doc = "docs/MODEL_SETTINGS.md, tested launch ";
    auto contains = [&](const char* k) { return model_name.find(k) != std::string::npos; };
    json L;
    switch (arch) {
    case ModelArch::kQwen35Dense: {
        const bool q8 = file_type == 7 || file_type < 0, q5 = file_type == 16 || file_type == 17;
        const std::string src = doc + "2026-10-18";
        L = {{"gpus", s(2, "the layer split over both cards; tensor parallel gives nothing on this dense model", src)},
             {"ctx", s(q8 ? 222222 : 250000, q8 ? "the most that fits beside the Q8_0 weights with the request lanes" : "the smaller Q6_K / Q5_K file leaves room for more context", src)},
             {"parallel", s(q5 ? 16 : 8, "request lanes decode together through the card pipe: more total tokens per second (one request ~23-29 tok/s, three together ~15 each)", src)},
             {"slot_ctx", s(q8 ? 60000 : 50000, "context of each extra lane; lane 0 keeps the full ctx", src)},
             {"prefill_chunk", s(2048, "prompt rows read per forward: the measured prompt-read rate (2,962-3,001 tok/s on Q8_0) is at 2048", src)},
             {"thinking", s(true, "the daily mode; requests can still turn it off", src)},
             {"reasoning_effort", s("xhigh", "the tested launch's effort", src)},
             {"spec", s(true, "the model's own MTP draft head, on every lane; exact at any temperature", src)},
             {"spec_k", s(3, "3 drafted tokens a round is the head's default and the tested setting", src)},
             {"int8_kv", s(false, "measured slower on this dense model and forces one card and one lane", "docs/MODEL_SETTINGS.md, Known")}};
        break; }
    case ModelArch::kQwen4Exp: {
        const std::string src = doc + "2026-10-23";
        L = {{"gpus", s(2, "the model and its expert cache need both cards", src)},
             {"ctx", s(262000, "the tested context; lane 0's buffers grow on demand, so unused context costs no expert slots", src)},
             {"parallel", s(4, "four request lanes (use 1 for a single conversation at a time)", src)},
             {"slot_ctx", s(75000, "context of each extra lane", src)},
             {"prefill_chunk", s(2048, "the tested prompt-read chunk", src)},
             {"thinking", s(true, "the daily mode", src)},
             {"reasoning_effort", s("xhigh", "the tested launch's effort", src)},
             {"spec", s(true, "the MTP draft head (needs mtp-head.gguf beside the model); exact at any temperature", src)},
             {"spec_k", s(3, "the certified draft length", src)},
             {"int8_kv", s(false, "not available on this model", "ie capabilities")}};
        break; }
    case ModelArch::kQwen35Moe: {
        const std::string src = doc + "2026-10-17";
        L = {{"gpus", s(2, "both cards", src)}, {"ctx", s(262144, "the tested context", src)},
             {"parallel", s(6, "six request lanes", src)}, {"slot_ctx", s(40000, "context of each extra lane", src)},
             {"prefill_chunk", s(2048, "the tested prompt-read chunk", src)}, {"thinking", s(true, "the daily mode", src)},
             {"spec", s(false, "not part of the tested launch (prompt-lookup drafts exist; not measured at this launch)", src)},
             {"int8_kv", s(false, "forces one card and one lane", "ie capabilities")}};
        break; }
    case ModelArch::kDeepSeek41: {
        const std::string src = doc + "2026-10-20";
        L = {{"gpus", s(2, "both cards; experts live in system RAM", src)}, {"ctx", s(265000, "the tested context (500K starved the expert reserve and thrashed)", src)},
             {"parallel", s(4, "four request lanes", src)}, {"slot_ctx", s(75000, "context of each extra lane", src)},
             {"prefill_chunk", s(2048, "the tested prompt-read chunk", src)}, {"thinking", s(true, "the daily mode", src)},
             {"reasoning_effort", s("high", "the tested launch's effort; the card uses max for its agent/coding benchmark", src)},
             {"vram_reserve_gib", s(0, "a reserve trades decode speed for nothing on this machine; never raise it", src)}};
        break; }
    case ModelArch::kMimo26: {
        const std::string src = doc + "2026-10-14";
        const bool base = !contains("UNCENSORED") && contains("Base");
        L = {{"gpus", s(2, "both cards; experts in system RAM", src)}, {"ctx", s(base ? 200000 : 250000, "the tested context", src)},
             {"parallel", s(1, "the tested launch serves one conversation at a time", src)}, {"slot_ctx", s(75000, "context of an extra lane if parallel is raised", src)},
             {"prefill_chunk", s(2048, "the tested prompt-read chunk", src)}, {"thinking", s(true, "the daily mode", src)},
             {"vram_reserve_gib", s(base ? 0.5 : 1.0, "headroom the expert tier leaves on each card at this launch", src)}};
        break; }
    case ModelArch::kGlm5Next: {
        const std::string src = doc + "2026-10-04";
        L = {{"gpus", s(2, "at most two layer stages", src)}, {"ctx", s(264000, "the tested context", src)},
             {"parallel", s(1, "the GLM server resets state per request; one at a time", src)},
             {"prefill_chunk", s(1024, "GLM's own chunk: 1024, not 2048", src)}, {"thinking", s(true, "GLM always reasons", src)},
             {"reasoning_effort", s("max", "the tested launch's effort", src)}};
        break; }
    case ModelArch::kDeepSeek4: {
        const std::string src = doc + "2025-09 (measured on an earlier driver stack)";
        const bool vision = contains("Vision") || contains("vision");
        L = {{"gpus", s(2, "both cards", src)}, {"ctx", s(300000, "the tested context", src)},
             {"parallel", s(1, "images work at one lane only", src)},
             {"prefill_chunk", s(vision ? 512 : 2048, vision ? "the vision variant reads prompts in 512-row chunks" : "the tested prompt-read chunk", src)},
             {"thinking", s(true, "the daily mode", src)}, {"reasoning_effort", s("high", "the tested launch's effort", src)}};
        break; }
    case ModelArch::kHy4Flash: {
        const std::string src = "README v0.2.44 (2026-11-19)";
        L = {{"gpus", s(2, "both cards; experts in system RAM", src)}, {"ctx", s(262144, "the measured context", src)},
             {"parallel", s(1, "one request at a time is what the backend serves", src)}, {"thinking", s(true, "the measured mode", src)}};
        break; }
    case ModelArch::kLlama3:
        if (contains("Nanbeige")) {
            const std::string src = doc + "2026-10-29";
            L = {{"gpus", s(2, "two cards: 28-29 tok/s (one card fits only ctx 8192, 19.3 tok/s)", src)},
                 {"ctx", s(65536, "the tested context on two cards", src)},
                 {"parallel", s(1, "no request lanes on this path yet; more requests queue", src)},
                 {"thinking", s(true, "the daily mode", src)}, {"spec", s(true, "prompt-lookup drafts: repeated text 2.7x faster, other text unchanged", src)}};
        }
        break;
    default: break;
    }
    if (L.is_null()) return nullptr;
    // The fields a request carries rather than the load: one answer for every model, with the reason.
    L["repeat_last_n"] = s(64, "how far back the presence / frequency / repetition penalties look; only matters when one of them is "
                              "non-zero (the instruct mode's presence 1.5). No card names a value; the engine default", "engine default");
    L["frequency_penalty"] = s(0, "no card recommends one", "engine default");
    L["seed"] = s(0, "0 = a new random seed each request; set a number only to reproduce a reply exactly", "engine default");
    L["stop"] = s(json::array(), "the chat template ends replies itself; extra stop strings can cut tool calls", "engine default");
    L["prompt_cache"] = s(true, "later turns reuse the read prompt instead of re-reading it", "engine default");
    return L;
}

std::string server_capabilities_json(const std::string& model_path) {
    using nlohmann::json;
    int64_t file_type = -1;
    ModelArch arch = ModelArch::kUnknown;
    std::string name;
    std::string chat_template, model_name, base_repo;
    bool dense_fleet_file = true;   // a dense GGUF the fleet class loads (the prompt cache's path); see below
    if (!model_path.empty() && Engine::ds41_dir(model_path)) { arch = ModelArch::kDeepSeek41; name = "deepseek_v41"; }
    else if (!model_path.empty() && Engine::mimo26_dir(model_path)) { arch = ModelArch::kMimo26; name = "mimo_v2"; }
    else if (!model_path.empty() && Engine::hy4flash_dir(model_path)) { arch = ModelArch::kHy4Flash; name = "hy_v4_flash"; }
    else if (!model_path.empty() && Engine::inkling2_dir(model_path)) { arch = ModelArch::kInkling2; name = "inkling2_loop"; }
    else if (!model_path.empty()) {
        GgufReader g;
        if (auto e = g.open(model_path); !e.empty()) throw std::runtime_error("gguf: " + e);
        arch = detect_arch(g);
        if (auto a = g.find_kv("general.architecture")) name = a->as_string();
        // one name per runtime: clients (Dream's launcher and model defaults) key on "glm5next";
        // Unsloth's 2026-10-05 rewrite of the GLM-5.3-Flash GGUF says "glm5-next"
        if (arch == ModelArch::kGlm5Next) name = "glm5next";
        if (auto t=g.find_kv("tokenizer.chat_template");t && t->type==GgufValueType::kString)chat_template=t->as_string();
        // P4 B20: what recommended_sampling matches the exact weights on
        if (auto t=g.find_kv("general.name");t && t->type==GgufValueType::kString)model_name=t->as_string();
        if (auto t=g.find_kv("general.base_model.0.repo_url");t && t->type==GgufValueType::kString)base_repo=t->as_string();
        if (auto t=g.find_kv("general.file_type"))file_type=t->as_int();
        // 2026-10-28 (serving campaign P1, its gate's finding): a dense file keeps a prompt cache only when it runs on the
        // fleet class, which loads the dtypes and geometry of dense_fleet_rules.hpp (a Q8_0 / Q5_K file stays on the
        // cache-less DenseModel). --int8-kv is a load option, not known here: it also keeps the old path.
        if (is_dense_arch(arch)) {
            DenseConfig dc;
            dense_fleet_file = read_dense_config_auto(g, dc).empty() &&
                               dense_fleet_loads(g, dc.n_layers, arch == ModelArch::kLlama3, dc.n_q_heads * dc.head_dim, dc.ffn);
        }
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
    // 2026-10-28 (serving campaign P1): the dense archs (llama / qwen3 GGUFs, Nanbeige4.3-33B) keep a FleetPrefixCache on
    // the fleet class (one card or tensor-parallel; --int8-kv and the F16-expansion files stay on the cache-less DenseModel).
    const bool cache = generic || (is_dense_arch(arch) && dense_fleet_file) || arch == ModelArch::kQwen35Moe ||
        arch == ModelArch::kQwen35Dense || arch == ModelArch::kQwen3Next ||
        arch == ModelArch::kGemma4 || arch == ModelArch::kQwen4Exp ||
        arch == ModelArch::kDeepSeek4 || arch == ModelArch::kDeepSeek41 || arch == ModelArch::kMimo26 ||
        arch == ModelArch::kLaguna || arch == ModelArch::kInkling || arch == ModelArch::kGlm5Next || arch == ModelArch::kHy4Flash ||
        arch == ModelArch::kXing4 || arch == ModelArch::kInkling2;
    // v0.2.21: the 35B-A3B class (lookup drafts on the request lanes, P4 B55) and Flash-Next (lookup drafts at one lane, B56; its
    // MTP head only with a --spec-head path, B63) take --spec too, so a client's load screen can offer it.
    // 2026-10-28 (serving campaign P2): the dense fleet class takes --spec too (prompt-lookup drafts at one lane).
    const bool spec = generic || (is_dense_arch(arch) && dense_fleet_file) || arch == ModelArch::kQwen35Dense || arch == ModelArch::kGemma4 ||
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
                               arch==ModelArch::kQwen35Dense || arch==ModelArch::kQwen35Moe || arch==ModelArch::kHy4Flash}}},
        {"defaults",{{"temperature",r6(sp.temperature)},{"top_k",sp.top_k},{"top_p",r6(sp.top_p)},{"min_p",r6(sp.min_p)},
                     {"repeat_penalty",r6(sp.repeat_penalty)},{"repeat_last_n",sp.repeat_window},{"presence_penalty",r6(sp.presence_penalty)},
                     {"frequency_penalty",r6(sp.frequency_penalty)},{"seed",sp.seed},
                     {"max_tokens",sp.max_tokens==kMaxTokensUnlimited?0u:sp.max_tokens},
                     {"stop",json::array()},{"thinking",default_thinking},{"threads",0},
                     {"prefill_chunk",glm?kGlm5PrefillChunk:256u},{"parallel",1},{"slot_ctx",0},{"prompt_cache",cache}}},
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
    // directory models (DS4.1 / MiMo / Hy4) have no general.name: match on the folder name instead
    j["recommended_launch"] = generic ? json(nullptr) : recommended_launch(arch, file_type, model_name.empty() ? model_path : model_name);
    // the reply limit: the publisher's answer budget where the card states one (a smaller cap cuts long files and tool
    // calls off mid-way, which a client then sees as a broken call)
    if (j["recommended_launch"].is_object() && rec && rec->max_output)
        j["recommended_launch"]["max_tokens"] = {{"value", rec->max_output},
            {"why", "the model card's answer budget; a lower cap cuts long files and tool calls off mid-way"},
            {"source", (rec->source && *rec->source) ? std::string(rec->source) : std::string("model card")}};
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
