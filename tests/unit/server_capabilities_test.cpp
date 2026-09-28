#undef NDEBUG
#include "ie/server_capabilities.hpp"
#include "ie/recommended_sampling.hpp"
#include "nlohmann/json.hpp"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>
using nlohmann::json;
int main() {
 using ie::ModelArch;
 assert(ie::server_supports_arch(ModelArch::kGlm5Next));
 assert(!ie::server_supports_arch(ModelArch::kGlmDsa));
 assert(!ie::server_supports_arch(ModelArch::kUnknown));
 assert(!ie::server_supports_arch(ModelArch::kHyv4));
 assert(ie::server_streams_experts(ModelArch::kGlm5Next));
 assert(!ie::server_streams_experts(ModelArch::kLlama3));
 auto general=json::parse(ie::server_capabilities_json());
 assert(general.at("schema_version")==1);
 assert(general.at("features").at("context_shift")==false);
 // A recognized architecture is not proof of a server backend. Exercise GGUF
 // metadata detection, not a parallel test-only architecture string parser.
 auto check=[](const std::string& arch, bool supported, const std::string& tmpl="",
               const std::vector<std::pair<std::string,std::string>>& kv={}) {
  const std::string path="/tmp/ie-capabilities-"+std::to_string(getpid())+".gguf";
  std::ofstream f(path,std::ios::binary);
  auto w=[&](auto x){f.write(reinterpret_cast<const char*>(&x),sizeof x);};
  auto s=[&](const std::string& x){w(uint64_t(x.size()));f.write(x.data(),x.size());};
  f.write("GGUF",4); w(uint32_t(3));w(uint64_t(0));w(uint64_t((tmpl.empty()?1:2)+kv.size()));
  s("general.architecture");w(uint32_t(8));s(arch);
  if(!tmpl.empty()){s("tokenizer.chat_template");w(uint32_t(8));s(tmpl);}
  for(const auto& [k,v]:kv){s(k);w(uint32_t(8));s(v);}
  while(f.tellp()%32)f.put(0);f.close();
  auto j=json::parse(ie::server_capabilities_json(path));
  if(arch=="glm5next") {
   assert(ie::server_reasoning_error(path,"low").empty());
   assert(!ie::server_reasoning_error(path,"medium").empty());
  }
  std::remove(path.c_str());
  assert(j.at("architecture")==arch);assert(j.at("supported")==supported);
  if(arch=="glm5next") {
   assert(j.at("reasoning").at("effort_levels")==json({"low","high","max"}));
   assert(j.at("defaults").at("reasoning_effort")==
       (std::getenv("IE_SERVE_REASONING_EFFORT")?std::getenv("IE_SERVE_REASONING_EFFORT"):"max"));
   assert(j.at("memory_planner")=="streaming");assert(j.at("max_gpus")==2);
   assert(!j.at("features").at("prompt_cache").get<bool>());
   assert(!j.at("features").at("speculative").get<bool>());
  }
  return j;
 };
 unsetenv("IE_G5_PIN_BANKS");unsetenv("IE_G5_ECACHE_MB");
 unsetenv("IE_G5_PIN_MAX_GIB");unsetenv("IE_G5_PIN_FLOOR_GIB");unsetenv("IE_G5_PIN_OVERHEAD");
 auto memory=check("glm5next",true)["memory_policy"];
 assert(memory["host_banks"]=="pinned_auto" && memory["expert_cache_bytes"]==0);
 assert(memory["pin_max_bytes"].is_null());
 setenv("IE_G5_PIN_BANKS","0",1);setenv("IE_G5_ECACHE_MB","10240",1);
 memory=check("glm5next",true)["memory_policy"];
 assert(memory["host_banks"]=="mmap" && memory["expert_cache_bytes"]==(10ull<<30));
 setenv("IE_G5_PIN_MAX_GIB","0",1);
 assert(check("glm5next",true)["memory_policy"]["pin_max_bytes"]==0);
 unsetenv("IE_G5_PIN_BANKS");unsetenv("IE_G5_ECACHE_MB");unsetenv("IE_G5_PIN_MAX_GIB");
 check("glm-dsa",false);check("clip",false);
 auto oss=check("gpt-oss",true);
 assert(oss["reasoning"]["effort_levels"]==json({"low","medium","high"}));
 auto has=[](const json& j,const char* key){for(const auto& item:j["load"])if(item==key)return true;return false;};
 assert(!has(oss,"thinking") && has(oss,"reasoning_effort"));
 auto coder=check("qwen3moe",true,"plain instruct");
 assert(!has(coder,"thinking") && !has(coder,"reasoning_effort"));
 auto think=check("qwen3moe",true,"<think> enable_thinking");
 assert(has(think,"thinking") && !has(think,"reasoning_effort"));
 auto q4=check("qwen4exp",true,"<think> reasoning_effort Reasoning effort is set to xhigh.");
 assert(q4["reasoning"]["effort_levels"]==json({"low","medium","high","xhigh"}));
 auto ds=check("deepseek4",true);
 assert(ds["reasoning"]["effort_levels"]==json({"low","high","max"}));
 assert(ds["defaults"]["reasoning_effort"]=="low");
 setenv("IE_SERVE_REASONING_EFFORT","high",1);
 setenv("IE_SERVE_NO_THINK","1",1);
 auto configured=check("glm5next",true);
 assert(configured["defaults"]["reasoning_effort"]=="high");
 assert(configured["reasoning"]["default_effort"]=="high");
 assert(configured["defaults"]["thinking"]==false);
 unsetenv("IE_SERVE_REASONING_EFFORT");unsetenv("IE_SERVE_NO_THINK");
 // P4 B20: "defaults" are the real per-model defaults; "recommended" the publisher's table in the shape Dream's reader
 // takes (DREAM-179): top-level "thinking"/"instruct" with label, when, temp, top_p, top_k, min_p, presence, repeat,
 // max_output; "effort" {level: note}; "note"; "source_url"
 assert(general["recommended"].is_null() && general["defaults"]["temperature"]==0.7 && general["defaults"]["top_k"]==40);
 assert(ds["recommended"].is_null() && ds["defaults"]["top_k"]==40);   // no row: library defaults
 const std::string q38="<think> reasoning_effort Reasoning effort is set to xhigh.";
 auto q27=check("qwen35",true,q38);
 const auto& rq=q27["recommended"];
 assert(rq["model"]=="Qwen3.8-27B" && rq["source_url"]=="https://huggingface.co/Qwen/Qwen3.8-27B" && !rq["note"].get<std::string>().empty());
 assert(rq["default_mode"]=="thinking" && rq["context"]==262144);
 const auto& th=rq["thinking"]; const auto& in=rq["instruct"];
 assert(th["label"]=="Thinking" && th["temp"]==1.0 && th["top_p"]==0.95 && th["top_k"]==20 && th["min_p"]==0.0 &&
        th["presence"]==0.0 && th["repeat"]==1.0 && !th["when"].get<std::string>().empty());
 assert(th["max_output"]==json({{"reasoning",262144},{"answer",131072}}));
 assert(in["label"]=="Instruct" && in["temp"]==0.7 && in["top_p"]==0.8 && in["presence"]==1.5 && in["max_output"]==131072 &&
        !in["when"].get<std::string>().empty());
 assert(rq["effort"].is_object() && rq["effort"].size()==3 && !rq["effort"]["xhigh"].get<std::string>().empty() &&
        rq["effort"].contains("medium") && rq["effort"].contains("low"));
 const auto& dq=q27["defaults"];
 assert(dq["temperature"]==1.0 && dq["top_p"]==0.95 && dq["top_k"]==20 && dq["presence_penalty"]==0.0 && dq["min_p"]==0.0);
 assert(dq["prefill_chunk"]==256 && dq["max_tokens"]==16384 && dq["thinking"]==true);   // load keys and max_tokens unchanged
 setenv("IE_SERVE_NO_THINK","1",1);   // server default instruct -> the instruct row is the default
 auto q27i=check("qwen35",true,q38);
 assert(q27i["recommended"]["default_mode"]=="instruct" && q27i["defaults"]["temperature"]==0.7 &&
        q27i["defaults"]["top_p"]==0.8 && q27i["defaults"]["presence_penalty"]==1.5);
 unsetenv("IE_SERVE_NO_THINK");
 setenv("IE_SERVE_TEMP","0.4",1);   // env outranks the recommendation
 auto q27e=check("qwen35",true,q38);
 assert(q27e["defaults"]["temperature"]==0.4 && q27e["defaults"]["top_k"]==20);
 unsetenv("IE_SERVE_TEMP");
 const std::string q36="https://huggingface.co/Qwen/Qwen3.6-35B-A3B";
 auto moe=check("qwen35moe",true,"<think> enable_thinking",{{"general.name","Ours"},{"general.base_model.0.repo_url",q36}});
 // the plain Qwen3.6-35B-A3B (same template as the distill): no row, library defaults
 auto base=check("qwen35moe",true,"<think> enable_thinking",{{"general.name","Qwen_Qwen3.6 35B A3B"},{"general.base_model.0.repo_url",q36}});
 assert(base["recommended"].is_null() && base["defaults"]["temperature"]==0.7 && base["defaults"]["top_k"]==40);
 assert(check("qwen35moe",true,"<think> enable_thinking")["recommended"].is_null());
 assert(moe["recommended"]["thinking"]["temp"]==0.6 && moe["recommended"]["thinking"]["max_output"]==16384);
 assert(!moe["recommended"]["instruct"].contains("max_output"));   // the distill states no instruct cap
 assert(moe["recommended"]["effort"]==json::object() && moe["defaults"]["temperature"]==0.6);
 auto fnx=check("qwen4exp",true,q38);
 assert(fnx["recommended"]["source_url"]=="https://huggingface.co/Qwen/Qwen3.8-Flash-Next");
 assert(check("glm5next",true)["recommended"].is_null());   // GLM by name only
 auto g=check("glm5next",true,"",{{"general.name","GLM 5.3 Flash"}});
 assert(g["recommended"]["instruct"].is_null() && !g["recommended"]["thinking"].contains("max_output"));
 assert(g["recommended"]["effort"].contains("max") && g["defaults"]["top_p"]==0.95 && g["defaults"]["top_k"]==0);
 // the directory-loaded DS4.1 / MiMo (a config.json naming the arch is what selects them)
 auto dir_caps=[](const char* cfg) {
  const std::string d="/tmp/ie-capabilities-dir-"+std::to_string(getpid());
  mkdir(d.c_str(),0700); std::ofstream(d+"/config.json")<<cfg;
  auto j=json::parse(ie::server_capabilities_json(d));
  std::remove((d+"/config.json").c_str()); rmdir(d.c_str());
  return j;
 };
 auto v41=dir_caps(R"({"model_type":"deepseek_v41"})");
 assert(v41["recommended"]["instruct"]["label"]=="Chat" && v41["recommended"]["thinking"]["max_output"]==262144);
 assert(v41["recommended"]["thinking"]["top_k"]==0 && v41["defaults"]["temperature"]==1.0 && v41["defaults"]["top_k"]==0);
 assert(v41["recommended"]["effort"].size()==3);
 auto mimo=dir_caps(R"({"model_type":"mimo_v2"})");
 assert(mimo["recommended"]["instruct"]["temp"]==1.0 && mimo["recommended"]["effort"]==json::object());
 assert(!mimo["recommended"]["thinking"].contains("max_output") && mimo["recommended"]["context"]==1048576);
 bool failed=false;try{ie::server_capabilities_json("/does/not/exist.gguf");}catch(...){failed=true;}
 assert(failed);std::puts("server_capabilities_test: OK");
}
