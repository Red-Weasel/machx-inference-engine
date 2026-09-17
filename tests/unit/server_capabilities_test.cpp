#undef NDEBUG
#include "ie/server_capabilities.hpp"
#include "nlohmann/json.hpp"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <unistd.h>
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
 auto check=[](const std::string& arch, bool supported, const std::string& tmpl="") {
  const std::string path="/tmp/ie-capabilities-"+std::to_string(getpid())+".gguf";
  std::ofstream f(path,std::ios::binary);
  auto w=[&](auto x){f.write(reinterpret_cast<const char*>(&x),sizeof x);};
  auto s=[&](const std::string& x){w(uint64_t(x.size()));f.write(x.data(),x.size());};
  f.write("GGUF",4); w(uint32_t(3));w(uint64_t(0));w(uint64_t(tmpl.empty()?1:2));
  s("general.architecture");w(uint32_t(8));s(arch);
  if(!tmpl.empty()){s("tokenizer.chat_template");w(uint32_t(8));s(tmpl);}
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
 bool failed=false;try{ie::server_capabilities_json("/does/not/exist.gguf");}catch(...){failed=true;}
 assert(failed);std::puts("server_capabilities_test: OK");
}
