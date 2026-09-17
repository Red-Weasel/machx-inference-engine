#undef NDEBUG
#include "ie/glm5next.hpp"
#include "ie/glm5_memory_policy.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace ie;
struct FixtureTensor { std::string name; uint32_t type; };
int main(int argc,char** argv) {
    if(argc==2) { // optional real-file metadata probe; no GPU initialization
        GgufReader g; auto error=g.open(argv[1]); assert(error.empty());
        Glm5NextConfig c;error=read_glm5next_config(g,c);assert(error.empty());
        auto hi=c.n_transformer_layers(), split=(hi+1)/2;
        for(unsigned stage=0;stage<2;++stage) {
            uint64_t bytes=0;error=Glm5NextModel::plan_device_weights(g,c,stage?split:0,stage?hi:split,bytes);
            if(!error.empty()){std::cerr<<error<<'\n';return 1;}
            std::cout<<"stage "<<stage<<": "<<bytes<<" bytes ("<<bytes/1073741824.<<" GiB)\n";
        }
        return 0;
    }
    unsetenv("IE_G5_MTP");unsetenv("IE_G5_DENSE_Q8");
    std::vector<FixtureTensor> tensors;
    uint64_t expected[2]={}, q8count[2]={};
    auto add=[&](unsigned stage,const std::string& name,uint32_t source,unsigned destination) {
        tensors.push_back({name,source});expected[stage]+=destination;
        if(source==8 && destination==34)++q8count[stage];
    };
    add(0,"token_embd.weight",1,64);add(1,"output.weight",8,34);add(1,"output_norm.weight",1,128);
    for(unsigned l=0;l<2;++l) {
        auto put=[&](const char* n,unsigned src,unsigned dst){add(l,"blk."+std::to_string(l)+"."+n,src,dst);};
        for(auto n:{"hc_attn_fn.weight","hc_attn_base.weight","hc_attn_scale.weight","hc_ffn_fn.weight","hc_ffn_base.weight","hc_ffn_scale.weight","attn_norm.weight","ffn_norm.weight"})put(n,1,128);
        if(l==0) {
            for(auto n:{"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight","ssm_f_a.weight","ssm_f_b.weight","ssm_g_a.weight","ssm_g_b.weight","ssm_beta.weight"})put(n,8,34);
            for(auto n:{"ssm_conv1d_q.weight","ssm_conv1d_k.weight","ssm_conv1d_v.weight","ssm_norm.weight"})put(n,1,64);
            for(auto n:{"ssm_a","ssm_dt.bias"})put(n,1,128);
            for(auto n:{"ffn_gate.weight","ffn_up.weight","ffn_down.weight"})put(n,8,34);
        } else {
            for(auto n:{"attn_q_a.weight","attn_q_b.weight","attn_kv_a_mqa.weight","attn_output.weight","ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"})put(n,8,34);
            for(auto n:{"attn_q_a_norm.weight","attn_kv_a_norm.weight","indexer.k_norm.weight","indexer.k_norm.bias","indexer.proj.weight","indexer_compressor_ape.weight","ffn_gate_inp.weight","exp_probs_b.bias"})put(n,1,128);
            for(auto n:{"attn_k_b.weight","attn_v_b.weight","indexer.attn_k.weight","indexer.attn_q_b.weight","indexer_compressor_gate.weight"})put(n,1,64);
            for(auto n:{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"})put(n,1,0);
        }
    }
    const std::string path="/tmp/glm5-weight-plan-"+std::to_string(getpid())+".gguf";
    auto write=[&] {
        std::ofstream f(path,std::ios::binary|std::ios::trunc);
        auto w=[&](auto v){f.write(reinterpret_cast<const char*>(&v),sizeof v);};
        auto str=[&](const std::string& s){w(uint64_t(s.size()));f.write(s.data(),s.size());};
        f.write("GGUF",4);w(uint32_t(3));w(uint64_t(tensors.size()));w(uint64_t(0));
        uint64_t offset=0;
        for(const auto& t:tensors){str(t.name);w(uint32_t(2));w(uint64_t(32));w(uint64_t(1));w(t.type);w(offset);offset+=64;}
        while(f.tellp()%32)f.put(0);
        for(uint64_t i=0;i<offset;++i)f.put(0);
    };
    write();Glm5NextConfig c;c.n_layers=2;c.leading_dense=1;c.attn_kind={0,1};c.n_experts=1;
    {
        GgufReader g;assert(g.open(path).empty());
        for(unsigned stage=0;stage<2;++stage) {
            uint64_t bytes=0;assert(Glm5NextModel::plan_device_weights(g,c,stage,stage+1,bytes).empty());
            assert(bytes==expected[stage]);
        }
        uint64_t bytes=0;assert(Glm5NextModel::plan_device_weights(g,c,0,2,bytes).empty());
        assert(bytes==expected[0]+expected[1]);
        setenv("IE_G5_DENSE_Q8","0",1);
        assert(Glm5NextModel::plan_device_weights(g,c,0,2,bytes).empty());
        assert(bytes==expected[0]+expected[1]+(q8count[0]+q8count[1])*30);
        unsetenv("IE_G5_DENSE_Q8");
        // Invalid ranges fail without creating an allocator or touching weights.
        assert(!Glm5NextModel::plan_device_weights(g,c,1,1,bytes).empty());
    }
    tensors.erase(tensors.begin());write();
    {GgufReader g;assert(g.open(path).empty());uint64_t bytes=123;
     assert(!Glm5NextModel::plan_device_weights(g,c,0,2,bytes).empty());assert(bytes==0);}
    std::remove(path.c_str());std::cout<<"glm5_weight_plan_test: OK\n";
}
