#undef NDEBUG
#include "ie/glm5_server.hpp"
#include "ie/openai_proto.hpp"
#include <cassert>
#include <iostream>
int main() {
    ie::Glm5NextConfig c;
    c.n_layers=46; c.nextn_predict_layers=1; c.hidden=4096;
    c.n_q_heads=64; c.kda_head_dim=128; c.kv_lora_rank=512;
    c.indexer_kpool=4; c.indexer_head_dim=128;
    c.indexer_top_k=2048; c.hc_count=4; c.conv_kernel=4;
    c.attn_kind.resize(46); for(unsigned i=0;i<46;i+=4)c.attn_kind[i]=1;
    c.expert_ffn=2048; c.ffn=12288; c.n_experts_used=8;
    auto a=ie::glm5_runtime_reserve(c,8192,256,0,23);
    auto b=ie::glm5_runtime_reserve(c,200000,256,0,23);
    assert(b>a);
    assert(ie::glm5_runtime_reserve(c,200000,1024,0,23)>b);
    assert(!ie::glm5_residency_fits(20ull<<30,4ull<<30,10ull<<30,32ull<<30));
    assert(ie::glm5_residency_fits(6ull<<30,4ull<<30,10ull<<30,30ull<<30));
    assert(!ie::glm5_residency_fits(UINT64_MAX,4,10,32));
    std::string err;
    std::vector<ie::ChatTurn> turns={{"user","Hello",{}}};
    assert(ie::build_glm5_prompt(turns,true,"",err)=="[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hello<|assistant|><think>");
    for(auto effort:{"low","high","max"}) {
        const auto request=ie::oai::parse_chat_request(std::string("{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"reasoning_effort\":\"")+effort+"\",\"enable_thinking\":true}");
        assert(request.error.empty());
        auto p=ie::build_glm5_prompt(request.turns,request.enable_thinking,"",err,request.reasoning_effort);
        std::string title=effort;title.front()-='a'-'A';
        assert(err.empty() && p=="[gMASK]<sop><|system|>Reasoning Effort: "+title+"<|user|>Hello<|assistant|><think>");
        auto off=ie::build_glm5_prompt(turns,false,"",err,effort);
        assert(off==p+"</think>");
    }
    assert(ie::build_glm5_prompt(turns,true,"",err,"medium").empty() && !err.empty());
    // a reply cut mid call (max_tokens): an earlier complete call is converted, the cut one stays as the model wrote it
    {
        std::vector<ie::ChatTurn> cut={{"user","look",{}},{"assistant",R"(<tool_call>{"name":"read","arguments":{"path":"a.txt"}}</tool_call>
Next <tool_call>read<arg_key>pa)",{}},{"user","go on",{}}};
        std::string cut_err;
        const auto p=ie::build_glm5_prompt(cut,false,"",cut_err);
        assert(cut_err.empty());
        assert(p.find("<tool_call>read<arg_key>path</arg_key><arg_value>a.txt</arg_value></tool_call>")!=std::string::npos);
        assert(p.find("Next <tool_call>read<arg_key>pa<|user|>go on")!=std::string::npos);
    }
    turns={{"assistant",R"(<tool_call>{"name":"read","arguments":{"path":"a.txt","n":2}}</tool_call>)",{}},{"tool","found",{}},{"tool","other",{}}};
    auto prompt=ie::build_glm5_prompt(turns,false,"",err);
    assert(err.empty());
    assert(prompt.find("<tool_call>read<arg_key>path</arg_key><arg_value>a.txt</arg_value><arg_key>n</arg_key><arg_value>2</arg_value></tool_call>")!=std::string::npos);
    assert(prompt.find("<|observation|><tool_response>found</tool_response><tool_response>other</tool_response>")!=std::string::npos);
    auto parsed=ie::parse_glm5_completion("let me think</think>answer",true);
    assert(parsed.reasoning=="let me think" && parsed.content=="answer");
    parsed=ie::parse_glm5_completion("</think><tool_call>read<arg_key>n</arg_key><arg_value>2</arg_value></tool_call>",true);
    assert(parsed.content.empty() && parsed.tool_calls.find("read")!=std::string::npos);
    parsed=ie::parse_glm5_completion("<tool_call>read<arg_key>path</arg_key><arg_value>42</arg_value></tool_call>",false,R"([{"type":"function","function":{"name":"read","parameters":{"properties":{"path":{"type":"string"}}}}}])");
    assert(parsed.tool_calls.find("\\\"42\\\"")!=std::string::npos);
    auto request=ie::oai::parse_chat_request(R"({"messages":[{"role":"assistant","content":null,"reasoning_content":"I should read both","tool_calls":[{"id":"a","type":"function","function":{"name":"read","arguments":"{\"path\":\"a.txt\"}"}},{"id":"b","type":"function","function":{"name":"read","arguments":"{\"path\":\"b.txt\"}"}}]},{"role":"tool","tool_call_id":"b","content":"second"},{"role":"tool","tool_call_id":"a","content":"first"}]})");
    assert(request.error.empty());
    prompt=ie::build_glm5_prompt(request.turns,true,"",err);
    assert(prompt.find("</tool_call><tool_call>")!=std::string::npos);
    assert(prompt.find("<think>I should read both</think>")!=std::string::npos);
    assert(prompt.find("<|observation|><tool_response>first</tool_response><tool_response>second</tool_response>")!=std::string::npos);
    request.turns[2].tool_call_id="b"; // duplicate ids must retain wire order
    prompt=ie::build_glm5_prompt(request.turns,true,"",err);
    assert(prompt.find("<|observation|><tool_response>second</tool_response><tool_response>first</tool_response>")!=std::string::npos);
    // The prompt cache's boundary: the render without its generation suffix, and the text the next turn starts
    // with -- after a reply, a tool result, a reply with reasoning and a new user turn, thinking on or off.
    for(bool think:{true,false}) {
        const std::string tools=R"([{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}])";
        std::vector<ie::ChatTurn> conv={{"system","Be brief.",{}},{"user","Read a.txt",{}}};
        std::string full=ie::build_glm5_prompt(conv,think,tools,err);
        assert(err.empty());
        for(int step=0;step<3;++step) {
            const auto stable=ie::glm5_stable_prefix(full,think);
            assert(!stable.empty() && full.starts_with(stable) && stable.size()<full.size());
            assert(stable.ends_with("</tool_response>") || stable.ends_with("Read a.txt") || stable.ends_with("Thanks"));
            if(step==0) { conv.push_back({"assistant",R"(<tool_call>{"name":"read","arguments":{"path":"a.txt"}}</tool_call>)",{}});
                          conv.push_back({"tool","contents",{}}); }
            if(step==1) { conv.push_back({"assistant","It says contents.",{}});
                          conv.back().reasoning_content="the file is short"; conv.push_back({"user","Thanks",{}}); }
            if(step==2) break;
            const std::string next=ie::build_glm5_prompt(conv,think,tools,err);
            assert(err.empty() && next.starts_with(stable));
            full=next;
        }
        assert(ie::glm5_stable_prefix(full,!think).empty());   // the other mode's suffix is not this render's
    }
    assert(ie::glm5_stable_prefix("no suffix",true).empty());
    turns={{"developer","bad",{}}};
    ie::build_glm5_prompt(turns,true,"",err); assert(!err.empty());
    std::cout << "glm5_server_test: OK\n";
}
