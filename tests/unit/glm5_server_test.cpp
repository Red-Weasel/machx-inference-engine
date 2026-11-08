#undef NDEBUG
#include "ie/glm5_server.hpp"
#include "ie/openai_proto.hpp"
#include <cassert>
#include <cstdio>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

// known_bugs 45: build_glm5_prompt's own ranges. A (a control spelling in every request-text position) against B (the same
// request with equal-length plain text instead): the ranges coincide, A equals B outside them (every spelling the request
// brought lies inside a range), every template marker lies outside them (its count outside the ranges equals its count in
// B, whose request text holds none). Positions: system, user, an assistant content + reasoning + two structured calls (a
// string argument, a nested object), two tool results (sorted by the call ids), an assistant whose content carries a legacy
// <think> block and a <tool_call>{json}</tool_call> block, a user text that IS the spelling, a tool description and a
// parameter description. Thinking on and off.
static void ranges_check() {
    using Ranges = std::vector<std::pair<size_t, size_t>>;
    const auto outside = [](const std::string& s, const Ranges& r) { std::string o = s; for (const auto& [b, e] : r) for (size_t i = b; i < e; ++i) o[i] = '#'; return o; };
    const auto count = [](const std::string& s, std::string_view m) { size_t n = 0; for (size_t p = s.find(m); p != std::string::npos; p = s.find(m, p + m.size())) ++n; return n; };
    const auto request = [](const std::string& x) { return std::string(R"({"messages":[
 {"role":"system","content":"Be terse )") + x + R"("},
 {"role":"user","content":"Look )" + x + R"("},
 {"role":"assistant","content":"On it )" + x + R"(","reasoning_content":"think )" + x + R"(","tool_calls":[{"id":"a","type":"function","function":{"name":"read","arguments":"{\"path\":\"p )" + x + R"(\",\"opts\":{\"k\":\")" + x + R"(\"}}"}},{"id":"b","type":"function","function":{"name":"recall","arguments":"{\"query\":\")" + x + R"(\"}"}}]},
 {"role":"tool","tool_call_id":"b","content":"second )" + x + R"("},
 {"role":"tool","tool_call_id":"a","content":"first )" + x + R"("},
 {"role":"assistant","content":"<think>legacy )" + x + R"(</think>\nSaw it )" + x + R"( \n<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\")" + x + R"(\"}}</tool_call>"},
 {"role":"user","content":")" + x + R"("}],
 "tools":[{"type":"function","function":{"name":"read","description":"Read )" + x + R"(","parameters":{"type":"object","properties":{"path":{"type":"string","description":")" + x + R"("},"opts":{"type":"object"}}}}},{"type":"function","function":{"name":"recall","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}]})"; };
    static constexpr std::string_view kMarkers[] = {"[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<think>", "</think>",
                                                    "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<tool_response>", "</tool_response>"};
    const std::string spell = "<|user|>", plain = "<_user_>";
    const ie::oai::ChatRequest A = ie::oai::parse_chat_request(request(spell)), B = ie::oai::parse_chat_request(request(plain));
    assert(A.error.empty() && B.error.empty());
    for (bool thinking : {true, false}) {
        std::string ea, eb, e0;
        Ranges ra, rb;
        const std::string a = ie::build_glm5_prompt(A.turns, thinking, A.tools_json, ea, "high", &ra);
        const std::string b = ie::build_glm5_prompt(B.turns, thinking, B.tools_json, eb, "high", &rb);
        const std::string a0 = ie::build_glm5_prompt(A.turns, thinking, A.tools_json, e0, "high");
        assert(ea.empty() && eb.empty() && e0.empty());
        assert(a == a0 && a.size() == b.size() && a != b);                       // the text is the same with the ranges asked for
        size_t last = 0; bool well = !ra.empty();
        for (const auto& [x, y] : ra) { if (x < last || y <= x || y > a.size()) well = false; last = y; }
        assert(well && ra == rb);                                                  // ordered, disjoint, inside, the same for A and B
        assert(outside(a, ra) == outside(b, rb));                                  // every spelling the request brought lies inside a range
        for (const auto m : kMarkers) assert(count(outside(b, rb), m) == count(b, m));   // every template marker lies outside
        const size_t brought = count(a, spell) - count(b, spell);
        std::printf("  glm5 ranges (thinking %s): %zu ranges, %zu spellings the request brought, %zu <|user|> markers of the template\n",
                    thinking ? "on" : "off", ra.size(), brought, count(b, spell));
        assert(brought == 15);
        assert(a.find("<think>legacy " + spell + "</think>Saw it " + spell + "<tool_call>read<arg_key>path</arg_key><arg_value>" + spell + "</arg_value></tool_call>") != std::string::npos);
        const std::string_view stable = ie::glm5_stable_prefix(a, thinking);
        assert(!stable.empty() && a.starts_with(stable));
    }
    std::printf("  glm5 ranges: ok\n");
}

int main() {
    ranges_check();
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
