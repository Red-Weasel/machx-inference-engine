// CPU-only JSON-lines bridge for comparison with the checkpoint's reference encoder.
#include "ie/deepseek41_prompt.hpp"
#include "ie/openai_proto.hpp"
#include "../third_party/nlohmann/json.hpp"
#include <iostream>
using J=nlohmann::ordered_json;
int main() {
    std::string line;
    while(std::getline(std::cin,line)) try {
        auto c=J::parse(line); J result;
        if(c.contains("request")) {
            const auto req = ie::oai::parse_chat_request(c["request"].dump());
            std::vector<ie::Ds41ChatMessage> msgs;
            for(const auto& t:req.turns) msgs.push_back({t.role,t.content_without_tool_calls.value_or(t.content),t.reasoning_content.value_or(""),!t.images.empty(),!t.tool_calls_json.empty(),t.tool_calls_json,t.tool_call_id,""});
            ie::Ds41PromptOptions o; o.thinking=req.enable_thinking; o.tools_json=req.tools_json;
            std::string error=req.error, prompt;
            if(error.empty()) prompt=ie::ds41_encode_messages(msgs,o,error);
            result={{"prompt",prompt},{"error",error}};
        } else if(c.contains("completion")) {
            auto p=ie::ds41_parse_completion(c["completion"].get<std::string>(),c.value("thinking",false));
            ie::GenerateResult generated;
            generated.text=p.content; generated.reasoning_content=p.reasoning_content;
            generated.tool_calls_json=p.tool_calls_json.empty()?"[]":p.tool_calls_json;
            generated.finish_reason=p.error.empty()?(p.tool_calls_json.empty()?"stop":"tool_calls"):"error: "+p.error;
            std::string sse;
            if(p.error.empty()) {
                sse+=ie::oai::chat_chunk_sse_reasoning("DeepSeek-V4.1-Flash","fixture",0,p.reasoning_content);
                sse+=ie::oai::chat_chunk_sse("DeepSeek-V4.1-Flash","fixture",0,p.content,"");
                sse+=ie::oai::chat_chunk_sse_tool_calls_json("DeepSeek-V4.1-Flash","fixture",0,generated.tool_calls_json);
                sse+=ie::oai::chat_chunk_sse("DeepSeek-V4.1-Flash","fixture",0,"",generated.finish_reason);
            } else sse="data: "+ie::oai::error_json(p.error)+"\n\n";
            sse+="data: [DONE]\n\n";
            result={{"sse",sse},{"response",J::parse(ie::oai::chat_completion_json("DeepSeek-V4.1-Flash",generated,"fixture",0))},{"content",p.content},{"reasoning_content",p.reasoning_content},{"error",p.error},{"tool_calls",p.tool_calls_json.empty()?J::array():J::parse(p.tool_calls_json)}};
        } else if(c.contains("partial")) result={{"content",ie::ds41_visible_content(c["partial"].get<std::string>())},
                                                 {"cut_tool",ie::ds41_cut_tool_name(c["partial"].get<std::string>())}};
        else {
            std::vector<ie::Ds41ChatMessage> msgs;
            for(const auto& m:c["messages"]) {
                ie::Ds41ChatMessage t; t.role=m.value("role","");
                if(m.contains("content") && m["content"].is_string()) t.content=m["content"].get<std::string>();
                else if(m.contains("content") && !m["content"].is_null()) t.has_images=true;
                t.reasoning_content=m.value("reasoning_content","");
                if(m.contains("tool_calls") && !m["tool_calls"].is_null()) t.tool_calls_json=m["tool_calls"].dump();
                if(m.contains("tools")) t.tools_json=m["tools"].dump();
                t.tool_call_id=m.value("tool_call_id",""); msgs.push_back(t);
            }
            ie::Ds41PromptOptions o; o.thinking=c.value("thinking_mode","chat")=="thinking";
            o.drop_thinking=c.value("drop_thinking",true); o.add_bos=c.value("add_default_bos_token",true);
            std::string error;
            if(c.contains("tools")) o.tools_json=c["tools"].dump();
            if(c.contains("reasoning_effort") && !c["reasoning_effort"].is_null())
                o.reasoning_effort=c["reasoning_effort"].is_number_integer()?c["reasoning_effort"].get<int>():ie::ds41_reasoning_effort_of(c["reasoning_effort"].get<std::string>(),error);
            auto prompt=ie::ds41_encode_messages(msgs,o,error);
            result={{"prompt",prompt},{"error",error}};
        }
        std::cout<<result.dump()<<std::endl;
    } catch(const std::exception& e) {std::cout<<J{{"error",e.what()}}.dump()<<std::endl;}
}
