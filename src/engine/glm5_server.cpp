#include "ie/glm5_server.hpp"
#include "ie/reasoning.hpp"
#include "nlohmann/json.hpp"
#include <cctype>
#include <unordered_set>
namespace ie {
namespace {
using Json=nlohmann::ordered_json;
std::string trim(std::string_view s) {
    while(!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))s.remove_prefix(1);
    while(!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))s.remove_suffix(1);
    return std::string(s);
}
// Match Jinja tojson's separators and UTF-8 policy.
std::string pyjson(const Json& j) {
    if(!j.is_structured())return j.dump(-1,' ',false);
    std::string out=j.is_object()?"{":"["; bool first=true;
    for(auto it=j.begin();it!=j.end();++it) {
        if(!first)out+=", ";first=false;
        if(j.is_object())out+=Json(it.key()).dump(-1,' ',false)+": ";
        out+=pyjson(it.value());
    }
    return out+(j.is_object()?"}":"]");
}
std::string render_history_calls(std::string text,std::string& error) {
    size_t pos=0;
    while((pos=text.find("<tool_call>",pos))!=std::string::npos) {
        const size_t begin=pos+11, end=text.find("</tool_call>",begin);
        if(end==std::string::npos){error="glm5next: incomplete assistant tool history";return {};}
        auto body=trim(std::string_view(text).substr(begin,end-begin));
        if(body.empty() || body.front()!='{'){pos=end+12;continue;}
        auto call=Json::parse(body,nullptr,false);
        if(!call.is_object() || !call.contains("name") || !call["name"].is_string()){
            error="glm5next: invalid assistant tool history";return {};
        }
        auto args=call.value("arguments",Json::object());
        if(args.is_string())args=Json::parse(args.get<std::string>(),nullptr,false);
        if(!args.is_object()){error="glm5next: tool arguments must be an object";return {};}
        std::string xml="<tool_call>"+call["name"].get<std::string>();
        for(auto it=args.begin();it!=args.end();++it)
            xml+="<arg_key>"+it.key()+"</arg_key><arg_value>"+
                (it->is_string()?it->get<std::string>():pyjson(*it))+"</arg_value>";
        xml+="</tool_call>";
        text.replace(pos,end+12-pos,xml);
        // The OpenAI parser separates flattened calls with a newline. The
        // GLM template trims prose and appends structured calls adjacently.
        while(pos && std::isspace(static_cast<unsigned char>(text[pos-1])))text.erase(--pos,1);
        pos+=xml.size();
    }
    return text;
}
}

std::string build_glm5_prompt(std::span<const ChatTurn> turns,bool thinking,
                             std::string_view tools_json,std::string& error,
                             std::string_view reasoning_effort) {
    error=reasoning_effort_error(reasoning_capabilities(ModelArch::kGlm5Next),reasoning_effort);
    if(!error.empty())return {};
    const std::string effort=reasoning_effort.empty()?"max":std::string(reasoning_effort);
    std::string out="[gMASK]<sop><|system|>Reasoning Effort: ";
    out+=char(std::toupper(static_cast<unsigned char>(effort.front())));out+=effort.substr(1);
    if(!tools_json.empty()) {
        auto tools=Json::parse(tools_json,nullptr,false);
        if(!tools.is_array()){error="glm5next tools must be an array";return {};}
        if(!tools.empty()) {
            out+="<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
            for(auto tool:tools) {
                if(tool.contains("function"))tool=tool["function"];
                if(!tool.is_object() || !tool.contains("name") || !tool["name"].is_string()){
                    error="glm5next tools need function names";return {};
                }
                if(tool.contains("defer_loading") && !tool["defer_loading"].is_boolean()) {
                    error="glm5next defer_loading must be boolean";return {};
                }
                if(tool.value("defer_loading",false)) {error="glm5next deferred tool loading is not supported";return {};}
                tool.erase("defer_loading");tool.erase("strict");out+=pyjson(tool)+"\n";
            }
            out+="</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";
        }
    }
    for(size_t ti=0;ti<turns.size();++ti) {
        const auto& t=turns[ti];
        if(t.role=="tool") {
            size_t end=ti+1;while(end<turns.size() && turns[end].role=="tool")++end;
            std::vector<size_t> order;
            const auto* prior=ti && turns[ti-1].role=="assistant"?&turns[ti-1]:nullptr;
            bool sortable=prior && !prior->tool_call_ids.empty();
            std::unordered_set<std::string> calls,results;
            if(sortable)for(const auto& id:prior->tool_call_ids)
                if(id.empty() || !calls.insert(id).second)sortable=false;
            for(size_t k=ti;k<end && sortable;++k) {
                const auto& id=turns[k].tool_call_id;
                if(id.empty() || !calls.contains(id) || !results.insert(id).second)sortable=false;
            }
            if(sortable)for(const auto& id:prior->tool_call_ids)
                for(size_t k=ti;k<end;++k)if(turns[k].tool_call_id==id)order.push_back(k);
            if(!sortable)for(size_t k=ti;k<end;++k)order.push_back(k);
            out+="<|observation|>";
            for(size_t k:order)out+="<tool_response>"+turns[k].content+"</tool_response>";
            ti=end-1;continue;
        }
        if(t.role=="user" || t.role=="system")out+="<|"+t.role+"|>"+t.content;
        else if(t.role=="assistant") {
            out+="<|assistant|>";
            std::string content=render_history_calls(t.content,error);
            if(!error.empty())return {};
            auto end=content.find("</think>");
            if(t.reasoning_content)out+="<think>"+*t.reasoning_content+"</think>";
            else if(end!=std::string::npos) {
                auto start=content.rfind("<think>",end);
                out+="<think>"+content.substr(start==std::string::npos?0:start+7,end-(start==std::string::npos?0:start+7))+"</think>";
                content.erase(0,content.rfind("</think>")+8);
            } else out+="<think></think>";
            out+=trim(content);
        } else {error="glm5next does not support role '"+t.role+"'";return {};}
    }
    out+="<|assistant|><think>";
    if(!thinking)out+="</think>";
    return out;
}

Glm5Completion parse_glm5_completion(std::string_view text,bool thinking,std::string_view tools_json) {
    Glm5Completion result;
    if(thinking) {
        auto end=text.find("</think>");
        if(end==text.npos){result.reasoning=std::string(text);return result;}
        auto reasoning=text.substr(0,end);if(reasoning.starts_with("<think>"))reasoning.remove_prefix(7);
        result.reasoning=std::string(reasoning);text.remove_prefix(end+8);
    }
    Json calls=Json::array();
    auto schemas=Json::parse(tools_json,nullptr,false);
    while(!text.empty()) {
        auto pos=text.find("<tool_call>");
        if(pos==text.npos){result.content+=text;break;}
        result.content+=text.substr(0,pos);text.remove_prefix(pos);
        auto end=text.find("</tool_call>");
        if(end==text.npos){result.content+=text;break;}
        auto body=text.substr(11,end-11);
        auto first=body.find("<arg_key>");
        std::string name=trim(body.substr(0,first));Json args=Json::object();
        bool valid=!name.empty();
        if(first!=body.npos)body.remove_prefix(first);else body={};
        while(!body.empty() && valid) {
            if(!body.starts_with("<arg_key>")){valid=false;break;}
            auto ke=body.find("</arg_key>");if(ke==body.npos){valid=false;break;}
            std::string key(body.substr(9,ke-9));body.remove_prefix(ke+10);
            if(!body.starts_with("<arg_value>")){valid=false;break;}
            auto ve=body.find("</arg_value>");if(ve==body.npos){valid=false;break;}
            std::string value(body.substr(11,ve-11));body.remove_prefix(ve+12);
            bool string_arg=false;
            if(schemas.is_array())for(const auto& entry:schemas) {
                const auto& fn=entry.contains("function")?entry["function"]:entry;
                if(!fn.is_object() || !fn.contains("name") || fn["name"]!=name || !fn.contains("parameters"))continue;
                const auto& params=fn["parameters"];
                if(params.contains("properties") && params["properties"].contains(key)) {
                    const auto& prop=params["properties"][key];
                    string_arg=prop.is_object() && prop.contains("type") && prop["type"]=="string";
                }
            }
            auto v=Json::parse(value,nullptr,false);args[key]=string_arg||v.is_discarded()?Json(value):v;
        }
        if(valid)calls.push_back({{"id","call_glm5_"+std::to_string(calls.size())},{"type","function"},
            {"function",{{"name",name},{"arguments",args.dump(-1,' ',false)}}}});
        else result.content+=text.substr(0,end+12);
        text.remove_prefix(end+12);
    }
    if(!calls.empty())result.tool_calls=calls.dump(-1,' ',false);
    return result;
}
}
