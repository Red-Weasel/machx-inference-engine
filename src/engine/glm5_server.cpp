#include "ie/glm5_server.hpp"
#include "ie/reasoning.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cctype>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>
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
// A piece's slice [a, b) appended to `out`: its text, and the content ranges inside it, shifted (known_bugs 45).
void append_slice(PromptPiece& out, const PromptPiece& f, size_t a, size_t b) {
    if (b > f.text.size()) b = f.text.size();
    if (a >= b) return;
    const size_t base = out.text.size();
    out.text.append(f.text, a, b - a);
    for (const auto& [rb, re] : f.content) {
        const size_t cb = std::max(rb, a), ce = std::min(re, b);
        if (cb >= ce) continue;
        const size_t nb = base + (cb - a), ne = base + (ce - a);
        if (!out.content.empty() && out.content.back().second == nb) out.content.back().second = ne;
        else out.content.emplace_back(nb, ne);
    }
}
// trim's bounds of s[a, b).
std::pair<size_t, size_t> trim_bounds(std::string_view s, size_t a, size_t b) {
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return {a, b};
}
// The assistant history's <tool_call>{json}</tool_call> blocks (the OpenAI parser's form) in the template's XML form; the
// prose between them, the names, the keys and the values are the request's text, the markers the template's.
PromptPiece render_history_calls(std::string_view text,std::string& error) {
    PromptPiece f;
    size_t pos=0, seg=0;   // seg: where the pending request-text segment starts
    while((pos=text.find("<tool_call>",pos))!=std::string_view::npos) {
        const size_t begin=pos+11, end=text.find("</tool_call>",begin);
        // a reply cut mid call (max_tokens) streamed the raw call as content; the history then carries what the model
        // wrote, so keep it verbatim (2026-10-21: it was refused, "incomplete assistant tool history", ending the chat)
        if(end==std::string_view::npos)break;
        auto body=trim(text.substr(begin,end-begin));
        if(body.empty() || body.front()!='{'){pos=end+12;continue;}
        auto call=Json::parse(body,nullptr,false);
        if(!call.is_object() || !call.contains("name") || !call["name"].is_string()){
            error="glm5next: invalid assistant tool history";return {};
        }
        auto args=call.value("arguments",Json::object());
        if(args.is_string())args=Json::parse(args.get<std::string>(),nullptr,false);
        if(!args.is_object()){error="glm5next: tool arguments must be an object";return {};}
        // The OpenAI parser separates flattened calls with a newline. The
        // GLM template trims prose and appends structured calls adjacently.
        size_t cut=pos;
        while(cut>seg && std::isspace(static_cast<unsigned char>(text[cut-1])))--cut;
        f.add_content(text.substr(seg,cut-seg));
        f.add_text("<tool_call>");f.add_content(call["name"].get_ref<const std::string&>());
        for(auto it=args.begin();it!=args.end();++it) {
            f.add_text("<arg_key>");f.add_content(it.key());f.add_text("</arg_key><arg_value>");
            if(it->is_string())f.add_content(it->get_ref<const std::string&>());else f.add_content(pyjson(*it));
            f.add_text("</arg_value>");
        }
        f.add_text("</tool_call>");
        pos=end+12;seg=pos;
    }
    f.add_content(text.substr(seg));
    return f;
}
}

std::string build_glm5_prompt(std::span<const ChatTurn> turns,bool thinking,
                             std::string_view tools_json,std::string& error,
                             std::string_view reasoning_effort,
                             std::vector<std::pair<size_t,size_t>>* content_out) {
    error=reasoning_effort_error(reasoning_capabilities(ModelArch::kGlm5Next),reasoning_effort);
    if(!error.empty())return {};
    const std::string effort=reasoning_effort.empty()?"max":std::string(reasoning_effort);
    PromptPiece p;
    p.add_text("[gMASK]<sop><|system|>Reasoning Effort: ");
    p.add_text(char(std::toupper(static_cast<unsigned char>(effort.front()))));p.add_text(effort.substr(1));
    if(!tools_json.empty()) {
        auto tools=Json::parse(tools_json,nullptr,false);
        if(!tools.is_array()){error="glm5next tools must be an array";return {};}
        if(!tools.empty()) {
            p.add_text("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
            for(auto tool:tools) {
                if(tool.contains("function"))tool=tool["function"];
                if(!tool.is_object() || !tool.contains("name") || !tool["name"].is_string()){
                    error="glm5next tools need function names";return {};
                }
                if(tool.contains("defer_loading") && !tool["defer_loading"].is_boolean()) {
                    error="glm5next defer_loading must be boolean";return {};
                }
                if(tool.value("defer_loading",false)) {error="glm5next deferred tool loading is not supported";return {};}
                tool.erase("defer_loading");tool.erase("strict");p.add_content(pyjson(tool));p.add_text('\n');
            }
            p.add_text("</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>");
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
            p.add_text("<|observation|>");
            for(size_t k:order){p.add_text("<tool_response>");p.add_content(turns[k].content);p.add_text("</tool_response>");}
            ti=end-1;continue;
        }
        if(t.role=="user" || t.role=="system"){p.add_text("<|"+t.role+"|>");p.add_content(t.content);}
        else if(t.role=="assistant") {
            p.add_text("<|assistant|>");
            const PromptPiece hist=render_history_calls(t.content,error);
            if(!error.empty())return {};
            const std::string& content=hist.text;
            const auto end=content.find("</think>");
            size_t rest=0;
            if(t.reasoning_content){p.add_text("<think>");p.add_content(*t.reasoning_content);p.add_text("</think>");}
            else if(end!=std::string::npos) {
                const auto start=content.rfind("<think>",end);
                const size_t s0=start==std::string::npos?0:start+7;
                p.add_text("<think>");append_slice(p,hist,s0,end);p.add_text("</think>");
                rest=content.rfind("</think>")+8;
            } else p.add_text("<think></think>");
            const auto [a,b]=trim_bounds(content,rest,content.size());
            append_slice(p,hist,a,b);
        } else {error="glm5next does not support role '"+t.role+"'";return {};}
    }
    p.add_text("<|assistant|><think>");
    if(!thinking)p.add_text("</think>");
    if(content_out)*content_out=std::move(p.content);
    return std::move(p.text);
}

std::string_view glm5_stable_prefix(std::string_view prompt,bool thinking) {
    const std::string_view gen=thinking?"<|assistant|><think>":"<|assistant|><think></think>";
    if(!prompt.ends_with(gen))return {};
    return prompt.substr(0,prompt.size()-gen.size());
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
