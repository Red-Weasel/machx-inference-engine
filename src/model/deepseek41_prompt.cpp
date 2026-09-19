// V4.1 native text/tool protocol, checked against the checkpoint encoding/encoding.py.
// Internal task/reminder roles and image inference are intentionally unsupported.
#include "ie/deepseek41_prompt.hpp"

#include <cstdlib>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <regex>
#include "../../third_party/nlohmann/json.hpp"

namespace ie {
namespace {
constexpr const char* kBos       = "<｜begin▁of▁sentence｜>";
constexpr const char* kEos       = "<｜end▁of▁sentence｜>";
constexpr const char* kSystem    = "<｜System｜>";
constexpr const char* kUser      = "<｜User｜>";
constexpr const char* kAssistant = "<｜Assistant｜>";
constexpr const char* kThinkOpen = "<think>";
constexpr const char* kThinkEnd  = "</think>";


using Json = nlohmann::ordered_json;
// Match Python json.dumps(ensure_ascii=False), including separator whitespace.
std::string py_json(const Json& j) {
    if (j.is_object() || j.is_array()) {
        std::string out = j.is_object() ? "{" : "[";
        bool first = true;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (!first) out += ", ";
            first = false;
            if (j.is_object()) out += py_json(it.key()) + ": ";
            out += py_json(it.value());
        }
        return out + (j.is_object() ? "}" : "]");
    }
    auto raw = j.dump(-1, ' ', false);
    // Python repr uses fixed notation through exponent 15; nlohmann switches at 15.
    if (j.is_number_float()) {
        const auto e = raw.find('e');
        if (e != std::string::npos) {
            const int exponent = std::stoi(raw.substr(e + 1));
            if (exponent >= -4 && exponent < 16) {
                const bool negative = raw.front() == '-';
                auto digits = raw.substr(negative ? 1 : 0, e - (negative ? 1 : 0));
                const auto dot = digits.find('.');
                if (dot != std::string::npos) digits.erase(dot, 1);
                const int position = exponent + 1;
                if (position <= 0) digits = "0." + std::string(-position, '0') + digits;
                else if (size_t(position) >= digits.size()) digits += std::string(position - digits.size(), '0') + ".0";
                else digits.insert(position, ".");
                raw = (negative ? "-" : "") + digits;
            }
        }
    }
    return raw;
}
Json array_json(const std::string& s) {
    if (s.empty()) return Json::array();
    auto j=Json::parse(s);
    if (!j.is_array()) throw std::runtime_error("tools/tool_calls must be arrays");
    return j;
}
std::string tool_name(Json f, const Json& tc) {
    if (tc.contains("namespace") && !tc["namespace"].is_null()) f["namespace"]=tc["namespace"];
    auto name=f.at("name").get<std::string>();
    std::string ns;
    if(f.contains("namespace") && !f["namespace"].is_null())
        ns=f["namespace"].is_object() ? f["namespace"].at("name").get<std::string>() : f["namespace"].get<std::string>();
    const auto at=name.find("::");
    if(at!=std::string::npos) {
        if(!ns.empty() && ns!=name.substr(0,at)) throw std::runtime_error("conflicting tool namespaces");
        ns=name.substr(0,at); name.erase(0,at+2);
    }
    if(name.empty() || name.find("::")!=std::string::npos || ns.find("::")!=std::string::npos ||
       (ns+name).find_first_of("\"<>\r\n")!=std::string::npos)
        throw std::runtime_error("invalid tool name");
    return ns.empty()?name:ns+"::"+name;
}
std::string render_tools(const std::string& raw) {
    auto tools=array_json(raw); if(tools.empty()) return {};
    std::string schemas;
    for(const auto& t:tools) {
        auto f=t.at("function");
        if(t.contains("namespace") && !t["namespace"].is_null()) f["namespace"]=t["namespace"];
        f["name"]=tool_name(f, t);
        if(f.contains("namespace") && f["namespace"].is_object() && !f["namespace"].value("description", "").empty())
            f["description"]=f["namespace"]["description"].get<std::string>()+"\n"+(f.contains("description") && f["description"].is_string()?f["description"].get<std::string>():"");
        f.erase("namespace");
        if(!schemas.empty()) schemas+='\n';
        schemas+=py_json(f);
    }
    std::string text=R"TOOLS(## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

__SCHEMAS__

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
)TOOLS";
    text.replace(text.find("__SCHEMAS__"), 11, schemas);
    return text;
}
std::string render_calls(const std::string& raw) {
    auto calls=array_json(raw); if(calls.empty()) return {};
    std::string text="\n\n<｜DSML｜ calls>\n";
    for(size_t i=0;i<calls.size();++i) {
        const auto& tc=calls[i]; const auto& f=tc.at("function");
        if(i) text+='\n';
        text+="<｜DSML｜ invoke name=\""+tool_name(f,tc)+"\">\n";
        auto args=f.at("arguments"); const auto original=args;
        for(int pass=0;pass<2 && args.is_string();++pass) {
            auto parsed=Json::parse(args.get<std::string>(),nullptr,false);
            if(parsed.is_discarded()) break;
            args=std::move(parsed);
        }
        if(!args.is_object()) args=Json{{"arguments",original}};
        bool first=true;
        for(auto it=args.begin();it!=args.end();++it) {
            if(it.key().find_first_of("\"<>\r\n")!=std::string::npos) throw std::runtime_error("invalid parameter name");
            if(!first) text+='\n'; first=false;
            text+="<｜DSML｜ parameter name=\""+it.key()+"\" string=\""+(it.value().is_string()?"true":"false")+"\">";
            text+=it.value().is_string()?it.value().get<std::string>():py_json(it.value());
            text+="</｜DSML｜ parameter>";
        }
        text+="\n</｜DSML｜ invoke>";
    }
    return text+"\n</｜DSML｜ calls>";
}
struct Block { bool tool; std::string id, text; };
std::vector<Ds41ChatMessage> merge_messages(const std::vector<Ds41ChatMessage>& input) {
    std::vector<Ds41ChatMessage> out;
    std::map<std::string,size_t> order;
    for(size_t i=0;i<input.size();) {
        auto m=input[i];
        if(m.role=="user" || m.role=="tool") {
            std::vector<Block> blocks;
            while(i<input.size() && (input[i].role=="user" || input[i].role=="tool")) {
                const auto& u=input[i++]; blocks.push_back({u.role=="tool",u.tool_call_id,u.content});
            }
            std::vector<Block> results;
            for(const auto& b:blocks) if(b.tool) results.push_back(b);
            std::stable_sort(results.begin(),results.end(),[&](const auto& a,const auto& b){return order[a.id]<order[b.id];});
            size_t n=0; m.role="user"; m.content.clear();
            for(size_t k=0;k<blocks.size();++k) {
                const auto& b=blocks[k].tool?results[n++]:blocks[k];
                if(k) m.content+="\n\n";
                m.content+=b.tool?"<tool_result>"+b.text+"</tool_result>":b.text;
            }
        } else {
            ++i;
            auto calls=array_json(m.tool_calls_json);
            if(m.role=="assistant" && !calls.empty()) {
                order.clear();
                for(size_t k=0;k<calls.size();++k) {
                    auto id=calls[k].value("id","");
                    if(id.empty()) id=calls[k].at("function").value("id","");
                    if(!id.empty()) order[id]=k;
                }
            }
        }
        out.push_back(std::move(m));
    }
    return out;
}

// find_last_user_index: the last user, or a system message past index 0
int last_user_index(const std::vector<Ds41ChatMessage>& m) {
    for (int i = int(m.size()) - 1; i >= 0; --i)
        if (m[size_t(i)].role == "user" || (m[size_t(i)].role == "system" && i > 0)) return i;
    return -1;
}

std::string render(size_t idx, const std::vector<Ds41ChatMessage>& m, const Ds41PromptOptions& o) {
    const auto& msg = m[idx];
    const int last_user = last_user_index(m);
    // render_reasoning_effort: thinking mode, index 0 only
    std::string effort;
    if (idx == 0 && o.thinking)
        effort = "Reasoning Effort: " + std::to_string(o.reasoning_effort) + " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
    std::string p = (idx == 0 && (!effort.empty() || msg.role == "system")) ? kSystem : "";
    p += effort;
    if (msg.role == "system") {
        if (idx > 0) p += kSystem;                       // mid-conversation system message
        p += msg.content;
        auto tools = render_tools(msg.tools_json);
        if (!tools.empty()) p += "\n\n" + tools;
    } else if (msg.role == "user") {
        p += kUser; p += msg.content;
    } else {                                              // assistant
        std::string thinking_part;
        if (o.thinking && (!o.drop_thinking || int(idx) > last_user)) thinking_part = msg.reasoning_content + kThinkEnd;
        p += thinking_part + msg.content + render_calls(msg.tool_calls_json) + kEos;          // assistant_msg_template: {reasoning}{content}{tool_calls}eos
    }
    // the transition tokens, only when what follows is an assistant turn (or nothing)
    if (idx + 1 < m.size() && m[idx + 1].role != "assistant") return p;
    if (msg.role == "user" || (msg.role == "system" && idx > 0)) {
        p += kAssistant;
        if (!o.drop_thinking && o.thinking)                       p += kThinkOpen;
        else if (o.drop_thinking && o.thinking && int(idx) >= last_user) p += kThinkOpen;
        else                                                       p += kThinkEnd;
    }
    return p;
}
}  // namespace

int ds41_reasoning_effort_of(const std::string& s, std::string& err) {
    if (s == "low") return 50;
    if (s == "high") return 75;
    if (s == "max") return 100;
    char* end = nullptr; const long v = std::strtol(s.c_str(), &end, 10);
    if (!s.empty() && end && *end == '\0' && v >= 1 && v <= 100) return int(v);
    err = "reasoning effort must be low / high / max or an integer in 1..100, got '" + s + "'";
    return -1;
}

std::string ds41_encode_messages(const std::vector<Ds41ChatMessage>& msgs, const Ds41PromptOptions& options, std::string& err) {
    err.clear();
    auto o=options;
    try {
    if (msgs.empty()) { err = "no messages"; return ""; }
    if (o.reasoning_effort < 1 || o.reasoning_effort > 100) { err = "reasoning_effort out of 1..100"; return ""; }
    if (o.has_tools && o.tools_json.empty()) throw std::runtime_error("missing structured tools_json");
    for (size_t i = 0; i < msgs.size(); ++i) {
        const auto& r = msgs[i].role;
        if (r == "latest_reminder" || r == "direct_search_results")
            { err = "message " + std::to_string(i) + ": role '" + r + "' (internal role) is not supported by this port -- refused, not faked"; return ""; }
        if (r != "system" && r != "user" && r != "assistant" && r != "tool") { err = "message " + std::to_string(i) + ": unknown role '" + r + "'"; return ""; }
        if (!msgs[i].reasoning_content.empty() && r != "assistant" && r != "tool") { err = "message " + std::to_string(i) + ": reasoning_content on a non-assistant turn"; return ""; }
        if (msgs[i].has_images) { err = "message " + std::to_string(i) + ": image content is not supported by this port -- refused, not faked"; return ""; }
        if (msgs[i].has_tool_calls && msgs[i].tool_calls_json.empty()) throw std::runtime_error("missing structured tool_calls_json");
        if (!msgs[i].tool_calls_json.empty() && r != "assistant") throw std::runtime_error("tool_calls require assistant role");
        if (!msgs[i].tools_json.empty() && r != "system") throw std::runtime_error("tools require system role");
    }
    // _drop_thinking_messages: in thinking mode with drop_thinking, assistant turns before the last
    // user turn lose their reasoning_content (user/system turns are kept as they are)
    std::vector<Ds41ChatMessage> full = msgs;
    if (!array_json(o.tools_json).empty()) {
        if (full.front().role != "system") full.insert(full.begin(), Ds41ChatMessage{"system", "", "", false, false, "", "", ""});
        full.front().tools_json=o.tools_json;
    }
    for(const auto& m:full) if(!array_json(m.tools_json).empty()) o.drop_thinking=false;
    full=merge_messages(full);
    if (o.thinking && o.drop_thinking) {
        const int last_user = last_user_index(full);
        for (size_t i = 0; i < full.size(); ++i)
            if (full[i].role == "assistant" && int(i) < last_user) full[i].reasoning_content.clear();
    }
    std::string prompt = o.add_bos ? kBos : "";
    for (size_t i = 0; i < full.size(); ++i) prompt += render(i, full, o);
    return prompt;
    } catch(const std::exception& e) { err=std::string("deepseek41 prompt: ")+e.what(); return {}; }
}

namespace {
constexpr std::string_view dsml = "<｜DSML｜";
constexpr std::string_view calls_open = "\n\n<｜DSML｜ calls>\n";
constexpr std::string_view calls_close = "</｜DSML｜ calls>";
constexpr std::string_view invoke_close = "</｜DSML｜ invoke>\n";
constexpr std::string_view parameter_close = "</｜DSML｜ parameter>\n";
}

std::string ds41_visible_content(std::string_view text) {
    auto at=text.find(dsml);
    if(at!=std::string_view::npos) text=text.substr(0,at);
    else for(size_t n=std::min(text.size(),dsml.size()-1);n>0;--n)
        if(text.ends_with(dsml.substr(0,n))) { text.remove_suffix(n); break; }
    return std::string(text);
}

std::string ds41_cut_tool_name(std::string_view text) {
    const auto at=text.find(dsml);
    if(at==std::string_view::npos) return {};
    static const std::regex invoke(R"TAG(<｜DSML｜ invoke name="([^"]+)")TAG");
    const std::string tail(text.substr(at));
    std::string name="unknown";                          // the LAST invoke is the one the cut landed in
    for(auto it=std::sregex_iterator(tail.begin(),tail.end(),invoke); it!=std::sregex_iterator(); ++it) name=(*it)[1].str();
    return name;
}

Ds41Completion ds41_parse_completion(std::string_view text, bool thinking) {
    Ds41Completion out;
    try {
        auto fail=[](const char* msg){throw std::runtime_error(msg);};
        if(thinking) {
            auto at=text.find(kThinkEnd);
            if(at==std::string_view::npos) fail("missing </think>");
            out.reasoning_content=std::string(text.substr(0,at)); text.remove_prefix(at+8);
        }
        if(!text.ends_with(kEos)) fail("missing EOS");
        text.remove_suffix(std::string_view(kEos).size());
        auto start=text.find(calls_open);
        if(start==std::string_view::npos) out.content=std::string(text);
        else {
            out.content=std::string(text.substr(0,start)); text.remove_prefix(start+calls_open.size());
            Json calls=Json::array();
            const std::regex invoke(R"TAG(^<｜DSML｜ invoke name="([^"]+)">\n)TAG");
            const std::regex parameter(R"TAG(^<｜DSML｜ parameter name="([^"]*)" string="(true|false)">)TAG");
            while(!text.starts_with(calls_close)) {
                std::match_results<std::string_view::const_iterator> match;
                if(!std::regex_search(text.begin(),text.end(),match,invoke)) fail("invalid invoke header");
                const std::string name=tool_name(Json{{"name",match[1].str()}},Json::object());
                text.remove_prefix(match.length());
                std::string args="{"; std::vector<std::string> keys;
                // Reference emits a blank arguments line for a zero-argument invocation.
                if(text.starts_with("\n")) text.remove_prefix(1);
                while(!text.starts_with(invoke_close)) {
                    if(!std::regex_search(text.begin(),text.end(),match,parameter)) fail("invalid parameter header");
                    auto key=match[1].str(); bool is_string=match[2]=="true";
                    if(std::find(keys.begin(),keys.end(),key)!=keys.end()) fail("duplicate parameter");
                    keys.push_back(key); text.remove_prefix(match.length());
                    auto end=text.find(parameter_close);
                    if(end==std::string_view::npos) fail("missing parameter terminator");
                    auto value=text.substr(0,end);
                    if(!is_string && Json::parse(value,nullptr,false).is_discarded()) fail("invalid JSON parameter");
                    if(keys.size()>1) args+=", ";
                    args+=py_json(key)+": "+(is_string?py_json(std::string(value)):std::string(value));
                    text.remove_prefix(end+parameter_close.size());
                }
                text.remove_prefix(invoke_close.size()); args+='}';
                calls.push_back(Json{{"type","function"},{"function",{{"name",name},{"arguments",args}}}});
            }
            text.remove_prefix(calls_close.size());
            if(!text.empty() || calls.empty()) fail("empty calls or trailing content");
            out.tool_calls_json=calls.dump();
        }
        for(const auto* token:{kBos,kEos,kThinkOpen,kThinkEnd,"｜DSML｜"})
            if(out.content.find(token)!=std::string::npos || out.reasoning_content.find(token)!=std::string::npos)
                fail("unexpected special token outside tool calls");
    } catch(const std::exception& e) {
        out.error=std::string("deepseek41 completion: ")+e.what(); out.tool_calls_json.clear();
        out.content=ds41_visible_content(out.content);
    }
    return out;
}

}  // namespace ie
