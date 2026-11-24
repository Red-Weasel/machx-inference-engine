// src/engine/tool_guard.cpp -- see include/ie/tool_guard.hpp
#include <cstdio>
#include <cstdlib>
#include "ie/tool_guard.hpp"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <bit>
#include <cctype>

namespace ie {

namespace {
bool starts_with(std::string_view a, std::string_view b) { return a.size() >= b.size() && a.compare(0, b.size(), b) == 0; }
bool ends_with(std::string_view a, std::string_view b) { return a.size() >= b.size() && a.compare(a.size() - b.size(), b.size(), b) == 0; }
// the longest suffix of v that is a proper prefix of tag (a tag being written)
size_t partial_tag(std::string_view v, std::string_view tag) {
    for (size_t n = std::min(v.size(), tag.size() - 1); n > 0; --n)
        if (v.compare(v.size() - n, n, tag, 0, n) == 0) return n;
    return 0;
}
}  // namespace

extern thread_local const ToolGuardPrompt* t_guard_prompt;

std::shared_ptr<const ToolGuard> ToolGuard::make(std::string_view tools_json, const Tags& tags, std::string& err) {
    err.clear();
    if (tools_json.empty()) return nullptr;
    const auto j = nlohmann::json::parse(tools_json, nullptr, false);
    if (j.is_discarded() || !j.is_array()) { err = "tools: not a JSON array"; return nullptr; }
    auto g = std::make_shared<ToolGuard>();
    g->t_ = tags;
    for (const auto& t0 : j) {
        const auto& f = t0.contains("function") ? t0["function"] : t0;
        if (!f.is_object() || !f.contains("name") || !f["name"].is_string()) continue;
        Tool tool;
        tool.name = f["name"].get<std::string>();
        const auto& ps = f.value("parameters", nlohmann::json::object());
        std::vector<std::string> req;
        if (ps.contains("required") && ps["required"].is_array())
            for (const auto& r : ps["required"]) if (r.is_string()) req.push_back(r.get<std::string>());
        if (ps.contains("properties") && ps["properties"].is_object())
            for (auto it = ps["properties"].begin(); it != ps["properties"].end(); ++it) {
                if (tool.params.size() >= 64) break;
                Param p;
                p.name = it.key();
                const auto& sc = it.value();
                std::string ty;
                if (sc.contains("type") && sc["type"].is_string()) ty = sc["type"].get<std::string>();
                if (sc.contains("enum") && sc["enum"].is_array() && !sc["enum"].empty() &&
                    std::all_of(sc["enum"].begin(), sc["enum"].end(), [](const auto& e) { return e.is_string(); })) {
                    p.kind = Kind::kEnum;
                    for (const auto& e : sc["enum"]) p.enums.push_back(e.get<std::string>());
                } else if (ty == "integer") p.kind = Kind::kInt;
                else if (ty == "number") p.kind = Kind::kNum;
                else if (ty == "boolean") p.kind = Kind::kBool;
                else if (ty == "object" || ty == "array") p.kind = Kind::kJson;
                else p.kind = Kind::kStr;   // string, or no / several types: any text
                p.required = std::find(req.begin(), req.end(), p.name) != req.end();
                if (p.required) tool.required |= uint64_t(1) << tool.params.size();
                tool.params.push_back(std::move(p));
            }
        g->tools_.push_back(std::move(tool));
    }
    if (g->tools_.empty()) return nullptr;
    for (const auto& t : g->tools_) if (t.name == "tool_call" || t.name == "tool_schema") g->open_names_ = true;
    if (const char* v = std::getenv("IE_TOOL_GUARD_OPEN"); v && *v) g->open_names_ = *v == '1';   // (override)
    if (!g->open_names_ && t_guard_prompt && !t_guard_prompt->text.empty()) {   // the names the conversation mentions
        const std::string& tx = t_guard_prompt->text;
        for (size_t i = 0; i < tx.size();) {
            if (!(std::isalpha(uint8_t(tx[i])) || tx[i] == '_')) { ++i; continue; }
            size_t j = i;
            while (j < tx.size() && ident(tx[j])) ++j;
            if (j - i >= 2 && j - i <= 64) g->prompt_words_.emplace_back(tx, i, j - i);
            i = j;
        }
        std::sort(g->prompt_words_.begin(), g->prompt_words_.end());
        g->prompt_words_.erase(std::unique(g->prompt_words_.begin(), g->prompt_words_.end()), g->prompt_words_.end());
        g->prompt_names_ = !g->prompt_words_.empty();
    }
    return g;
}

bool ToolGuard::name_allowed(std::string_view name, bool complete) const {
    if (open_names_) return true;
    if (!prompt_names_) return false;
    auto it = std::lower_bound(prompt_words_.begin(), prompt_words_.end(), name, [](const std::string& a, std::string_view b) { return a < b; });
    if (it == prompt_words_.end()) return false;
    return complete ? *it == name : std::string_view(*it).substr(0, name.size()) == name;
}

thread_local const ToolGuardPrompt* t_guard_prompt = nullptr;
ToolGuardPrompt::ToolGuardPrompt(std::string t) : text(std::move(t)) { t_guard_prompt = this; }
ToolGuardPrompt::~ToolGuardPrompt() { if (t_guard_prompt == this) t_guard_prompt = nullptr; }

int ToolGuard::open_match(std::string_view buf, std::initializer_list<std::string_view> ends) {
    size_t k = 0;
    while (k < buf.size() && ident(buf[k])) ++k;
    if (k == 0) return 0;
    const std::string_view rest = buf.substr(k);
    if (rest.empty()) return 1;
    int i = 0;
    for (auto e : ends) {
        if (rest == e) return 2 + i;
        if (e.substr(0, rest.size()) == rest) return 1;
        ++i;
    }
    return 0;
}

std::shared_ptr<const ToolGuard> ToolGuard::make_dsml(std::string_view tools_json, std::string& err) {
    auto g = make(tools_json, Tags{}, err);
    if (!g) return g;
    auto d = std::make_shared<ToolGuard>(*g);
    d->dsml_ = true;
    return d;
}

std::shared_ptr<const ToolGuard> ToolGuard::make_xml(std::string_view tools_json, std::string& err) {
    auto g = make(tools_json, Tags{}, err);
    if (!g) return g;
    auto x = std::make_shared<ToolGuard>(*g);
    x->xml_ = true;
    return x;
}

std::shared_ptr<const ToolGuard> ToolGuard::with_choice(std::string_view choice, std::string& err) const {
    err.clear();
    auto g = std::make_shared<ToolGuard>(*this);
    if (choice.empty() || choice == "auto") return g;
    if (choice == "none") { g->no_calls_ = true; return g; }
    g->need_call_ = true;
    if (choice == "required") return g;
    if (choice.substr(0, 3) == "fn:") {
        const std::string_view nm = choice.substr(3);
        for (size_t i = 0; i < tools_.size(); ++i) if (tools_[i].name == nm) { g->only_tool_ = int32_t(i); return g; }
        err = "tool_choice names a function the request's tools do not have";
        return nullptr;
    }
    err = "tool_choice: unknown value";
    return nullptr;
}

bool ToolGuard::value_prefix_ok(const Param& p, std::string_view v) const {
    switch (p.kind) {
    case Kind::kStr: case Kind::kJson: case Kind::kAny: return true;   // (JSON: checked whole at the close)
    case Kind::kBool: return starts_with("true", v) || starts_with("false", v);
    case Kind::kEnum: return std::any_of(p.enums.begin(), p.enums.end(), [&](const std::string& e) { return starts_with(e, v); });
    case Kind::kInt: {
        for (size_t i = 0; i < v.size(); ++i) if (!(std::isdigit(uint8_t(v[i])) || (i == 0 && v[i] == '-'))) return false;
        return true;
    }
    case Kind::kNum: {
        // -?digits(.digits)?([eE][+-]?digits)?, as a prefix
        size_t i = 0; int st = 0;
        if (i < v.size() && v[i] == '-') ++i;
        for (; i < v.size(); ++i) {
            const char c = v[i];
            if (std::isdigit(uint8_t(c))) { if (st == 3) st = 4; continue; }
            if (c == '.' && st == 0) { st = 1; continue; }
            if ((c == 'e' || c == 'E') && st <= 1) { st = 2; continue; }
            if ((c == '+' || c == '-') && st == 2) { st = 3; continue; }
            return false;
        }
        return true;
    }
    }
    return true;
}

bool ToolGuard::value_ok(const Param& p, std::string_view v) const {
    switch (p.kind) {
    case Kind::kStr: return true;
    case Kind::kBool: return v == "true" || v == "false";
    case Kind::kEnum: return std::find(p.enums.begin(), p.enums.end(), v) != p.enums.end();
    case Kind::kInt: return !v.empty() && v != "-" && value_prefix_ok(p, v);
    case Kind::kNum: return !v.empty() && std::isdigit(uint8_t(v.back())) && value_prefix_ok(p, v);
    case Kind::kJson: { const auto j = nlohmann::json::parse(v, nullptr, false); return !j.is_discarded() && (j.is_object() || j.is_array()); }
    case Kind::kAny: return !nlohmann::json::parse(v, nullptr, false).is_discarded();
    }
    return true;
}

namespace {
constexpr std::string_view kD = "<｜DSML｜", kDOpenRest = " calls>\n", kDInv = "<｜DSML｜ invoke name=\"", kDNameEnd = "\">\n",
    kDPar = "<｜DSML｜ parameter name=\"", kDKeyEnd = "\" string=\"", kDPClose = "</｜DSML｜ parameter>\n",
    kDIClose = "</｜DSML｜ invoke>\n", kDCClose = "</｜DSML｜ calls>";
}

bool ToolGuard::step_dsml(State& s, char c) const {
    switch (s.mode) {
    case Mode::kThink:
        s.buf += c;
        if (ends_with(s.buf, t_.think_close)) { s.mode = Mode::kFree; s.buf.clear(); }
        else if (s.buf.size() > 64) s.buf.erase(0, s.buf.size() - t_.think_close.size());
        return true;
    case Mode::kFree:
        if (need_call_ && s.calls == 0) {   // tool_choice required: the block opens at once
            static const std::string lead = "\n\n" + std::string(kD);
            s.buf += c;
            if (!starts_with(lead, s.buf)) return false;
            if (s.buf == lead) { s.mode = Mode::kDOpen; s.buf.clear(); }
            return true;
        }
        s.buf += c;
        if (ends_with(s.buf, kD)) {   // a DSML block opens only after a blank line (the parser's "\n\n<｜DSML｜ calls>\n")
            if (no_calls_) return false;    // tool_choice none
            const std::string_view before(s.buf.data(), s.buf.size() - kD.size());
            if (!ends_with(before, "\n\n")) return false;
            s.mode = Mode::kDOpen; s.buf.clear();
        } else if (s.buf.size() > 64) s.buf.erase(0, s.buf.size() - kD.size() - 2);
        return true;
    case Mode::kDOpen:
        s.buf += c;
        if (!starts_with(kDOpenRest, s.buf)) return false;
        if (s.buf.size() == kDOpenRest.size()) { s.mode = Mode::kDInvoke; s.buf.clear(); }
        return true;
    case Mode::kDInvoke: case Mode::kDAfterInvoke: {
        s.buf += c;
        if (starts_with(kDInv, s.buf)) {
            if (s.buf.size() == kDInv.size()) { s.mode = Mode::kDName; s.buf.clear(); s.tool = -1; s.used = 0; }
            return true;
        }
        if (s.mode == Mode::kDAfterInvoke && starts_with(kDCClose, s.buf)) {
            if (s.buf.size() == kDCClose.size()) { s.mode = Mode::kDDone; s.buf.clear(); }
            return true;
        }
        return false;
    }
    case Mode::kDName: {
        s.buf += c;
        for (size_t i = 0; i < tools_.size(); ++i) {
            if (!name_ok(i)) continue;
            const std::string whole = tools_[i].name + std::string(kDNameEnd);
            if (starts_with(whole, s.buf)) {
                if (s.buf == whole) { s.tool = int32_t(i); s.mode = Mode::kDParams; s.buf.clear(); }
                return true;
            }
        }
        if ((open_names_ || prompt_names_) && only_tool_ < 0) {
            const int m = open_match(s.buf, {kDNameEnd});
            if (m) { size_t k = 0; while (k < s.buf.size() && ident(s.buf[k])) ++k;
                     if (!name_allowed(std::string_view(s.buf).substr(0, k), m >= 2 || k < s.buf.size())) return false; }
            if (m == 2) { s.tool = kOpen; s.keys.clear(); s.used = 0; s.mode = Mode::kDParams; s.buf.clear(); }
            return m != 0;
        }
        return false;
    }
    case Mode::kDParams: {
        s.buf += c;
        const bool open = s.tool == kOpen;
        const Tool* tp = open ? nullptr : &tools_[size_t(s.tool)];
        const bool more = open || std::popcount(s.used) < int(tp->params.size()), done = open || (s.used & tp->required) == tp->required;
        if (open && !s.keys.empty()) s.used = 1;   // (the zero-argument blank line only before any key)
        if (more && starts_with(kDPar, s.buf)) {
            if (s.buf.size() == kDPar.size()) { s.mode = Mode::kDKey; s.buf.clear(); }
            return true;
        }
        if (done) {
            const std::string blank = "\n" + std::string(kDIClose);   // (a call with no arguments: the reference's blank line)
            if (starts_with(kDIClose, s.buf) || (s.used == 0 && starts_with(blank, s.buf))) {
                if (s.buf == kDIClose || s.buf == blank) { s.mode = Mode::kDAfterInvoke; s.buf.clear(); ++s.calls; }
                return true;
            }
        }
        return false;
    }
    case Mode::kDKey: {
        s.buf += c;
        if (s.tool == kOpen) {
            const int m = open_match(s.buf, {kDKeyEnd});
            if (m == 2) {
                const std::string key = "\x1f" + s.buf.substr(0, s.buf.size() - kDKeyEnd.size()) + "\x1f";
                if (s.keys.find(key) != std::string::npos) return false;
                s.keys += key; s.mode = Mode::kDFlag; s.buf.clear(); s.param = -1;
            }
            return m != 0;
        }
        const Tool& t = tools_[size_t(s.tool)];
        for (size_t i = 0; i < t.params.size(); ++i) {
            if (s.used >> i & 1) continue;
            const std::string whole = t.params[i].name + std::string(kDKeyEnd);
            if (starts_with(whole, s.buf)) {
                if (s.buf == whole) { s.param = int32_t(i); s.used |= uint64_t(1) << i; s.mode = Mode::kDFlag; s.buf.clear(); }
                return true;
            }
        }
        return false;
    }
    case Mode::kDFlag: {
        s.buf += c;
        if (s.tool == kOpen) {   // either flag; "false" = a JSON value
            const bool t1 = starts_with("true\">", s.buf), f1 = starts_with("false\">", s.buf);
            if (!t1 && !f1) return false;
            if (s.buf == "true\">") { s.param = -1; s.mode = Mode::kDValue; s.buf.clear(); }
            else if (s.buf == "false\">") { s.param = -2; s.mode = Mode::kDValue; s.buf.clear(); }
            return true;
        }
        const Param& p = tools_[size_t(s.tool)].params[size_t(s.param)];
        const std::string want = (p.kind == Kind::kStr || p.kind == Kind::kEnum) ? "true\">" : "false\">";
        if (!starts_with(want, s.buf)) return false;
        if (s.buf == want) { s.mode = Mode::kDValue; s.buf.clear(); }
        return true;
    }
    case Mode::kDValue: {
        s.buf += c;
        const Param& p = s.tool == kOpen ? (s.param == -2 ? open_json_ : open_param_) : tools_[size_t(s.tool)].params[size_t(s.param)];
        if (ends_with(s.buf, kDPClose)) {
            const std::string_view v(s.buf.data(), s.buf.size() - kDPClose.size());
            if (!value_ok(p, v)) return false;
            s.mode = Mode::kDParams; s.buf.clear();
            return true;
        }
        if (p.kind == Kind::kStr) return true;
        const size_t pt = partial_tag(s.buf, kDPClose);
        const std::string_view v(s.buf.data(), s.buf.size() - pt);
        return pt ? value_ok(p, v) : value_prefix_ok(p, v);
    }
    case Mode::kDDone: return false;   // the block is the reply's end
    default: return false;
    }
}

namespace {
constexpr std::string_view kXFnOpen = "<function=", kXParOpen = "<parameter=", kXParClose = "</parameter>", kXFnClose = "</function>",
    kXCallClose = "</tool_call>";
bool is_ws(char c) { return c == ' ' || c == '\n'; }
std::string_view strip_nl(std::string_view v) {
    if (!v.empty() && v.front() == '\n') v.remove_prefix(1);
    if (!v.empty() && v.back() == '\n') v.remove_suffix(1);
    return v;
}
}

// Qwen XML: whitespace between the tags is free (at most 2 characters at each place), as the parser trims it
bool ToolGuard::step_xml(State& s, char c) const {
    switch (s.mode) {
    case Mode::kXFn: {   // after <tool_call>: "<function="
        if (s.buf.empty() && is_ws(c)) return ++s.ws <= 2;
        s.buf += c;
        if (!starts_with(kXFnOpen, s.buf)) return false;
        if (s.buf.size() == kXFnOpen.size()) { s.mode = Mode::kXName; s.buf.clear(); s.tool = -1; s.used = 0; }
        return true;
    }
    case Mode::kXName: {
        s.buf += c;
        for (size_t i = 0; i < tools_.size(); ++i) {
            if (!name_ok(i)) continue;
            const std::string whole = tools_[i].name + ">";
            if (starts_with(whole, s.buf)) {
                if (s.buf == whole) { s.tool = int32_t(i); s.mode = Mode::kXParams; s.buf.clear(); s.ws = 0; }
                return true;
            }
        }
        if ((open_names_ || prompt_names_) && only_tool_ < 0) {
            const int m = open_match(s.buf, {">"});
            if (m) { size_t k = 0; while (k < s.buf.size() && ident(s.buf[k])) ++k;
                     if (!name_allowed(std::string_view(s.buf).substr(0, k), m >= 2 || k < s.buf.size())) return false; }
            if (m == 2) { s.tool = kOpen; s.keys.clear(); s.mode = Mode::kXParams; s.buf.clear(); s.ws = 0; }
            return m != 0;
        }
        return false;
    }
    case Mode::kXParams: {
        if (s.buf.empty() && is_ws(c)) return ++s.ws <= 2;
        s.buf += c;
        const bool open = s.tool == kOpen;
        const Tool* tp = open ? nullptr : &tools_[size_t(s.tool)];
        const bool more = open || std::popcount(s.used) < int(tp->params.size()), done = open || (s.used & tp->required) == tp->required;
        if (more && starts_with(kXParOpen, s.buf)) {
            if (s.buf.size() == kXParOpen.size()) { s.mode = Mode::kXKey; s.buf.clear(); }
            return true;
        }
        if (done && starts_with(kXFnClose, s.buf)) {
            if (s.buf.size() == kXFnClose.size()) { s.mode = Mode::kXAfterFn; s.buf.clear(); s.ws = 0; }
            return true;
        }
        return false;
    }
    case Mode::kXKey: {
        s.buf += c;
        if (s.tool == kOpen) {
            const int m = open_match(s.buf, {">"});
            if (m == 2) {
                const std::string key = "\x1f" + s.buf.substr(0, s.buf.size() - 1) + "\x1f";
                if (s.keys.find(key) != std::string::npos) return false;
                s.keys += key; s.mode = Mode::kXValue; s.buf.clear();
            }
            return m != 0;
        }
        const Tool& t = tools_[size_t(s.tool)];
        for (size_t i = 0; i < t.params.size(); ++i) {
            if (s.used >> i & 1) continue;
            const std::string whole = t.params[i].name + ">";
            if (starts_with(whole, s.buf)) {
                if (s.buf == whole) { s.param = int32_t(i); s.used |= uint64_t(1) << i; s.mode = Mode::kXValue; s.buf.clear(); }
                return true;
            }
        }
        return false;
    }
    case Mode::kXValue: {
        s.buf += c;
        const Param& p = s.tool == kOpen ? open_param_ : tools_[size_t(s.tool)].params[size_t(s.param)];
        if (ends_with(s.buf, kXParClose)) {
            const std::string_view v = strip_nl(std::string_view(s.buf.data(), s.buf.size() - kXParClose.size()));
            if (!value_ok(p, v)) return false;
            s.mode = Mode::kXParams; s.buf.clear(); s.ws = 0;
            return true;
        }
        if (p.kind == Kind::kStr) return true;
        const size_t pt = partial_tag(s.buf, kXParClose);
        std::string_view v(s.buf.data(), s.buf.size() - pt);
        if (!v.empty() && v.front() == '\n') v.remove_prefix(1);
        if (pt) return value_ok(p, strip_nl(v));
        if (!v.empty() && v.back() == '\n') return value_ok(p, strip_nl(v));   // (the format's newline before the closing tag)
        return value_prefix_ok(p, v);
    }
    case Mode::kXAfterFn: {
        if (s.buf.empty() && is_ws(c)) return ++s.ws <= 2;
        s.buf += c;
        if (!starts_with(kXCallClose, s.buf)) return false;
        if (s.buf.size() == kXCallClose.size()) { s.mode = Mode::kFree; s.buf.clear(); ++s.calls; }
        return true;
    }
    default: return false;
    }
}

bool ToolGuard::step(State& s, char c) const {
    if (dsml_) return step_dsml(s, c);
    if (xml_ && s.mode != Mode::kThink && s.mode != Mode::kFree) return step_xml(s, c);
    switch (s.mode) {
    case Mode::kThink:
        s.buf += c;
        if (ends_with(s.buf, t_.think_close)) { s.mode = Mode::kFree; s.buf.clear(); }
        else if (s.buf.size() > 64) s.buf.erase(0, s.buf.size() - t_.think_close.size());
        return true;
    case Mode::kFree:
        if (need_call_ && s.calls == 0) {   // tool_choice required: the call opens at once (a word-start space or newline before it)
            if (s.buf.empty() && (c == ' ' || c == '\n')) return ++s.ws <= 2;
            s.buf += c;
            if (!starts_with(t_.open, s.buf)) return false;
            if (s.buf == t_.open) { s.mode = xml_ ? Mode::kXFn : Mode::kName; s.buf.clear(); s.tool = -1; s.used = 0; s.ws = 0; }
            return true;
        }
        s.buf += c;
        if (ends_with(s.buf, t_.open)) {
            if (no_calls_) return false;    // tool_choice none
            s.mode = xml_ ? Mode::kXFn : Mode::kName; s.buf.clear(); s.tool = -1; s.used = 0; s.ws = 0;
        }
        else if (s.buf.size() > 64) s.buf.erase(0, s.buf.size() - t_.open.size());
        return true;
    case Mode::kName: {
        if (s.buf.empty() && (c == ' ' || c == '\n')) return ++s.ws <= 2;   // (a word-start space before the name)
        s.buf += c;
        for (size_t i = 0; i < tools_.size(); ++i) {
            if (!name_ok(i)) continue;
            const Tool& t = tools_[i];
            const std::string with_key = t.name + t_.key_open;
            if (!t.params.empty() && starts_with(with_key, s.buf)) {
                if (s.buf == with_key) { s.tool = int32_t(i); s.mode = Mode::kKey; s.buf.clear(); }
                return true;
            }
        }
        for (size_t i = 0; i < tools_.size(); ++i) {
            if (!name_ok(i)) continue;
            const Tool& t = tools_[i];
            const std::string closed = t.name + t_.close;
            if (t.required == 0 && starts_with(closed, s.buf)) {
                if (s.buf == closed) { s.mode = Mode::kFree; s.buf.clear(); ++s.calls; }
                return true;
            }
        }
        if ((open_names_ || prompt_names_) && only_tool_ < 0) {
            const int m = open_match(s.buf, {t_.key_open, t_.close});
            if (m) { size_t k = 0; while (k < s.buf.size() && ident(s.buf[k])) ++k;
                     if (!name_allowed(std::string_view(s.buf).substr(0, k), m >= 2 || k < s.buf.size())) return false; }
            if (m == 2) { s.tool = kOpen; s.keys.clear(); s.mode = Mode::kKey; s.buf.clear(); }
            else if (m == 3) { s.mode = Mode::kFree; s.buf.clear(); ++s.calls; }
            return m != 0;
        }
        return false;
    }
    case Mode::kKey: {
        if (s.buf.empty() && (c == ' ' || c == '\n')) return true;
        s.buf += c;
        if (s.tool == kOpen) {
            const int m = open_match(s.buf, {t_.key_close + t_.val_open});
            if (m == 2) {
                const std::string key = "\x1f" + s.buf.substr(0, s.buf.size() - t_.key_close.size() - t_.val_open.size()) + "\x1f";
                if (s.keys.find(key) != std::string::npos) return false;
                s.keys += key; s.mode = Mode::kValue; s.buf.clear();
            }
            return m != 0;
        }
        const Tool& t = tools_[size_t(s.tool)];
        for (size_t i = 0; i < t.params.size(); ++i) {
            if (s.used >> i & 1) continue;
            const std::string whole = t.params[i].name + t_.key_close + t_.val_open;
            if (starts_with(whole, s.buf)) {
                if (s.buf == whole) { s.param = int32_t(i); s.used |= uint64_t(1) << i; s.mode = Mode::kValue; s.buf.clear(); }
                return true;
            }
        }
        return false;
    }
    case Mode::kValue: {
        s.buf += c;
        const Param& p = s.tool == kOpen ? open_param_ : tools_[size_t(s.tool)].params[size_t(s.param)];
        if (ends_with(s.buf, t_.val_close)) {
            const std::string_view v(s.buf.data(), s.buf.size() - t_.val_close.size());
            if (!value_ok(p, v)) return false;
            s.mode = Mode::kAfter; s.buf.clear();
            return true;
        }
        if (p.kind == Kind::kStr) return true;
        const size_t pt = partial_tag(s.buf, t_.val_close);
        const std::string_view v(s.buf.data(), s.buf.size() - pt);
        return pt ? value_ok(p, v) : value_prefix_ok(p, v);
    }
    case Mode::kAfter: {
        s.buf += c;
        const bool open = s.tool == kOpen;
        const Tool* t = open ? nullptr : &tools_[size_t(s.tool)];
        const bool more = open || std::popcount(s.used) < int(t->params.size());
        if (more && starts_with(t_.key_open, s.buf)) {
            if (s.buf == t_.key_open) { s.mode = Mode::kKey; s.buf.clear(); }
            return true;
        }
        if ((open || (s.used & t->required) == t->required) && starts_with(t_.close, s.buf)) {
            if (s.buf == t_.close) { s.mode = Mode::kFree; s.buf.clear(); ++s.calls; }
            return true;
        }
        return false;
    }
    default: return false;
    }
    return false;
}

bool ToolGuard::feed(State& s, std::string_view text) const {
    for (char c : text) if (!step(s, c)) return false;
    return true;
}

// ------------------------------------------------------------------------------------------- a request's guard state
void LaneGuard::apply_raw(float* logits, size_t Vin) const {
    const uint32_t V = uint32_t(std::min(Vin, pieces->size()));
    const bool call = ToolGuard::in_call(st);
    const uint32_t N = std::min<uint32_t>(V, call ? 256 : 48);
    std::vector<int32_t> idx(V);
    for (uint32_t i = 0; i < V; ++i) idx[i] = int32_t(i);
    std::partial_sort(idx.begin(), idx.begin() + N, idx.end(), [&](int32_t a, int32_t b) { return logits[size_t(a)] > logits[size_t(b)]; });
    auto ok = [&](int32_t id) {
        if (stop && stop(id)) return g->may_stop(st);
        ToolGuard::State s = st;
        return g->feed(s, (*pieces)[size_t(id)]);
    };
    constexpr float kOff = -1e30f;
    uint32_t valid = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (ok(idx[i])) ++valid;
        else { logits[size_t(idx[i])] = kOff; ++masked; }
    }
    if (valid) {         // outside a call only the likely tokens are checked (the sampler's top-k is within them)
        if (call) for (uint32_t i = N; i < V; ++i) logits[size_t(idx[i])] = kOff;
        return;
    }
    ++scans;             // no likely token continues the call: every token is checked
    uint32_t any = 0;
    for (uint32_t i = N; i < V; ++i) { if (ok(idx[i])) ++any; else logits[size_t(idx[i])] = kOff; }
    (void)any;
}

void LaneGuard::advance(int32_t id) {
    static const bool trace = std::getenv("IE_TOOL_GUARD_TRACE") && *std::getenv("IE_TOOL_GUARD_TRACE") == '1';
    if (stop && stop(id)) return;
    ToolGuard::State s = st;
    const bool fed = g->feed(s, (*pieces)[size_t(id)]);
    if (trace) std::fprintf(stderr, "[guard] id %d %-20s ok=%d mode %d -> %d ws %d\n", id, ("'" + (*pieces)[size_t(id)] + "'").c_str(), fed,
                            int(st.mode), int(fed ? s.mode : st.mode), int(fed ? s.ws : st.ws));
    if (fed) st = std::move(s);
    else { st = g->start(false); }   // (an unchecked token broke the call: the guard lets go of it rather than wedge the lane)
}

std::vector<std::string> tool_guard_pieces(uint32_t V, const std::function<std::string(const int32_t*, size_t)>& detok) {
    std::vector<std::string> p(V);
    for (uint32_t id = 0; id < V; ++id) {
        const int32_t two[2] = {int32_t(id), int32_t(id)};
        const std::string a = detok(two, 1), b = detok(two, 2);
        p[id] = b.size() >= a.size() && b.compare(0, a.size(), a) == 0 ? b.substr(a.size()) : a;
    }
    return p;
}

}  // namespace ie
