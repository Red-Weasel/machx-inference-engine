#pragma once
// `ie serve --config <layout.json>` (docs/serve_config.md): a launch layout in JSON (third_party/nlohmann, vendored).
//
// A layout lists servers: {"servers": [{"name": ..., "model": ..., "cards": [0], "port": ..., <launch keys>}]}.
// Each launch key is a command-line flag spelled with '_' ("max_queue" = --max-queue); the file is turned into those
// flags and validated by the SAME parser as the command line (parse_launch_options), then the command line's own flags
// are appended, so a flag given on the command line overrides the file. `ie serve --config` runs one server (the only
// one, or the one --server NAME picks); `ie supervise --config` runs every server of the layout as its own `ie serve`
// child behind one routing endpoint (ie/supervisor.hpp). The top-level "default", "host" and "port" keys and the
// per-server "restart" key are the supervisor's; `ie serve` ignores them.
#include "ie/serve_options.hpp"
#include "nlohmann/json.hpp"
#include <netdb.h>
#include <sys/socket.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ie {

enum class ConfigKind { Int, Num, Str, Path, Flag, NotFlag, OnOff, StrList, CardList };
struct ConfigKey { const char* key; const char* flag; ConfigKind kind; };

// Every launch flag, keyed. tests/unit/cli_help_test.cpp checks this table against the flags the parser accepts.
inline const std::vector<ConfigKey>& serve_config_keys() {
    using K = ConfigKind;
    static const std::vector<ConfigKey> keys = {
        {"ctx", "--ctx", K::Int},                     {"gpus", "--gpus", K::Int},
        {"cards", "--cards", K::CardList},            {"host", "--host", K::Str},
        {"port", "--port", K::Int},                   {"parallel", "--parallel", K::Int},
        {"max_queue", "--max-queue", K::Int},         {"slot_ctx", "--slot-ctx", K::Int},
        {"prefill_chunk", "--prefill-chunk", K::Int}, {"threads", "--threads", K::Int},
        {"prompt_cache", "--no-prompt-cache", K::NotFlag},
        {"vram_reserve_gib", "--vram-reserve-gib", K::Num},
        {"int8_kv", "--int8-kv", K::Flag},            {"spec", "--spec", K::Flag},
        {"spec_k", "--spec-k", K::Int},               {"spec_head", "--spec-head", K::Path},
        {"spec_draft", "--spec-draft", K::Path},     {"mmproj", "--mmproj", K::Path},
        {"temp", "--temp", K::Num},                   {"top_k", "--top-k", K::Int},
        {"top_p", "--top-p", K::Num},                 {"min_p", "--min-p", K::Num},
        {"repeat_penalty", "--repeat-penalty", K::Num}, {"repeat_last_n", "--repeat-last-n", K::Int},
        {"presence_penalty", "--presence-penalty", K::Num},
        {"frequency_penalty", "--frequency-penalty", K::Num},
        {"seed", "--seed", K::Int},                   {"max_tokens", "--max-tokens", K::Int},
        {"stop", "--stop", K::StrList},               {"thinking", "--thinking", K::OnOff},
        {"reasoning_effort", "--reasoning-effort", K::Str},
    };
    return keys;
}

struct ServeConfigServer {
    std::string name;                                      // "" when not given (single-server layouts)
    std::string model;                                     // resolved path ("" when not given)
    bool restart_on_failure = false;                       // "restart": "on-failure" (supervisor only; default "none")
    std::vector<std::string> args;                         // the launch flags the entry stands for
    std::vector<std::pair<std::string, std::string>> env;  // "env": {"IE_X": "1"}
    LaunchOptions launch;                                  // args parsed (validation; ports and cards for the layout)
};
struct ServeConfig {
    std::vector<ServeConfigServer> servers;
    std::string default_server;   // top-level "default": the server a request without "model" goes to ("" = none)
    std::string front_host;       // top-level "host" / "port": the supervisor's front endpoint ("" / 0 = not given)
    uint16_t front_port = 0;
};

// The model id `ie serve` reports for a model path: the file name without ".gguf" (a checkpoint directory: its name).
inline std::string model_id_from_path(const std::string& path) {
    const auto s = path.find_last_of("/\\");
    const auto base = path.substr(s == std::string::npos ? 0 : s + 1);
    const auto e = base.rfind(".gguf");
    return e == std::string::npos ? base : base.substr(0, e);
}

namespace serve_config_detail {
inline std::string resolve_path(const std::string& p, const std::string& base_dir) {
    if (p.rfind("~/", 0) == 0) {
        const char* home = std::getenv("HOME");
        if (!home || !*home) throw std::runtime_error("cannot expand \"" + p + "\": HOME is not set");
        return std::string(home) + p.substr(1);
    }
    if (p.empty() || p[0] == '/' || base_dir.empty()) return p;
    return base_dir + "/" + p;
}
inline bool env_name_ok(const std::string& k) {
    if (k.empty() || (k[0] >= '0' && k[0] <= '9')) return false;
    for (char c : k) if (!((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) return false;
    return true;
}
}  // namespace serve_config_detail

// The addresses a bind host stands for: resolved (getaddrinfo, numeric form), so "localhost" is 127.0.0.1 / ::1;
// a host that does not resolve stands for itself.
inline std::vector<std::string> bind_addresses(const std::string& host) {
    std::vector<std::string> out;
    addrinfo hints{};
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) == 0) {
        for (addrinfo* p = res; p; p = p->ai_next) {
            char buf[NI_MAXHOST];
            if (getnameinfo(p->ai_addr, p->ai_addrlen, buf, sizeof buf, nullptr, 0, NI_NUMERICHOST) == 0) out.emplace_back(buf);
        }
        freeaddrinfo(res);
    }
    if (out.empty()) out.push_back(host);
    return out;
}
// Would two listeners on the same port with these bind hosts collide? Compared as resolved addresses; a wildcard
// (0.0.0.0 or ::) collides with everything (Linux binds :: dual-stack).
inline bool binds_overlap(const std::string& a, const std::string& b) {
    const auto wild = [](const std::string& x) { return x == "0.0.0.0" || x == "::"; };
    for (const auto& x : bind_addresses(a))
        for (const auto& y : bind_addresses(b))
            if (wild(x) || wild(y) || x == y) return true;
    return false;
}

// Parse a layout. base_dir resolves relative model/spec paths (the config file's directory). Throws with the entry
// and key named.
inline ServeConfig parse_serve_config(const std::string& text, const std::string& base_dir = "") {
    using nlohmann::json;
    namespace d = serve_config_detail;
    json root;
    try { root = json::parse(text); }
    catch (const json::parse_error& e) { throw std::runtime_error(std::string("not valid JSON: ") + e.what()); }
    if (!root.is_object()) throw std::runtime_error("the layout must be a JSON object with a \"servers\" array");
    for (auto it = root.begin(); it != root.end(); ++it)
        if (it.key() != "servers" && it.key() != "version" && it.key() != "default" && it.key() != "host" && it.key() != "port")
            throw std::runtime_error("unknown top-level key \"" + it.key() +
                                     "\" (expected \"servers\", \"version\", \"default\", \"host\", \"port\")");
    if (root.contains("version") && root["version"] != 1)
        throw std::runtime_error("\"version\" must be 1");
    if (!root.contains("servers") || !root["servers"].is_array() || root["servers"].empty())
        throw std::runtime_error("\"servers\" must be a non-empty array of server objects");

    ServeConfig cfg;
    if (root.contains("default")) {
        if (!root["default"].is_string() || root["default"].get<std::string>().empty())
            throw std::runtime_error("\"default\" must be the name of one of the servers");
        cfg.default_server = root["default"].get<std::string>();
    }
    if (root.contains("host")) {
        if (!root["host"].is_string() || root["host"].get<std::string>().empty())
            throw std::runtime_error("\"host\" (the supervisor's front address) must be a non-empty string");
        cfg.front_host = root["host"].get<std::string>();
    }
    if (root.contains("port")) {
        if (!root["port"].is_number_unsigned() || root["port"].get<uint64_t>() < 1 || root["port"].get<uint64_t>() > 65535)
            throw std::runtime_error("\"port\" (the supervisor's front port) must be an integer in [1, 65535]");
        cfg.front_port = uint16_t(root["port"].get<uint64_t>());
    }
    const auto& list = root["servers"];
    for (size_t i = 0; i < list.size(); ++i) {
        const auto& s = list[i];
        const std::string where = "servers[" + std::to_string(i) + "]";
        if (!s.is_object()) throw std::runtime_error(where + " must be an object");
        ServeConfigServer out;
        for (auto it = s.begin(); it != s.end(); ++it) {
            const std::string& k = it.key();
            const json& v = it.value();
            const std::string at = where + "." + k;
            if (k == "name") {
                if (!v.is_string() || v.get<std::string>().empty()) throw std::runtime_error(at + " must be a non-empty string");
                out.name = v.get<std::string>();
                continue;
            }
            if (k == "model") {
                if (!v.is_string() || v.get<std::string>().empty()) throw std::runtime_error(at + " must be a non-empty path string");
                out.model = d::resolve_path(v.get<std::string>(), base_dir);
                continue;
            }
            if (k == "restart") {
                if (!v.is_string() || (v != "none" && v != "on-failure"))
                    throw std::runtime_error(at + " must be \"none\" or \"on-failure\"");
                out.restart_on_failure = v == "on-failure";
                continue;
            }
            if (k == "env") {
                if (!v.is_object()) throw std::runtime_error(at + " must be an object of NAME: value");
                for (auto e = v.begin(); e != v.end(); ++e) {
                    if (!d::env_name_ok(e.key())) throw std::runtime_error(at + ": \"" + e.key() + "\" is not an environment variable name");
                    std::string val;
                    if (e.value().is_string()) val = e.value().get<std::string>();
                    else if (e.value().is_number_integer() || e.value().is_number_float()) val = e.value().dump();
                    else if (e.value().is_boolean()) val = e.value().get<bool>() ? "1" : "0";
                    else throw std::runtime_error(at + "." + e.key() + " must be a string, number or boolean");
                    out.env.emplace_back(e.key(), val);
                }
                continue;
            }
            const ConfigKey* ck = nullptr;
            for (const auto& c : serve_config_keys()) if (k == c.key) { ck = &c; break; }
            if (!ck) throw std::runtime_error("unknown key " + at + " (docs/serve_config.md lists the keys)");
            const std::string flag = ck->flag;
            switch (ck->kind) {
            case ConfigKind::Int:
                if (v.is_number_unsigned()) out.args.insert(out.args.end(), {flag, std::to_string(v.get<uint64_t>())});
                else if (v.is_number_integer()) out.args.insert(out.args.end(), {flag, std::to_string(v.get<int64_t>())});
                else throw std::runtime_error(at + " must be an integer");
                break;
            case ConfigKind::Num:
                if (!v.is_number()) throw std::runtime_error(at + " must be a number");
                out.args.insert(out.args.end(), {flag, v.dump()});
                break;
            case ConfigKind::Str:
            case ConfigKind::Path:
                if (!v.is_string()) throw std::runtime_error(at + " must be a string");
                out.args.insert(out.args.end(), {flag, ck->kind == ConfigKind::Path
                                                           ? d::resolve_path(v.get<std::string>(), base_dir) : v.get<std::string>()});
                break;
            case ConfigKind::Flag:
                if (!v.is_boolean()) throw std::runtime_error(at + " must be true or false");
                if (v.get<bool>()) out.args.push_back(flag);
                break;
            case ConfigKind::NotFlag:
                if (!v.is_boolean()) throw std::runtime_error(at + " must be true or false");
                if (!v.get<bool>()) out.args.push_back(flag);
                break;
            case ConfigKind::OnOff:
                if (v.is_boolean()) out.args.insert(out.args.end(), {flag, v.get<bool>() ? "on" : "off"});
                else if (v.is_string()) out.args.insert(out.args.end(), {flag, v.get<std::string>()});
                else throw std::runtime_error(at + " must be true/false or \"on\"/\"off\"");
                break;
            case ConfigKind::StrList:
                if (!v.is_array()) throw std::runtime_error(at + " must be an array of strings");
                for (const auto& x : v) {
                    if (!x.is_string()) throw std::runtime_error(at + " must be an array of strings");
                    out.args.insert(out.args.end(), {flag, x.get<std::string>()});
                }
                break;
            case ConfigKind::CardList: {
                if (!v.is_array() || v.empty()) throw std::runtime_error(at + " must be a non-empty array of card numbers, e.g. [1] or [0, 1]");
                std::string list;
                for (const auto& x : v) {
                    if (!x.is_number_unsigned()) throw std::runtime_error(at + " must be a non-empty array of card numbers, e.g. [1] or [0, 1]");
                    list += (list.empty() ? "" : ",") + std::to_string(x.get<uint64_t>());
                }
                out.args.insert(out.args.end(), {flag, list});
                break;
            }
            }
        }
        try { out.launch = parse_launch_options(out.args); }
        catch (const std::exception& e) { throw std::runtime_error(where + (out.name.empty() ? "" : " (" + out.name + ")") + ": " + e.what()); }
        cfg.servers.push_back(std::move(out));
    }

    // The multi-server shape (run by the P2 supervisor): every entry named, on its own port, on its own cards.
    if (cfg.servers.size() > 1) {
        for (size_t i = 0; i < cfg.servers.size(); ++i) {
            const auto& a = cfg.servers[i];
            const std::string where = "servers[" + std::to_string(i) + "]";
            if (a.name.empty()) throw std::runtime_error(where + ": every server of a multi-server layout needs a \"name\"");
            if (a.model.empty()) throw std::runtime_error(where + " (" + a.name + "): \"model\" is required");
            if (!list[i].contains("port")) throw std::runtime_error(where + " (" + a.name + "): every server of a multi-server layout needs a \"port\"");
            if (a.launch.cards.empty()) throw std::runtime_error(where + " (" + a.name + "): every server of a multi-server layout needs \"cards\"");
            for (size_t j = 0; j < i; ++j) {
                const auto& b = cfg.servers[j];
                if (a.name == b.name) throw std::runtime_error("two servers are named \"" + a.name + "\"");
                if (a.launch.port == b.launch.port && binds_overlap(a.launch.host, b.launch.host))
                    throw std::runtime_error("\"" + a.name + "\" and \"" + b.name + "\" both use port " + std::to_string(a.launch.port));
                for (uint32_t c : a.launch.cards)
                    for (uint32_t c2 : b.launch.cards)
                        if (c == c2) throw std::runtime_error("\"" + a.name + "\" and \"" + b.name + "\" both use card " + std::to_string(c) +
                                                              " (one server per card)");
            }
        }
    }
    if (!cfg.default_server.empty()) {
        bool found = false;
        for (const auto& s : cfg.servers) found = found || s.name == cfg.default_server;
        if (!found) throw std::runtime_error("\"default\": no server is named \"" + cfg.default_server + "\"");
    }
    return cfg;
}

inline ServeConfig load_serve_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    const auto slash = path.find_last_of('/');
    const std::string dir = slash == std::string::npos ? std::string(".") : path.substr(0, slash == 0 ? 1 : slash);
    try { return parse_serve_config(ss.str(), dir); }
    catch (const std::exception& e) { throw std::runtime_error(path + ": " + e.what()); }
}

// The single server an `ie serve --config` process runs. Several = `ie supervise`, or pick one with --server NAME.
inline const ServeConfigServer& single_server(const ServeConfig& cfg) {
    if (cfg.servers.size() != 1)
        throw std::runtime_error("this layout lists " + std::to_string(cfg.servers.size()) +
                                 " servers; run them all with `ie supervise --config <layout>`, or one with --server NAME");
    return cfg.servers.front();
}

// --server NAME: the layout's server of that name (how `ie supervise` starts each child).
inline const ServeConfigServer& named_server(const ServeConfig& cfg, const std::string& name) {
    std::string names;
    for (const auto& s : cfg.servers) {
        if (s.name == name) return s;
        names += (names.empty() ? "" : ", ") + (s.name.empty() ? std::string("(unnamed)") : s.name);
    }
    throw std::runtime_error("--server " + name + ": the layout has no server of that name (it has: " + names + ")");
}

// The flags a process parses: the file's, then the command line's (last one wins in parse_launch_options). --stop
// accumulates, so stop strings given on the command line REPLACE the file's instead of adding to them.
inline std::vector<std::string> merge_launch_args(const std::vector<std::string>& file_args,
                                                  const std::vector<std::string>& cli_args) {
    bool cli_stop = false;
    for (size_t i = 0; i < cli_args.size(); ++i) {
        if (cli_args[i] == "--stop") { cli_stop = true; break; }
        if (!launch_flag_is_boolean(cli_args[i])) ++i;   // skip the value
    }
    std::vector<std::string> out;
    for (size_t i = 0; i < file_args.size(); ++i) {
        const bool takes_value = !launch_flag_is_boolean(file_args[i]);
        if (!(cli_stop && file_args[i] == "--stop")) {
            out.push_back(file_args[i]);
            if (takes_value && i + 1 < file_args.size()) out.push_back(file_args[i + 1]);
        }
        if (takes_value) ++i;
    }
    out.insert(out.end(), cli_args.begin(), cli_args.end());
    return out;
}

// The layout's "env" block: a variable already set in the process environment wins over the file. Returns the names
// actually set.
inline std::vector<std::string> apply_config_env(const std::vector<std::pair<std::string, std::string>>& env) {
    std::vector<std::string> set;
    for (const auto& [k, v] : env)
        if (!std::getenv(k.c_str())) { setenv(k.c_str(), v.c_str(), 0); set.push_back(k); }
    return set;
}

}  // namespace ie
