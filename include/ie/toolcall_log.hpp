// include/ie/toolcall_log.hpp — the server's ToolCallObserver (docs/deepseek41/107): every tool call the parsers had to repair, and
// every one they refused, leaves one stderr line and one JSON line with the raw text of the call, so a failure that costs the agent a
// turn is captured instead of being a mystery. Only those rare events are written, never ordinary traffic.
//   IE_TOOLCALL_LOG=0            turns it off;  IE_TOOLCALL_LOG=<path>  moves it (default ~/.cache/machx-ie/toolcall-events.jsonl)
//   IE_TOOLCALL_LOG_MAX_BYTES    rotation size (default 8 MiB: the file is renamed to <path>.1 and started again)
// The raw text is cut to the call itself (a little context before the first DSML marker), not the reasoning that precedes it.
#pragma once

#include "ie/tool_call_repair.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

#include "../../third_party/nlohmann/json.hpp"

namespace ie {
namespace toolcall_log_detail {

inline std::string path() {
    const char* e = std::getenv("IE_TOOLCALL_LOG");
    if (e && *e) return std::string(e) == "0" ? std::string() : std::string(e);
    const char* h = std::getenv("HOME");
    return h && *h ? std::string(h) + "/.cache/machx-ie/toolcall-events.jsonl" : std::string();
}

inline void on_event(const ToolCallEvent& e) {
    try {
        std::fprintf(stderr, "[toolcall] %s %s: %s\n", e.family, e.kind, e.detail.c_str());
        const std::string file = path();
        if (file.empty()) return;
        static std::mutex mu;
        std::lock_guard<std::mutex> lk(mu);
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(fs::path(file).parent_path(), ec);
        uint64_t cap = 8ull << 20;
        if (const char* m = std::getenv("IE_TOOLCALL_LOG_MAX_BYTES"); m && *m) cap = std::strtoull(m, nullptr, 10);
        if (fs::exists(file, ec) && fs::file_size(file, ec) > cap) fs::rename(file, file + ".1", ec);
        const size_t at = e.raw.find("\xef\xbd\x9c" "DSML");                                  // ｜DSML
        const std::string raw = at == std::string::npos ? e.raw : e.raw.substr(at > 160 ? at - 160 : 0);
        char ts[32]; const std::time_t now = std::time(nullptr); std::strftime(ts, sizeof ts, "%FT%TZ", std::gmtime(&now));
        nlohmann::ordered_json j = {{"ts", ts}, {"family", e.family}, {"kind", e.kind}, {"detail", e.detail}, {"raw", raw}};
        std::ofstream out(file, std::ios::app);
        if (out) out << j.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace) << '\n';
    } catch (...) { /* a diagnostic must never take the server down */ }
}

}  // namespace toolcall_log_detail

inline void install_toolcall_log() { tool_call_observer() = toolcall_log_detail::on_event; }

}  // namespace ie
