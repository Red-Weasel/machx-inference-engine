#pragma once
#include "ie/openai_proto.hpp"
#include "ie/reasoning.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ie {
struct LaunchOptions {
    EngineOptions engine;
    oai::ChatRequest defaults;
    std::string host = "127.0.0.1";
    uint16_t port = 11435;
    uint32_t max_queue = 8;   // requests allowed to WAIT for a generation slot; beyond → HTTP 429
    bool max_queue_set = false;   // --max-queue given (P4 B30: else serve_max_queue picks the default)
    double vram_reserve_gib = -1;   // --vram-reserve-gib: VRAM the auto expert tier leaves free per card (mimo_v2, deepseek41); -1 = the engine default
    std::vector<uint32_t> cards;    // --cards: physical cards this process may use (ascending, unique); empty = every card (today's default)
};

// The launch flags that take no value; every other flag parse_launch_options accepts takes exactly one.
// tests/unit/cli_help_test.cpp derives both sets from the parser below and checks this list against it.
inline bool launch_flag_is_boolean(const std::string& f) {
    return f == "--no-prompt-cache" || f == "--int8-kv" || f == "--spec";
}

// "--cards 0,1" -> {0, 1}: card numbers 0..63, unique, returned ascending (docs/serve_config.md).
inline std::vector<uint32_t> parse_card_list(const std::string& v) {
    std::vector<uint32_t> cards;
    size_t pos = 0;
    while (true) {
        const size_t comma = v.find(',', pos);
        const std::string item = v.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        uint32_t n = 0;
        auto [p, ec] = std::from_chars(item.data(), item.data() + item.size(), n);
        if (item.empty() || ec != std::errc{} || p != item.data() + item.size() || n > 63)
            throw std::runtime_error("--cards requires a comma-separated list of card numbers in [0, 63], e.g. 1 or 0,1");
        if (std::find(cards.begin(), cards.end(), n) != cards.end())
            throw std::runtime_error("--cards lists card " + std::to_string(n) + " twice");
        cards.push_back(n);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    std::sort(cards.begin(), cards.end());
    return cards;
}

// Parse without side effects: validation completes before loading a model or
// publishing defaults to HTTP workers. Every accepted flag has a consumer.
inline LaunchOptions parse_launch_options(const std::vector<std::string>& args) {
    LaunchOptions out;
    out.engine.n_gpus = 0;
    out.engine.parallel = kLanesAuto;   // P4 B30: no --parallel = the load picks the lanes (ie/lanes_auto.hpp)
    out.defaults = oai::server_defaults_from_environment();
    auto& e = out.engine;
    auto& s = out.defaults.sampling;
    auto& set = out.defaults.sampling_set;
    for (size_t i = 0; i < args.size(); ++i) {
        const auto& flag = args[i];
        auto value = [&]() -> const std::string& {
            if (i + 1 >= args.size()) throw std::runtime_error(flag + " requires a value");
            return args[++i];
        };
        auto integer = [&](uint64_t lo, uint64_t hi) {
            const auto& v = value(); uint64_t n = 0;
            auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), n);
            if (ec != std::errc{} || p != v.data() + v.size() || n < lo || n > hi)
                throw std::runtime_error(flag + " requires an integer in [" +
                    std::to_string(lo) + ", " + std::to_string(hi) + "]");
            return n;
        };
        auto number = [&](double lo, double hi, bool exclusive_lo = false) {
            const auto& v = value(); double n = 0;
            auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), n);
            if (ec != std::errc{} || p != v.data() + v.size() || !std::isfinite(n) ||
                n < lo || n > hi || (exclusive_lo && n == lo))
                throw std::runtime_error(flag + " requires a finite number in " +
                    (exclusive_lo ? "(" : "[") + std::to_string(lo) + ", " + std::to_string(hi) + "]");
            const float narrowed = float(n);
            if (!std::isfinite(narrowed) || (n != 0 && narrowed == 0))
                throw std::runtime_error(flag + " is not representable as a sampling float");
            return narrowed;
        };
        if (flag == "--ctx") e.max_ctx = uint32_t(integer(9, INT32_MAX));
        else if (flag == "--gpus") e.n_gpus = uint32_t(integer(0, 64));
        else if (flag == "--host") {
            out.host = value();
            if (out.host.empty()) throw std::runtime_error("--host cannot be empty");
        }
        else if (flag == "--port") out.port = uint16_t(integer(1, 65535));
        // P4 B20: a CLI sampling flag outranks the model's recommended sampling (oai::apply_recommended)
        else if (flag == "--temp") { s.temperature = number(0, 2); set |= oai::kSetTemperature; }
        else if (flag == "--top-k") { s.top_k = uint32_t(integer(0, 1024)); set |= oai::kSetTopK; }
        else if (flag == "--top-p") { s.top_p = number(0, 1, true); set |= oai::kSetTopP; }
        else if (flag == "--min-p") { s.min_p = number(0, 1); set |= oai::kSetMinP; }
        else if (flag == "--repeat-penalty") { s.repeat_penalty = number(0, 10, true); set |= oai::kSetRepeat; }
        else if (flag == "--repeat-last-n") s.repeat_window = uint32_t(integer(0, 512));
        else if (flag == "--presence-penalty") { s.presence_penalty = number(-2, 2); set |= oai::kSetPresence; }
        else if (flag == "--frequency-penalty") s.frequency_penalty = number(-2, 2);
        else if (flag == "--seed") s.seed = integer(0, UINT64_MAX);
        else if (flag == "--max-tokens") {
            s.max_tokens = uint32_t(integer(0, UINT32_MAX));
            if (s.max_tokens == 0) s.max_tokens = kMaxTokensUnlimited;
        }
        else if (flag == "--stop") {
            const auto& v = value();
            if (v.empty() || out.defaults.stop.size() >= 4)
                throw std::runtime_error("--stop accepts up to 4 nonempty literal strings");
            out.defaults.stop.push_back(v);
        }
        else if (flag == "--threads") e.cpu_threads = uint32_t(integer(1, 1024));
        else if (flag == "--no-prompt-cache") e.prompt_cache = false;
        else if (flag == "--thinking") {
            const auto& v = value();
            if (v != "on" && v != "off") throw std::runtime_error("--thinking requires on or off");
            out.defaults.enable_thinking = v == "on";
        }
        else if (flag == "--reasoning-effort") {
            const auto& v=value();
            if(!known_reasoning_effort(v))throw std::runtime_error("--reasoning-effort requires a supported model effort level");
            out.defaults.reasoning_effort=v;
        }
        else if (flag == "--prefill-chunk") e.prefill_chunk = uint32_t(integer(1, INT32_MAX));
        else if (flag == "--vram-reserve-gib") out.vram_reserve_gib = double(number(0, 24));
        else if (flag == "--parallel") {   // P4 B30: "auto" = the default
            if (i + 1 < args.size() && args[i + 1] == "auto") { ++i; e.parallel = kLanesAuto; }
            else e.parallel = uint32_t(integer(1, kMaxParallel));
        }
        else if (flag == "--max-queue") { out.max_queue = uint32_t(integer(0, 1024)); out.max_queue_set = true; }
        else if (flag == "--slot-ctx") e.slot_ctx = uint32_t(integer(0, INT32_MAX));
        else if (flag == "--int8-kv") e.int8_kv = true;
        else if (flag == "--spec") e.spec = true;
        else if (flag == "--spec-k") e.spec_k = uint32_t(integer(1, 64));
        else if (flag == "--spec-head") e.spec_head = value();
        else if (flag == "--spec-draft") e.spec_draft = value();
        else if (flag == "--mmproj") {   // P4 B45: the vision projector GGUF
            e.mmproj = value();
            if (e.mmproj.empty()) throw std::runtime_error("--mmproj cannot be empty");
        }
        else if (flag == "--cards") out.cards = parse_card_list(value());
        else throw std::runtime_error("unknown option: " + flag);
    }
    if (e.parallel > 1 && e.int8_kv)
        throw std::runtime_error("--parallel > 1 does not support --int8-kv");
    if (e.parallel == kLanesAuto && e.int8_kv) e.parallel = 1;   // P4 B30: the lanes refuse --int8-kv, so auto = one lane
    if (e.slot_ctx > 0 && e.slot_ctx < 9)
        throw std::runtime_error("--slot-ctx must be 0 (automatic) or at least 9");
    if (e.slot_ctx > e.max_ctx)
        throw std::runtime_error("--slot-ctx cannot exceed --ctx");
    if (!out.cards.empty() && e.n_gpus > out.cards.size())
        throw std::runtime_error("--gpus " + std::to_string(e.n_gpus) + " exceeds the " +
                                 std::to_string(out.cards.size()) + " card(s) chosen with --cards");
    return out;
}

// P4 B30: the --max-queue a server keeps with `lanes` request lanes running: the flag when given, 8 beside an explicit
// --parallel (as before), else lanes + 8 -- the load picked N, so N + 8 may wait behind the N running.
inline uint32_t serve_max_queue(const LaunchOptions& l, uint32_t lanes) {
    return l.max_queue_set || l.engine.parallel != kLanesAuto ? l.max_queue : lanes + 8;
}
} // namespace ie
