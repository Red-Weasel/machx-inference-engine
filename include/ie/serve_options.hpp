#pragma once
#include "ie/openai_proto.hpp"
#include "ie/reasoning.hpp"
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
};

// Parse without side effects: validation completes before loading a model or
// publishing defaults to HTTP workers. Every accepted flag has a consumer.
inline LaunchOptions parse_launch_options(const std::vector<std::string>& args) {
    LaunchOptions out;
    out.engine.n_gpus = 0;
    out.defaults = oai::server_defaults_from_environment();
    auto& e = out.engine;
    auto& s = out.defaults.sampling;
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
        else if (flag == "--temp") s.temperature = number(0, 2);
        else if (flag == "--top-k") s.top_k = uint32_t(integer(0, 1024));
        else if (flag == "--top-p") s.top_p = number(0, 1, true);
        else if (flag == "--min-p") s.min_p = number(0, 1);
        else if (flag == "--repeat-penalty") s.repeat_penalty = number(0, 10, true);
        else if (flag == "--repeat-last-n") s.repeat_window = uint32_t(integer(0, 512));
        else if (flag == "--presence-penalty") s.presence_penalty = number(-2, 2);
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
        else if (flag == "--parallel") e.parallel = uint32_t(integer(1, 4));
        else if (flag == "--max-queue") out.max_queue = uint32_t(integer(0, 1024));
        else if (flag == "--slot-ctx") e.slot_ctx = uint32_t(integer(0, INT32_MAX));
        else if (flag == "--int8-kv") e.int8_kv = true;
        else if (flag == "--spec") e.spec = true;
        else if (flag == "--spec-k") e.spec_k = uint32_t(integer(1, 64));
        else if (flag == "--spec-head") e.spec_head = value();
        else if (flag == "--spec-draft") e.spec_draft = value();
        else throw std::runtime_error("unknown option: " + flag);
    }
    if (e.parallel > 1 && e.int8_kv)
        throw std::runtime_error("--parallel > 1 does not support --int8-kv");
    if (e.slot_ctx > 0 && e.slot_ctx < 9)
        throw std::runtime_error("--slot-ctx must be 0 (automatic) or at least 9");
    if (e.slot_ctx > e.max_ctx)
        throw std::runtime_error("--slot-ctx cannot exceed --ctx");
    return out;
}
} // namespace ie
