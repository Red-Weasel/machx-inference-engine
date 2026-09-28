// Host-only: `ie serve --config` layouts (ie/serve_config.hpp), --cards parsing (ie/serve_options.hpp) and the card
// numbering (ie/card_select.hpp). No GPU, no model.
// usage: serve_config_test [docs/serve_config.md]   (with the doc, every ```json example in it must parse)
#undef NDEBUG
#include "ie/serve_config.hpp"
#include "ie/card_select.hpp"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using V = std::vector<std::string>;

static std::string error_of(const std::string& text) {
    try { (void)ie::parse_serve_config(text, "/cfg"); }
    catch (const std::exception& e) { return e.what(); }
    return {};
}
static bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

int main(int argc, char** argv) {
    // ---- --cards on the command line
    assert(ie::parse_launch_options({"--cards", "1"}).cards == std::vector<uint32_t>{1});
    assert((ie::parse_launch_options({"--cards", "1,0"}).cards == std::vector<uint32_t>{0, 1}));
    assert(ie::parse_launch_options({}).cards.empty());   // default: every card, as before
    for (auto args : {V{"--cards"}, V{"--cards", ""}, V{"--cards", "0,"}, V{"--cards", ",1"}, V{"--cards", "a"},
                      V{"--cards", "1,1"}, V{"--cards", "64"}, V{"--cards", "-1"}, V{"--cards", "0 1"},
                      V{"--cards", "1", "--gpus", "2"}}) {
        bool rejected = false;
        try { (void)ie::parse_launch_options(args); } catch (const std::exception&) { rejected = true; }
        assert(rejected);
    }
    (void)ie::parse_launch_options({"--cards", "0,1", "--gpus", "2"});

    // ---- card numbering: discrete GPUs in PCI order; the integrated GPU is never a card
    {
        std::vector<ie::LevelZeroGpu> gpus = {{0, "0000:00:02.0", "Intel(R) Graphics", true},
                                              {1, "0000:09:00.0", "B70", false},
                                              {2, "0000:04:00.0", "B70", false}};
        const auto cards = ie::cards_of(gpus);
        assert(cards.size() == 2 && cards[0].pci == "0000:04:00.0" && cards[1].pci == "0000:09:00.0");
        std::string err;
        std::vector<std::string> pcis;
        assert(ie::card_selector_for(gpus, {1}, err, &pcis) == "level_zero:1" && pcis == V{"0000:09:00.0"});
        assert(ie::card_selector_for(gpus, {0}, err) == "level_zero:2");
        assert(ie::card_selector_for(gpus, {0, 1}, err) == "level_zero:2,1");
        assert(ie::card_selector_for(gpus, {2}, err).empty() && has(err, "2 card(s)"));
        err.clear();
        assert(ie::card_selector_for({{0, "0000:00:02.0", "iGPU", true}}, {0}, err).empty() && has(err, "no discrete"));
    }

    // ---- a single-server layout: keys become flags, the same parser validates them
    {
        const auto cfg = ie::parse_serve_config(R"({"version": 1, "servers": [{
            "name": "mimo", "model": "models/MiMo", "cards": [1, 0], "port": 11460, "host": "127.0.0.1",
            "ctx": 250000, "gpus": 2, "parallel": 1, "max_queue": 16, "slot_ctx": 0, "prefill_chunk": 512,
            "threads": 12, "prompt_cache": false, "vram_reserve_gib": 2.5, "int8_kv": false, "spec": false,
            "spec_k": 3, "temp": 0.6, "top_k": 20, "top_p": 0.9, "min_p": 0.05, "repeat_penalty": 1.1,
            "repeat_last_n": 128, "presence_penalty": 0.5, "frequency_penalty": -0.25, "seed": 18446744073709551615,
            "max_tokens": 0, "stop": ["</s>", "END"], "thinking": false, "reasoning_effort": "high",
            "env": {"IE_MIMO26_DFLASH": 7, "IE_DS41_LOOKUP": "0", "IE_FLAG": true}}]})", "/cfg");
        const auto& s = ie::single_server(cfg);
        assert(s.name == "mimo" && s.model == "/cfg/models/MiMo");
        const auto& l = s.launch;
        assert((l.cards == std::vector<uint32_t>{0, 1}) && l.port == 11460 && l.max_queue == 16);
        assert(l.engine.max_ctx == 250000 && l.engine.n_gpus == 2 && l.engine.prefill_chunk == 512);
        assert(l.engine.cpu_threads == 12 && !l.engine.prompt_cache && !l.engine.int8_kv && !l.engine.spec);
        assert(l.engine.spec_k == 3 && l.vram_reserve_gib == 2.5);
        const auto& sp = l.defaults.sampling;
        assert(sp.temperature == 0.6f && sp.top_k == 20 && sp.top_p == 0.9f && sp.min_p == 0.05f);
        assert(sp.repeat_penalty == 1.1f && sp.repeat_window == 128 && sp.presence_penalty == 0.5f);
        assert(sp.frequency_penalty == -0.25f && sp.seed == UINT64_MAX && sp.max_tokens == ie::kMaxTokensUnlimited);
        assert((l.defaults.stop == V{"</s>", "END"}) && !l.defaults.enable_thinking && l.defaults.reasoning_effort == "high");
        // (the JSON object's keys come back sorted)
        assert(s.env.size() == 3 && s.env[0].first == "IE_DS41_LOOKUP" && s.env[0].second == "0");
        assert(s.env[1].first == "IE_FLAG" && s.env[1].second == "1");
        assert(s.env[2].first == "IE_MIMO26_DFLASH" && s.env[2].second == "7");

        // command-line flags override the file; command-line --stop replaces the file's stops
        auto m = ie::parse_launch_options(ie::merge_launch_args(s.args, {"--ctx", "9000", "--port", "11461"}));
        assert(m.engine.max_ctx == 9000 && m.port == 11461 && (m.cards == std::vector<uint32_t>{0, 1}));
        assert((m.defaults.stop == V{"</s>", "END"}));
        // ... except that --gpus 2 (file) with --cards 1 (command line) is refused: the combination is checked
        m = ie::parse_launch_options(ie::merge_launch_args(s.args, {"--stop", "X", "--gpus", "1", "--cards", "1"}));
        assert(m.defaults.stop == V{"X"} && m.engine.n_gpus == 1 && m.cards == std::vector<uint32_t>{1});
        bool refused = false;
        try { (void)ie::parse_launch_options(ie::merge_launch_args(s.args, {"--cards", "1"})); }
        catch (const std::exception& e) { refused = has(e.what(), "exceeds"); }
        assert(refused);
        // a boolean flag on the command line does not swallow the next flag
        m = ie::parse_launch_options(ie::merge_launch_args({"--spec", "--ctx", "100"}, {"--int8-kv", "--ctx", "50"}));
        assert(m.engine.spec && m.engine.int8_kv && m.engine.max_ctx == 50);
    }

    // ---- a minimal layout keeps today's defaults
    {
        const auto cfg = ie::parse_serve_config(R"({"servers": [{"model": "/m.gguf"}]})");
        const auto& l = ie::single_server(cfg).launch;
        const auto d = ie::parse_launch_options({});
        assert(l.port == d.port && l.host == d.host && l.engine.max_ctx == d.engine.max_ctx);
        assert(l.engine.n_gpus == 0 && l.cards.empty() && l.engine.prompt_cache && l.max_queue == 8);
        assert(ie::single_server(cfg).model == "/m.gguf" && ie::single_server(cfg).args.empty());
    }
    // "~/" expands to $HOME; absolute paths stay
    {
        setenv("HOME", "/home/u", 1);
        const auto cfg = ie::parse_serve_config(R"({"servers": [{"model": "~/models/x", "spec_draft": "d.gguf"}]})", "/cfg");
        assert(ie::single_server(cfg).model == "/home/u/models/x");
        assert(ie::single_server(cfg).launch.engine.spec_draft == "/cfg/d.gguf");
    }

    // ---- validation: each names what is wrong
    assert(has(error_of("{"), "not valid JSON"));
    assert(has(error_of("[]"), "JSON object"));
    assert(has(error_of(R"({"servers": []})"), "non-empty array"));
    assert(has(error_of(R"({"server": [{}]})"), "unknown top-level key \"server\""));
    assert(has(error_of(R"({"servers": [{"name": "a"}], "default": "b"})"), "no server is named \"b\""));
    assert(has(error_of(R"({"servers": [{"name": "a"}], "default": 1})"), "\"default\" must be"));
    assert(has(error_of(R"({"servers": [{}], "port": 0})"), "front port"));
    assert(has(error_of(R"({"servers": [{}], "port": "1"})"), "front port"));
    assert(has(error_of(R"({"servers": [{}], "host": ""})"), "front address"));
    assert(has(error_of(R"({"servers": [{"restart": "always"}]})"), "\"none\" or \"on-failure\""));
    assert(has(error_of(R"({"version": 2, "servers": [{}]})"), "\"version\" must be 1"));
    assert(has(error_of(R"({"servers": [{"ctxx": 5}]})"), "unknown key servers[0].ctxx"));
    assert(has(error_of(R"({"servers": [{"ctx": "8192"}]})"), "servers[0].ctx must be an integer"));
    assert(has(error_of(R"({"servers": [{"ctx": 8192.5}]})"), "must be an integer"));
    assert(has(error_of(R"({"servers": [{"ctx": 5}]})"), "--ctx requires an integer"));
    assert(has(error_of(R"({"servers": [{"temp": "hot"}]})"), "must be a number"));
    assert(has(error_of(R"({"servers": [{"temp": 3}]})"), "--temp requires"));
    assert(has(error_of(R"({"servers": [{"spec": 1}]})"), "true or false"));
    assert(has(error_of(R"({"servers": [{"cards": []}]})"), "non-empty array of card numbers"));
    assert(has(error_of(R"({"servers": [{"cards": [-1]}]})"), "card numbers"));
    assert(has(error_of(R"({"servers": [{"cards": [0, 0]}]})"), "twice"));
    assert(has(error_of(R"({"servers": [{"cards": [1], "gpus": 2}]})"), "exceeds"));
    assert(has(error_of(R"({"servers": [{"stop": "x"}]})"), "array of strings"));
    assert(has(error_of(R"({"servers": [{"thinking": "maybe"}]})"), "--thinking requires on or off"));
    assert(has(error_of(R"({"servers": [{"name": ""}]})"), "non-empty string"));
    assert(has(error_of(R"({"servers": [{"env": {"lower": "1"}}]})"), "not an environment variable name"));
    assert(has(error_of(R"({"servers": [{"env": {"IE_X": [1]}}]})"), "string, number or boolean"));
    assert(has(error_of(R"({"servers": [{"int8_kv": true, "parallel": 2}]})"), "--int8-kv"));

    // ---- multi-server: the full shape is validated, then P1 refuses to run it
    {
        const std::string two = R"({"servers": [
            {"name": "a", "model": "/a.gguf", "cards": [0], "port": 11460},
            {"name": "b", "model": "/b.gguf", "cards": [1], "port": 11461}]})";
        const auto cfg = ie::parse_serve_config(two);
        assert(cfg.servers.size() == 2 && cfg.servers[1].launch.cards == std::vector<uint32_t>{1});
        std::string msg;
        try { (void)ie::single_server(cfg); } catch (const std::exception& e) { msg = e.what(); }
        assert(has(msg, "ie supervise --config") && has(msg, "--server NAME"));
        // --server NAME picks one (how `ie supervise` starts each child)
        assert(ie::named_server(cfg, "b").model == "/b.gguf" && ie::named_server(cfg, "b").launch.port == 11461);
        msg.clear();
        try { (void)ie::named_server(cfg, "c"); } catch (const std::exception& e) { msg = e.what(); }
        assert(has(msg, "no server of that name") && has(msg, "a, b"));
        assert(cfg.default_server.empty() && cfg.front_port == 0 && cfg.front_host.empty());
        // the supervisor's keys: top-level default / host / port, per-server restart
        const auto sup = ie::parse_serve_config(R"({"version": 1, "default": "b", "host": "0.0.0.0", "port": 11470,
            "servers": [{"name": "a", "model": "/a.gguf", "cards": [0], "port": 11460, "restart": "on-failure"},
                        {"name": "b", "model": "/m/B-Q4_K_M.gguf", "cards": [1], "port": 11461, "restart": "none"}]})");
        assert(sup.default_server == "b" && sup.front_host == "0.0.0.0" && sup.front_port == 11470);
        assert(sup.servers[0].restart_on_failure && !sup.servers[1].restart_on_failure);
        assert(sup.servers[0].args == V({"--cards", "0", "--port", "11460"}));   // restart is not a flag
        assert(ie::model_id_from_path(sup.servers[1].model) == "B-Q4_K_M");
        // port clashes compare resolved bind addresses, not strings
        assert(ie::binds_overlap("localhost", "127.0.0.1") && ie::binds_overlap("::", "127.0.0.1") &&
               ie::binds_overlap("0.0.0.0", "::1") && ie::binds_overlap("127.0.0.1", "127.0.0.1"));
        assert(!ie::binds_overlap("127.0.0.1", "127.0.0.2") && !ie::binds_overlap("::1", "127.0.0.1"));
        assert(has(error_of(R"({"servers": [{"name": "a", "model": "/a", "cards": [0], "port": 7, "host": "localhost"},
                                            {"name": "b", "model": "/b", "cards": [1], "port": 7}]})"), "both use port 7"));
        assert(has(error_of(R"({"servers": [{"name": "a", "model": "/a", "cards": [0], "port": 7, "host": "::"},
                                            {"name": "b", "model": "/b", "cards": [1], "port": 7}]})"), "both use port 7"));
        (void)ie::parse_serve_config(R"({"servers": [{"name": "a", "model": "/a", "cards": [0], "port": 7, "host": "127.0.0.2"},
                                                     {"name": "b", "model": "/b", "cards": [1], "port": 7}]})");
        assert(ie::model_id_from_path("/models/MiMo-V2.6-Flash-RL") == "MiMo-V2.6-Flash-RL");
        assert(has(error_of(R"({"servers": [{"name": "a", "model": "/a", "cards": [0], "port": 1},
                                            {"model": "/b", "cards": [1], "port": 2}]})"), "needs a \"name\""));
        assert(has(error_of(R"({"servers": [{"name": "a", "model": "/a", "cards": [0], "port": 1},
                                            {"name": "b", "model": "/b", "cards": [1]}]})"), "needs a \"port\""));
        assert(has(error_of(R"({"servers": [{"name": "a", "model": "/a", "cards": [0], "port": 1},
                                            {"name": "b", "model": "/b", "port": 2}]})"), "needs \"cards\""));
        assert(has(error_of(R"({"servers": [{"name": "a", "model": "/a", "cards": [0], "port": 1},
                                            {"name": "b", "cards": [1], "port": 2}]})"), "\"model\" is required"));
        assert(has(error_of(R"({"servers": [{"name": "a", "model": "/a", "cards": [0], "port": 1},
                                            {"name": "a", "model": "/b", "cards": [1], "port": 2}]})"), "two servers are named"));
        assert(has(error_of(R"({"servers": [{"name": "a", "model": "/a", "cards": [0], "port": 1},
                                            {"name": "b", "model": "/b", "cards": [1], "port": 1}]})"), "both use port 1"));
        // 0.0.0.0 binds every address: the same port on any other host collides at bind
        assert(has(error_of(R"({"servers": [{"name": "a", "model": "/a", "cards": [0], "port": 1, "host": "0.0.0.0"},
                                            {"name": "b", "model": "/b", "cards": [1], "port": 1}]})"), "both use port 1"));
        assert(has(error_of(R"({"servers": [{"name": "a", "model": "/a", "cards": [0], "port": 1, "host": "127.0.0.1"},
                                            {"name": "b", "model": "/b", "cards": [1], "port": 1, "host": "0.0.0.0"}]})"), "both use port 1"));
        assert(error_of(R"({"servers": [{"name": "a", "model": "/a", "cards": [0], "port": 1, "host": "127.0.0.1"},
                                        {"name": "b", "model": "/b", "cards": [1], "port": 1, "host": "192.168.1.5"}]})").empty());
        assert(has(error_of(R"({"servers": [{"name": "a", "model": "/a", "cards": [0, 1], "port": 1},
                                            {"name": "b", "model": "/b", "cards": [1], "port": 2}]})"), "both use card 1"));
        assert(has(error_of(R"({"servers": [{"name": "a", "model": "/a", "cards": [0], "port": 1, "ctx": 3},
                                            {"name": "b", "model": "/b", "cards": [1], "port": 2}]})"), "servers[0] (a): --ctx"));
    }

    // ---- env: the process environment wins over the file
    {
        unsetenv("IE_P1_TEST_A");
        setenv("IE_P1_TEST_B", "from-env", 1);
        const auto set = ie::apply_config_env({{"IE_P1_TEST_A", "file"}, {"IE_P1_TEST_B", "file"}});
        assert(set == V{"IE_P1_TEST_A"});
        assert(std::string(std::getenv("IE_P1_TEST_A")) == "file" && std::string(std::getenv("IE_P1_TEST_B")) == "from-env");
    }

    // ---- the doc's examples parse
    if (argc > 1) {
        std::ifstream f(argv[1]);
        if (!f) { std::printf("serve config: %s not present, doc examples skipped\n", argv[1]); }
        else {
            std::stringstream ss;
            ss << f.rdbuf();
            const std::string doc = ss.str();
            size_t pos = 0, n = 0;
            while ((pos = doc.find("```json\n", pos)) != std::string::npos) {
                const size_t start = pos + 8, end = doc.find("```", start);
                assert(end != std::string::npos);
                try { (void)ie::parse_serve_config(doc.substr(start, end - start)); }
                catch (const std::exception& e) { std::fprintf(stderr, "doc example %zu: %s\n", n, e.what()); return 1; }
                ++n;
                pos = end + 3;
            }
            assert(n >= 2);
            std::printf("serve config: %zu doc examples parse\n", n);
        }
    }
    std::puts("serve config layouts, overrides, validation and card numbering passed");
    return 0;
}
