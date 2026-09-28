// Host-only: `ie` help coverage. The flag list is DERIVED from the parser's source (include/ie/serve_options.hpp), not
// hand-kept: every flag parse_launch_options accepts must appear in the run and serve help, must have a layout key
// (ie/serve_config.hpp), and launch_flag_is_boolean must agree with whether the parser reads a value for it. Every
// command main.cpp dispatches must have help.
// usage: cli_help_test <include/ie/serve_options.hpp> <src/cli/main.cpp>
#undef NDEBUG
#include "ie/cli_help.hpp"
#include "ie/serve_config.hpp"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>

static std::string read_file(const char* p) {
    std::ifstream f(p);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", p); std::exit(1); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static bool names_flag(const std::string& text, const std::string& flag) {   // a whole token: --spec is not --spec-k
    const std::regex re("(^|[^a-z0-9-])" + flag + "([^a-z0-9-]|$)");
    return std::regex_search(text, re);
}

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: cli_help_test <serve_options.hpp> <main.cpp>\n"); return 2; }
    const std::string parser = read_file(argv[1]);
    const auto body_at = parser.find("inline LaunchOptions parse_launch_options");
    assert(body_at != std::string::npos);
    const std::string body = parser.substr(body_at);

    // The flags, and for each the parser text up to the next branch (does it read a value?).
    const std::regex flag_re("flag == \"(--[a-z0-9-]+)\"");
    std::vector<std::pair<std::string, size_t>> found;
    for (auto it = std::sregex_iterator(body.begin(), body.end(), flag_re); it != std::sregex_iterator(); ++it)
        found.emplace_back((*it)[1].str(), size_t(it->position(0)));
    assert(found.size() >= 30);   // 30 today (with --cards); a regex that stops matching must not pass vacuously
    const auto end_of_chain = body.find("else throw std::runtime_error(\"unknown option");
    assert(end_of_chain != std::string::npos);

    std::set<std::string> flags;
    const std::string serve = ie::cli_help("serve"), run = ie::cli_help("run");
    int missing = 0;
    for (size_t i = 0; i < found.size(); ++i) {
        const auto& [flag, pos] = found[i];
        flags.insert(flag);
        const size_t next = i + 1 < found.size() ? found[i + 1].second : end_of_chain;
        const std::string branch = body.substr(pos, next - pos);
        const bool reads_value = branch.find("value()") != std::string::npos ||
                                 branch.find("integer(") != std::string::npos ||
                                 branch.find("number(") != std::string::npos;
        if (ie::launch_flag_is_boolean(flag) == reads_value) {
            std::fprintf(stderr, "launch_flag_is_boolean(%s) disagrees with the parser\n", flag.c_str()); ++missing;
        }
        for (auto [name, text] : {std::pair{"serve", &serve}, {"run", &run}})
            if (!names_flag(*text, flag)) { std::fprintf(stderr, "ie %s --help does not name %s\n", name, flag.c_str()); ++missing; }
    }
    // --config is main.cpp's, not the parser's
    assert(names_flag(serve, "--config") && names_flag(run, "--config"));
    // main.cpp takes <model> only as the first argument after the command: the synopsis must say so
    for (const std::string* text : {&serve, &run}) {
        const std::regex before(R"(ie (serve|run) \[<model>\] --config <layout\.json>)");
        const std::regex after(R"(--config <[^>]*>[^\n]*<model>)");
        assert(std::regex_search(*text, before));
        assert(!std::regex_search(*text, after));
    }

    // The layout keys cover exactly the parser's flags.
    std::set<std::string> keyed;
    for (const auto& k : ie::serve_config_keys()) keyed.insert(k.flag);
    for (const auto& f : flags) if (!keyed.count(f)) { std::fprintf(stderr, "no layout key for %s\n", f.c_str()); ++missing; }
    for (const auto& f : keyed) if (!flags.count(f)) { std::fprintf(stderr, "layout key for unknown flag %s\n", f.c_str()); ++missing; }

    // Every command main.cpp dispatches has help.
    const std::string main_src = read_file(argv[2]);
    const std::regex cmd_re("cmd [!=]= \"([a-z-]+)\"");
    std::set<std::string> cmds;
    for (auto it = std::sregex_iterator(main_src.begin(), main_src.end(), cmd_re); it != std::sregex_iterator(); ++it)
        cmds.insert((*it)[1].str());
    assert(cmds.size() >= 8);
    for (const auto& c : cmds) {
        if (c.rfind("-", 0) == 0) continue;   // --help / -h
        if (ie::cli_help(c).empty()) { std::fprintf(stderr, "no help for command %s\n", c.c_str()); ++missing; }
    }
    assert(ie::cli_help("nope").empty());
    assert(std::string(ie::cli_help_overview()).find("serve") != std::string::npos);

    // cli_wants_help: --help / -h anywhere, but not as a flag's value.
    using V = std::vector<std::string>;
    assert(ie::cli_wants_help(V{"--help"}));
    assert(ie::cli_wants_help(V{"-h"}));
    assert(ie::cli_wants_help(V{"model.gguf", "--ctx", "9", "--help"}));
    assert(ie::cli_wants_help(V{"model.gguf", "--spec", "--help"}));
    assert(!ie::cli_wants_help(V{"model.gguf", "--stop", "--help"}));
    assert(!ie::cli_wants_help(V{"model.gguf", "--ctx", "9"}));
    assert(!ie::cli_wants_help(V{}));

    if (missing) { std::fprintf(stderr, "%d help/parser mismatches\n", missing); return 1; }
    std::printf("help names all %zu parser flags; layout keys match; %zu commands have help\n", flags.size(), cmds.size());
    return 0;
}
