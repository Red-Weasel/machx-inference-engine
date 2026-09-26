// tests/unit/ds41_prefix_key_test.cpp -- #48 (docs/deepseek41/103): the V4.1 disk prompt cache's key. CPU only, no SYCL:
//   g++ -std=c++20 -O1 -Wall -Wextra -I include tests/unit/ds41_prefix_key_test.cpp src/model/deepseek41_prefix_key.cpp
//   ./a.out <repository root> [<input list to check instead of src/model/deepseek41_numerics_inputs.txt>]
//           [--objects <ie_core's object files>...]
// Part 1, the derivation: FNV-1a 64 against its published vectors; the same inputs give the same key; a changed flag,
// file hash, compiler line, model, format or runtime gives another. Part 2, the input list (needs the root): every
// listed file exists, nothing is listed twice, and every header a listed file includes is listed too -- so a new #include
// in a numerics source cannot slip past the key. Part 3, the calls (needs the objects; ctest passes ie_core's): with `nm`
// over them, every function a listed source calls that is DEFINED in another source must be defined in a listed one or in
// one the list names on an `unlisted:` comment line (with its reason) -- so a new call into an unlisted file cannot slip
// past the key either. A symbol some listed object defines counts as keyed (an inline or template a listed header emits).
#include "ie/deepseek41_prefix_key.hpp"

#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <filesystem>
#include <map>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

// A manifest in the build script's shape (src/model/deepseek41_numerics.cmake).
std::string manifest(const std::string& compiler, const std::vector<std::string>& flags, const std::vector<std::pair<std::string, std::string>>& files) {
    std::string m = "ds41-numerics-manifest 1\ncompiler " + compiler + "\n";
    for (const auto& f : flags) m += "flag " + f + "\n";
    for (const auto& [p, h] : files) m += "file " + p + " " + h + "\n";
    return m;
}

void part1() {
    // FNV-1a 64: the published test vectors (Fowler/Noll/Vo reference implementation)
    check(ie::ds41_fnv1a64("") == 0xcbf29ce484222325ull, "fnv1a64(\"\") == cbf29ce484222325");
    check(ie::ds41_fnv1a64("a") == 0xaf63dc4c8601ec8cull, "fnv1a64(\"a\") == af63dc4c8601ec8c");
    check(ie::ds41_fnv1a64("foobar") == 0x85944171f73967e8ull, "fnv1a64(\"foobar\") == 85944171f73967e8");
    check(ie::ds41_numerics_fingerprint("") == "cbf29ce484222325", "the fingerprint is 16 lowercase hex digits");

    const std::string cc = "Intel(R) oneAPI DPC++/C++ Compiler 2026.0.0 (2026.0.0.20251025)";
    const std::vector<std::string> fl = {"build_type=Release", "cxx_flags_build_type=-O3 -DNDEBUG", "sycl_target=spir64",
                                         "sycl_device_hint=bmg_g31", "onednn=ON"};
    const std::vector<std::pair<std::string, std::string>> fs = {
        {"src/model/deepseek41_forward.cpp", "0f3c0e4d2b1a99887766554433221100aabbccddeeff00112233445566778899"},
        {"src/ops/deepseek4_attn.cpp", "1111111111111111111111111111111111111111111111111111111111111111"}};
    const std::string model = "L43 H4096 HD512 IHD128 W128 R0s0s2s4 E123456789";
    const std::string rt = "oneDNN 3.9.1 abcdef0 | Intel(R) Arc(TM) Pro B70 Graphics driver 1.6.33578+38; Intel(R) Arc(TM) Pro B70 Graphics driver 1.6.33578+38";
    const std::string m0 = manifest(cc, fl, fs);
    const std::string k0 = ie::ds41_prefix_disk_key(model, ie::kDs41PrefixDiskFormat, m0, rt);

    check(k0 == ie::ds41_prefix_disk_key(model, ie::kDs41PrefixDiskFormat, manifest(cc, fl, fs), rt), "same inputs -> the same key");
    check(k0.rfind(model, 0) == 0, "the key starts with the model's identity");
    check(k0.find("| format " + std::to_string(ie::kDs41PrefixDiskFormat) + " |") != std::string::npos, "the key carries the format version");
    check(k0.find("| numerics " + ie::ds41_numerics_fingerprint(m0) + " |") != std::string::npos, "the key carries the manifest's fingerprint");
    check(k0.find(rt) != std::string::npos, "the key carries the runtime verbatim");

    auto differs = [&](const std::string& k, const std::string& what) { check(k != k0, what + " -> another key"); };
    { auto f = fl; f[0] = "build_type=RelWithDebInfo"; differs(ie::ds41_prefix_disk_key(model, ie::kDs41PrefixDiskFormat, manifest(cc, f, fs), rt), "a changed build type"); }
    { auto f = fl; f[3] = "sycl_device_hint=bmg_g21"; differs(ie::ds41_prefix_disk_key(model, ie::kDs41PrefixDiskFormat, manifest(cc, f, fs), rt), "a changed device hint"); }
    { auto f = fl; f[4] = "onednn=OFF"; differs(ie::ds41_prefix_disk_key(model, ie::kDs41PrefixDiskFormat, manifest(cc, f, fs), rt), "oneDNN switched off"); }
    { auto f = fl; f.push_back("source_options src/ops/cpu_moe_mxfp4.cpp=-mavx2 -mfma -fopenmp"); differs(ie::ds41_prefix_disk_key(model, ie::kDs41PrefixDiskFormat, manifest(cc, f, fs), rt), "an added per-source option"); }
    { auto s = fs; s[1].second[63] = '2'; differs(ie::ds41_prefix_disk_key(model, ie::kDs41PrefixDiskFormat, manifest(cc, fl, s), rt), "one changed hex digit of one file's SHA-256"); }
    { auto s = fs; s.push_back({"src/ops/new_kernel.cpp", std::string(64, 'a')}); differs(ie::ds41_prefix_disk_key(model, ie::kDs41PrefixDiskFormat, manifest(cc, fl, s), rt), "a file added to the list"); }
    differs(ie::ds41_prefix_disk_key(model, ie::kDs41PrefixDiskFormat, manifest("Intel(R) oneAPI DPC++/C++ Compiler 2026.1.0 (2026.1.0.20260301)", fl, fs), rt), "another compiler");
    differs(ie::ds41_prefix_disk_key(model + "0", ie::kDs41PrefixDiskFormat, m0, rt), "another model");
    differs(ie::ds41_prefix_disk_key(model, ie::kDs41PrefixDiskFormat + 1, m0, rt), "another entry format");
    differs(ie::ds41_prefix_disk_key(model, ie::kDs41PrefixDiskFormat, m0, "oneDNN 3.10.0 0000000 | Intel(R) Arc(TM) Pro B70 Graphics driver 1.6.33578+38; Intel(R) Arc(TM) Pro B70 Graphics driver 1.6.33578+38"), "another oneDNN");
    differs(ie::ds41_prefix_disk_key(model, ie::kDs41PrefixDiskFormat, m0, "oneDNN 3.9.1 abcdef0 | Intel(R) Arc(TM) Pro B70 Graphics driver 1.6.34000+1; Intel(R) Arc(TM) Pro B70 Graphics driver 1.6.34000+1"), "another GPU driver");
    // the key is a pure function of its four inputs: nothing of the executable (its path, size or mtime) enters it
    check(k0.find("/proc") == std::string::npos && k0.find(" X") == std::string::npos, "no executable identity in the key");
}

// ---- part 2: the input list -----------------------------------------------------------------------------------------------
std::vector<std::string> read_list(const std::filesystem::path& p) {
    std::vector<std::string> out;
    std::ifstream f(p);
    for (std::string line; std::getline(f, line);) {
        if (const auto h = line.find('#'); h != std::string::npos) line.erase(h);
        const auto a = line.find_first_not_of(" \t\r"), b = line.find_last_not_of(" \t\r");
        if (a == std::string::npos) continue;
        out.push_back(line.substr(a, b - a + 1));
    }
    return out;
}

// the "..." includes of a file, resolved the way the build resolves them: beside the file, then include/, then third_party/
std::vector<std::pair<std::string, std::string>> quoted_includes(const std::filesystem::path& root, const std::string& rel) {
    std::vector<std::pair<std::string, std::string>> out;   // (as written, resolved relative to root or "")
    std::ifstream f(root / rel);
    for (std::string line; std::getline(f, line);) {
        const auto s = line.find_first_not_of(" \t");
        if (s == std::string::npos || line[s] != '#') continue;
        std::istringstream is(line.substr(s + 1)); std::string kw; is >> kw;
        if (kw != "include") continue;
        const auto q0 = line.find('"'), q1 = q0 == std::string::npos ? q0 : line.find('"', q0 + 1);
        if (q1 == std::string::npos) continue;                  // <...>: a system header
        const std::string name = line.substr(q0 + 1, q1 - q0 - 1);
        std::string hit;
        for (const auto& base : {std::filesystem::path(rel).parent_path(), std::filesystem::path("include"), std::filesystem::path("third_party")}) {
            const auto cand = (base / name).lexically_normal();
            if (std::filesystem::is_regular_file(root / cand)) { hit = cand.generic_string(); break; }
        }
        out.push_back({name, hit});
    }
    return out;
}

void part2(const std::filesystem::path& root, const std::filesystem::path& list_path) {
    const auto list = read_list(list_path);
    check(list.size() >= 20, "the input list has " + std::to_string(list.size()) + " entries (" + list_path.string() + ")");
    std::set<std::string> listed;
    bool dup = false, bad_char = false, missing = false;
    for (const auto& p : list) {
        if (!listed.insert(p).second) { dup = true; std::printf("       listed twice: %s\n", p.c_str()); }
        if (p.find_first_of(";[]") != std::string::npos) { bad_char = true; std::printf("       a path CMake's list syntax would split: %s\n", p.c_str()); }
        if (!std::filesystem::is_regular_file(root / p)) { missing = true; std::printf("       listed but absent: %s\n", p.c_str()); }
    }
    check(!dup, "nothing is listed twice");
    check(!bad_char, "no listed path holds ';', '[' or ']'");
    check(!missing, "every listed file exists");
    // the closure: every header a listed file includes is listed (the generated manifest header is the one exception)
    const std::set<std::string> generated = {"deepseek41_numerics_manifest.h"};
    size_t n_inc = 0; bool closed = true;
    for (const auto& p : list)
        for (const auto& [name, hit] : quoted_includes(root, p)) {
            ++n_inc;
            if (hit.empty()) {
                if (!generated.count(name)) { closed = false; std::printf("       %s includes \"%s\", which resolves to no file in the tree\n", p.c_str(), name.c_str()); }
            } else if (!listed.count(hit)) {
                closed = false; std::printf("       %s includes %s, which is not listed\n", p.c_str(), hit.c_str());
            }
        }
    check(closed, "the list is closed under #include (" + std::to_string(n_inc) + " quoted includes checked)");
    for (const char* must : {"src/model/deepseek41_forward.cpp", "src/model/deepseek41_prefix_cache.cpp", "src/model/deepseek41_generate.cpp",
                             "src/model/deepseek41_experts.cpp", "src/ops/deepseek4_attn.cpp", "src/ops/gemm_onednn.cpp"})
        check(listed.count(must) == 1, std::string("listed: ") + must);
}

// the `unlisted: <path>` comment lines of the input list: the sources it names as reachable but not keyed
std::set<std::string> read_unlisted(const std::filesystem::path& p) {
    std::set<std::string> out;
    std::ifstream f(p);
    for (std::string line; std::getline(f, line);) {
        const auto h = line.find('#'), u = line.find("unlisted:");
        if (h == std::string::npos || u == std::string::npos || u < h) continue;
        std::istringstream is(line.substr(u + 9)); std::string path;
        if (is >> path) out.insert(path);
    }
    return out;
}

// `nm <flags> <object>`: (type letter, mangled name) per symbol line; `ok` false when nm could not run or failed
std::vector<std::pair<char, std::string>> nm(const char* flags, const std::string& obj, bool& ok) {
    std::string quoted = "'";
    for (char c : obj) { if (c == '\'') quoted += "'\\''"; else quoted += c; }
    quoted += "'";
    std::vector<std::pair<char, std::string>> out;
    FILE* p = popen(("nm " + std::string(flags) + " " + quoted + " 2>/dev/null").c_str(), "r");
    if (!p) { ok = false; return out; }
    char buf[8192];
    while (std::fgets(buf, sizeof buf, p)) {
        std::istringstream is(buf); std::vector<std::string> w; for (std::string t; is >> t;) w.push_back(t);
        if (w.size() == 2 && w[0].size() == 1) out.push_back({w[0][0], w[1]});                     // "U name"
        else if (w.size() >= 3 && w[1].size() == 1) out.push_back({w[1][0], w[2]});                // "addr T name"
    }
    ok = pclose(p) == 0 && ok;
    return out;
}

std::string demangle(const std::string& s) {
    int st = 0; char* d = abi::__cxa_demangle(s.c_str(), nullptr, nullptr, &st);
    std::string r = st == 0 && d ? d : s; std::free(d);
    return r.size() > 160 ? r.substr(0, 160) + "..." : r;
}

// an object of ie_core (<build>/src/CMakeFiles/ie_core.dir/<dir>/<name>.cpp.o) -> its source relative to the root
std::string source_of(const std::filesystem::path& root, const std::string& obj) {
    const auto d = obj.rfind(".dir/");
    if (d == std::string::npos || obj.size() < d + 8 || obj.compare(obj.size() - 2, 2, ".o") != 0) return "";
    const std::string rest = obj.substr(d + 5, obj.size() - 2 - (d + 5));   // CMake may write "ie_core.dir/./core/..."
    for (const std::string& cand : {"src/" + rest, rest}) {
        const std::string norm = std::filesystem::path(cand).lexically_normal().generic_string();
        if (std::filesystem::is_regular_file(root / norm)) return norm;
    }
    return "";
}

void part3(const std::filesystem::path& root, const std::filesystem::path& list_path, const std::vector<std::string>& objects) {
    const auto list = read_list(list_path);
    const std::set<std::string> listed(list.begin(), list.end());
    const auto unlisted = read_unlisted(list_path);
    bool clean = true, nm_ok = true, mapped = true, have_all = true;
    for (const auto& u : unlisted) if (listed.count(u)) { clean = false; std::printf("       %s is both listed and named unlisted\n", u.c_str()); }
    check(clean, "no source is both listed and named unlisted (" + std::to_string(unlisted.size()) + " named)");
    // who defines what: every global symbol each source's object defines, strong or weak
    std::map<std::string, std::string> obj_of;                  // source -> object
    std::map<std::string, std::set<std::string>> defined_in;    // symbol -> sources
    for (const auto& o : objects) {
        const std::string src = source_of(root, o);
        if (src.empty()) { mapped = false; std::printf("       no source found for object %s\n", o.c_str()); continue; }
        obj_of[src] = o;
        for (const auto& [t, sym] : nm("--defined-only", o, nm_ok))
            if (std::string("TDBRWVGS").find(t) != std::string::npos) defined_in[sym].insert(src);
    }
    check(mapped, "every object maps to a source (" + std::to_string(objects.size()) + " objects)");
    for (const auto& p : list)
        if (p.size() > 4 && p.compare(p.size() - 4, 4, ".cpp") == 0 && !obj_of.count(p)) { have_all = false; std::printf("       no object for listed %s\n", p.c_str()); }
    check(have_all, "every listed source has an object");
    // the calls: a listed object's undefined symbol that no listed object defines, but an unnamed source does
    std::map<std::string, std::set<std::string>> gaps;          // "definer <- listed caller" -> symbols
    size_t n_calls = 0;
    for (const auto& p : list) {
        const auto it = obj_of.find(p);
        if (it == obj_of.end()) continue;
        for (const auto& [t, sym] : nm("--undefined-only", it->second, nm_ok)) {
            const auto d = defined_in.find(sym);
            if (d == defined_in.end()) continue;                // outside the project: the SYCL runtime, oneDNN, libc
            ++n_calls;
            bool keyed = false;
            for (const auto& s : d->second) keyed = keyed || listed.count(s);
            if (keyed) continue;
            for (const auto& s : d->second)
                if (!unlisted.count(s)) gaps[s + " <- " + p].insert(sym);
        }
    }
    check(nm_ok, "nm read every object");
    for (const auto& [where, syms] : gaps) {
        std::printf("       %s:\n", where.c_str());
        for (const auto& s : syms) std::printf("         %s\n", demangle(s).c_str());
    }
    check(gaps.empty(), "every project function a listed source calls is defined in a listed or a named-unlisted source (" +
                        std::to_string(n_calls) + " calls checked)");
}

}  // namespace

int main(int argc, char** argv) {
    part1();
    std::vector<std::string> objects;
    int n_pos = argc;   // the positional arguments end at --objects
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--objects") { n_pos = i; for (int j = i + 1; j < argc; ++j) objects.push_back(argv[j]); break; }
    if (n_pos >= 2) {
        const std::filesystem::path root = argv[1];
        const std::filesystem::path list = n_pos >= 3 ? std::filesystem::path(argv[2]) : root / "src/model/deepseek41_numerics_inputs.txt";
        part2(root, list);
        if (!objects.empty()) part3(root, list, objects);
        else std::printf("[skip] the call check needs ie_core's objects after --objects\n");
    } else std::printf("[skip] the input list checks need the repository root as the first argument\n");
    std::printf("\nDS41 PREFIX KEY: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
