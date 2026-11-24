#include "solution.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "check failed line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)
using quiz_SV_ = std::vector<std::string_view>;
static std::vector<std::string> S(const quiz_SV_& v) { return {v.begin(), v.end()}; }
using quiz_VS_ = std::vector<std::string>;
static bool inside(const std::string& base, const quiz_SV_& v) {
    for (auto x : v) if (x.data() < base.data() || x.data() + x.size() > base.data() + base.size()) return false;
    return true;
}
int main(int argc, char** argv) {
    CHECK((S(split("a,,b", ',')) == quiz_VS_{"a", "", "b"})); CHECK((S(split("a,", ',')) == quiz_VS_{"a", ""}));
    CHECK((S(split(",", ',')) == quiz_VS_{"", ""})); CHECK((S(split("", ',')) == quiz_VS_{""})); CHECK((S(split("abc", ',')) == quiz_VS_{"abc"}));
    CHECK((S(split("a;b;c", ';')) == quiz_VS_{"a", "b", "c"}));
    std::string line(200, 'x'); line += ",second field that is long enough to live on the heap,"; line += std::string(100, 'y');
    quiz_SV_ parts = split(line, ',');
    CHECK(parts.size() == 3 && inside(line, parts));
    CHECK(parts[1] == "second field that is long enough to live on the heap" && parts[2] == std::string(100, 'y'));
    CHECK((S(split_ws("  a  b\n")) == quiz_VS_{"a", "b"})); CHECK(split_ws("").empty()); CHECK(split_ws(" \t\n\v\f\r ").empty());
    CHECK((S(split_ws("x  ")) == quiz_VS_{"x"})); CHECK((S(split_ws("one")) == quiz_VS_{"one"})); CHECK((S(split_ws("\xc3\xa9 t")) == quiz_VS_{"\xc3\xa9", "t"}));
    std::string text = "   alpha\tbeta\r\ngamma   " + std::string(300, 'z') + "  ";
    quiz_SV_ w = split_ws(text); CHECK(w.size() == 4 && inside(text, w) && w[3].size() == 300);
    CHECK(join({"a", "b"}, ", ") == "a, b"); CHECK(join({}, ",") == ""); CHECK(join({"solo"}, "--") == "solo");
    CHECK(join({"", ""}, ",") == ",");
    { std::string_view nul("a\0,b", 4); auto f = split(nul, ','); CHECK(f.size() == 2 && f[0] == std::string_view("a\0", 2) && f[1] == "b"); } CHECK(join(split(line, ','), ",") == line);
    std::printf("%s\n", argc > 1 ? argv[1] : "");
}
