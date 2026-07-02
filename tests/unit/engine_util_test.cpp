#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/engine.hpp"
#include <cassert>
#include <cstdio>
#include <string>
int main() {
    using ie::utf8_complete_prefix_len;
    assert(utf8_complete_prefix_len("abc") == 3);
    assert(utf8_complete_prefix_len("") == 0);
    std::string e = "\xE2\x82\xAC";              // "€" complete
    assert(utf8_complete_prefix_len(e) == 3);
    std::string partial = "ab" + e.substr(0, 2); // split 3-byte char
    assert(utf8_complete_prefix_len(partial) == 2);
    std::string emoji4 = "\xF0\x9F\x98\x80";     // 4-byte emoji
    assert(utf8_complete_prefix_len("x" + emoji4.substr(0, 3)) == 1);
    std::puts("engine_util_test: all OK");
    return 0;
}
