`solution.hpp` below holds three string helpers used by a log parser. Bug reports: tokens returned by `split` sometimes contain garbage or crash the program later (AddressSanitizer reports heap-use-after-free); `split("a,b,", ',')` loses the last, empty field; `split_ws("x  ")` returns an empty token at the end; and `join` leaves a separator at the end of its result.

Fix it. The intended behaviour:

- `split(s, delim)` returns every field between delimiters, including empty ones: `"a,,b"` gives `["a", "", "b"]`, `"a,"` gives `["a", ""]`, `","` gives `["", ""]`, and `""` gives `[""]`. The returned views must point into `s`'s own characters (not into a copy).
- `split_ws(s)` splits on runs of whitespace (space, `\t`, `\n`, `\v`, `\f`, `\r`) and never returns an empty token: `"  a  b\n"` gives `["a", "b"]`, and `""` or all-whitespace gives `[]`. Its views also point into `s`.
- `join(parts, sep)` puts `sep` between consecutive parts only: `join({"a","b"}, ", ")` is `"a, b"`, and joining no parts gives `""`.

Keep the function names and signatures. The code is compiled with AddressSanitizer.

```cpp
#pragma once
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

// Split on a single-character delimiter. The returned views point into the input.
inline std::vector<std::string_view> split(std::string_view s, char delim) {
    std::string copy(s);
    std::vector<std::string_view> out;
    size_t start = 0;
    for (size_t i = 0; i < copy.size(); ++i) {
        if (copy[i] == delim) {
            out.emplace_back(copy.data() + start, i - start);
            start = i + 1;
        }
    }
    if (start < copy.size()) out.emplace_back(copy.data() + start, copy.size() - start);
    return out;
}

// Split on runs of whitespace; never returns empty tokens.
inline std::vector<std::string_view> split_ws(std::string_view s) {
    std::vector<std::string_view> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        size_t j = i;
        while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
        out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

// Join with a separator between parts.
inline std::string join(const std::vector<std::string_view>& parts, std::string_view sep) {
    std::string r;
    for (size_t i = 0; i < parts.size(); ++i) {
        r += parts[i];
        r += sep;
    }
    return r;
}
```
