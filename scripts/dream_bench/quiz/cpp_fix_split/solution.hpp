#pragma once
#include <string>
#include <string_view>
#include <vector>

// Split on a single-character delimiter. The returned views point into the input.
inline std::vector<std::string_view> split(std::string_view s, char delim) {
    std::vector<std::string_view> out;
    size_t start = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == delim) {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    out.push_back(s.substr(start));
    return out;
}

inline bool is_ws_(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r'; }

// Split on runs of whitespace; never returns empty tokens.
inline std::vector<std::string_view> split_ws(std::string_view s) {
    std::vector<std::string_view> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && is_ws_(s[i])) ++i;
        size_t j = i;
        while (j < s.size() && !is_ws_(s[j])) ++j;
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

// Join with a separator between parts.
inline std::string join(const std::vector<std::string_view>& parts, std::string_view sep) {
    std::string r;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) r += sep;
        r += parts[i];
    }
    return r;
}
