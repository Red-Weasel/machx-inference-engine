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
