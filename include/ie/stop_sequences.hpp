#pragma once
#include <algorithm>
#include <span>
#include <string>
#include <string_view>
namespace ie {
// A later entry can match earlier in the text. Stops inside a tool block are
// ignored by passing its opening offset as the exclusive upper bound.
inline size_t first_stop_match(std::string_view text, std::span<const std::string> stops,
                              size_t begin=0, size_t before=std::string_view::npos) {
    size_t result=std::string_view::npos;
    for(const auto& s:stops) if(!s.empty()) {
        auto pos=text.find(s,begin);
        if(pos<before) result=std::min(result,pos);
    }
    return result;
}
}
