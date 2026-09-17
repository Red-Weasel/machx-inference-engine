// src/core/tensor.cpp

#include "ie/tensor.hpp"

#include <cstdio>

namespace ie {

std::span<char> Tensor::format_shape(std::span<char> buf) const noexcept {
    char* const beg = buf.data();
    char* p = beg;
    char* const end = beg + buf.size();
    auto put = [&](const char* fmt, auto v) {
        if (p >= end) return;
        int n = std::snprintf(p, size_t(end - p), fmt, v);
        if (n > 0) p += n;
    };
    put("%s", "[");
    for (uint32_t i = 0; i < n_dims; ++i) {
        if (i) put("%s", ", ");
        put("%llu", static_cast<unsigned long long>(shape[i]));
    }
    put("%s", "]");
    return std::span<char>(beg, size_t(p - beg));
}

}  // namespace ie
