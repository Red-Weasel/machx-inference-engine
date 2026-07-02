// include/ie/tensor.hpp — Tensor descriptor.
//
// A Tensor is just a non-owning view by default. The concrete byte buffer is
// owned by either:
//   - a GGUF mmap (read-only, host pointer) — the loader hands these out
//   - a USM device allocation (gpu pointer) — the engine creates these
//
// Up to 4 dims (matches GGML_MAX_DIMS = 4). Strides are in bytes when the
// dtype is quantized (block-aligned), or in elements for non-quantized types
// — the loader fills these in correctly for GGUF tensors.

#pragma once

#include "ie/dtype.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace ie {

inline constexpr size_t kMaxDims = 4;

enum class Device : uint8_t {
    kHost,    // pointer is in CPU address space (incl. mmap'd files)
    kGpu,     // pointer is a device USM allocation
};

struct Tensor {
    DType                          dtype = DType::kF32;
    Device                         device = Device::kHost;
    uint32_t                       n_dims = 0;
    std::array<uint64_t, kMaxDims> shape{};        // logical shape, leading dim first
    void*                          data = nullptr; // raw bytes
    uint64_t                       nbytes = 0;     // total byte size of the buffer
    std::string_view               name{};         // borrowed (e.g. from GGUF mmap)

    bool valid() const noexcept {
        return data != nullptr && n_dims > 0 && n_dims <= kMaxDims && type_info(dtype);
    }

    uint64_t n_elements() const noexcept {
        uint64_t n = 1;
        for (uint32_t i = 0; i < n_dims; ++i) n *= shape[i];
        return n;
    }

    std::span<const uint64_t> shape_span() const noexcept {
        return {shape.data(), n_dims};
    }

    // Returns a row-major shape string like "[8, 16, 256]".
    // Caller-supplied buffer; returns the slice that was written.
    std::span<char> format_shape(std::span<char> buf) const noexcept;
};

}  // namespace ie
