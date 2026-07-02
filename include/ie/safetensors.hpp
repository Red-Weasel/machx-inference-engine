// include/ie/safetensors.hpp — mmap reader for HuggingFace `.safetensors`.
//
// Format (huggingface/safetensors spec):
//   [0 .. 8)      u64 LE  N        ← byte length of the JSON header
//   [8 .. 8+N)    UTF-8 JSON       ← { "<name>": {dtype, shape, data_offsets:[b,e]}, ... ,
//                                       "__metadata__": { ... } }
//   [8+N .. EOF)  raw tensor bytes ← row-major (C order), contiguous; data_offsets
//                                     are relative to THIS data region.
//
// This is the ingestion entry point for AWQ/GPTQ checkpoints (P3e) — a format
// llama.cpp cannot load natively. The reader is dependency-light (mmap +
// nlohmann/json header) and hands out non-owning pointers into the mmap.
//
// ⚠ Alignment: the spec does NOT guarantee per-tensor 4-byte alignment within
// the data region. Read int32 tensors (AWQ qweight/qzeros) via std::memcpy or a
// byte-wise accessor, NOT a raw reinterpret_cast<const int32_t*>.
#pragma once

#include "ie/dtype.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ie {

struct SafeTensorInfo {
    std::string          name;
    DType                dtype = DType::kCount;  // kCount if the dtype string is unmapped
    std::string          dtype_str;              // raw, e.g. "F16","BF16","I32"
    std::vector<int64_t> shape;
    const uint8_t*       data   = nullptr;       // points into the mmap
    uint64_t             nbytes = 0;

    uint64_t numel() const {
        uint64_t n = 1;
        for (auto d : shape) n *= (d < 0 ? 0 : uint64_t(d));
        return shape.empty() ? 0 : n;
    }
};

// Single-file safetensors reader. (Sharded checkpoints — `model.safetensors.
// index.json` + `model-0000k-of-N.safetensors` — are handled by opening one
// reader per shard and merging name→tensor; see P3e Task 2.)
class SafetensorsReader {
public:
    SafetensorsReader() = default;
    ~SafetensorsReader();
    SafetensorsReader(const SafetensorsReader&)            = delete;
    SafetensorsReader& operator=(const SafetensorsReader&) = delete;
    SafetensorsReader(SafetensorsReader&&) noexcept;
    SafetensorsReader& operator=(SafetensorsReader&&) noexcept;

    // Returns "" on success, human-readable error text otherwise.
    std::string open(const std::string& path);
    void        close();

    const std::vector<SafeTensorInfo>& tensors() const { return tensors_; }
    const SafeTensorInfo*              find(std::string_view name) const;

    // __metadata__ string entry ("" if absent).
    std::string metadata(std::string_view key) const;

    // safetensors dtype string → engine DType (kCount if unmapped).
    static DType dtype_from_string(std::string_view s);

private:
    int      fd_   = -1;
    uint8_t* base_ = nullptr;
    size_t   size_ = 0;
    std::vector<SafeTensorInfo>                  tensors_;
    std::unordered_map<std::string, std::string> meta_;
};

// A whole HF checkpoint that may be ONE `model.safetensors` or MANY shards
// (`model.safetensors.index.json` + `model-0000k-of-N.safetensors`). Big AWQ
// models (e.g. Qwen-72B-AWQ) are always sharded. Presents a unified view.
class SafetensorsModel {
public:
    // Open a model directory. Returns "" on success, error text otherwise.
    std::string open(const std::string& dir);

    const SafeTensorInfo*               find(std::string_view name) const;
    std::vector<const SafeTensorInfo*>  all() const;   // every tensor across shards
    size_t                              shard_count() const { return shards_.size(); }

private:
    std::vector<std::unique_ptr<SafetensorsReader>> shards_;
};

}  // namespace ie
