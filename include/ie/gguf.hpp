// include/ie/gguf.hpp — GGUF (mmap) reader.
//
// Spec: research/03_quant_formats.md §1, derived from ggml/docs/gguf.md.
// Container layout:
//   [24-byte header]
//   [n_kv KV pairs]
//   [n_tensor tensor-info entries]
//   [pad to general.alignment]
//   [tensor data blob]
//
// All offsets in tensor-info are relative to the start of the tensor-data blob.

#pragma once

#include "ie/dtype.hpp"
#include "ie/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ie {

// 1.2 — KV value-type ids. Same values as the on-disk encoding.
enum class GgufValueType : uint32_t {
    kU8     = 0,
    kI8     = 1,
    kU16    = 2,
    kI16    = 3,
    kU32    = 4,
    kI32    = 5,
    kF32    = 6,
    kBool   = 7,
    kString = 8,
    kArray  = 9,
    kU64    = 10,
    kI64    = 11,
    kF64    = 12,
};

std::string_view name_of(GgufValueType v) noexcept;

// Lightweight non-owning reference into the mmap'd file. The memory
// belongs to the GgufReader; KvRef is invalidated when the reader closes.
struct KvRef {
    std::string_view  key;
    GgufValueType     type;
    GgufValueType     inner_type{};   // valid only if type == kArray
    uint64_t          n_array{};      // valid only if type == kArray
    const uint8_t*    payload = nullptr; // points at the value bytes (or array elements)

    // Scalar accessors. Return 0 / "" if the dynamic type doesn't match.
    bool             as_bool()   const noexcept;
    int64_t          as_int()    const noexcept;
    uint64_t         as_uint()   const noexcept;
    double           as_float()  const noexcept;
    std::string_view as_string() const noexcept;

    // Array accessors. Return empty span if type is wrong.
    std::vector<std::string_view> as_string_array() const;
    template <typename T> std::span<const T> as_pod_array() const noexcept {
        if (type != GgufValueType::kArray) return {};
        return {reinterpret_cast<const T*>(payload), n_array};
    }
};

// Tensor info entry parsed from §1.3. `data` points into the mmap.
struct GgufTensorInfo {
    std::string_view  name;
    DType             dtype;
    uint32_t          n_dims;
    std::array<uint64_t, kMaxDims> shape{};
    uint64_t          offset_in_data;  // bytes from start of tensor-data section
    uint64_t          nbytes;          // computed from dtype and shape
    const uint8_t*    data = nullptr;  // = mmap_base + tensor_data_offset + offset_in_data
};

class GgufReader {
public:
    GgufReader() = default;
    ~GgufReader();
    GgufReader(const GgufReader&) = delete;
    GgufReader& operator=(const GgufReader&) = delete;
    GgufReader(GgufReader&&) noexcept;
    GgufReader& operator=(GgufReader&&) noexcept;

    // Returns empty error string on success.
    std::string open(const std::string& path);
    void        close() noexcept;

    bool                                valid() const noexcept { return base_ != nullptr; }
    uint32_t                            version() const noexcept { return version_; }
    uint64_t                            n_tensors() const noexcept { return tensors_.size(); }
    uint64_t                            n_kv()      const noexcept { return kvs_.size(); }
    uint64_t                            alignment() const noexcept { return alignment_; }
    uint64_t                            file_size() const noexcept { return size_; }
    uint64_t                            tensor_data_offset() const noexcept { return tensor_data_off_; }

    std::span<const KvRef>              kvs()     const noexcept { return kvs_; }
    std::span<const GgufTensorInfo>     tensors() const noexcept { return tensors_; }

    const KvRef*                        find_kv(std::string_view key) const noexcept;
    const GgufTensorInfo*               find_tensor(std::string_view name) const noexcept;

    // Wrap a tensor info as an `ie::Tensor` viewing the mmap'd bytes.
    Tensor make_host_view(const GgufTensorInfo& info) const noexcept;

private:
    // Parse one GGUF shard's header into kvs_ (primary only) + tensors_ (every shard,
    // with each tensor's `data` resolved against THIS shard's mmap base). For a split
    // model (split.count>1) the metadata lives only in shard 0; secondary shards
    // contribute tensors only. Returns "" on success.
    std::string parse_shard(const uint8_t* base, size_t size, bool primary);

    // Extra mmaps for the secondary shards of a split GGUF — kept alive for the
    // reader's lifetime because tensors_/names point into them.
    struct ExtraShard { int fd = -1; uint8_t* base = nullptr; size_t size = 0; };

    int            fd_   = -1;
    uint8_t*       base_ = nullptr;
    size_t         size_ = 0;
    uint32_t       version_ = 0;
    uint64_t       alignment_ = 32;
    uint64_t       tensor_data_off_ = 0;
    std::vector<KvRef>           kvs_;
    std::vector<GgufTensorInfo>  tensors_;
    std::vector<ExtraShard>      extra_shards_;
};

}  // namespace ie
