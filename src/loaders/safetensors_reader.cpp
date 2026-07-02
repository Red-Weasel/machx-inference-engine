// src/loaders/safetensors_reader.cpp — mmap-based `.safetensors` reader.
// Mirrors gguf_reader.cpp's mmap/RAII + error-string conventions.
#include "ie/safetensors.hpp"

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ie {

using json = nlohmann::json;

SafetensorsReader::~SafetensorsReader() { close(); }

SafetensorsReader::SafetensorsReader(SafetensorsReader&& o) noexcept
    : fd_(o.fd_), base_(o.base_), size_(o.size_),
      tensors_(std::move(o.tensors_)), meta_(std::move(o.meta_)) {
    o.fd_ = -1; o.base_ = nullptr; o.size_ = 0;
}

SafetensorsReader& SafetensorsReader::operator=(SafetensorsReader&& o) noexcept {
    if (this != &o) {
        close();
        fd_ = o.fd_; base_ = o.base_; size_ = o.size_;
        tensors_ = std::move(o.tensors_); meta_ = std::move(o.meta_);
        o.fd_ = -1; o.base_ = nullptr; o.size_ = 0;
    }
    return *this;
}

void SafetensorsReader::close() {
    if (base_ && base_ != MAP_FAILED) ::munmap(base_, size_);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1; base_ = nullptr; size_ = 0;
    tensors_.clear(); meta_.clear();
}

DType SafetensorsReader::dtype_from_string(std::string_view s) {
    if (s == "F64")  return DType::kF64;
    if (s == "F32")  return DType::kF32;
    if (s == "F16")  return DType::kF16;
    if (s == "BF16") return DType::kBF16;
    if (s == "I64")  return DType::kI64;
    if (s == "I32")  return DType::kI32;
    if (s == "I16")  return DType::kI16;
    if (s == "I8" || s == "U8" || s == "BOOL") return DType::kI8;
    // F8_E4M3 / F8_E5M2 / U16/U32/U64 left unmapped (kCount); dtype_str kept.
    return DType::kCount;
}

const SafeTensorInfo* SafetensorsReader::find(std::string_view name) const {
    for (const auto& t : tensors_)
        if (t.name == name) return &t;
    return nullptr;
}

std::string SafetensorsReader::metadata(std::string_view key) const {
    auto it = meta_.find(std::string(key));
    return it == meta_.end() ? std::string() : it->second;
}

std::string SafetensorsReader::open(const std::string& path) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
        return std::string("open(") + path + ") failed: " + std::strerror(errno);

    struct stat st{};
    if (::fstat(fd_, &st) != 0) { close(); return std::string("fstat failed: ") + std::strerror(errno); }
    size_ = static_cast<size_t>(st.st_size);
    if (size_ < 8) { close(); return "file shorter than safetensors header length"; }

    void* m = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (m == MAP_FAILED) { close(); return std::string("mmap failed: ") + std::strerror(errno); }
    base_ = static_cast<uint8_t*>(m);
    ::madvise(base_, size_, MADV_RANDOM);

    // [0..8) header length (u64 LE). x86 is little-endian → direct memcpy.
    uint64_t header_len = 0;
    std::memcpy(&header_len, base_, 8);
    if (header_len == 0 || 8 + header_len > size_)
        { close(); return "invalid safetensors header length"; }

    const char* json_begin = reinterpret_cast<const char*>(base_) + 8;
    const uint64_t data_base   = 8 + header_len;           // start of tensor data
    const uint64_t data_region = size_ - data_base;

    json hdr;
    try {
        hdr = json::parse(json_begin, json_begin + header_len);
    } catch (const std::exception& e) {
        close();
        return std::string("safetensors header JSON parse error: ") + e.what();
    }
    if (!hdr.is_object()) { close(); return "safetensors header is not a JSON object"; }

    tensors_.reserve(hdr.size());
    for (auto it = hdr.begin(); it != hdr.end(); ++it) {
        const std::string& key = it.key();
        const json& v = it.value();

        if (key == "__metadata__") {
            if (v.is_object())
                for (auto m2 = v.begin(); m2 != v.end(); ++m2)
                    if (m2.value().is_string())
                        meta_[m2.key()] = m2.value().get<std::string>();
            continue;
        }

        if (!v.is_object() || !v.contains("dtype") || !v.contains("shape") ||
            !v.contains("data_offsets")) {
            close();
            return "tensor '" + key + "' missing dtype/shape/data_offsets";
        }

        SafeTensorInfo t;
        t.name      = key;
        t.dtype_str = v["dtype"].get<std::string>();
        t.dtype     = dtype_from_string(t.dtype_str);
        for (const auto& d : v["shape"]) t.shape.push_back(d.get<int64_t>());

        const auto& off = v["data_offsets"];
        if (!off.is_array() || off.size() != 2) {
            close();
            return "tensor '" + key + "' has malformed data_offsets";
        }
        const uint64_t b = off[0].get<uint64_t>();
        const uint64_t e = off[1].get<uint64_t>();
        if (b > e || e > data_region) {
            close();
            return "tensor '" + key + "' data_offsets out of range";
        }
        t.nbytes = e - b;
        t.data   = base_ + data_base + b;
        tensors_.push_back(std::move(t));
    }
    return {};
}

// ---- SafetensorsModel (single file or sharded directory) ----

static bool file_exists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0;
}

std::string SafetensorsModel::open(const std::string& dir) {
    shards_.clear();
    auto add_shard = [&](const std::string& path) -> std::string {
        auto r = std::make_unique<SafetensorsReader>();
        if (auto e = r->open(path); !e.empty()) return e;
        shards_.push_back(std::move(r));
        return {};
    };

    const std::string single = dir + "/model.safetensors";
    if (file_exists(single)) return add_shard(single);

    const std::string index = dir + "/model.safetensors.index.json";
    if (!file_exists(index))
        return "no model.safetensors or model.safetensors.index.json in " + dir;

    std::ifstream f(index, std::ios::binary);
    if (!f) return "cannot open " + index;
    std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    json j;
    try { j = json::parse(body); }
    catch (const std::exception& e) { return std::string("index.json parse error: ") + e.what(); }
    if (!j.contains("weight_map") || !j["weight_map"].is_object())
        return "index.json missing weight_map";

    // Unique shard filenames, opened once each (in sorted order for determinism).
    std::vector<std::string> files;
    for (auto it = j["weight_map"].begin(); it != j["weight_map"].end(); ++it) {
        const std::string fn = it.value().get<std::string>();
        if (std::find(files.begin(), files.end(), fn) == files.end()) files.push_back(fn);
    }
    std::sort(files.begin(), files.end());
    for (const auto& fn : files)
        if (auto e = add_shard(dir + "/" + fn); !e.empty()) return "shard " + fn + ": " + e;
    return {};
}

const SafeTensorInfo* SafetensorsModel::find(std::string_view name) const {
    for (const auto& s : shards_)
        if (const auto* t = s->find(name)) return t;
    return nullptr;
}

std::vector<const SafeTensorInfo*> SafetensorsModel::all() const {
    std::vector<const SafeTensorInfo*> out;
    for (const auto& s : shards_)
        for (const auto& t : s->tensors()) out.push_back(&t);
    return out;
}

}  // namespace ie
