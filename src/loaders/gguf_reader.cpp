// src/loaders/gguf_reader.cpp — mmap-based GGUF v3 reader.
//
// Cross-references research/03_quant_formats.md §1.

#include "ie/gguf.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ie {

namespace {

constexpr uint8_t kMagic[4] = {'G', 'G', 'U', 'F'};

// Scalar value sizes (excluding kBool which is encoded as 1 byte).
constexpr uint8_t kScalarSize[13] = {
    1, 1, 2, 2, 4, 4, 4, 1, /*string*/0, /*array*/0, 8, 8, 8
};

bool is_scalar(GgufValueType t) noexcept {
    return t != GgufValueType::kString && t != GgufValueType::kArray;
}

// Returns size of a single scalar value of type t. 0 for non-scalars.
size_t scalar_size(GgufValueType t) noexcept {
    auto i = static_cast<uint32_t>(t);
    return (i < 13) ? kScalarSize[i] : 0;
}

// Reads a uint64-prefixed UTF-8 string. Returns the byte cursor *after* the string.
// Sets out_view to the in-place view (no copy).
const uint8_t* read_string(const uint8_t* p, const uint8_t* end, std::string_view& out, bool& ok) {
    if (end - p < 8) { ok = false; return p; }
    uint64_t len;
    std::memcpy(&len, p, 8);
    p += 8;
    if (uint64_t(end - p) < len) { ok = false; return p; }
    out = std::string_view(reinterpret_cast<const char*>(p), len);
    return p + len;
}

// Skip a value of type `t`. Returns cursor after the value.
const uint8_t* skip_value(const uint8_t* p, const uint8_t* end, GgufValueType t, bool& ok) {
    if (is_scalar(t)) {
        size_t s = scalar_size(t);
        if (uint64_t(end - p) < s) { ok = false; return p; }
        return p + s;
    }
    if (t == GgufValueType::kString) {
        std::string_view dummy;
        return read_string(p, end, dummy, ok);
    }
    // ARRAY: { uint32_t inner_type; uint64_t n; values[n]; }
    if (end - p < 12) { ok = false; return p; }
    uint32_t inner_t;
    uint64_t n;
    std::memcpy(&inner_t, p, 4);
    std::memcpy(&n, p + 4, 8);
    p += 12;
    auto inner = static_cast<GgufValueType>(inner_t);
    if (is_scalar(inner)) {
        size_t s = scalar_size(inner);
        if (s == 0) { ok = false; return p; }
        if (uint64_t(end - p) < n * s) { ok = false; return p; }
        return p + n * s;
    }
    if (inner == GgufValueType::kString) {
        for (uint64_t i = 0; i < n; ++i) {
            std::string_view dummy;
            p = read_string(p, end, dummy, ok);
            if (!ok) return p;
        }
        return p;
    }
    // Nested arrays are not in the spec.
    ok = false;
    return p;
}

}  // namespace

std::string_view name_of(GgufValueType v) noexcept {
    switch (v) {
        case GgufValueType::kU8:     return "u8";
        case GgufValueType::kI8:     return "i8";
        case GgufValueType::kU16:    return "u16";
        case GgufValueType::kI16:    return "i16";
        case GgufValueType::kU32:    return "u32";
        case GgufValueType::kI32:    return "i32";
        case GgufValueType::kF32:    return "f32";
        case GgufValueType::kBool:   return "bool";
        case GgufValueType::kString: return "string";
        case GgufValueType::kArray:  return "array";
        case GgufValueType::kU64:    return "u64";
        case GgufValueType::kI64:    return "i64";
        case GgufValueType::kF64:    return "f64";
    }
    return "?";
}

// ===== KvRef accessors =====

bool KvRef::as_bool() const noexcept {
    if (type != GgufValueType::kBool) return false;
    return *payload != 0;
}

int64_t KvRef::as_int() const noexcept {
    switch (type) {
        case GgufValueType::kU8:  return *reinterpret_cast<const uint8_t*>(payload);
        case GgufValueType::kI8:  return *reinterpret_cast<const int8_t*>(payload);
        case GgufValueType::kU16: { uint16_t v; std::memcpy(&v, payload, 2); return v; }
        case GgufValueType::kI16: { int16_t  v; std::memcpy(&v, payload, 2); return v; }
        case GgufValueType::kU32: { uint32_t v; std::memcpy(&v, payload, 4); return int64_t(v); }
        case GgufValueType::kI32: { int32_t  v; std::memcpy(&v, payload, 4); return v; }
        case GgufValueType::kU64: { uint64_t v; std::memcpy(&v, payload, 8); return int64_t(v); }
        case GgufValueType::kI64: { int64_t  v; std::memcpy(&v, payload, 8); return v; }
        case GgufValueType::kBool: return *payload != 0;
        default: return 0;
    }
}
uint64_t KvRef::as_uint() const noexcept { return uint64_t(as_int()); }
double KvRef::as_float() const noexcept {
    if (type == GgufValueType::kF32) { float v; std::memcpy(&v, payload, 4); return v; }
    if (type == GgufValueType::kF64) { double v; std::memcpy(&v, payload, 8); return v; }
    return 0.0;
}
std::string_view KvRef::as_string() const noexcept {
    if (type != GgufValueType::kString) return {};
    uint64_t len;
    std::memcpy(&len, payload, 8);
    return std::string_view(reinterpret_cast<const char*>(payload + 8), len);
}
std::vector<std::string_view> KvRef::as_string_array() const {
    std::vector<std::string_view> out;
    if (type != GgufValueType::kArray || inner_type != GgufValueType::kString) return out;
    out.reserve(n_array);
    const uint8_t* p = payload;
    for (uint64_t i = 0; i < n_array; ++i) {
        uint64_t len;
        std::memcpy(&len, p, 8);
        p += 8;
        out.emplace_back(reinterpret_cast<const char*>(p), len);
        p += len;
    }
    return out;
}

// ===== GgufReader =====

GgufReader::~GgufReader() { close(); }

GgufReader::GgufReader(GgufReader&& other) noexcept { *this = std::move(other); }
GgufReader& GgufReader::operator=(GgufReader&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_; base_ = other.base_; size_ = other.size_;
        version_ = other.version_; alignment_ = other.alignment_;
        tensor_data_off_ = other.tensor_data_off_;
        kvs_ = std::move(other.kvs_);
        tensors_ = std::move(other.tensors_);
        extra_shards_ = std::move(other.extra_shards_);
        other.fd_ = -1; other.base_ = nullptr; other.size_ = 0;
    }
    return *this;
}

void GgufReader::close() noexcept {
    if (base_) ::munmap(base_, size_);
    if (fd_ >= 0) ::close(fd_);
    for (auto& s : extra_shards_) {
        if (s.base) ::munmap(s.base, s.size);
        if (s.fd >= 0) ::close(s.fd);
    }
    extra_shards_.clear();
    base_ = nullptr;
    fd_ = -1;
    size_ = 0;
    kvs_.clear();
    tensors_.clear();
}

namespace {
// Parse a "...-NNNNN-of-MMMMM.gguf" split path. false if it doesn't match. On success
// prefix = chars before NNNNN, tail = "-of-MMMMM.gguf", shard_no = NNNNN, count = MMMMM.
bool parse_shard_suffix(const std::string& path, std::string& prefix, std::string& tail,
                        uint32_t& shard_no, uint32_t& count) {
    if (path.size() < 19) return false;
    if (path.compare(path.size() - 5, 5, ".gguf") != 0) return false;
    const size_t mmmmm = path.size() - 10;            // MMMMM start (5 digits before ".gguf")
    if (mmmmm < 9) return false;
    if (path.compare(mmmmm - 4, 4, "-of-") != 0) return false;
    const size_t nnnnn = mmmmm - 9;                   // NNNNN start (= mmmmm - 4 - 5)
    auto digits = [&](size_t s) {
        for (size_t i = s; i < s + 5; ++i) if (path[i] < '0' || path[i] > '9') return false;
        return true;
    };
    if (!digits(nnnnn) || !digits(mmmmm)) return false;
    shard_no = uint32_t(std::stoul(path.substr(nnnnn, 5)));
    count    = uint32_t(std::stoul(path.substr(mmmmm, 5)));
    prefix   = path.substr(0, nnnnn);
    tail     = path.substr(mmmmm - 4);                // "-of-MMMMM.gguf"
    return true;
}
std::string shard_path(const std::string& prefix, const std::string& tail, uint32_t i_1based) {
    char num[8];
    std::snprintf(num, sizeof(num), "%05u", i_1based);
    return prefix + num + tail;
}
}  // namespace

std::string GgufReader::open(const std::string& path) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return std::string("open(") + path + ") failed: " + std::strerror(errno);
    struct stat st{};
    if (::fstat(fd_, &st) != 0) { close(); return std::string("fstat failed: ") + std::strerror(errno); }
    size_ = static_cast<size_t>(st.st_size);
    if (size_ < 24) { close(); return "file shorter than GGUF header"; }
    void* m = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (m == MAP_FAILED) { close(); return std::string("mmap failed: ") + std::strerror(errno); }
    base_ = static_cast<uint8_t*>(m);
    ::madvise(base_, size_, MADV_RANDOM);   // tensor data is touched sparsely

    if (auto e = parse_shard(base_, size_, /*primary=*/true); !e.empty()) { close(); return e; }

    // Multi-shard split GGUF: shard 0 carries all metadata + split.count; siblings are
    // "...-NNNNN-of-MMMMM.gguf" and contribute tensors only. Trigger off the filename
    // convention (robust vs the KV scalar type), cross-checked with split.count.
    std::string prefix, tail; uint32_t shard_no = 0, fcount = 0;
    const bool named = parse_shard_suffix(path, prefix, tail, shard_no, fcount);
    uint32_t count = named ? fcount : 1;
    if (const KvRef* kc = find_kv("split.count")) {
        const uint64_t kv = kc->as_uint();
        if (kv > count) count = uint32_t(kv);
    }
    if (count > 1) {
        if (!named) { close(); return "split GGUF (split.count>1) but path is not '...-NNNNN-of-MMMMM.gguf'"; }
        if (shard_no != 1) { close(); return "open the FIRST shard (…-00001-of-…) of a split GGUF"; }
        for (uint32_t i = 2; i <= count; ++i) {
            const std::string sp = shard_path(prefix, tail, i);
            int sfd = ::open(sp.c_str(), O_RDONLY);
            if (sfd < 0) { close(); return std::string("split shard open(") + sp + ") failed: " + std::strerror(errno); }
            struct stat ss{};
            if (::fstat(sfd, &ss) != 0) { ::close(sfd); close(); return "split shard fstat failed"; }
            const size_t ssz = size_t(ss.st_size);
            void* sm = ::mmap(nullptr, ssz, PROT_READ, MAP_PRIVATE, sfd, 0);
            if (sm == MAP_FAILED) { ::close(sfd); close(); return std::string("split shard mmap failed: ") + sp; }
            ::madvise(sm, ssz, MADV_RANDOM);
            extra_shards_.push_back({sfd, static_cast<uint8_t*>(sm), ssz});
            if (auto e = parse_shard(static_cast<uint8_t*>(sm), ssz, /*primary=*/false); !e.empty()) {
                close(); return std::string("split shard ") + sp + ": " + e;
            }
        }
        if (const KvRef* kt = find_kv("split.tensors.count")) {
            const uint64_t want = kt->as_uint();
            if (want != 0 && want != tensors_.size()) {
                close();
                char buf[160];
                std::snprintf(buf, sizeof(buf),
                              "split tensor count mismatch: merged %zu, split.tensors.count=%llu",
                              tensors_.size(), (unsigned long long)want);
                return buf;
            }
        }
    }
    return {};   // success
}

std::string GgufReader::parse_shard(const uint8_t* base, size_t size, bool primary) {
    const uint8_t* end = base + size;
    const uint8_t* p   = base;
    if (size < 24) return "shard shorter than GGUF header";
    if (std::memcmp(p, kMagic, 4) != 0) return "magic mismatch (not a GGUF file)";
    p += 4;
    uint32_t version;
    std::memcpy(&version, p, 4); p += 4;
    if (version != 3) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "GGUF version %u not supported (need 3)", version);
        return buf;
    }
    uint64_t n_tensor, n_kv;
    std::memcpy(&n_tensor, p, 8); p += 8;
    std::memcpy(&n_kv,     p, 8); p += 8;

    // KV pairs: stored only for the primary; secondary shards just advance the cursor.
    uint64_t shard_align = 32;
    if (primary) kvs_.reserve(n_kv);
    for (uint64_t i = 0; i < n_kv; ++i) {
        bool ok = true;
        std::string_view key;
        p = read_string(p, end, key, ok);
        if (!ok) return "failed to read KV key";
        if (end - p < 4) return "truncated KV type";
        uint32_t type_raw;
        std::memcpy(&type_raw, p, 4); p += 4;
        auto t = static_cast<GgufValueType>(type_raw);
        KvRef ref; ref.key = key; ref.type = t;
        if (t == GgufValueType::kArray) {
            if (end - p < 12) return "truncated array header";
            uint32_t inner_t;
            std::memcpy(&inner_t, p, 4);
            std::memcpy(&ref.n_array, p + 4, 8);
            ref.inner_type = static_cast<GgufValueType>(inner_t);
            ref.payload = p + 12;
        } else {
            ref.payload = p;
        }
        if (t == GgufValueType::kU32 && key == "general.alignment") {
            uint32_t a; std::memcpy(&a, p, 4); shard_align = a;   // this shard's own padding
        }
        if (primary) kvs_.push_back(ref);
        p = skip_value(p, end, t, ok);
        if (!ok) return "failed to skip KV value";
    }
    if (shard_align == 0) shard_align = 32;
    if (primary) { version_ = version; alignment_ = shard_align; }

    // Tensor-info table → appended to tensors_ (merged across shards).
    const size_t first = tensors_.size();
    tensors_.reserve(first + n_tensor);
    for (uint64_t i = 0; i < n_tensor; ++i) {
        bool ok = true;
        std::string_view name;
        p = read_string(p, end, name, ok);
        if (!ok) return "failed to read tensor name";
        if (end - p < 4) return "truncated tensor n_dims";
        uint32_t nd;
        std::memcpy(&nd, p, 4); p += 4;
        if (nd == 0 || nd > kMaxDims) return "tensor n_dims out of range";
        if (uint64_t(end - p) < 8u * nd) return "truncated shape";
        std::array<uint64_t, kMaxDims> shape{};
        std::memcpy(shape.data(), p, 8u * nd);
        p += 8u * nd;
        if (end - p < 4 + 8) return "truncated tensor info trailer";
        uint32_t type_raw; uint64_t off;
        std::memcpy(&type_raw, p, 4); p += 4;
        std::memcpy(&off, p, 8);      p += 8;
        auto dtype = static_cast<DType>(type_raw);
        if (!type_info(dtype)) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "unsupported tensor dtype id %u", type_raw);
            return buf;
        }
        uint64_t row_elems = shape[0], hi = 1;
        for (uint32_t d = 1; d < nd; ++d) hi *= shape[d];
        GgufTensorInfo info{};
        info.name = name; info.dtype = dtype; info.n_dims = nd; info.shape = shape;
        info.offset_in_data = off;
        info.nbytes = bytes_for(dtype, row_elems) * hi;
        tensors_.push_back(info);
    }

    // This shard's data section: pad the post-header cursor to the shard's alignment,
    // then resolve the tensors THIS shard contributed against `base` (each shard has
    // its own data blob — a split layer straddles shards, so resolve per-shard).
    const size_t cursor_off = size_t(p - base);
    const size_t pad = (shard_align - (cursor_off % shard_align)) % shard_align;
    const uint64_t data_off = cursor_off + pad;
    if (primary) tensor_data_off_ = data_off;
    if (data_off > size) return "tensor data starts past EOF";
    for (size_t ti = first; ti < tensors_.size(); ++ti) {
        auto& t = tensors_[ti];
        const uint64_t abs_off = data_off + t.offset_in_data;
        if (abs_off + t.nbytes > size) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "tensor '%.*s' overruns shard (off=%llu, nbytes=%llu, shard=%zu)",
                          int(t.name.size()), t.name.data(),
                          (unsigned long long)abs_off, (unsigned long long)t.nbytes, size);
            return buf;
        }
        t.data = base + abs_off;
    }
    return {};
}

const KvRef* GgufReader::find_kv(std::string_view key) const noexcept {
    for (const auto& kv : kvs_) if (kv.key == key) return &kv;
    return nullptr;
}

const GgufTensorInfo* GgufReader::find_tensor(std::string_view name) const noexcept {
    for (const auto& t : tensors_) if (t.name == name) return &t;
    return nullptr;
}

Tensor GgufReader::make_host_view(const GgufTensorInfo& info) const noexcept {
    Tensor t{};
    t.dtype = info.dtype;
    t.device = Device::kHost;
    t.n_dims = info.n_dims;
    t.shape = info.shape;
    t.data = const_cast<uint8_t*>(info.data);
    t.nbytes = info.nbytes;
    t.name = info.name;
    return t;
}

}  // namespace ie
