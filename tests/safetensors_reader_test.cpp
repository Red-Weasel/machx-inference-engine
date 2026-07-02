// tests/safetensors_reader_test.cpp — P3e Task 0. Host-only: synthesize a
// minimal .safetensors file, read it back, assert layout/dtype/pointers.
#undef NDEBUG  // Release build (-DNDEBUG); asserts must stay live
#include "ie/safetensors.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Write a minimal single-tensor safetensors file (header + raw bytes).
void write_one(const std::string& path, const std::string& name,
               const std::string& dtype, const std::string& shape_json,
               const void* data, size_t nbytes) {
    const std::string hdr = "{\"" + name + "\":{\"dtype\":\"" + dtype +
        "\",\"shape\":" + shape_json + ",\"data_offsets\":[0," +
        std::to_string(nbytes) + "]}}";
    const uint64_t hlen = hdr.size();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(hdr.data(), std::streamsize(hlen));
    f.write(reinterpret_cast<const char*>(data), std::streamsize(nbytes));
}

// Two tensors: w = F16[2,3] (12 B) at [0,12); qw = I32[4] (16 B) at [12,28).
const char* kHeader =
    "{\"w\":{\"dtype\":\"F16\",\"shape\":[2,3],\"data_offsets\":[0,12]},"
    "\"qw\":{\"dtype\":\"I32\",\"shape\":[4],\"data_offsets\":[12,28]},"
    "\"__metadata__\":{\"format\":\"pt\"}}";

std::string write_synthetic(const uint16_t f16[6], const int32_t i32[4]) {
    const std::string path = "/tmp/ie_safetensors_test.safetensors";
    const uint64_t hlen = std::strlen(kHeader);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(&hlen), 8);   // u64 LE (x86)
    f.write(kHeader, std::streamsize(hlen));
    f.write(reinterpret_cast<const char*>(f16), 12);
    f.write(reinterpret_cast<const char*>(i32), 16);
    f.close();
    return path;
}

}  // namespace

int main() {
    // 1.0, 2.0, -1.0 in IEEE half + filler; check exact bit patterns survive.
    const uint16_t f16[6] = {0x3C00, 0x4000, 0xBC00, 0x0000, 0x3C00, 0x4000};
    const int32_t  i32[4] = {10, -5, 1000, 0x01020304};
    const std::string path = write_synthetic(f16, i32);

    ie::SafetensorsReader r;
    const std::string err = r.open(path);
    if (!err.empty()) { std::fprintf(stderr, "open failed: %s\n", err.c_str()); return 1; }

    assert(r.tensors().size() == 2);
    assert(r.metadata("format") == "pt");
    assert(r.metadata("absent").empty());
    assert(r.find("missing") == nullptr);

    const ie::SafeTensorInfo* w = r.find("w");
    assert(w && w->dtype == ie::DType::kF16 && w->dtype_str == "F16");
    assert((w->shape == std::vector<int64_t>{2, 3}));
    assert(w->nbytes == 12 && w->numel() == 6);
    // memcpy (alignment-safe) — the data pointer views the mmap.
    uint16_t got16[6];
    std::memcpy(got16, w->data, 12);
    for (int i = 0; i < 6; ++i) assert(got16[i] == f16[i]);

    const ie::SafeTensorInfo* qw = r.find("qw");
    assert(qw && qw->dtype == ie::DType::kI32 && qw->dtype_str == "I32");
    assert((qw->shape == std::vector<int64_t>{4}));
    assert(qw->nbytes == 16 && qw->numel() == 4);
    int32_t got32[4];
    std::memcpy(got32, qw->data, 16);
    for (int i = 0; i < 4; ++i) assert(got32[i] == i32[i]);

    // qw must sit exactly 12 bytes after w in the mmap'd data region.
    assert(qw->data == w->data + 12);

    // dtype string mapping spot-checks.
    assert(ie::SafetensorsReader::dtype_from_string("BF16") == ie::DType::kBF16);
    assert(ie::SafetensorsReader::dtype_from_string("F32")  == ie::DType::kF32);
    assert(ie::SafetensorsReader::dtype_from_string("F8_E4M3") == ie::DType::kCount);

    // Move-construct must transfer ownership without invalidating pointers.
    ie::SafetensorsReader r2(std::move(r));
    assert(r2.find("w") != nullptr);

    // ---- SafetensorsModel: sharded directory (index.json + 2 shards) ----
    namespace fs = std::filesystem;
    const std::string sdir = "/tmp/ie_st_shard_test";
    fs::create_directories(sdir);
    const uint16_t af16[2] = {0x3C00, 0x4000};
    const int32_t  bi32[2] = {7, -3};
    write_one(sdir + "/model-00001-of-00002.safetensors", "a", "F16", "[2]", af16, 4);
    write_one(sdir + "/model-00002-of-00002.safetensors", "b", "I32", "[2]", bi32, 8);
    {
        std::ofstream idx(sdir + "/model.safetensors.index.json", std::ios::binary | std::ios::trunc);
        idx << R"({"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}})";
    }
    ie::SafetensorsModel m;
    const std::string merr = m.open(sdir);
    if (!merr.empty()) { std::fprintf(stderr, "model.open: %s\n", merr.c_str()); return 1; }
    assert(m.shard_count() == 2);
    assert(m.all().size() == 2);
    const auto* ta = m.find("a");
    const auto* tb = m.find("b");
    assert(ta && ta->dtype == ie::DType::kF16 && ta->nbytes == 4);
    assert(tb && tb->dtype == ie::DType::kI32 && tb->nbytes == 8);
    int32_t gb[2]; std::memcpy(gb, tb->data, 8);
    assert(gb[0] == 7 && gb[1] == -3);
    assert(m.find("missing") == nullptr);

    // Single-file directory → one shard.
    const std::string single_dir = "/tmp/ie_st_single_test";
    fs::create_directories(single_dir);
    write_one(single_dir + "/model.safetensors", "w", "F32", "[1]", af16, 4);
    ie::SafetensorsModel m1;
    assert(m1.open(single_dir).empty());
    assert(m1.shard_count() == 1 && m1.find("w") != nullptr);

    std::printf("safetensors_reader_test: all OK (reader + sharded/single model)\n");
    return 0;
}
