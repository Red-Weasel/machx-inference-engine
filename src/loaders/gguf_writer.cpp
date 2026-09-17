// src/loaders/gguf_writer.cpp — minimal GGUF v3 writer. See gguf_writer.hpp.
#include "ie/gguf_writer.hpp"

#include <cstring>
#include <fstream>

namespace ie {

namespace {

// GGUF metadata value type ids (mirror GgufValueType in gguf.hpp).
enum : uint32_t {
    GT_U8 = 0, GT_I8 = 1, GT_U16 = 2, GT_I16 = 3, GT_U32 = 4, GT_I32 = 5,
    GT_F32 = 6, GT_BOOL = 7, GT_STRING = 8, GT_ARRAY = 9, GT_U64 = 10,
    GT_I64 = 11, GT_F64 = 12,
};

template <typename T>
void put(std::vector<uint8_t>& b, const T& v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

}  // namespace

void GgufWriter::kv_u32(const std::string& k, uint32_t v) {
    std::vector<uint8_t> body; put<uint32_t>(body, GT_U32); put<uint32_t>(body, v);
    kvs_.push_back({k, std::move(body)});
}
void GgufWriter::kv_i32(const std::string& k, int32_t v) {
    std::vector<uint8_t> body; put<uint32_t>(body, GT_I32); put<int32_t>(body, v);
    kvs_.push_back({k, std::move(body)});
}
void GgufWriter::kv_f32(const std::string& k, float v) {
    std::vector<uint8_t> body; put<uint32_t>(body, GT_F32); put<float>(body, v);
    kvs_.push_back({k, std::move(body)});
}
void GgufWriter::kv_bool(const std::string& k, bool v) {
    std::vector<uint8_t> body; put<uint32_t>(body, GT_BOOL); body.push_back(v ? 1 : 0);
    kvs_.push_back({k, std::move(body)});
}
void GgufWriter::kv_string(const std::string& k, const std::string& v) {
    std::vector<uint8_t> body; put<uint32_t>(body, GT_STRING); put_str(body, v);
    kvs_.push_back({k, std::move(body)});
}
void GgufWriter::kv_string_array(const std::string& k, const std::vector<std::string>& v) {
    std::vector<uint8_t> body;
    put<uint32_t>(body, GT_ARRAY); put<uint32_t>(body, GT_STRING); put<uint64_t>(body, v.size());
    for (const auto& s : v) put_str(body, s);
    kvs_.push_back({k, std::move(body)});
}
void GgufWriter::kv_i32_array(const std::string& k, const std::vector<int32_t>& v) {
    std::vector<uint8_t> body;
    put<uint32_t>(body, GT_ARRAY); put<uint32_t>(body, GT_I32); put<uint64_t>(body, v.size());
    for (int32_t x : v) put<int32_t>(body, x);
    kvs_.push_back({k, std::move(body)});
}
void GgufWriter::kv_f32_array(const std::string& k, const std::vector<float>& v) {
    std::vector<uint8_t> body;
    put<uint32_t>(body, GT_ARRAY); put<uint32_t>(body, GT_F32); put<uint64_t>(body, v.size());
    for (float x : v) put<float>(body, x);
    kvs_.push_back({k, std::move(body)});
}

void GgufWriter::tensor(const std::string& name, DType dtype,
                        const std::vector<uint64_t>& shape,
                        const void* data, uint64_t nbytes) {
    tensors_.push_back({name, uint32_t(dtype), shape, data, nbytes});
}

std::vector<uint8_t> GgufWriter::build_meta(const std::vector<uint64_t>& offsets,
                                            uint32_t alignment) const {
    std::vector<uint8_t> meta;
    const char magic[4] = {'G', 'G', 'U', 'F'};
    meta.insert(meta.end(), magic, magic + 4);
    put<uint32_t>(meta, 3u);                              // version
    put<uint64_t>(meta, uint64_t(tensors_.size()));       // n_tensors
    put<uint64_t>(meta, uint64_t(kvs_.size()) + 1);       // n_kv (+ general.alignment)

    // general.alignment first so a reader honors our padding.
    put_str(meta, "general.alignment");
    put<uint32_t>(meta, GT_U32); put<uint32_t>(meta, alignment);

    for (const auto& kv : kvs_) {
        put_str(meta, kv.key);
        meta.insert(meta.end(), kv.body.begin(), kv.body.end());
    }
    for (size_t i = 0; i < tensors_.size(); ++i) {
        const auto& t = tensors_[i];
        put_str(meta, t.name);
        put<uint32_t>(meta, uint32_t(t.shape.size()));
        for (uint64_t d : t.shape) put<uint64_t>(meta, d);
        put<uint32_t>(meta, t.dtype);
        put<uint64_t>(meta, offsets[i]);
    }
    return meta;
}

std::string GgufWriter::write(const std::string& path, uint32_t alignment) {
    if (alignment == 0) return "alignment must be > 0";
    auto align_up = [&](uint64_t x) { return (x + alignment - 1) / alignment * alignment; };

    std::vector<uint64_t> offsets(tensors_.size());
    uint64_t running = 0;
    for (size_t i = 0; i < tensors_.size(); ++i) {
        offsets[i] = align_up(running);
        running    = offsets[i] + tensors_[i].nbytes;
    }

    const std::vector<uint8_t> meta = build_meta(offsets, alignment);
    const uint64_t meta_size  = meta.size();
    const uint64_t data_start = align_up(meta_size);

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return "cannot open output: " + path;
    f.write(reinterpret_cast<const char*>(meta.data()), std::streamsize(meta_size));

    static const char zeros[64] = {0};
    auto pad = [&](uint64_t n) {
        while (n > 0) { uint64_t c = n < 64 ? n : 64; f.write(zeros, std::streamsize(c)); n -= c; }
    };
    pad(data_start - meta_size);                          // pad metadata → data_start

    uint64_t pos = 0;                                     // relative to data_start
    for (size_t i = 0; i < tensors_.size(); ++i) {
        pad(offsets[i] - pos);                            // inter-tensor alignment
        f.write(reinterpret_cast<const char*>(tensors_[i].data),
                std::streamsize(tensors_[i].nbytes));
        pos = offsets[i] + tensors_[i].nbytes;
    }
    if (!f) return "write error on: " + path;
    return {};
}

// ---- streaming write (RAM-safe; one tensor in RAM at a time) -----------
void GgufWriter::tensor_info(const std::string& name, DType dtype,
                             const std::vector<uint64_t>& shape, uint64_t nbytes) {
    tensors_.push_back({name, uint32_t(dtype), shape, nullptr, nbytes});
}

std::string GgufWriter::begin_streaming(const std::string& path, uint32_t alignment) {
    if (alignment == 0) return "alignment must be > 0";
    stream_align_ = alignment;
    auto align_up = [&](uint64_t x) { return (x + alignment - 1) / alignment * alignment; };

    stream_offsets_.assign(tensors_.size(), 0);
    uint64_t running = 0;
    for (size_t i = 0; i < tensors_.size(); ++i) {
        stream_offsets_[i] = align_up(running);
        running            = stream_offsets_[i] + tensors_[i].nbytes;
    }

    const std::vector<uint8_t> meta = build_meta(stream_offsets_, alignment);
    const uint64_t meta_size  = meta.size();
    const uint64_t data_start = align_up(meta_size);

    stream_f_.open(path, std::ios::binary | std::ios::trunc);
    if (!stream_f_) return "cannot open output: " + path;
    stream_f_.write(reinterpret_cast<const char*>(meta.data()), std::streamsize(meta_size));
    static const char zeros[64] = {0};
    for (uint64_t n = data_start - meta_size; n > 0; ) {
        uint64_t c = n < 64 ? n : 64; stream_f_.write(zeros, std::streamsize(c)); n -= c;
    }
    stream_pos_ = 0;
    stream_idx_ = 0;
    return stream_f_ ? std::string{} : std::string("header write error on: " + path);
}

std::string GgufWriter::stream_next(const void* data, uint64_t nbytes) {
    if (stream_idx_ >= tensors_.size()) return "stream_next: more tensors than declared";
    if (nbytes != tensors_[stream_idx_].nbytes)
        return "stream_next: nbytes " + std::to_string(nbytes) + " != declared " +
               std::to_string(tensors_[stream_idx_].nbytes) + " for '" + tensors_[stream_idx_].name + "'";
    static const char zeros[64] = {0};
    for (uint64_t n = stream_offsets_[stream_idx_] - stream_pos_; n > 0; ) {  // inter-tensor align
        uint64_t c = n < 64 ? n : 64; stream_f_.write(zeros, std::streamsize(c)); n -= c;
    }
    stream_f_.write(reinterpret_cast<const char*>(data), std::streamsize(nbytes));
    stream_pos_ = stream_offsets_[stream_idx_] + nbytes;
    ++stream_idx_;
    return stream_f_ ? std::string{} : std::string("stream write error");
}

std::string GgufWriter::end_streaming() {
    if (stream_idx_ != tensors_.size())
        return "end_streaming: streamed " + std::to_string(stream_idx_) + " of " +
               std::to_string(tensors_.size()) + " tensors";
    stream_f_.flush();
    const bool okf = bool(stream_f_);
    stream_f_.close();
    return okf ? std::string{} : std::string("stream flush error");
}

}  // namespace ie
