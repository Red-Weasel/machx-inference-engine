// include/ie/gguf_writer.hpp — minimal GGUF v3 writer (P3e import target).
//
// Emits a file the engine's own GgufReader parses, so an imported HF/AWQ model
// becomes a standard GGUF the existing loader/forward runs unchanged. Streams
// tensor data (no giant in-RAM buffer). Little-endian (x86); GGUF is LE.
//
// Tensor `shape` is GGML ne[] order — ne[0] is the CONTIGUOUS leading dim. For
// a Linear weight mapping in→out, write shape = {in, out} with data laid out as
// out rows × in cols ([out][in] row-major). (Callers transpose as needed.)
#pragma once

#include "ie/dtype.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace ie {

class GgufWriter {
public:
    void kv_u32(const std::string& key, uint32_t v);
    void kv_i32(const std::string& key, int32_t v);
    void kv_f32(const std::string& key, float v);
    void kv_bool(const std::string& key, bool v);
    void kv_string(const std::string& key, const std::string& v);
    void kv_string_array(const std::string& key, const std::vector<std::string>& v);
    void kv_i32_array(const std::string& key, const std::vector<int32_t>& v);
    void kv_f32_array(const std::string& key, const std::vector<float>& v);

    // `data` is NON-OWNING and must outlive write(). nbytes must equal the GGUF
    // size for (dtype, shape).
    void tensor(const std::string& name, DType dtype,
                const std::vector<uint64_t>& shape,
                const void* data, uint64_t nbytes);

    // Write the GGUF to `path`. Returns "" on success, error text otherwise.
    std::string write(const std::string& path, uint32_t alignment = 32);

    // --- Streaming write (RAM-safe import of huge models: only one tensor's
    // bytes live in RAM at a time, vs write() which needs every tensor resident).
    // Usage: tensor_info() for EVERY tensor (specs only, no data, in the order
    // they will stream) → begin_streaming() (writes the header + offsets) →
    // stream_next(data, nbytes) once per tensor IN THE SAME ORDER → end_streaming().
    // Produces a byte-identical file to write() for the same tensors. ---
    void tensor_info(const std::string& name, DType dtype,
                     const std::vector<uint64_t>& shape, uint64_t nbytes);
    std::string begin_streaming(const std::string& path, uint32_t alignment = 32);
    std::string stream_next(const void* data, uint64_t nbytes);
    std::string end_streaming();

private:
    struct Kv { std::string key; std::vector<uint8_t> body; };  // body = type(u32)+value
    struct Tensor {
        std::string name; uint32_t dtype;
        std::vector<uint64_t> shape; const void* data; uint64_t nbytes;
    };
    // Build the metadata block (magic + version + counts + KVs + tensor infos
    // with `offsets`). Shared by write() and begin_streaming() so they stay in
    // lockstep. `offsets[i]` is tensor i's start within the data section.
    std::vector<uint8_t> build_meta(const std::vector<uint64_t>& offsets,
                                    uint32_t alignment) const;

    std::vector<Kv>     kvs_;
    std::vector<Tensor> tensors_;

    // Streaming state (used between begin_streaming and end_streaming).
    std::ofstream         stream_f_;
    std::vector<uint64_t> stream_offsets_;
    size_t                stream_idx_  = 0;
    uint64_t              stream_pos_  = 0;   // relative to data_start
    uint32_t              stream_align_ = 32;
};

}  // namespace ie
