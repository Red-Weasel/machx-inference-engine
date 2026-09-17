// tests/gguf_writer_test.cpp — P3e Task 2 (Step C). Round-trip: write a GGUF
// with the GgufWriter, read it back with GgufReader, assert KVs + tensors match.
#undef NDEBUG
#include "ie/gguf_writer.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    const std::string path = "/tmp/ie_gguf_writer_test.gguf";

    const float    f32[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const uint16_t f16[6] = {0x3C00, 0x4000, 0xBC00, 0x0000, 0x3C00, 0x4000};

    {
        ie::GgufWriter w;
        w.kv_string("general.architecture", "qwen3");
        w.kv_u32("qwen3.block_count", 2);
        w.kv_u32("qwen3.embedding_length", 2560);
        w.kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
        w.kv_string_array("tokenizer.ggml.tokens", {"a", "bb", "ccc"});
        w.kv_i32_array("tokenizer.ggml.token_type", {1, 1, 1});
        // F32 [4] and F16 [2,3] tensors (block_size 1 → nbytes = elems * bytes).
        w.tensor("blk.0.attn_norm.weight", ie::DType::kF32, {4}, f32, sizeof(f32));
        w.tensor("blk.0.attn_q.weight",   ie::DType::kF16, {2, 3}, f16, sizeof(f16));
        const std::string err = w.write(path);
        if (!err.empty()) { std::fprintf(stderr, "write: %s\n", err.c_str()); return 1; }
    }

    ie::GgufReader g;
    const std::string err = g.open(path);
    if (!err.empty()) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }

    // Arch + KV round-trip (also exercises detect_arch on a written file).
    assert(ie::detect_arch(g) == ie::ModelArch::kQwen3Dense);
    assert(g.alignment() == 32);
    assert(g.find_kv("qwen3.block_count")->as_uint() == 2);
    assert(g.find_kv("qwen3.embedding_length")->as_uint() == 2560);
    assert(std::fabs(float(g.find_kv("qwen3.attention.layer_norm_rms_epsilon")->as_float()) - 1e-6f) <= 1e-9f);
    const auto* toks = g.find_kv("tokenizer.ggml.tokens");
    assert(toks && toks->type == ie::GgufValueType::kArray && toks->n_array == 3);

    // Tensor round-trip: shape, dtype, and exact bytes via the mmap view.
    const auto* tn = g.find_tensor("blk.0.attn_norm.weight");
    assert(tn && tn->dtype == ie::DType::kF32 && tn->n_dims == 1 && tn->shape[0] == 4);
    assert(tn->nbytes == 16);
    assert(std::memcmp(tn->data, f32, 16) == 0);

    const auto* tq = g.find_tensor("blk.0.attn_q.weight");
    assert(tq && tq->dtype == ie::DType::kF16 && tq->n_dims == 2);
    assert(tq->shape[0] == 2 && tq->shape[1] == 3);
    assert(tq->nbytes == 12);
    assert(std::memcmp(tq->data, f16, 12) == 0);
    // Data must be 32-byte aligned within the file.
    assert((g.tensor_data_offset() % 32) == 0);

    // Streaming write must produce a BYTE-IDENTICAL file to write() (P-C: the
    // RAM-safe import path). Same KVs + tensors via tensor_info/begin/stream/end.
    {
        const std::string spath = "/tmp/ie_gguf_writer_stream_test.gguf";
        ie::GgufWriter sw;
        sw.kv_string("general.architecture", "qwen3");
        sw.kv_u32("qwen3.block_count", 2);
        sw.kv_u32("qwen3.embedding_length", 2560);
        sw.kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
        sw.kv_string_array("tokenizer.ggml.tokens", {"a", "bb", "ccc"});
        sw.kv_i32_array("tokenizer.ggml.token_type", {1, 1, 1});
        sw.tensor_info("blk.0.attn_norm.weight", ie::DType::kF32, {4}, sizeof(f32));
        sw.tensor_info("blk.0.attn_q.weight",   ie::DType::kF16, {2, 3}, sizeof(f16));
        assert(sw.begin_streaming(spath).empty());
        assert(sw.stream_next(f32, sizeof(f32)).empty());
        assert(sw.stream_next(f16, sizeof(f16)).empty());
        assert(sw.end_streaming().empty());

        auto slurp = [](const std::string& p) {
            FILE* fp = std::fopen(p.c_str(), "rb"); assert(fp);
            std::fseek(fp, 0, SEEK_END); const size_t n = size_t(std::ftell(fp)); std::fseek(fp, 0, SEEK_SET);
            std::vector<uint8_t> b; b.resize(n);
            assert(std::fread(b.data(), 1, n, fp) == n);
            std::fclose(fp); return b;
        };
        const std::vector<uint8_t> a = slurp(path);
        const std::vector<uint8_t> b = slurp(spath);
        assert(a.size() == b.size());
        assert(std::memcmp(a.data(), b.data(), a.size()) == 0);
        std::printf("  streaming write byte-identical to write() (%zu bytes)\n", a.size());
    }

    std::printf("gguf_writer_test: all OK (KVs + 2 tensors round-trip via GgufReader)\n");
    return 0;
}
