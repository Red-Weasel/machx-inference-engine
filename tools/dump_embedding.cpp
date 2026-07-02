// tools/dump_embedding.cpp — debug: compare embedding_lookup_q4k against
// the proven dequant_q4_K_buffer reference for a few token IDs.

#include "ie/allocator.hpp"
#include "ie/dequant_ref.hpp"
#include "ie/gguf.hpp"
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    std::string gguf_path = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    int32_t token = 1814;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"  && i+1 < argc) gguf_path = argv[++i];
        else if (a == "--token" && i+1 < argc) token = std::atoi(argv[++i]);
    }

    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "gguf: %s\n", err.c_str());
        return 1;
    }
    auto* t = g.find_tensor("token_embd.weight");
    if (!t) { std::fprintf(stderr, "no token_embd\n"); return 1; }
    std::printf("token_embd: %.*s shape=[%llu, %llu]\n",
                int(ie::type_name(t->dtype).size()), ie::type_name(t->dtype).data(),
                (unsigned long long)t->shape[0], (unsigned long long)t->shape[1]);

    const uint32_t H = uint32_t(t->shape[0]);

    // CPU reference: dequant the FULL table (slow but proven), pick token T's row.
    // Row layout: token T's H elements live at flat [T * H .. T * H + H).
    // We only dequant the first 16 super-blocks worth of tokens to keep this fast.
    // Actually, since dequant_q4_K_buffer takes a contiguous element count, we can
    // dequant only `(token+1) * H` elements.
    std::vector<float> ref((uint64_t(token) + 1) * H);
    ie::ref::dequant_q4_K_buffer(t->data, ref.size(), ref.data());
    std::printf("CPU ref token %d first 8 values:\n  ", token);
    for (int i = 0; i < 8; ++i) std::printf("%+.4f ", ref[uint64_t(token) * H + i]);
    std::putchar('\n');

    // GPU lookup
    ie::DeviceAllocator alloc;
    if (auto err = alloc.init(); !err.empty()) {
        std::fprintf(stderr, "alloc: %s\n", err.c_str()); return 1;
    }
    auto& q = alloc.queue();
    auto* d_table = alloc.malloc(t->nbytes);
    q.memcpy(d_table, t->data, t->nbytes).wait();
    auto* d_id = sycl::malloc_device<int32_t>(1, q);
    auto* d_y  = sycl::malloc_device<sycl::half>(H, q);
    q.memcpy(d_id, &token, sizeof(int32_t)).wait();
    if (t->dtype == ie::DType::kQ4_K) {
        ie::embedding_lookup_q4k(q, d_id, d_table, d_y, 1, H).wait();
    } else {
        ie::embedding_lookup_q6k(q, d_id, d_table, d_y, 1, H).wait();
    }
    std::vector<sycl::half> got(H);
    q.memcpy(got.data(), d_y, H * sizeof(sycl::half)).wait();
    std::printf("GPU lookup first 8 values:\n  ");
    for (int i = 0; i < 8; ++i) std::printf("%+.4f ", float(got[i]));
    std::putchar('\n');

    // Diff
    float max_abs = 0.f;
    int worst = 0;
    for (uint32_t i = 0; i < H; ++i) {
        const float a = ref[uint64_t(token) * H + i];
        const float b = float(got[i]);
        const float d = std::abs(a - b);
        if (d > max_abs) { max_abs = d; worst = int(i); }
    }
    std::printf("max_abs over %u dims = %.5f (worst@%d ref=%.4f got=%.4f)\n",
                H, max_abs, worst,
                ref[uint64_t(token) * H + worst], float(got[worst]));
    alloc.free(d_table);
    sycl::free(d_id, q); sycl::free(d_y, q);
    return max_abs > 1e-2f ? 1 : 0;
}
