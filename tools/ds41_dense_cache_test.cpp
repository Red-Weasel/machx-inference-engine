// tools/ds41_dense_cache_test.cpp — Phase 8 step 1: the dense text path resident on a device.
//   every layer + head uploaded once; bytes reported; one layer's wq_b bit-identical to a fresh
//   dequant; upload is idempotent; free_all returns the VRAM.
#include "ie/deepseek41_weights.hpp"
#include "ie/deepseek41_upload.hpp"

#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what, const std::string& d = "") {
    std::printf("%s%s%s\n", ok ? "[ ok ] " : "[FAIL] ", what.c_str(), d.empty() ? "" : ("  (" + d + ")").c_str());
    if (!ok) ++g_fail;
}
double gib(uint64_t b) { return double(b) / double(1ull << 30); }
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    const auto& c = m.config();

    sycl::device dev;
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; goto found; }
    std::fprintf(stderr, "no Arc GPU\n"); return 1;
found:
    sycl::queue q(dev, sycl::property::queue::in_order{});
    const bool hf = dev.has(sycl::aspect::ext_intel_free_memory);
    auto free_vram = [&]() -> uint64_t { return hf ? dev.get_info<sycl::ext::intel::info::device::free_memory>() : 0; };
    const uint64_t f0 = free_vram();
    std::printf("%s: %.3f GiB free\n", dev.get_info<sycl::info::device::name>().c_str(), gib(f0));

    ie::Ds41DenseCache cache;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t L = 0; L < c.n_layers; ++L)
        if (auto e = cache.upload_layer(q, m, L); !e.empty()) { std::fprintf(stderr, "layer %u: %s\n", L, e.c_str()); return 1; }
    if (auto e = cache.upload_head(q, m); !e.empty()) { std::fprintf(stderr, "head: %s\n", e.c_str()); return 1; }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const uint64_t f1 = free_vram();
    std::printf("       %u layers + head resident: %.3f GiB by the cache's count, %.3f GiB by the driver, %.1f s\n",
                c.n_layers, gib(cache.bytes()), gib(f0 - f1), secs);
    // fp16 for the FP8 dense (2 B/elem vs 1 B in the file) and BF16->fp16; fp32 for the small F32/BF16 tensors:
    // the census said 8.87 GiB of text-path dense weights in the FILE, so ~2x that resident is expected.
    check(cache.bytes() > 12ull << 30 && cache.bytes() < 24ull << 30, "resident bytes in the expected range (file 8.87 GiB, fp16 on device ~2x)",
          std::to_string(gib(cache.bytes())).substr(0, 5) + " GiB");
    check(hf && (f0 - f1) >= cache.bytes() && (f0 - f1) < cache.bytes() * 5 / 4 + (256ull << 20),
          "VRAM the driver reports consumed matches the cache's count within allocator slack",
          std::to_string(gib(f0 - f1)).substr(0, 5) + " vs " + std::to_string(gib(cache.bytes())).substr(0, 5));

    // idempotent
    const uint64_t b1 = cache.bytes();
    cache.upload_layer(q, m, 5);
    check(cache.bytes() == b1 && free_vram() == f1, "re-uploading a resident layer is a no-op");

    // content: layer 7's resident wq_b == a fresh dequant, bit for bit
    {
        const auto& t = m.layers()[7].wq_b;
        const uint32_t N = uint32_t(t.w->shape[0]), K = uint32_t(t.w->shape[1]);
        const uint32_t bn = N / uint32_t(t.s->shape[0]), bk = K / uint32_t(t.s->shape[1]);
        uint8_t* dw = sycl::malloc_device<uint8_t>(size_t(N) * K, q); uint8_t* ds = sycl::malloc_device<uint8_t>(t.s->nbytes, q);
        sycl::half* fresh = sycl::malloc_device<sycl::half>(size_t(N) * K, q);
        q.memcpy(dw, t.w->data, size_t(N) * K); q.memcpy(ds, t.s->data, t.s->nbytes).wait();
        ie::ds41_dense_dequant_f16(q, dw, ds, N, K, bn, bk, fresh).wait();
        std::vector<sycl::half> a(size_t(N) * K), b(size_t(N) * K);
        q.memcpy(a.data(), cache.layer(7)->wq_b, a.size() * 2); q.memcpy(b.data(), fresh, b.size() * 2).wait();
        check(std::memcmp(a.data(), b.data(), a.size() * 2) == 0, "resident wq_b (layer 7) is bit-identical to a fresh dequant",
              std::to_string(a.size()) + " halves");
        sycl::free(dw, q); sycl::free(ds, q); sycl::free(fresh, q);
    }
    // kinds: the conditional tensors are present exactly where the kind says
    {
        size_t bad = 0;
        for (uint32_t L = 0; L < c.n_layers; ++L) {
            const auto* d = cache.layer(L); const auto& k = m.layers()[L].kind;
            if ((d->comp_wkv != nullptr) != k.is_kv_source) ++bad;
            if ((d->comp_wgate != nullptr) != k.has_compressor_gate) ++bad;
            if ((d->idx_wq_b != nullptr) != k.is_index_source) ++bad;
            if ((d->idx_wk != nullptr) != k.is_kv_source) ++bad;
            if ((d->engram_wkv != nullptr) != k.has_engram) ++bad;
        }
        check(bad == 0, "conditional tensors resident exactly where the layer kind says", std::to_string(bad) + " mismatches");
    }
    cache.free_all(q); q.wait_and_throw();
    const uint64_t f2 = free_vram();
    check(hf && int64_t(f0) - int64_t(f2) < int64_t(128ull << 20), "free_all returns the VRAM within 128 MiB",
          std::to_string(double(int64_t(f0) - int64_t(f2)) / (1 << 20)).substr(0, 7) + " MiB unreturned");
    std::printf("\n%s\n", g_fail ? ("DENSE CACHE TEST: " + std::to_string(g_fail) + " FAILURE(S)").c_str() : "DENSE CACHE TEST: PASS");
    return g_fail ? 1 : 0;
}
