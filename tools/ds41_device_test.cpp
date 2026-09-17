// tools/ds41_device_test.cpp — V4.1 Phase 4: the bytes on the GPU.
//
// Checks that V4.1's routed experts upload with no repack and that the engine's existing MXFP4
// expert GEMV reads them correctly, and that dense FP8 dequantises on device exactly as it does
// on the host. Criteria: docs/deepseek41/07_PHASE4_CRITERIA_2026-09-12.md
#include "ie/deepseek41_upload.hpp"
#include "ie/fp8.hpp"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    const std::string l = std::string(ok ? "[ ok ] " : "[FAIL] ") + what +
                          (detail.empty() ? "" : "  (" + detail + ")");
    std::printf("%s\n", l.c_str());
    if (!ok) ++g_fail;
}
uint32_t f32_bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
uint16_t f16_bits(sycl::half h) { uint16_t b; std::memcpy(&b, &h, 2); return b; }

double gib(uint64_t b) { return double(b) / double(1ull << 30); }

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const uint32_t   E_take = argc > 2 ? uint32_t(std::atoi(argv[2])) : 384;

    // ---- C1: GPU clearance ---------------------------------------------------------------
    std::printf("=== C1 GPU clearance ===\n");
    sycl::device dev;
    try {
        std::vector<sycl::device> gpus;
        for (const auto& p : sycl::platform::get_platforms())
            for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
                if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos)
                    gpus.push_back(d);
        if (gpus.empty()) { std::fprintf(stderr, "no Arc GPU found\n"); return 1; }
        dev = gpus.front();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "device enumeration failed: %s\n", e.what()); return 1;
    }
    sycl::queue q(dev, sycl::property::queue::in_order{});
    const uint64_t total_vram = dev.get_info<sycl::info::device::global_mem_size>();
    const bool has_free_mem = dev.has(sycl::aspect::ext_intel_free_memory);
    auto free_vram = [&]() -> uint64_t {
        return has_free_mem ? dev.get_info<sycl::ext::intel::info::device::free_memory>() : 0;
    };
    const uint64_t free_at_start = free_vram();
    std::printf("       %s, %.1f GiB global, %.2f GiB free at start%s\n",
                dev.get_info<sycl::info::device::name>().c_str(), gib(total_vram),
                gib(free_at_start), has_free_mem ? "" : " (free_memory UNSUPPORTED)");
    check(has_free_mem, "device reports free memory, so criterion 7 is measurable");

    ie::DeepSeek41Model m;
    if (const auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    const auto& c = m.config();
    const auto& L0 = m.layers()[0];
    const uint32_t H = c.dim, EF = c.moe_inter_dim;

    // Sized for everything the test allocates, not one bank: w1 and w3 at [inter, dim] plus w2
    // at [dim, inter] (the same bytes, different K), and the dense dequant buffers for C4. The
    // earlier version sized a single bank and under-stated by 3x once C6 began uploading three.
    const uint64_t per_expert = uint64_t(c.moe_inter_dim) * (c.dim / 2) +
                                uint64_t(c.moe_inter_dim) * (c.dim / 32);
    const uint64_t need = uint64_t(E_take) * per_expert * 3
                        + uint64_t(c.n_heads) * c.head_dim * c.q_lora_rank * 3;  // C4 dequant
    check(need + (1ull << 30) < total_vram, "enough VRAM for everything this test allocates",
          std::to_string(E_take) + " experts = " + std::to_string(gib(need)).substr(0, 5) +
          " GiB of " + std::to_string(gib(total_vram)).substr(0, 5));
    if (g_fail) return 1;

    // ---- C5: device decoders vs the host, all 256 codes, fp32 bit patterns ----------------
    std::printf("\n=== C5 device decode == host decode (all 256 codes each) ===\n");
    {
        uint32_t* d_out = sycl::malloc_device<uint32_t>(512, q);
        q.parallel_for(sycl::range<1>(512), [=](sycl::id<1> i) {
            const uint8_t b = uint8_t(i[0] & 0xFF);
            const float v = (i[0] < 256) ? ie::ds41_e4m3(b) : ie::ds41_e8m0(b);
            d_out[i[0]] = sycl::bit_cast<uint32_t>(v);
        }).wait();
        std::vector<uint32_t> got(512);
        q.memcpy(got.data(), d_out, 512 * 4).wait();
        sycl::free(d_out, q);

        int bad4 = 0, bad8 = 0;
        for (int i = 0; i < 256; ++i) {
            const float h = ie::e4m3_to_f32(uint8_t(i));
            const bool hn = h != h, gn = (got[i] & 0x7F800000u) == 0x7F800000u && (got[i] & 0x7FFFFFu);
            if (hn ? !gn : (f32_bits(h) != got[i])) {
                if (++bad4 <= 3) std::printf("       E4M3 0x%02X host 0x%08X device 0x%08X\n",
                                             i, f32_bits(h), got[i]);
            }
        }
        for (int i = 0; i < 256; ++i) {
            const float h = ie::e8m0_to_f32(uint8_t(i));
            const bool hn = h != h, gn = (got[256+i] & 0x7F800000u) == 0x7F800000u && (got[256+i] & 0x7FFFFFu);
            if (hn ? !gn : (f32_bits(h) != got[256 + i])) {
                if (++bad8 <= 3) std::printf("       E8M0 0x%02X host 0x%08X device 0x%08X\n",
                                             i, f32_bits(h), got[256 + i]);
            }
        }
        check(bad4 == 0, "device E4M3 == host on all 256 codes", std::to_string(bad4) + " differ");
        check(bad8 == 0, "device E8M0 == host on all 256 codes", std::to_string(bad8) + " differ");
        // The subnormal path specifically: 2^-127 must survive the device's fast float model.
        check(got[256 + 0] == 0x00400000u,
              "E8M0 byte 0 decodes to 2^-127 on device, not flushed to zero",
              "device bits 0x" + [&]{ char t[16]; std::snprintf(t, 16, "%08X", got[256]); return std::string(t); }());
    }

    // ---- C2: expert upload is a copy -----------------------------------------------------
    std::printf("\n=== C2 expert upload (layer 0 w1, %u experts) ===\n", E_take);
    // Host recomputation of the permutation, reused for all three banks. w2 matters on its own
    // because it runs the repack at a DIFFERENT K (moe_inter_dim 2304, so K/2 = 1152) than w1
    // and w3 (dim 5120, K/2 = 2560); the shuffle is per-16-byte-block and therefore K-independent
    // by construction, but "by construction" is what Phase 4 already got wrong once.
    auto verify_planes = [&](const ie::DS4ExpertBank& b, const std::vector<ie::Ds41Tensor>& src,
                             const char* name) {
        std::vector<uint8_t> back(b.mx_qs_stride), backe(b.mx_e_stride), want(b.mx_qs_stride);
        size_t qs_diff = 0, e_diff = 0;
        for (uint32_t i = 0; i < E_take; ++i) {
            q.memcpy(back.data(),  b.mx_qs + uint64_t(i) * b.mx_qs_stride, b.mx_qs_stride);
            q.memcpy(backe.data(), b.mx_e  + uint64_t(i) * b.mx_e_stride,  b.mx_e_stride).wait();
            const uint8_t* v0 = src[i].w->data;
            for (uint64_t blk = 0; blk < b.mx_qs_stride / 16; ++blk) {
                const uint8_t* v = v0 + blk * 16;
                uint8_t* o = want.data() + blk * 16;
                for (int j = 0; j < 16; ++j) {
                    const int sh = (j & 1) * 4;
                    o[j] = uint8_t(((v[j >> 1] >> sh) & 0xF) | (((v[(j >> 1) + 8] >> sh) & 0xF) << 4));
                }
            }
            if (std::memcmp(back.data(),  want.data(),       b.mx_qs_stride) != 0) ++qs_diff;
            if (std::memcmp(backe.data(), src[i].s->data,    b.mx_e_stride)  != 0) ++e_diff;
        }
        check(qs_diff == 0, std::string(name) + ": FP4 planes are exactly the host-computed repack",
              std::to_string(E_take) + " experts, " + std::to_string(qs_diff) + " differ");
        check(e_diff == 0, std::string(name) + ": E8M0 scale planes are byte-identical to the source",
              std::to_string(E_take) + " experts, " + std::to_string(e_diff) + " differ");
    };

    // One-hot GEMV probe over every within-block index of two blocks, for any bank.
    auto probe_bank = [&](const ie::DS4ExpertBank& b, const std::vector<ie::Ds41Tensor>& src,
                          uint32_t K, uint32_t N, const char* name) {
        sycl::half* x = sycl::malloc_device<sycl::half>(K, q);
        sycl::half* y = sycl::malloc_device<sycl::half>(N, q);
        std::vector<sycl::half> hx(K), hy(N);
        std::vector<uint32_t> probes;
        for (uint32_t j = 0; j < 32; ++j) probes.push_back(j);
        for (uint32_t j = 0; j < 32; ++j) probes.push_back(K - 32 + j);
        size_t probe_bad = 0; std::string first;
        for (uint32_t k0 : probes) {
            for (auto& v : hx) v = sycl::half(0.0f);
            hx[k0] = sycl::half(1.0f);
            q.memcpy(x, hx.data(), K * 2).wait();
            ie::ds4_expert_gemv(q, b, 0, nullptr, x, y).wait();
            q.memcpy(hy.data(), y, N * 2).wait();
            const uint8_t* qs = src[0].w->data;
            const uint8_t* ep = src[0].s->data;
            size_t bad = 0;
            for (uint32_t n = 0; n < N; ++n) {
                const uint8_t byte = qs[uint64_t(n) * (K / 2) + k0 / 2];
                const uint8_t nb   = (k0 & 1u) ? uint8_t(byte >> 4) : uint8_t(byte & 0x0F);
                const int mag = int((0xC8643210u >> ((nb & 7u) * 4u)) & 0xFu);
                const float w = ((nb & 8u) ? -1.0f : 1.0f) * float(mag) * 0.5f;
                const float ref = w * ie::e8m0_to_f32(ep[uint64_t(n) * (K / 32) + k0 / 32]);
                const bool both_zero = (ref == 0.0f) && (float(hy[n]) == 0.0f);
                if (!both_zero && f16_bits(sycl::half(ref)) != f16_bits(hy[n])) ++bad;
            }
            if (bad && first.empty())
                first = "k0=" + std::to_string(k0) + " (within-block " + std::to_string(k0 % 32) +
                        "): " + std::to_string(bad) + " of " + std::to_string(N) + " rows";
            if (bad) ++probe_bad;
        }
        check(probe_bad == 0, std::string(name) + ": y[n] == fp16(W[n,k0]) at EVERY one of the 32 "
              "within-block indices, in two blocks (K=" + std::to_string(K) + ")",
              probe_bad ? first + " (" + std::to_string(probe_bad) + "/" +
                          std::to_string(probes.size()) + " probes failed)"
                        : std::to_string(probes.size()) + " probes, all exact");
        sycl::free(x, q); sycl::free(y, q);
    };

    ie::DS4ExpertBank bank;
    if (const auto e = ie::ds41_expert_bank_upload(q, L0.exp_w1, c.dim, c.moe_inter_dim, E_take, bank);
        !e.empty()) {
        std::fprintf(stderr, "upload: %s\n", e.c_str()); return 1;
    }
    verify_planes(bank, L0.exp_w1, "w1");

    uint64_t file_bytes = 0;
    for (uint32_t i = 0; i < E_take; ++i)
        file_bytes += L0.exp_w1[i].w->nbytes + L0.exp_w1[i].s->nbytes;
    // NOT checked here: `bank_bytes == file_bytes`. ds41_expert_bank_upload REFUSES the upload
    // unless every tensor's nbytes equals its stride, so reaching this line already guarantees
    // it — asserting it would restate a precondition, which is the same decoration criterion 6
    // was failed for the first time round. The driver is the only independent witness, so it is
    // the only thing compared.
    const uint64_t consumed_w1 = free_at_start - free_vram();
    check(has_free_mem && consumed_w1 >= file_bytes && consumed_w1 < file_bytes * 5 / 4 + (256ull << 20),
          "VRAM the driver reports consumed matches the bytes the CHECKPOINT declares",
          std::to_string(gib(consumed_w1)).substr(0, 6) + " GiB consumed for " +
          std::to_string(gib(file_bytes)).substr(0, 6) + " GiB declared");

    // ---- C3: the existing expert GEMV reads them, proved with a one-hot activation --------
    std::printf("\n=== C3 ds4_expert_gemv on the uploaded bank (one-hot, exact) ===\n");
    probe_bank(bank, L0.exp_w1, H, EF, "w1");

    // The criterion named a whole layer's 384 experts at 17.93 MiB each = 6.72 GiB, and one
    // bank is only w1. Upload w3 and w2 as well so the number the contract actually named is
    // the number that gets tested.
    std::printf("\n=== C6 whole-layer residency (w1 + w3 + w2) ===\n");
    {
        ie::DS4ExpertBank b_up, b_down;
        if (auto e = ie::ds41_expert_bank_upload(q, L0.exp_w3, c.dim, c.moe_inter_dim, E_take, b_up);
            !e.empty()) { std::fprintf(stderr, "w3: %s\n", e.c_str()); return 1; }
        if (auto e = ie::ds41_expert_bank_upload(q, L0.exp_w2, c.moe_inter_dim, c.dim, E_take, b_down);
            !e.empty()) { std::fprintf(stderr, "w2: %s\n", e.c_str()); return 1; }
        verify_planes(b_up,   L0.exp_w3, "w3");
        verify_planes(b_down, L0.exp_w2, "w2");
        // w2 contracts over moe_inter_dim, so its K is 2304 and K/2 is 1152 — a different width
        // through the same repack. Probed for its own sake, not inferred from w1.
        probe_bank(b_down, L0.exp_w2, EF, H, "w2");

        uint64_t layer_file = 0;
        for (uint32_t i = 0; i < E_take; ++i)
            layer_file += L0.exp_w1[i].w->nbytes + L0.exp_w1[i].s->nbytes +
                          L0.exp_w3[i].w->nbytes + L0.exp_w3[i].s->nbytes +
                          L0.exp_w2[i].w->nbytes + L0.exp_w2[i].s->nbytes;
        const uint64_t consumed = free_at_start - free_vram();
        std::printf("       %u experts x (w1+w3+w2) = %.3f GiB in the checkpoint, "
                    "%.2f MiB per expert\n", E_take, gib(layer_file),
                    double(layer_file) / E_take / double(1 << 20));
        // Proportional, not absolute: a flat 2 GiB slack swallowed a 2x over-allocation at
        // E=64, which the gate demonstrated. 25% + 256 MiB holds at every bank size.
        check(has_free_mem && consumed >= layer_file &&
              consumed < layer_file * 5 / 4 + (256ull << 20),
              "VRAM consumed for a whole layer's experts matches the checkpoint",
              std::to_string(gib(consumed)).substr(0, 6) + " GiB consumed vs " +
              std::to_string(gib(layer_file)).substr(0, 6) + " GiB declared");
        ie::ds4_expert_bank_free(q, b_up);
        ie::ds4_expert_bank_free(q, b_down);
    }
    ie::ds4_expert_bank_free(q, bank);

    // ---- C4: dense FP8 dequant on device == host -----------------------------------------
    std::printf("\n=== C4 dense FP8 dequant (layers.0.attn.wq_b) ===\n");
    {
        const auto& t = L0.wq_b;
        const uint32_t N = uint32_t(t.w->shape[0]), K = uint32_t(t.w->shape[1]);
        const uint32_t bn = N / uint32_t(t.s->shape[0]), bk = K / uint32_t(t.s->shape[1]);
        const uint64_t nel = uint64_t(N) * K;
        uint8_t* dw = sycl::malloc_device<uint8_t>(nel, q);
        uint8_t* ds = sycl::malloc_device<uint8_t>(t.s->nbytes, q);
        sycl::half* dout = sycl::malloc_device<sycl::half>(nel, q);
        q.memcpy(dw, t.w->data, nel);
        q.memcpy(ds, t.s->data, t.s->nbytes).wait();
        ie::ds41_dense_dequant_f16(q, dw, ds, N, K, bn, bk, dout).wait();
        std::vector<sycl::half> got(nel);
        q.memcpy(got.data(), dout, nel * 2).wait();
        sycl::free(dw, q); sycl::free(ds, q); sycl::free(dout, q);

        const uint32_t SK = K / bk;
        size_t bad = 0; uint64_t first = 0;
        for (uint64_t i = 0; i < nel; ++i) {
            const uint32_t n = uint32_t(i / K), k = uint32_t(i % K);
            const float ref = ie::e4m3_to_f32(t.w->data[i]) *
                              ie::e8m0_to_f32(t.s->data[uint64_t(n / bn) * SK + k / bk]);
            if (f16_bits(sycl::half(ref)) != f16_bits(got[i])) { if (!bad) first = i; ++bad; }
        }
        check(bad == 0, "device dequant == host, as fp16 BIT PATTERNS",
              bad ? "first at " + std::to_string(first) + ", " + std::to_string(bad) + " of " +
                    std::to_string(nel) + " differ"
                  : std::to_string(nel) + " elements, block " + std::to_string(bn) + "x" +
                    std::to_string(bk));
    }

    // ---- C7: teardown actually returns the memory ----------------------------------------
    std::printf("\n=== C7 teardown ===\n");
    {
        q.wait_and_throw();
        const uint64_t end = free_vram();
        const int64_t leaked = int64_t(free_at_start) - int64_t(end);
        std::printf("       free VRAM: %.3f GiB at start, %.3f GiB at end\n",
                    gib(free_at_start), gib(end));
        // 128 MiB against a measured, stable ~70 MiB runtime/kernel-cache baseline.
        check(has_free_mem && leaked < int64_t(128ull << 20),
              "free VRAM returns to its starting value after every free",
              std::to_string(double(leaked) / double(1 << 20)).substr(0, 8) + " MiB not returned");
    }

    std::printf("\n%s\n", g_fail ? ("DEVICE TEST: " + std::to_string(g_fail) + " FAILURE(S)").c_str()
                                 : "DEVICE TEST: PASS");
    return g_fail ? 1 : 0;
}
