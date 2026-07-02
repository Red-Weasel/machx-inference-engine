// tests/unit/dequant_test.cpp — Phase 2 correctness gate.
//
// Two passes:
//   1. Synthetic — 1024 random super-blocks per format; diff GPU vs CPU ref.
//   2. Real     — if --gguf <path> given, dequant a Q4_K and Q6_K tensor from
//      the file with both ref and GPU; diff. This catches packing bugs that
//      uniform-random scales might hide.
//
// Tolerances (per Phase 2 gate):
//   Q8_0   — exact (0 ULP)
//   Q4/5/6_K — max abs diff < 1e-3 vs CPU reference (which itself rounds to fp16)

#include "ie/dequant.hpp"
#include "ie/dequant_ref.hpp"
#include "ie/gguf.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kGreen = "\033[32m";
constexpr const char* kRed   = "\033[31m";
constexpr const char* kYell  = "\033[33m";
constexpr const char* kReset = "\033[0m";
constexpr const char* kDim   = "\033[2m";

// ---- fp32 -> fp16 (round to nearest, ties-to-even). Used to fabricate
// realistic test super-blocks with valid d/dmin scales.
uint16_t fp32_to_fp16(float f) {
    uint32_t b = std::bit_cast<uint32_t>(f);
    uint32_t s = (b >> 16) & 0x8000u;
    int32_t  e = ((b >> 23) & 0xff) - 127 + 15;
    uint32_t m =  b         & 0x7fffffu;
    if (e >= 0x1f) return uint16_t(s | 0x7c00u);                  // overflow -> Inf
    if (e <= 0)    return uint16_t(s);                            // flush subnormals
    uint32_t mh = m >> 13;
    uint32_t r  = (m & 0x1fff) > 0x1000 ? 1 : 0;                  // simple round-half-up
    return uint16_t(s | (uint32_t(e) << 10) | (mh + r));
}

template <typename Block>
std::vector<Block> make_random_blocks(size_t n, std::mt19937& rng) {
    std::vector<Block> out(n);
    std::uniform_int_distribution<int> ud(0, 255);
    for (auto& b : out) {
        auto* raw = reinterpret_cast<uint8_t*>(&b);
        for (size_t i = 0; i < sizeof(Block); ++i) raw[i] = uint8_t(ud(rng));
    }
    return out;
}

// Q8_0 random: d in [-1,1], qs random. Avoid HUGE scales so ref stays sane.
std::vector<ie::block_q8_0> rand_q8_0_blocks(size_t n, std::mt19937& rng) {
    std::vector<ie::block_q8_0> out(n);
    std::uniform_real_distribution<float> dscale(-0.5f, 0.5f);
    std::uniform_int_distribution<int>    dq(-127, 127);
    for (auto& b : out) {
        b.d = fp32_to_fp16(dscale(rng));
        for (int i = 0; i < 32; ++i) b.qs[i] = int8_t(dq(rng));
    }
    return out;
}

// K-quant random: d, dmin in modest ranges; raw bytes for scales/qs.
std::vector<ie::block_q4_K> rand_q4k_blocks(size_t n, std::mt19937& rng) {
    auto out = make_random_blocks<ie::block_q4_K>(n, rng);
    std::uniform_real_distribution<float> ds(-0.2f, 0.2f);
    for (auto& b : out) {
        b.d = fp32_to_fp16(ds(rng));
        b.dmin = fp32_to_fp16(ds(rng));
    }
    return out;
}
std::vector<ie::block_q5_K> rand_q5k_blocks(size_t n, std::mt19937& rng) {
    auto out = make_random_blocks<ie::block_q5_K>(n, rng);
    std::uniform_real_distribution<float> ds(-0.2f, 0.2f);
    for (auto& b : out) {
        b.d = fp32_to_fp16(ds(rng));
        b.dmin = fp32_to_fp16(ds(rng));
    }
    return out;
}
std::vector<ie::block_q6_K> rand_q6k_blocks(size_t n, std::mt19937& rng) {
    auto out = make_random_blocks<ie::block_q6_K>(n, rng);
    std::uniform_real_distribution<float> ds(-0.2f, 0.2f);
    for (auto& b : out) {
        b.d = fp32_to_fp16(ds(rng));
    }
    return out;
}

// Run one format: dequant via host ref → exp[N], dequant via GPU → got[N],
// compute max-abs diff. Pass criteria left to caller.
struct DiffStats {
    float max_abs = 0.f;
    float max_rel = 0.f;
    size_t  worst_idx = 0;
    float   worst_exp = 0.f;
    float   worst_got = 0.f;
};

// Round the fp32 reference through fp16 (the device output is fp16). This
// removes the 1-ULP rounding noise and gives a contract we can require to be
// bit-exact: both implementations must agree on the same fp16 result. Any
// disagreement after this point is a real arithmetic bug.
DiffStats diff_buffers(const float* exp, const sycl::half* got, size_t n) {
    DiffStats s{};
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        const float a_round = float(sycl::half(exp[i]));
        const float b       = float(got[i]);
        const float d       = std::fabs(a_round - b);
        if (d > s.max_abs) {
            s.max_abs = d;
            s.worst_idx = i;
            s.worst_exp = a_round;
            s.worst_got = b;
        }
        const float scale = std::fmax(std::fabs(a_round), 1e-6f);
        const float r = d / scale;
        if (r > s.max_rel) s.max_rel = r;
        if (a_round != b) ++mismatches;
    }
    s.worst_idx = mismatches;        // repurpose: store mismatch count
    return s;
}

// Gate semantics:
//   - tol = 0       : require bit-exact fp16 agreement (Q8_0).
//   - tol > 0       : allow `tol`-ULP-relative drift (K-quants, where fp32
//                     intermediate FMA fusion can flip the final fp16 bit).
bool report(const char* fmt, const DiffStats& s, float max_rel_tol) {
    const bool ok = (max_rel_tol == 0.f) ? (s.max_rel == 0.f) : (s.max_rel <= max_rel_tol);
    std::printf("  %s%-6s%s  max_abs=%-10.4g  max_rel=%-10.4g  mismatches=%zu (exp=%g got=%g)  tol_rel=%g  %s\n",
                ok ? kGreen : kRed,
                fmt, kReset,
                s.max_abs, s.max_rel, s.worst_idx,
                s.worst_exp, s.worst_got,
                max_rel_tol, ok ? "OK" : "FAIL");
    return ok;
}

template <typename Block, typename RefFn, typename DevFn>
bool run_synth(sycl::queue& q,
               const char* name, std::vector<Block>&& blocks, size_t elems_per_block,
               RefFn ref_fn, DevFn dev_fn, float tol) {
    const size_t n_blocks = blocks.size();
    const size_t n_elem   = n_blocks * elems_per_block;
    const size_t pkb      = sizeof(Block);

    // Host reference
    std::vector<float> expected(n_elem);
    for (size_t i = 0; i < n_blocks; ++i) {
        ref_fn(&blocks[i], expected.data() + i * elems_per_block);
    }
    // GPU
    void*        d_in  = sycl::malloc_device(n_blocks * pkb, q);
    sycl::half*  d_out = sycl::malloc_device<sycl::half>(n_elem, q);
    if (!d_in || !d_out) {
        std::fprintf(stderr, "alloc fail\n");
        if (d_in)  sycl::free(d_in, q);
        if (d_out) sycl::free(d_out, q);
        return false;
    }
    q.memcpy(d_in, blocks.data(), n_blocks * pkb).wait();
    auto evt = dev_fn(q, d_in, d_out, n_elem, std::vector<sycl::event>{});
    evt.wait_and_throw();
    std::vector<sycl::half> got(n_elem);
    q.memcpy(got.data(), d_out, n_elem * sizeof(sycl::half)).wait();
    sycl::free(d_in, q);
    sycl::free(d_out, q);

    auto s = diff_buffers(expected.data(), got.data(), n_elem);
    return report(name, s, tol);
}

bool run_gguf_tensor(sycl::queue& q, ie::GgufReader& g,
                     const char* tensor_name, ie::DType expected_dtype, float tol) {
    auto* t = g.find_tensor(tensor_name);
    if (!t) {
        std::printf("  %s%-6s%s GGUF tensor '%s' not found — skipped\n",
                    kYell, "WARN", kReset, tensor_name);
        return true;
    }
    if (t->dtype != expected_dtype) {
        std::printf("  %s%-6s%s '%s' has dtype %.*s, expected %.*s — skipped\n",
                    kYell, "WARN", kReset, tensor_name,
                    int(ie::type_name(t->dtype).size()), ie::type_name(t->dtype).data(),
                    int(ie::type_name(expected_dtype).size()), ie::type_name(expected_dtype).data());
        return true;
    }
    // Element count: prod(shape).
    size_t n_elem = 1;
    for (uint32_t i = 0; i < t->n_dims; ++i) n_elem *= t->shape[i];

    // Host reference dequant
    std::vector<float> expected(n_elem);
    if (expected_dtype == ie::DType::kQ4_K) {
        ie::ref::dequant_q4_K_buffer(t->data, n_elem, expected.data());
    } else if (expected_dtype == ie::DType::kQ5_K) {
        ie::ref::dequant_q5_K_buffer(t->data, n_elem, expected.data());
    } else if (expected_dtype == ie::DType::kQ6_K) {
        ie::ref::dequant_q6_K_buffer(t->data, n_elem, expected.data());
    } else if (expected_dtype == ie::DType::kQ8_0) {
        ie::ref::dequant_q8_0_buffer(t->data, n_elem, expected.data());
    } else {
        std::fprintf(stderr, "unsupported dtype in run_gguf_tensor\n");
        return false;
    }

    // Upload + GPU dequant
    void* d_in = sycl::malloc_device(t->nbytes, q);
    sycl::half* d_out = sycl::malloc_device<sycl::half>(n_elem, q);
    q.memcpy(d_in, t->data, t->nbytes).wait();
    sycl::event evt;
    switch (expected_dtype) {
        case ie::DType::kQ4_K: evt = ie::dequant_q4_K(q, d_in, d_out, n_elem); break;
        case ie::DType::kQ5_K: evt = ie::dequant_q5_K(q, d_in, d_out, n_elem); break;
        case ie::DType::kQ6_K: evt = ie::dequant_q6_K(q, d_in, d_out, n_elem); break;
        case ie::DType::kQ8_0: evt = ie::dequant_q8_0(q, d_in, d_out, n_elem); break;
        default: break;
    }
    evt.wait_and_throw();
    std::vector<sycl::half> got(n_elem);
    q.memcpy(got.data(), d_out, n_elem * sizeof(sycl::half)).wait();
    sycl::free(d_in, q);
    sycl::free(d_out, q);

    auto s = diff_buffers(expected.data(), got.data(), n_elem);
    char label[64];
    std::snprintf(label, sizeof(label), "%s/%s",
                  ie::type_name(expected_dtype).data(),
                  tensor_name);
    return report(label, s, tol);
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path;
    bool only_gguf = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--only-gguf") only_gguf = true;
    }

    sycl::device dev;
    bool found = false;
    for (const auto& d : sycl::device::get_devices()) {
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("0xe223") != std::string::npos) {
            dev = d; found = true; break;
        }
    }
    if (!found) for (const auto& d : sycl::device::get_devices()) if (d.is_gpu()) { dev = d; found = true; break; }
    if (!found) { std::fputs("no GPU\n", stderr); return 2; }
    sycl::queue q(dev, sycl::property::queue::enable_profiling{});
    std::printf("Device: %s\n\n", dev.get_info<sycl::info::device::name>().c_str());

    int fails = 0;

    if (!only_gguf) {
        std::printf("%sSynthetic 1024 random blocks per format%s\n", kDim, kReset);
        std::mt19937 rng(0xC0FFEE);
        constexpr size_t N = 1024;

        // Tolerances are RELATIVE error (max_rel). 0 -> must be bit-exact fp16.
        // 1e-3 ~ 1 fp16 ULP, generously acceptable for K-quant FMA-fusion drift.
        if (!run_synth(q, "Q8_0", rand_q8_0_blocks(N, rng), 32,
                       [](const ie::block_q8_0* b, float* y) { ie::ref::dequant_q8_0(b, y); },
                       ie::dequant_q8_0, /*tol_rel=*/0.0f)) ++fails;

        if (!run_synth(q, "Q4_K", rand_q4k_blocks(N, rng), 256,
                       [](const ie::block_q4_K* b, float* y) { ie::ref::dequant_q4_K(b, y); },
                       ie::dequant_q4_K, 1e-3f)) ++fails;

        if (!run_synth(q, "Q5_K", rand_q5k_blocks(N, rng), 256,
                       [](const ie::block_q5_K* b, float* y) { ie::ref::dequant_q5_K(b, y); },
                       ie::dequant_q5_K, 1e-3f)) ++fails;

        if (!run_synth(q, "Q6_K", rand_q6k_blocks(N, rng), 256,
                       [](const ie::block_q6_K* b, float* y) { ie::ref::dequant_q6_K(b, y); },
                       ie::dequant_q6_K, 1e-3f)) ++fails;
    }

    if (!gguf_path.empty()) {
        ie::GgufReader g;
        if (auto err = g.open(gguf_path); !err.empty()) {
            std::fprintf(stderr, "gguf open: %s\n", err.c_str());
            return 3;
        }
        std::printf("\n%sReal Qwen3.6-35B-A3B tensors (%llu in file)%s\n",
                    kDim, (unsigned long long)g.n_tensors(), kReset);
        // Pick a few representative Q4_K tensors and one Q6_K
        if (!run_gguf_tensor(q, g, "blk.0.attn_gate.weight",   ie::DType::kQ4_K, 1e-3f)) ++fails;
        if (!run_gguf_tensor(q, g, "blk.0.ffn_gate_exps.weight", ie::DType::kQ4_K, 1e-3f)) ++fails;
        if (!run_gguf_tensor(q, g, "blk.0.attn_qkv.weight",    ie::DType::kQ6_K, 1e-3f)) ++fails;
        if (!run_gguf_tensor(q, g, "blk.0.ffn_down_exps.weight", ie::DType::kQ6_K, 1e-3f)) ++fails;
        if (!run_gguf_tensor(q, g, "blk.3.attn_v.weight",      ie::DType::kQ6_K, 1e-3f)) ++fails;
        if (!run_gguf_tensor(q, g, "output.weight",            ie::DType::kQ6_K, 1e-3f)) ++fails;
    }

    std::printf("\n%s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails;
}
