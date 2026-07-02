// tests/unit/exl3_decode_test.cpp — EXL3 host decode bit-matches the numpy oracle.
// Reads the git-ignored one-layer vectors (tests/data/exl3/onelayer.*, regenerate
// via tools/exl3/make_oracle.py — see tests/data/exl3/README.md). SKIPS gracefully
// (exit 0) when the binaries are absent so a fresh checkout's ctest still passes.
#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/exl3.hpp"

#include <sycl/sycl.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef EXL3_DATA_DIR
#define EXL3_DATA_DIR "tests/data/exl3"
#endif

namespace {

std::vector<uint8_t> read_bytes(const std::string& path, bool& ok) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { ok = false; return {}; }
    f.seekg(0, std::ios::end);
    const std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(size_t(n < 0 ? 0 : n));
    if (n > 0) f.read(reinterpret_cast<char*>(buf.data()), n);
    ok = bool(f);
    return buf;
}

// Minimal extractor for a flat "key": <int> field in the meta JSON.
long meta_int(const std::string& js, const char* key) {
    const std::string needle = std::string("\"") + key + "\"";
    auto p = js.find(needle);
    if (p == std::string::npos) return -1;
    p = js.find(':', p);
    if (p == std::string::npos) return -1;
    ++p;
    while (p < js.size() && (js[p] == ' ' || js[p] == '\t')) ++p;
    return std::strtol(js.c_str() + p, nullptr, 10);
}

float half_bits_to_float(uint16_t b) {
    sycl::half h;
    std::memcpy(&h, &b, sizeof h);
    return float(h);
}

}  // namespace

int main() {
    const std::string dir = EXL3_DATA_DIR;
    bool ok = true;
    const auto trellis_bytes = read_bytes(dir + "/onelayer.trellis", ok);
    bool ok2 = true;
    const auto wrot_bytes = read_bytes(dir + "/onelayer.wrot.f16", ok2);
    std::ifstream mf(dir + "/onelayer.meta.json");
    if (!ok || !ok2 || !mf) {
        std::printf("exl3_decode_test: SKIP (vectors absent — regenerate via "
                    "tools/exl3/make_oracle.py)\n");
        return 0;
    }
    std::string meta((std::istreambuf_iterator<char>(mf)),
                     std::istreambuf_iterator<char>());

    ie::Exl3Tensor t;
    t.K      = uint32_t(meta_int(meta, "K"));
    t.N      = uint32_t(meta_int(meta, "N"));
    t.bits   = uint32_t(meta_int(meta, "bits"));
    t.cb     = uint32_t(meta_int(meta, "cb"));
    t.tile_k = uint32_t(meta_int(meta, "tile_k"));
    t.tile_n = uint32_t(meta_int(meta, "tile_n"));
    assert(t.K && t.N && t.bits && t.tile_k && t.tile_n);

    // trellis: raw int16 → uint16.
    assert(trellis_bytes.size() == size_t(t.tile_k) * t.tile_n * 16 * t.bits * 2);
    t.trellis.resize(trellis_bytes.size() / 2);
    std::memcpy(t.trellis.data(), trellis_bytes.data(), trellis_bytes.size());

    // Reference W_rot (fp16) → float.
    assert(wrot_bytes.size() == size_t(t.K) * t.N * 2);
    const uint16_t* ref16 = reinterpret_cast<const uint16_t*>(wrot_bytes.data());

    std::vector<float> got(size_t(t.K) * t.N);
    const std::string err = ie::exl3_decode_host(t, got.data());
    assert(err.empty() && "exl3_decode_host returned an error");

    double maxerr = 0.0;
    size_t mism = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float r = half_bits_to_float(ref16[i]);
        const double e = std::fabs(double(got[i]) - double(r));
        if (e > maxerr) maxerr = e;
        if (e > 1e-3) ++mism;
    }
    std::printf("exl3_decode_test: K=%u N=%u bits=%u  maxerr=%.3e  mism(>1e-3)=%zu/%zu\n",
                t.K, t.N, t.bits, maxerr, mism, got.size());
    assert(maxerr < 1e-3 && "host decode does not match numpy oracle");
    std::puts("exl3_decode_test: all OK");
    return 0;
}
