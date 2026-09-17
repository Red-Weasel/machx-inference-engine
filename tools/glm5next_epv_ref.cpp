// glm5next_epv_ref — CPU ground-truth arbiter for the EP verify oracle.
//
// Recomputes the dumped layer's peer-half MoE contribution from the mmap'd
// bank bytes (disk-verified) entirely on the CPU, with the same f16
// roundings at the stage boundaries the GPU path applies (gate/up outputs,
// swiglu output, down output), then reports max|gt-pa| (peer partial) and
// max|gt-pb| (owner recompute). fp64 accumulators; residual fp noise is
// ~1e-3-scale, far below the disputed 0.1-0.35 diffs — whichever side sits
// near gt is right.
//
// usage: ie-glm5next-epv-ref <model.gguf> <dumpdir>   (IE_G5_EPV_DUMP output)

#include "ie/gguf.hpp"
#include "ie/quant_blocks.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ie;

static float f16f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float v;
    if (e == 0) v = std::ldexp(float(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(float(m + 1024), int(e) - 25);
    return s ? -v : v;
}
static uint16_t ff16(float f) {   // round-to-nearest-even float -> f16
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000;
    int e = int((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e >= 31) return uint16_t(s | 0x7C00);
    if (e <= 0) {
        if (e < -10) return uint16_t(s);
        m |= 0x800000;
        const int sh = 14 - e;
        uint32_t r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return uint16_t(s | r);
    }
    uint32_t r = m >> 13, rem = m & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;
    uint32_t out = (uint32_t(e) << 10) + r;   // mantissa carry bumps exponent
    return uint16_t(s | out);
}
static float r16(float f) { return f16f(ff16(f)); }   // round through f16

static void scale_min_k4(int sub, const uint8_t* sc, uint8_t& s, uint8_t& m) {
    if (sub < 4) { s = sc[sub] & 0x3F; m = sc[sub + 4] & 0x3F; }
    else {
        s = (sc[sub + 4] & 0x0F) | ((sc[sub - 4] >> 6) << 4);
        m = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4);
    }
}
// y[n] += dot of x[0..K) with column n of a Q4_K/Q5_K/Q6_K [K,N] slab.
static double col_dot(const uint8_t* col, DType dt, const float* x, uint32_t K) {
    const uint32_t bpc = K / 256;
    double acc = 0;
    for (uint32_t b = 0; b < bpc; ++b) {
        if (dt == DType::kQ4_K) {
            const auto& blk = reinterpret_cast<const block_q4_K*>(col)[b];
            const float d = f16f(blk.d), dm = f16f(blk.dmin);
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t s, m;
                scale_min_k4(sub, blk.scales, s, m);
                const float ds = d * s, ms = dm * m;
                const int g = sub >> 1, hi = sub & 1;
                for (int j = 0; j < 32; ++j) {
                    const int q = (blk.qs[g * 32 + j] >> (hi * 4)) & 0xF;
                    acc += double(x[b * 256 + g * 64 + hi * 32 + j]) * (ds * q - ms);
                }
            }
        } else if (dt == DType::kQ5_K) {
            const auto& blk = reinterpret_cast<const block_q5_K*>(col)[b];
            const float d = f16f(blk.d), dm = f16f(blk.dmin);
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t s, m;
                scale_min_k4(sub, blk.scales, s, m);
                const float ds = d * s, ms = dm * m;
                const int g = sub >> 1, hi = sub & 1;
                for (int j = 0; j < 32; ++j) {
                    const int q = ((blk.qs[g * 32 + j] >> (hi * 4)) & 0xF) +
                                  ((blk.qh[j] >> sub) & 1) * 16;
                    acc += double(x[b * 256 + g * 64 + hi * 32 + j]) * (ds * q - ms);
                }
            }
        } else {   // kQ6_K — kernel lane lattice iterated serially
            const auto& blk = reinterpret_cast<const block_q6_K*>(col)[b];
            const float d = f16f(blk.d);
            for (int lane = 0; lane < 16; ++lane) {
                const int half = lane >> 3, sub = (lane >> 1) & 3, lh = lane & 1;
                const int ls = lh * 16;
                const int ql_off = half * 64 + (sub & 1) * 32 + ls;
                const int qh_off = half * 32 + ls;
                const float ds = d * blk.scales[half * 8 + sub * 2 + lh];
                const int qh_sh = sub * 2, ql_sh = (sub & 2) ? 4 : 0;
                const int out = half * 128 + sub * 32 + ls;
                for (int i = 0; i < 16; ++i) {
                    const int q = ((blk.ql[ql_off + i] >> ql_sh) & 0xF) |
                                  (((blk.qh[qh_off + i] >> qh_sh) & 3) << 4);
                    acc += double(x[b * 256 + out + i]) * (ds * (q - 32));
                }
            }
        }
    }
    return acc;
}

static std::vector<uint8_t> slurp(const std::string& p) {
    std::vector<uint8_t> v;
    if (FILE* f = std::fopen(p.c_str(), "rb")) {
        std::fseek(f, 0, SEEK_END);
        v.resize(size_t(std::ftell(f)));
        std::fseek(f, 0, SEEK_SET);
        if (std::fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
        std::fclose(f);
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s <gguf> <dumpdir>\n", argv[0]); return 2; }
    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    const std::string dir = argv[2];
    uint32_t L = 0, T = 0, H = 0, EF = 0;
    size_t n_exp = 0;
    double clampd = 0;
    {
        FILE* f = std::fopen((dir + "/meta.txt").c_str(), "r");
        if (!f || std::fscanf(f, "L %u T %u H %u EF %u n_exp %zu clamp %lf",
                              &L, &T, &H, &EF, &n_exp, &clampd) != 6) {
            std::fprintf(stderr, "bad meta.txt\n");
            return 1;
        }
        std::fclose(f);
    }
    const float clamp = float(clampd);
    if (T != 1) { std::fprintf(stderr, "arbiter handles T==1 dumps only\n"); return 1; }
    auto xb = slurp(dir + "/x.f16");
    auto pa = slurp(dir + "/pa.f32");
    auto pb = slurp(dir + "/pb.f32");
    auto el = slurp(dir + "/experts.i32");
    auto ew = slurp(dir + "/weights.f32");
    if (xb.size() != H * 2 || pa.size() != H * 4 || pb.size() != H * 4 ||
        el.size() != n_exp * 4 || ew.size() != n_exp * 4) {
        std::fprintf(stderr, "dump size mismatch\n");
        return 1;
    }
    std::vector<float> x(H);
    for (uint32_t k = 0; k < H; ++k)
        x[k] = f16f(reinterpret_cast<const uint16_t*>(xb.data())[k]);

    char nm[64];
    std::snprintf(nm, sizeof(nm), "blk.%u.ffn_gate_exps.weight", L);
    const GgufTensorInfo* tg = g.find_tensor(nm);
    std::snprintf(nm, sizeof(nm), "blk.%u.ffn_up_exps.weight", L);
    const GgufTensorInfo* tu = g.find_tensor(nm);
    std::snprintf(nm, sizeof(nm), "blk.%u.ffn_down_exps.weight", L);
    const GgufTensorInfo* td = g.find_tensor(nm);
    if (!tg || !tu || !td) { std::fprintf(stderr, "bank tensors missing\n"); return 1; }
    const uint32_t E = 288;
    const uint64_t gsl = tg->nbytes / E, usl = tu->nbytes / E, dsl = td->nbytes / E;
    const uint64_t gcol = gsl / EF, ucol = usl / EF, dcol = dsl / H;

    std::vector<double> gt(H, 0.0);
    std::vector<float> yg(EF), yu(EF);
    for (size_t i = 0; i < n_exp; ++i) {
        const uint32_t e = uint32_t(reinterpret_cast<const int32_t*>(el.data())[i]);
        const float w = reinterpret_cast<const float*>(ew.data())[i];
        const uint8_t* gb = static_cast<const uint8_t*>(tg->data) + uint64_t(e) * gsl;
        const uint8_t* ub = static_cast<const uint8_t*>(tu->data) + uint64_t(e) * usl;
        const uint8_t* db = static_cast<const uint8_t*>(td->data) + uint64_t(e) * dsl;
        for (uint32_t n = 0; n < EF; ++n) {
            yg[n] = r16(float(col_dot(gb + uint64_t(n) * gcol, tg->dtype, x.data(), H)));
            yu[n] = r16(float(col_dot(ub + uint64_t(n) * ucol, tu->dtype, x.data(), H)));
        }
        std::vector<float> ys(EF);
        for (uint32_t n = 0; n < EF; ++n) {
            const float gg = std::fmin(yg[n], clamp);
            const float uu = std::fmin(std::fmax(yu[n], -clamp), clamp);
            ys[n] = r16((gg / (1.0f + std::exp(-gg))) * uu);
        }
        for (uint32_t h = 0; h < H; ++h) {
            const float yd = r16(float(col_dot(db + uint64_t(h) * dcol, td->dtype, ys.data(), EF)));
            gt[h] += double(w) * double(yd);
        }
        std::fprintf(stderr, "  expert %u done\n", e);
    }
    double mxa = 0, mxb = 0;
    for (uint32_t h = 0; h < H; ++h) {
        mxa = std::max(mxa, std::abs(gt[h] - double(reinterpret_cast<const float*>(pa.data())[h])));
        mxb = std::max(mxb, std::abs(gt[h] - double(reinterpret_cast<const float*>(pb.data())[h])));
    }
    std::printf("[epv-ref] L%u n_exp %zu  max|gt-pa| %.6f (peer)  max|gt-pb| %.6f (owner recompute)\n",
                L, n_exp, mxa, mxb);
    // 10x separation = a clear culprit; both small = noise; both large = the
    // arbiter's assumptions need checking.
    std::printf("[epv-ref] verdict: %s\n",
                mxa < 0.005 && mxb < 0.005      ? "BOTH within noise" :
                mxb > mxa * 10.0                ? "PEER RIGHT — owner recompute corrupt" :
                mxa > mxb * 10.0                ? "OWNER RIGHT — peer partial corrupt" :
                                                  "AMBIGUOUS — check arbiter assumptions");
    return 0;
}
