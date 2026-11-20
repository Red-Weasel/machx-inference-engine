// tools/hy4flash_quant.cpp -- ie-hy4flash-quant: the Hy4-flash routed experts at a lower bit width, on the card.
//
//   ie-hy4flash-quant <ie_dir> <calib_prefix> <out_dir> --bits G,D [--layers 1,2,..|all] [--flips N] [--damp 0.01]
//                     [--layout native|q8] [--chunk E] [--card C]
//
// Source: <ie_dir>/experts-q8-LNN.bin (the 8-bit pack, dequantized; its own error is ~1/10 of a 5-bit rounding).
// Statistics: <calib_prefix>.xtx.f64 / .json (ie-hy4flash-run --cov-out --cov-inputs-only): per expert layer the
// summed outer products of the experts' latent inputs.
//
// Per layer, gate / up (whose input statistics are known):
//   1. scales: one fp16 per 32 weights, the clip of max|w| (1.0 .. 0.64) with the least squared rounding error
//   2. GPTQ: columns in order, each column's rounding error pushed onto the later columns along the upper Cholesky
//      factor of the inverse input covariance (the arithmetic of scripts/hy4flash_oracle/pack_lowbit_trial.py)
//   3. flips (MACaQ's refinement, arXiv 2611.10649): per row, the move of one code to a neighbouring level that lowers
//      the row's output error r H r^T the most, repeated up to N times or until no move helps (r = w - w_hat)
// down (no recorded input statistics): the same scales, round to nearest.
// Printed per layer: the gate / up output error sqrt(sum r H r^T / sum w H w^T) after GPTQ and after the flips.
//
// --layout native: <out_dir>/experts-q<G><D>-LNN.bin (Hy4fQFmt: 5 bits = nibbles [K/2] + a high-bit word per 32 [K/8],
//   q + 16; 4 bits = nibbles, q + 8; 8 bits = int8), the layout ie-hy4flash-run --qbits G,D reads.
// --layout q8: <out_dir>/experts-q8-LNN.bin in the 8-bit layout with the narrow integers (the 8-bit path computes the
//   same products -- any per-layer mix of widths can be measured without a native kernel).
#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr uint32_t E = 512, MI = 1536, LAT = 1536, NL = 42;
constexpr float CLIPS[7] = {1.0f, 0.94f, 0.88f, 0.82f, 0.76f, 0.70f, 0.64f};

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

uint32_t row_bytes(uint32_t bits, uint32_t K) { return bits == 8 ? K : (bits == 5 ? K / 2 + K / 8 : K / 2); }

// U (upper, float) with H^-1 = U^T U, from H (double, n x n, symmetric positive definite).
// With J the reversal: J H J = L L^T (lower Cholesky) -> H = V V^T, V = J L J upper -> U = V^-1.
std::vector<float> chol_inv_upper(const std::vector<double>& H, uint32_t n) {
    std::vector<double> M(size_t(n) * n), Lw(size_t(n) * n, 0.0);
    for (uint32_t i = 0; i < n; ++i) for (uint32_t j = 0; j < n; ++j) M[size_t(i) * n + j] = H[size_t(n - 1 - i) * n + (n - 1 - j)];
    for (uint32_t j = 0; j < n; ++j) {                      // lower Cholesky of M, column by column
        double s = M[size_t(j) * n + j];
        for (uint32_t k = 0; k < j; ++k) s -= Lw[size_t(j) * n + k] * Lw[size_t(j) * n + k];
        const double djj = std::sqrt(s > 1e-300 ? s : 1e-300);
        Lw[size_t(j) * n + j] = djj;
        for (uint32_t i = j + 1; i < n; ++i) {
            double t = M[size_t(i) * n + j];
            const double* li = &Lw[size_t(i) * n]; const double* lj = &Lw[size_t(j) * n];
            for (uint32_t k = 0; k < j; ++k) t -= li[k] * lj[k];
            Lw[size_t(i) * n + j] = t / djj;
        }
    }
    std::vector<double> V(size_t(n) * n, 0.0);              // V = J L J (upper)
    for (uint32_t i = 0; i < n; ++i) for (uint32_t j = 0; j < n; ++j) V[size_t(i) * n + j] = Lw[size_t(n - 1 - i) * n + (n - 1 - j)];
    std::vector<double> Ui(size_t(n) * n, 0.0);             // U = V^-1, upper: back substitution column by column
    for (uint32_t c = 0; c < n; ++c) {
        Ui[size_t(c) * n + c] = 1.0 / V[size_t(c) * n + c];
        for (int i = int(c) - 1; i >= 0; --i) {
            double t = 0.0;
            for (uint32_t k = uint32_t(i) + 1; k <= c; ++k) t += V[size_t(i) * n + k] * Ui[size_t(k) * n + c];
            Ui[size_t(i) * n + c] = -t / V[size_t(i) * n + i];
        }
    }
    std::vector<float> U(size_t(n) * n);
    for (size_t i = 0; i < U.size(); ++i) U[i] = float(Ui[i]);
    return U;
}

// w [R, K] fp32 from the 8-bit slots of n experts (gate rows, up rows: R = n * 2 * MI, K = LAT; down: R = n * LAT, K = MI)
void dequant(sycl::queue& q, const uint8_t* slots, size_t slot, uint32_t n, float* wgu, float* wdn) {
    const size_t gq = size_t(MI) * LAT, dq = size_t(LAT) * MI, gd = size_t(MI) * (LAT / 32) * 2;
    q.parallel_for(sycl::range<2>(size_t(n) * 2 * MI, LAT), [=](sycl::id<2> i) {
        const size_t r = i[0]; const uint32_t k = uint32_t(i[1]);
        const size_t e = r / (2 * MI), rr = r % (2 * MI);
        const uint8_t* s = slots + e * slot;
        const int8_t qv = int8_t(s[rr * LAT + k]);                              // gate rows then up rows: contiguous
        const sycl::half* d = reinterpret_cast<const sycl::half*>(s + 2 * gq + dq);
        wgu[r * LAT + k] = float(qv) * float(d[rr * (LAT / 32) + k / 32]);
    });
    q.parallel_for(sycl::range<2>(size_t(n) * LAT, MI), [=](sycl::id<2> i) {
        const size_t r = i[0]; const uint32_t k = uint32_t(i[1]);
        const size_t e = r / LAT, rr = r % LAT;
        const uint8_t* s = slots + e * slot;
        const int8_t qv = int8_t(s[2 * gq + rr * MI + k]);
        const sycl::half* d = reinterpret_cast<const sycl::half*>(s + 2 * gq + dq + 2 * gd);
        wdn[r * MI + k] = float(qv) * float(d[rr * (MI / 32) + k / 32]);
    });
}

// d [R, K/32]: the fp16 scale with the least squared rounding error over the clips
void scales(sycl::queue& q, const float* w, float* d, size_t R, uint32_t K, int qmax) {
    q.parallel_for(sycl::range<1>(R * (K / 32)), [=](sycl::id<1> i) {
        const float* b = w + i[0] * 32;
        float amax = 0.f;
        for (int t = 0; t < 32; ++t) amax = sycl::fmax(amax, sycl::fabs(b[t]));
        float best = 0.f, best_err = 0.f;
        for (int c = 0; c < 7; ++c) {
            float dv = float(sycl::half(amax * (CLIPS[c] / float(qmax))));
            if (!(dv > 0.f)) dv = 1e-8f;
            float err = 0.f;
            for (int t = 0; t < 32; ++t) {
                const float qv = sycl::clamp(sycl::rint(b[t] / dv), float(-qmax), float(qmax));
                const float r = b[t] - qv * dv; err += r * r;
            }
            if (c == 0 || err < best_err) { best = dv; best_err = err; }
        }
        d[i[0]] = best;
    });
}

void round_plain(sycl::queue& q, const float* w, const float* d, int8_t* Q, size_t R, uint32_t K, int qmax) {
    q.parallel_for(sycl::range<1>(R * K), [=](sycl::id<1> i) {
        const size_t r = i[0] / K; const uint32_t k = uint32_t(i[0] % K);
        Q[i[0]] = int8_t(sycl::clamp(sycl::rint(w[i[0]] / d[r * (K / 32) + k / 32]), float(-qmax), float(qmax)));
    });
}

// GPTQ: 8 rows a work-group, the rows in local memory; per column the 8 errors, then the later columns updated
template <uint32_t K>
void round_gptq(sycl::queue& q, const float* w, const float* d, const float* U, int8_t* Q, size_t R, int qmax) {
    constexpr uint32_t RW = 8, WG = 256;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ws(sycl::range<1>(RW * K), h), es(sycl::range<1>(RW), h);
        h.parallel_for(sycl::nd_range<1>((R / RW) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t lid = uint32_t(it.get_local_id(0)); const size_t r0 = it.get_group(0) * RW;
            for (uint32_t x = lid; x < RW * K; x += WG) ws[x] = w[r0 * K + x];
            sycl::group_barrier(it.get_group());
            for (uint32_t j = 0; j < K; ++j) {
                if (lid < RW) {
                    const float dj = d[(r0 + lid) * (K / 32) + j / 32];
                    const float v = ws[lid * K + j];
                    const float qv = sycl::clamp(sycl::rint(v / dj), float(-qmax), float(qmax));
                    Q[(r0 + lid) * K + j] = int8_t(qv);
                    es[lid] = (v - qv * dj) / U[size_t(j) * K + j];
                }
                sycl::group_barrier(it.get_group());
                for (uint32_t k = j + 1 + lid; k < K; k += WG) {
                    const float u = U[size_t(j) * K + k];
                    #pragma unroll
                    for (uint32_t r = 0; r < RW; ++r) ws[r * K + k] -= es[r] * u;
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

// The flips, one work-group a row: g = H r, then up to F best single-code moves; accumulates sum r H r^T (before and
// after) and sum w H w^T. H here: the covariance (fp32, symmetric, K x K).
template <uint32_t K>
void refine(sycl::queue& q, const float* w, const float* d, const float* H, int8_t* Q, size_t R, int qmax, uint32_t F,
            float* acc /* [R, 3]: rHr before, rHr after, wHw, per row */) {
    constexpr uint32_t WG = 256, PL = K / WG;
    static_assert(K % WG == 0);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> rs(sycl::range<1>(K), h), ws(sycl::range<1>(K), h);
        h.parallel_for(sycl::nd_range<1>(R * WG, WG), [=](sycl::nd_item<1> it) {
            auto g = it.get_group(); const uint32_t lid = uint32_t(it.get_local_id(0)); const size_t row = it.get_group(0);
            const float* wr = w + row * K; int8_t* qr = Q + row * K; const float* dr = d + row * (K / 32);
            for (uint32_t k = lid; k < K; k += WG) { ws[k] = wr[k]; rs[k] = wr[k] - float(qr[k]) * dr[k / 32]; }
            sycl::group_barrier(g);
            float gv[PL], rv[PL]; int qv[PL]; float dv[PL], hd[PL];
            float rhr = 0.f, whw = 0.f, bv[PL];
            #pragma unroll
            for (uint32_t m = 0; m < PL; ++m) { gv[m] = 0.f; bv[m] = 0.f; }
            for (uint32_t j = 0; j < K; ++j) {               // g = H r, H symmetric: row j's entries for this lane's k (coalesced)
                const float rj = rs[j], wj = ws[j]; const float* hj = H + size_t(j) * K;
                #pragma unroll
                for (uint32_t m = 0; m < PL; ++m) { const float hv = hj[lid + m * WG]; gv[m] += hv * rj; bv[m] += hv * wj; }
            }
            #pragma unroll
            for (uint32_t m = 0; m < PL; ++m) {
                const uint32_t k = lid + m * WG;
                rv[m] = rs[k]; qv[m] = qr[k]; dv[m] = dr[k / 32]; hd[m] = H[size_t(k) * K + k];
                rhr += rv[m] * gv[m]; whw += ws[k] * bv[m];
            }
            const float rhr0 = sycl::reduce_over_group(g, rhr, sycl::plus<float>());
            const float whw0 = sycl::reduce_over_group(g, whw, sycl::plus<float>());
            float gain_sum = 0.f;
            for (uint32_t f = 0; f < F; ++f) {
                float best = 0.f; uint32_t bk = 0xffffffffu; int bs = 0;
                #pragma unroll
                for (uint32_t m = 0; m < PL; ++m) {
                    #pragma unroll
                    for (int s = -1; s <= 1; s += 2) {
                        if (qv[m] + s < -qmax || qv[m] + s > qmax) continue;
                        const float del = float(s) * dv[m];
                        const float gn = 2.f * del * gv[m] - del * del * hd[m];
                        if (gn > best) { best = gn; bk = lid + m * WG; bs = s; }
                    }
                }
                const float gbest = sycl::reduce_over_group(g, best, sycl::maximum<float>());
                if (!(gbest > 0.f)) break;
                const uint32_t mine = (best == gbest) ? bk : 0xffffffffu;
                const uint32_t kk = sycl::reduce_over_group(g, mine, sycl::minimum<uint32_t>());
                const int ss = sycl::group_broadcast(g, bs, kk % WG);
                const float del = float(ss) * dr[kk / 32];
                if (lid == kk % WG) { const uint32_t m = kk / WG; qv[m] += ss; rv[m] -= del; }
                const float* hk = H + size_t(kk) * K;
                #pragma unroll
                for (uint32_t m = 0; m < PL; ++m) gv[m] -= del * hk[lid + m * WG];
                gain_sum += gbest;
            }
            #pragma unroll
            for (uint32_t m = 0; m < PL; ++m) qr[lid + m * WG] = int8_t(qv[m]);
            if (lid == 0) { acc[row * 3] = rhr0; acc[row * 3 + 1] = rhr0 - gain_sum; acc[row * 3 + 2] = whw0; }
        });
    });
}

void pack_row(const int8_t* q, uint32_t K, uint32_t bits, uint8_t* out) {
    if (bits == 8) { std::memcpy(out, q, K); return; }
    const int off = bits == 5 ? 16 : 8;
    for (uint32_t k = 0; k < K; k += 2) out[k / 2] = uint8_t(((q[k] + off) & 15) | (((q[k + 1] + off) & 15) << 4));
    if (bits == 5) {
        uint8_t* hb = out + K / 2;
        for (uint32_t k = 0; k < K; k += 8) {
            uint8_t v = 0;
            for (uint32_t t = 0; t < 8; ++t) v |= uint8_t((((q[k + t] + off) >> 4) & 1) << t);
            hb[k / 8] = v;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: ie-hy4flash-quant <ie_dir> <calib_prefix> <out_dir> --bits G,D [--layers a,b|all] [--flips N] [--damp X] [--layout native|q8] [--chunk E] [--card C]\n"); return 2; }
    const std::string ie = argv[1], cal = argv[2], out = argv[3];
    uint32_t bg = 5, bd = 5, flips = 64, chunk = 32, card = 0; double damp = 0.01; std::string layers_s = "all", layout = "native"; bool rtn = false;   // --rtn: gate / up round to nearest (the baseline the error is compared with)
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i]; auto val = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--bits") { const std::string v = val(); bg = uint32_t(std::atoi(v.c_str())); const size_t c = v.find(','); bd = c == std::string::npos ? bg : uint32_t(std::atoi(v.c_str() + c + 1)); }
        else if (a == "--layers") layers_s = val();
        else if (a == "--flips") flips = uint32_t(std::atoi(val().c_str()));
        else if (a == "--damp") damp = std::atof(val().c_str());
        else if (a == "--layout") layout = val();
        else if (a == "--chunk") chunk = uint32_t(std::atoi(val().c_str()));
        else if (a == "--rtn") rtn = true;
        else if (a == "--card") card = uint32_t(std::atoi(val().c_str()));
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    auto okb = [](uint32_t b) { return b == 4 || b == 5 || b == 8; };
    if (!okb(bg) || !okb(bd) || E % chunk) { std::fprintf(stderr, "bits must be 4, 5 or 8; the chunk must divide 512\n"); return 2; }
    std::vector<uint32_t> layers;
    if (layers_s == "all") for (uint32_t L = 1; L < NL; ++L) layers.push_back(L);
    else for (size_t p = 0; p < layers_s.size();) { layers.push_back(uint32_t(std::atoi(layers_s.c_str() + p))); const size_t c = layers_s.find(',', p); if (c == std::string::npos) break; p = c + 1; }

    // the statistics: one token count per expert layer (layer L -> index L - 1: layer 0 is dense)
    std::ifstream js(cal + ".json"); std::string meta((std::istreambuf_iterator<char>(js)), {});
    std::vector<double> ntok;
    { const size_t a = meta.find("\"tokens\""), b = meta.find('[', a), c = meta.find(']', b);
      for (size_t p = b + 1; p < c;) { ntok.push_back(std::atof(meta.c_str() + p)); const size_t q2 = meta.find(',', p); if (q2 == std::string::npos || q2 > c) break; p = q2 + 1; } }
    if (ntok.size() != NL - 1) { std::fprintf(stderr, "%s.json: expected %u token counts, found %zu\n", cal.c_str(), NL - 1, ntok.size()); return 1; }
    const int xfd = open((cal + ".xtx.f64").c_str(), O_RDONLY); if (xfd < 0) { std::perror("xtx"); return 1; }
    const double* xtx = static_cast<const double*>(mmap(nullptr, size_t(NL - 1) * LAT * LAT * 8, PROT_READ, MAP_SHARED, xfd, 0));

    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (card >= gpus.size()) { std::fprintf(stderr, "no card %u\n", card); return 1; }
    sycl::queue q{gpus[card], sycl::property::queue::in_order{}};
    std::printf("card %u: %s; gate / up %u bits (GPTQ + up to %u flips a row), down %u bits (nearest); layout %s\n", card,
                q.get_device().get_info<sycl::info::device::name>().c_str(), bg, flips, bd, layout.c_str());

    const size_t gq8 = size_t(MI) * LAT, dq8 = size_t(LAT) * MI, gd = size_t(MI) * (LAT / 32) * 2, dd = size_t(LAT) * (MI / 32) * 2;
    const size_t slot8 = (2 * gq8 + dq8 + 2 * gd + dd + 63) / 64 * 64;
    const bool native = layout == "native";
    const uint32_t ob_g = native ? bg : 8, ob_d = native ? bd : 8;
    const size_t gqo = size_t(MI) * row_bytes(ob_g, LAT), dqo = size_t(LAT) * row_bytes(ob_d, MI);
    const size_t sloto = (2 * gqo + dqo + 2 * gd + dd + 63) / 64 * 64;
    const int qg = (1 << (bg - 1)) - 1, qd = (1 << (bd - 1)) - 1;
    const size_t Rg = size_t(chunk) * 2 * MI, Rd = size_t(chunk) * LAT;

    uint8_t* d_slots = sycl::malloc_device<uint8_t>(chunk * slot8, q);
    float* wgu = sycl::malloc_device<float>(Rg * LAT, q); float* wdn = sycl::malloc_device<float>(Rd * MI, q);
    float* dgu = sycl::malloc_device<float>(Rg * (LAT / 32), q); float* ddn = sycl::malloc_device<float>(Rd * (MI / 32), q);
    int8_t* Qg = sycl::malloc_device<int8_t>(Rg * LAT, q); int8_t* Qd = sycl::malloc_device<int8_t>(Rd * MI, q);
    float* dU = sycl::malloc_device<float>(size_t(LAT) * LAT, q); float* dH = sycl::malloc_device<float>(size_t(LAT) * LAT, q);
    float* acc = sycl::malloc_device<float>(Rg * 3, q); std::vector<float> hacc(Rg * 3); double sum3[3];
    std::vector<int8_t> hQg(Rg * LAT), hQd(Rd * MI); std::vector<float> hdg(Rg * (LAT / 32)), hdd(Rd * (MI / 32));
    std::vector<uint8_t> obuf(chunk * sloto);
    const double t_all = now();
    for (const uint32_t L : layers) {
        const double t0 = now();
        char nm[64]; std::snprintf(nm, sizeof nm, "/experts-q8-L%02u.bin", L);
        const int fd = open((ie + nm).c_str(), O_RDONLY); if (fd < 0) { std::perror((ie + nm).c_str()); return 1; }
        struct stat st; fstat(fd, &st);
        if (size_t(st.st_size) != E * slot8) { std::fprintf(stderr, "%s: not the 8-bit layout\n", nm); return 1; }
        const uint8_t* src = static_cast<const uint8_t*>(mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0));
        // the statistics: H = XtX / N (dead dims 1), damped for the GPTQ factor
        const double* X = xtx + size_t(L - 1) * LAT * LAT; const double N = ntok[L - 1];
        std::vector<double> H(size_t(LAT) * LAT); double mdiag = 0.0;
        for (size_t i = 0; i < H.size(); ++i) H[i] = X[i] / N;
        for (uint32_t i = 0; i < LAT; ++i) { double& h = H[size_t(i) * LAT + i]; if (h <= 0) h = 1.0; mdiag += h; }
        mdiag /= LAT;
        std::vector<float> Hf(H.size()); for (size_t i = 0; i < H.size(); ++i) Hf[i] = float(H[i]);
        for (uint32_t i = 0; i < LAT; ++i) H[size_t(i) * LAT + i] += damp * mdiag;
        const std::vector<float> U = chol_inv_upper(H, LAT);
        q.memcpy(dU, U.data(), U.size() * 4); q.memcpy(dH, Hf.data(), Hf.size() * 4);
        sum3[0] = sum3[1] = sum3[2] = 0.0;
        const double t_f = now();
        char on[64];
        if (native && !(bg == 8 && bd == 8)) std::snprintf(on, sizeof on, "/experts-q%u%u-L%02u.bin", bg, bd, L);
        else std::snprintf(on, sizeof on, "/experts-q8-L%02u.bin", L);
        const std::string opath = out + on, tmp = opath + ".tmp";
        FILE* fo = std::fopen(tmp.c_str(), "wb"); if (!fo) { std::perror(tmp.c_str()); return 1; }
        for (uint32_t e0 = 0; e0 < E; e0 += chunk) {
            q.memcpy(d_slots, src + size_t(e0) * slot8, chunk * slot8);
            dequant(q, d_slots, slot8, chunk, wgu, wdn);
            scales(q, wgu, dgu, Rg, LAT, qg); scales(q, wdn, ddn, Rd, MI, qd);
            if (bg < 8 && rtn) { round_plain(q, wgu, dgu, Qg, Rg, LAT, qg); refine<LAT>(q, wgu, dgu, dH, Qg, Rg, qg, flips, acc); }
            else if (bg < 8) { round_gptq<LAT>(q, wgu, dgu, dU, Qg, Rg, qg); if (flips) refine<LAT>(q, wgu, dgu, dH, Qg, Rg, qg, flips, acc); }
            else round_plain(q, wgu, dgu, Qg, Rg, LAT, qg);
            if (bg < 8 && !flips && !rtn) refine<LAT>(q, wgu, dgu, dH, Qg, Rg, qg, 0, acc);          // (the error only)
            round_plain(q, wdn, ddn, Qd, Rd, MI, qd);
            q.memcpy(hQg.data(), Qg, hQg.size()); q.memcpy(hQd.data(), Qd, hQd.size());
            q.memcpy(hdg.data(), dgu, hdg.size() * 4); q.memcpy(hdd.data(), ddn, hdd.size() * 4);
            if (bg < 8) q.memcpy(hacc.data(), acc, hacc.size() * 4);
            q.wait();
            if (bg < 8) for (size_t r = 0; r < Rg; ++r) for (int t = 0; t < 3; ++t) sum3[t] += double(hacc[r * 3 + t]);
            std::fill(obuf.begin(), obuf.end(), 0);
            for (uint32_t e = 0; e < chunk; ++e) {
                uint8_t* s = obuf.data() + size_t(e) * sloto;
                const uint32_t rbg = row_bytes(ob_g, LAT), rbd = row_bytes(ob_d, MI);
                for (uint32_t r = 0; r < 2 * MI; ++r) pack_row(&hQg[(size_t(e) * 2 * MI + r) * LAT], LAT, ob_g, s + size_t(r) * rbg);
                for (uint32_t r = 0; r < LAT; ++r) pack_row(&hQd[(size_t(e) * LAT + r) * MI], MI, ob_d, s + 2 * gqo + size_t(r) * rbd);
                sycl::half* sd = reinterpret_cast<sycl::half*>(s + 2 * gqo + dqo);
                for (size_t x = 0; x < size_t(2 * MI) * (LAT / 32); ++x) sd[x] = sycl::half(hdg[size_t(e) * 2 * MI * (LAT / 32) + x]);
                sycl::half* sdd = reinterpret_cast<sycl::half*>(s + 2 * gqo + dqo + 2 * gd);
                for (size_t x = 0; x < size_t(LAT) * (MI / 32); ++x) sdd[x] = sycl::half(hdd[size_t(e) * LAT * (MI / 32) + x]);
            }
            if (std::fwrite(obuf.data(), 1, obuf.size(), fo) != obuf.size()) { std::perror("write"); return 1; }
        }
        std::fclose(fo); std::rename(tmp.c_str(), opath.c_str());
        munmap(const_cast<uint8_t*>(src), st.st_size); close(fd);
        if (bg < 8) std::printf("layer %u: gate / up output error %.3f %% after GPTQ, %.3f %% after the flips; %.1f s (factor %.1f s)\n", L,
                                100.0 * std::sqrt(sum3[0] / sum3[2]), 100.0 * std::sqrt(std::max(0.0, sum3[1]) / sum3[2]), now() - t0, t_f - t0);
        else std::printf("layer %u: %.1f s\n", L, now() - t0);
        std::fflush(stdout);
    }
    std::printf("done: %zu layers in %.1f s\n", layers.size(), now() - t_all);
    return 0;
}
