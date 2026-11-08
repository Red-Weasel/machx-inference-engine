// tools/qwen4_route_test.cpp — the device-reproducible Flash-Next routing (include/ie/qwen4_route.hpp) gate.
//
//   1. qwen4_route_dev == qwen4_route_host bit for bit (ids and weight bits) on random F16 logit rows, a fifth of them with
//      exact ties forced at the k-th / (k+1)-th place, and on real router rows when a dump is given.
//   2. Against route_from_logits (the routing qwen4exp runs today; copied here): rows whose pick SET differs, and the
//      weights' max / mean absolute difference where the sets agree. This is the numerics change the device routing brings.
//   3. qwen4_route_jobs_dev's job table (qwen4exp's host layout, a random slot map) == the table built on the host from
//      qwen4_route_host, every word, at T = 1 / 3 / 4.
//   4. The kernel's time for one row and for four rows (the decode and verify shapes).
//
// usage: ie-qwen4-route-test [router_rows.f16 [E]]   (a dump: rows of E F16 logits, IE_Q4E_ROUTER_DUMP; E default 512)
// exit 0 on GATE PASSED (1 = device != host).
#include "ie/qwen4_route.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

using namespace ie;

namespace {

// COPY of qwen4exp.cpp:route_from_logits (the routing today).
void route_from_logits(const float* logit, uint32_t E, uint32_t k, std::vector<std::pair<uint32_t, float>>& out) {
    float m = logit[0];
    for (uint32_t e = 1; e < E; ++e) m = std::max(m, logit[e]);
    double sum = 0.0;
    std::vector<float> prob(E);
    for (uint32_t e = 0; e < E; ++e) { prob[e] = std::exp(logit[e] - m); sum += prob[e]; }
    for (uint32_t e = 0; e < E; ++e) prob[e] = float(prob[e] / sum);
    std::vector<uint32_t> idx(E);
    for (uint32_t e = 0; e < E; ++e) idx[e] = e;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](uint32_t a, uint32_t b) { return prob[a] > prob[b]; });
    float wsum = 0.f;
    for (uint32_t j = 0; j < k; ++j) wsum += prob[idx[j]];
    out.clear();
    for (uint32_t j = 0; j < k; ++j) out.emplace_back(idx[j], prob[idx[j]] / wsum);
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
}

struct Tally {
    uint64_t rows = 0, dev_diff = 0, set_diff = 0, set_diff_tie = 0, w_n = 0;
    double w_max = 0, w_sum = 0;
};

void run_rows(sycl::queue& q, const std::vector<sycl::half>& L, uint32_t E, uint32_t k, const char* what, Tally& t) {
    const uint32_t rows = uint32_t(L.size() / E);
    sycl::half* dl = sycl::malloc_device<sycl::half>(L.size(), q);
    int32_t* di = sycl::malloc_device<int32_t>(size_t(rows) * k, q);
    float* dw = sycl::malloc_device<float>(size_t(rows) * k, q);
    q.memcpy(dl, L.data(), L.size() * 2).wait();
    qwen4_route_dev(q, dl, rows, E, k, di, dw).wait();
    std::vector<int32_t> gi(size_t(rows) * k);
    std::vector<float> gw(size_t(rows) * k);
    q.memcpy(gi.data(), di, gi.size() * 4).wait();
    q.memcpy(gw.data(), dw, gw.size() * 4).wait();
    std::vector<int32_t> hi(k);
    std::vector<float> hw(k), lf(E);
    std::vector<std::pair<uint32_t, float>> old;
    for (uint32_t r = 0; r < rows; ++r) {
        const sycl::half* lr = L.data() + size_t(r) * E;
        qwen4_route_host(lr, E, k, hi.data(), hw.data());
        ++t.rows;
        if (std::memcmp(hi.data(), gi.data() + size_t(r) * k, k * 4) || std::memcmp(hw.data(), gw.data() + size_t(r) * k, k * 4)) {
            if (t.dev_diff < 3) {
                std::printf("    device != host at row %u:", r);
                for (uint32_t j = 0; j < k; ++j) std::printf(" %d/%d %.9g/%.9g", hi[j], gi[size_t(r) * k + j], hw[j], gw[size_t(r) * k + j]);
                std::printf("\n");
            }
            ++t.dev_diff;
        }
        for (uint32_t e = 0; e < E; ++e) lf[e] = float(lr[e]);
        route_from_logits(lf.data(), E, k, old);
        bool same = true;
        for (uint32_t j = 0; j < k; ++j) same = same && int32_t(old[j].first) == hi[j];
        if (!same) {
            ++t.set_diff;
            // is the difference an exact tie at the boundary? (the k-th and (k+1)-th largest logits equal)
            std::vector<float> s(lf);
            std::nth_element(s.begin(), s.begin() + k, s.end(), std::greater<float>());
            const float kth1 = s[k];
            std::nth_element(s.begin(), s.begin() + (k - 1), s.end(), std::greater<float>());
            if (s[k - 1] == kth1) ++t.set_diff_tie;
        } else {
            for (uint32_t j = 0; j < k; ++j) {
                const double d = std::fabs(double(old[j].second) - double(hw[j]));
                t.w_max = std::max(t.w_max, d);
                t.w_sum += d;
                ++t.w_n;
            }
        }
    }
    std::printf("  %-34s rows %7llu | device != host %llu | pick set != today's %llu (%llu at an exact tie) | weights vs today's: "
                "max %.2e mean %.2e\n", what, (unsigned long long)t.rows, (unsigned long long)t.dev_diff,
                (unsigned long long)t.set_diff, (unsigned long long)t.set_diff_tie, t.w_max, t.w_n ? t.w_sum / double(t.w_n) : 0.0);
    sycl::free(dl, q); sycl::free(di, q); sycl::free(dw, q);
}

}  // namespace

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v};
    const uint32_t k = 10;
    const uint32_t E = argc > 2 ? uint32_t(std::atoi(argv[2])) : 512;
    std::printf("\nqwen4exp device-reproducible routing gate  device: %s  E=%u k=%u\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), E, k);
    int fail = 0;

    // random rows: a router-like spread, a fifth with the k-th logit copied into another expert (an exact boundary tie),
    // and every row's values F16-rounded (as the router GEMV writes them)
    for (float sigma : {1.0f, 2.5f}) {
        std::mt19937 rng(1234 + int(sigma * 10));
        std::normal_distribution<float> nd(0.0f, sigma);
        const uint32_t rows = 20000;
        std::vector<sycl::half> L(size_t(rows) * E);
        for (uint32_t r = 0; r < rows; ++r) {
            sycl::half* lr = L.data() + size_t(r) * E;
            for (uint32_t e = 0; e < E; ++e) lr[e] = sycl::half(nd(rng));
            if (r % 5 == 0) {
                std::vector<float> s(E);
                for (uint32_t e = 0; e < E; ++e) s[e] = float(lr[e]);
                std::nth_element(s.begin(), s.begin() + (k - 1), s.end(), std::greater<float>());
                const float kth = s[k - 1];
                uint32_t victim = rng() % E;
                while (float(lr[victim]) >= kth) victim = rng() % E;
                lr[victim] = sycl::half(kth);
            }
        }
        Tally t;
        char what[64];
        std::snprintf(what, sizeof what, "random rows, sigma %.1f", double(sigma));
        run_rows(q, L, E, k, what, t);
        if (t.dev_diff) ++fail;
    }
    if (argc > 1) {
        FILE* f = std::fopen(argv[1], "rb");
        if (!f) { std::printf("  cannot open %s\n", argv[1]); return 2; }
        std::vector<sycl::half> L;
        sycl::half buf[4096];
        size_t n;
        while ((n = std::fread(buf, 2, 4096, f)) > 0) L.insert(L.end(), buf, buf + n);
        std::fclose(f);
        L.resize(L.size() / E * E);
        Tally t;
        run_rows(q, L, E, k, "real router rows (dump)", t);
        if (t.dev_diff) ++fail;
    }
    {   // qwen4_route_jobs_dev: the job table in qwen4exp's host layout, against the same table built on the host
        std::mt19937 rng(99);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        const uint32_t S = 60, NU = k, cnt_off = 3 * 16 * NU, w_off = cnt_off + 16;
        bool ok = true;
        for (uint32_t T : {1u, 3u, 4u}) {
            for (int rep = 0; rep < 50; ++rep) {
                std::vector<sycl::half> L(size_t(T) * E);
                for (auto& v : L) v = sycl::half(nd(rng));
                std::vector<int16_t> slot_of(E, -1);   // ~40 % of the experts cached, at random slots < S
                for (uint32_t e = 0, nxt = 0; e < E; ++e) if (rng() % 5 < 2 && nxt < S) slot_of[e] = int16_t((nxt++ * 7) % S);
                const uint32_t words = w_off + T * NU;
                sycl::half* dl = sycl::malloc_device<sycl::half>(L.size(), q);
                int16_t* dso = sycl::malloc_device<int16_t>(E, q);
                int32_t* dj = sycl::malloc_device<int32_t>(words, q);
                int32_t* did = sycl::malloc_device<int32_t>(T * NU, q);
                q.memcpy(dl, L.data(), L.size() * 2).wait();
                q.memcpy(dso, slot_of.data(), E * 2).wait();
                q.memset(dj, 0, words * 4).wait();
                qwen4_route_jobs_dev(q, dl, T, E, k, dso, S, dj, cnt_off, w_off, did).wait();
                std::vector<int32_t> gj(words), gid(T * NU), hj(words, 0), hid(T * NU);
                q.memcpy(gj.data(), dj, words * 4).wait();
                q.memcpy(gid.data(), did, T * NU * 4).wait();
                float* hw = reinterpret_cast<float*>(hj.data() + w_off);
                for (uint32_t vt = 0; vt < T; ++vt) {
                    std::vector<int32_t> ids(k); std::vector<float> w(k);
                    qwen4_route_host(L.data() + size_t(vt) * E, E, k, ids.data(), w.data());
                    hj[cnt_off + vt] = int32_t(k);
                    for (uint32_t j = 0; j < k; ++j) {
                        const uint32_t pp = vt * k + j, e = uint32_t(ids[j]);
                        hj[3 * pp] = int32_t(vt); hj[3 * pp + 1] = slot_of[e] >= 0 ? slot_of[e] : int32_t(S + e); hj[3 * pp + 2] = int32_t(pp);
                        hw[pp] = w[j]; hid[pp] = int32_t(e);
                    }
                }
                ok = ok && gj == hj && gid == hid;
                sycl::free(dl, q); sycl::free(dso, q); sycl::free(dj, q); sycl::free(did, q);
            }
        }
        std::printf("  %-34s %s\n", "jobs table (T = 1 / 3 / 4, 150 sets)", ok ? "device == host-built, every word" : "DIFFERS");
        if (!ok) ++fail;
    }
    {   // the kernel's time at the decode (1 row) and verify (4 rows) shapes
        std::vector<sycl::half> L(size_t(4) * E);
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        for (auto& v : L) v = sycl::half(nd(rng));
        sycl::half* dl = sycl::malloc_device<sycl::half>(L.size(), q);
        int32_t* di = sycl::malloc_device<int32_t>(4 * k, q);
        float* dw = sycl::malloc_device<float>(4 * k, q);
        q.memcpy(dl, L.data(), L.size() * 2).wait();
        for (uint32_t rows : {1u, 4u, 1u}) {   // 1 row twice: the first pass includes the clock ramp
            for (int i = 0; i < 50; ++i) qwen4_route_dev(q, dl, rows, E, k, di, dw);
            q.wait();
            const int N = 2000;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; ++i) qwen4_route_dev(q, dl, rows, E, k, di, dw);
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
            std::printf("  kernel: %u row%s %.1f us a launch (back to back, incl. launch gaps)\n", rows, rows > 1 ? "s" : "", us);
        }
        {   // qwen4_route_jobs_dev at one row (the decode shape), the model's buffers' sizes
            const uint32_t NU = k, words = 3 * 16 * NU + 16 + 16 * NU;
            int16_t* dso = sycl::malloc_device<int16_t>(E, q);
            int32_t* dj = sycl::malloc_device<int32_t>(words, q);
            int32_t* did = sycl::malloc_device<int32_t>(16 * NU, q);
            q.memset(dso, 0xff, E * 2).wait();
            for (int i = 0; i < 50; ++i) qwen4_route_jobs_dev(q, dl, 1, E, k, dso, 60, dj, 3 * 16 * NU, 3 * 16 * NU + 16, did);
            q.wait();
            const int N = 2000;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; ++i) qwen4_route_jobs_dev(q, dl, 1, E, k, dso, 60, dj, 3 * 16 * NU, 3 * 16 * NU + 16, did);
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
            std::printf("  kernel: jobs variant, 1 row %.1f us a launch (back to back, incl. launch gaps)\n", us);
            sycl::free(dso, q); sycl::free(dj, q); sycl::free(did, q);
        }
        sycl::free(dl, q); sycl::free(di, q); sycl::free(dw, q);
    }
    std::printf("\n%s  (device != host in %d set%s)\n\n", fail ? "GATE FAILED" : "GATE PASSED", fail, fail == 1 ? "" : "s");
    return fail ? 1 : 0;
}
