// p2p_soak — sustained bidirectional peer-WRITE stress with per-transfer
// verification, under concurrent compute load on both cards. Decides whether
// the 2026-08-16 "peer writes unreliable under load" finding still holds on
// the p2pwl kernel + 26.22 stack. Peer READS are never used (known-broken);
// all verification happens on the buffer's OWNER device.
// usage: p2p_soak [seconds=300] [mb=64]
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const double secs = argc > 1 ? std::atof(argv[1]) : 300.0;
    const size_t NB = (argc > 2 ? size_t(std::atoll(argv[2])) : 64) << 20;
    std::vector<sycl::device> gpus;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_info<sycl::info::platform::name>().find("Level-Zero") ==
            std::string::npos) continue;
        for (const auto& d : p.get_devices())
            if (d.is_gpu() &&
                d.get_info<sycl::info::device::name>().find("B70") !=
                    std::string::npos)
                gpus.push_back(d);
    }
    if (gpus.size() < 2) { std::printf("need 2 B70\n"); return 1; }
    gpus.resize(2);
    gpus[0].ext_oneapi_enable_peer_access(gpus[1]);
    gpus[1].ext_oneapi_enable_peer_access(gpus[0]);
    sycl::context ctx(gpus);
    sycl::queue q0(ctx, gpus[0], sycl::property::queue::in_order{});
    sycl::queue q1(ctx, gpus[1], sycl::property::queue::in_order{});
    // Compute-load queues (separate, same devices) keep both GPUs busy.
    sycl::queue l0(ctx, gpus[0], sycl::property::queue::in_order{});
    sycl::queue l1(ctx, gpus[1], sycl::property::queue::in_order{});

    const size_t NW = NB / 4;
    auto* s0 = sycl::malloc_device<uint32_t>(NW, q0);   // src on dev0
    auto* r1 = sycl::malloc_device<uint32_t>(NW, q1);   // dst on dev1
    auto* s1 = sycl::malloc_device<uint32_t>(NW, q1);   // src on dev1
    auto* r0 = sycl::malloc_device<uint32_t>(NW, q0);   // dst on dev0
    auto* e0 = sycl::malloc_shared<uint64_t>(2, q0);    // err count, checksum
    auto* e1 = sycl::malloc_shared<uint64_t>(2, q1);
    auto* w0 = sycl::malloc_device<float>(1 << 22, l0); // load working set
    auto* w1 = sycl::malloc_device<float>(1 << 22, l1);
    l0.fill(w0, 1.0f, 1 << 22).wait();
    l1.fill(w1, 1.0f, 1 << 22).wait();

    auto fill = [&](sycl::queue& q, uint32_t* p, uint32_t seed) {
        return q.parallel_for(sycl::range<1>(NW), [=](sycl::id<1> i) {
            p[i] = uint32_t(i) * 2654435761u + seed;
        });
    };
    auto verify = [&](sycl::queue& q, uint32_t* p, uint32_t seed, uint64_t* e) {
        return q.parallel_for(sycl::range<1>(NW), [=](sycl::id<1> i) {
            if (p[i] != uint32_t(i) * 2654435761u + seed) {
                sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>(e[0])++;
            }
        });
    };
    auto churn = [&](sycl::queue& q, float* w) {   // ~ms-scale compute burst
        return q.parallel_for(sycl::range<1>(1 << 22), [=](sycl::id<1> i) {
            float v = w[i];
            for (int k = 0; k < 64; ++k) v = sycl::fma(v, 1.0000001f, 1e-7f);
            w[i] = v;
        });
    };

    e0[0] = e1[0] = 0;
    uint64_t iters = 0, bytes = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(
               std::chrono::steady_clock::now() - t0).count() < secs) {
        const uint32_t seed = uint32_t(iters * 0x9E3779B9u + 1);
        // Keep both devices busy with compute during the peer writes.
        churn(l0, w0); churn(l1, w1);
        fill(q0, s0, seed); fill(q1, s1, seed ^ 0xA5A5A5A5u);
        q0.memcpy(r1, s0, NB);            // dev0 PUSHES into dev1
        q1.memcpy(r0, s1, NB);            // dev1 PUSHES into dev0
        q0.wait(); q1.wait();
        verify(q1, r1, seed, e1);              // verify on OWNER device
        verify(q0, r0, seed ^ 0xA5A5A5A5u, e0);
        q0.wait(); q1.wait();
        if (e0[0] || e1[0]) {
            std::printf("CORRUPTION at iter %llu: dev0-recv errs %llu, dev1-recv errs %llu\n",
                        (unsigned long long)iters, (unsigned long long)e0[0],
                        (unsigned long long)e1[0]);
            return 2;
        }
        ++iters; bytes += 2 * NB;
        if ((iters & 63) == 0) { std::printf("  ... iter %llu clean\n",
                (unsigned long long)iters); std::fflush(stdout); }
    }
    l0.wait(); l1.wait();
    const double s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("SOAK CLEAN: %llu iters, %.1f GB pushed bidirectionally in %.0fs "
                "(%.2f GB/s agg), 0 corrupt words\n",
                (unsigned long long)iters, bytes / 1e9, s, bytes / s / 1e9);
    return 0;
}
