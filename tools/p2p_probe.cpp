// p2p_probe — is PCIe P2P live between the two B70s on this kernel?
// Probe can_access_peer both ways, then measure a PUSH memcpy (dev0 queue,
// dev0 buffer -> dev1 buffer) in one shared context, verifying bytes via
// each device's own queue (peer READS are known-broken on this bridge).
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>

int main() {
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
    if (gpus.size() > 2) gpus.resize(2);
    std::printf("GPUs: %zu\n", gpus.size());
    if (gpus.size() < 2) return 1;
    for (int a = 0; a < 2; ++a) {
        int b = 1 - a;
        bool can = false;
        try {
            can = gpus[a].ext_oneapi_can_access_peer(
                gpus[b], sycl::ext::oneapi::peer_access::access_supported);
        } catch (const sycl::exception& e) {
            std::printf("probe %d->%d threw: %s\n", a, b, e.what());
        }
        std::printf("can_access_peer %d -> %d : %s\n", a, b, can ? "YES" : "no");
        if (!can) return 2;
    }
    try {
        gpus[0].ext_oneapi_enable_peer_access(gpus[1]);
        gpus[1].ext_oneapi_enable_peer_access(gpus[0]);
    } catch (const sycl::exception& e) {
        std::printf("enable_peer_access threw: %s\n", e.what());
        return 3;
    }
    sycl::context ctx(gpus);
    sycl::queue q0(ctx, gpus[0], sycl::property::queue::in_order{});
    sycl::queue q1(ctx, gpus[1], sycl::property::queue::in_order{});
    const size_t NB = 256ull << 20;   // 256 MB
    auto* a0 = sycl::malloc_device<uint8_t>(NB, q0);
    auto* a1 = sycl::malloc_device<uint8_t>(NB, q1);
    std::vector<uint8_t> h(NB);
    for (size_t i = 0; i < NB; i += 4096) h[i] = uint8_t(i >> 12);
    q0.memcpy(a0, h.data(), NB).wait();
    q1.memset(a1, 0, NB).wait();
    // PUSH: source queue (dev0) writes into dev1's buffer.
    q0.memcpy(a1, a0, NB).wait();               // warm + correctness copy
    std::vector<uint8_t> r(NB);
    q1.memcpy(r.data(), a1, NB).wait();         // read back via OWNER queue
    size_t bad = 0;
    for (size_t i = 0; i < NB; i += 4096) if (r[i] != uint8_t(i >> 12)) ++bad;
    std::printf("push correctness: %zu bad pages of %zu\n", bad, NB / 4096);
    if (bad) return 4;
    const int reps = 10;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) q0.memcpy(a1, a0, NB);
    q0.wait();
    double s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("push bandwidth dev0->dev1: %.2f GB/s\n", NB * double(reps) / s / 1e9);
    return 0;
}
