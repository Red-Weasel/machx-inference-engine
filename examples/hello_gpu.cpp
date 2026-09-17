// examples/hello_gpu.cpp — Phase 0 smoke test.
// Enumerates SYCL devices, prints info for the selected one, runs a trivial kernel.

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace ie {

void print_aspects(const sycl::device& d) {
    struct Pair { sycl::aspect a; const char* name; };
    static const Pair pairs[] = {
        {sycl::aspect::gpu, "gpu"},
        {sycl::aspect::fp16, "fp16"},
        {sycl::aspect::fp64, "fp64"},
        {sycl::aspect::usm_device_allocations, "usm_device"},
        {sycl::aspect::usm_host_allocations, "usm_host"},
        {sycl::aspect::usm_shared_allocations, "usm_shared"},
        {sycl::aspect::ext_intel_matrix, "ext_intel_matrix (XMX)"},
        {sycl::aspect::ext_intel_esimd, "ext_intel_esimd"},
    };
    std::fputs("    aspects:", stdout);
    for (const auto& p : pairs) {
        if (d.has(p.a)) std::printf(" %s", p.name);
    }
    std::putchar('\n');
}

void print_device(const sycl::device& d) {
    std::printf("  Name      : %s\n", d.get_info<sycl::info::device::name>().c_str());
    std::printf("  Vendor    : %s\n", d.get_info<sycl::info::device::vendor>().c_str());
    std::printf("  Driver    : %s\n", d.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("  Type      : %s\n",
                d.is_gpu() ? "GPU" : d.is_cpu() ? "CPU" : "other");
    std::printf("  EUs       : %u\n", d.get_info<sycl::info::device::max_compute_units>());
    std::printf("  Max WG    : %zu\n", d.get_info<sycl::info::device::max_work_group_size>());
    std::printf("  Max alloc : %.2f GiB\n",
                d.get_info<sycl::info::device::max_mem_alloc_size>() / double(1ull<<30));
    std::printf("  Global mem: %.2f GiB\n",
                d.get_info<sycl::info::device::global_mem_size>() / double(1ull<<30));
    std::printf("  Local mem : %zu KiB\n",
                d.get_info<sycl::info::device::local_mem_size>() / 1024);
    auto sg_sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    std::fputs("  SG sizes  :", stdout);
    for (auto s : sg_sizes) std::printf(" %zu", s);
    std::putchar('\n');
    print_aspects(d);
}

}  // namespace ie

int main(int argc, char** argv) {
    bool list_only = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--list" || a == "-l") list_only = true;
    }

    auto devices = sycl::device::get_devices();
    if (devices.empty()) {
        std::fputs("No SYCL devices found.\n", stderr);
        return 2;
    }

    std::printf("Found %zu SYCL device(s):\n", devices.size());
    int idx = 0;
    for (const auto& d : devices) {
        std::printf("\n[#%d] (%s)\n", idx++,
                    d.is_gpu() ? "gpu" : d.is_cpu() ? "cpu" : "other");
        ie::print_device(d);
    }
    if (list_only) return 0;

    // Pick the first GPU whose name contains "0xe223" (the B70) or any GPU.
    sycl::device picked;
    bool found = false;
    for (const auto& d : devices) {
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("0xe223") != std::string::npos) {
            picked = d; found = true; break;
        }
    }
    if (!found) {
        for (const auto& d : devices) if (d.is_gpu()) { picked = d; found = true; break; }
    }
    if (!found) {
        std::fputs("\nNo GPU device available; not running kernel.\n", stderr);
        return 3;
    }

    std::printf("\nSelected: %s\n", picked.get_info<sycl::info::device::name>().c_str());
    sycl::queue q(picked, sycl::property::queue::enable_profiling{});
    constexpr size_t N = 1u << 20;          // 1M floats
    auto* a = sycl::malloc_device<float>(N, q);
    auto* b = sycl::malloc_device<float>(N, q);
    auto* c = sycl::malloc_device<float>(N, q);
    if (!a || !b || !c) {
        std::fputs("device alloc failed\n", stderr);
        return 4;
    }

    std::vector<float> ha(N, 1.0f), hb(N, 2.0f), hc(N, 0.0f);
    q.memcpy(a, ha.data(), N*sizeof(float)).wait();
    q.memcpy(b, hb.data(), N*sizeof(float)).wait();

    auto evt = q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        c[i] = a[i] + b[i];
    });
    evt.wait();
    q.memcpy(hc.data(), c, N*sizeof(float)).wait();

    if (hc[0] != 3.0f || hc[N-1] != 3.0f) {
        std::printf("kernel result wrong: hc[0]=%f hc[N-1]=%f\n", hc[0], hc[N-1]);
        return 5;
    }

    auto t0 = evt.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t1 = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
    std::printf("\nVector add OK: %zu elements in %.3f ms (%.1f GB/s effective)\n",
                N, (t1 - t0) / 1.0e6,
                (3.0 * N * sizeof(float)) / double(t1 - t0));

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    return 0;
}
