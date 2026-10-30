// ds41_hugetlb_staging_test (2026-10-27): the mmap tier's staging slots as MAP_HUGETLB pages -- the way
// src/model/deepseek41_experts.cpp allocates them since docs/research/2026-10-27-card0-kernel-queue-wedge-recurrence.md.
// Proves on a real card that (1) a slot of huge pages is really huge pages (KernelPageSize in /proc/self/smaps),
// (2) an O_DIRECT pread lands in it, (3) the driver imports it (zexDriverImportExternalPointer), (4) the card DMAs
// from it byte for byte, 20 rounds, with CPU writes between rounds seen by the DMA, (5) release + munmap are clean.
// Exit 0 = pass, 1 = fail, 77 = skip (no GPU, no import extension, or no reserved hugetlb pool: vm.nr_hugepages).
// IE_TEST_PCI=09:00.0 picks the card (a substring of its PCI address); the default is the first Level Zero GPU.
// IE_TEST_TMPDIR holds the 18 MiB file (a filesystem that takes O_DIRECT; default /tmp).
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

size_t hugepage_bytes() {
    size_t kb = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f)) if (std::sscanf(line, "Hugepagesize: %zu kB", &kb) == 1) break;
        std::fclose(f);
    }
    return kb ? kb << 10 : size_t(2) << 20;
}

// KernelPageSize of the mapping that holds `p`, from /proc/self/smaps (0 when not found)
size_t kernel_page_size_of(const void* p) {
    std::ifstream in("/proc/self/smaps");
    std::string line; bool inside = false;
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    while (std::getline(in, line)) {
        unsigned long lo = 0, hi = 0;
        if (std::sscanf(line.c_str(), "%lx-%lx ", &lo, &hi) == 2) { inside = (a >= lo && a < hi); continue; }
        size_t kb = 0;
        if (inside && std::sscanf(line.c_str(), "KernelPageSize: %zu kB", &kb) == 1) return kb << 10;
    }
    return 0;
}

int fail(const char* what) { std::fprintf(stderr, "FAIL: %s\n", what); return 1; }
int skip(const char* why) { std::fprintf(stderr, "SKIP: %s\n", why); return 77; }

}  // namespace

int main() {
    // ---- the card ------------------------------------------------------------------------------------------------
    const char* want = std::getenv("IE_TEST_PCI");
    std::vector<sycl::device> gpus;
    for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) gpus.push_back(d);
    if (gpus.empty()) return skip("no Level Zero GPU");
    int pick = -1;
    for (size_t i = 0; i < gpus.size(); ++i) {
        std::string pci = "?";
        try { if (gpus[i].has(sycl::aspect::ext_intel_pci_address)) pci = gpus[i].get_info<sycl::ext::intel::info::device::pci_address>(); } catch (...) {}
        std::fprintf(stderr, "gpu %zu: %s (PCI %s)\n", i, gpus[i].get_info<sycl::info::device::name>().c_str(), pci.c_str());
        if (pick < 0 && (!want || !*want || pci.find(want) != std::string::npos)) pick = int(i);
    }
    if (pick < 0) return skip("IE_TEST_PCI matches no GPU");
    sycl::queue q(gpus[size_t(pick)], sycl::property::queue::in_order{});
    std::fprintf(stderr, "using gpu %d\n", pick);

    // ---- the import extension -------------------------------------------------------------------------------------
    ze_driver_handle_t drv = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context().get_platform());
    void* lib = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_NOLOAD); if (!lib) lib = dlopen("libze_loader.so.1", RTLD_NOW);
    auto getext = lib ? reinterpret_cast<ze_result_t (*)(ze_driver_handle_t, const char*, void**)>(dlsym(lib, "zeDriverGetExtensionFunctionAddress")) : nullptr;
    ze_result_t (*imp)(ze_driver_handle_t, void*, size_t) = nullptr; ze_result_t (*rel)(ze_driver_handle_t, void*) = nullptr;
    if (!getext || getext(drv, "zexDriverImportExternalPointer", reinterpret_cast<void**>(&imp)) != ZE_RESULT_SUCCESS || !imp
                || getext(drv, "zexDriverReleaseImportedPointer", reinterpret_cast<void**>(&rel)) != ZE_RESULT_SUCCESS || !rel)
        return skip("the driver has no zexDriverImportExternalPointer");

    // ---- the slots: an expert-sized buffer (not a multiple of the huge page) rounded up, all or none ------------------
    const size_t huge = hugepage_bytes();
    constexpr size_t kSlots = 2;
    constexpr size_t kBytes = (18ull << 20) + (16ull << 12);   // 18 MiB + 64 KiB: 4 KiB-aligned, as an expert slot is; rounds up
    const size_t slot_huge = (kBytes + huge - 1) / huge * huge;
    std::vector<uint8_t*> slots;
    for (size_t i = 0; i < kSlots; ++i) {
        void* m = mmap(nullptr, slot_huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (m == MAP_FAILED) {
            const int en = errno;
            for (uint8_t* s : slots) munmap(s, slot_huge);
            std::fprintf(stderr, "MAP_HUGETLB(%zu bytes) failed: %s -- reserve vm.nr_hugepages >= %zu pages of %zu MiB for this test\n",
                         slot_huge, std::strerror(en), kSlots * (slot_huge / huge), huge >> 20);
            return skip("no hugetlb pool");
        }
        slots.push_back(static_cast<uint8_t*>(m));
    }
    for (uint8_t* s : slots) std::memset(s, 0, slot_huge);   // the pages exist from here
    for (uint8_t* s : slots) {
        const size_t kps = kernel_page_size_of(s);
        std::fprintf(stderr, "slot at %p: KernelPageSize %zu kB\n", static_cast<void*>(s), kps >> 10);
        if (kps != huge) return fail("the slot is not backed by huge pages (KernelPageSize)");
    }

    // ---- the file, read with O_DIRECT into the slots --------------------------------------------------------------
    const char* tmpdir = std::getenv("IE_TEST_TMPDIR");
    const std::string path = std::string(tmpdir && *tmpdir ? tmpdir : "/tmp") + "/ds41_hugetlb_staging_test." + std::to_string(getpid()) + ".bin";
    std::vector<uint8_t> expect(kBytes);
    {
        std::mt19937_64 g(20261027); for (size_t i = 0; i < kBytes; i += 8) { const uint64_t v = g(); std::memcpy(&expect[i], &v, std::min<size_t>(8, kBytes - i)); }
        const int wfd = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
        if (wfd < 0) return fail("cannot create the test file (IE_TEST_TMPDIR)");
        size_t done = 0; while (done < kBytes) { const ssize_t n = write(wfd, expect.data() + done, kBytes - done); if (n <= 0) { close(wfd); return fail("write"); } done += size_t(n); }
        fsync(wfd); close(wfd);
    }
    const int dfd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (dfd < 0) { unlink(path.c_str()); return fail("open O_DIRECT (IE_TEST_TMPDIR must be a filesystem that takes it)"); }
    auto direct_read = [&](uint8_t* dst) -> bool {
        size_t done = 0;
        while (done < kBytes) { const ssize_t n = pread(dfd, dst + done, kBytes - done, off_t(done)); if (n <= 0) return false; done += size_t(n); }
        return true;
    };

    // ---- import + DMA, 20 rounds ----------------------------------------------------------------------------------
    for (uint8_t* s : slots)
        if (const ze_result_t r = imp(drv, s, slot_huge); r != ZE_RESULT_SUCCESS) { std::fprintf(stderr, "import 0x%x\n", unsigned(r)); return fail("zexDriverImportExternalPointer on a hugetlb slot"); }
    uint8_t* dev = sycl::malloc_device<uint8_t>(kBytes * kSlots, q);
    uint8_t* chk = sycl::malloc_host<uint8_t>(kBytes, q);
    if (!dev || !chk) return fail("device / host allocation");
    double dma_ms = 0; size_t dma_n = 0;
    int rc = 0;
    for (int round = 0; round < 20 && rc == 0; ++round) {
        for (size_t si = 0; si < kSlots && rc == 0; ++si) {
            uint8_t* s = slots[si];
            if (!direct_read(s)) { rc = fail("O_DIRECT pread into the hugetlb slot"); break; }
            std::vector<uint8_t> want_bytes = expect;
            // a CPU write after the direct read, as the engine's packers do: the DMA must see both
            std::memset(s + 4096, uint8_t(round), 4096); std::memset(want_bytes.data() + 4096, uint8_t(round), 4096);
            const auto t0 = std::chrono::steady_clock::now();
            q.memcpy(dev + si * kBytes, s, kBytes).wait();
            dma_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); ++dma_n;
            q.memcpy(chk, dev + si * kBytes, kBytes).wait();
            if (std::memcmp(chk, want_bytes.data(), kBytes) != 0) {
                size_t off = 0; while (off < kBytes && chk[off] == want_bytes[off]) ++off;
                std::fprintf(stderr, "round %d slot %zu: first difference at byte %zu (%02x vs %02x)\n", round, si, off, chk[off], want_bytes[off]);
                rc = fail("the DMA from the hugetlb slot is not byte for byte");
            }
        }
    }
    if (rc == 0) std::fprintf(stderr, "OK: %zu DMAs of %.1f MiB from hugetlb slots, byte for byte; %.1f GB/s host->card from the slot\n",
                              dma_n, double(kBytes) / 1048576.0, double(kBytes) * dma_n / (dma_ms * 1e6));

    // ---- release ---------------------------------------------------------------------------------------------------
    q.wait();
    sycl::free(dev, q); sycl::free(chk, q);
    for (uint8_t* s : slots) if (rel(drv, s) != ZE_RESULT_SUCCESS && rc == 0) rc = fail("zexDriverReleaseImportedPointer");
    for (uint8_t* s : slots) munmap(s, slot_huge);
    close(dfd); unlink(path.c_str());
    return rc;
}
