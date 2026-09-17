// tests/unit/runtime_defaults_test.cpp — ie::apply_runtime_defaults():
//   1. the static initializer in allocator.cpp applies SYCL_CACHE_PERSISTENT=1
//      before main() in any binary that links DeviceAllocator;
//   2. an explicit environment value is never overwritten (overwrite=0), checked
//      by re-executing this binary with SYCL_CACHE_PERSISTENT=0 preset;
//   3. repeated calls are idempotent.
#include "ie/allocator.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int, char** argv) {
    const char* child = std::getenv("IE_RT_DEFAULTS_CHILD");
    const char* v = std::getenv("SYCL_CACHE_PERSISTENT");
    if (child) {
        // Parent preset SYCL_CACHE_PERSISTENT=0: the default must not clobber it.
        if (!v || std::strcmp(v, "0") != 0) {
            std::printf("FAIL child: explicit SYCL_CACHE_PERSISTENT=0 was overwritten (got %s)\n",
                        v ? v : "(unset)");
            return 1;
        }
        return 0;
    }
    if (!v || std::strcmp(v, "1") != 0) {
        std::printf("FAIL: SYCL_CACHE_PERSISTENT not defaulted to 1 before main (got %s)\n",
                    v ? v : "(unset)");
        return 1;
    }
    ie::apply_runtime_defaults();   // idempotent
    if (std::strcmp(std::getenv("SYCL_CACHE_PERSISTENT"), "1") != 0) {
        std::printf("FAIL: second apply changed the value\n");
        return 1;
    }
    const std::string cmd = std::string("IE_RT_DEFAULTS_CHILD=1 SYCL_CACHE_PERSISTENT=0 \"") +
                            argv[0] + "\"";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) { std::printf("FAIL: child re-exec returned %d\n", rc); return 1; }
    std::printf("runtime_defaults_test: 3/3 OK\n");
    return 0;
}
