// include/ie/dp4a_native.hpp — P4 B32: the hardware integer dot on packed words, without IGC's byte shuffles.
//
// ie::dp4a_ss (dp4a.hpp) is plain byte arithmetic that IGC pattern-matches to dp4a, but over vector-loaded words it
// rebuilds every operand byte by byte: 12 byte movs per operand word (P4 B31, ~/ds41_work/p60/b31/BUILD.md). This calls
// IGC's own integer-dot builtin (what its OpenCL dot-product builtins lower to): GenISA.dp4a.ss(c, a, b, sat = false),
// the same instruction and the same exact integer as ie::dp4a_ss. IGC-only: -DIE_DP4A_PORTABLE=1 builds the plain
// arithmetic for another SYCL device (e.g. the OpenCL CPU device). Kept out of dp4a.hpp, which keys the DS4.1 prompt
// cache (src/model/deepseek41_numerics_inputs.txt).
#pragma once

#include "ie/dp4a.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

#if defined(__SYCL_DEVICE_ONLY__) && !defined(IE_DP4A_PORTABLE)
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_ss(int c, int a, int b, bool sat);
#endif

namespace ie {

// (s8x4 · s8x4) + c, exact: the same integer as ie::dp4a_ss(a, b, c).
inline int32_t dp4a_ss_native(uint32_t a, uint32_t b, int32_t c) {
#if defined(__SYCL_DEVICE_ONLY__) && !defined(IE_DP4A_PORTABLE)
    return __builtin_IB_dp4a_ss(c, int32_t(a), int32_t(b), false);
#else
    return ie::dp4a_ss(int32_t(a), int32_t(b), c);
#endif
}

}  // namespace ie
