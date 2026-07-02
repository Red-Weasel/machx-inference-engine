// include/ie/exl3.hpp — EXL3 (QTIP-based) weight decode, host reference.
//
// EXL3 stores each Linear as a tail-biting trellis (int16, per 16x16 tile) plus
// per-feature sign+scale vectors (suh/svh). The decode is pure ALU: extract a
// `bits`-wide code per weight, run a 3-instruction integer hash (codebook cb=0),
// reinterpret as two fp16 and sum. The 256 codes per tile come out in tensor-core
// lane order and must be un-permuted to row-major.
//
// This header declares the host (CPU) reference decode — the ground truth the SYCL
// kernel (gemv_exl3) is validated against. Spec + file:line cites:
// docs/exl3_format_notes.md (§2 extraction, §3 codebook, §7 tile permutation).
// Decode-only: NO Hadamard / suh / svh (those are the forward-reconstruction step).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

struct Exl3Tensor {
    uint32_t K = 0;       // in_features  (= tile_k * 16)
    uint32_t N = 0;       // out_features (= tile_n * 16)
    uint32_t bits = 0;    // bpw 1..8 (= trellis.shape[-1] / 16)
    uint32_t cb = 0;      // codebook id (0 = default; 1/2 not yet supported)
    uint32_t tile_k = 0;  // K / 16
    uint32_t tile_n = 0;  // N / 16
    // Raw trellis: int16 reinterpreted as uint16, row-major [tile_k, tile_n, 16*bits].
    std::vector<uint16_t> trellis;
};

// Decode `t.trellis` into `out` = W_rot[K, N] row-major (fp16 values widened to
// float). `out` must hold K*N floats. Returns "" on success, else an error string.
// Trellis decode + tensor-core un-permute only (the "incoherent"/rotated weights);
// the Hadamard + suh/svh reconstruction is a separate forward step. cb=0 only.
std::string exl3_decode_host(const Exl3Tensor& t, float* out);

}  // namespace ie
