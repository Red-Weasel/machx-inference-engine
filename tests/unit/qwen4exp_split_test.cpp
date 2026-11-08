// tests/unit/qwen4exp_split_test.cpp -- qwen4exp_split_merged (include/ie/qwen4exp.hpp): llama.cpp's converter writes
// attn_qkv_gate = [attn_qkv | attn_gate], ssm_beta_alpha = [ssm_beta | ssm_alpha] and hc_*_down_inject = [hc_*_down |
// hc_*_inject] since 2026-10-25; the loader splits such a tensor into two views at an output column. Host-only, no GPU.
#include "ie/qwen4exp.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) { std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str()); if (!ok) ++g_fail; }
ie::GgufTensorInfo tensor(const char* name, ie::DType dt, uint64_t k, uint64_t n, uint64_t row_bytes, const uint8_t* data) {
    ie::GgufTensorInfo t;
    t.name = name; t.dtype = dt; t.n_dims = 2; t.shape[0] = k; t.shape[1] = n;
    t.offset_in_data = 4096; t.nbytes = row_bytes * n; t.data = data;
    return t;
}
}  // namespace

int main() {
    std::vector<uint8_t> bytes(16384 * 34 * 2, 0);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i * 31 + 7);
    {   // attn_qkv_gate [2560, 16384] in Q8_0: a row of 2560 = 80 blocks x 34 B = 2720 B; split at 10240 (qkv) | 6144 (gate)
        const ie::GgufTensorInfo m = tensor("blk.0.attn_qkv_gate.weight", ie::DType::kQ8_0, 2560, 16384, 2720, bytes.data());
        ie::GgufTensorInfo a, b;
        check(ie::qwen4exp_split_merged(m, 10240, a, b), "a Q8_0 [2560, 16384] splits at 10240");
        check(a.shape[0] == 2560 && a.shape[1] == 10240 && a.nbytes == 2720ull * 10240 && a.data == m.data && a.offset_in_data == m.offset_in_data &&
              a.dtype == m.dtype && a.n_dims == 2 && a.name == m.name, "the first half: the first 10240 output columns, same start");
        check(b.shape[0] == 2560 && b.shape[1] == 6144 && b.nbytes == 2720ull * 6144 && b.data == m.data + a.nbytes &&
              b.offset_in_data == m.offset_in_data + a.nbytes && b.dtype == m.dtype, "the second half: the next 6144 columns, right after the first");
        check(a.nbytes + b.nbytes == m.nbytes, "the halves tile the merged tensor exactly");
    }
    {   // ssm_beta_alpha [2560, 96] in F32 (1-D-ish small matrices stay F32): split at 48 | 48
        const ie::GgufTensorInfo m = tensor("blk.3.ssm_beta_alpha.weight", ie::DType::kF32, 2560, 96, 2560 * 4, bytes.data());
        ie::GgufTensorInfo a, b;
        check(ie::qwen4exp_split_merged(m, 48, a, b) && a.shape[1] == 48 && b.shape[1] == 48 && b.data == m.data + 2560ull * 4 * 48,
              "an F32 [2560, 96] splits at 48 | 48");
    }
    {   // hc_attn_down_inject [10240, 324] in Q4_K: a row of 10240 = 40 super-blocks x 144 B; split at 320 | 4
        const ie::GgufTensorInfo m = tensor("blk.5.hc_attn_down_inject.weight", ie::DType::kQ4_K, 10240, 324, 40 * 144, bytes.data());
        ie::GgufTensorInfo a, b;
        check(ie::qwen4exp_split_merged(m, 320, a, b) && a.shape[1] == 320 && b.shape[1] == 4 && b.nbytes == 4ull * 40 * 144 &&
              b.data == m.data + 320ull * 40 * 144, "a Q4_K [10240, 324] splits at 320 | 4 (the inject's four columns)");
    }
    {   // refusals: not 2-D, n_a at or past the end, zero n_a, bytes that do not divide by the row count
        ie::GgufTensorInfo a, b;
        ie::GgufTensorInfo m3 = tensor("x", ie::DType::kQ8_0, 2560, 16, 2720, bytes.data()); m3.n_dims = 3; m3.shape[2] = 2;
        check(!ie::qwen4exp_split_merged(m3, 8, a, b), "a 3-D bank is not split");
        const ie::GgufTensorInfo m = tensor("y", ie::DType::kQ8_0, 2560, 16, 2720, bytes.data());
        check(!ie::qwen4exp_split_merged(m, 16, a, b) && !ie::qwen4exp_split_merged(m, 17, a, b) && !ie::qwen4exp_split_merged(m, 0, a, b),
              "n_a at the end, past it, or zero: not split");
        ie::GgufTensorInfo odd = m; odd.nbytes = 2720ull * 16 + 1;
        check(!ie::qwen4exp_split_merged(odd, 8, a, b), "bytes the row count does not divide: not split");
    }
    std::printf("\nQWEN4EXP SPLIT: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
