// tests/unit/dense_fleet_rules_test.cpp -- the dtype rule that sends a one-card dense GGUF to the fleet class
// (DenseModelTP) or keeps it on DenseModel (include/ie/dense_fleet_rules.hpp). CPU only.
#include "ie/dense_fleet_rules.hpp"

#include <cstdio>

using ie::DType;
using ie::DenseFleetDtypes;

static int fails = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++fails; } } while (0)

int main() {
    const DType Q6 = DType::kQ6_K, Q4 = DType::kQ4_K, Q8 = DType::kQ8_0, Q5 = DType::kQ5_K, F16 = DType::kF16;
    // Nanbeige4.3-33B Q6_K (every projection, the embedding and the lm_head Q6_K; llama arch): the fleet class
    CHECK(ie::dense_fleet_dtypes_ok({Q6, Q6, Q6, Q6, Q6, Q6, Q6, Q6, Q6, true}));
    // a Q4_K_M-style mix (Q4_K projections, Q6_K ffn_down / v / output): the fleet class
    CHECK(ie::dense_fleet_dtypes_ok({Q4, Q4, Q6, Q4, Q4, Q4, Q6, Q4, Q6, true}));
    // qwen3 dense (non-llama) with an F16 attn_v (qwen3-8b's shape): column-parallel F16 is fine
    CHECK(ie::dense_fleet_dtypes_ok({Q6, Q6, F16, Q6, Q6, Q6, Q6, Q6, Q6, false}));
    // ... but a llama file's F16 q or k is not (the row permute reads Q4_K / Q6_K rows)
    CHECK(!ie::dense_fleet_dtypes_ok({F16, Q6, Q6, Q6, Q6, Q6, Q6, Q6, Q6, true}));
    CHECK(!ie::dense_fleet_dtypes_ok({Q6, F16, Q6, Q6, Q6, Q6, Q6, Q6, Q6, true}));
    // an F16 row-parallel weight (o-proj or ffn_down) has no TP loader
    CHECK(!ie::dense_fleet_dtypes_ok({Q6, Q6, Q6, F16, Q6, Q6, Q6, Q6, Q6, false}));
    CHECK(!ie::dense_fleet_dtypes_ok({Q6, Q6, Q6, Q6, Q6, Q6, F16, Q6, Q6, false}));
    // Q8_0 / Q5_K anywhere: DenseModel's F16 expansion, the previous one-card path
    CHECK(!ie::dense_fleet_dtypes_ok({Q6, Q6, Q6, Q6, Q6, Q6, Q8, Q6, Q6, true}));
    CHECK(!ie::dense_fleet_dtypes_ok({Q5, Q5, Q5, Q5, Q5, Q5, Q5, Q5, Q5, true}));
    CHECK(!ie::dense_fleet_dtypes_ok({Q6, Q6, Q6, Q6, Q6, Q6, Q6, Q6, Q8, true}));    // a Q8_0 lm_head
    CHECK(!ie::dense_fleet_dtypes_ok({Q6, Q6, Q6, Q6, Q6, Q6, Q6, Q8, Q6, true}));    // a Q8_0 embedding
    // a missing tensor (DType::kCount) is unloadable
    CHECK(!ie::dense_fleet_dtypes_ok({Q6, Q6, Q6, Q6, Q6, DType::kCount, Q6, Q6, Q6, true}));
    // the geometry: Nanbeige (56 x 128 = 7168, FFN 22016), qwen3-8b (32 x 128, 12288) and llama-7b (32 x 128, 11008 =
    // 43 x 256) fit the 256-superblock split; a 40 x 80 head layout (3200) or an FFN of 11000 does not
    CHECK(ie::dense_fleet_geometry_ok(56 * 128, 22016));
    CHECK(ie::dense_fleet_geometry_ok(32 * 128, 12288));
    CHECK(ie::dense_fleet_geometry_ok(32 * 128, 11008));
    CHECK(!ie::dense_fleet_geometry_ok(40 * 80, 12288));
    CHECK(!ie::dense_fleet_geometry_ok(32 * 128, 11000));
    CHECK(!ie::dense_fleet_geometry_ok(0, 12288));
    if (fails) { std::fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    std::printf("dense_fleet_rules_test: PASS\n");
    return 0;
}
