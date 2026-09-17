// include/ie/ds41_engine.hpp — the DeepSeek-V4.1-Flash bundle behind the Engine (Phase 12,
// docs/deepseek41/31): the model, its engram tables, the tokenizer.json tokenizer, one in-order
// queue per Arc card, and the resident two-card runtime. Complete here so Engine's destructor
// (engine.cpp) can destroy the unique_ptr; built and used in src/engine/ds41_engine.cpp.
#pragma once
#include "ie/deepseek41.hpp"
#include "ie/deepseek41_engram.hpp"
#include <cstdio>

#include "ie/deepseek41_dspark.hpp"
#include "ie/deepseek41_forward.hpp"
#include "ie/tokenizer.hpp"

#include <memory>
#include <vector>

namespace ie {

struct Ds41Bundle {
    DeepSeek41Model                             model;
    Ds41EngramTables                            tables;
    Tokenizer                                   tok;
    std::vector<std::unique_ptr<sycl::queue>>   queues;
    std::vector<sycl::queue*>                   qs;
    Ds41Forward                                 fwd;
    Ds41Drafter                                 drafter;   // DSpark P4 (docs/deepseek41/58): built under IE_DS41_SPEC=1
    // Phase 48 (docs/deepseek41/88): IE_DS41_PROFILE_OUT -- the DECODE routing of every request served, summed per
    // (layer, expert) and rewritten as a residency ranking after each request, so a client's own traffic can be profiled
    std::string                                 profile_out;
    std::vector<std::vector<uint64_t>>          profile_acc;
    uint64_t                                    profile_steps = 0;
    // Drain BEFORE any free: the drafter's and the runtime's device memory can still be referenced by
    // kernels an aborted generation left in flight (see Ds41Forward::free_resident). The final wait is
    // kept as a belt-and-braces flush of anything the frees themselves submitted.
    ~Ds41Bundle() {
        for (auto& q : queues) if (q) { try { q->wait_and_throw(); } catch (const sycl::exception& e) { std::fprintf(stderr, "[deepseek41] teardown drain: %s\n", e.what()); } }
        if (drafter.ready() && !qs.empty()) drafter.free(*qs.back());
        fwd.free_resident();
        for (auto& q : queues) if (q) { try { q->wait_and_throw(); } catch (const sycl::exception&) {} }
    }
};

}  // namespace ie
