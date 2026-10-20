// include/ie/laguna_engine.hpp — the Laguna-S-2.1 bundle behind the Engine (docs/laguna/00_PORT_PLAN.md P5): the runtime
// (one card or two), its queues, the GGUF's tokenizer and the live conversation (the ids whose K/V the runtime holds). Built
// and used in src/engine/laguna_engine.cpp; complete here so Engine's destructor can destroy the unique_ptr.
#pragma once
#include "ie/laguna.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace ie {

struct LagunaBundle {
    LagunaConfig                 cfg;
    std::unique_ptr<sycl::queue> q;       // declared before the model: destroyed after it
    std::unique_ptr<sycl::queue> q2;      // the second card (two cards), likewise
    LagunaModel                  model;
    Tokenizer                    tok;
    std::vector<int32_t>         live;    // the tokens the model's KV holds (prefix reuse)
    std::vector<int32_t>         eos;     // 〈|EOS|〉 and </assistant> (the template's turn end)
    uint32_t                     chunk = 512;
    bool                         reuse = true;   // --no-prompt-cache: every request reads its whole prompt
};

}  // namespace ie
