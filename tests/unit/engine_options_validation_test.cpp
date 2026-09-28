#undef NDEBUG
#include "ie/engine.hpp"
#include <cassert>
#include <iostream>
int main() {
    for(uint32_t ctx:{0u,1u,7u,8u}) {
        ie::EngineOptions opts;opts.max_ctx=ctx;
        std::string error;
        auto engine=ie::Engine::load("/does/not/exist.gguf",opts,error);
        assert(!engine && error.find("context")!=error.npos && error.find("9")!=error.npos);
    }
    ie::EngineOptions opts;opts.slot_ctx=8;
    std::string error;
    auto engine=ie::Engine::load("/does/not/exist.gguf",opts,error);
    assert(!engine && error.find("slot")!=error.npos);
    {   // P4 B14: --parallel above ie::kMaxParallel is refused before any file is opened
        ie::EngineOptions po;po.parallel=ie::kMaxParallel+1;
        std::string perr;
        assert(!ie::Engine::load("/does/not/exist.gguf",po,perr) && perr.find("parallel 17")!=perr.npos && perr.find("16")!=perr.npos);
        po.parallel=ie::kMaxParallel; perr.clear();
        assert(!ie::Engine::load("/does/not/exist.gguf",po,perr) && perr.find("parallel")==perr.npos);   // fails later: the file
    }
    std::cout<<"engine_options_validation_test: OK\n";
}
