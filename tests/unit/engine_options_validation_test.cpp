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
    std::cout<<"engine_options_validation_test: OK\n";
}
