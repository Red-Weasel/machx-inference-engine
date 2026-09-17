#undef NDEBUG
#include "ie/glm5_memory_policy.hpp"
#include <cassert>
#include <iostream>
int main() {
    using namespace ie;
    constexpr uint64_t G=1ull<<30;
    Glm5MemoryPolicy p;
    assert(glm5_memory_policy(nullptr,nullptr,nullptr,nullptr,nullptr,p).empty());
    assert(p.pin_banks && p.cache_bytes==0);
    assert(glm5_cache_budget(30*G,6*G,2*G,80*G,p)==22*G);
    assert(glm5_cache_budget(30*G,6*G,2*G,4*G,p)==4*G);
    assert(glm5_cache_budget(7*G,6*G,2*G,80*G,p)==0);
    assert(glm5_cache_budget(30*G,UINT64_MAX,2*G,80*G,p)==0);
    auto cap=glm5_pin_budget(226*G,10*G,2,p);
    assert(cap==88*G);   // (226-10-40) GiB / 1.0 overhead / 2 stages (was 64.2 GiB at the retired 1.37)
    assert(glm5_pin_budget(49*G,10*G,2,p)==0);
    assert(glm5_pin_budget(226*G,10*G,0,p)==0);
    assert(glm5_memory_policy("0","10240","40","32","1.37",p).empty());
    assert(!p.pin_banks && p.cache_bytes==10*G);
    assert(glm5_pin_budget(226*G,10*G,2,p)==0);
    assert(glm5_memory_policy("1",nullptr,nullptr,"32",nullptr,p).empty());
    assert(glm5_pin_budget(226*G,10*G,2,p)==32*G);
    assert(glm5_memory_policy("1",nullptr,nullptr,"0",nullptr,p).empty());
    assert(glm5_pin_budget(226*G,10*G,2,p)==0);
    for(auto s:{"nan","inf","-1","1x"})
        assert(!glm5_memory_policy(nullptr,nullptr,nullptr,s,nullptr,p).empty());
    assert(glm5_memory_policy(nullptr,"32768",nullptr,nullptr,nullptr,p).empty());
    // Explicit oversized requests survive planning for the caller to reject.
    assert(glm5_cache_budget(30*G,6*G,2*G,80*G,p)==32*G);
    for (auto s:{"", "no", "10", "-1"})
        assert(!glm5_memory_policy(s,nullptr,nullptr,nullptr,nullptr,p).empty());
    for (auto s:{"-1","0","12MB","9999999999999999999999999"})
        assert(!glm5_memory_policy(nullptr,s,nullptr,nullptr,nullptr,p).empty());
    for (auto s:{"nan","inf","0","-4","40x"})
        assert(!glm5_memory_policy(nullptr,nullptr,s,nullptr,nullptr,p).empty());
    for (auto s:{"nan","inf","0.99","-1"})
        assert(!glm5_memory_policy(nullptr,nullptr,nullptr,nullptr,s,p).empty());
    std::cout << "glm5_memory_policy_test: OK\n";
}
