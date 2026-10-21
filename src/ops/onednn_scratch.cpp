// src/ops/onednn_scratch.cpp -- the oneDNN products' shared scratch (include/ie/onednn_scratch.hpp).

#include "ie/onednn_scratch.hpp"

#include "ie/allocator.hpp"

#include <dnnl_sycl.hpp>

#include <memory>
#include <mutex>

namespace ie {

namespace {

struct QueueScratch {
    std::mutex mu;
    void*      p = nullptr;
    size_t     bytes = 0;
};

QueueScratch& scratch_for(sycl::queue& q) {
    static std::mutex mu;
    static auto* by_queue = new std::unordered_map<sycl::queue, std::unique_ptr<QueueScratch>>();   // never destroyed:
    std::lock_guard<std::mutex> lk(mu);                                                            // no exit-order hazard
    auto& s = (*by_queue)[q];
    if (!s) s = std::make_unique<QueueScratch>();
    return *s;
}

}  // namespace

void onednn_scratch_desc(const dnnl::primitive_desc_base& pd, const dnnl::engine& eng, dnnl::memory& sp, size_t& bytes) {
    const dnnl::memory::desc sd = pd.scratchpad_desc();
    bytes = sd.get_size();
    if (bytes) sp = dnnl::memory(sd, eng, nullptr);
}

sycl::event onednn_execute(sycl::queue& q, const dnnl::stream& strm, const dnnl::primitive& prim,
                           std::unordered_map<int, dnnl::memory> args, dnnl::memory& sp, size_t bytes,
                           const std::vector<sycl::event>& deps, std::string& err) {
    if (!bytes) return dnnl::sycl_interop::execute(prim, strm, args, deps);
    QueueScratch& s = scratch_for(q);
    std::lock_guard<std::mutex> lk(s.mu);   // a growth never frees a buffer another thread's product was just handed
    if (s.bytes < bytes) {
        q.wait();                           // in order: every product that read the old buffer has finished
        if (s.p) sycl::free(s.p, q);
        s.p = malloc_device_checked(q, bytes);
        s.bytes = s.p ? bytes : 0;
        if (!s.p) {
            err = "oneDNN scratch: " + std::to_string(bytes) + " bytes not allocated";
            return {};
        }
    }
    sp.set_data_handle(s.p);
    args.emplace(DNNL_ARG_SCRATCHPAD, sp);
    return dnnl::sycl_interop::execute(prim, strm, args, deps);
}

}  // namespace ie
