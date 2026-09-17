#include "ie/deepseek4.hpp"
#include <sycl/sycl.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
// Frozen GPU ordering plus the separate bounded CPU oracle below.
sycl::event frozen_mix(sycl::queue& q,
                       const float* streams, const float* post, const float* comb,
                       const float* sub, float* out,
                       uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                       const std::vector<sycl::event>& deps) {
    const uint64_t n = uint64_t(n_tokens) * hc_mult * hidden;
    constexpr uint32_t WG = 256;
    return q.submit( [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>((n + WG - 1) / WG * WG, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            const uint64_t d  = i % hidden;
            const uint64_t hh = (i / hidden) % hc_mult;
            const uint64_t t  = i / (uint64_t(hc_mult) * hidden);
            // comb is consumed TRANSPOSED: Σ_j comb[t, j, h] · streams[t, j, d].
            float s = 0.f;
            for (uint32_t j = 0; j < hc_mult; ++j)
                s = sycl::fma(comb[t * hc_mult * hc_mult + uint64_t(j) * hc_mult + hh],
                              streams[t * hc_mult * hidden + uint64_t(j) * hidden + d], s);
            out[i] = sycl::fma(post[t * hc_mult + hh], sub[t * hidden + d], s);
        });
    });
}

struct Memory {
    sycl::queue& q;
    std::vector<void*> allocations;
    template<class T> T* get(size_t n) {
        T* p = sycl::malloc_device<T>(n, q);
        if (!p) throw std::bad_alloc();
        allocations.push_back(p);
        return p;
    }
    ~Memory() { q.wait(); for (void* p : allocations) sycl::free(p, q); }
};

void check(sycl::queue& q, unsigned T, unsigned H, unsigned HC, bool inplace, unsigned shift) {
    constexpr size_t guard = 16;
    const size_t n = size_t(T) * H * HC;
    std::vector<float> x(n), p(size_t(T)*HC), c(size_t(T)*HC*HC), sub(size_t(T)*H);
    for (size_t i=0; i<n; ++i) x[i] = float(int((i*73+19)%211)-105)/113.f;
    for (size_t i=0; i<p.size(); ++i) p[i] = float(int(i%17)-8)/31.f;
    for (size_t i=0; i<c.size(); ++i) c[i] = float(int((i*43+7)%197)-98)/509.f;
    for (size_t i=0; i<sub.size(); ++i) sub[i] = float(int(i%11)-5)/7.f;
    // Independent scalar reference, including the TRANSPOSED comb indexing.
    std::vector<float> ref(n+guard*2, 12345.f), got(ref.size());
    for (size_t t=0; t<T; ++t) for (unsigned h=0; h<HC; ++h) for (unsigned d=0; d<H; ++d) {
        float acc=0;
        for (unsigned j=0; j<HC; ++j)
            acc=std::fma(c[(t*HC+j)*HC+h], x[(t*HC+j)*H+d], acc);
        ref[guard+(t*HC+h)*H+d]=std::fma(p[t*HC+h],sub[t*H+d],acc);
    }
    Memory mem{q,{}};
    auto xb=mem.get<float>(n+guard*2+shift), xi=xb+guard+shift;
    auto pi=mem.get<float>(p.size()), ci=mem.get<float>(c.size()), si=mem.get<float>(sub.size());
    auto yb=inplace?xb:mem.get<float>(n+guard*2+shift), y=yb+guard+shift;
    auto fill=q.fill(xb,12345.f,n+guard*2+shift);
    auto copy=q.submit([&](sycl::handler& h){h.depends_on(fill);h.memcpy(xi,x.data(),n*sizeof(float));});
    std::vector<sycl::event> deps{copy,q.memcpy(pi,p.data(),p.size()*sizeof(float)),
        q.memcpy(ci,c.data(),c.size()*sizeof(float)),q.memcpy(si,sub.data(),sub.size()*sizeof(float))};
    if (!inplace) deps.push_back(q.fill(yb,12345.f,n+guard*2+shift));
    auto reference=mem.get<float>(ref.size());
    auto rf=q.fill(reference,12345.f,ref.size());
    auto rd=deps; rd.push_back(rf);
    auto re=frozen_mix(q,xi,pi,ci,si,reference+guard,T,H,HC,rd);
    std::vector<float> gpu_ref(ref.size());
    q.submit([&](sycl::handler& h){h.depends_on(re);h.memcpy(gpu_ref.data(),reference,gpu_ref.size()*sizeof(float));}).wait_and_throw();
    for(size_t i=guard;i<guard+n;++i) {
        const float bound=2e-6f*float(HC+1)*(1.f+std::fabs(ref[i]));
        if(!std::isfinite(gpu_ref[i]) || std::fabs(gpu_ref[i]-ref[i])>bound)
            throw std::runtime_error("frozen GPU output violates CPU bound");
    }
    ref.swap(gpu_ref);
    // Poison then restore after reference completion. The candidate must honor
    // this fresh dependency, including on an out-of-order queue.
    auto poison=q.fill(xi,0.f,n);
    auto restore=q.submit([&](sycl::handler& h){h.depends_on(poison);h.memcpy(xi,x.data(),n*sizeof(float));});
    deps.push_back(restore);
    auto done=ie::ds4_hc_mix(q,xi,pi,ci,si,y,T,H,HC,deps);
    q.submit([&](sycl::handler& h){h.depends_on(done);h.memcpy(got.data(),yb+shift,got.size()*sizeof(float));}).wait_and_throw();
    if (std::memcmp(got.data(),ref.data(),got.size()*sizeof(float))) {
        size_t i=0; while(i<got.size() && std::memcmp(&got[i],&ref[i],sizeof(float))==0) ++i;
        std::fprintf(stderr,"mismatch T=%u H=%u HC=%u inplace=%d shift=%u i=%zu got=%g ref=%g\n",T,H,HC,inplace,shift,i,got[i],ref[i]);
        throw std::runtime_error("mix output or guard differs from independent reference");
    }
}
}

int main() { try {
    unsigned devices=0,cases=0;
    for (const auto& d:sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (d.get_info<sycl::info::device::name>().find("B70")==std::string::npos) continue;
        ++devices; sycl::queue q(d);
        for (auto dims : {std::array<unsigned,3>{std::numeric_limits<unsigned>::max(),std::numeric_limits<unsigned>::max(),1},
                          {1,1,std::numeric_limits<unsigned>::max()}}) {
            bool rejected=false;
            try { ie::ds4_hc_mix(q,nullptr,nullptr,nullptr,nullptr,nullptr,dims[0],dims[1],dims[2]); }
            catch(const std::invalid_argument&) { rejected=true; }
            if(!rejected) throw std::runtime_error("overflowing tensor span was accepted");
            ++cases;
        }
        std::printf("device=%s\n",d.get_info<sycl::info::device::name>().c_str());
        // Put the original cross-workgroup in-place race first.
        for(bool alias:{false,true}) {check(q,1,4096,4,alias,0);++cases;}
        for(unsigned hc:{1u,2u,3u,4u,5u,16u,17u})
            for(unsigned h:{1u,255u,256u,257u,4096u})
                for(bool alias:{false,true}) {check(q,2,h,hc,alias,1);++cases;}
        for(unsigned t:{63u,64u,65u,512u})
            for(bool alias:{false,true}) {check(q,t,4096,4,alias,0);++cases;}
        sycl::queue ordered(d,sycl::property::queue::in_order{});
        for(unsigned hc:{3u,4u}) for(bool alias:{false,true}) {check(ordered,65,257,hc,alias,0);++cases;}
    }
    if(!devices) throw std::runtime_error("no B70 available");
    std::printf("PASS %u cases on %u devices\n",cases,devices);
} catch(const std::exception& e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;} }
