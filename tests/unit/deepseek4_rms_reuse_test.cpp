#include "ie/deepseek4_ops.hpp"
#include <array>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <vector>
namespace rms_oracle {
constexpr unsigned kWG=256,kSG=16;
sycl::event frozen_rms(sycl::queue& q,
                                    const float* x, float* y,
                                    uint32_t n_rows, uint32_t hidden,
                                    float eps,
                                    const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(n_rows) * kWG, kWG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const float* xr = x + size_t(row) * hidden;
            float*       yr = y + size_t(row) * hidden;

            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            uint32_t i = lid;
            for (; i + 3u * kWG < hidden; i += 4u * kWG) {
                const float v0 = xr[i], v1 = xr[i + kWG];
                const float v2 = xr[i + 2u * kWG], v3 = xr[i + 3u * kWG];
                s0 += v0 * v0; s1 += v1 * v1; s2 += v2 * v2; s3 += v3 * v3;
            }
            for (; i < hidden; i += kWG) { const float v = xr[i]; s0 += v * v; }
            const float ss = sycl::reduce_over_group(it.get_group(),
                                                     (s0 + s1) + (s2 + s3),
                                                     sycl::plus<float>());
            // Reference: x * rsqrt(x.square().mean(-1) + eps)
            const float r = sycl::rsqrt(ss / float(hidden) + eps);
            for (uint32_t j = lid; j < hidden; j += kWG) yr[j] = xr[j] * r;
        });
    });
}
} // namespace rms_oracle

namespace {
struct Mem {
    sycl::queue& q; std::vector<void*> ptrs;
    float* get(size_t n){auto p=sycl::malloc_device<float>(n,q);if(!p)throw std::bad_alloc();ptrs.push_back(p);return p;}
    ~Mem(){q.wait();for(auto p:ptrs)sycl::free(p,q);}
};
void check(sycl::queue& q,unsigned rows,unsigned H,bool alias,unsigned shift,unsigned pattern,float eps){
    constexpr size_t G=16;size_t n=size_t(rows)*H;
    std::vector<float> src(n),expected(n+2*G),got(expected.size());
    for(size_t i=0;i<n;++i){float v=float(int((i*83+7)%257)-128)/137;
        if(pattern==1)v=0; if(pattern==2)v*=1e-30f; if(pattern==3)v*=1e10f;
        if(pattern==5)v*=std::numeric_limits<float>::denorm_min();
        if(pattern==4){const float special[]={0.f,-0.f,1.f,-1.f,std::numeric_limits<float>::max(),std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()};v=special[i%7];}
        src[i]=v;
    }
    Mem m{q,{}};auto xb=m.get(n+2*G+shift),x=xb+G+shift;
    auto yb=alias?xb:m.get(n+2*G+shift),y=yb+G+shift,ref=m.get(n+2*G);
    auto fill=q.fill(xb,12345.f,n+2*G+shift);
    auto copy=q.submit([&](sycl::handler& h){h.depends_on(fill);h.memcpy(x,src.data(),n*4);});
    std::vector<sycl::event> deps{copy,q.fill(ref,12345.f,n+2*G)};
    if(!alias)deps.push_back(q.fill(yb,12345.f,n+2*G+shift));
    auto e=rms_oracle::frozen_rms(q,x,ref+G,rows,H,eps,deps);
    q.submit([&](sycl::handler& h){h.depends_on(e);h.memcpy(expected.data(),ref,expected.size()*4);}).wait_and_throw();
    // Independent double-precision reference for finite, non-underflow cases.
    if(pattern==0||pattern==3)for(unsigned r=0;r<rows;++r){
        double ss=0;for(unsigned j=0;j<H;++j){double v=src[size_t(r)*H+j];ss+=v*v;}
        double scale=1/std::sqrt(ss/H+eps);
        for(unsigned j=0;j<H;++j){double v=double(src[size_t(r)*H+j])*scale;
            if(!std::isfinite(expected[G+size_t(r)*H+j])||std::fabs(double(expected[G+size_t(r)*H+j])-v)>5e-6*(1+std::fabs(v)))throw std::runtime_error("CPU RMS bound");}
    }
    auto poison=q.fill(x,0.f,n);
    poison.wait_and_throw();
    sycl::queue producer(q.get_context(),q.get_device());
    auto delay=producer.submit([&](sycl::handler& h){h.depends_on(poison);h.host_task([]{std::this_thread::sleep_for(std::chrono::milliseconds(20));});});
    auto restore=producer.submit([&](sycl::handler& h){h.depends_on(delay);h.memcpy(x,src.data(),n*4);});deps.push_back(restore);
#ifdef IE_RMS_NEGATIVE_CONTROL
    deps.clear();
#endif
    e=ie::ds4_unweighted_rms_norm(q,x,y,rows,H,eps,deps);
    q.submit([&](sycl::handler& h){h.depends_on(e);h.memcpy(got.data(),yb+shift,got.size()*4);}).wait_and_throw();
    producer.wait_and_throw();
    if(std::memcmp(got.data(),expected.data(),got.size()*4)){
        std::fprintf(stderr,"mismatch rows=%u H=%u alias=%d shift=%u pattern=%u eps=%g\n",rows,H,alias,shift,pattern,eps);
        throw std::runtime_error("RMS result or guard differs from frozen GPU");
    }
}
}
int main(){try{
    unsigned cards=0,cases=0;
    for(auto d:sycl::device::get_devices(sycl::info::device_type::gpu)){
        if(d.get_info<sycl::info::device::name>().find("B70")==std::string::npos)continue;
        ++cards;sycl::queue q(d);
        for(unsigned H:{1u,128u,255u,256u,257u,511u,512u,513u,1024u,4096u})
            for(bool alias:{false,true}){check(q,3,H,alias,1,0,1e-20f);++cases;}
        for(unsigned rows:{1u,31u,32u,33u,257u})for(bool alias:{false,true}){
            check(q,rows,512,alias,2,0,1e-5f);++cases;}
        for(unsigned pattern:{1u,2u,3u,4u,5u})for(bool alias:{false,true}){
            check(q,3,512,alias,1,pattern,1e-20f);++cases;}
        for(bool alias:{false,true}){check(q,3,512,alias,0,1,0.f);++cases;}
        sycl::queue ordered(d,sycl::property::queue::in_order{});
        for(bool alias:{false,true}){check(ordered,257,512,alias,0,0,1e-20f);++cases;}
    }if(!cards)throw std::runtime_error("no B70");std::printf("PASS %u exact/guard/alias/dependency cases on %u devices\n",cases,cards);
}catch(std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
