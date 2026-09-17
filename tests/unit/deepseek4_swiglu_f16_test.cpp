#include "ie/deepseek4_ops.hpp"
#include "ie/ops.hpp"
#include <sycl/sycl.hpp>
#include <bit>
#include <cmath>
#include <cstdio>
#include <future>
#include <limits>
#include <stdexcept>
#include <vector>

static void check_case(sycl::queue& q, size_t n, unsigned shift, float limit) {
    std::vector<float> gates(n), ups(n);
    const float edge[] = {0.f,-0.f,1e-40f,-1e-40f,7.f,std::nextafter(7.f,0.f),std::nextafter(7.f,8.f),-7.f,65504.f,-65504.f,1e30f,-1e30f,INFINITY,-INFINITY,NAN};
    for (size_t i=0;i<n;++i) {
        // All half bit patterns promoted exactly, with interleaved fp32-only
        // rounding/clamp boundaries and nonfinite values.
        gates[i]=i%3 ? float(std::bit_cast<sycl::half>(uint16_t(i))) : edge[(i/3)%std::size(edge)];
        ups[i]=i%5 ? float(std::bit_cast<sycl::half>(uint16_t(i*73+19))) : edge[(i/5)%std::size(edge)];
    }
    auto gb=sycl::malloc_device<float>(n+shift,q),ub=sycl::malloc_device<float>(n+shift+1,q),tmp=sycl::malloc_device<float>(n,q);
    auto yb=sycl::malloc_device<sycl::half>(n+shift+32,q),ref=sycl::malloc_device<sycl::half>(n,q);
    if (!gb||!ub||!tmp||!yb||!ref)throw std::runtime_error("allocation failed");
    auto g=gb+shift,u=ub+shift+1;auto y=yb+shift+16;
    auto a=q.memcpy(g,gates.data(),n*4),b=q.memcpy(u,ups.data(),n*4);
    auto old=ie::ds4_swiglu_clamped(q,g,u,tmp,n,limit,{a,b});
    auto done=ie::cast_fp32_to_fp16(q,tmp,ref,n,{old});done.wait_and_throw();
    // Fresh input dependencies on the out-of-order queue: consuming poison
    // rather than either restoration must be observable.
    auto poison_g=q.fill(g,123.f,n),poison_u=q.fill(u,-123.f,n);
    a=q.submit([&](sycl::handler& h){h.depends_on(poison_g);h.memcpy(g,gates.data(),n*4);});
    b=q.submit([&](sycl::handler& h){h.depends_on(poison_u);h.memcpy(u,ups.data(),n*4);});
    auto c=q.fill(yb,sycl::half(12345.f),n+shift+32);
    ie::ds4_swiglu_clamped_to_f16(q,g,u,y,n,limit,{a,b,c}).wait_and_throw();
    std::vector<sycl::half> want(n),got(n+shift+32);
    q.memcpy(want.data(),ref,n*2).wait_and_throw();q.memcpy(got.data(),yb,got.size()*2).wait_and_throw();
    for (size_t i=0;i<n;++i)if (std::bit_cast<uint16_t>(want[i])!=std::bit_cast<uint16_t>(got[i+shift+16])) {
        std::fprintf(stderr,"mismatch n=%zu shift=%u limit=%g i=%zu want=%04x got=%04x\n",n,shift,limit,i,std::bit_cast<uint16_t>(want[i]),std::bit_cast<uint16_t>(got[i+shift+16]));throw std::runtime_error("not bit-exact");
    }
    for (size_t i=0;i<got.size();++i)if ((i<shift+16||i>=shift+16+n)&&got[i]!=sycl::half(12345.f))throw std::runtime_error("output guard changed");
    sycl::free(gb,q);sycl::free(ub,q);sycl::free(tmp,q);sycl::free(yb,q);sycl::free(ref,q);
}
static void empty_and_invalid(sycl::queue& q) {
    std::promise<void> promise;auto ready=promise.get_future().share();
    auto dep=q.submit([&](sycl::handler& h){h.host_task([ready]{ready.wait();});});
    auto e=ie::ds4_swiglu_clamped_to_f16(q,nullptr,nullptr,nullptr,0,7.f,{dep});
    bool early=e.get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::complete;
    promise.set_value();e.wait_and_throw();if (early)throw std::runtime_error("empty lost dependency");
    bool rejected=false;try {ie::ds4_swiglu_clamped_to_f16(q,nullptr,nullptr,nullptr,1,7.f,{});}catch (const std::invalid_argument&){rejected=true;}
    if (!rejected)throw std::runtime_error("null operands accepted");
    // Nonnull operands isolate the size check from null-pointer validation.
    float input=1.f;sycl::half output=0.f;rejected=false;
    try {ie::ds4_swiglu_clamped_to_f16(q,&input,&input,&output,std::numeric_limits<size_t>::max(),7.f,{});}catch (const std::invalid_argument&){rejected=true;}
    if (!rejected)throw std::runtime_error("oversized span accepted");
}
int main(){try {unsigned cases=0,devices=0;for (auto d:sycl::device::get_devices(sycl::info::device_type::gpu)){
    if (d.get_info<sycl::info::device::name>().find("B70")==std::string::npos)continue;++devices;std::printf("device=%s\n",d.get_info<sycl::info::device::name>().c_str());
    for (bool in_order:{false,true}){sycl::queue q(d,in_order?sycl::property_list{sycl::property::queue::in_order{}}:sycl::property_list{});
        for (size_t n:{size_t(1),size_t(255),size_t(256),size_t(257),size_t(196611)})for (unsigned shift:{0u,1u})for (float limit:{0.f,7.f,-7.f,INFINITY,NAN}){check_case(q,n,shift,limit);++cases;}
        empty_and_invalid(q);++cases;
    }}if (!devices)throw std::runtime_error("no B70 devices tested");std::printf("PASS %u cases on %u devices\n",cases,devices);
}catch (const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
