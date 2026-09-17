#include "ie/ds4_decode_gemv.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <sycl/sycl.hpp>
#include <vector>
namespace dq_oracle {
sycl::event frozen_dequant(sycl::queue& q, const int8_t* qs, const sycl::half* d,
                              sycl::half* out, uint32_t K, uint32_t N,
                              const std::vector<sycl::event>& deps) {
    const uint32_t nb = K / 32;
    constexpr uint32_t WG = 256;
    // 2-D, one row per group-1 index, rather than a flat 1-D index divided by K.
    // The flat form spent a 64-BIT INTEGER DIVISION per element to recover the
    // row, and Xe has no hardware 64-bit divide — IGC emulates it.  At the real
    // per-card set that is 3.4e9 emulated divisions per chunk on a kernel whose
    // entire job is to move 10.45 GB.  The row now comes from the launch
    // geometry, `k` is the fast axis so both the int8 read and the fp16 write
    // stay coalesced, and the scale index is a shift.
    //
    // The value written is unchanged, bit for bit: same fp16 scale, same int8,
    // same single fp32 multiply, same one rounding to fp16.  Only the address
    // arithmetic moved.
    const uint32_t kg = (K + WG - 1) / WG * WG;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(N, kg), sycl::range<2>(1, WG)),
                       [=](sycl::nd_item<2> it) {
            const uint32_t n = uint32_t(it.get_global_id(0));
            const uint32_t k = uint32_t(it.get_global_id(1));
            if (k >= K) return;
            const uint64_t i = uint64_t(n) * K + k;
            out[i] = sycl::half(float(d[uint64_t(n) * nb + (k >> 5)]) * float(qs[i]));
        });
    });
}

}

namespace {
struct Memory {
    sycl::queue& q; std::vector<void*> ptrs;
    template<class T> T* get(size_t n) { auto p=sycl::malloc_device<T>(n,q); if(!p)throw std::bad_alloc(); ptrs.push_back(p); return p; }
    ~Memory(){q.wait();for(auto p:ptrs)sycl::free(p,q);}
};
void check(sycl::queue& q,unsigned K,unsigned N,unsigned shift,bool exhaustive){
    constexpr size_t G=16; const size_t count=size_t(K)*N,ns=count/32;
    std::vector<int8_t> bytes(count);std::vector<sycl::half> scales(ns),want(count+2*G),got(want.size());
    for(size_t i=0;i<count;++i)bytes[i]=int8_t(int((i*71+13)%256)-128);
    for(size_t i=0;i<ns;++i){
        if(exhaustive){uint16_t bits=uint16_t(i);std::memcpy(&scales[i],&bits,2);}
        else scales[i]=sycl::half(float(int((i*31)%257)-128)/512);
    }
    Memory m{q,{}};auto xb=m.get<int8_t>(count+2*G+shift),x=xb+G+shift;
    auto db=m.get<sycl::half>(ns+2*G+shift),d=db+G+shift;
    auto yb=m.get<sycl::half>(count+2*G+shift),y=yb+G+shift,ref=m.get<sycl::half>(count+2*G);
    std::vector<sycl::event> deps{q.fill(yb,sycl::half(123),count+2*G+shift),q.fill(ref,sycl::half(123),count+2*G),q.memcpy(x,bytes.data(),count),q.memcpy(d,scales.data(),ns*2)};
    auto e=dq_oracle::frozen_dequant(q,x,d,ref+G,K,N,deps);
    q.submit([&](sycl::handler& h){h.depends_on(e);h.memcpy(want.data(),ref,want.size()*2);}).wait_and_throw();
    // Independent scalar host product/half rounding, allowing only NaN payload differences.
    for(size_t i=0;i<count;++i){sycl::half v=sycl::half(float(scales[i/32])*float(bytes[i]));
        if(std::memcmp(&v,&want[G+i],2) && !(std::isnan(float(v))&&std::isnan(float(want[G+i]))))throw std::runtime_error("CPU half oracle");}
    auto poison=q.fill(x,int8_t(0),count);
    poison.wait_and_throw();
    // Separate queue and delayed fresh input ensure omission of deps is observable.
    sycl::queue producer(q.get_context(),q.get_device());
    auto delay=producer.submit([&](sycl::handler& h){h.depends_on(poison);h.host_task([]{std::this_thread::sleep_for(std::chrono::milliseconds(10));});});
    auto restore=producer.submit([&](sycl::handler& h){h.depends_on(delay);h.memcpy(x,bytes.data(),count);});deps.push_back(restore);
    #ifdef IE_DEQUANT_NEGATIVE_CONTROL
    deps.clear();
#endif
    e=ie::ds4_q8_soa_to_f16(q,x,d,y,K,N,deps);
    q.submit([&](sycl::handler& h){h.depends_on(e);h.memcpy(got.data(),yb+shift,got.size()*2);}).wait_and_throw();
    producer.wait_and_throw();
    if(std::memcmp(got.data(),want.data(),got.size()*2))throw std::runtime_error("dequant output/guard/fresh producer mismatch");
}
void reuse(sycl::queue& q){
    constexpr unsigned K=96,N=17;Memory m{q,{}};auto x=m.get<int8_t>(K*N);auto d=m.get<sycl::half>(K*N/32);auto a=m.get<sycl::half>(K*N),b=m.get<sycl::half>(K*N);
    std::vector<sycl::half> ah(K*N),bh(K*N);sycl::event prev=q.fill(d,sycl::half(.125),K*N/32);
    for(unsigned i=0;i<1024;++i){
        auto fresh=q.submit([&](sycl::handler& h){h.depends_on(prev);h.fill(x,int8_t(int(i%255)-127),K*N);});
        auto expected=dq_oracle::frozen_dequant(q,x,d,a,K,N,{fresh});
        auto actual=ie::ds4_q8_soa_to_f16(q,x,d,b,K,N,{fresh});
        prev=q.submit([&](sycl::handler& h){h.depends_on({expected,actual});h.single_task([]{});});
        if(i%127==0||i==1023){prev.wait_and_throw();q.memcpy(ah.data(),a,ah.size()*2).wait();q.memcpy(bh.data(),b,bh.size()*2).wait();if(ah!=bh)throw std::runtime_error("dequant reuse mismatch");}
    }
}
}
int main(){try{
    unsigned cards=0,cases=0;
    for(auto dev:sycl::device::get_devices(sycl::info::device_type::gpu)){
        if(dev.get_info<sycl::info::device::name>().find("B70")==std::string::npos)continue;
        ++cards;sycl::queue q(dev);
        for(auto shape:{std::array<unsigned,2>{32,1},{96,17},{224,3},{256,32},{288,257},{1024,256},{4096,32}})
            for(unsigned shift:{0u,1u,3u}){check(q,shape[0],shape[1],shift,false);++cases;}
        check(q,32,65536,1,true);++cases;reuse(q);
        sycl::queue ordered(dev,sycl::property::queue::in_order{});check(ordered,96,17,1,false);++cases;
    }
    if(!cards)throw std::runtime_error("no B70");
    std::printf("PASS %u cases, all 65536 half scale patterns, 1024 reuse rounds/device on %u devices\n",cases,cards);
}catch(std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
