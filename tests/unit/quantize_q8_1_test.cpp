// Actual quantize_q8_1 correctness gate. The frozen one-SG GPU oracle
// preserves the 2026-09-10 baseline bytes. Independent host math checks
// finite data, with a bounded exception for host/device tie rounding.
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/kernel_profiler.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

using Block = ie::block_q8_1x;
using Fn = sycl::event (*)(sycl::queue&,const sycl::half*,void*,uint32_t,const std::vector<sycl::event>&);
static sycl::event frozen(sycl::queue& q,const sycl::half* x,void* out_q8,uint32_t K,const std::vector<sycl::event>& deps) {
    const uint32_t n_blocks = K / 32;
    auto* out = static_cast<Block*>(out_q8);
    return ie::ps(q, "quant_q8_1", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_blocks) * 32, 32),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const uint32_t b = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const float v = float(x[b * 32 + lane]);
            const float amax = sycl::reduce_over_group(sg, sycl::fabs(v), sycl::maximum<float>());
            const float d = amax / 127.0f;
            const float inv = (amax > 0.f) ? 127.0f / amax : 0.f;
            const int32_t qi = int32_t(sycl::round(v * inv));
            out[b].qs[lane] = int8_t(qi);
            const int32_t qsum = sycl::reduce_over_group(sg, qi, sycl::plus<int32_t>());
            if (lane == 0) { out[b].d = d; out[b].s = d * float(qsum); }
        });
    });
}

static const std::array<Fn,2> functions{frozen,ie::quantize_q8_1};
static void require(bool ok,const char* what) { if(!ok) throw std::runtime_error(what); }
static sycl::half half_bits(uint16_t bits) { sycl::half h; std::memcpy(&h,&bits,2); return h; }
static std::vector<sycl::half> data(size_t n,unsigned seed) {
    std::vector<sycl::half> x(n);
    const uint16_t edge[]{0,0x8000,1,0x8001,0x3ff,0x83ff,0x400,0x8400,0x7bff,0xfbff,0x3c00,0xbc00,0x3555,0xb555};
    for(size_t i=0;i<n;++i) {
        uint32_t z=uint32_t(i)*1664525u+seed*1013904223u;
        uint16_t bits=uint16_t(z^(z>>16));
        if((bits&0x7c00)==0x7c00) bits^=0x400;
        if((i/32)%7==0) bits=edge[(i+seed)%14];
        if((i/32)%7==1) bits=(i&1)?0x8000:0;
        if((i/32)%7==2) bits=uint16_t((i%1023)+1)|((i&1)?0x8000:0);
        x[i]=half_bits(bits);
    }
    return x;
}
static void host_check(const sycl::half* x,const Block* got,size_t blocks) {
    for(size_t b=0;b<blocks;++b) {
        float a=0; for(unsigned l=0;l<32;++l) a=std::max(a,std::abs(float(x[32*b+l])));
        const float d=a/127.f,inv=a>0?127.f/a:0;
        int sum=0;
        for(unsigned l=0;l<32;++l) {
            const float product=float(x[32*b+l])*inv;
            const int qi=int(std::round(product));
            const int observed=got[b].qs[l];
            // Host and device reciprocal approximations may straddle exact
            // half-integers. Only accept the adjacent quant at that boundary;
            // the independent frozen GPU oracle still requires exact bytes.
            const float tie=std::floor(std::abs(product))+0.5f;
            const float bound=4*std::numeric_limits<float>::epsilon()*std::max(1.f,std::abs(product));
            const bool adjacent=observed==int(std::floor(product)) || observed==int(std::ceil(product));
            const bool rounding_boundary=adjacent && std::abs(std::abs(product)-tie)<=bound;
            if(observed!=qi && !rounding_boundary) {
                std::fprintf(stderr,"host mismatch block=%zu lane=%u value=%.9g product=%.9g expected=%d got=%d\n",b,l,float(x[32*b+l]),product,qi,observed);
                throw std::runtime_error("host quant mismatch");
            }
            sum+=observed;
        }
        require(std::abs(got[b].d-d)<=std::max(1e-12f,std::abs(d)*2e-7f),"host scale mismatch");
        const float s=d*sum;
        require(std::abs(got[b].s-s)<=std::max(1e-12f,std::abs(s)*3e-7f),"host sum mismatch");
    }
}
static void correctness(sycl::queue& q) {
    sycl::queue producer(q.get_context(),q.get_device());
    size_t cases=0;
    for(uint32_t K: {0u,32u,64u,96u,128u,160u,224u,256u,288u,480u,512u,544u,992u,1024u,1056u,4096u,7168u,18432u,65536u,1048576u}) {
        for(unsigned offset: {0u,1u,3u,15u}) {
            const size_t blocks=K/32, count=blocks+8;
            auto input=data(size_t(K)+offset+32,K+offset);
            auto* dx=sycl::malloc_device<sycl::half>(input.size(),q);
            auto* dy=sycl::malloc_device<Block>(count,q);
            std::vector<Block> reference(count),actual(count);
            std::memset(reference.data(),0xa5,count*sizeof(Block));
            for(size_t f=0;f<functions.size();++f) {
                auto fill=producer.memset(dy,0xa5,count*sizeof(Block));
                auto copy=producer.memcpy(dx,input.data(),input.size()*2);
                auto e=functions[f](q,dx+offset,dy+4,K,{fill,copy});
                q.submit([&](sycl::handler& h){h.depends_on(e);h.memcpy(actual.data(),dy,count*sizeof(Block));}).wait_and_throw();
                if(f==0) { reference=actual; host_check(input.data()+offset,actual.data()+4,blocks); }
                require(std::memcmp(reference.data(),actual.data(),count*sizeof(Block))==0,"frozen byte/guard mismatch");
                const auto* bytes=reinterpret_cast<const uint8_t*>(actual.data());
                for(size_t j=0;j<4*sizeof(Block);++j) {
                    require(bytes[j]==0xa5,"prefix guard changed");
                    require(bytes[(blocks+4)*sizeof(Block)+j]==0xa5,"suffix guard changed");
                }
                ++cases;
            }
            sycl::free(dx,q); sycl::free(dy,q);
        }
    }
    // Every finite binary16 bit pattern, including both signed zeros.
    std::vector<sycl::half> exhaustive;
    for(unsigned bits=0;bits<65536;++bits) if((bits&0x7c00)!=0x7c00) exhaustive.push_back(half_bits(bits));
    auto* dx=sycl::malloc_device<sycl::half>(exhaustive.size(),q);
    auto* dy=sycl::malloc_device<Block>(exhaustive.size()/32,q);
    auto copy=q.memcpy(dx,exhaustive.data(),exhaustive.size()*2);
    std::vector<Block> ref(exhaustive.size()/32),got(ref.size());
    for(size_t f=0;f<functions.size();++f) {
        auto e=functions[f](q,dx,dy,exhaustive.size(),{copy});
        q.submit([&](sycl::handler& h){h.depends_on(e);h.memcpy(got.data(),dy,got.size()*sizeof(Block));}).wait_and_throw();
        if(f==0) {ref=got;host_check(exhaustive.data(),got.data(),got.size());}
        require(std::memcmp(ref.data(),got.data(),got.size()*sizeof(Block))==0,"finite exhaustive mismatch");
    }
    sycl::free(dx,q);sycl::free(dy,q);
    // Explicit empty dependency chain across queues, with null pointers.
    auto* marker=sycl::malloc_device<int>(1,q);int answer=0;
    int expected=73;
    for(auto fn:functions) {
        ++expected;
        auto delay=producer.submit([&](sycl::handler& h){h.host_task([]{std::this_thread::sleep_for(std::chrono::milliseconds(20));});});
        auto set=producer.submit([&](sycl::handler& h){h.depends_on(delay);h.single_task([=]{*marker=expected;});});
        auto empty=fn(q,nullptr,nullptr,0,{set});
        producer.submit([&](sycl::handler& h){h.depends_on(empty);h.memcpy(&answer,marker,sizeof(int));}).wait_and_throw();
        require(answer==expected,"empty dependency lost");
    }
    sycl::free(marker,q);
    std::printf("correctness cases=%zu finite_half_patterns=%zu guards=PASS dependencies=PASS empty=PASS\n",cases,exhaustive.size());
}
static void stress(sycl::queue& q) {
    constexpr uint32_t K=32*257,iterations=2048;
    auto* dx=sycl::malloc_device<sycl::half>(K,q);
    auto* ref=sycl::malloc_device<Block>(K/32,q);
    auto* got=sycl::malloc_device<Block>(K/32,q);
    auto* errors=sycl::malloc_device<unsigned>(iterations,q);
    std::vector<unsigned> host_errors(iterations);
    sycl::event prior=q.memset(errors,0,iterations*sizeof(unsigned));
    for(unsigned i=0;i<iterations;++i) {
        auto p=q.submit([&](sycl::handler& h){h.depends_on(prior);h.parallel_for(sycl::range<1>(K),[=](sycl::id<1> id){dx[id]=sycl::half(float(int((id[0]*13+i*17)%2047)-1023)*0.0625f);});});
        auto a=frozen(q,dx,ref,K,{p});
        auto b=ie::quantize_q8_1(q,dx,got,K,{p});
        prior=q.submit([&](sycl::handler& h){h.depends_on({a,b});h.single_task([=]{
            const auto* r=reinterpret_cast<const uint8_t*>(ref);const auto* g=reinterpret_cast<const uint8_t*>(got);
            for(unsigned j=0;j<K/32*sizeof(Block);++j) if(r[j]!=g[j]) {errors[i]=1;break;}
        });});
    }
    q.submit([&](sycl::handler& h){h.depends_on(prior);h.memcpy(host_errors.data(),errors,iterations*sizeof(unsigned));}).wait_and_throw();
    for(unsigned e:host_errors) require(e==0,"repeated buffer reuse mismatch");
    sycl::free(dx,q);sycl::free(ref,q);sycl::free(got,q);sycl::free(errors,q);
    std::printf("stress iterations=%u shared_input_and_output_reuse=PASS\n",iterations);
}
int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        std::printf("device=%s out_of_order=%d\n",q.get_device().get_info<sycl::info::device::name>().c_str(),!q.is_in_order());
        correctness(q);
        stress(q);
        q.wait_and_throw();
        std::puts("PASS");
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what());
        return 1;
    }
}
