#include "ie/deepseek4_attn.hpp"
#include <algorithm>
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
namespace rope_oracle {
constexpr uint32_t kWG=256;
sycl::event frozen_rope(sycl::queue& q,
                           const float* x, const float* cos_in, const float* sin_in,
                           float* y,
                           uint32_t n_tokens, uint32_t n_heads,
                           uint32_t head_dim, uint32_t rope_dim,
                           float sin_sign,
                           const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t half   = rope_dim / 2u;
        const uint32_t nope   = head_dim - rope_dim;
        const uint32_t total  = n_tokens * n_heads * head_dim;
        const uint32_t global = ((total + kWG - 1) / kWG) * kWG;
        h.parallel_for(sycl::nd_range<1>(global, kWG), [=](sycl::nd_item<1> it) {
            const uint32_t i = uint32_t(it.get_global_id(0));
            if (i >= total) return;
            const uint32_t d = i % head_dim;
            const uint32_t t = i / (n_heads * head_dim);
            if (d < nope) {                       // untouched "nope" channels
                if (y != x) y[i] = x[i];
                return;
            }
            const uint32_t rd = d - nope;         // index inside the rope slice
            if (rd & 1u) return;                  // the even lane owns the whole pair
            const uint32_t p = rd / 2u;           // interleaved pair index
            const float c = cos_in[size_t(t) * half + p];
            const float s = sin_in[size_t(t) * half + p] * sin_sign;
            // rotate_half pairs (2p, 2p+1):
            //   y[2p]   = x[2p]  *cos - x[2p+1]*sin
            //   y[2p+1] = x[2p+1]*cos + x[2p]  *sin
            // Both halves are read before either is written, so x and y may
            // alias; a per-element work-item split could not (the partner lane
            // would race with the write).
            const float a = x[i];
            const float b = x[i + 1];
            y[i]     = a * c - b * s;
            y[i + 1] = b * c + a * s;
        });
    });
}

} // namespace rope_oracle

namespace {
struct Mem {
    sycl::queue& q; std::vector<void*> ptrs;
    float* get(size_t n) {auto p=sycl::malloc_device<float>(n,q);if(!p)throw std::bad_alloc();ptrs.push_back(p);return p;}
    ~Mem(){q.wait();for(auto p:ptrs)sycl::free(p,q);}
};
void check(sycl::queue& q,unsigned T,unsigned H,unsigned D,unsigned R,bool alias,unsigned shift,float sign,bool special=false){
    const size_t n=size_t(T)*H*D,nc=size_t(T)*R/2,G=16;
    std::vector<float> src(n),co(std::max(nc,size_t(1))),si(co.size()),got(n+2*G),ref(got.size(),12345.f);
    for(size_t i=0;i<n;++i)src[i]=float(int((i*71+13)%197)-98)/137;
    if(special){const float values[]={0.f,-0.f,1e-38f,-1e-38f,std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::max(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()};
        for(size_t i=0;i<n;++i)src[i]=values[i%9];}
    for(size_t i=0;i<co.size();++i){co[i]=std::cos(float(i%79)/37);si[i]=std::sin(float(i%79)/37);}
    Mem m{q,{}};auto xb=m.get(n+2*G+shift),x=xb+G+shift;
    auto yb=alias?xb:m.get(n+2*G+shift),y=yb+G+shift;
    auto cbase=m.get(co.size()+4),c=cbase+1,sbase=m.get(si.size()+4),s=sbase+2;
    auto out=m.get(n+2*G);
    auto xf=q.fill(xb,12345.f,n+2*G+shift);
    auto xc=q.submit([&](sycl::handler& h){h.depends_on(xf);h.memcpy(x,src.data(),n*4);});
    std::vector<sycl::event> deps{xc,q.memcpy(c,co.data(),co.size()*4),q.memcpy(s,si.data(),si.size()*4),q.fill(out,12345.f,n+2*G)};
    if(!alias)deps.push_back(q.fill(yb,12345.f,n+2*G+shift));
    auto e=rope_oracle::frozen_rope(q,x,c,s,out+G,T,H,D,R,sign,deps);
    q.submit([&](sycl::handler& h){h.depends_on(e);h.memcpy(ref.data(),out,ref.size()*4);}).wait_and_throw();
    if(!special){
        for(size_t row=0;row<size_t(T)*H;++row)for(unsigned d=0;d<D;++d){
            double expected=src[row*D+d];
            if(d>=D-R){unsigned pair=(d-(D-R))/2;size_t j=row*D+D-R+2*pair,ci=(row/H)*(R/2)+pair;
                double a=src[j],b=src[j+1],cs=co[ci],ss=double(si[ci])*sign;
                expected=((d-(D-R))&1)?b*cs+a*ss:a*cs-b*ss;}
            if(std::fabs(double(ref[G+row*D+d])-expected)>2e-6*(1+std::fabs(expected)))throw std::runtime_error("independent CPU bound");
        }
    }
    // Fresh restore after reference completion makes dependency coverage real.
    auto poison=q.fill(x,0.f,n);poison.wait_and_throw();
    sycl::queue producer(q.get_context(),q.get_device());
    auto delay=producer.submit([&](sycl::handler& h){h.host_task([]{std::this_thread::sleep_for(std::chrono::milliseconds(20));});});
    auto restore=producer.submit([&](sycl::handler& h){h.depends_on(delay);h.memcpy(x,src.data(),n*4);});deps.push_back(restore);
#ifdef IE_ROPE_NEGATIVE_CONTROL
    deps.clear();
#endif
    e=ie::ds4_rope_apply(q,x,c,s,y,T,H,D,R,sign,deps);
    q.submit([&](sycl::handler& h){h.depends_on(e);h.memcpy(got.data(),yb+shift,got.size()*4);}).wait_and_throw();
    producer.wait_and_throw();
    if(std::memcmp(ref.data(),got.data(),got.size()*4)){
        size_t i=0;while(i<got.size()&&std::memcmp(&ref[i],&got[i],4)==0)++i;
        std::fprintf(stderr,"mismatch T=%u H=%u D=%u R=%u alias=%d shift=%u sign=%g special=%d i=%zu\n",T,H,D,R,alias,shift,sign,special,i);
        throw std::runtime_error("output or guard differs from frozen GPU");
    }
}
void invalid(sycl::queue& q){
    Mem m{q,{}};auto x=m.get(16),y=m.get(16),c=m.get(16),s=m.get(16);
    q.fill(x,1.f,16);q.fill(c,1.f,16);q.fill(s,1.f,16);q.fill(y,12345.f,16);q.wait();
    bool rejected=false;
    try{ie::ds4_rope_apply(q,x,c,s,y,1,1,4,3,1).wait_and_throw();}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected){float guard;q.memcpy(&guard,y+4,4).wait();std::fprintf(stderr,"odd rope accepted; output guard=%g expected=12345\n",guard);throw std::runtime_error("odd rotary dimension accepted");}
    const unsigned max=std::numeric_limits<unsigned>::max();
    for(auto dim:{std::array<unsigned,4>{1,1,4,6},{1,1,0,0},{max,max,max,2}}){
        rejected=false;try{ie::ds4_rope_apply(q,x,c,s,y,dim[0],dim[1],dim[2],dim[3],1).wait_and_throw();}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("invalid dimensions or byte span accepted");
    }
    for(unsigned operand=0;operand<4;++operand){
        rejected=false;try{ie::ds4_rope_apply(q,operand==0?nullptr:x,operand==1?nullptr:c,operand==2?nullptr:s,operand==3?nullptr:y,1,1,4,2,1).wait_and_throw();}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("null live operand accepted");
    }
    q.fill(x,0.f,1).wait_and_throw();
    sycl::queue producer(q.get_context(),q.get_device());
    auto delay=producer.submit([&](sycl::handler& h){h.host_task([]{std::this_thread::sleep_for(std::chrono::milliseconds(20));});});
    auto fill=producer.submit([&](sycl::handler& h){h.depends_on(delay);h.fill(x,7.f,1);});
    std::vector<sycl::event> empty_deps{fill};
#ifdef IE_ROPE_EMPTY_NEGATIVE_CONTROL
    empty_deps.clear();
#endif
    auto done=ie::ds4_rope_apply(q,nullptr,nullptr,nullptr,nullptr,0,1,4,2,1,empty_deps);
    float marker=0;q.submit([&](sycl::handler& h){h.depends_on(done);h.memcpy(&marker,x,4);}).wait_and_throw();
    producer.wait_and_throw();
    if(marker!=7.f)throw std::runtime_error("empty dependency dropped");
}
}
int main(){try{
    unsigned cards=0,cases=0;
    for(auto d:sycl::device::get_devices(sycl::info::device_type::gpu)){
        if(d.get_info<sycl::info::device::name>().find("B70")==std::string::npos)continue;
        ++cards;sycl::queue q(d);invalid(q);
        for(auto sh:{std::array<unsigned,4>{1,1,1,0},{1,32,512,64},{3,3,65,64},{17,3,128,64},
                     {255,3,512,64},{256,3,512,64},{257,3,512,64},{17,1,7,2},{17,3,64,64},{3,3,65,0}})
            for(bool alias:{false,true})for(float sign:{-1.f,1.f}){check(q,sh[0],sh[1],sh[2],sh[3],alias,1,sign);++cases;}
        for(bool alias:{false,true})for(float sign:{-1.f,1.f}){check(q,3,3,65,64,alias,2,sign,true);++cases;}
        sycl::queue ordered(d,sycl::property::queue::in_order{});
        for(bool alias:{false,true}){check(ordered,17,32,512,64,alias,0,-1);++cases;}
    }if(!cards)throw std::runtime_error("no B70");std::printf("PASS %u numerical/guard cases and host validation on %u devices\n",cases,cards);
}catch(std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}

