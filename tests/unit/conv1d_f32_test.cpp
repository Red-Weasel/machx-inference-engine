#include "ie/ops.hpp"
#include "qwen_glm_kernel_reference.hpp"
#ifdef QG_PRIVATE_FUSION
#include "candidate.hpp"
namespace impl = candidate;
#define CONV_F32 depthwise_conv1d_causal
#else
namespace impl = ie;
#define CONV_F32 depthwise_conv1d_causal_f32
#endif
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

using half=sycl::half;
constexpr size_t G=17;
void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
struct Memory {
    sycl::queue& q;std::vector<void*> ps;
    template<class T>T* get(size_t n){auto p=sycl::malloc_device<T>(std::max(size_t(1),n),q);if(!p)throw std::bad_alloc();ps.push_back(p);return p;}
    ~Memory(){q.wait();for(auto p:ps)sycl::free(p,q);}
};
template<class T>sycl::event copy(sycl::queue& q,T* dst,const T* src,size_t n,const std::vector<sycl::event>& deps){
    return q.submit([&](sycl::handler& h){h.depends_on(deps);h.memcpy(dst,src,n*sizeof(T));});
}
void run(sycl::queue& q,unsigned K,unsigned C,unsigned chunk,bool stateful,bool specials){
    constexpr unsigned T=17;size_t n=size_t(T)*C,sn=size_t(K-1)*C;
    Memory m{q,{}};
    auto x=m.get<half>(n+2*G),w=m.get<half>(size_t(K)*C+2*G);
    auto a=m.get<half>(n+2*G),sa=m.get<half>(sn+2*G),sb=m.get<half>(sn+2*G);
    auto b=m.get<float>(n+2*G);
    std::vector<half> hx(n+2*G,half(7)),hw(size_t(K)*C+2*G,half(7)),hs(sn+2*G,half(7));
    for(size_t i=0;i<n;++i)hx[G+i]=half(float(int(i*31%251)-125)/97);
    for(size_t i=0;i<size_t(K)*C;++i)hw[G+i]=half(float(int(i*13%127)-63)/119);
    for(size_t i=0;i<sn;++i)hs[G+i]=half(float(i%31)/37);
    if(specials)for(size_t i=0;i<n;++i){
        const float vals[]={0.f,-0.f,1e-7f,-1e-7f,65504.f,-65504.f,
                           std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()};
        hx[G+i]=half(vals[i%8]);
    }
    for(unsigned repeat=0;repeat<2;++repeat){
        auto held=q.submit([](sycl::handler& h){h.host_task([]{std::this_thread::sleep_for(std::chrono::milliseconds(5));});});
        auto ex=copy(q,x,hx.data(),hx.size(),{held}),ew=copy(q,w,hw.data(),hw.size(),{held});
        auto ea=copy(q,sa,hs.data(),hs.size(),{held}),eb=copy(q,sb,hs.data(),hs.size(),{held});
        auto fa=q.fill(a,half(7),n+2*G),fb=q.fill(b,7.f,n+2*G);
        auto old=frozen::depthwise_conv1d_causal(q,x+G,w+G,stateful?sa+G:nullptr,a+G,T,C,K,{ex,ew,ea,fa});
        std::vector<sycl::event> deps{ex,ew,eb,fb};
        for(unsigned pos=0;pos<T;pos+=chunk)deps={impl::CONV_F32(q,x+G+size_t(pos)*C,w+G,
            stateful?sb+G:nullptr,b+G+size_t(pos)*C,std::min(chunk,T-pos),C,K,deps)};
        // Stateless chunks must be independent; compare a single full operation instead.
        if(!stateful)deps={impl::CONV_F32(q,x+G,w+G,nullptr,b+G,T,C,K,deps)};
        std::vector<half> av(n+2*G),sav(sn+2*G),sbv(sn+2*G);std::vector<float> bv(n+2*G);
        copy(q,av.data(),a,av.size(),{old}).wait_and_throw();
        copy(q,sav.data(),sa,sav.size(),{old}).wait_and_throw();
        copy(q,bv.data(),b,bv.size(),deps).wait_and_throw();
        copy(q,sbv.data(),sb,sbv.size(),deps).wait_and_throw();
        for(size_t i=0;i<av.size();++i){float want=float(av[i]);
            require((std::isnan(want)&&std::isnan(bv[i]))||!std::memcmp(&want,&bv[i],4),"conv half-rounded output");}
        require(!std::memcmp(sav.data(),sbv.data(),sav.size()*2),"conv full state mismatch");
        for(size_t i=0;i<G;++i)require(bv[i]==7&&bv[G+n+i]==7&&float(sbv[i])==7&&float(sbv[G+sn+i])==7,"conv guard");
        if(stateful)for(size_t i=0;i<sn;++i){
            size_t at=n+i;half want=at<sn?hs[G+at]:hx[G+at-sn];
            require(!std::memcmp(&want,&sbv[G+i],2),"conv independent history");
        }
        if(!specials)for(unsigned t=0;t<T;++t)for(unsigned c=0;c<C;++c){
            double sum=0;for(unsigned k=0;k<K;++k){int pos=int(t)-int(k);double v=pos>=0?float(hx[G+size_t(pos)*C+c]):
                (stateful?float(hs[G+size_t(int(K-1)+pos)*C+c]):0.);
                sum+=v*float(hw[G+size_t(c)*K+(K-1-k)]);}
            double want=sum/(1+std::exp(-sum));float got=bv[G+size_t(t)*C+c];
            require(std::isfinite(got)&&std::abs(got-want)<.002*(1+std::abs(want)),"conv independent arithmetic");
        }
    }
}
void empty(sycl::queue& q){
    for(auto dims:{std::array<unsigned,3>{0,65,4},{3,0,4},{3,65,0}}){
        std::atomic<bool> ready=false;
        auto e=q.submit([&](sycl::handler& h){h.host_task([&]{std::this_thread::sleep_for(std::chrono::milliseconds(20));ready=true;});});
        auto done=impl::CONV_F32(q,nullptr,nullptr,nullptr,nullptr,dims[0],dims[1],dims[2],{e});
        done.wait_and_throw();bool observed=ready.load();e.wait_and_throw();require(observed,"empty float conv dependency");
    }
}
int main(){try{
    sycl::queue q(sycl::gpu_selector_v);unsigned cases=0;
    for(unsigned k:{1u,2u,4u,7u,65u})for(unsigned c:{1u,65u,4096u})
        for(unsigned chunk:{1u,2u,3u,4u,8u,17u}){run(q,k,c,chunk,true,false);++cases;}
    for(unsigned k:{1u,4u,7u}){run(q,k,65,17,false,false);++cases;run(q,k,65,2,true,true);++cases;}
    empty(q);printf("PASS %u convolution cases twice: half-rounded float, complete state, guards, OOO reuse and empty events\n",cases);
}catch(std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
