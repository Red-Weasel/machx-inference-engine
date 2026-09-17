// Public descriptor, scalar-aligned loads, and host validation regression.
#ifdef PASS7_BASELINE
#include "deepseek4_experts.before.hpp"
#else
#include "ie/deepseek4_experts.hpp"
#endif
#include "ie/ops.hpp"
#include <sycl/sycl.hpp>
#include <thread>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
#ifdef PASS7_BASELINE
using Job = ie::DS4GemmJob;
Job job(ie::DS4ExpertBank b, unsigned m, unsigned row, float* y) { return {b, 1, m, nullptr, reinterpret_cast<sycl::half*>(y), row}; }
#else
using Job = ie::DS4XmxGemmJob;
static_assert(std::is_same_v<decltype(Job::y), float*>);
static_assert(!std::is_convertible_v<sycl::half*, decltype(Job::y)>);
Job job(ie::DS4ExpertBank b, unsigned m, unsigned row, float* y) { return {b, 1, m, row, y}; }
#endif
void require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
struct Memory {
    sycl::queue& q; std::vector<void*> p;
    template<class T> T* get(size_t n) { auto v=sycl::malloc_shared<T>(n,q); if(!v)throw std::bad_alloc();p.push_back(v);return v; }
    ~Memory(){q.wait();for(auto v:p)sycl::free(v,q);}
};
void run(sycl::queue& q, unsigned offset, unsigned M) {
    constexpr unsigned K=64,N=128,R=3,G=16; constexpr float guard=-12345.f;
    Memory mem{q,{}};
    auto qs=mem.get<uint8_t>(2*N*K/2+16)+(offset&1);
    auto ep=mem.get<uint8_t>(2*N*K/32+16)+(offset&1);
    auto x=mem.get<sycl::half>((M+R)*K+8)+((offset>>1)&1);
    auto ybase=mem.get<float>(M*N+2*G+4)+((offset>>3)&1);auto y=ybase+G;
    auto scratch=mem.get<sycl::half>(N*K+8)+((offset>>2)&1);
    for(unsigned i=0;i<2*N*K/2;++i)qs[i]=uint8_t((i*73+19)%256);
    for(unsigned i=0;i<2*N*K/32;++i)ep[i]=uint8_t(124+i%5);
    for(unsigned i=0;i<(M+R)*K;++i)x[i]=sycl::half(float(int(i%17)-8)/16);
    std::fill(ybase,ybase+M*N+2*G,guard);
    ie::DS4ExpertBank b{};b.dtype=ie::DType::kMXFP4;b.K=K;b.N=N;b.E=2;
    b.mx_qs=qs;b.mx_e=ep;b.mx_qs_stride=N*K/2;b.mx_e_stride=N*K/32;
    auto live=job(b,M,R,y), empty=job(b,0,0,y);
    Job jobs[3]={empty,live,empty};
    auto err=ie::ds4_expert_gemm_xmx_grouped(q,jobs,3,x,K,scratch,N*K);
    require(err.empty(),err.c_str());q.wait_and_throw();
    const float mag[8]={0,1,2,3,4,6,8,12};
    for(unsigned m=0;m<M;++m)for(unsigned n=0;n<N;++n){
        double expected=0;
        for(unsigned k=0;k<K;++k){
            auto byte=qs[N*K/2+n*K/2+(k/32)*16+k%16];
            unsigned nib=k%32<16?byte&15:byte>>4;
            float w=std::ldexp(mag[nib&7]*(nib&8?-1.f:1.f),int(ep[N*K/32+n*K/32+k/32])-128);
            expected+=double(float(x[(R+m)*K+k]))*w;
        }
        require(std::abs(y[m*N+n]-expected)<1.e-5,"independent numerical oracle mismatch");
    }
    for(unsigned i=0;i<G;++i)require(ybase[i]==guard&&y[M*N+i]==guard,"float output canary overwritten");
#ifdef PASS7_BASELINE
    // This is a correctly sized half output plus safe extra backing. The old
    // public descriptor promises half but its wrapper overwrites the guard.
    auto halfout=mem.get<sycl::half>(2*M*N+G);
    std::fill(halfout,halfout+2*M*N+G,sycl::half(-123));
    auto old=live;old.y=halfout;
    err=ie::ds4_expert_gemm_xmx_grouped(q,&old,1,x,K,scratch,N*K);
    require(err.empty(),err.c_str());q.wait_and_throw();
    unsigned changed=0;for(unsigned i=M*N;i<2*M*N;++i)changed+=halfout[i]!=sycl::half(-123);
    std::printf("BASELINE_DEFECT half output guard overwritten: %u / %u halves\n",changed,M*N);
    require(changed!=0,"baseline defect did not reproduce");
#else
    // Two different expert slices reuse one materialized weight scratch in
    // one wrapper call. This also runs on the second same-context queue.
    auto other_y=mem.get<float>(M*N);
    auto other=live;other.e=0;other.y=other_y;
    Job pair[2]={other,live};
    err=ie::ds4_expert_gemm_xmx_grouped(q,pair,2,x,K,scratch,N*K);
    require(err.empty(),err.c_str());q.wait_and_throw();
    for(unsigned m=0;m<M;++m)for(unsigned n=0;n<N;++n){
        double expected=0;
        for(unsigned k=0;k<K;++k){
            auto byte=qs[n*K/2+(k/32)*16+k%16];
            unsigned nib=k%32<16?byte&15:byte>>4;
            float w=std::ldexp(mag[nib&7]*(nib&8?-1.f:1.f),int(ep[n*K/32+k/32])-128);
            expected+=double(float(x[(R+m)*K+k]))*w;
        }
        require(std::abs(other_y[m*N+n]-expected)<1.e-5,"weight scratch reuse changed previous expert output");
    }
    auto reject=[&](Job bad,const char* label){
        std::fill(y,y+M*N,guard);Job request[2]={live,bad};
        auto e=ie::ds4_expert_gemm_xmx_grouped(q,request,2,x,K,scratch,N*K);
        require(!e.empty(),label);q.wait_and_throw();
        for(unsigned i=0;i<M*N;++i)require(y[i]==guard,"partial submission before validation");
    };
    auto bad=live;bad.y=nullptr;reject(bad,"null output accepted");
    bad=live;bad.e=b.E;reject(bad,"invalid expert accepted");
    bad=live;bad.bank.dtype=ie::DType::kIQ3_XXS;reject(bad,"unsupported dtype accepted");
    bad=live;bad.bank.K=33;reject(bad,"invalid K accepted");
    bad=live;bad.bank.N=129;reject(bad,"invalid N accepted");
    bad=live;bad.bank.mx_qs=nullptr;reject(bad,"null weights accepted");
    bad=live;bad.bank.mx_qs_stride=0;reject(bad,"short stride accepted");
    bad=live;bad.bank.mx_e_stride=UINT64_MAX;reject(bad,"stride overflow accepted");
    bad=live;bad.M=131072;bad.bank.N=134217728;bad.bank.mx_qs_stride=uint64_t(bad.bank.N)*K/2;bad.bank.mx_e_stride=uint64_t(bad.bank.N)*K/32;reject(bad,"grid overflow accepted");
    bad=live;bad.row0=UINT32_MAX;bad.M=UINT32_MAX;bad.bank.N=4294967168u;
    bad.bank.mx_qs_stride=uint64_t(bad.bank.N)*K/2;bad.bank.mx_e_stride=uint64_t(bad.bank.N)*K/32;reject(bad,"output byte span overflow accepted");
    require(!ie::ds4_expert_gemm_xmx_grouped(q,nullptr,1,x,K,scratch,N*K).empty(),"null jobs accepted");
    require(!ie::ds4_expert_gemm_xmx_grouped(q,&live,1025,x,K,scratch,N*K).empty(),"oversized count accepted");
    require(ie::ds4_expert_gemm_xmx_grouped(q,nullptr,0,nullptr,0,nullptr,0).empty(),"zero count failed");
    Job zero{};require(ie::ds4_expert_gemm_xmx_grouped(q,&zero,1,nullptr,0,nullptr,0).empty(),"empty job failed");
    const bool fused=!(std::getenv("IE_DS4_EXPERT_XMX_FUSED") && *std::getenv("IE_DS4_EXPERT_XMX_FUSED")=='0');
    std::fill(y,y+M*N,guard);
    auto scratch_result=ie::ds4_expert_gemm_xmx_grouped(q,&live,1,x,K,nullptr,0);
    require(scratch_result.empty()==fused,"null scratch dispatch contract");
    q.wait_and_throw();
    if(!fused){
        for(unsigned i=0;i<M*N;++i)require(y[i]==guard,"null scratch submitted work");
        require(!ie::ds4_expert_gemm_xmx_grouped(q,&live,1,x,K,scratch,N*K-1).empty(),"short scratch accepted");
    }
    require(ie::ds4_expert_gemm_column_tiled(128,UINT32_MAX),"column ceiling overflow");
    require(ie::ds4_expert_gemm_m1_ncols(192,UINT32_MAX)==8,"M1 column ceiling overflow");
    if (offset==0 && M==1) {
        ie::DS4GemmGroupWs ws{};
        require(ie::ds4_gemm_group_ws_alloc(q,16,ws).empty(), "group workspace allocation");
        std::array<ie::DS4GemmJob,16> huge{};
        for(auto& J:huge){J.bank=b;J.bank.N=1u<<30;J.M=1u<<30;J.x_q8=x;J.y=scratch;}
        auto e=ie::ds4_expert_gemm_q8_grouped(q,huge.data(),huge.size(),ws);
        require(e.find("grid size overflow")!=std::string::npos,"grouped grid overflow not rejected on host");
        ie::ds4_gemm_group_ws_free(q,ws);
    }
    sycl::queue unordered(q.get_context(),q.get_device());
    require(!ie::ds4_expert_gemm_xmx_grouped(unordered,&live,1,x,K,scratch,N*K).empty(),"out-of-order wrapper accepted");
#endif
}
#ifndef PASS7_BASELINE
void s8_queue_case(sycl::queue& q, float value) {
    constexpr unsigned M=3,K=256,N=128;
    Memory mem{q,{}};auto x=mem.get<sycl::half>(M*K);auto w=mem.get<int8_t>(K*N);
    auto d=mem.get<sycl::half>(K/32*N);auto y=mem.get<float>(M*N);
    std::fill(x,x+M*K,sycl::half(value));std::fill(w,w+K*N,int8_t(1));
    std::fill(d,d+K/32*N,sycl::half(.5f));std::fill(y,y+M*N,-12345.f);
    sycl::event done;
    if(!ie::gemm_nt_s8_onednn(q,x,w,d,y,M,N,K,32,{},&done)){
        std::printf("SKIP unsupported oneDNN S8 queue case\n");return;
    }
    done.wait_and_throw();
    for(unsigned i=0;i<M*N;++i)require(y[i]==128*value,"S8 queue/thread cache mismatch");
}
#endif

}
int main(){try{unsigned devices=0,cases=0;
    for(auto d:sycl::device::get_devices(sycl::info::device_type::gpu)){
        if(d.get_info<sycl::info::device::name>().find("B70")==std::string::npos)continue;
        sycl::queue q(d,sycl::property::queue::in_order{});++devices;
        for(unsigned offset:{0u,1u,2u,4u,7u,8u,15u})for(unsigned M:{1u,3u,31u,32u,33u}){
            run(q,offset,M);++cases;std::printf("PASS device=%u offset=%u M=%u\n",devices-1,offset,M);
        }
#ifndef PASS7_BASELINE
        // Warm q first, then exercise a different in-order queue in the SAME
        // context, including staged output and immediate scratch reuse.
        sycl::queue second(q.get_context(),d,sycl::property::queue::in_order{});
        for(unsigned offset:{0u,8u,15u}) {run(second,offset,3);++cases;}
        std::printf("PASS device=%u second-queue same-context\n",devices-1);
        s8_queue_case(q,.125f);s8_queue_case(second,.25f);
        for(unsigned t=0;t<4;++t){
            std::exception_ptr error;
            std::thread worker([&]{try {
                sycl::queue tq(q.get_context(),d,sycl::property::queue::in_order{});
                s8_queue_case(tq,float(t+1)/16);
            } catch(...) {error=std::current_exception();}});
            worker.join();if(error)std::rethrow_exception(error);
        }
        std::printf("PASS device=%u S8 crossqueue and sequential host threads\n",devices-1);
#endif
    }
    require(devices>0,"expected at least one B70 card");std::printf("PASS %u boundary cases on %u cards\n",cases,devices);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
