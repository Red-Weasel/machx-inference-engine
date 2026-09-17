#include "ie/deepseek4_experts.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <array>
#include <barrier>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
struct Memory {
    sycl::queue& q; std::vector<void*> allocations;
    template<class T> T* get(size_t n) {
        T* p=sycl::malloc_device<T>(n,q); if(!p)throw std::bad_alloc();
        allocations.push_back(p); return p;
    }
    ~Memory(){q.wait();for(void* p:allocations)sycl::free(p,q);}
};
void ring_case(sycl::queue& q, bool empty_wrap, bool invalid_wrap=false) {
    Memory mem{q,{}}; constexpr unsigned K=32,N=128,JOBS=9,GUARD=16;
    auto w=mem.get<uint8_t>(N*K/2), e=mem.get<uint8_t>(N*K/32);
    auto x=mem.get<sycl::half>(JOBS*K);auto y=mem.get<float>(JOBS*(N+GUARD));
    q.fill(w,uint8_t(0x11),N*K/2).wait_and_throw();q.fill(e,uint8_t(127),N*K/32).wait_and_throw();
    std::vector<ie::Ds4MxXmxJob> jobs;
    for(unsigned j=0;j<JOBS;++j){q.fill(x+j*K,sycl::half(float(j+1)/16),K).wait_and_throw();jobs.push_back({w,e,x+j*K,y+j*(N+GUARD),1,K,N});}
    // Warm the kernel before gating the queue, avoiding JIT timing ambiguity.
    ie::ds4_gemm_mxfp4_xmx(q,jobs.data(),1).wait_and_throw();q.fill(y,-12345.f,JOBS*(N+GUARD)).wait_and_throw();
    std::promise<void> release;auto ready=release.get_future().share();
    auto gate=q.submit([&](sycl::handler& h){h.host_task([ready]{ready.wait();});});
    // A repaired ring can apply backpressure on its ninth active submission.
    // Release from another host thread so that waiting for reuse cannot deadlock.
    std::thread releaser([&]{std::this_thread::sleep_for(std::chrono::milliseconds(200));release.set_value();});
    std::vector<sycl::event> done;
    try {
        done.push_back(ie::ds4_gemm_mxfp4_xmx(q,jobs.data(),1,{gate}));
        for(unsigned j=1;j<JOBS;++j){
            if(invalid_wrap){
                auto bad=jobs[j];bad.K=33;auto prior=jobs[j];prior.M=0;
                const ie::Ds4MxXmxJob request[2]={prior,bad};bool rejected=false;
                try{ie::ds4_gemm_mxfp4_xmx(q,request,2,{gate});}catch(const std::invalid_argument&){rejected=true;}
                if(!rejected)throw std::runtime_error("invalid job accepted");
            }else{
                auto job=jobs[j];if(empty_wrap)job.M=0;
                done.push_back(ie::ds4_gemm_mxfp4_xmx(q,&job,1,{gate}));
            }
        }
    } catch(...) {releaser.join();throw;}
    releaser.join();for(auto& event:done)event.wait_and_throw();q.wait_and_throw();
    std::vector<float> got(JOBS*(N+GUARD));q.memcpy(got.data(),y,got.size()*sizeof(float)).wait_and_throw();
    for(unsigned j=0;j<JOBS;++j)for(unsigned n=0;n<N+GUARD;++n){
        const float expected=n<N && (j==0 || (!empty_wrap&&!invalid_wrap))?float(j+1):-12345.f;
        if(got[j*(N+GUARD)+n]!=expected){
            std::fprintf(stderr,"ring failure empty=%d invalid=%d job=%u col=%u got=%g expected=%g\n",empty_wrap,invalid_wrap,j,n,got[j*(N+GUARD)+n],expected);
            throw std::runtime_error("descriptor ring changed an outstanding job or guard");
        }
    }
}
void concurrent_case(sycl::queue& q) {
    Memory mem{q,{}}; constexpr unsigned K=32,N=128,JOBS=32,GUARD=16;
    auto w=mem.get<uint8_t>(N*K/2), e=mem.get<uint8_t>(N*K/32);
    auto x=mem.get<sycl::half>(JOBS*K); auto y=mem.get<float>(JOBS*(N+GUARD));
    q.fill(w,uint8_t(0x11),N*K/2).wait_and_throw();q.fill(e,uint8_t(127),N*K/32).wait_and_throw();
    std::vector<ie::Ds4MxXmxJob> jobs;
    for(unsigned j=0;j<JOBS;++j){q.fill(x+j*K,sycl::half(float(j+1)/16),K).wait_and_throw();jobs.push_back({w,e,x+j*K,y+j*(N+GUARD),1,K,N});}
    q.fill(y,-12345.f,JOBS*(N+GUARD)).wait_and_throw();
    std::barrier start(4);std::array<std::exception_ptr,4> errors{};
    std::array<std::vector<sycl::event>,4> done;std::vector<std::thread> threads;
    for(unsigned t=0;t<4;++t)threads.emplace_back([&,t]{
        start.arrive_and_wait();
        try{for(unsigned j=t*8;j<(t+1)*8;++j)done[t].push_back(ie::ds4_gemm_mxfp4_xmx(q,&jobs[j],1));}
        catch(...){errors[t]=std::current_exception();}
    });
    for(auto& t:threads)t.join();for(auto error:errors)if(error)std::rethrow_exception(error);
    for(auto& list:done)for(auto& event:list)event.wait_and_throw();
    std::vector<float> got(JOBS*(N+GUARD));q.memcpy(got.data(),y,got.size()*sizeof(float)).wait_and_throw();
    for(unsigned j=0;j<JOBS;++j)for(unsigned n=0;n<N+GUARD;++n)
        if(got[j*(N+GUARD)+n]!=(n<N?float(j+1):-12345.f))throw std::runtime_error("concurrent submission corrupted descriptor or guard");
}
void empty_dependency_case(sycl::queue& q, bool zero_jobs) {
    Memory mem{q,{}};auto bytes=mem.get<uint8_t>(1);auto x=mem.get<sycl::half>(1);auto y=mem.get<float>(1);
    ie::Ds4MxXmxJob empty{bytes,bytes,x,y,0,32,128};
    std::promise<void> release;auto ready=release.get_future().share();
    auto gate=q.submit([&](sycl::handler& h){h.host_task([ready]{ready.wait();});});
    auto event=ie::ds4_gemm_mxfp4_xmx(q,zero_jobs?nullptr:&empty,zero_jobs?0:1,{gate});
    const bool early=event.get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::complete;
    release.set_value();event.wait_and_throw();gate.wait_and_throw();
    if(early)throw std::runtime_error("empty submission lost its dependency");
}

}
int main(){try{unsigned cases=0,devices=0;
    for(auto d:sycl::device::get_devices(sycl::info::device_type::gpu)){
        if(d.get_info<sycl::info::device::name>().find("B70")==std::string::npos)continue;
        ++devices;std::printf("device=%s\n",d.get_info<sycl::info::device::name>().c_str());
        for(bool ordered:{true,false}){
            sycl::queue q=ordered?sycl::queue(d,sycl::property::queue::in_order{}):sycl::queue(d);
            ring_case(q,true);ring_case(q,false);ring_case(q,false,true);
            concurrent_case(q);empty_dependency_case(q,true);empty_dependency_case(q,false);cases+=6;
        }
    }
    if(!devices)throw std::runtime_error("no B70 available");
    std::printf("PASS %u cases on %u devices\n",cases,devices);
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
