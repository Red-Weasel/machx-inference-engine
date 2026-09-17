// Router scheduling gate: frozen scalar-token GPU logits, exact scalar routing,
// guard regions, ragged shapes, and an optional profiled performance experiment.
#include "ie/deepseek4_ops.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// Frozen pre-optimization subgroup mapping and reduction. Keep independent of
// the production helper so indexing/scheduling changes have an exact oracle.
sycl::event reference(sycl::queue& q, const float* x, const float* w, float* y,
                      unsigned T, unsigned H, unsigned E) {
    const size_t rows = size_t(T) * E;
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(((rows + 15) / 16) * 256, 256),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const size_t row = it.get_group(0) * 16 + sg.get_group_linear_id();
                if (row >= rows) return;
                const float* a = x + (row / E) * H;
                const float* b = w + (row % E) * H;
                unsigned i = sg.get_local_linear_id();
                float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                for (; i + 48 < H; i += 64) {
                    a0 += a[i] * b[i];
                    a1 += a[i + 16] * b[i + 16];
                    a2 += a[i + 32] * b[i + 32];
                    a3 += a[i + 48] * b[i + 48];
                }
                for (; i < H; i += 16) a0 += a[i] * b[i];
                const float v = sycl::reduce_over_group(sg, (a0+a1)+(a2+a3), sycl::plus<float>());
                if (sg.leader()) y[row] = v;
            });
    });
}

struct Memory {
    sycl::queue& q;
    std::vector<void*> allocations;
    template<class T> T* get(size_t n) {
        auto p = sycl::malloc_shared<T>(n, q);
        if (!p) throw std::bad_alloc();
        allocations.push_back(p);
        return p;
    }
    ~Memory() { q.wait(); for (auto p : allocations) sycl::free(p, q); }
};

void require(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}

bool shape(sycl::queue& q, unsigned T, unsigned H, unsigned E, unsigned mode,
           bool time, bool require_speedup) {
    constexpr unsigned K = 6, guard = 16;
    Memory mem{q, {}};
    auto x = mem.get<float>(size_t(T)*H);
    auto w = mem.get<float>(size_t(E)*H);
    auto b = mem.get<float>(E);
    auto bv = mem.get<float>(E);
    auto ids = mem.get<int32_t>(T);
    auto mask = mem.get<int32_t>(T);
    auto table = mem.get<int32_t>(T*K);
    auto raw = mem.get<float>(size_t(T)*E + 2*guard);
    auto out = raw + guard;
    auto ref = mem.get<float>(size_t(T)*E);
    auto weights = mem.get<float>(T*K + guard);
    auto indices = mem.get<int32_t>(T*K + guard);
    auto rowlogits = mem.get<float>(E);
    auto rowweights = mem.get<float>(K);
    auto rowindices = mem.get<int32_t>(K);
    for (size_t i=0; i<size_t(T)*H; ++i) x[i] = float(int((i*73+19)%211)-105)/113.f;
    for (size_t i=0; i<size_t(E)*H; ++i) w[i] = float(int((i*43+7)%197)-98)/509.f;
    // Exact ties and nearby logits challenge the selection boundary.
    std::copy(w, w+H, w+H);
    for (unsigned e=0; e<E; ++e) { b[e] = float(int(e%7)-3)*0.001f; bv[e] = -b[e]; }
    b[0]=b[1]=bv[0]=bv[1]=0;
    for (unsigned t=0; t<T; ++t) {
        ids[t]=int(t); mask[t]=int(t%2);
        for (unsigned k=0; k<K; ++k) table[t*K+k]=int((t*3+k)%E);
    }
    auto route = [&](unsigned n, unsigned off, float* l, float* rw, int32_t* ri,
                     const std::vector<sycl::event>& deps) {
        if (mode==0) return ie::ds4_router_topk(q,x+size_t(off)*H,w,b,l,rw,ri,n,H,E,K,1.5f,deps);
        if (mode==1) return ie::ds4_router_hash(q,x+size_t(off)*H,w,table,ids+off,l,rw,ri,n,H,E,K,1.5f,deps);
        if (mode==2) return ie::ds4_router_topk_vl(q,x+size_t(off)*H,w,b,bv,mask+off,l,rw,ri,n,H,E,K,1.5f,deps);
        return ie::ds4_router_hash_vl(q,x+size_t(off)*H,w,table,ids+off,bv,mask+off,l,rw,ri,n,H,E,K,1.5f,deps);
    };
    auto a=q.fill(raw, 12345.f, size_t(T)*E+2*guard);
    auto c=q.fill(weights, 12345.f, T*K+guard);
    auto d=q.fill(indices, int32_t(-77), T*K+guard);
    route(T,0,out,weights,indices,{a,c,d}).wait_and_throw();
    reference(q,x,w,ref,T,H,E).wait_and_throw();
    require(std::memcmp(out,ref,size_t(T)*E*sizeof(float))==0, "logits differ from frozen GPU reference");
    for (unsigned i=0;i<guard;++i) {
        require(raw[i]==12345.f && out[size_t(T)*E+i]==12345.f,"logit guard overwritten");
        require(weights[T*K+i]==12345.f && indices[T*K+i]==-77,"routing guard overwritten");
    }
    // Independent scalar-token dispatch checks indices and normalized weights,
    // including the partial last tile and both text/image selection branches.
    for (unsigned t=0;t<T;++t) if (T<=65 || t<4 || t>=T-4) {
        route(1,t,rowlogits,rowweights,rowindices,{}).wait_and_throw();
        require(std::memcmp(out+size_t(t)*E,rowlogits,E*sizeof(float))==0,"scalar logits differ");
        require(std::memcmp(weights+t*K,rowweights,K*sizeof(float))==0,"scalar weights differ");
        require(std::memcmp(indices+t*K,rowindices,K*sizeof(int32_t))==0,"scalar indices differ");
    }
    // Demonstrate that the exactness assertion detects a one-bit corruption.
    uint32_t word;
    std::memcpy(&word,ref,sizeof(word)); word^=1;
    std::memcpy(ref,&word,sizeof(word));
    require(std::memcmp(out,ref,size_t(T)*E*sizeof(float))!=0,"negative control escaped");
    if (!time) return true;
    std::vector<double> old_us, new_us;
    for (unsigned repeat=0;repeat<13;++repeat) {
        auto control=[&] {
            auto e=reference(q,x,w,ref,T,H,E); e.wait_and_throw();
            return double(e.get_profiling_info<sycl::info::event_profiling::command_end>()-
                          e.get_profiling_info<sycl::info::event_profiling::command_start>())/1000;
        };
        auto candidate=[&] {
            ie::KernelProfiler p; ie::g_profiler=&p;
            route(T,0,out,weights,indices,{}).wait_and_throw();
            ie::g_profiler=nullptr;
            for (const auto& stat:p.harvest()) if (stat.name=="ds4_router_logits") return stat.avg_ms()*1000;
            throw std::runtime_error("router profile missing");
        };
        double before,after;
        if (repeat%2) { after=candidate(); before=control(); }
        else { before=control(); after=candidate(); }
        if (repeat>=3) { old_us.push_back(before); new_us.push_back(after); }
    }
    std::sort(old_us.begin(),old_us.end()); std::sort(new_us.begin(),new_us.end());
    const double old_med=(old_us[4]+old_us[5])/2, new_med=(new_us[4]+new_us[5])/2;
    const double gain=old_med/new_med;
    std::printf("T=%u H=%u E=%u reference_us=%.3f production_us=%.3f speedup=%.3f\n",T,H,E,old_med,new_med,gain);
    return !require_speedup || T<512 || gain>=1.20;
}
} // namespace

int main(int argc, char** argv) {
    const bool speed=argc>1 && std::string(argv[1])=="--require-speedup";
    try {
        unsigned checked=0,devices=0; bool performance=true;
        for (const auto& dev:sycl::device::get_devices(sycl::info::device_type::gpu)) {
            if (dev.get_info<sycl::info::device::name>().find("B70")==std::string::npos) continue;
            ++devices;
            std::printf("device=%s\n",dev.get_info<sycl::info::device::name>().c_str());
            sycl::queue q(dev,sycl::property::queue::enable_profiling{}); // out of order
            for (unsigned T:{1u,3u,4u,15u,31u,32u,33u,63u,64u,65u,129u,512u,2048u}) {
                performance=shape(q,T,4096,256,0,true,speed)&&performance; ++checked;
            }
            for (unsigned mode=0;mode<4;++mode)
                for (auto dims:{std::array<unsigned,3>{65,67,17},{64,4095,257},{65,4095,257},
                                {65,1023,65},{65,1024,63},{65,1024,65},{129,4096,256}}) {
                    shape(q,dims[0],dims[1],dims[2],mode,false,false); ++checked;
                }
            sycl::queue ordered(dev,{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}});
            shape(ordered,65,4096,256,0,false,false); ++checked;
        }
        require(devices>0,"no B70 GPU available");
        std::printf("correctness PASS: %u cases, %u devices\n",checked,devices);
        if (!performance) { std::puts("performance target FAILED: require 1.20x at T>=512"); return 2; }
        return 0;
    } catch (const std::exception& e) { ie::g_profiler=nullptr; std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
