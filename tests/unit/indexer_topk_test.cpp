// Exact packed-key selection: CPU sort, frozen GPU sort, guards and OOO queues.
#include <sycl/sycl.hpp>
#include <vector>
#include <limits>
constexpr int kSG=16;
constexpr float kNegInf=-std::numeric_limits<float>::infinity();
inline uint64_t pack_key(float v, uint32_t idx) {
    uint32_t u = sycl::bit_cast<uint32_t>(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (uint64_t(u) << 32) | uint64_t(0xFFFFFFFFu - idx);
}

template <uint32_t WG>
sycl::event frozen(sycl::queue& q,
                                     const float* scores, const int32_t* positions,
                                     int32_t* out,
                                     uint32_t T, uint32_t n_keys, uint32_t top_k,
                                     uint32_t compress_rate, uint32_t N,
                                     const std::vector<sycl::event>& deps) {
    return q.submit( [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint64_t, 1> buf(sycl::range<1>(N), h);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t   = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const int64_t thr = (int64_t(positions[t]) + 1) / int64_t(compress_rate);

            for (uint32_t e = lid; e < N; e += WG) {
                const float v = (e < n_keys && int64_t(e) < thr)
                                    ? scores[size_t(t) * n_keys + e] : kNegInf;
                buf[e] = pack_key(v, e);
            }
            it.barrier(sycl::access::fence_space::local_space);

            // Textbook bitonic sort, ascending.  Every work-item reaches every
            // barrier: the compare-exchange loop is inside, the barrier is not.
            for (uint32_t k = 2; k <= N; k <<= 1) {
                for (uint32_t j = k >> 1; j > 0; j >>= 1) {
                    for (uint32_t i = lid; i < N; i += WG) {
                        const uint32_t ixj = i ^ j;
                        if (ixj > i) {
                            const bool up = ((i & k) == 0);
                            const uint64_t a = buf[i], b = buf[ixj];
                            if ((a > b) == up) { buf[i] = b; buf[ixj] = a; }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
            }

            // Descending read-out: the j-th largest key sits at N-1-j.
            for (uint32_t j = lid; j < top_k; j += WG) {
                const int32_t idx =
                    int32_t(0xFFFFFFFFu - uint32_t(buf[N - 1 - j] & 0xFFFFFFFFu));
                out[size_t(t) * top_k + j] = (int64_t(idx) >= thr) ? -1 : idx;
            }
        });
    });
}


#include "ie/deepseek4_attn.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
using Fn=sycl::event(*)(sycl::queue&,const float*,const int32_t*,int32_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,const std::vector<sycl::event>&);
struct V{const char*name;Fn fn;};
sycl::event production(sycl::queue&q,const float*s,const int32_t*p,int32_t*o,uint32_t T,uint32_t E,uint32_t K,uint32_t C,uint32_t,const std::vector<sycl::event>&d){return ie::ds4_indexer_topk(q,s,p,o,T,E,K,C,d);}
std::vector<V>vs={{"production",production}};
void check(sycl::queue&q,uint32_t E,uint32_t K,uint32_t T,int pattern,bool timing){
 uint32_t N=1;while(N<E)N*=2;const size_t nr=size_t(T)*E,ny=size_t(T)*K+32;std::mt19937 rng(1273+E+K+T);
 std::vector<float>r(nr);std::vector<int32_t>p(T);for(auto&v:r)v=pattern==1?0.f:pattern==2?-INFINITY:(int(rng()%24001)-12000)/1000.f;
 for(uint32_t t=0;t<T;++t)p[t]=pattern==3?int32_t((t%3)*(E/3)*4)-1:int32_t(E*4)-1;
 auto*dr=sycl::malloc_device<float>(nr,q);auto*dp=sycl::malloc_device<int32_t>(T,q);auto*y=sycl::malloc_device<int32_t>(ny,q);if(!dr||!dp||!y)throw std::runtime_error("allocation");
 std::vector<int32_t>ref(ny),got(ny),oracle(ny,-123);
 for(uint32_t t=0;t<T;++t){std::vector<uint64_t> keys;for(uint32_t e=0;e<E;++e)keys.push_back(pack_key(int64_t(e)<(int64_t(p[t])+1)/4?r[size_t(t)*E+e]:kNegInf,e));std::sort(keys.begin(),keys.end(),std::greater<uint64_t>());for(uint32_t k=0;k<K;++k){auto i=int32_t(0xffffffffu-uint32_t(keys[k]));oracle[16+size_t(t)*K+k]=int64_t(i)>=(int64_t(p[t])+1)/4?-1:i;}}
 auto run=[&](Fn f,const std::vector<sycl::event>&d={}){return f(q,dr,dp,y+16,T,E,K,4,N,d);};
 auto validate=[&](Fn f,bool base){std::vector<sycl::event>d={q.memcpy(dr,r.data(),nr*4),q.memcpy(dp,p.data(),T*4),q.fill(y,int32_t(-123),ny)};auto e=run(f,d);q.submit([&](sycl::handler&h){h.depends_on(e);h.memcpy(base?ref.data():got.data(),y,ny*4);}).wait_and_throw();};
 validate(frozen<1024>,true);if(ref!=oracle)throw std::runtime_error("frozen vs oracle");
 for(auto v:vs){validate(v.fn,false);if(ref!=got){std::fprintf(stderr,"FAIL %s E%u K%u T%u pattern%d\n",v.name,E,K,T,pattern);for(size_t z=0;z<ny;++z)if(ref[z]!=got[z]){std::fprintf(stderr,"at%zu %d/%d\n",z,ref[z],got[z]);break;}std::exit(2);}std::printf("EXACT,%s,%u,%u,%u,%d\n",v.name,E,K,T,pattern);std::fflush(stdout);}

 sycl::free(dr,q);sycl::free(dp,q);sycl::free(y,q);
}
int main(){
 unsigned devices=0;
 for(auto d:sycl::device::get_devices(sycl::info::device_type::gpu)) {
  if(d.get_backend()!=sycl::backend::ext_oneapi_level_zero||d.get_info<sycl::info::device::name>().find("B70")==std::string::npos)continue;
  ++devices;sycl::queue q(d);std::printf("DEVICE %s\n",d.get_info<sycl::ext::intel::info::device::pci_address>().c_str());
  for(uint32_t E:{1u,7u,16u,32u,33u,127u,255u,256u,257u,512u,513u,1023u,1024u,1025u,2048u,2049u,4096u,4097u,6001u,8192u})
   for(int pattern:{0,1,2,3})check(q,E,std::min(E,512u),3,pattern,false);
  for(uint32_t E:{256u,1024u,2048u,4096u,8192u})for(uint32_t T:{1u,2u,16u,127u,128u,129u,1024u})check(q,E,std::min(E,512u),T,0,false);
  for(uint32_t K:{1u,7u,127u,511u,1024u})check(q,2048,K,3,1,false);
 }
 if(!devices)return 77;std::puts("GATE PASSED");
}
