// Exact KDA outputs and persistent state across four dependent updates.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <vector>
sycl::event frozen(sycl::queue& q,
                           const float* q_in, const float* k_in, const float* v_in,
                           const float* g_in, const float* beta_in,
                           float* state,
                           float* out,
                           uint32_t B, uint32_t T,
                           uint32_t n_heads, uint32_t k_head_dim, uint32_t v_head_dim,
                           const std::vector<sycl::event>& deps) {
    constexpr int K_DIM_MAX = 128;
    if (k_head_dim != K_DIM_MAX || v_head_dim != K_DIM_MAX) {
        sycl::event e;   // only the glm5next shape (128/128) is supported
        return e;
    }
    return q.submit( [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> q_slm(K_DIM_MAX, h);
        sycl::local_accessor<float, 1> k_slm(K_DIM_MAX, h);
        sycl::local_accessor<float, 1> v_slm(K_DIM_MAX, h);
        sycl::local_accessor<float, 1> a_slm(K_DIM_MAX, h);   // exp(g_t) per key
        sycl::local_accessor<float, 1> sc_slm(1, h);

        const uint32_t WG_ITEMS = K_DIM_MAX;
        h.parallel_for(sycl::nd_range<2>({uint64_t(B) * n_heads, WG_ITEMS}, {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t bh = uint32_t(it.get_group(0));
            const uint32_t b  = bh / n_heads;
            const uint32_t hh = bh % n_heads;
            const uint32_t vv = uint32_t(it.get_local_id(1));

            float S_col[K_DIM_MAX];
            const uint64_t state_base = (uint64_t(b) * n_heads + hh) * K_DIM_MAX * K_DIM_MAX;
            #pragma unroll
            for (int kk = 0; kk < K_DIM_MAX; ++kk)
                S_col[kk] = state[state_base + uint64_t(kk) * K_DIM_MAX + vv];

            for (uint32_t t = 0; t < T; ++t) {
                const uint64_t row = (uint64_t(b) * T + t) * n_heads + hh;
                q_slm[vv] = q_in[row * K_DIM_MAX + vv];
                k_slm[vv] = k_in[row * K_DIM_MAX + vv];
                v_slm[vv] = v_in[row * K_DIM_MAX + vv];
                a_slm[vv] = sycl::native::exp(g_in[row * K_DIM_MAX + vv]);
                if (vv == 0) sc_slm[0] = beta_in[row];
                sycl::group_barrier(it.get_group());

                const float beta_t = sc_slm[0];
                const float v_t    = v_slm[vv];

                // decay (per key channel) + kv_mem in one pass
                float kv_mem = 0.f;
                #pragma unroll
                for (int kk = 0; kk < K_DIM_MAX; ++kk) {
                    S_col[kk] *= a_slm[kk];
                    kv_mem += S_col[kk] * k_slm[kk];
                }
                const float delta = (v_t - kv_mem) * beta_t;
                float out_v = 0.f;
                #pragma unroll
                for (int kk = 0; kk < K_DIM_MAX; ++kk) {
                    S_col[kk] += k_slm[kk] * delta;
                    out_v += S_col[kk] * q_slm[kk];
                }
                out[(uint64_t(b) * T + t) * n_heads * K_DIM_MAX + hh * K_DIM_MAX + vv] = out_v;

                sycl::group_barrier(it.get_group());
            }

            #pragma unroll
            for (int kk = 0; kk < K_DIM_MAX; ++kk)
                state[state_base + uint64_t(kk) * K_DIM_MAX + vv] = S_col[kk];
        });
    });
}
#include "ie/ops.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
using Fn=decltype(&frozen);
struct V{const char*name;Fn fn;};
std::vector<V> vs={{"production",ie::kda_recurrence}};
void check(sycl::queue&q,uint32_t B,uint32_t H,uint32_t T,bool timing){
 const size_t nr=size_t(B)*T*H*128,ns=size_t(B)*H*128*128,ny=nr+32;std::mt19937 rng(271+H+T+B);
 std::vector<float>hq(nr),hk(nr),hv(nr),hg(nr),hb(nr/128),hs(ns+32);
 for(auto&v:hq)v=(int(rng()%2001)-1000)/12000.f;
 for(auto&v:hk)v=(int(rng()%2001)-1000)/12000.f;
 for(auto&v:hv)v=(int(rng()%2001)-1000)/1000.f;
 for(auto&v:hg)v=-(1+int(rng()%2001))/10000.f;
 for(auto&v:hb)v=(1+int(rng()%1000))/1000.f;
 for(auto&v:hs)v=(int(rng()%2001)-1000)/10000.f;
 for(size_t z=0;z<16;++z)hs[z]=hs[ns+16+z]=-123.f;
 auto*dq=sycl::malloc_device<float>(nr,q);auto*dk=sycl::malloc_device<float>(nr,q);auto*dv=sycl::malloc_device<float>(nr,q);auto*dg=sycl::malloc_device<float>(nr,q);auto*db=sycl::malloc_device<float>(nr/128,q);auto*st=sycl::malloc_device<float>(ns+32,q);auto*y=sycl::malloc_device<float>(ny,q);if(!dq||!dk||!dv||!dg||!db||!st||!y)throw std::runtime_error("allocation");
 std::vector<float>ref(ny),got(ny),sr(ns+32),sg(ns+32);
 auto run=[&](Fn f,const std::vector<sycl::event>&d={}){return f(q,dq,dk,dv,dg,db,st+16,y+16,B,T,H,128,128,d);};
 auto validate=[&](Fn f,bool base){std::vector<sycl::event>d={q.memcpy(dq,hq.data(),nr*4),q.memcpy(dk,hk.data(),nr*4),q.memcpy(dv,hv.data(),nr*4),q.memcpy(dg,hg.data(),nr*4),q.memcpy(db,hb.data(),hb.size()*4),q.memcpy(st,hs.data(),hs.size()*4),q.fill(y,-123.f,ny)};auto e=run(f,d);for(int step=0;step<3;++step)e=run(f,{e});q.submit([&](sycl::handler&h){h.depends_on(e);h.memcpy(base?ref.data():got.data(),y,ny*4);}).wait_and_throw();q.submit([&](sycl::handler&h){h.depends_on(e);h.memcpy(base?sr.data():sg.data(),st,(ns+32)*4);}).wait_and_throw();};
 validate(frozen,true);
 for(size_t z=0;z<16;++z)if(ref[z]!=-123.f||ref[ny-16+z]!=-123.f||sr[z]!=-123.f||sr[ns+16+z]!=-123.f)throw std::runtime_error("reference guards");
 for(auto v:vs){validate(v.fn,false);if(std::memcmp(ref.data(),got.data(),ny*4)||std::memcmp(sr.data(),sg.data(),(ns+32)*4)){std::fprintf(stderr,"FAIL %s B%u H%u T%u\n",v.name,B,H,T);for(size_t z=0;z<ny;++z)if(sycl::bit_cast<uint32_t>(ref[z])!=sycl::bit_cast<uint32_t>(got[z])){std::fprintf(stderr,"out%zu %.9g/%.9g\n",z,ref[z],got[z]);break;}std::exit(2);}std::printf("EXACT,%s,%u,%u,%u\n",v.name,B,H,T);std::fflush(stdout);}

 for(auto*p:{dq,dk,dv,dg,db,st,y})sycl::free(p,q);
}
int main(){
 unsigned devices=0;
 for(auto d:sycl::device::get_devices(sycl::info::device_type::gpu)) {
  if(d.get_backend()!=sycl::backend::ext_oneapi_level_zero||d.get_info<sycl::info::device::name>().find("B70")==std::string::npos)continue;
  ++devices;sycl::queue q(d);
  std::printf("DEVICE %s\n",d.get_info<sycl::ext::intel::info::device::pci_address>().c_str());
  for(uint32_t B:{1u,2u})for(uint32_t H:{1u,3u,16u,32u,64u})for(uint32_t T:{1u,2u,3u,17u,128u})check(q,B,H,T,false);
  check(q,1,32,1024,false);
  for(auto shape:{std::pair{64u,128u},std::pair{128u,64u},std::pair{0u,0u}}) {
   bool threw=false;
   try{ie::kda_recurrence(q,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1,1,1,shape.first,shape.second);}
   catch(const std::invalid_argument&){threw=true;}
   if(!threw)throw std::runtime_error("unsupported KDA shape must throw");
  }
  for(uint32_t dim=0;dim<3;++dim) {
   auto*m=sycl::malloc_device<int>(1,q);auto fill=q.fill(m,42,1);
   auto e=ie::kda_recurrence(q,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,dim==0?0:1,dim==1?0:1,dim==2?0:1,128,128,{fill});
   int got=0;q.submit([&](sycl::handler&h){h.depends_on(e);h.memcpy(&got,m,4);}).wait_and_throw();sycl::free(m,q);
   if(got!=42)throw std::runtime_error("empty KDA dependency");
  }
 }
 if(!devices)return 77;std::puts("GATE PASSED");
}
