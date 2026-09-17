#include "ie/deepseek4_attn.hpp"
#include "ie/kernel_profiler.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace {
constexpr unsigned kSG=16;
// Frozen pre-optimization score kernels: independent numerical oracles.
template <typename TK>
sycl::event frozen_dimlane(sycl::queue& q,
                                      const float* q_in, const TK* keys,
                                      const float* w_proj, float* scores,
                                      uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                      uint32_t n_keys,
                                      float softmax_scale, float weights_scaling,
                                      const std::vector<sycl::event>& deps) {
    return ie::ps(q, "ds4_indexer_score", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({T, size_t(n_keys) * kSG}, {1, kSG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t    = uint32_t(it.get_group(0));
            const uint32_t e    = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            const TK* krow = keys + size_t(e) * head_dim;

            float total = 0.f;
            for (uint32_t hh = 0; hh < n_heads; ++hh) {
                const float* qrow = q_in + (size_t(t) * n_heads + hh) * head_dim;
                float part = 0.f;
                for (uint32_t d = lane; d < head_dim; d += kSG) part += qrow[d] * float(krow[d]);
                const float dot = sycl::reduce_over_group(sg, part, sycl::plus<float>());
                // scores = relu(q·K) * softmax_scale ; weights = w_proj * weights_scaling
                total += sycl::fmax(dot, 0.f) * softmax_scale
                       * (w_proj[size_t(t) * n_heads + hh] * weights_scaling);
            }
            if (lane == 0) scores[size_t(t) * n_keys + e] = total;
        });
    });
}

// SGS keys per work-group (one per sub-group); lane l owns heads l, l+kSG, ...
template <uint32_t SGS, typename TK>
sycl::event frozen_headlane(sycl::queue& q,
                                       const float* q_in, const TK* keys,
                                       const float* w_proj, float* scores,
                                       uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                       uint32_t n_keys,
                                       float softmax_scale, float weights_scaling,
                                       const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = SGS * kSG;
    const uint32_t n_grp = (n_keys + SGS - 1u) / SGS;
    return ie::ps(q, "ds4_indexer_score", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({T, size_t(n_grp) * WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t t = uint32_t(it.get_group(0));
            auto sg = it.get_sub_group();
            // `e` is uniform inside the sub-group, so this early return retires
            // whole sub-groups and never leaves the reduce below unreached.
            const uint32_t e = uint32_t(it.get_group(1)) * SGS
                             + uint32_t(sg.get_group_linear_id());
            if (e >= n_keys) return;
            const uint32_t lane = uint32_t(sg.get_local_linear_id());
            const TK* krow = keys + size_t(e) * head_dim;

            float tot = 0.f;
            // The loop bound is uniform; only the CONTRIBUTION is predicated, so
            // n_heads that is not a multiple of kSG costs idle lanes, not
            // divergent control flow.
            for (uint32_t hb = 0; hb < n_heads; hb += kSG) {
                const uint32_t hh = hb + lane;
                float dot = 0.f;
                if (hh < n_heads) {
                    const float* qrow = q_in + (size_t(t) * n_heads + hh) * head_dim;
                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                    uint32_t d = 0;
                    for (; d + 3u < head_dim; d += 4u) {
                        a0 += qrow[d]      * float(krow[d]);
                        a1 += qrow[d + 1u] * float(krow[d + 1u]);
                        a2 += qrow[d + 2u] * float(krow[d + 2u]);
                        a3 += qrow[d + 3u] * float(krow[d + 3u]);
                    }
                    for (; d < head_dim; ++d) a0 += qrow[d] * float(krow[d]);
                    dot = (a0 + a1) + (a2 + a3);
                    tot += sycl::fmax(dot, 0.f) * softmax_scale
                         * (w_proj[size_t(t) * n_heads + hh] * weights_scaling);
                }
            }
            const float r = sycl::reduce_over_group(sg, tot, sycl::plus<float>());
            if (lane == 0) scores[size_t(t) * n_keys + e] = r;
        });
    });
}


}

struct Memory {
 sycl::queue& q;std::vector<void*> p;
 template<typename T>T* get(size_t n){auto a=sycl::malloc_device<T>(n,q);if(!a)throw std::bad_alloc();p.push_back(a);return a;}
 ~Memory(){q.wait();for(auto a:p)sycl::free(a,q);}
};
void require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
template<typename TK>
void check(sycl::queue& q,unsigned T,unsigned H,unsigned D,unsigned E,unsigned qshift=0,unsigned kshift=0,bool zeros=false) {
 Memory mem{q,{}};constexpr size_t guard=16;
 size_t qn=size_t(T)*H*D,kn=size_t(E)*D,wn=size_t(T)*H,on=size_t(T)*E;
 std::vector<float> hq(qn),hw(wn),ref(on+guard*2),got(ref.size());std::vector<TK> hk(kn);
 for(size_t i=0;i<qn;++i)hq[i]=zeros?0.f:float(int((i*73+19)%211)-105)/113.f;
 for(size_t i=0;i<kn;++i)hk[i]=TK(float(int((i*43+7)%197)-98)/509.f);
 if(E>=2)std::copy(hk.begin(),hk.begin()+D,hk.begin()+D);
 for(size_t i=0;i<wn;++i)hw[i]=float(int(i%17)-8)/31.f;
 auto qi=mem.get<float>(qn+qshift)+qshift;auto keys=mem.get<TK>(kn+kshift)+kshift;
 auto wp=mem.get<float>(wn);auto ro=mem.get<float>(ref.size());auto yo=mem.get<float>(got.size());
 std::vector<sycl::event> deps={q.memcpy(qi,hq.data(),qn*4),q.memcpy(keys,hk.data(),kn*sizeof(TK)),
    q.memcpy(wp,hw.data(),wn*4),q.fill(ro,12345.f,ref.size()),q.fill(yo,12345.f,got.size())};
 auto r=H>=16?frozen_headlane<16>(q,qi,keys,wp,ro+guard,T,H,D,E,.08838835f,.125f,deps):
              frozen_dimlane(q,qi,keys,wp,ro+guard,T,H,D,E,.08838835f,.125f,deps);
 auto y=ie::ds4_indexer_score(q,qi,keys,wp,yo+guard,T,H,D,E,.08838835f,.125f,deps);
 q.submit([&](sycl::handler& h){h.depends_on(r);h.memcpy(ref.data(),ro,ref.size()*4);}).wait_and_throw();
 q.submit([&](sycl::handler& h){h.depends_on(y);h.memcpy(got.data(),yo,got.size()*4);}).wait_and_throw();
 require(!std::memcmp(ref.data(),got.data(),ref.size()*4),"scores differ from frozen kernel");
 for(size_t i=0;i<guard;++i)require(got[i]==12345.f&&got[guard+on+i]==12345.f,"score guard overwritten");
 for(size_t i=guard;i<guard+on;++i)require(std::isfinite(got[i]),"nonfinite score");
 // Negative control: the exact comparator must catch a one-bit corruption.
 uint32_t bit;std::memcpy(&bit,got.data()+guard,4);bit^=1;std::memcpy(got.data()+guard,&bit,4);
 require(std::memcmp(ref.data(),got.data(),ref.size()*4)!=0,"score negative control escaped");
 const unsigned top=std::min(33u,E);size_t tn=size_t(T)*top+2*guard;
 auto pos=mem.get<int32_t>(T);auto rt=mem.get<int32_t>(tn);auto yt=mem.get<int32_t>(tn);
 std::vector<int32_t> hp(T),hr(tn),hy(tn);
 for(unsigned t=0;t<T;++t)hp[t]=int32_t((t%3)*(E/2)*4)-1;
 auto pe=q.memcpy(pos,hp.data(),T*4);auto re=q.fill(rt,int32_t(-123),tn);auto ye=q.fill(yt,int32_t(-123),tn);
 auto tr=ie::ds4_indexer_topk(q,ro+guard,pos,rt+guard,T,E,top,4,{r,pe,re});
 auto ty=ie::ds4_indexer_topk(q,yo+guard,pos,yt+guard,T,E,top,4,{y,pe,ye});
 q.submit([&](sycl::handler& h){h.depends_on(tr);h.memcpy(hr.data(),rt,tn*4);}).wait_and_throw();
 q.submit([&](sycl::handler& h){h.depends_on(ty);h.memcpy(hy.data(),yt,tn*4);}).wait_and_throw();
 require(hr==hy,"selected entries differ");
 for(size_t i=0;i<guard;++i)require(hy[i]==-123&&hy[tn-1-i]==-123,"top-k guard overwritten");
}
int main(){try{
 unsigned devices=0,cases=0;
 for(const auto& dev:sycl::device::get_devices(sycl::info::device_type::gpu)){
  if(dev.get_info<sycl::info::device::name>().find("B70")==std::string::npos)continue;
  ++devices;sycl::queue q(dev); // out-of-order: every input dependency matters
  std::printf("device=%s\n",dev.get_info<sycl::info::device::name>().c_str());
  for(auto d:{std::array<unsigned,4>{1,64,128,128},{63,64,128,128},{64,64,128,128},
              {65,64,128,131},{65,15,128,131},{65,16,128,131},{65,17,128,131},
              {65,63,128,131},{65,65,128,131},{65,64,63,131},{65,64,64,131},
              {65,64,65,131},{65,64,124,131},{65,64,127,131},{65,64,132,131},
              {65,64,128,1},{65,64,128,2},{65,64,128,31},{65,64,128,32},
              {65,64,128,33},{65,64,128,63},{65,64,128,64},{65,64,128,65},
              {512,64,128,128},{2048,64,128,512}}){
   check<float>(q,d[0],d[1],d[2],d[3]);check<sycl::half>(q,d[0],d[1],d[2],d[3]);cases+=2;
  }
  for(unsigned qshift:{0u,1u})for(unsigned kshift:{0u,1u})for(bool zeros:{false,true}){
   check<float>(q,65,64,128,131,qshift,kshift,zeros);check<sycl::half>(q,65,64,128,131,qshift,kshift,zeros);cases+=2;
  }
  sycl::queue ordered(dev,sycl::property::queue::in_order{});
  check<float>(ordered,65,64,128,131);check<sycl::half>(ordered,65,64,128,131);cases+=2;
 }
 require(devices>0,"no B70 available");std::printf("PASS %u cases on %u devices\n",cases,devices);
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
