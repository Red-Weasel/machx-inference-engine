// Exact pre-round GPU oracle plus independent arithmetic, guards and OOO events.
#include "ie/ops.hpp"
#include "qwen_glm_kernel_reference.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

using half = sycl::half;
constexpr size_t G = 17; // intentionally misalign every operand
struct Memory {
    sycl::queue& q;
    std::vector<void*> pointers;
    template<class T> T* get(size_t n) {
        auto* p = sycl::malloc_device<T>(std::max(size_t(1), n), q);
        if (!p) throw std::bad_alloc();
        pointers.push_back(p); return p;
    }
    ~Memory() { q.wait(); for (auto p : pointers) sycl::free(p, q); }
};
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
template<class T> void exact(const std::vector<T>& a, const std::vector<T>& b) {
    require(a.size() == b.size(), "oracle shape");
    for (size_t i = 0; i < a.size(); ++i)
        require((std::isnan(float(a[i])) && std::isnan(float(b[i]))) ||
                !std::memcmp(&a[i], &b[i], sizeof(T)), "frozen numerical mismatch");
}
template<class T> void guards(const std::vector<T>& v) {
    for (size_t i = 0; i < G; ++i)
        require(float(v[i]) == 7 && float(v[v.size()-1-i]) == 7, "guard modified");
}
template<class T> std::vector<T> read(sycl::queue& q, T* p, size_t n, sycl::event e) {
    std::vector<T> v(n);
    q.submit([&](sycl::handler& h) { h.depends_on(e); h.memcpy(v.data(), p, n*sizeof(T)); }).wait_and_throw();
    return v;
}
sycl::event delay(sycl::queue& q) {
    return q.submit([](sycl::handler& h) {
        h.host_task([] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); });
    });
}
template<class T> sycl::event upload(sycl::queue& q, T* p, const std::vector<T>& v, sycl::event e) {
    return q.submit([&](sycl::handler& h) { h.depends_on(e); h.memcpy(p, v.data(), v.size()*sizeof(T)); });
}
void near(double got, double expected, double tolerance) {
    require(std::isfinite(got) && std::isfinite(expected) &&
            std::abs(got-expected) <= tolerance*(1+std::abs(expected)), "independent arithmetic oracle");
}

void rope(sycl::queue& q, unsigned T, unsigned H, unsigned D, unsigned R, bool inter, bool alias, bool special) {
    Memory m{q,{}}; const size_t n=size_t(T)*H*D;
    auto x=m.get<half>(n+2*G), a=m.get<half>(n+2*G), b=m.get<half>(n+2*G);
    auto pos=m.get<int32_t>(3*T+2*G);
    std::vector<half> src(n+2*G,half(7)), initial(n+2*G,half(7));
    std::vector<int32_t> hp(3*T+2*G);
    for (size_t i=0;i<n;++i) src[G+i]=half(float(int(i*31%251)-125)/97);
    if (special) for(size_t i=0;i<n;i+=101) src[G+i]=half(i%2 ? std::numeric_limits<float>::infinity() : -0.f);
    for(size_t i=0;i<3*T;++i) hp[G+i]=int32_t(i%2 ? 32768+i*17 : -int(i*13));
    auto held=delay(q);
    std::vector<sycl::event> deps{upload(q,x,src,held),upload(q,a,alias?src:initial,held),
        upload(q,b,alias?src:initial,held),upload(q,pos,hp,held)};
    constexpr float theta=10000000.f;
    auto old=inter ? frozen::rope_imrope3(q,alias?a+G:x+G,pos+G,a+G,T,H,D,R,theta,deps)
                   : frozen::rope_partial(q,alias?a+G:x+G,pos+G,a+G,T,H,D,R,theta,deps);
    auto now=inter ? ie::rope_imrope3(q,alias?b+G:x+G,pos+G,b+G,T,H,D,R,theta,deps)
                   : ie::rope_partial(q,alias?b+G:x+G,pos+G,b+G,T,H,D,R,theta,deps);
    auto av=read(q,a,n+2*G,old),bv=read(q,b,n+2*G,now);
    exact(av,bv); guards(bv);
    for(unsigned t=0;t<T;++t)for(unsigned h=0;h<H;++h) {
        size_t off=G+(size_t(t)*H+h)*D;
        for(unsigned d=R;d<D;++d) require(!std::memcmp(&bv[off+d],&src[off+d],2),"RoPE passthrough");
        if(!special)for(unsigned r=0;r<R/2;++r) {
            // Match the documented float angle precision, not the device native approximation.
            float angle=float(hp[G+(inter?size_t(r%3)*T:0)+t])*
                std::exp(-float(2*r)/float(R)*std::log(theta));
            double u=float(src[off+r]),v=float(src[off+r+R/2]);
            near(float(bv[off+r]),u*std::cos(angle)-v*std::sin(angle),.03);
            near(float(bv[off+r+R/2]),u*std::sin(angle)+v*std::cos(angle),.03);
        }
    }
}
void gate(sycl::queue& q,unsigned T,unsigned H,unsigned D,bool alias,bool special) {
    Memory m{q,{}};size_t width=size_t(H)*D,n=size_t(T)*width;
    auto x=m.get<float>(n+2*G),a=m.get<float>(n+2*G),b=m.get<float>(n+2*G);
    auto bias=m.get<float>(width+2*G),scale=m.get<float>(H+2*G);
    std::vector<float> src(n+2*G,7),initial(n+2*G,7),hb(width+2*G,7),hs(H+2*G,7);
    for(size_t i=0;i<n;++i)src[G+i]=float(int(i*41%251)-125)/71;
    if(special)for(size_t i=0;i<n;i+=101)src[G+i]=i%2?std::numeric_limits<float>::infinity():-std::numeric_limits<float>::infinity();
    for(size_t i=0;i<width;++i)hb[G+i]=float(int(i*13%127)-63)/119;
    for(unsigned i=0;i<H;++i)hs[G+i]=-float(i%17+1)/21;
    auto held=delay(q);std::vector<sycl::event> deps{upload(q,x,src,held),upload(q,a,alias?src:initial,held),
        upload(q,b,alias?src:initial,held),upload(q,bias,hb,held),upload(q,scale,hs,held)};
    auto old=frozen::kda_gate(q,alias?a+G:x+G,bias+G,scale+G,a+G,T,H,D,-5,deps);
    auto now=ie::kda_gate(q,alias?b+G:x+G,bias+G,scale+G,b+G,T,H,D,-5,deps);
    auto av=read(q,a,n+2*G,old),bv=read(q,b,n+2*G,now);exact(av,bv);guards(bv);
    if(!special)for(size_t i=0;i<n;++i) {
        double v=-(double(src[G+i])+hb[G+i%width])*hs[G+(i%width)/D];
        near(bv[G+i],-5/(1+std::exp(-v)),1e-6);
    }
}

// An empty operation must carry a dependency even when it has no data to touch.
void empty_conv(sycl::queue& q) {
    std::atomic<bool> ready=false;
    auto e=q.submit([&](sycl::handler& h){h.host_task([&]{
        std::this_thread::sleep_for(std::chrono::milliseconds(50));ready=true;});});
    auto done=ie::depthwise_conv1d_causal(q,nullptr,nullptr,nullptr,nullptr,0,65,4,{e});
    done.wait_and_throw(); bool observed=ready.load();e.wait_and_throw();
    require(observed,"empty conv dropped dependency");
}
int main(int argc,char** argv) { try {
    sycl::queue q(sycl::gpu_selector_v); // deliberately out of order
    if(argc>1 && std::string(argv[1])=="empty") {empty_conv(q);puts("PASS empty conv dependency");return 0;}
    unsigned count=0;
    for(bool inter:{false,true})for(bool alias:{false,true})
      for(auto s:{std::array<unsigned,4>{1,32,256,64},{15,32,256,64},{16,31,256,64},
            {16,32,256,64},{17,3,65,64},{16,32,128,64},{16,32,256,128},
            {16,32,128,128},{17,32,65,64},{16,32,256,0},
            {0,32,256,64},{16,0,256,64},{16,32,0,0}}) {
        rope(q,s[0],s[1],s[2],s[3],inter,alias,false);++count;
      }
    for(bool inter:{false,true})for(bool alias:{false,true}) {rope(q,16,32,256,64,inter,alias,true);++count;}
    for(bool alias:{false,true})for(auto s:{std::array<unsigned,3>{1,64,128},{16,64,128},
            {128,64,128},{129,64,128},{3,7,65},{16,128,64},{0,64,128}}) {
        gate(q,s[0],s[1],s[2],alias,false);++count;
    }
    for(bool alias:{false,true}){gate(q,16,64,128,alias,true);++count;}
    printf("PASS %u Qwen RoPE / GLM gate cases: frozen outputs, host arithmetic, guards, delayed OOO producers\n",count);
}catch(std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
