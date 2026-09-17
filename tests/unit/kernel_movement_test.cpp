// Frozen indexing references plus independent host layout oracles.
// Define IE_MOVEMENT_PRIVATE_HEADER to a quoted accepted.hpp path for standalone probes.
#include "ie/ops.hpp"
#ifdef IE_MOVEMENT_PRIVATE_HEADER
#include IE_MOVEMENT_PRIVATE_HEADER
namespace tested = movement_accepted;
#else
namespace tested = ie;
#endif
#include "ie/kernel_profiler.hpp"
namespace frozen {
sycl::event cast_qkv_split_fp16_to_fp32(sycl::queue& q,
                                        const sycl::half* src,
                                        float* q_dst, float* k_dst, float* v_dst,
                                        uint32_t T, uint32_t k_total, uint32_t v_total,
                                        const std::vector<sycl::event>& deps) {
    // Each row of `src` is laid out [Q (k_total), K (k_total), V (v_total)].
    // Total elements per row = 2*k_total + v_total = SI2.
    const uint32_t row_stride = 2 * k_total + v_total;
    const uint64_t total_q = uint64_t(T) * k_total;
    const uint64_t total_k = uint64_t(T) * k_total;
    const uint64_t total_v = uint64_t(T) * v_total;
    const uint64_t total = total_q + total_k + total_v;
    return ie::ps(q, "cast_qkv_split", [&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint64_t WG = 256;
        const uint64_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            // Map a flat index across (Q∥K∥V) into one of the three buffers.
            uint64_t local = i;
            if (local < total_q) {
                const uint32_t t = uint32_t(local / k_total);
                const uint32_t d = uint32_t(local % k_total);
                q_dst[uint64_t(t) * k_total + d] =
                    float(src[uint64_t(t) * row_stride + d]);
                return;
            }
            local -= total_q;
            if (local < total_k) {
                const uint32_t t = uint32_t(local / k_total);
                const uint32_t d = uint32_t(local % k_total);
                k_dst[uint64_t(t) * k_total + d] =
                    float(src[uint64_t(t) * row_stride + k_total + d]);
                return;
            }
            local -= total_k;
            // V
            const uint32_t t = uint32_t(local / v_total);
            const uint32_t d = uint32_t(local % v_total);
            v_dst[uint64_t(t) * v_total + d] =
                float(src[uint64_t(t) * row_stride + 2 * k_total + d]);
        });
    });
}

sycl::event split_q_gate_per_head(sycl::queue& q,
                                  const sycl::half* qk,
                                  sycl::half* q_out, sycl::half* gate_out,
                                  uint32_t T, uint32_t n_heads, uint32_t head_dim,
                                  const std::vector<sycl::event>& deps) {
    return ie::ps(q, "split_q_gate", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint64_t total = uint64_t(T) * n_heads * head_dim;
        constexpr uint64_t WG = 256;
        const uint64_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t idx = it.get_global_id(0);
            if (idx >= total) return;
            const uint64_t d   = idx % head_dim;
            const uint64_t h   = (idx / head_dim) % n_heads;
            const uint64_t t   = idx / (n_heads * head_dim);
            // Source (qk): index in [T, n_heads, 2, head_dim] flat.
            const uint64_t src_q   = ((t * n_heads + h) * 2 + 0) * head_dim + d;
            const uint64_t src_g   = ((t * n_heads + h) * 2 + 1) * head_dim + d;
            q_out   [idx] = qk[src_q];
            gate_out[idx] = qk[src_g];
        });
    });
}

sycl::event repeat_interleave_heads(sycl::queue& q,
                                    const float* x, float* y,
                                    uint32_t T, uint32_t n_in_heads,
                                    uint32_t head_dim, uint32_t repeat,
                                    bool interleave,
                                    const std::vector<sycl::event>& deps) {
    return ie::ps(q, "repeat_heads", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint32_t n_out_heads = n_in_heads * repeat;
        const uint64_t total = uint64_t(T) * n_out_heads * head_dim;
        constexpr uint64_t WG = 256;
        const uint64_t global = ((total + WG - 1) / WG) * WG;
        h.parallel_for(sycl::nd_range<1>(global, WG),
                       [=](sycl::nd_item<1> it) {
            const uint64_t idx = it.get_global_id(0);
            if (idx >= total) return;
            const uint64_t d  = idx % head_dim;
            const uint64_t h_out = (idx / head_dim) % n_out_heads;
            const uint64_t t  = idx / (n_out_heads * head_dim);
            const uint64_t h_in = interleave ? (h_out / repeat)      // qwen3next
                                             : (h_out % n_in_heads); // 27B tiled
            y[idx] = x[(t * n_in_heads + h_in) * head_dim + d];
        });
    });
}

}
#include <bit>
#include <chrono>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
using H=sycl::half;
using Deps=const std::vector<sycl::event>&;
using Cast=sycl::event(*)(sycl::queue&,const H*,float*,float*,float*,uint32_t,uint32_t,uint32_t,Deps);
using Split=sycl::event(*)(sycl::queue&,const H*,H*,H*,uint32_t,uint32_t,uint32_t,Deps);
using Repeat=sycl::event(*)(sycl::queue&,const float*,float*,uint32_t,uint32_t,uint32_t,uint32_t,bool,Deps);
const Cast casts[]={frozen::cast_qkv_split_fp16_to_fp32,tested::cast_qkv_split_fp16_to_fp32};
const Split splits[]={frozen::split_q_gate_per_head,tested::split_q_gate_per_head};
const Repeat repeats[]={frozen::repeat_interleave_heads,tested::repeat_interleave_heads};
void require(bool b,const std::string& s) { if(!b) throw std::runtime_error(s); }
template<class T> struct Buffer {
    sycl::queue& q; T* raw; T* p; size_t n,offset,size;
    std::vector<T> initial;
    Buffer(sycl::queue& queue,size_t count,size_t shift=0):q(queue),n(count),offset(32+shift),size(n+offset+32),initial(size) {
        raw=sycl::malloc_shared<T>(size,q); require(raw!=nullptr,"allocation"); p=raw+offset;
        std::memset(initial.data(),0xa5,size*sizeof(T)); std::memcpy(raw,initial.data(),size*sizeof(T));
    }
    ~Buffer(){sycl::free(raw,q);}
    void guards(const std::string& label) {
        require(std::memcmp(raw,initial.data(),offset*sizeof(T))==0,label+" prefix");
        require(std::memcmp(p+n,initial.data()+offset+n,32*sizeof(T))==0,label+" suffix");
    }
    void equal(const std::vector<T>& expected,const std::string& label) {
        require(expected.size()==n,"wrong expected size");
        for(size_t i=0;i<n;++i) if(std::memcmp(p+i,expected.data()+i,sizeof(T)))
            throw std::runtime_error(label+" element "+std::to_string(i));
        guards(label);
    }
};
H half_value(size_t i) { return std::bit_cast<H>(uint16_t(((i*7919)%0x7c00) | ((i&1)<<15))); }
float float_value(size_t i) {
    static const uint32_t special[]={0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0x7f800123,1,0x807fffff};
    return std::bit_cast<float>(i%13==0?special[(i/13)%8]:uint32_t(i*2654435761u));
}
template<class T> sycl::event delayed_input(sycl::queue& q,Buffer<T>& src) {
    std::memset(src.p,0x5a,src.n*sizeof(T));
    auto delay=q.submit([&](sycl::handler& h){h.host_task([]{std::this_thread::sleep_for(std::chrono::milliseconds(2));});});
    return q.submit([&](sycl::handler& h){h.depends_on(delay);h.memcpy(src.raw,src.initial.data(),src.size*sizeof(T));});
}
template<class T> void consume(sycl::queue& q,sycl::event e,Buffer<T>& out) {
    std::vector<T> consumed(out.n);
    if(out.n) q.submit([&](sycl::handler& h){h.depends_on(e);h.memcpy(consumed.data(),out.p,out.n*sizeof(T));}).wait_and_throw();
    else e.wait_and_throw();
    require(std::memcmp(consumed.data(),out.p,out.n*sizeof(T))==0,"consumer event");
}
void check_cast(sycl::queue& q,uint32_t T,uint32_t K,uint32_t V,size_t shift) {
    const size_t stride=2ull*K+V;
    Buffer<H> src(q,T*stride,shift);
    for(size_t i=0;i<src.n;++i) src.initial[src.offset+i]=half_value(i);
    std::vector<float> a(T*size_t(K)),b(a.size()),c(T*size_t(V));
    size_t input=0;
    for(uint32_t row=0;row<T;++row) {
        for(uint32_t d=0;d<K;++d) a[size_t(row)*K+d]=float(src.initial[src.offset+input++]);
        for(uint32_t d=0;d<K;++d) b[size_t(row)*K+d]=float(src.initial[src.offset+input++]);
        for(uint32_t d=0;d<V;++d) c[size_t(row)*V+d]=float(src.initial[src.offset+input++]);
    }
    for(auto f:casts) {
        Buffer<float> ao(q,a.size(),shift+1),bo(q,b.size(),shift+2),co(q,c.size(),shift+3);
        auto ready=delayed_input(q,src); auto e=f(q,src.p,ao.p,bo.p,co.p,T,K,V,{ready});
        consume(q,e,ao); consume(q,e,bo); consume(q,e,co);
        ao.equal(a,"cast Q"); bo.equal(b,"cast K"); co.equal(c,"cast V");
        require(std::memcmp(src.raw,src.initial.data(),src.size*sizeof(H))==0,"cast changed input");
    }
}
void check_split(sycl::queue& q,uint32_t T,uint32_t heads,uint32_t dim,size_t shift) {
    const size_t n=size_t(T)*heads*dim;
    Buffer<H> src(q,2*n,shift);
    for(size_t i=0;i<src.n;++i) src.initial[src.offset+i]=std::bit_cast<H>(uint16_t(i*7919));
    std::vector<H> a(n),b(n); size_t input=0,output=0;
    for(uint32_t t=0;t<T;++t) for(uint32_t h=0;h<heads;++h) {
        for(uint32_t d=0;d<dim;++d) a[output+d]=src.initial[src.offset+input++];
        for(uint32_t d=0;d<dim;++d) b[output+d]=src.initial[src.offset+input++];
        output+=dim;
    }
    for(auto f:splits) {
        Buffer<H> ao(q,n,shift+1),bo(q,n,shift+2);
        auto ready=delayed_input(q,src); auto e=f(q,src.p,ao.p,bo.p,T,heads,dim,{ready});
        consume(q,e,ao); consume(q,e,bo); ao.equal(a,"split Q");bo.equal(b,"split gate");
        require(std::memcmp(src.raw,src.initial.data(),src.size*sizeof(H))==0,"split changed input");
        if(T==1 && heads==1) {
            auto ready2=delayed_input(q,src); f(q,src.p,src.p,src.p+dim,T,heads,dim,{ready2}).wait_and_throw();
            require(std::memcmp(src.raw,src.initial.data(),src.size*sizeof(H))==0,"split identity alias");
        }
    }
}
void check_repeat(sycl::queue& q,uint32_t T,uint32_t heads,uint32_t dim,uint32_t rep,bool interleave,size_t shift) {
    const size_t n=size_t(T)*heads*dim;
    Buffer<float> src(q,n,shift); for(size_t i=0;i<n;++i) src.initial[src.offset+i]=float_value(i);
    std::vector<float> expected(n*rep);size_t o=0;
    for(uint32_t t=0;t<T;++t) {
        if(interleave) {
            for(uint32_t head=0;head<heads;++head) for(uint32_t r=0;r<rep;++r)
                for(uint32_t d=0;d<dim;++d) expected[o++]=src.initial[src.offset+(size_t(t)*heads+head)*dim+d];
        } else {
            for(uint32_t r=0;r<rep;++r) for(uint32_t head=0;head<heads;++head)
                for(uint32_t d=0;d<dim;++d) expected[o++]=src.initial[src.offset+(size_t(t)*heads+head)*dim+d];
        }
    }
    for(auto f:repeats) {
        Buffer<float> out(q,n*rep,shift+1);auto ready=delayed_input(q,src);
        auto e=f(q,src.p,out.p,T,heads,dim,rep,interleave,{ready});consume(q,e,out);out.equal(expected,"repeat");
        require(std::memcmp(src.raw,src.initial.data(),src.size*sizeof(float))==0,"repeat changed input");
        if(rep==1) {
            auto r=delayed_input(q,src);f(q,src.p,src.p,T,heads,dim,rep,interleave,{r}).wait_and_throw();
            require(std::memcmp(src.raw,src.initial.data(),src.size*sizeof(float))==0,"repeat identity alias");
        }
    }
}
void check_zero_dependency(sycl::queue& q) {
    int* flag=sycl::malloc_shared<int>(2,q);
    for(int op=0;op<3;++op) for(int v=0;v<2;++v) for(int dim=0;dim<4;++dim) {
        flag[0]=0;flag[1]=0;
        auto delay=q.submit([&](sycl::handler& h){h.host_task([=]{std::this_thread::sleep_for(std::chrono::milliseconds(2));flag[0]=17;});});
        sycl::event e;
        if(op==0) e=casts[v](q,nullptr,nullptr,nullptr,nullptr,dim==0?0:1,0,0,{delay});
        if(op==1) e=splits[v](q,nullptr,nullptr,nullptr,dim==0?0:1,dim==1?0:1,dim>=2?0:64,{delay});
        if(op==2) e=repeats[v](q,nullptr,nullptr,dim==0?0:1,dim==1?0:1,dim==2?0:64,dim==3?0:2,false,{delay});
        q.submit([&](sycl::handler& h){h.depends_on(e);h.single_task([=]{flag[1]=flag[0];});}).wait_and_throw();
        require(flag[1]==17,"empty event lost dependency");
    }
    sycl::free(flag,q);
}
void check_cast_all_half_bits(sycl::queue& q) {
    struct Shape { uint32_t T, K, V; };
    // Exercise both the flat fallback and the accepted vector-cast dispatch.
    // Each input includes every half bit pattern, including signaling NaNs.
    for (const Shape shape : {Shape{1, 32768, 0}, Shape{16, 2048, 4096}}) {
        const auto [T, K, V] = shape;
        const size_t q_count = size_t(T) * K;
        const size_t v_count = size_t(T) * V;
        Buffer<H> src(q, 2 * q_count + v_count, 1);
        for (size_t i = 0; i < src.n; ++i)
            src.initial[src.offset + i] = std::bit_cast<H>(uint16_t(i));
        std::vector<float> expected_a(q_count), expected_b(q_count), expected_c(v_count);
        for (int variant = 0; variant < 2; ++variant) {
            Buffer<float> a(q, q_count, 1), b(q, q_count, 3), c(q, v_count, 2);
            auto ready = delayed_input(q, src);
            auto event = casts[variant](q, src.p, a.p, b.p, c.p, T, K, V, {ready});
            consume(q, event, a); consume(q, event, b); consume(q, event, c);
            if (variant == 0) {
                std::memcpy(expected_a.data(), a.p, a.n * sizeof(float));
                std::memcpy(expected_b.data(), b.p, b.n * sizeof(float));
                if (v_count) std::memcpy(expected_c.data(), c.p, c.n * sizeof(float));
            } else {
                a.equal(expected_a, "all-half-bit cast Q");
                b.equal(expected_b, "all-half-bit cast K");
                c.equal(expected_c, "all-half-bit cast V");
            }
            a.guards("all-half cast Q"); b.guards("all-half cast K"); c.guards("all-half cast V");
            require(std::memcmp(src.raw, src.initial.data(), src.size * sizeof(H)) == 0,
                    "all-half cast input");
        }
    }
}
void correctness(sycl::queue& q) {
    for(auto T:{1u,3u,64u}) for(auto dim:{1u,3u,31u,64u,127u,128u,256u}) {
        check_cast(q,T,dim*3,dim*5,1);check_split(q,T,T==1?1:3,dim,1);
        for(bool interleave:{false,true}) check_repeat(q,T,3,dim,2,interleave,1);
    }
    check_cast(q,2,0,19,3);check_cast(q,2,19,0,3);
    check_cast(q,1,32768,0,1); // wide half conversion, finite patterns cover signs and denormals
    check_cast_all_half_bits(q);
    for(bool mode:{false,true}) for(auto rep:{1u,3u,4u}) check_repeat(q,3,5,67,rep,mode,3);

    // Exact accepted shapes and both sides of every dispatch boundary.
    for(uint32_t T:{1u,15u,16u,17u,1024u,1025u}) {
        check_cast(q,T,1024,2048,1);
        check_cast(q,T,2048,4096,3);
        check_cast(q,T,2048,6144,1);
    }
    for(uint32_t K:{1023u,1025u}) check_cast(q,16,K,2048,3);
    for(uint32_t V:{4095u,4097u,6143u,6145u}) check_cast(q,16,2048,V,3);
    for(uint32_t T:{1u,16u,64u}) for(uint32_t H:{15u,16u,17u,23u,24u,25u,31u,32u,33u})
        for(uint32_t D:{127u,128u,129u,255u,256u,257u}) check_split(q,T,H,D,3);
    for(uint32_t T:{1024u,1025u}) check_split(q,T,16,128,1);
    for(bool mode:{false,true}) {
        for(uint32_t T:{1u,16u,1024u,1025u}) for(uint32_t D:{128u,256u}) for(uint32_t R:{2u,3u})
            check_repeat(q,T,16,D,R,mode,3);
        for(uint32_t H:{15u,17u}) check_repeat(q,16,H,128,2,mode,1);
        for(uint32_t D:{127u,129u,255u,257u}) check_repeat(q,16,16,D,2,mode,1);
        for(uint32_t R:{1u,4u}) check_repeat(q,16,16,128,R,mode,1);
    }
    check_zero_dependency(q);
    std::cout<<"CORRECTNESS PASS finite cast bit oracle; raw half/float movement; frozen equivalence; guards; shifts; tails; identity aliases; delayed OOO input and consumer; empty dependencies\n";
}
int main(int argc,char** argv) {
    try {
        std::cout<<std::fixed<<std::setprecision(3);

        int i=0;
        for(auto d:sycl::device::get_devices(sycl::info::device_type::gpu)) {
            auto name=d.get_info<sycl::info::device::name>();if(name.find("B70")==std::string::npos) continue;
            std::cout<<"DEVICE,"<<i<<","<<name<<"\n";
            sycl::queue q(d,sycl::property_list{sycl::property::queue::enable_profiling{}});
            require(!q.has_property<sycl::property::queue::in_order>(),"expected OOO queue");
            correctness(q);++i;
        }
        require(i>0,"expected at least one selected B70 GPU");std::cout<<"ALL PASS\n";
    } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
