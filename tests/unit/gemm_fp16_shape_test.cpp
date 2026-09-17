#include "ie/ops.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>

// Dyadic inputs make the integer dot product / 512 an exact FP32 oracle.
// Check sampled columns of every row, padded allocation canaries, and events
// on an out-of-order queue. No model files are needed.
int main() {
    std::vector<sycl::device> devices;
    for (auto d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_info<sycl::info::device::name>().find("B70") != std::string::npos)
            devices.push_back(d);
    if (devices.empty()) { std::puts("No B70 available"); return 77; }
    sycl::queue q(devices.back());
    struct Shape { unsigned m, n, k; };
    for (auto [m,n,k] : std::vector<Shape>{{1,2048,1024},{5,2048,1024},
             {32,4096,1536},{64,2048,1024},{96,2048,1024},{97,2048,1024},
             {128,4096,1024},{129,4096,1024},{17,2176,1024},
             {5,128,63},{17,128,65},{5,2048,1025}}) {
        constexpr size_t guard=32;
        const size_t nc=size_t((m+7)/8*8)*n+2*guard;
        std::vector<sycl::half> a(size_t(m)*k), b(size_t(k)*n);
        std::vector<float> c(nc);
        auto ai=[](size_t i) { return int((i*37+i/97)%17)-8; };
        auto bi=[](size_t i) { return int((i*17+i/101)%19)-9; };
        for(size_t i=0;i<a.size();++i)a[i]=sycl::half(ai(i)/16.f);
        for(size_t i=0;i<b.size();++i)b[i]=sycl::half(bi(i)/32.f);
        auto* da=sycl::malloc_device<sycl::half>(a.size(),q);
        auto* db=sycl::malloc_device<sycl::half>(b.size(),q);
        auto* dc=sycl::malloc_device<float>(nc,q);
        if(!da||!db||!dc)return 2;
        std::vector<sycl::event> deps={q.memcpy(da,a.data(),a.size()*2),
            q.memcpy(db,b.data(),b.size()*2),q.fill(dc,12345.f,nc)};
        auto done=ie::gemm_fp16(q,da,db,dc+guard,m,n,k,deps);
        q.submit([&](sycl::handler& h){h.depends_on(done);h.memcpy(c.data(),dc,nc*4);}).wait_and_throw();
        for(size_t i=0;i<guard;++i)
            if(c[i]!=12345.f||c[nc-guard+i]!=12345.f)return 3;
        for(unsigned row=0;row<m;++row)for(unsigned col : {0u,1u,15u,16u,n/2,n-1}) {
            int sum=0;
            for(unsigned j=0;j<k;++j)sum+=ai(size_t(row)*k+j)*bi(size_t(j)*n+col);
            float expected=sum/512.f;
            if(c[guard+size_t(row)*n+col]!=expected) {
                std::printf("FAIL M%u N%u K%u row%u col%u: %g != %g\n",m,n,k,row,col,c[guard+size_t(row)*n+col],expected);
                return 4;
            }
        }
        sycl::free(da,q);sycl::free(db,q);sycl::free(dc,q);
        std::printf("PASS M%u N%u K%u exact oracle, canaries, OOO events\n",m,n,k);
    }
}
