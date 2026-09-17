#pragma once
#include <sycl/sycl.hpp>
namespace kda_norm_frozen {
sycl::event kda_l2norm(sycl::queue& q,
                       const float* x, float* y,
                       uint32_t n_rows, uint32_t head_dim, float scale, float eps,
                       const std::vector<sycl::event>& deps) {
    constexpr int SG = 16;
    return q.submit( [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_rows) * SG, SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto sg = it.get_sub_group();
            const float* xr = x + uint64_t(row) * head_dim;
            float* yr = y + uint64_t(row) * head_dim;
            float partial = 0.f;
            for (uint32_t i = lid; i < head_dim; i += SG) partial += xr[i] * xr[i];
            const float sum_sq = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
            const float r = scale / sycl::fmax(sycl::sqrt(sum_sq), eps);
            for (uint32_t i = lid; i < head_dim; i += SG) yr[i] = xr[i] * r;
        });
    });
}

}
