// src/ops/conv1d.cpp — depthwise causal conv1d (kernel=4 default).
//
// One work-item per (output_token t, channel c). Each item reads up to `kernel`
// inputs along time — first from `conv_state` (the past kernel-1 tokens), then
// from `x` itself for any t-k >= 0.
//
// After the chunk is processed, the kernel writes the last (kernel-1) tokens
// of the current x back into `conv_state` for the next call's left-context.
//
// `apply_silu` (default true): fuses the silu activation that the DeltaNet
// pipeline would otherwise launch as a separate kernel right after.

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

sycl::event depthwise_conv1d_causal(sycl::queue& q,
                                    const sycl::half* x, const sycl::half* w,
                                    sycl::half* conv_state,
                                    sycl::half* y,
                                    uint32_t T, uint32_t channels, uint32_t kernel,
                                    const std::vector<sycl::event>& deps) {
    if (T == 0 || channels == 0 || kernel == 0) return {};

    auto compute_evt = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        constexpr uint32_t WG_C = 64;
        const uint64_t global_t = T;
        const uint64_t global_c = ((channels + WG_C - 1) / WG_C) * WG_C;
        h.parallel_for(sycl::nd_range<2>({global_t, global_c}, {1, WG_C}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t t = uint32_t(it.get_global_id(0));
            const uint32_t c = uint32_t(it.get_global_id(1));
            if (c >= channels) return;
            float acc = 0.f;
            for (uint32_t k = 0; k < kernel; ++k) {
                const int32_t src_t = int32_t(t) - int32_t(k);
                float v = 0.f;
                if (src_t >= 0) {
                    v = float(x[uint64_t(src_t) * channels + c]);
                } else if (conv_state) {
                    // Past tokens stored in conv_state[(kernel-1) + src_t, c].
                    const int32_t past_idx = int32_t(kernel - 1) + src_t;  // 0..kernel-2
                    if (past_idx >= 0)
                        v = float(conv_state[uint64_t(past_idx) * channels + c]);
                }
                // GGUF conv weight has shape [kernel, channels] with kernel as
                // the leading (contiguous) dim — element (k, c) at offset k + c*kernel.
                // PyTorch/llama.cpp convention: W[0] multiplies the oldest tap
                // (t-(K-1)), W[K-1] multiplies the current tap (t). Since our
                // loop variable k indexes "lag from current" (src_t = t - k),
                // we need W[K-1-k] here.
                const uint32_t wk = (kernel - 1) - k;
                acc += v * float(w[uint64_t(wk) + uint64_t(c) * kernel]);
            }
            // Inline SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
            // Saves a separate silu launch in the DeltaNet pipeline.
            const float silu = acc / (1.0f + sycl::native::exp(-acc));
            y[uint64_t(t) * channels + c] = sycl::half(silu);

            // v1.4 fusion: at T==1 each item owns channel c outright (its
            // state taps were read above), so the state shift-and-append
            // happens here and the separate writeback launch is skipped.
            // Ascending p reads slot p+1 before any overwrite of p+1.
            if (T == 1 && conv_state) {
                for (uint32_t p = 0; p + 1 < kernel - 1; ++p)
                    conv_state[uint64_t(p) * channels + c] =
                        conv_state[uint64_t(p + 1) * channels + c];
                conv_state[uint64_t(kernel - 2) * channels + c] = x[c];
            }
        });
    });

    if (!conv_state || T == 1) return compute_evt;

    // Update conv_state with the last (kernel-1) tokens of x (after compute completes).
    return ie::ps(q, "conv1d", [&](sycl::handler& h) {
        h.depends_on(compute_evt);
        const uint32_t past = kernel - 1;
        constexpr uint32_t WG_C = 64;
        const uint64_t global_c = ((channels + WG_C - 1) / WG_C) * WG_C;
        const uint64_t global_p = past;
        h.parallel_for(sycl::nd_range<2>({global_p, global_c}, {1, WG_C}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t p = uint32_t(it.get_global_id(0));   // 0..past-1
            const uint32_t c = uint32_t(it.get_global_id(1));
            if (c >= channels) return;
            // Token index in x corresponding to slot `p` of the new conv_state.
            // We want conv_state[p, c] = x[T - past + p, c] when T >= past.
            // If T < past, fold the existing state forward.
            sycl::half v;
            if (T >= past) {
                const uint32_t src_t = T - past + p;
                v = x[uint64_t(src_t) * channels + c];
            } else {
                // T < past: shift old state down by T, copy what fits from x.
                const int32_t src = int32_t(p) + int32_t(T) - int32_t(past);
                if (src < 0) {
                    // Read from old conv_state at index (past + src) = p + T - past + past = p + T
                    // Wait, let's redo: we want the new state to be the last `past` tokens of
                    // the concatenation [old_state, x]. old_state has `past` slots; x has T.
                    // Total length = past + T. Take the last `past`: indices [T, T+past).
                    // Index `i` of new state corresponds to old-or-x position `T + i`.
                    // For i in [0..past), pos = T + i. If pos < past => from old state.
                    const uint32_t src_old = uint32_t(int32_t(p) + int32_t(T));
                    v = conv_state[uint64_t(src_old) * channels + c];
                } else {
                    v = x[uint64_t(src) * channels + c];
                }
            }
            conv_state[uint64_t(p) * channels + c] = v;
        });
    });
}

}  // namespace ie
