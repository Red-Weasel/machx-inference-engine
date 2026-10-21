// include/ie/onednn_scratch.hpp -- 2026-10-21: the oneDNN products' scratch, one device buffer per caller queue.
//
// gemm_onednn.cpp and gemm_onednn_b48.cpp cache their primitives per host thread and per shape (row count included). In
// oneDNN's default (library) scratchpad mode every cached primitive owned its own scratch on the card -- 12 to 20 MiB for
// a mid-sized fp16-output product -- for the life of the process, so every server pool thread x prompt-piece length kept
// another set (the 27B: ~100 MiB a card per request, measured with unitrace: zeMemAllocDevice and never zeMemFree). The
// products now use scratchpad_mode::user and run through onednn_execute, which hands them the queue's one buffer. The
// queue is in order, so its products never overlap; a larger buffer waits for the queue before the old one is freed.
// The mode does not change which kernel oneDNN picks (impl_info_str, library vs user: 720 / 720 shapes equal).
#pragma once

#include <sycl/sycl.hpp>

#include <dnnl.hpp>

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace ie {

// The scratch a primitive desc built with scratchpad_mode::user needs: `sp` is a handle-less memory for it (left empty and
// `bytes` 0 when it needs none). Call once per cached primitive.
void onednn_scratch_desc(const dnnl::primitive_desc_base& pd, const dnnl::engine& eng, dnnl::memory& sp, size_t& bytes);

// `prim` on `strm` (which wraps `q`) with `args`, plus DNNL_ARG_SCRATCHPAD = q's buffer when bytes > 0 (grown first when
// smaller; the handle and the submission are one critical section per queue). A non-empty `err` = nothing was submitted.
sycl::event onednn_execute(sycl::queue& q, const dnnl::stream& strm, const dnnl::primitive& prim,
                           std::unordered_map<int, dnnl::memory> args, dnnl::memory& sp, size_t bytes,
                           const std::vector<sycl::event>& deps, std::string& err);

}  // namespace ie
