add_executable(hello_gpu hello_gpu.cpp)
target_link_libraries(hello_gpu PRIVATE ie_sycl_flags)
