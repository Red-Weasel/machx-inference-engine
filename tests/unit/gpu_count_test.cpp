#include "ie/memory_plan.hpp"
#include <cstdio>
int main() {
 unsigned n=0;
 for (auto d:sycl::device::get_devices())
  if(d.is_gpu() && d.get_backend()==sycl::backend::ext_oneapi_level_zero && d.get_info<sycl::info::device::name>().find("B70")!=std::string::npos) ++n;
 unsigned actual=ie::count_matching_gpus("B70");
 std::printf("Level Zero physical B70s=%u planner B70s=%u\n",n,actual);
 if (!n) return 77;
 return actual != n ? 1 : 0;
}
