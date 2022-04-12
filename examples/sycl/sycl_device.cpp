// This program pulls out all platforms and devices using SYCL.

#include <taskflow/sycl/syclflow.hpp>

int main() {

  std::vector<sycl::platform> platforms = sycl::platform::get_platforms();

  // looping over platforms
  for (const auto& platform : platforms) {

    std::cout << "Platform   : "
	            << platform.get_info<sycl::info::platform::name>() << '\n'
              << "is_host    : "
              << platform.is_host() << '\n'
              << "version    : "
              << platform.get_info<sycl::info::platform::version>() << '\n'
              << "vendor     : "
              << platform.get_info<sycl::info::platform::vendor>() << '\n'
              << "profile    : "
              << platform.get_info<sycl::info::platform::profile>() << '\n';
              //<< "extensions :"
              //<< platform.get_info<sycl::info::platform::extensions>() << '\n';

    // getting the list of devices from the platform
    std::vector<sycl::device> devices = platform.get_devices();

    // looping over devices
    for (const auto& device : devices) {

      std::cout << "  Device             : "
		            << device.get_info<sycl::info::device::name>() << '\n'
                << "  vendor             : "
                << device.get_info<sycl::info::device::vendor>() << '\n'
                << "  version            : "
                << device.get_info<sycl::info::device::version>() << '\n'
                << "  is_host            : " << device.is_host() << '\n'
                << "  is_cpu             : " << device.is_cpu() << '\n'
                << "  is_gpu             : " << device.is_gpu() << '\n'
                << "  is_accelerator     : " << device.is_accelerator() << '\n'
                << "  max_work_group_size: "
                << device.get_info<sycl::info::device::max_work_group_size>() << '\n'
                << "  local_mem_size     : "
                << device.get_info<sycl::info::device::local_mem_size>() << '\n';

      // submitting a kernel to the sycl device
      auto queue = sycl::queue(device);
      queue.submit([](sycl::handler& handler){
        handler.single_task([](){});
      });
    }

    std::cout << std::endl;
  }

  return 0;
}
