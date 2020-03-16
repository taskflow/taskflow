#include <taskflow/taskflow.hpp>

int main() {

  // Number of CUDA devices
  auto num_cuda_devices = tf::cuda_num_devices();

  std::cout << "There are " << num_cuda_devices << " CUDA devices.\n";

  // Iterate through devices
  for(unsigned i = 0; i < num_cuda_devices; ++i) {
    std::cout << "CUDA device #" << i << '\n';
    tf::cuda_dump_device_property(std::cout, tf::cuda_get_device_property(i));
  }

  return 0;
}
