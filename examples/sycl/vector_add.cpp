#include <iostream>
#include <vector>
#include <SYCL/sycl.hpp>

#define N 10

class add;

int main() {
  
  std::cout << "SYCL VERSION: " << CL_SYCL_LANGUAGE_VERSION << '\n';

  std::vector<int> dA(N), dB(N), dC(N);

  for(size_t i=0; i<N; ++i) {
    dA[i] = 1;
    dB[i] = 2;
    dC[i] = 0;
  }
  
  {
    cl::sycl::queue gpuQueue{cl::sycl::default_selector{}};
    
    auto device = gpuQueue.get_device();
    auto deviceName = device.get_info<cl::sycl::info::device::name>();
    std::cout << "running vector-add on device: " << deviceName << '\n';

    cl::sycl::buffer<int, 1> bufA(dA.data(), cl::sycl::range<1>(dA.size()));
    cl::sycl::buffer<int, 1> bufB(dB.data(), cl::sycl::range<1>(dB.size()));
    cl::sycl::buffer<int, 1> bufC(dC.data(), cl::sycl::range<1>(dC.size()));

    gpuQueue.submit([&](cl::sycl::handler& cgh){
      auto inA = bufA.get_access<cl::sycl::access::mode::read>(cgh);
      auto inB = bufB.get_access<cl::sycl::access::mode::read>(cgh);
      auto out = bufC.get_access<cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<add>(cl::sycl::range<1>(dA.size()), [=](cl::sycl::id<1> i) {
        out[i] = inA[i] + inB[i];
      });
    });
  }
  
  bool correct = true;
  for(int i=0; i<N; i++) {
    if(dC[i] != dA[i] + dB[i]) {
      correct = false;
    }
  }

  std::cout << (correct ? "result is correct" : "result is incorrect")
            << std::endl;

  return 0;
}


