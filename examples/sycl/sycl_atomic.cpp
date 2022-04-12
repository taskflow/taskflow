// This program demonstrates how to create a simple vector-add
// application using syclFlow and unified shared memory (USM).

#include <taskflow/sycl/syclflow.hpp>

constexpr size_t N = 10000;

int main() {

  // create a standalone scylFlow
  sycl::queue queue;

  tf::syclFlow syclflow(queue);

  // allocate a shared memory and initialize the data
  auto data = sycl::malloc_shared<int>(N, queue);

  for(size_t i=0; i<N; i++) {
    data[i] = i;
  }

  // reduce the summation to the first element using ONEAPI atomic_ref
  syclflow.parallel_for(
    sycl::range<1>(N), [=](sycl::id<1> id) {

      auto ref = sycl::atomic_ref<
        int, 
        sycl::memory_order_relaxed, 
        sycl::memory_scope::device,
        sycl::access::address_space::global_space
      >{data[0]};

      ref.fetch_add(data[id]);
    }
  );

  // run the syclflow
  syclflow.offload();

  // create a deallocate task that checks the result and frees the memory
  if(data[0] != (N-1)*N/2) {
    std::cout << data[0] << '\n';
    throw std::runtime_error("incorrect result");
  }

  std::cout << "correct result\n";

  // deallocates the memory
  sycl::free(data, queue);


  return 0;
}


