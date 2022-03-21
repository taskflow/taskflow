// This program demonstrates how to rebind syclFlow tasks
// to different device operations.

#include <taskflow/syclflow.hpp>

int main() {

  size_t N = 10000;

  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(N, queue);

  tf::syclFlow syclflow(queue);

  // fill data with -1
  std::cout << "filling data with -1 ...\n";

  tf::syclTask task = syclflow.fill(data, -1, N);
  syclflow.offload();

  for(size_t i=0; i<N; i++) {
    if(data[i] != -1) {
      throw std::runtime_error("unexpected result after fill");
    }
  }
  std::cout << "correct result after fill\n";

  // rebind the task to for-each task setting each element to 100
  // You can rebind a syclTask to any other task type.
  std::cout << "rebind task to for_each task setting each element to 100\n";

  syclflow.for_each(
    task, data, data+N, [](int& i){ i = 100; }
  );
  syclflow.offload();

  for(size_t i=0; i<N; i++) {
    if(data[i] != 100) {
      throw std::runtime_error("unexpected result after for_each");
    }
  }
  std::cout << "correct result after updating for_each\n";


  return 0;
}



