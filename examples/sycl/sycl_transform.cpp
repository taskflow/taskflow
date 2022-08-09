// This program demonstrates how to performs a parallel transform
// using syclFlow.

#include <taskflow/syclflow.hpp>

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./sycl_transform num_items\n";
    std::exit(EXIT_FAILURE);
  }

  size_t N = std::atoi(argv[1]);

  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(N, queue);
  auto src1 = sycl::malloc_shared<int>(N, queue);
  auto src2 = sycl::malloc_shared<int>(N, queue);
  auto src3 = sycl::malloc_shared<int>(N, queue);

  // initialize the data
  for(size_t i=0; i<N; i++) {
    data[i] = 0;
    src1[i] = 1;
    src2[i] = 2;
    src3[i] = 3;
  }

  // perform parallel transform
  tf::syclFlow syclflow(queue);

  // data[i] = src1[i] + src2[i] + src3[i]
  syclflow.transform(
    data, data+N, [](int a, int b, int c) { return a+b+c; }, src1, src2, src3
  );

  syclflow.offload();

  // inspect the result
  for(size_t i=0; i<N; i++) {
    if(data[i] != (src1[i] + src2[i] + src3[i])) {
      throw std::runtime_error("incorrect result");
    }
  }

  std::cout << "correct result\n";

  return 0;
}
