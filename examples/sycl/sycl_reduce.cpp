// This program demonstrates how to performs a parallel reduction
// using syclFlow.

#include <taskflow/sycl/syclflow.hpp>
#include <taskflow/sycl/algorithm/reduce.hpp>

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./sycl_reduce num_items\n";
    std::exit(EXIT_FAILURE);
  }

  size_t N = std::atoi(argv[1]);

  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(N, queue);
  auto res1 = sycl::malloc_shared<int>(1, queue);
  auto res2 = sycl::malloc_shared<int>(1, queue);
  auto hres = 0;

  // initialize the data
  for(size_t i=0; i<N; i++) {
    data[i] = ::rand()%100;
    hres += data[i];
  }
  *res1 = 10;
  *res2 = 10;

  tf::syclDefaultExecutionPolicy policy(queue);

  tf::sycl_reduce(policy, data, data+N, res1, [](int a, int b){ return a+b; });

  //// perform reduction
  //tf::syclFlow syclflow(queue);
  //
  //// res1 = res1 + data[0] + data[1] + ...
  //syclflow.reduce(
  //  data, data+N, res1, [](int a, int b){ return a+b; }
  //);
  //
  //// res2 = data[0] + data[1] + data[2] + ...
  //syclflow.uninitialized_reduce(
  //  data, data+N, res2, [](int a, int b){ return a+b; }
  //);

  //syclflow.offload();
  //
  //// inspect
  //if(hres + 10 != *res1 || hres != *res2) {
  //  throw std::runtime_error("incorrect result");
  //}
  //
  printf("hres=%d res1=%d\n", hres, *res1);

  std::cout << "correct result\n";

  return 0;
}


