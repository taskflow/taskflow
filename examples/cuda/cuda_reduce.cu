// This program demonstrate how to perform a parallel reduction 
// using cudaFlow.

#include <taskflow/cudaflow.hpp>

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./cuda_reduce num_items\n";
    std::exit(EXIT_FAILURE);
  }

  size_t N = std::atoi(argv[1]);

  auto data = tf::cuda_malloc_shared<int>(N);
  auto res1 = tf::cuda_malloc_shared<int>(1);
  auto res2 = tf::cuda_malloc_shared<int>(1);
  auto hres = 0;
  
  // initialize the data
  for(size_t i=0; i<N; i++) {
    data[i] = ::rand()%100;
    hres += data[i];
  }
  *res1 = 0;
  *res2 = 10;
  
  tf::cudaDefaultExecutionPolicy p;

  // using STL-styled reduction
  tf::cuda_transform_reduce(p, data, data+N, res1, tf::cuda_plus<int>{}, []__device__(int a){ return a + 1;});
  tf::cuda_uninitialized_reduce(p, data, data+N, res2, tf::cuda_plus<int>{});

  printf("href=%d, res1=%d(%d), res2=%d\n", hres, *res1, *res1 - hres, *res2);
  
  //if(hres + 10 != *res1 || hres != *res2) {
  //  throw std::runtime_error("incorrect result");
  //}

  //// perform reduction
  //tf::cudaFlow cudaflow;
  //
  //// res1 = res1 + data[0] + data[1] + ... 
  //cudaflow.reduce(
  //  data, data+N, res1, [] __device__ (int a, int b){ return a+b; }
  //);
  //
  //// res2 = data[0] + data[1] + data[2] + ...
  //cudaflow.uninitialized_reduce(
  //  data, data+N, res2, [] __device__ (int a, int b){ return a+b; }
  //);

  //cudaflow.offload();
  //
  //// inspect 
  //if(hres + 10 != *res1 || hres != *res2) {
  //  throw std::runtime_error("incorrect result");
  //}

  //std::cout << "correct result\n";

  return 0;
}


