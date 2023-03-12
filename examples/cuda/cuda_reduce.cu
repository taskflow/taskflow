// This program demonstrate how to perform a parallel reduction 
// using cudaFlow.

#include <taskflow/cuda/cudaflow.hpp>
#include <taskflow/cuda/algorithm/reduce.hpp>

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
  *res1 = 10;
  *res2 = 10;
  
  // perform reduction
  tf::cudaStream stream;
  tf::cudaDefaultExecutionPolicy policy(stream);

  // get the buffer size needed for reduction
  void* buff;
  cudaMalloc(&buff, policy.reduce_bufsz<int>(N));
  
  // res1 = res1 + data[0] + data[1] + ... 
  tf::cuda_reduce(policy,
    data, data+N, res1, [] __device__ (int a, int b){ return a+b; }, buff
  );
  
  // res2 = data[0] + data[1] + data[2] + ...
  tf::cuda_uninitialized_reduce(policy,
    data, data+N, res2, [] __device__ (int a, int b){ return a+b; }, buff
  );

  stream.synchronize();
  
  // inspect 
  if(hres + 10 != *res1 || hres != *res2) {
    throw std::runtime_error("incorrect result");
  }

  std::cout << "correct result\n";
  
  cudaFree(data);
  cudaFree(res1);
  cudaFree(res2);

  return 0;
}


