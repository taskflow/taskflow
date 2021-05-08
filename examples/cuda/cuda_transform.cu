// This program demonstrates how to performs a parallel transform
// using cudaFlow.

#include <taskflow/cudaflow.hpp>

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./cuda_transform num_items\n";
    std::exit(EXIT_FAILURE);
  }

  size_t N = std::atoi(argv[1]);

  auto data = tf::cuda_malloc_shared<int>(N);
  auto src1 = tf::cuda_malloc_shared<int>(N);
  auto src2 = tf::cuda_malloc_shared<int>(N);
  auto src3 = tf::cuda_malloc_shared<int>(N);
  
  // initialize the data
  for(size_t i=0; i<N; i++) {
    data[i] = 0;
    src1[i] = 1;
    src2[i] = 2;
    src3[i] = 3;
  }
  
  // perform parallel transform
  tf::cudaFlow cudaflow;
  
  // data[i] = src1[i] + src2[i] + src3[i]
  cudaflow.transform(
    data, data + N, 
    [] __device__ (int a, int b, int c) { return a+b+c; }, 
    src1, src2, src3
  );

  cudaflow.offload();

  // inspect the result
  for(size_t i=0; i<N; i++) {
    if(data[i] != (src1[i] + src2[i] + src3[i])) {
      throw std::runtime_error("incorrect result");
    }
  }

  std::cout << "correct result\n";

  return 0;
}
