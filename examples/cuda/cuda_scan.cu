// This program demonstrate how to perform a parallel scan
// using cudaFlowCapturer.

#include <taskflow/cudaflow.hpp>

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./cuda_scan num_items\n";
    std::exit(EXIT_FAILURE);
  }

  int N = std::atoi(argv[1]);

  auto data1 = tf::cuda_malloc_shared<int>(N);
  auto scan1 = tf::cuda_malloc_shared<int>(N);
  auto data2 = tf::cuda_malloc_shared<int>(N);
  auto scan2 = tf::cuda_malloc_shared<int>(N);

  // initialize the data
  for(int i=0; i<N; i++) {
    data1[i] = i;
    data2[i] = i;
  }
  
  // perform reduction
  tf::cudaFlowCapturer cudaflow;
  
  // inclusive scan
  cudaflow.inclusive_scan(
    data1, data1+N, scan1, [] __device__ (int a, int b){ return a+b; }
  );
  
  // exclusive scan
  cudaflow.exclusive_scan(
    data2, data2+N, scan2, [] __device__ (int a, int b){ return a+b; }
  );

  cudaflow.offload();
  
  // inspect 
  for(int i=1; i<N; i++) {
    //printf("data1[%d]/scan1[%d]=%d/%d\n", i, i, data1[i], scan1[i]);
    if(scan1[i] != scan1[i-1] + data1[i]) {
      throw std::runtime_error("incorrect inclusive scan result");
    }
    //printf("data2[%d]/scan2[%d]=%d/%d\n", i, i, data2[i], scan2[i]);
    if(scan2[i] != scan2[i-1] + data2[i-1]) {
      throw std::runtime_error("incorrect exclusive scan result");
    }
  }

  std::cout << "correct result\n";

  return 0;
}


