// This program demonstrate how to perform a parallel scan
// using cudaFlow.

#include <taskflow/cudaflow.hpp>

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./cuda_scan N\n";
    std::exit(EXIT_FAILURE);
  }

  int N = std::atoi(argv[1]);

  auto data1 = tf::cuda_malloc_shared<int>(N);
  auto data2 = tf::cuda_malloc_shared<int>(N);
  auto scan1 = tf::cuda_malloc_shared<int>(N);
  auto scan2 = tf::cuda_malloc_shared<int>(N);

  // --------------------------------------------------------------------------
  // inclusive/exclusive scan
  // --------------------------------------------------------------------------

  // initialize the data
  std::iota(data1, data1 + N, 0);
  std::iota(data2, data2 + N, 0);
  
  tf::cudaFlow cudaflow;
  
  // create inclusive and exclusive scan tasks
  cudaflow.inclusive_scan(data1, data1+N, scan1, tf::cuda_plus<int>{});
  cudaflow.exclusive_scan(data2, data2+N, scan2, tf::cuda_plus<int>{});

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

  std::cout << "scan done\n";
  
  // --------------------------------------------------------------------------
  // transform inclusive/exclusive scan
  // --------------------------------------------------------------------------
  
  cudaflow.clear();

  // initialize the data
  std::iota(data1, data1 + N, 0);
  std::iota(data2, data2 + N, 0);
  
  // transform inclusive scan
  cudaflow.transform_inclusive_scan(
    data1, data1+N, scan1, tf::cuda_plus<int>{},
    [] __device__ (int a) { return a*10; }
  );

  // transform exclusive scan
  cudaflow.transform_exclusive_scan(
    data2, data2+N, scan2, tf::cuda_plus<int>{},
    [] __device__ (int a) { return a*11; }
  );

  cudaflow.offload();
  
  // inspect 
  for(int i=1; i<N; i++) {
    //printf("data1[%d]/scan1[%d]=%d/%d\n", i, i, data1[i], scan1[i]);
    if(scan1[i] != scan1[i-1] + data1[i] * 10) {
      throw std::runtime_error("incorrect transform inclusive scan result");
    }
    //printf("data2[%d]/scan2[%d]=%d/%d\n", i, i, data2[i], scan2[i]);
    if(scan2[i] != scan2[i-1] + data2[i-1] * 11) {
      throw std::runtime_error("incorrect transform exclusive scan result");
    }
  }

  std::cout << "transform scan done\n";

  return 0;
}


