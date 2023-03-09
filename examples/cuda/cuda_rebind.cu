// This program demonstrates how to rebind a cudaFlowCapturer task
// to another GPU operation.

#include <taskflow/cuda/cudaflow.hpp>
#include <taskflow/cuda/algorithm/for_each.hpp>

int main() {

  size_t N = 10000;

  auto data = tf::cuda_malloc_shared<int>(N);
  
  tf::cudaFlowCapturer cudaflow;
  tf::cudaStream stream;

  // set data to -1
  for(size_t i=0; i<N; i++) {
    data[i] = -1;
  }
  
  // clear data with 0
  std::cout << "clearing data with 0 ...\n";

  tf::cudaTask task = cudaflow.memset(data, 0, N*sizeof(int));
  cudaflow.run(stream);
  stream.synchronize();

  for(size_t i=0; i<N; i++) {
    if(data[i] != 0) {
      std::cout << data[i] << '\n';
      throw std::runtime_error("unexpected result after fill");
    }
  }
  std::cout << "correct result after fill\n";

  // Rebind the task to for-each task setting each element to 100.
  // You can rebind a capture task to any other task type.
  std::cout << "rebind to for_each task setting each element to 100 ...\n";

  cudaflow.for_each(
    task, data, data+N, [] __device__ (int& i){ i = 100; }
  );
  cudaflow.run(stream);
  stream.synchronize();
  
  for(size_t i=0; i<N; i++) {
    if(data[i] != 100) {
      throw std::runtime_error("unexpected result after for_each");
    }
  }
  std::cout << "correct result after updating for_each\n";

  return 0;
}



