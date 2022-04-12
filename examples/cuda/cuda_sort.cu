// This program demonstrates how to perform parallel sort with CUDA.

#include <taskflow/cuda/cudaflow.hpp>
#include <taskflow/cuda/algorithm/sort.hpp>

int main(int argc, char* argv[]) {
  
  if(argc != 2) {
    std::cerr << "usage: ./cuda_sort N\n";
    std::exit(EXIT_FAILURE);
  }

  unsigned N = std::atoi(argv[1]);

  // gpu data
  auto d_keys = tf::cuda_malloc_shared<int>(N);

  // cpu data
  std::vector<int> h_keys(N);

  for(unsigned i=0; i<N; i++) {
    int k = rand() % 10000;
    d_keys[i] = k;
    h_keys[i] = k;
  }
  
  // --------------------------------------------------------------------------
  // Standard GPU sort
  // --------------------------------------------------------------------------

  auto p = tf::cudaDefaultExecutionPolicy{};
  
  auto beg = std::chrono::steady_clock::now();
  tf::cudaStream s;
  auto bufsz = tf::cuda_sort_buffer_size<decltype(p), int>(N);
  tf::cudaDeviceVector<std::byte> buf(bufsz);
  tf::cuda_sort(p, d_keys, d_keys+N, tf::cuda_less<int>{}, buf.data());
  s.synchronize();
  auto end = std::chrono::steady_clock::now();

  std::cout << "GPU sort: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end-beg).count()
            << " us\n";
  
  // --------------------------------------------------------------------------
  // CPU sort
  // --------------------------------------------------------------------------
  beg = std::chrono::steady_clock::now();
  std::sort(h_keys.begin(), h_keys.end());
  end = std::chrono::steady_clock::now();
  
  std::cout << "CPU sort: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end-beg).count()
            << " us\n";

  // --------------------------------------------------------------------------
  // verify the result
  // --------------------------------------------------------------------------
  
  for(unsigned i=0; i<N; i++) {
    if(d_keys[i] != h_keys[i]) {
      throw std::runtime_error("incorrect result");
    }
  }

  std::cout << "correct result\n";
};
