#include "binary_tree.hpp"
#include <omp.h>

// binary_tree_omp
void binary_tree_omp(size_t num_layers, unsigned num_threads) {

  std::atomic<size_t> counter {0};
  
  size_t N = 1 << num_layers;
  size_t *D = new size_t [N]; 

  #pragma omp parallel num_threads(num_threads)
  {
    #pragma omp single
    {
      for(size_t i = 1; i<N; ++i) {
        
        size_t p = i / 2;
        size_t l = i * 2;
        size_t r = l + 1;
        
        if(l < N && r < N) {
          #pragma omp task firstprivate(i) depend(out:D[l], D[r]) depend(in:D[p])
          {
            //printf("%d\n", i);
            counter.fetch_add(1, std::memory_order_relaxed);
          }
        }
        else {
          #pragma omp task firstprivate(i) depend(in:D[p])
          {
            //printf("%d\n", i);
            counter.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    }
  }

  assert((counter + 1) == N);

  delete [] D;
}

std::chrono::microseconds measure_time_omp(
  size_t num_layers,
  unsigned num_threads
) {
  auto beg = std::chrono::high_resolution_clock::now();
  binary_tree_omp(num_layers, num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


