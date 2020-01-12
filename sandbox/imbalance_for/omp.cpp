#include "sparse.hpp"
#include <omp.h>

void imbalance_omp(unsigned num_threads) {
  omp_set_num_threads(num_threads);
  ////#pragma omp parallel for schedule (dynamic)  
  ////#pragma omp parallel for schedule (runtime) 
  ////#pragma omp parallel for schedule (static)  
  #pragma omp parallel for schedule (static)
  for(int i=0; i<M; i++) {
    compute_one_iteration(i);
  } 
}

std::chrono::microseconds measure_time_omp(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  imbalance_omp(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
