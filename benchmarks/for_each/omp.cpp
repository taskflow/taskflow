#include "for_each.hpp"
#include <omp.h>

// for_each_omp
void for_each_omp(size_t nthreads) {
  #pragma omp parallel for num_threads(nthreads)
  for(size_t i=0; i<vec.size(); i++) {
    vec[i] = std::tan(vec[i]);
  }
}

std::chrono::microseconds measure_time_omp(size_t num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  for_each_omp(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

