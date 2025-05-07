#include "fibonacci.hpp"
#include <omp.h>

// fibonacci computation 
size_t fibonacci_omp(size_t num_fibonacci) {

  if (num_fibonacci < 2) { 
    return num_fibonacci;
  }

  size_t x, y;

  #pragma omp task shared(x)
  x = fibonacci_omp(num_fibonacci - 1);

  #pragma omp task shared(y)
  y = fibonacci_omp(num_fibonacci - 2);

  #pragma omp taskwait
  return x + y;
}

std::chrono::microseconds measure_time_omp(size_t num_threads, size_t num_fibonacci) {
  auto beg = std::chrono::high_resolution_clock::now();
  omp_set_num_threads(num_threads);
  #pragma omp parallel
  {
    #pragma omp single
    fibonacci_omp(num_fibonacci);
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


