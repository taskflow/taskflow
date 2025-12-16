#include "embarrassing_parallelism.hpp"

#include <omp.h>

// embarrassing_parallelism computation
void embarrassing_parallelism_omp(unsigned num_threads, size_t num_tasks) {

  omp_set_num_threads(num_threads);

  #pragma omp parallel
  {
    #pragma omp single
    {
      for(size_t i = 0; i < num_tasks; ++i) {
        #pragma omp task firstprivate(i)
        {
          dummy(i);
        }
      }
  
      #pragma omp taskwait
    }
  }
}

std::chrono::microseconds measure_time_omp(unsigned num_threads, size_t num_tasks) {
  auto beg = std::chrono::high_resolution_clock::now();
  embarrassing_parallelism_omp(num_threads, num_tasks);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
