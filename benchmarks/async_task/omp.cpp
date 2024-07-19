#include "async_task.hpp"
#include <omp.h>

// async_task computation
void async_task_omp(unsigned num_threads, size_t num_tasks) {

  omp_set_num_threads(num_threads);

  #pragma omp parallel
  {
    #pragma omp single
    {
      for (size_t i=0; i<num_tasks; i++) {
        #pragma omp task
        func();
      }  
    }  
  }
}

std::chrono::microseconds measure_time_omp(unsigned num_threads, size_t num_tasks) {
  auto beg = std::chrono::high_resolution_clock::now();
  async_task_omp(num_threads, num_tasks);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
