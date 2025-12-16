#include "embarrassing_parallelism.hpp"

// embarrassing_parallelism computation
void embarrassing_parallelism_seq(size_t num_tasks) {

  for(size_t i=0; i<num_tasks; ++i) {
    dummy(i); 
  }
}

std::chrono::microseconds measure_time_seq(size_t num_tasks) {
  auto beg = std::chrono::high_resolution_clock::now();
  embarrassing_parallelism_seq(num_tasks);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
}

