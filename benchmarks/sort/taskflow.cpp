#include "sort.hpp"
#include <taskflow/taskflow.hpp> 
#include <taskflow/algorithm/sort.hpp>

void sort_taskflow(unsigned num_threads) {

  tf::Executor executor(num_threads); 
  tf::Taskflow taskflow;

  taskflow.sort(vec.begin(), vec.end());

  executor.run(taskflow).get(); 
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  sort_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


