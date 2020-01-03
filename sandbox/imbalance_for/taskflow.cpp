#include "sparse.hpp"
#include <taskflow/taskflow.hpp> 

void imbalance_taskflow(unsigned num_threads) {
  tf::Executor executor(num_threads);
  tf::Taskflow taskflow;

  taskflow.dynamic_parallel_for(0, M, 1, [&](uint64_t i) {
    compute_one_iteration(i);
  }, num_threads);

  executor.run(taskflow).get();
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  imbalance_taskflow(num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

