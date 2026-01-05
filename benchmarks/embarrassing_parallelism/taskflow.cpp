#include "embarrassing_parallelism.hpp"

#include <taskflow/taskflow.hpp>

// embarrassing_parallelism computing
void embarrassing_parallelism_taskflow(tf::Executor& executor, size_t num_tasks) {
  tf::Taskflow taskflow;
  for(size_t i=0; i<num_tasks; ++i) {
    taskflow.emplace([i](){ dummy(i); });
  }
  executor.run(taskflow).get();
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads, size_t num_tasks) {
  static tf::Executor executor(num_threads);
  auto beg = std::chrono::high_resolution_clock::now();
  embarrassing_parallelism_taskflow(executor, num_tasks);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


