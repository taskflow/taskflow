#include "async_task.hpp"
#include <taskflow/taskflow.hpp>

// async_task computing
void async_task_taskflow(unsigned num_threads, size_t num_tasks) {

  tf::Executor executor(num_threads);

  for(size_t i=0; i<num_tasks; i++) {
    executor.silent_async(func);
  }

  executor.wait_for_all();
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads, size_t num_tasks) {
  auto beg = std::chrono::high_resolution_clock::now();
  async_task_taskflow(num_threads, num_tasks);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


