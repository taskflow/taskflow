#include <taskflow/taskflow.hpp>
#include "async_task.hpp"

// async_task computing
void async_task_taskflow(unsigned num_threads, size_t num_tasks) {

  static tf::Executor executor(num_threads);

  std::atomic<size_t> counter(0);

  for(size_t i=0; i<num_tasks; i++) {
    executor.silent_async([&] { func(counter); });
  }

  executor.wait_for_all();
  
  if(counter.load(std::memory_order_relaxed) != num_tasks) {
    throw std::runtime_error("incorrect result");
  }
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads, size_t num_tasks) {
  auto beg = std::chrono::high_resolution_clock::now();
  async_task_taskflow(num_threads, num_tasks);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}


