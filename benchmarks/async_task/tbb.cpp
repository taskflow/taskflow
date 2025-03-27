#include "async_task.hpp"
#include <tbb/task_group.h>
#include <tbb/global_control.h>

// async_task computation
void async_task_tbb(unsigned num_threads, size_t num_tasks) {
  
  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_threads
  );

  std::atomic<size_t> counter(0);
  tbb::task_group tg;

  for (size_t i = 0; i < num_tasks; i++) {
    tg.run([&] { func(counter); });
  }

  tg.wait();  // Ensures all tasks are completed before returning

  if(counter.load(std::memory_order_relaxed) != num_tasks) {
    throw std::runtime_error("incorrect result");
  }
}

std::chrono::microseconds measure_time_tbb(unsigned num_threads, size_t num_tasks) {
  auto beg = std::chrono::high_resolution_clock::now();
  async_task_tbb(num_threads, num_tasks);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
